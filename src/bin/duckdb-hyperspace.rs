//! CLI for `duckdb-hyperspace` — Apache Arrow IPC & DuckDB Zero-Copy Bridge for HyperspaceDB.

use clap::{Parser, Subcommand};
use duckdb_hyperspace::{
    generate_duckdb_init_sql, read_record_batches_ipc_file, record_batch_to_vector_tuples,
    search_results_to_record_batch, vector_points_to_record_batch, write_record_batch_ipc_file,
};
use hyperspace_sdk::{Client, DurabilityLevel, SearchOptions};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "duckdb-hyperspace",
    version = "4.2.0",
    about = "DuckDB & Apache Arrow IPC Zero-Copy Analytics Bridge for HyperspaceDB"
)]
struct Cli {
    /// HyperspaceDB gRPC endpoint URL
    #[arg(long, env = "HYPERSPACE_URL", default_value = "http://127.0.0.1:50051")]
    url: String,

    /// Optional HyperspaceDB API key
    #[arg(long, env = "HYPERSPACE_API_KEY")]
    api_key: Option<String>,

    /// Target collection name in HyperspaceDB
    #[arg(short, long, env = "HYPERSPACE_COLLECTION", default_value = "default")]
    collection: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Print the DuckDB SQL macro bootstrap script (`hyperspace_lorentz_dist`, `hyperspace_poincare_dist`, etc.)
    InitSql,
    /// Scroll a HyperspaceDB collection and write points to a zero-copy Apache Arrow IPC file (`.arrow`) for DuckDB
    ExportArrow {
        /// Output `.arrow` file path
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum number of points to export
        #[arg(long, default_value_t = 10000)]
        limit: u32,
    },
    /// Execute a vector search on HyperspaceDB and materialize results to an Apache Arrow IPC file (`.arrow`)
    SearchArrow {
        /// Query vector as comma-separated floats
        #[arg(short, long)]
        query: String,

        /// Top-K nearest neighbors
        #[arg(short = 'k', long, default_value_t = 50)]
        top_k: u32,

        /// Output `.arrow` file path
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Ingest an Apache Arrow IPC file (`.arrow`) exported from DuckDB directly into HyperspaceDB
    ImportArrow {
        /// Input `.arrow` file path
        #[arg(short, long)]
        input: PathBuf,

        /// Batch size for gRPC insertion
        #[arg(long, default_value_t = 500)]
        batch_size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::InitSql => {
            println!("{}", generate_duckdb_init_sql());
        }
        Commands::ExportArrow { output, limit } => {
            let mut client = Client::connect(cli.url, cli.api_key, None).await?;
            let points = client
                .scroll(limit, 0, vec![], cli.collection.clone())
                .await?;
            let batch = vector_points_to_record_batch(&points)?;
            let ipc_bytes = write_record_batch_ipc_file(&batch)?;
            std::fs::write(&output, ipc_bytes)?;
            println!(
                "Exported {} points from '{}' to {}",
                points.len(),
                cli.collection,
                output.display()
            );
        }
        Commands::SearchArrow {
            query,
            top_k,
            output,
        } => {
            let q_vec: Vec<f64> = query
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();
            let mut client = Client::connect(cli.url, cli.api_key, None).await?;
            let results = client
                .search_with_options(
                    q_vec,
                    top_k,
                    SearchOptions::new().collection(cli.collection.clone()),
                )
                .await?;
            let batch = search_results_to_record_batch(&results)?;
            let ipc_bytes = write_record_batch_ipc_file(&batch)?;
            std::fs::write(&output, ipc_bytes)?;
            println!(
                "Wrote {} search results to {}",
                results.len(),
                output.display()
            );
        }
        Commands::ImportArrow { input, batch_size } => {
            let bytes = std::fs::read(&input)?;
            let batches = read_record_batches_ipc_file(&bytes)?;
            let mut client = Client::connect(cli.url, cli.api_key, None).await?;
            let mut total_inserted = 0usize;

            for b in &batches {
                let tuples = record_batch_to_vector_tuples(b)?;
                for chunk in tuples.chunks(batch_size.max(1)) {
                    client
                        .batch_insert(
                            chunk.to_vec(),
                            Some(cli.collection.clone()),
                            DurabilityLevel::default(),
                        )
                        .await?;
                    total_inserted += chunk.len();
                }
            }
            println!(
                "Imported {} vectors from {} into '{}'",
                total_inserted,
                input.display(),
                cli.collection
            );
        }
    }

    Ok(())
}

