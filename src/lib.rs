//! # `duckdb_hyperspace`
//!
//! DuckDB & Apache Arrow IPC Zero-Copy Analytics Extension for **HyperspaceDB**.
//!
//! ## Core Modules
//! - [`arrow_bridge`]: Zero-copy conversion between HyperspaceDB [`SearchResult`] / [`VectorData`]
//!   slices and Apache Arrow [`RecordBatch`]es + Arrow IPC Stream/File encoding.
//! - [`kernels`]: Multi-manifold analytical distance kernels (`L2`, `Cosine`, `Dot`, `L1`,
//!   `Lorentz`, `Poincare`, `Wasserstein`, `Hybrid801`).
//! - [`sql`]: Generates DuckDB SQL initialization macros (`hyperspace_lorentz_dist`,
//!   `hyperspace_poincare_dist`, `hyperspace_lorentz_lift`, `hyperspace_cosine_dist`).
//! - [`ffi`]: C-ABI entrypoints (`hs_duckdb_version`, `hs_duckdb_distance`) for native loading.

pub mod arrow_bridge;
pub mod ffi;
pub mod kernels;
pub mod sql;

pub use arrow_bridge::{
    decode_record_batches_ipc_stream, encode_record_batch_ipc_stream,
    evaluate_columnar_distances, read_record_batches_ipc_file, record_batch_to_vector_tuples,
    search_results_arrow_schema, search_results_to_record_batch, vector_points_arrow_schema,
    vector_points_to_record_batch, write_record_batch_ipc_file,
};
pub use kernels::{compute_distance, lorentz_lift, ManifoldMetric};
pub use sql::generate_duckdb_init_sql;
