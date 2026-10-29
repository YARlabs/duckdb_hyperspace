//! Zero-Copy Apache Arrow Columnar Bridge (`RecordBatch` & Arrow IPC Stream/File).
//!
//! Converts HyperspaceDB [`SearchResult`] and [`VectorData`] slices into columnar Apache Arrow
//! `RecordBatch`es and encodes/decodes Arrow IPC buffers (`read_ipc` / Arrow Flight payload format)
//! so DuckDB, Polars, and PyArrow can join analytical tables directly with HyperspaceDB without
//! JSON serialization overhead.

use crate::kernels::{compute_distance, ManifoldMetric};
use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder, UInt32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_ipc::writer::{FileWriter, StreamWriter};
use arrow_schema::{DataType, Field, Schema};
use hyperspace_sdk::{SearchResult, VectorData};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

/// Returns the canonical Apache Arrow [`Schema`] for HyperspaceDB search results:
/// - `id`: `UInt32`
/// - `distance`: `Float64`
/// - `metadata_json`: `Utf8`
#[must_use]
pub fn search_results_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("distance", DataType::Float64, false),
        Field::new("metadata_json", DataType::Utf8, false),
    ]))
}

/// Returns the canonical Apache Arrow [`Schema`] for HyperspaceDB vector points:
/// - `id`: `UInt32`
/// - `vector`: `List<Float64>`
/// - `metadata_json`: `Utf8`
#[must_use]
pub fn vector_points_arrow_schema() -> Arc<Schema> {
    let item_field = Arc::new(Field::new("item", DataType::Float64, true));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("vector", DataType::List(item_field), false),
        Field::new("metadata_json", DataType::Utf8, false),
    ]))
}

/// Converts a slice of HyperspaceDB [`SearchResult`]s into an Apache Arrow [`RecordBatch`].
pub fn search_results_to_record_batch(results: &[SearchResult]) -> Result<RecordBatch, String> {
    let schema = search_results_arrow_schema();
    let mut id_builder = UInt32Builder::with_capacity(results.len());
    let mut dist_builder = Float64Builder::with_capacity(results.len());
    let mut meta_builder = StringBuilder::with_capacity(results.len(), results.len() * 64);

    for r in results {
        id_builder.append_value(r.id);
        dist_builder.append_value(r.distance);
        let meta_str = serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".to_string());
        meta_builder.append_value(&meta_str);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(dist_builder.finish()),
            Arc::new(meta_builder.finish()),
        ],
    )
    .map_err(|e| format!("Failed to construct SearchResult RecordBatch: {e}"))
}

/// Converts a slice of HyperspaceDB [`VectorData`] points into an Apache Arrow [`RecordBatch`].
pub fn vector_points_to_record_batch(points: &[VectorData]) -> Result<RecordBatch, String> {
    let schema = vector_points_arrow_schema();
    let mut id_builder = UInt32Builder::with_capacity(points.len());
    let mut vec_builder = ListBuilder::new(Float64Builder::new());
    let mut meta_builder = StringBuilder::with_capacity(points.len(), points.len() * 64);

    for p in points {
        id_builder.append_value(p.id);
        vec_builder.values().append_slice(&p.vector);
        vec_builder.append(true);
        let meta_str = serde_json::to_string(&p.metadata).unwrap_or_else(|_| "{}".to_string());
        meta_builder.append_value(&meta_str);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(vec_builder.finish()),
            Arc::new(meta_builder.finish()),
        ],
    )
    .map_err(|e| format!("Failed to construct VectorData RecordBatch: {e}"))
}

/// Tuple representation of a HyperspaceDB point `(id, vector, metadata)` decoded from an Arrow batch.
pub type ArrowVectorTuple = (u32, Vec<f64>, HashMap<String, String>);

/// Extracts `(id, vector, metadata)` tuples from an Apache Arrow [`RecordBatch`] matching
/// [`vector_points_arrow_schema`] (or any DuckDB/Parquet Arrow batch with `id: UInt32`,
/// `vector: List<Float64>`, and optional `metadata_json: Utf8`).
pub fn record_batch_to_vector_tuples(
    batch: &RecordBatch,
) -> Result<Vec<ArrowVectorTuple>, String> {
    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| "Missing 'id' column in Arrow RecordBatch".to_string())?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| "Column 'id' must be UInt32Array".to_string())?;

    let vec_col = batch
        .column_by_name("vector")
        .ok_or_else(|| "Missing 'vector' column in Arrow RecordBatch".to_string())?
        .as_list::<i32>();

    let meta_col = batch
        .column_by_name("metadata_json")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let id = id_col.value(row);
        let list_val = vec_col.value(row);
        let float_arr = list_val.as_primitive::<Float64Type>();
        let vector: Vec<f64> = float_arr.values().to_vec();

        let metadata: HashMap<String, String> = meta_col
            .and_then(|mc| {
                if mc.is_null(row) {
                    None
                } else {
                    serde_json::from_str(mc.value(row)).ok()
                }
            })
            .unwrap_or_default();

        out.push((id, vector, metadata));
    }

    Ok(out)
}

/// Evaluates vectorized manifold distances between a query vector and every row in a
/// [`RecordBatch`] containing an `id` (`UInt32`) and `vector` (`List<Float64>`) column,
/// returning a new `RecordBatch` (`id: UInt32`, `distance: Float64`).
pub fn evaluate_columnar_distances(
    batch: &RecordBatch,
    query: &[f64],
    metric: ManifoldMetric,
) -> Result<RecordBatch, String> {
    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| "Missing 'id' column".to_string())?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| "'id' must be UInt32Array".to_string())?;

    let vec_col = batch
        .column_by_name("vector")
        .ok_or_else(|| "Missing 'vector' column".to_string())?
        .as_list::<i32>();

    let mut dists = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let arr = vec_col.value(row);
        let slice = arr.as_primitive::<Float64Type>().values();
        dists.push(compute_distance(slice, query, metric));
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("distance", DataType::Float64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_col.clone()),
            Arc::new(Float64Array::from(dists)),
        ],
    )
    .map_err(|e| format!("Failed to construct distance RecordBatch: {e}"))
}

/// Encodes a [`RecordBatch`] into an in-memory Apache Arrow IPC Stream byte buffer.
pub fn encode_record_batch_ipc_stream(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
            .map_err(|e| format!("Arrow StreamWriter init failed: {e}"))?;
        writer
            .write(batch)
            .map_err(|e| format!("Arrow StreamWriter write failed: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("Arrow StreamWriter finish failed: {e}"))?;
    }
    Ok(buf)
}

/// Decodes all [`RecordBatch`]es from an Apache Arrow IPC Stream byte buffer.
pub fn decode_record_batches_ipc_stream(bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("Arrow StreamReader init failed: {e}"))?;
    let mut batches = Vec::new();
    for item in reader {
        let batch = item.map_err(|e| format!("Arrow StreamReader batch decode error: {e}"))?;
        batches.push(batch);
    }
    Ok(batches)
}

/// Writes a [`RecordBatch`] to an Apache Arrow IPC File (`.arrow` / Feather v2 format) compatible
/// with DuckDB's `read_ipc` / `scan_arrow_ipc`.
pub fn write_record_batch_ipc_file(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buf, batch.schema().as_ref())
            .map_err(|e| format!("Arrow FileWriter init failed: {e}"))?;
        writer
            .write(batch)
            .map_err(|e| format!("Arrow FileWriter write failed: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("Arrow FileWriter finish failed: {e}"))?;
    }
    Ok(buf)
}

/// Reads all [`RecordBatch`]es from an Apache Arrow IPC File byte buffer.
pub fn read_record_batches_ipc_file(bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
    let cursor = Cursor::new(bytes);
    let reader = FileReader::try_new(cursor, None)
        .map_err(|e| format!("Arrow FileReader init failed: {e}"))?;
    let mut batches = Vec::new();
    for item in reader {
        let batch = item.map_err(|e| format!("Arrow FileReader batch decode error: {e}"))?;
        batches.push(batch);
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_ipc_roundtrip_and_columnar_distance() {
        let mut meta = HashMap::new();
        meta.insert("topic".to_string(), "robotics".to_string());

        let points = vec![
            VectorData {
                id: 1,
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: meta.clone(),
                typed_metadata: HashMap::new(),
            },
            VectorData {
                id: 2,
                vector: vec![2.0_f64.sqrt(), 1.0, 0.0, 0.0],
                metadata: meta,
                typed_metadata: HashMap::new(),
            },
        ];

        let batch = vector_points_to_record_batch(&points).expect("Should build Arrow batch");
        assert_eq!(batch.num_rows(), 2);

        let ipc_bytes = encode_record_batch_ipc_stream(&batch).expect("Should encode IPC stream");
        let decoded = decode_record_batches_ipc_stream(&ipc_bytes).expect("Should decode IPC");
        assert_eq!(decoded.len(), 1);

        let tuples = record_batch_to_vector_tuples(&decoded[0]).expect("Should extract tuples");
        assert_eq!(tuples.len(), 2);
        assert_eq!(tuples[0].0, 1);
        assert_eq!(tuples[1].0, 2);

        let dist_batch = evaluate_columnar_distances(
            &decoded[0],
            &[1.0, 0.0, 0.0, 0.0],
            ManifoldMetric::Lorentz,
        )
        .expect("Should evaluate Lorentz columnar distance");
        let dist_col = dist_batch
            .column_by_name("distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(dist_col.value(0).abs() < 1e-9);
        assert!(dist_col.value(1) > 0.8);
    }
}
