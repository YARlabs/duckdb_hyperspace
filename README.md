# `duckdb-hyperspace` — DuckDB & Apache Arrow IPC Zero-Copy Analytics Bridge for HyperspaceDB

[![Crates.io](https://img.shields.io/crates/v/duckdb-hyperspace.svg)](https://crates.io/crates/duckdb-hyperspace)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

`duckdb-hyperspace` bridges **[DuckDB](https://duckdb.org)**, **Apache Arrow IPC**, and **[HyperspaceDB](https://yar.ink)** so data scientists and analytical pipelines can join local DuckDB/Parquet tables directly with HyperspaceDB's multi-manifold HNSW and CSR KAG graph without JSON serialization overhead.

It depends exclusively on the published [`hyperspace-sdk`](https://crates.io/crates/hyperspace-sdk) (`v4.2.0`) gRPC client and Apache Arrow (`arrow-array`, `arrow-ipc`, `arrow-schema`).

---

## Key Capabilities

1. **Zero-Copy Apache Arrow IPC Columnar Bridge**
   - Converts HyperspaceDB `VectorData` and `SearchResult` batches directly into columnar Apache Arrow `RecordBatch`es (`id: UInt32`, `vector: List<Float64>`, `distance: Float64`, `metadata_json: Utf8`) and writes/reads Arrow IPC streams and `.arrow` files.
2. **Multi-Manifold Columnar Distance Kernels**
   - Vectorized distance kernels across Arrow slices:
     - `Lorentz` ($\mathbb{H}^{d-1}$ hyperboloid geodesic distance)
     - `Poincare` (Poincaré open ball distance)
     - `Wasserstein` ($W_1$ Earth Mover's distance)
     - `Hybrid801` (129D Lorentz prefix + 672D Cosine tail)
     - `Cosine`, `L2`, `Dot`, `L1`
3. **DuckDB SQL Macro Bootstrap & C-ABI (`cdylib`)**
   - Generates pure-DuckDB SQL macros (`hyperspace_lorentz_dist`, `hyperspace_poincare_dist`, `hyperspace_lorentz_lift`, `hyperspace_cosine_dist`) and exports C-ABI entrypoints (`hs_duckdb_distance`, `hs_duckdb_version`).

---

## Quick Start Usage

### 1. Load Multi-Manifold SQL Macros in DuckDB
```bash
duckdb-hyperspace init-sql > hyperspace_macros.sql
duckdb -init hyperspace_macros.sql
```

```sql
-- Inside DuckDB: lift Euclidean embeddings onto the Lorentz hyperboloid and compute geodesic distance
SELECT
    id,
    hyperspace_lorentz_dist(
        hyperspace_lorentz_lift(embedding),
        hyperspace_lorentz_lift([0.12, -0.45, 0.88])
    ) AS lorentz_dist
FROM read_parquet('analytics_events.parquet')
ORDER BY lorentz_dist ASC
LIMIT 10;
```

### 2. Zero-Copy Export & Join via Apache Arrow IPC
```bash
# Materialize top-100 HyperspaceDB search results as an Arrow IPC file
duckdb-hyperspace \
  --url http://127.0.0.1:50051 \
  --collection enterprise_kg \
  search-arrow \
  --query "1.05,0.21,-0.18,0.14" \
  --top-k 100 \
  --output ./hs_results.arrow
```

---

## License

Apache-2.0. Copyright © 2026 **YARlabs**.

