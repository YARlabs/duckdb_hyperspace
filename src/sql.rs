//! DuckDB SQL Macro & Analytical Table Join Generator.
//!
//! Generates DuckDB-compatible SQL bootstrap scripts defining multi-manifold scalar macros
//! (`hyperspace_l2_dist`, `hyperspace_cosine_dist`, `hyperspace_lorentz_dist`,
//! `hyperspace_poincare_dist`, `hyperspace_lorentz_lift`) and Arrow IPC view helpers.

/// Generates the DuckDB SQL initialization script registering multi-manifold vector distance
/// macros and Lorentz hyperboloid lifting functions.
#[must_use]
pub fn generate_duckdb_init_sql() -> &'static str {
    r#"-- HyperspaceDB Multi-Manifold Analytical Macros for DuckDB
-- Compatible with DuckDB v1.0+ list functions

CREATE OR REPLACE MACRO hyperspace_l2_dist(u, v) AS (
    sqrt(list_sum(list_transform(list_zip(u, v), x -> pow(x[1] - x[2], 2))))
);

CREATE OR REPLACE MACRO hyperspace_cosine_dist(u, v) AS (
    1.0 - (
        list_sum(list_transform(list_zip(u, v), x -> x[1] * x[2])) /
        greatest(
            1e-12,
            sqrt(list_sum(list_transform(u, a -> a * a))) *
            sqrt(list_sum(list_transform(v, b -> b * b)))
        )
    )
);

CREATE OR REPLACE MACRO hyperspace_lorentz_lift(spatial) AS (
    list_prepend(sqrt(1.0 + list_sum(list_transform(spatial, x -> x * x))), spatial)
);

CREATE OR REPLACE MACRO hyperspace_lorentz_dist(u, v) AS (
    acosh(
        greatest(
            1.0,
            (u[1] * v[1]) - list_sum(
                list_transform(
                    list_zip(list_slice(u, 2, len(u)), list_slice(v, 2, len(v))),
                    x -> x[1] * x[2]
                )
            )
        )
    )
);

CREATE OR REPLACE MACRO hyperspace_poincare_dist(u, v) AS (
    acosh(
        greatest(
            1.0,
            1.0 + (
                2.0 * list_sum(list_transform(list_zip(u, v), x -> pow(x[1] - x[2], 2)))
            ) / (
                greatest(1e-7, 1.0 - list_sum(list_transform(u, a -> a * a))) *
                greatest(1e-7, 1.0 - list_sum(list_transform(v, b -> b * b)))
            )
        )
    )
);
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duckdb_sql_generation() {
        let sql = generate_duckdb_init_sql();
        assert!(sql.contains("hyperspace_lorentz_dist"));
        assert!(sql.contains("hyperspace_poincare_dist"));
        assert!(sql.contains("hyperspace_lorentz_lift"));
    }
}

