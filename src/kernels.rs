//! Multi-Manifold Analytical Distance Kernels for Columnar & DuckDB Queries.
//!
//! Provides SIMD-friendly distance evaluations across Euclidean, Cosine, Dot, Manhattan,
//! Lorentz Hyperboloid ($\mathbb{H}^{d-1}$), Poincaré Ball, Wasserstein 1D ($W_1$), and
//! 801D Hybrid MRL spaces.

/// Supported geometric distance metrics for DuckDB & Arrow analytical scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldMetric {
    L2,
    Cosine,
    Dot,
    L1,
    Lorentz,
    Poincare,
    Wasserstein,
    Hybrid801,
}

impl ManifoldMetric {
    /// Parses a metric identifier string (`"l2"`, `"cosine"`, `"lorentz"`, `"poincare"`, `"wasserstein"`, `"hybrid"`).
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "cosine" | "cos" | "angular" => Self::Cosine,
            "dot" | "ip" | "inner_product" => Self::Dot,
            "l1" | "manhattan" | "taxicab" => Self::L1,
            "lorentz" | "hyperbolic" | "minkowski" => Self::Lorentz,
            "poincare" | "poincaré" | "ball" => Self::Poincare,
            "wasserstein" | "emd" | "w1" => Self::Wasserstein,
            "hybrid" | "hybrid801" | "801d" => Self::Hybrid801,
            _ => Self::L2,
        }
    }
}

/// Evaluates the pairwise distance between two vectors under the specified [`ManifoldMetric`].
#[must_use]
pub fn compute_distance(u: &[f64], v: &[f64], metric: ManifoldMetric) -> f64 {
    let n = u.len().min(v.len());
    if n == 0 {
        return 0.0;
    }
    let u = &u[..n];
    let v = &v[..n];

    match metric {
        ManifoldMetric::L2 => {
            let sum_sq: f64 = u
                .iter()
                .zip(v)
                .map(|(&a, &b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            sum_sq.sqrt()
        }
        ManifoldMetric::Cosine => {
            let mut dot = 0.0_f64;
            let mut nu = 0.0_f64;
            let mut nv = 0.0_f64;
            for (&a, &b) in u.iter().zip(v) {
                dot += a * b;
                nu += a * a;
                nv += b * b;
            }
            let denom = (nu * nv).sqrt();
            if denom <= 1e-12 {
                1.0
            } else {
                (1.0 - (dot / denom)).clamp(0.0, 2.0)
            }
        }
        ManifoldMetric::Dot => {
            let dot: f64 = u.iter().zip(v).map(|(&a, &b)| a * b).sum();
            -dot
        }
        ManifoldMetric::L1 => u.iter().zip(v).map(|(&a, &b)| (a - b).abs()).sum(),
        ManifoldMetric::Lorentz => {
            let spatial_dot: f64 = u[1..].iter().zip(&v[1..]).map(|(&a, &b)| a * b).sum();
            let neg_minkowski = (u[0] * v[0] - spatial_dot).max(1.0);
            neg_minkowski.acosh()
        }
        ManifoldMetric::Poincare => {
            let mut diff_sq = 0.0_f64;
            let mut u_sq = 0.0_f64;
            let mut v_sq = 0.0_f64;
            for (&a, &b) in u.iter().zip(v) {
                let d = a - b;
                diff_sq += d * d;
                u_sq += a * a;
                v_sq += b * b;
            }
            let denom = (1.0 - u_sq).max(1e-7) * (1.0 - v_sq).max(1e-7);
            let arg = (1.0 + (2.0 * diff_sq) / denom).max(1.0);
            arg.acosh()
        }
        ManifoldMetric::Wasserstein => {
            let mut u_sorted = u.to_vec();
            let mut v_sorted = v.to_vec();
            u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let sum_abs: f64 = u_sorted
                .iter()
                .zip(&v_sorted)
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            sum_abs / (n as f64)
        }
        ManifoldMetric::Hybrid801 => {
            let cutoff = n.min(129);
            let lorentz_part = if cutoff >= 2 {
                compute_distance(&u[..cutoff], &v[..cutoff], ManifoldMetric::Lorentz)
            } else {
                0.0
            };
            let cosine_part = if n > cutoff {
                compute_distance(&u[cutoff..], &v[cutoff..], ManifoldMetric::Cosine)
            } else {
                0.0
            };
            0.35 * lorentz_part + 0.65 * cosine_part
        }
    }
}

/// Lifts a $(d-1)$-dimensional spatial slice onto a $d$-dimensional Lorentz hyperboloid
/// by prepending $x_0 = \sqrt{1 + \|\mathbf{x}\|^2}$.
#[must_use]
pub fn lorentz_lift(spatial: &[f64]) -> Vec<f64> {
    let norm_sq: f64 = spatial.iter().map(|&x| x * x).sum();
    let mut out = Vec::with_capacity(spatial.len() + 1);
    out.push((1.0 + norm_sq).sqrt());
    out.extend_from_slice(spatial);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_manifold_kernels() {
        let u_lor = lorentz_lift(&[0.0, 0.0, 0.0]);
        let v_lor = lorentz_lift(&[1.0, 0.0, 0.0]);
        assert!((compute_distance(&u_lor, &u_lor, ManifoldMetric::Lorentz)).abs() < 1e-9);
        assert!(compute_distance(&u_lor, &v_lor, ManifoldMetric::Lorentz) > 0.8);

        let p1 = [0.1, 0.2, -0.1];
        let p2 = [0.4, -0.2, 0.3];
        let d_p = compute_distance(&p1, &p2, ManifoldMetric::Poincare);
        assert!(d_p > 0.5);

        let w = compute_distance(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], ManifoldMetric::Wasserstein);
        assert!(w.abs() < 1e-9); // Identical empirical distributions
    }
}

