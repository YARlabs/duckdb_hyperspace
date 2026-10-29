//! C-ABI Entrypoints for Native Shared Library (`cdylib`) Integration.

use crate::kernels::{compute_distance, ManifoldMetric};
use std::ffi::CStr;
use std::os::raw::c_char;

static VERSION_CSTR: &[u8] = b"v4.2.0\0";

/// Returns the null-terminated version string of `duckdb_hyperspace`.
#[no_mangle]
pub extern "C" fn hs_duckdb_version() -> *const c_char {
    VERSION_CSTR.as_ptr().cast::<c_char>()
}

/// Evaluates pairwise manifold distance over raw `f64` pointers from C / DuckDB UDF callers.
///
/// # Safety
/// `u_ptr` and `v_ptr` must be valid pointers to at least `len` `f64` elements.
/// `metric_ptr` must be a valid null-terminated UTF-8 C string (or null for L2).
#[no_mangle]
pub unsafe extern "C" fn hs_duckdb_distance(
    u_ptr: *const f64,
    v_ptr: *const f64,
    len: usize,
    metric_ptr: *const c_char,
) -> f64 {
    if u_ptr.is_null() || v_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let u = std::slice::from_raw_parts(u_ptr, len);
    let v = std::slice::from_raw_parts(v_ptr, len);
    let metric = if metric_ptr.is_null() {
        ManifoldMetric::L2
    } else {
        let s = CStr::from_ptr(metric_ptr).to_str().unwrap_or("l2");
        ManifoldMetric::from_name(s)
    };
    compute_distance(u, v, metric)
}
