//! Chronos Python bindings - Core integration layer
//!
//! This module establishes the bidirectional interoperability bridge between
//! Rust's high-performance computational core and Python's educational framework,
//! leveraging category-theoretic principles to ensure type-safe, zero-copy data exchange.
//!
//! The architecture employs a functorial mapping between Rust and Python type systems,
//! preserving semantic invariants across language boundaries through PyO3's trait-based
//! type conversion framework.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::types::PyDict;

// Import submodules for registration
mod algorithm;
mod data_structures;
mod execution;
mod temporal;
mod insights;
mod buffer;
mod utils;

/// Chronos Python module exposing the Rust computational core
///
/// This module implements a category-theoretic bridge between Rust and Python,
/// preserving semantic invariants across language boundaries while enabling
/// high-performance algorithm execution with minimal overhead.
#[pymodule]
fn _core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register exception types for type-safe error propagation
    let exceptions = PyDict::new(py);
    exceptions.set_item("ChronosError", py.get_type::<exceptions::ChronosError>())?;
    exceptions.set_item("AlgorithmError", py.get_type::<exceptions::AlgorithmError>())?;
    exceptions.set_item("StateError", py.get_type::<exceptions::StateError>())?;
    exceptions.set_item("VisualizationError", py.get_type::<exceptions::VisualizationError>())?;
    m.add("_exceptions", exceptions)?;

    // Register submodules using categorical composition pattern
    // Each module maintains independent type boundaries but shares
    // common zero-copy protocols through buffer protocol traits
    m.add_wrapped(wrap_pymodule!(algorithm::algorithm))?;
    m.add_wrapped(wrap_pymodule!(data_structures::data_structures))?;
    m.add_wrapped(wrap_pymodule!(execution::execution))?;
    m.add_wrapped(wrap_pymodule!(temporal::temporal))?;
    m.add_wrapped(wrap_pymodule!(insights::insights))?;
    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    
    // Core functionality available at top level with zero overhead
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    
    // Register Python buffer protocol implementation
    // Enables zero-copy interchange between languages
    buffer::register(py, m)?;
    
    // Register metadata
    let meta = PyDict::new(py);
    meta.set_item("version", env!("CARGO_PKG_VERSION"))?;
    meta.set_item("author", "Mohammad Atashi")?;
    meta.set_item("license", "MIT")?;
    meta.set_item("build_timestamp", option_env!("BUILD_TIMESTAMP").unwrap_or("unknown"))?;
    m.add("__meta__", meta)?;

    Ok(())
}

/// Get Chronos version with semantic versioning
#[pyfunction]
fn version() -> String {
    chronos_core::VERSION.to_string()
}

/// Initialize Chronos system with optional configuration
/// 
/// This function initializes the Chronos computational core with thread pools,
/// memory allocators, and other resources based on the system configuration.
#[pyfunction]
#[pyo3(signature = (config = None))]
fn init(py: Python<'_>, config: Option<&PyDict>) -> PyResult<()> {
    // Convert Python config to Rust config using zero-copy serialization
    let rust_config = if let Some(cfg) = config {
        Some(utils::config::py_to_rust_config(py, cfg)?)
    } else {
        None
    };
    
    // Initialize the core with the configuration
    match chronos_core::init_with_config(rust_config) {
        Ok(_) => Ok(()),
        Err(e) => Err(exceptions::ChronosError::new_err(e.to_string()))
    }
}

/// Shutdown Chronos system gracefully
/// 
/// This function ensures proper cleanup of resources, flushing of caches,
/// and termination of background threads.
#[pyfunction]
fn shutdown(py: Python<'_>) -> PyResult<()> {
    // Release GIL during potentially long shutdown process
    py.allow_threads(|| {
        match chronos_core::shutdown() {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::ChronosError::new_err(e.to_string()))
        }
    })
}

/// Custom exception types for type-safe error propagation
pub mod exceptions {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;

    create_exception!(chronos, ChronosError, PyException);
    create_exception!(chronos, AlgorithmError, ChronosError);
    create_exception!(chronos, StateError, ChronosError);
    create_exception!(chronos, VisualizationError, ChronosError);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    
    #[test]
    fn test_module_initialization() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            _core(py, module).unwrap();
            
            // Verify module structure
            assert!(module.hasattr("version").unwrap());
            assert!(module.hasattr("init").unwrap());
            assert!(module.hasattr("shutdown").unwrap());
            assert!(module.hasattr("_exceptions").unwrap());
            assert!(module.hasattr("__meta__").unwrap());
        });
    }

    #[test]
    fn test_version_function() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            _core(py, module).unwrap();
            
            let version_fn = module.getattr("version").unwrap();
            let version: String = version_fn.call0().unwrap().extract().unwrap();
            assert_eq!(version, chronos_core::VERSION);
        });
    }
}