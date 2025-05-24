//! Chronos Execution Bindings
//!
//! This module provides Python bindings for algorithm execution and tracing,
//! enabling detailed monitoring and analysis of algorithm behavior with
//! minimal performance overhead.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::PyTraverseError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chronos_core::execution::{ExecutionTracer, ExecutionConfig, TracePoint, ExecutionFilter};
use chronos_core::algorithm::{Algorithm, NodeId, PathResult};
use crate::exceptions::{ChronosError, AlgorithmError};
use crate::algorithm::PyAlgorithm;
use crate::data_structures::PyGraph;

/// Python module for algorithm execution and tracing
#[pymodule]
pub fn execution(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExecutionTracer>()?;
    m.add_class::<PyExecutionConfig>()?;
    m.add_class::<PyTracePoint>()?;
    m.add_class::<PyExecutionFilter>()?;
    
    // Add execution functions
    m.add_function(wrap_pyfunction!(execute_algorithm, m)?)?;
    m.add_function(wrap_pyfunction!(execute_with_tracing, m)?)?;
    
    Ok(())
}

/// Execution configuration for algorithm tracing
#[pyclass(name = "ExecutionConfig")]
#[derive(Clone)]
pub struct PyExecutionConfig {
    inner: ExecutionConfig,
}

#[pymethods]
impl PyExecutionConfig {
    /// Create a new execution configuration with default settings
    #[new]
    fn new() -> Self {
        Self {
            inner: ExecutionConfig::default(),
        }
    }
    
    /// Set the capture interval for state snapshots
    /// 
    /// Parameters:
    /// - interval: Number of steps between full state captures
    #[pyo3(text_signature = "(self, interval)")]
    fn set_capture_interval(&mut self, interval: usize) -> PyResult<()> {
        self.inner.capture_interval = interval;
        Ok(())
    }
    
    /// Set the trace depth
    /// 
    /// Parameters:
    /// - depth: Maximum depth of the trace (0 for unlimited)
    #[pyo3(text_signature = "(self, depth)")]
    fn set_trace_depth(&mut self, depth: usize) -> PyResult<()> {
        self.inner.trace_depth = if depth == 0 {
            None
        } else {
            Some(depth)
        };
        Ok(())
    }
    
    /// Enable or disable decision point detection
    #[pyo3(text_signature = "(self, enabled)")]
    fn set_decision_detection(&mut self, enabled: bool) -> PyResult<()> {
        self.inner.detect_decisions = enabled;
        Ok(())
    }
    
    /// Set compression level for state storage
    /// 
    /// Parameters:
    /// - level: Compression level (0-9, 0 for no compression)
    #[pyo3(text_signature = "(self, level)")]
    fn set_compression_level(&mut self, level: u32) -> PyResult<()> {
        if level > 9 {
            return Err(PyValueError::new_err("Compression level must be between 0 and 9"));
        }
        self.inner.compression_level = level;
        Ok(())
    }
    
    /// Set filters for tracing specific parts of the algorithm
    #[pyo3(text_signature = "(self, filter)")]
    fn set_filter(&mut self, filter: &PyExecutionFilter) -> PyResult<()> {
        self.inner.filter = filter.inner.clone();
        Ok(())
    }
    
    /// Create an execution configuration from a Python dictionary
    #[classmethod]
    #[pyo3(text_signature = "(cls, config_dict)")]
    fn from_dict(cls: &PyType, config_dict: &PyDict) -> PyResult<Self> {
        let mut config = Self::new();
        
        if let Some(interval) = config_dict.get_item("capture_interval") {
            let interval: usize = interval.extract()?;
            config.set_capture_interval(interval)?;
        }
        
        if let Some(depth) = config_dict.get_item("trace_depth") {
            let depth: usize = depth.extract()?;
            config.set_trace_depth(depth)?;
        }
        
        if let Some(detect_decisions) = config_dict.get_item("detect_decisions") {
            let detect_decisions: bool = detect_decisions.extract()?;
            config.set_decision_detection(detect_decisions)?;
        }
        
        if let Some(compression_level) = config_dict.get_item("compression_level") {
            let compression_level: u32 = compression_level.extract()?;
            config.set_compression_level(compression_level)?;
        }
        
        // Additional settings from dictionary
        
        Ok(config)
    }
    
    /// Convert the configuration to a Python dictionary
    #[pyo3(text_signature = "(self)")]
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        
        dict.set_item("capture_interval", self.inner.capture_interval)?;
        dict.set_item("trace_depth", match self.inner.trace_depth {
            Some(depth) => depth,
            None => 0,
        })?;
        dict.set_item("detect_decisions", self.inner.detect_decisions)?;
        dict.set_item("compression_level", self.inner.compression_level)?;
        
        // Additional settings to dictionary
        
        Ok(dict)
    }
}

/// Execution filter for selective tracing
#[pyclass(name = "ExecutionFilter")]
#[derive(Clone)]
pub struct PyExecutionFilter {
    inner: ExecutionFilter,
}

#[pymethods]
impl PyExecutionFilter {
    /// Create a new execution filter
    #[new]
    fn new() -> Self {
        Self {
            inner: ExecutionFilter::default(),
        }
    }
    
    /// Include specific nodes in the trace
    #[pyo3(text_signature = "(self, nodes)")]
    fn include_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.inner.include_nodes = Some(nodes.into_iter().collect());
        Ok(())
    }
    
    /// Exclude specific nodes from the trace
    #[pyo3(text_signature = "(self, nodes)")]
    fn exclude_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.inner.exclude_nodes = Some(nodes.into_iter().collect());
        Ok(())
    }
    
    /// Include specific state changes in the trace
    #[pyo3(text_signature = "(self, changes)")]
    fn include_changes(&mut self, changes: Vec<String>) -> PyResult<()> {
        self.inner.include_changes = Some(changes.into_iter().collect());
        Ok(())
    }
    
    /// Only trace steps with state changes
    #[pyo3(text_signature = "(self, enabled)")]
    fn only_state_changes(&mut self, enabled: bool) -> PyResult<()> {
        self.inner.only_state_changes = enabled;
        Ok(())
    }
}

/// Trace point in algorithm execution
#[pyclass(name = "TracePoint")]
#[derive(Clone)]
pub struct PyTracePoint {
    inner: Arc<TracePoint>,
}

#[pymethods]
impl PyTracePoint {
    /// Get the step index of this trace point
    #[getter]
    fn step(&self) -> usize {
        self.inner.step
    }
    
    /// Get the current node at this trace point
    #[getter]
    fn current_node(&self) -> Option<NodeId> {
        self.inner.current_node
    }
    
    /// Check if this trace point is a decision point
    #[getter]
    fn is_decision_point(&self) -> bool {
        self.inner.is_decision_point
    }
    
    /// Get the timestamp of this trace point
    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }
    
    /// Get the state data at this trace point
    #[pyo3(text_signature = "(self)")]
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        
        // Open set
        let open_set = PyList::new(py, &self.inner.state.open_set);
        dict.set_item("open_set", open_set)?;
        
        // Closed set
        let closed_set = PyList::new(py, &self.inner.state.closed_set);
        dict.set_item("closed_set", closed_set)?;
        
        // Current node
        if let Some(node) = self.inner.state.current_node {
            dict.set_item("current_node", node)?;
        } else {
            dict.set_item("current_node", py.None())?;
        }
        
        // Custom state data
        let data = PyDict::new(py);
        for (key, value) in &self.inner.state.data {
            data.set_item(key, value)?;
        }
        dict.set_item("data", data)?;
        
        Ok(dict)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "TracePoint(step={}, current_node={:?}, is_decision_point={})",
            self.inner.step,
            self.inner.current_node,
            self.inner.is_decision_point
        )
    }
}

/// Execution tracer for algorithm monitoring
#[pyclass(name = "ExecutionTracer")]
pub struct PyExecutionTracer {
    inner: Mutex<ExecutionTracer>,
}

#[pymethods]
impl PyExecutionTracer {
    /// Create a new execution tracer
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PyExecutionConfig>) -> Self {
        match config {
            Some(config) => Self {
                inner: Mutex::new(ExecutionTracer::with_config(config.inner.clone())),
            },
            None => Self {
                inner: Mutex::new(ExecutionTracer::new()),
            },
        }
    }
    
    /// Start tracing algorithm execution
    #[pyo3(text_signature = "(self, algorithm_name)")]
    fn start_trace(&self, algorithm_name: &str) -> PyResult<()> {
        let mut tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        tracer.start_trace(algorithm_name)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start trace: {}", e)))
    }
    
    /// End tracing algorithm execution
    #[pyo3(text_signature = "(self)")]
    fn end_trace(&self) -> PyResult<()> {
        let mut tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        tracer.end_trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to end trace: {}", e)))
    }
    
    /// Add a trace point manually
    #[pyo3(text_signature = "(self, point)")]
    fn add_trace_point(&self, point: &PyTracePoint) -> PyResult<()> {
        let mut tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        tracer.add_trace_point((*point.inner).clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add trace point: {}", e)))
    }
    
    /// Get all trace points
    #[pyo3(text_signature = "(self)")]
    fn get_trace_points<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        let trace = tracer.get_trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trace: {}", e)))?;
        
        let points = trace.points.iter()
            .map(|point| {
                PyTracePoint {
                    inner: Arc::new(point.clone()),
                }
            })
            .collect::<Vec<_>>();
        
        Ok(PyList::new(py, points))
    }
    
    /// Get decision points in the trace
    #[pyo3(text_signature = "(self)")]
    fn get_decision_points<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        let trace = tracer.get_trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trace: {}", e)))?;
        
        let points = trace.points.iter()
            .filter(|point| point.is_decision_point)
            .map(|point| {
                PyTracePoint {
                    inner: Arc::new(point.clone()),
                }
            })
            .collect::<Vec<_>>();
        
        Ok(PyList::new(py, points))
    }
    
    /// Get the trace point at a specific step
    #[pyo3(text_signature = "(self, step)")]
    fn get_trace_point(&self, step: usize) -> PyResult<Option<PyTracePoint>> {
        let tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        let trace = tracer.get_trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trace: {}", e)))?;
        
        Ok(trace.points.iter()
            .find(|point| point.step == step)
            .map(|point| {
                PyTracePoint {
                    inner: Arc::new(point.clone()),
                }
            }))
    }
    
    /// Get trace statistics
    #[pyo3(text_signature = "(self)")]
    fn get_statistics<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        let trace = tracer.get_trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trace: {}", e)))?;
        
        let stats = PyDict::new(py);
        stats.set_item("algorithm_name", trace.algorithm_name)?;
        stats.set_item("step_count", trace.points.len())?;
        stats.set_item("decision_count", trace.points.iter().filter(|p| p.is_decision_point).count())?;
        stats.set_item("start_time", trace.start_time.to_string())?;
        
        if let Some(end_time) = trace.end_time {
            stats.set_item("end_time", end_time.to_string())?;
            
            // Calculate execution time
            let duration = end_time - trace.start_time;
            stats.set_item("execution_time_ms", duration.as_millis() as u64)?;
        }
        
        Ok(stats)
    }
    
    /// Clear the trace
    #[pyo3(text_signature = "(self)")]
    fn clear(&self) -> PyResult<()> {
        let mut tracer = self.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        tracer.clear();
        Ok(())
    }
}

/// Execute an algorithm on a graph and return the result
#[pyfunction]
#[pyo3(text_signature = "(algorithm, graph, start, goal)")]
fn execute_algorithm(
    py: Python<'_>,
    algorithm: &PyAny,
    graph: &PyGraph,
    start: NodeId,
    goal: NodeId,
) -> PyResult<PyObject> {
    // Extract algorithm implementation
    let algo: &PyAlgorithm = algorithm.extract()?;
    
    // Release GIL during potentially long computation
    let result = py.allow_threads(|| {
        algo.inner.find_path(&graph.inner, start, goal)
    });
    
    // Convert result to Python
    match result {
        Ok(path_result) => {
            let result_dict = PyDict::new(py);
            
            // Path
            if let Some(path) = path_result.path {
                result_dict.set_item("path", PyList::new(py, path))?;
            } else {
                result_dict.set_item("path", py.None())?;
            }
            
            // Cost
            if let Some(cost) = path_result.cost {
                result_dict.set_item("cost", cost)?;
            } else {
                result_dict.set_item("cost", py.None())?;
            }
            
            // Algorithm result
            let result = path_result.result;
            result_dict.set_item("steps", result.steps)?;
            result_dict.set_item("nodes_visited", result.nodes_visited)?;
            result_dict.set_item("execution_time_ms", result.execution_time_ms)?;
            
            Ok(result_dict.to_object(py))
        },
        Err(e) => Err(AlgorithmError::new_err(e.to_string())),
    }
}

/// Execute an algorithm with tracing and return both the result and tracer
#[pyfunction]
#[pyo3(signature = (algorithm, graph, start, goal, config = None))]
fn execute_with_tracing(
    py: Python<'_>,
    algorithm: &PyAny,
    graph: &PyGraph,
    start: NodeId,
    goal: NodeId,
    config: Option<&PyExecutionConfig>,
) -> PyResult<(PyObject, PyExecutionTracer)> {
    // Extract algorithm implementation
    let algo: &PyAlgorithm = algorithm.extract()?;
    
    // Create tracer
    let tracer = PyExecutionTracer::new(config);
    
    // Get algorithm result
    let result = {
        let mut tracer_inner = tracer.inner.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire tracer lock: {}", e))
        })?;
        
        // Release GIL during potentially long computation
        py.allow_threads(|| {
            algo.inner.execute_with_tracing(&graph.inner, &mut tracer_inner)
        })
    };
    
    // Convert result to Python
    match result {
        Ok(result) => {
            let result_dict = PyDict::new(py);
            
            result_dict.set_item("steps", result.steps)?;
            result_dict.set_item("nodes_visited", result.nodes_visited)?;
            result_dict.set_item("execution_time_ms", result.execution_time_ms)?;
            
            Ok((result_dict.to_object(py), tracer))
        },
        Err(e) => Err(AlgorithmError::new_err(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    
    #[test]
    fn test_execution_config() {
        Python::with_gil(|py| {
            let config = PyExecutionConfig::new();
            
            // Test default values
            let dict = config.to_dict(py).unwrap();
            assert_eq!(dict.get_item("capture_interval").unwrap().extract::<usize>().unwrap(), 10);
            assert_eq!(dict.get_item("trace_depth").unwrap().extract::<usize>().unwrap(), 0);
            assert_eq!(dict.get_item("detect_decisions").unwrap().extract::<bool>().unwrap(), true);
            
            // Test setting values
            let mut config = PyExecutionConfig::new();
            config.set_capture_interval(20).unwrap();
            config.set_trace_depth(100).unwrap();
            config.set_decision_detection(false).unwrap();
            
            let dict = config.to_dict(py).unwrap();
            assert_eq!(dict.get_item("capture_interval").unwrap().extract::<usize>().unwrap(), 20);
            assert_eq!(dict.get_item("trace_depth").unwrap().extract::<usize>().unwrap(), 100);
            assert_eq!(dict.get_item("detect_decisions").unwrap().extract::<bool>().unwrap(), false);
        });
    }
    
    #[test]
    fn test_execution_filter() {
        Python::with_gil(|py| {
            let mut filter = PyExecutionFilter::new();
            
            // Test including nodes
            filter.include_nodes(vec![1, 2, 3]).unwrap();
            assert!(filter.inner.include_nodes.as_ref().unwrap().contains(&1));
            assert!(filter.inner.include_nodes.as_ref().unwrap().contains(&2));
            assert!(filter.inner.include_nodes.as_ref().unwrap().contains(&3));
            
            // Test excluding nodes
            filter.exclude_nodes(vec![4, 5]).unwrap();
            assert!(filter.inner.exclude_nodes.as_ref().unwrap().contains(&4));
            assert!(filter.inner.exclude_nodes.as_ref().unwrap().contains(&5));
            
            // Test only state changes
            filter.only_state_changes(true).unwrap();
            assert_eq!(filter.inner.only_state_changes, true);
        });
    }
    
    #[test]
    fn test_execution_tracer() {
        Python::with_gil(|py| {
            let tracer = PyExecutionTracer::new(None);
            
            // Test start/end trace
            assert!(tracer.start_trace("test_algorithm").is_ok());
            assert!(tracer.end_trace().is_ok());
            
            // Test getting statistics
            let stats = tracer.get_statistics(py).unwrap();
            assert_eq!(stats.get_item("algorithm_name").unwrap().extract::<String>().unwrap(), "test_algorithm");
            assert_eq!(stats.get_item("step_count").unwrap().extract::<usize>().unwrap(), 0);
            
            // Test clearing
            assert!(tracer.clear().is_ok());
        });
    }
}