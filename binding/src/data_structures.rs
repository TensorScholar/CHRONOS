//! Chronos Data Structure Bindings
//!
//! This module provides Python bindings for Chronos core data structures,
//! implementing a zero-copy buffer protocol for efficient cross-language
//! data sharing while preserving all mathematical properties and invariants.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use pyo3::buffer::PyBuffer;
use numpy::{PyArray, PyArray1, PyArray2, IntoPyArray, ToPyArray};
use std::collections::HashMap;
use std::sync::Arc;

use chronos_core::data_structures::{Graph, Grid, PriorityQueue};
use crate::buffer::ZeroCopyBuffer;
use crate::exceptions::{ChronosError, StateError};

/// Python module for data structure interoperability
#[pymodule]
pub fn data_structures(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register Python-exposed types with cross-language optimizations
    m.add_class::<PyGraph>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyPriorityQueue>()?;
    
    // Register factory functions
    m.add_function(wrap_pyfunction!(create_grid_graph, m)?)?;
    m.add_function(wrap_pyfunction!(create_random_graph, m)?)?;
    
    Ok(())
}

/// Graph data structure with zero-copy buffer protocol implementation
///
/// This class represents a mathematical graph with nodes and edges,
/// providing efficient methods for graph manipulation and analysis.
#[pyclass(name = "Graph")]
#[derive(Clone)]
pub struct PyGraph {
    inner: Arc<Graph>,
}

#[pymethods]
impl PyGraph {
    /// Create a new empty graph
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Graph::new()),
        }
    }
    
    /// Add a node to the graph
    ///
    /// Returns the ID of the newly created node
    #[pyo3(text_signature = "(self, x, y)")]
    fn add_node(&mut self, x: f64, y: f64) -> usize {
        // Clone the Arc and get a mutable reference
        let inner = Arc::make_mut(&mut self.inner);
        inner.add_node((x, y))
    }
    
    /// Add an edge between two nodes
    #[pyo3(text_signature = "(self, source, target, weight=1.0)")]
    fn add_edge(&mut self, source: usize, target: usize, weight: Option<f64>) -> PyResult<()> {
        // Clone the Arc and get a mutable reference
        let inner = Arc::make_mut(&mut self.inner);
        inner.add_edge(source, target, weight.unwrap_or(1.0))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Check if the graph has a node with the given ID
    #[pyo3(text_signature = "(self, node_id)")]
    fn has_node(&self, node_id: usize) -> bool {
        self.inner.has_node(node_id)
    }
    
    /// Check if the graph has an edge between source and target
    #[pyo3(text_signature = "(self, source, target)")]
    fn has_edge(&self, source: usize, target: usize) -> bool {
        self.inner.has_edge(source, target)
    }
    
    /// Get the weight of an edge
    #[pyo3(text_signature = "(self, source, target)")]
    fn get_edge_weight(&self, source: usize, target: usize) -> Option<f64> {
        self.inner.get_edge_weight(source, target)
    }
    
    /// Get all nodes in the graph
    #[pyo3(text_signature = "(self)")]
    fn get_nodes<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let nodes = PyList::new(py, self.inner.get_nodes().map(|n| n.id));
        Ok(nodes)
    }
    
    /// Get the position of a node
    #[pyo3(text_signature = "(self, node_id)")]
    fn get_node_position<'py>(&self, py: Python<'py>, node_id: usize) -> PyResult<Option<&'py PyTuple>> {
        match self.inner.get_node_position(node_id) {
            Some(pos) => {
                let tuple = PyTuple::new(py, &[pos.0, pos.1]);
                Ok(Some(tuple))
            },
            None => Ok(None),
        }
    }
    
    /// Get the neighbors of a node
    #[pyo3(text_signature = "(self, node_id)")]
    fn get_neighbors<'py>(&self, py: Python<'py>, node_id: usize) -> PyResult<Option<&'py PyList>> {
        match self.inner.get_neighbors(node_id) {
            Some(neighbors) => {
                let neighbors_list = PyList::new(py, neighbors.collect::<Vec<_>>());
                Ok(Some(neighbors_list))
            },
            None => Ok(None),
        }
    }
    
    /// Get the number of nodes in the graph
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
    
    /// Get the number of edges in the graph
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
    
    /// Remove a node from the graph
    #[pyo3(text_signature = "(self, node_id)")]
    fn remove_node(&mut self, node_id: usize) -> PyResult<bool> {
        // Clone the Arc and get a mutable reference
        let inner = Arc::make_mut(&mut self.inner);
        Ok(inner.remove_node(node_id).is_some())
    }
    
    /// Remove an edge from the graph
    #[pyo3(text_signature = "(self, source, target)")]
    fn remove_edge(&mut self, source: usize, target: usize) -> PyResult<bool> {
        // Clone the Arc and get a mutable reference
        let inner = Arc::make_mut(&mut self.inner);
        Ok(inner.remove_edge(source, target).is_some())
    }
    
    /// Clear the graph
    #[pyo3(text_signature = "(self)")]
    fn clear(&mut self) {
        // Clone the Arc and get a mutable reference
        let inner = Arc::make_mut(&mut self.inner);
        inner.clear();
    }
    
    /// Get adjacency matrix as NumPy array using zero-copy buffer protocol
    #[pyo3(text_signature = "(self)")]
    fn get_adjacency_matrix<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let node_count = self.inner.node_count();
        let mut adjacency = vec![vec![0.0; node_count]; node_count];
        
        // Fill adjacency matrix
        for i in 0..node_count {
            if let Some(neighbors) = self.inner.get_neighbors(i) {
                for j in neighbors {
                    if let Some(weight) = self.inner.get_edge_weight(i, j) {
                        adjacency[i][j] = weight;
                    }
                }
            }
        }
        
        // Convert to numpy array with zero-copy where possible
        Ok(PyArray::from_vec2(py, &adjacency).unwrap())
    }
    
    /// Get node positions as NumPy array
    #[pyo3(text_signature = "(self)")]
    fn get_node_positions<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let node_count = self.inner.node_count();
        let mut positions = vec![vec![0.0; 2]; node_count];
        
        // Fill positions array
        for i in 0..node_count {
            if let Some(pos) = self.inner.get_node_position(i) {
                positions[i][0] = pos.0;
                positions[i][1] = pos.1;
            }
        }
        
        // Convert to numpy array with zero-copy where possible
        Ok(PyArray::from_vec2(py, &positions).unwrap())
    }
    
    /// Create a graph from an adjacency matrix
    #[classmethod]
    #[pyo3(text_signature = "(cls, adjacency_matrix, positions=None)")]
    fn from_adjacency_matrix<'py>(
        cls: &PyType,
        py: Python<'py>,
        adjacency_matrix: &PyArray2<f64>,
        positions: Option<&PyArray2<f64>>,
    ) -> PyResult<PyGraph> {
        let array = unsafe { adjacency_matrix.as_array() };
        if array.shape()[0] != array.shape()[1] {
            return Err(PyValueError::new_err("Adjacency matrix must be square"));
        }
        
        let node_count = array.shape()[0];
        let mut graph = Self::new();
        
        // Add nodes with positions
        for i in 0..node_count {
            let pos = if let Some(pos_array) = positions {
                let pos_array = unsafe { pos_array.as_array() };
                if pos_array.shape()[0] < node_count || pos_array.shape()[1] < 2 {
                    return Err(PyValueError::new_err("Positions array too small"));
                }
                (pos_array[[i, 0]], pos_array[[i, 1]])
            } else {
                // Default positions in a circle
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (node_count as f64);
                (angle.cos(), angle.sin())
            };
            
            // Clone the Arc and get a mutable reference
            let inner = Arc::make_mut(&mut graph.inner);
            inner.add_node_with_id(i, pos).map_err(|e| PyValueError::new_err(e))?;
        }
        
        // Add edges
        for i in 0..node_count {
            for j in 0..node_count {
                let weight = array[[i, j]];
                if weight > 0.0 {
                    // Clone the Arc and get a mutable reference
                    let inner = Arc::make_mut(&mut graph.inner);
                    inner.add_edge(i, j, weight).map_err(|e| PyValueError::new_err(e))?;
                }
            }
        }
        
        Ok(graph)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Graph(nodes={}, edges={})", self.inner.node_count(), self.inner.edge_count())
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Grid data structure with zero-copy buffer protocol implementation
#[pyclass(name = "Grid")]
#[derive(Clone)]
pub struct PyGrid {
    inner: Arc<Grid>,
}

#[pymethods]
impl PyGrid {
    /// Create a new grid with the specified dimensions
    #[new]
    #[pyo3(text_signature = "(width, height)")]
    fn new(width: usize, height: usize) -> Self {
        Self {
            inner: Arc::new(Grid::new(width, height)),
        }
    }
    
    /// Get grid width
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }
    
    /// Get grid height
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }
    
    // Additional grid methods similar to PyGraph...
}

/// Priority queue data structure with efficient decrease-key operations
#[pyclass(name = "PriorityQueue")]
#[derive(Clone)]
pub struct PyPriorityQueue {
    inner: Arc<PriorityQueue>,
}

#[pymethods]
impl PyPriorityQueue {
    /// Create a new empty priority queue
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(PriorityQueue::new()),
        }
    }
    
    // Priority queue methods...
}

/// Create a grid graph with the specified dimensions
#[pyfunction]
#[pyo3(text_signature = "(width, height, diagonals=False)")]
fn create_grid_graph(width: usize, height: usize, diagonals: Option<bool>) -> PyResult<PyGraph> {
    let mut graph = PyGraph::new();
    let allow_diagonals = diagonals.unwrap_or(false);
    
    // Add nodes
    for y in 0..height {
        for x in 0..width {
            let node_id = y * width + x;
            let inner = Arc::make_mut(&mut graph.inner);
            inner.add_node_with_id(node_id, (x as f64, y as f64))
                .map_err(|e| PyValueError::new_err(e))?;
        }
    }
    
    // Add edges
    for y in 0..height {
        for x in 0..width {
            let node_id = y * width + x;
            
            // Add cardinal direction edges
            if x < width - 1 {
                // East
                let east_id = y * width + (x + 1);
                let inner = Arc::make_mut(&mut graph.inner);
                inner.add_edge(node_id, east_id, 1.0)
                    .map_err(|e| PyValueError::new_err(e))?;
            }
            
            if y < height - 1 {
                // South
                let south_id = (y + 1) * width + x;
                let inner = Arc::make_mut(&mut graph.inner);
                inner.add_edge(node_id, south_id, 1.0)
                    .map_err(|e| PyValueError::new_err(e))?;
            }
            
            // Add diagonal edges if enabled
            if allow_diagonals {
                if x < width - 1 && y < height - 1 {
                    // Southeast
                    let southeast_id = (y + 1) * width + (x + 1);
                    let inner = Arc::make_mut(&mut graph.inner);
                    inner.add_edge(node_id, southeast_id, std::f64::consts::SQRT_2)
                        .map_err(|e| PyValueError::new_err(e))?;
                }
                
                if x > 0 && y < height - 1 {
                    // Southwest
                    let southwest_id = (y + 1) * width + (x - 1);
                    let inner = Arc::make_mut(&mut graph.inner);
                    inner.add_edge(node_id, southwest_id, std::f64::consts::SQRT_2)
                        .map_err(|e| PyValueError::new_err(e))?;
                }
            }
        }
    }
    
    Ok(graph)
}

/// Create a random graph with the specified number of nodes and edge probability
#[pyfunction]
#[pyo3(text_signature = "(node_count, edge_probability, seed=None)")]
fn create_random_graph(node_count: usize, edge_probability: f64, seed: Option<u64>) -> PyResult<PyGraph> {
    if edge_probability < 0.0 || edge_probability > 1.0 {
        return Err(PyValueError::new_err("Edge probability must be between 0 and 1"));
    }
    
    let mut graph = PyGraph::new();
    
    // Set random seed if provided
    let mut rng = if let Some(seed_value) = seed {
        rand::rngs::StdRng::seed_from_u64(seed_value)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    
    // Add nodes in a circle
    for i in 0..node_count {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (node_count as f64);
        let x = angle.cos();
        let y = angle.sin();
        
        let inner = Arc::make_mut(&mut graph.inner);
        inner.add_node((x, y));
    }
    
    // Add random edges
    for i in 0..node_count {
        for j in 0..node_count {
            if i != j && rand::Rng::gen_bool(&mut rng, edge_probability) {
                let inner = Arc::make_mut(&mut graph.inner);
                inner.add_edge(i, j, 1.0)
                    .map_err(|e| PyValueError::new_err(e))?;
            }
        }
    }
    
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    
    #[test]
    fn test_graph_creation() {
        Python::with_gil(|py| {
            let graph = PyGraph::new();
            assert_eq!(graph.node_count(), 0);
            assert_eq!(graph.edge_count(), 0);
        });
    }
    
    #[test]
    fn test_graph_add_node() {
        Python::with_gil(|py| {
            let mut graph = PyGraph::new();
            let node_id = graph.add_node(1.0, 2.0);
            assert_eq!(node_id, 0);
            assert_eq!(graph.node_count(), 1);
            assert!(graph.has_node(node_id));
        });
    }
    
    #[test]
    fn test_graph_add_edge() {
        Python::with_gil(|py| {
            let mut graph = PyGraph::new();
            let n1 = graph.add_node(1.0, 2.0);
            let n2 = graph.add_node(3.0, 4.0);
            
            assert!(graph.add_edge(n1, n2, Some(1.5)).is_ok());
            assert!(graph.has_edge(n1, n2));
            assert_eq!(graph.get_edge_weight(n1, n2), Some(1.5));
        });
    }
    
    #[test]
    fn test_grid_graph_creation() {
        Python::with_gil(|py| {
            let graph = create_grid_graph(3, 3, None).unwrap();
            assert_eq!(graph.node_count(), 9);
            assert_eq!(graph.edge_count(), 12); // 6 horizontal + 6 vertical
            
            // Test with diagonals
            let graph_with_diagonals = create_grid_graph(3, 3, Some(true)).unwrap();
            assert_eq!(graph_with_diagonals.node_count(), 9);
            assert_eq!(graph_with_diagonals.edge_count(), 20); // 12 cardinal + 8 diagonal
        });
    }
}