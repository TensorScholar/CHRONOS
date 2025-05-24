//! Temporal system Python bindings
//!
//! This module provides Python bindings for the Chronos temporal debugging
//! system, enabling advanced timeline manipulation, counterfactual analysis,
//! and state exploration through a seamless bridge between the high-performance
//! Rust core and the Python educational layer.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyTuple, PyBool, PyType};
use pyo3::PyObjectProtocol;
use pyo3::exceptions::PyException;
use pyo3::wrap_pyfunction;
use pyo3::PyGCProtocol;
use log::{debug, info, error, warn};

use chronos_core::algorithm::{Algorithm, AlgorithmState, NodeId, PathResult};
use chronos_core::temporal::{
    StateManager, TimelineBranch, StateDelta, StateId, DeltaChain,
    compression::CompressedState, storage::{StorageTier, StorageConfig, DiskStorage}
};

use crate::algorithm::{PyAlgorithm, PyAlgorithmState};
use crate::data_structures::PyGraph;
use crate::utils::py_datetime_from_instant;

/// Create a Python module for temporal bindings
#[pymodule]
pub fn temporal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTimelineManager>()?;
    m.add_class::<PyTimeline>()?;
    m.add_class::<PyTimelineBranch>()?;
    m.add_class::<PyStateModifier>()?;
    
    m.add_function(wrap_pyfunction!(create_timeline_manager, m)?)?;
    
    Ok(())
}

/// Timeline manager Python wrapper
#[pyclass(name = "TimelineManager")]
pub struct PyTimelineManager {
    /// Inner timeline manager
    pub inner: Arc<Mutex<StateManager>>,
    
    /// Current timeline
    pub current_timeline: Option<String>,
    
    /// Decision points
    pub decision_points: Vec<StateId>,
    
    /// Timelines
    pub timelines: HashMap<String, PyTimeline>,
    
    /// Storage configuration
    pub storage_config: StorageConfig,
}

#[pymethods]
impl PyTimelineManager {
    /// Create a new timeline manager
    #[new]
    fn new(py: Python, storage_dir: Option<String>) -> PyResult<Self> {
        // Create storage configuration
        let mut config = StorageConfig::default();
        
        if let Some(dir) = storage_dir {
            config.base_dir = PathBuf::from(dir);
        }
        
        // Create storage
        let storage = match DiskStorage::new(config.clone()) {
            Ok(storage) => storage,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to create storage: {}", e))),
        };
        
        // Create state manager
        let manager = match StateManager::new(storage) {
            Ok(manager) => manager,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to create state manager: {}", e))),
        };
        
        Ok(Self {
            inner: Arc::new(Mutex::new(manager)),
            current_timeline: None,
            decision_points: Vec::new(),
            timelines: HashMap::new(),
            storage_config: config,
        })
    }
    
    /// Create a new timeline
    fn create_timeline(&mut self, py: Python, name: String) -> PyResult<PyTimeline> {
        let timeline = PyTimeline {
            name: name.clone(),
            manager: Arc::clone(&self.inner),
            current_state_id: None,
            states: Vec::new(),
            branches: HashMap::new(),
        };
        
        self.timelines.insert(name.clone(), timeline.clone());
        self.current_timeline = Some(name);
        
        Ok(timeline)
    }
    
    /// Get a timeline by name
    fn get_timeline(&self, py: Python, name: String) -> PyResult<PyTimeline> {
        if let Some(timeline) = self.timelines.get(&name) {
            Ok(timeline.clone())
        } else {
            Err(PyValueError::new_err(format!("Timeline not found: {}", name)))
        }
    }
    
    /// Get the current timeline
    fn get_current_timeline(&self, py: Python) -> PyResult<PyTimeline> {
        if let Some(name) = &self.current_timeline {
            self.get_timeline(py, name.clone())
        } else {
            Err(PyValueError::new_err("No current timeline"))
        }
    }
    
    /// Set the current timeline
    fn set_current_timeline(&mut self, name: String) -> PyResult<()> {
        if self.timelines.contains_key(&name) {
            self.current_timeline = Some(name);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("Timeline not found: {}", name)))
        }
    }
    
    /// Get all timeline names
    fn get_timeline_names(&self, py: Python) -> PyResult<PyObject> {
        let names: Vec<&String> = self.timelines.keys().collect();
        let py_list = PyList::new(py, names);
        Ok(py_list.into())
    }
    
    /// Get storage statistics
    fn get_storage_stats(&self, py: Python) -> PyResult<PyObject> {
        let manager = match self.inner.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        let stats = manager.get_storage_stats();
        
        let dict = PyDict::new(py);
        dict.set_item("memory_states", stats.memory_states)?;
        dict.set_item("cache_states", stats.cache_states)?;
        dict.set_item("disk_states", stats.disk_states)?;
        dict.set_item("archive_states", stats.archive_states)?;
        dict.set_item("memory_size", stats.memory_size)?;
        dict.set_item("cache_size", stats.cache_size)?;
        dict.set_item("disk_size", stats.disk_size)?;
        dict.set_item("archive_size", stats.archive_size)?;
        dict.set_item("cache_hits", stats.cache_hits)?;
        dict.set_item("cache_misses", stats.cache_misses)?;
        dict.set_item("disk_reads", stats.disk_reads)?;
        dict.set_item("disk_writes", stats.disk_writes)?;
        
        Ok(dict.into())
    }
    
    /// Add a state to a timeline
    fn add_state(&mut self, py: Python, timeline_name: String, state: &PyAlgorithmState) -> PyResult<StateId> {
        let mut manager = match self.inner.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        let state_id = match manager.add_state(&state.inner) {
            Ok(id) => id,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to add state: {}", e))),
        };
        
        // Add to timeline
        if let Some(timeline) = self.timelines.get_mut(&timeline_name) {
            timeline.states.push(state_id);
            timeline.current_state_id = Some(state_id);
        } else {
            return Err(PyValueError::new_err(format!("Timeline not found: {}", timeline_name)));
        }
        
        Ok(state_id)
    }
    
    /// Mark a state as a decision point
    fn mark_decision_point(&mut self, state_id: StateId) -> PyResult<()> {
        let mut manager = match self.inner.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        match manager.mark_decision_point(state_id) {
            Ok(_) => {
                if !self.decision_points.contains(&state_id) {
                    self.decision_points.push(state_id);
                }
                Ok(())
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to mark decision point: {}", e))),
        }
    }
    
    /// Get all decision points
    fn get_decision_points(&self, py: Python) -> PyResult<PyObject> {
        let py_list = PyList::new(py, &self.decision_points);
        Ok(py_list.into())
    }
    
    /// Close the timeline manager
    fn close(&self) -> PyResult<()> {
        let mut manager = match self.inner.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        match manager.close() {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to close manager: {}", e))),
        }
    }
    
    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("TimelineManager(timelines={}, current={})",
            self.timelines.len(),
            self.current_timeline.as_ref().unwrap_or(&"None".to_string())
        ))
    }
    
    /// String representation
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

impl Drop for PyTimelineManager {
    fn drop(&mut self) {
        debug!("TimelineManager dropped");
        
        // Attempt to close the manager gracefully
        if let Ok(mut manager) = self.inner.lock() {
            if let Err(e) = manager.close() {
                error!("Failed to close state manager: {}", e);
            }
        }
    }
}

/// Timeline Python wrapper
#[pyclass(name = "Timeline")]
#[derive(Clone)]
pub struct PyTimeline {
    /// Timeline name
    pub name: String,
    
    /// State manager
    pub manager: Arc<Mutex<StateManager>>,
    
    /// Current state ID
    pub current_state_id: Option<StateId>,
    
    /// States in this timeline
    pub states: Vec<StateId>,
    
    /// Branches from this timeline
    pub branches: HashMap<String, PyTimelineBranch>,
}

#[pymethods]
impl PyTimeline {
    /// Step forward in the timeline
    fn step_forward(&mut self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state index
        let current_idx = if let Some(current_id) = self.current_state_id {
            self.states.iter().position(|&id| id == current_id)
                .ok_or_else(|| PyValueError::new_err("Current state not found in timeline"))?
        } else {
            return Err(PyValueError::new_err("No current state"));
        };
        
        // Check if we're at the end
        if current_idx >= self.states.len() - 1 {
            return Err(PyValueError::new_err("Already at the end of timeline"));
        }
        
        // Get next state ID
        let next_id = self.states[current_idx + 1];
        
        // Get state
        let state = match manager.get_state(next_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(next_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Step backward in the timeline
    fn step_backward(&mut self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state index
        let current_idx = if let Some(current_id) = self.current_state_id {
            self.states.iter().position(|&id| id == current_id)
                .ok_or_else(|| PyValueError::new_err("Current state not found in timeline"))?
        } else {
            return Err(PyValueError::new_err("No current state"));
        };
        
        // Check if we're at the beginning
        if current_idx == 0 {
            return Err(PyValueError::new_err("Already at the beginning of timeline"));
        }
        
        // Get previous state ID
        let prev_id = self.states[current_idx - 1];
        
        // Get state
        let state = match manager.get_state(prev_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(prev_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Jump to a specific state in the timeline
    fn jump_to_state(&mut self, py: Python, state_id: StateId) -> PyResult<PyAlgorithmState> {
        // Check if state exists in this timeline
        if !self.states.contains(&state_id) {
            return Err(PyValueError::new_err("State not found in timeline"));
        }
        
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get state
        let state = match manager.get_state(state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(state_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Jump to a decision point
    fn jump_to_decision_point(&mut self, py: Python, decision_idx: usize) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get decision points in this timeline
        let decision_points: Vec<_> = self.states.iter()
            .filter(|&&id| manager.is_decision_point(id))
            .copied()
            .collect();
        
        // Check if index is valid
        if decision_idx >= decision_points.len() {
            return Err(PyValueError::new_err(format!(
                "Invalid decision point index: {}. Only {} decision points available",
                decision_idx, decision_points.len()
            )));
        }
        
        // Get decision point state ID
        let decision_id = decision_points[decision_idx];
        
        // Get state
        let state = match manager.get_state(decision_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(decision_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Get the current state
    fn get_current_state(&self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state ID
        let state_id = self.current_state_id
            .ok_or_else(|| PyValueError::new_err("No current state"))?;
        
        // Get state
        let state = match manager.get_state(state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Get the current state ID
    fn get_current_state_id(&self) -> PyResult<Option<StateId>> {
        Ok(self.current_state_id)
    }
    
    /// Get the current state index
    fn get_current_state_index(&self) -> PyResult<Option<usize>> {
        if let Some(id) = self.current_state_id {
            Ok(self.states.iter().position(|&s| s == id))
        } else {
            Ok(None)
        }
    }
    
    /// Get all states in the timeline
    fn get_states(&self, py: Python) -> PyResult<PyObject> {
        let py_list = PyList::new(py, &self.states);
        Ok(py_list.into())
    }
    
    /// Get the number of states in the timeline
    fn get_state_count(&self) -> PyResult<usize> {
        Ok(self.states.len())
    }
    
    /// Create a branch from the current state
    fn create_branch(&mut self, py: Python, name: String, modifier: &PyStateModifier) -> PyResult<PyTimelineBranch> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state ID
        let state_id = self.current_state_id
            .ok_or_else(|| PyValueError::new_err("No current state"))?;
        
        // Get current state
        let mut state = match manager.get_state(state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Create Python state
        let py_state = PyAlgorithmState {
            inner: state.clone(),
        };
        
        // Apply modifier
        if let Err(e) = modifier.call(py, py_state.into_py(py)) {
            return Err(e);
        }
        
        // Create branch
        let branch_id = match manager.create_branch(state_id) {
            Ok(id) => id,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to create branch: {}", e))),
        };
        
        // Create branch object
        let branch = PyTimelineBranch {
            name: name.clone(),
            parent_timeline: self.name.clone(),
            branch_id,
            base_state_id: state_id,
            manager: Arc::clone(&self.manager),
            current_state_id: Some(state_id),
            states: vec![state_id],
        };
        
        // Add to branches
        self.branches.insert(name, branch.clone());
        
        Ok(branch)
    }
    
    /// Get a branch by name
    fn get_branch(&self, py: Python, name: String) -> PyResult<PyTimelineBranch> {
        if let Some(branch) = self.branches.get(&name) {
            Ok(branch.clone())
        } else {
            Err(PyValueError::new_err(format!("Branch not found: {}", name)))
        }
    }
    
    /// Get all branch names
    fn get_branch_names(&self, py: Python) -> PyResult<PyObject> {
        let names: Vec<&String> = self.branches.keys().collect();
        let py_list = PyList::new(py, names);
        Ok(py_list.into())
    }
    
    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Timeline(name='{}', states={}, branches={})",
            self.name,
            self.states.len(),
            self.branches.len()
        ))
    }
    
    /// String representation
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

/// Timeline branch Python wrapper
#[pyclass(name = "TimelineBranch")]
#[derive(Clone)]
pub struct PyTimelineBranch {
    /// Branch name
    pub name: String,
    
    /// Parent timeline name
    pub parent_timeline: String,
    
    /// Branch ID
    pub branch_id: String,
    
    /// Base state ID
    pub base_state_id: StateId,
    
    /// State manager
    pub manager: Arc<Mutex<StateManager>>,
    
    /// Current state ID
    pub current_state_id: Option<StateId>,
    
    /// States in this branch
    pub states: Vec<StateId>,
}

#[pymethods]
impl PyTimelineBranch {
    /// Continue execution from the current state
    fn continue_execution(
        &mut self, 
        py: Python, 
        algorithm: &PyAlgorithm, 
        graph: &PyGraph, 
        start: NodeId, 
        goal: NodeId
    ) -> PyResult<PyObject> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state ID
        let state_id = self.current_state_id
            .ok_or_else(|| PyValueError::new_err("No current state"))?;
        
        // Continue execution
        let result = match manager.continue_execution(&mut algorithm.inner, &graph.inner, state_id, start, goal) {
            Ok(result) => result,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to continue execution: {}", e))),
        };
        
        // Update states list with new states
        for &id in &result.state_ids {
            if !self.states.contains(&id) {
                self.states.push(id);
            }
        }
        
        // Update current state to the last state
        if let Some(&last_id) = result.state_ids.last() {
            self.current_state_id = Some(last_id);
        }
        
        // Convert to Python result
        let dict = PyDict::new(py);
        dict.set_item("path", result.path)?;
        dict.set_item("cost", result.cost)?;
        dict.set_item("nodes_visited", result.nodes_visited)?;
        dict.set_item("execution_time_ms", result.execution_time_ms)?;
        dict.set_item("state_ids", result.state_ids)?;
        
        Ok(dict.into())
    }
    
    /// Step forward in the branch
    fn step_forward(&mut self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state index
        let current_idx = if let Some(current_id) = self.current_state_id {
            self.states.iter().position(|&id| id == current_id)
                .ok_or_else(|| PyValueError::new_err("Current state not found in branch"))?
        } else {
            return Err(PyValueError::new_err("No current state"));
        };
        
        // Check if we're at the end
        if current_idx >= self.states.len() - 1 {
            return Err(PyValueError::new_err("Already at the end of branch"));
        }
        
        // Get next state ID
        let next_id = self.states[current_idx + 1];
        
        // Get state
        let state = match manager.get_state(next_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(next_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Step backward in the branch
    fn step_backward(&mut self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state index
        let current_idx = if let Some(current_id) = self.current_state_id {
            self.states.iter().position(|&id| id == current_id)
                .ok_or_else(|| PyValueError::new_err("Current state not found in branch"))?
        } else {
            return Err(PyValueError::new_err("No current state"));
        };
        
        // Check if we're at the beginning
        if current_idx == 0 {
            return Err(PyValueError::new_err("Already at the beginning of branch"));
        }
        
        // Get previous state ID
        let prev_id = self.states[current_idx - 1];
        
        // Get state
        let state = match manager.get_state(prev_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(prev_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Jump to a specific state in the branch
    fn jump_to_state(&mut self, py: Python, state_id: StateId) -> PyResult<PyAlgorithmState> {
        // Check if state exists in this branch
        if !self.states.contains(&state_id) {
            return Err(PyValueError::new_err("State not found in branch"));
        }
        
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get state
        let state = match manager.get_state(state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Update current state
        self.current_state_id = Some(state_id);
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Get the current state
    fn get_current_state(&self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get current state ID
        let state_id = self.current_state_id
            .ok_or_else(|| PyValueError::new_err("No current state"))?;
        
        // Get state
        let state = match manager.get_state(state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get state: {}", e))),
        };
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Get the base state
    fn get_base_state(&self, py: Python) -> PyResult<PyAlgorithmState> {
        let mut manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get base state
        let state = match manager.get_state(self.base_state_id) {
            Ok(state) => state,
            Err(e) => return Err(PyValueError::new_err(format!("Failed to get base state: {}", e))),
        };
        
        // Convert to Python state
        Ok(PyAlgorithmState {
            inner: state,
        })
    }
    
    /// Compare with another branch
    fn compare_with(&self, py: Python, other: &PyTimelineBranch) -> PyResult<PyObject> {
        let manager = match self.manager.lock() {
            Ok(manager) => manager,
            Err(e) => return Err(PyRuntimeError::new_err(format!("Failed to lock state manager: {}", e))),
        };
        
        // Get all states in both branches
        let self_states: HashSet<StateId> = self.states.iter().copied().collect();
        let other_states: HashSet<StateId> = other.states.iter().copied().collect();
        
        // Find common and unique states
        let common: Vec<_> = self_states.intersection(&other_states).copied().collect();
        let only_self: Vec<_> = self_states.difference(&other_states).copied().collect();
        let only_other: Vec<_> = other_states.difference(&self_states).copied().collect();
        
        // Create comparison result
        let dict = PyDict::new(py);
        dict.set_item("common_states", common)?;
        dict.set_item("only_self", only_self)?;
        dict.set_item("only_other", only_other)?;
        dict.set_item("self_name", &self.name)?;
        dict.set_item("other_name", &other.name)?;
        
        Ok(dict.into())
    }
    
    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("TimelineBranch(name='{}', parent='{}', states={})",
            self.name,
            self.parent_timeline,
            self.states.len()
        ))
    }
    
    /// String representation
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

/// State modifier Python wrapper
#[pyclass(name = "StateModifier")]
pub struct PyStateModifier {
    /// Python callable
    callback: PyObject,
}

#[pymethods]
impl PyStateModifier {
    /// Create a new state modifier
    #[new]
    fn new(callback: PyObject) -> Self {
        Self {
            callback,
        }
    }
    
    /// Call the modifier
    pub fn call(&self, py: Python, state: PyObject) -> PyResult<()> {
        // Call the Python function
        self.callback.call1(py, (state,))?;
        Ok(())
    }
    
    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        Ok("StateModifier(...)".to_string())
    }
    
    /// String representation
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

/// Create a new timeline manager
#[pyfunction]
fn create_timeline_manager(py: Python, storage_dir: Option<String>) -> PyResult<PyTimelineManager> {
    PyTimelineManager::new(py, storage_dir)
}

/// State execution result
#[derive(Debug, Clone)]
pub struct PyExecutionResult {
    /// Path found
    pub path: Vec<NodeId>,
    
    /// Path cost
    pub cost: Option<f64>,
    
    /// Number of nodes visited
    pub nodes_visited: usize,
    
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    
    /// State IDs in the execution
    pub state_ids: Vec<StateId>,
}

impl From<chronos_core::temporal::ExecutionResult> for PyExecutionResult {
    fn from(result: chronos_core::temporal::ExecutionResult) -> Self {
        Self {
            path: result.path,
            cost: result.cost,
            nodes_visited: result.nodes_visited,
            execution_time_ms: result.execution_time_ms,
            state_ids: result.state_ids,
        }
    }
}