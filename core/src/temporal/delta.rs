//! Differential state computation
//!
//! This module provides an efficient state delta computation system that enables
//! memory-efficient storage of algorithm execution states and high-performance
//! bidirectional navigation through the execution timeline.
//!
//! The system employs structural delta encoding to represent the minimal set of
//! changes between algorithm states, supporting both forward and backward state
//! transitions with optimal space complexity.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use log::{debug, trace};

use crate::algorithm::{AlgorithmState, NodeId};
use crate::utils::math;
use crate::temporal::compression::CompressedState;

/// State delta encoding type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeltaEncodingType {
    /// Full state snapshot (baseline for delta chains)
    FullSnapshot,
    
    /// Structural delta (minimal set changes)
    StructuralDelta,
    
    /// Incremental delta (depends on previous state)
    IncrementalDelta,
    
    /// Compressed delta (information-theoretic encoding)
    CompressedDelta,
}

/// State modification operation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateOperation {
    /// Add an element to a set
    AddToSet {
        /// The set to modify
        set_name: String,
        /// The element to add
        element: NodeId,
    },
    
    /// Remove an element from a set
    RemoveFromSet {
        /// The set to modify
        set_name: String,
        /// The element to remove
        element: NodeId,
    },
    
    /// Update a scalar value
    UpdateScalar {
        /// The scalar name
        scalar_name: String,
        /// The new value (serialized)
        value: String,
    },
    
    /// Update a map entry
    UpdateMapEntry {
        /// The map name
        map_name: String,
        /// The key (serialized)
        key: String,
        /// The value (serialized)
        value: Option<String>,
    },
    
    /// Set the current node
    SetCurrentNode {
        /// The new current node
        node: Option<NodeId>,
    },
    
    /// Custom operation (for algorithm-specific state)
    Custom {
        /// Operation type
        operation_type: String,
        /// Operation data (serialized)
        data: String,
    },
}

impl StateOperation {
    /// Compute the reverse operation
    pub fn reverse(&self, prev_state: &AlgorithmState) -> StateOperation {
        match self {
            StateOperation::AddToSet { set_name, element } => {
                StateOperation::RemoveFromSet {
                    set_name: set_name.clone(),
                    element: *element,
                }
            }
            
            StateOperation::RemoveFromSet { set_name, element } => {
                StateOperation::AddToSet {
                    set_name: set_name.clone(),
                    element: *element,
                }
            }
            
            StateOperation::UpdateScalar { scalar_name, value: _ } => {
                // Get previous value
                let prev_value = prev_state.data.get(scalar_name)
                    .map(|v| v.clone())
                    .unwrap_or_default();
                
                StateOperation::UpdateScalar {
                    scalar_name: scalar_name.clone(),
                    value: prev_value,
                }
            }
            
            StateOperation::UpdateMapEntry { map_name, key, value: _ } => {
                // Get previous value
                let prev_value = match map_name.as_str() {
                    "g_scores" => prev_state.g_scores.get(&key.parse::<NodeId>().unwrap_or_default()).map(|v| v.to_string()),
                    "f_scores" => prev_state.f_scores.get(&key.parse::<NodeId>().unwrap_or_default()).map(|v| v.to_string()),
                    _ => None,
                };
                
                StateOperation::UpdateMapEntry {
                    map_name: map_name.clone(),
                    key: key.clone(),
                    value: prev_value,
                }
            }
            
            StateOperation::SetCurrentNode { node: _ } => {
                StateOperation::SetCurrentNode {
                    node: prev_state.current_node,
                }
            }
            
            StateOperation::Custom { operation_type, data: _ } => {
                // Custom operations must provide their own reverse logic
                // This is a placeholder that would be replaced with actual implementation
                StateOperation::Custom {
                    operation_type: format!("reverse_{}", operation_type),
                    data: "{}".to_string(),
                }
            }
        }
    }
    
    /// Apply the operation to a state
    pub fn apply(&self, state: &mut AlgorithmState) -> Result<(), DeltaError> {
        match self {
            StateOperation::AddToSet { set_name, element } => {
                match set_name.as_str() {
                    "open_set" => {
                        state.open_set.push(*element);
                        Ok(())
                    }
                    "closed_set" => {
                        state.closed_set.push(*element);
                        Ok(())
                    }
                    _ => Err(DeltaError::UnknownSet(set_name.clone())),
                }
            }
            
            StateOperation::RemoveFromSet { set_name, element } => {
                match set_name.as_str() {
                    "open_set" => {
                        state.open_set.retain(|&e| e != *element);
                        Ok(())
                    }
                    "closed_set" => {
                        state.closed_set.retain(|&e| e != *element);
                        Ok(())
                    }
                    _ => Err(DeltaError::UnknownSet(set_name.clone())),
                }
            }
            
            StateOperation::UpdateScalar { scalar_name, value } => {
                state.data.insert(scalar_name.clone(), value.clone());
                Ok(())
            }
            
            StateOperation::UpdateMapEntry { map_name, key, value } => {
                match map_name.as_str() {
                    "g_scores" => {
                        let node_id = key.parse::<NodeId>().map_err(|_| DeltaError::InvalidKey(key.clone()))?;
                        
                        match value {
                            Some(v) => {
                                let score = v.parse::<f32>().map_err(|_| DeltaError::InvalidValue(v.clone()))?;
                                state.g_scores.insert(node_id, score);
                            }
                            None => {
                                state.g_scores.remove(&node_id);
                            }
                        }
                        
                        Ok(())
                    }
                    "f_scores" => {
                        let node_id = key.parse::<NodeId>().map_err(|_| DeltaError::InvalidKey(key.clone()))?;
                        
                        match value {
                            Some(v) => {
                                let score = v.parse::<f32>().map_err(|_| DeltaError::InvalidValue(v.clone()))?;
                                state.f_scores.insert(node_id, score);
                            }
                            None => {
                                state.f_scores.remove(&node_id);
                            }
                        }
                        
                        Ok(())
                    }
                    _ => Err(DeltaError::UnknownMap(map_name.clone())),
                }
            }
            
            StateOperation::SetCurrentNode { node } => {
                state.current_node = *node;
                Ok(())
            }
            
            StateOperation::Custom { operation_type, data } => {
                // Custom operations would be handled by algorithm-specific code
                // This is a placeholder
                debug!("Applying custom operation: {} with data: {}", operation_type, data);
                Err(DeltaError::CustomOperationNotSupported(operation_type.clone()))
            }
        }
    }
    
    /// Get the estimated memory impact of this operation in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            StateOperation::AddToSet { set_name, .. } | 
            StateOperation::RemoveFromSet { set_name, .. } => {
                // Set name + node ID (usually a small integer)
                set_name.len() + std::mem::size_of::<NodeId>()
            }
            
            StateOperation::UpdateScalar { scalar_name, value } => {
                scalar_name.len() + value.len()
            }
            
            StateOperation::UpdateMapEntry { map_name, key, value } => {
                map_name.len() + key.len() + value.as_ref().map_or(0, |v| v.len())
            }
            
            StateOperation::SetCurrentNode { .. } => {
                std::mem::size_of::<Option<NodeId>>()
            }
            
            StateOperation::Custom { operation_type, data } => {
                operation_type.len() + data.len()
            }
        }
    }
}

/// Unique identifier for a state
pub type StateId = u64;

/// State delta computation error
#[derive(Debug, thiserror::Error)]
pub enum DeltaError {
    /// Unknown set name
    #[error("Unknown set: {0}")]
    UnknownSet(String),
    
    /// Unknown map name
    #[error("Unknown map: {0}")]
    UnknownMap(String),
    
    /// Invalid key
    #[error("Invalid key: {0}")]
    InvalidKey(String),
    
    /// Invalid value
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    /// Custom operation not supported
    #[error("Custom operation not supported: {0}")]
    CustomOperationNotSupported(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Delta application error
    #[error("Delta application error: {0}")]
    ApplicationError(String),
    
    /// Missing base state
    #[error("Missing base state for incremental delta")]
    MissingBaseState,
    
    /// Invalid delta chain
    #[error("Invalid delta chain, expected base state ID {0}, got {1}")]
    InvalidDeltaChain(StateId, StateId),
}

/// State delta representing the changes between two algorithm states
#[derive(Clone, Serialize, Deserialize)]
pub struct StateDelta {
    /// Delta encoding type
    pub encoding_type: DeltaEncodingType,
    
    /// Base state ID (for incremental deltas)
    pub base_state_id: Option<StateId>,
    
    /// Target state ID
    pub target_state_id: StateId,
    
    /// Step number in the algorithm execution
    pub step: usize,
    
    /// Operations to transform from base to target state
    pub operations: Vec<StateOperation>,
    
    /// Full state snapshot (for full snapshots or for efficient delta chaining)
    pub snapshot: Option<CompressedState>,
    
    /// Size of this delta in bytes
    #[serde(skip)]
    pub size_bytes: usize,
    
    /// Creation timestamp
    #[serde(skip)]
    pub created_at: std::time::Instant,
}

impl StateDelta {
    /// Create a new state delta
    pub fn new(
        encoding_type: DeltaEncodingType,
        base_state_id: Option<StateId>,
        target_state_id: StateId,
        step: usize,
        operations: Vec<StateOperation>,
        snapshot: Option<CompressedState>,
    ) -> Self {
        // Calculate size
        let ops_size: usize = operations.iter().map(|op| op.memory_size()).sum();
        let snapshot_size = snapshot.as_ref().map_or(0, |s| s.size_bytes());
        
        Self {
            encoding_type,
            base_state_id,
            target_state_id,
            step,
            operations,
            snapshot,
            size_bytes: ops_size + snapshot_size,
            created_at: std::time::Instant::now(),
        }
    }
    
    /// Create a full snapshot delta
    pub fn new_snapshot(
        state_id: StateId,
        step: usize,
        snapshot: CompressedState,
    ) -> Self {
        Self::new(
            DeltaEncodingType::FullSnapshot,
            None,
            state_id,
            step,
            Vec::new(),
            Some(snapshot),
        )
    }
    
    /// Apply this delta to a state
    pub fn apply_to(&self, state: &mut AlgorithmState) -> Result<(), DeltaError> {
        match self.encoding_type {
            DeltaEncodingType::FullSnapshot => {
                // Replace with snapshot
                if let Some(snapshot) = &self.snapshot {
                    *state = snapshot.decompress().map_err(|e| DeltaError::ApplicationError(e.to_string()))?;
                    Ok(())
                } else {
                    Err(DeltaError::ApplicationError("Missing snapshot for full snapshot delta".to_string()))
                }
            }
            
            DeltaEncodingType::StructuralDelta | DeltaEncodingType::IncrementalDelta | DeltaEncodingType::CompressedDelta => {
                // Apply operations
                for op in &self.operations {
                    op.apply(state)?;
                }
                
                // Update step
                state.step = self.step;
                
                Ok(())
            }
        }
    }
    
    /// Create a delta between two states
    pub fn compute(
        base_state: &AlgorithmState,
        target_state: &AlgorithmState,
        base_id: Option<StateId>,
        target_id: StateId,
    ) -> Self {
        let mut operations = Vec::new();
        
        // Compare current_node
        if base_state.current_node != target_state.current_node {
            operations.push(StateOperation::SetCurrentNode {
                node: target_state.current_node,
            });
        }
        
        // Compare open_set
        let base_open: HashSet<NodeId> = base_state.open_set.iter().copied().collect();
        let target_open: HashSet<NodeId> = target_state.open_set.iter().copied().collect();
        
        // Added to open set
        for &node in target_open.difference(&base_open) {
            operations.push(StateOperation::AddToSet {
                set_name: "open_set".to_string(),
                element: node,
            });
        }
        
        // Removed from open set
        for &node in base_open.difference(&target_open) {
            operations.push(StateOperation::RemoveFromSet {
                set_name: "open_set".to_string(),
                element: node,
            });
        }
        
        // Compare closed_set
        let base_closed: HashSet<NodeId> = base_state.closed_set.iter().copied().collect();
        let target_closed: HashSet<NodeId> = target_state.closed_set.iter().copied().collect();
        
        // Added to closed set
        for &node in target_closed.difference(&base_closed) {
            operations.push(StateOperation::AddToSet {
                set_name: "closed_set".to_string(),
                element: node,
            });
        }
        
        // Removed from closed set
        for &node in base_closed.difference(&target_closed) {
            operations.push(StateOperation::RemoveFromSet {
                set_name: "closed_set".to_string(),
                element: node,
            });
        }
        
        // Compare g_scores
        let all_nodes: HashSet<NodeId> = base_state.g_scores.keys()
            .chain(target_state.g_scores.keys())
            .copied()
            .collect();
        
        for node in all_nodes {
            let base_score = base_state.g_scores.get(&node);
            let target_score = target_state.g_scores.get(&node);
            
            if base_score != target_score {
                operations.push(StateOperation::UpdateMapEntry {
                    map_name: "g_scores".to_string(),
                    key: node.to_string(),
                    value: target_score.map(|s| s.to_string()),
                });
            }
        }
        
        // Compare f_scores
        let all_nodes: HashSet<NodeId> = base_state.f_scores.keys()
            .chain(target_state.f_scores.keys())
            .copied()
            .collect();
        
        for node in all_nodes {
            let base_score = base_state.f_scores.get(&node);
            let target_score = target_state.f_scores.get(&node);
            
            if base_score != target_score {
                operations.push(StateOperation::UpdateMapEntry {
                    map_name: "f_scores".to_string(),
                    key: node.to_string(),
                    value: target_score.map(|s| s.to_string()),
                });
            }
        }
        
        // Compare custom data
        let all_keys: HashSet<String> = base_state.data.keys()
            .chain(target_state.data.keys())
            .cloned()
            .collect();
        
        for key in all_keys {
            let base_value = base_state.data.get(&key);
            let target_value = target_state.data.get(&key);
            
            if base_value != target_value {
                operations.push(StateOperation::UpdateScalar {
                    scalar_name: key,
                    value: target_value.map(|v| v.clone()).unwrap_or_default(),
                });
            }
        }
        
        // Check if we need a full snapshot
        let optimal_encoding = if operations.len() > 20 { // Heuristic threshold
            DeltaEncodingType::FullSnapshot
        } else {
            DeltaEncodingType::StructuralDelta
        };
        
        match optimal_encoding {
            DeltaEncodingType::FullSnapshot => {
                // Create compressed snapshot
                let snapshot = CompressedState::from_state(target_state);
                Self::new_snapshot(target_id, target_state.step, snapshot)
            }
            DeltaEncodingType::StructuralDelta => {
                Self::new(
                    DeltaEncodingType::StructuralDelta,
                    base_id,
                    target_id,
                    target_state.step,
                    operations,
                    None,
                )
            }
            _ => unreachable!()
        }
    }
    
    /// Create a reverse delta (for backward navigation)
    pub fn reverse(&self, base_state: &AlgorithmState) -> Result<Self, DeltaError> {
        match self.encoding_type {
            DeltaEncodingType::FullSnapshot => {
                // For snapshots, we need to create a new snapshot of the base state
                let snapshot = CompressedState::from_state(base_state);
                Ok(Self::new_snapshot(
                    self.base_state_id.unwrap_or(0),
                    base_state.step,
                    snapshot,
                ))
            }
            
            DeltaEncodingType::StructuralDelta | DeltaEncodingType::IncrementalDelta => {
                // Reverse operations in reverse order
                let mut rev_operations = Vec::with_capacity(self.operations.len());
                
                for op in self.operations.iter().rev() {
                    rev_operations.push(op.reverse(base_state));
                }
                
                Ok(Self::new(
                    self.encoding_type,
                    Some(self.target_state_id),
                    self.base_state_id.unwrap_or(0),
                    base_state.step,
                    rev_operations,
                    None,
                ))
            }
            
            DeltaEncodingType::CompressedDelta => {
                // For compressed deltas, we need a different approach
                // This is a placeholder
                Err(DeltaError::ApplicationError("Reverse not implemented for compressed deltas".to_string()))
            }
        }
    }
    
    /// Optimize this delta for storage
    pub fn optimize(&mut self) -> Result<(), DeltaError> {
        match self.encoding_type {
            DeltaEncodingType::FullSnapshot => {
                // Already a snapshot, nothing to optimize
                Ok(())
            }
            
            DeltaEncodingType::StructuralDelta => {
                // Check if we should convert to a compressed delta
                if self.operations.len() > 10 {
                    self.encoding_type = DeltaEncodingType::CompressedDelta;
                    
                    // This would involve additional compression, currently a placeholder
                    debug!("Converted structural delta to compressed delta");
                }
                Ok(())
            }
            
            DeltaEncodingType::IncrementalDelta | DeltaEncodingType::CompressedDelta => {
                // Already optimized
                Ok(())
            }
        }
    }
    
    /// Get the size of this delta in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
    
    /// Check if this delta is a full snapshot
    pub fn is_snapshot(&self) -> bool {
        self.encoding_type == DeltaEncodingType::FullSnapshot
    }
    
    /// Check if this delta can be applied to a state with the given ID
    pub fn is_applicable_to(&self, state_id: StateId) -> bool {
        match self.encoding_type {
            DeltaEncodingType::FullSnapshot => true, // Snapshots can be applied to any state
            DeltaEncodingType::StructuralDelta | DeltaEncodingType::IncrementalDelta | DeltaEncodingType::CompressedDelta => {
                // Need matching base state ID
                self.base_state_id.map_or(false, |id| id == state_id)
            }
        }
    }
}

impl fmt::Debug for StateDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateDelta")
            .field("encoding_type", &self.encoding_type)
            .field("base_state_id", &self.base_state_id)
            .field("target_state_id", &self.target_state_id)
            .field("step", &self.step)
            .field("operations", &self.operations.len())
            .field("has_snapshot", &self.snapshot.is_some())
            .field("size_bytes", &self.size_bytes)
            .finish()
    }
}

/// Delta chain for efficient state reconstruction
pub struct DeltaChain {
    /// Chain ID
    pub id: String,
    
    /// Base state ID
    pub base_state_id: StateId,
    
    /// Target state ID
    pub target_state_id: StateId,
    
    /// Chain of deltas
    pub deltas: Vec<Arc<StateDelta>>,
    
    /// Total size in bytes
    pub size_bytes: usize,
}

impl DeltaChain {
    /// Create a new delta chain
    pub fn new(
        base_state_id: StateId,
        target_state_id: StateId,
        deltas: Vec<Arc<StateDelta>>,
    ) -> Self {
        let id = format!("chain_{}_{}", base_state_id, target_state_id);
        let size_bytes = deltas.iter().map(|d| d.size_bytes).sum();
        
        Self {
            id,
            base_state_id,
            target_state_id,
            deltas,
            size_bytes,
        }
    }
    
    /// Apply this chain to a state
    pub fn apply_to(&self, state: &mut AlgorithmState) -> Result<(), DeltaError> {
        // Validate base state
        if self.deltas[0].base_state_id != Some(self.base_state_id) && !self.deltas[0].is_snapshot() {
            return Err(DeltaError::InvalidDeltaChain(
                self.base_state_id,
                self.deltas[0].base_state_id.unwrap_or(0),
            ));
        }
        
        // Apply deltas in order
        for delta in &self.deltas {
            delta.apply_to(state)?;
        }
        
        Ok(())
    }
    
    /// Optimize this chain
    pub fn optimize(&mut self) -> Result<(), DeltaError> {
        if self.deltas.len() <= 3 {
            // Short chains are already optimal
            return Ok(());
        }
        
        // Check total size
        let avg_delta_size = self.size_bytes / self.deltas.len();
        let snapshot_threshold = avg_delta_size * 3; // Heuristic
        
        // Create optimized chain
        let mut optimized = Vec::with_capacity(self.deltas.len() / 3 + 1);
        
        // Start with first delta
        optimized.push(Arc::clone(&self.deltas[0]));
        
        let mut current_size = 0;
        let mut last_snapshot_idx = 0;
        
        for (i, delta) in self.deltas.iter().enumerate().skip(1) {
            current_size += delta.size_bytes;
            
            // Add delta to chain
            optimized.push(Arc::clone(delta));
            
            // Check if we should insert a snapshot
            if current_size > snapshot_threshold && i - last_snapshot_idx > 5 {
                // This would involve creating a new snapshot delta
                // Currently a placeholder
                debug!("Would insert snapshot at delta {}", i);
                
                current_size = 0;
                last_snapshot_idx = i;
            }
        }
        
        // Update chain
        self.deltas = optimized;
        self.size_bytes = self.deltas.iter().map(|d| d.size_bytes).sum();
        
        Ok(())
    }
    
    /// Create a chain from a single delta
    pub fn from_delta(delta: Arc<StateDelta>) -> Self {
        let base_id = delta.base_state_id.unwrap_or(0);
        let target_id = delta.target_state_id;
        
        Self::new(base_id, target_id, vec![delta])
    }
    
    /// Get the total size of this chain in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    /// Create a test algorithm state
    fn create_test_state(step: usize) -> AlgorithmState {
        let mut state = AlgorithmState {
            step,
            open_set: vec![1, 2, 3],
            closed_set: vec![4, 5],
            current_node: Some(3),
            g_scores: HashMap::new(),
            f_scores: HashMap::new(),
            data: HashMap::new(),
        };
        
        state.g_scores.insert(1, 0.0);
        state.g_scores.insert(2, 1.0);
        state.g_scores.insert(3, 2.0);
        
        state.f_scores.insert(1, 5.0);
        state.f_scores.insert(2, 4.0);
        state.f_scores.insert(3, 3.0);
        
        state.data.insert("visited".to_string(), "3".to_string());
        
        state
    }
    
    #[test]
    fn test_state_operation_apply() {
        let mut state = create_test_state(1);
        
        // Test AddToSet
        let op = StateOperation::AddToSet {
            set_name: "open_set".to_string(),
            element: 6,
        };
        
        assert!(op.apply(&mut state).is_ok());
        assert!(state.open_set.contains(&6));
        
        // Test RemoveFromSet
        let op = StateOperation::RemoveFromSet {
            set_name: "open_set".to_string(),
            element: 2,
        };
        
        assert!(op.apply(&mut state).is_ok());
        assert!(!state.open_set.contains(&2));
        
        // Test UpdateScalar
        let op = StateOperation::UpdateScalar {
            scalar_name: "visited".to_string(),
            value: "4".to_string(),
        };
        
        assert!(op.apply(&mut state).is_ok());
        assert_eq!(state.data.get("visited"), Some(&"4".to_string()));
        
        // Test SetCurrentNode
        let op = StateOperation::SetCurrentNode {
            node: Some(6),
        };
        
        assert!(op.apply(&mut state).is_ok());
        assert_eq!(state.current_node, Some(6));
    }
    
    #[test]
    fn test_compute_delta() {
        let state1 = create_test_state(1);
        let mut state2 = state1.clone();
        
        // Modify state2
        state2.step = 2;
        state2.open_set.push(6);
        state2.open_set.retain(|&n| n != 1);
        state2.closed_set.push(1);
        state2.current_node = Some(6);
        state2.g_scores.insert(6, 3.0);
        state2.f_scores.insert(6, 6.0);
        state2.data.insert("visited".to_string(), "6".to_string());
        
        // Compute delta
        let delta = StateDelta::compute(&state1, &state2, Some(1), 2);
        
        // Check delta
        assert_eq!(delta.step, 2);
        assert_eq!(delta.base_state_id, Some(1));
        assert_eq!(delta.target_state_id, 2);
        assert!(!delta.operations.is_empty());
        
        // Apply delta
        let mut state1_copy = state1.clone();
        assert!(delta.apply_to(&mut state1_copy).is_ok());
        
        // Check result
        assert_eq!(state1_copy.step, state2.step);
        assert_eq!(state1_copy.open_set, state2.open_set);
        assert_eq!(state1_copy.closed_set, state2.closed_set);
        assert_eq!(state1_copy.current_node, state2.current_node);
        assert_eq!(state1_copy.g_scores, state2.g_scores);
        assert_eq!(state1_copy.f_scores, state2.f_scores);
        assert_eq!(state1_copy.data, state2.data);
    }
    
    #[test]
    fn test_reverse_delta() {
        let state1 = create_test_state(1);
        let mut state2 = state1.clone();
        
        // Modify state2
        state2.step = 2;
        state2.open_set.push(6);
        state2.current_node = Some(6);
        
        // Compute delta
        let delta = StateDelta::compute(&state1, &state2, Some(1), 2);
        
        // Reverse delta
        let reverse_delta = delta.reverse(&state1).unwrap();
        
        // Apply forward delta
        let mut state1_copy = state1.clone();
        assert!(delta.apply_to(&mut state1_copy).is_ok());
        assert_eq!(state1_copy.step, 2);
        assert!(state1_copy.open_set.contains(&6));
        
        // Apply reverse delta
        assert!(reverse_delta.apply_to(&mut state1_copy).is_ok());
        
        // Check we're back to state1
        assert_eq!(state1_copy.step, state1.step);
        assert_eq!(state1_copy.open_set, state1.open_set);
        assert_eq!(state1_copy.current_node, state1.current_node);
    }
}