//! Execution history management for algorithm tracing
//!
//! This module provides comprehensive execution history management for algorithm
//! execution, enabling bidirectional navigation, counterfactual analysis, and
//! detailed algorithm behavior inspection with minimal memory overhead.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

use crate::execution::tracer::{ExecutionTracer, TracePoint};
use crate::temporal::delta::StateDelta;
use crate::temporal::compression::StateCompressor;
use crate::algorithm::state::AlgorithmState;

/// Execution history manager for algorithm traces
///
/// Maintains an efficient, queryable record of algorithm execution
/// with state transition history and decision point identification.
#[derive(Debug)]
pub struct ExecutionHistory {
    /// Trace metadata
    metadata: ExecutionMetadata,
    
    /// Complete trace points in execution order
    trace_points: Vec<Arc<TracePoint>>,
    
    /// Decision points with their indices in the trace
    decision_points: Vec<DecisionPointEntry>,
    
    /// State compressor for efficient storage
    compressor: StateCompressor,
    
    /// State transition deltas between consecutive states
    state_deltas: Vec<StateDelta>,
    
    /// Full state snapshots at regular intervals
    state_snapshots: BTreeMap<usize, Arc<AlgorithmState>>,
    
    /// Index structures for efficient querying
    indices: ExecutionIndices,
}

/// Metadata for execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Algorithm name
    pub algorithm_name: String,
    
    /// Algorithm parameters
    pub parameters: HashMap<String, String>,
    
    /// Execution start timestamp
    pub start_time: DateTime<Utc>,
    
    /// Execution end timestamp
    pub end_time: Option<DateTime<Utc>>,
    
    /// Total number of execution steps
    pub step_count: usize,
    
    /// Custom metadata for algorithm-specific information
    pub custom: HashMap<String, String>,
}

/// Decision point in execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPointEntry {
    /// Index in the trace
    pub index: usize,
    
    /// Step number in execution
    pub step: usize,
    
    /// Decision type
    pub decision_type: String,
    
    /// Decision description
    pub description: String,
    
    /// Decision importance score (0.0-1.0)
    pub importance: f64,
    
    /// Alternative branches from this decision point
    pub alternatives: Vec<AlternativeBranch>,
}

/// Alternative branch from a decision point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeBranch {
    /// Branch ID
    pub branch_id: String,
    
    /// Branch description
    pub description: String,
    
    /// Branch creation timestamp
    pub creation_time: DateTime<Utc>,
    
    /// Reference to branch history if available
    pub branch_history: Option<Arc<ExecutionHistory>>,
}

/// Index structures for efficient querying
#[derive(Debug)]
struct ExecutionIndices {
    /// Maps state signature to trace point index
    state_signatures: HashMap<String, usize>,
    
    /// Maps node ID to trace points where it was visited
    node_visits: HashMap<usize, Vec<usize>>,
    
    /// Maps decision type to trace points containing that decision
    decision_types: HashMap<String, Vec<usize>>,
    
    /// Maps custom event types to trace points
    custom_events: HashMap<String, Vec<usize>>,
}

/// Error types for execution history operations
#[derive(Debug, Error)]
pub enum HistoryError {
    #[error("Failed to create history: {0}")]
    CreationError(String),
    
    #[error("Invalid trace point index: {0}")]
    InvalidIndex(usize),
    
    #[error("State compression error: {0}")]
    CompressionError(#[from] crate::temporal::compression::CompressionError),
    
    #[error("State delta error: {0}")]
    DeltaError(#[from] crate::temporal::delta::DeltaError),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Decision point error: {0}")]
    DecisionPointError(String),
    
    #[error("Branch creation error: {0}")]
    BranchCreationError(String),
    
    #[error("Other history error: {0}")]
    Other(String),
}

/// Serializable history format for storage/transmission
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedHistory {
    /// Metadata
    pub metadata: ExecutionMetadata,
    
    /// Trace points (excluding state data)
    pub trace_points: Vec<SerializedTracePoint>,
    
    /// Decision points
    pub decision_points: Vec<DecisionPointEntry>,
    
    /// Compressed state deltas
    pub compressed_deltas: Vec<u8>,
    
    /// Snapshot indices
    pub snapshot_indices: Vec<usize>,
    
    /// Compressed state snapshots
    pub compressed_snapshots: Vec<u8>,
}

/// Serialized trace point (without full state data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTracePoint {
    /// Step number
    pub step: usize,
    
    /// Current node
    pub current_node: Option<usize>,
    
    /// Whether this is a decision point
    pub is_decision_point: bool,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// State signature
    pub state_signature: String,
}

impl ExecutionHistory {
    /// Create a new execution history from a tracer
    pub fn from_tracer(tracer: &ExecutionTracer) -> Result<Self, HistoryError> {
        let trace = tracer.get_trace()
            .map_err(|e| HistoryError::CreationError(e.to_string()))?;
        
        // Create metadata
        let metadata = ExecutionMetadata {
            algorithm_name: trace.algorithm_name.clone(),
            parameters: trace.parameters.clone(),
            start_time: trace.start_time,
            end_time: trace.end_time,
            step_count: trace.points.len(),
            custom: HashMap::new(),
        };
        
        // Create state compressor
        let mut compressor = StateCompressor::new();
        
        // Process trace points
        let mut trace_points = Vec::with_capacity(trace.points.len());
        let mut state_deltas = Vec::with_capacity(trace.points.len() - 1);
        let mut state_snapshots = BTreeMap::new();
        let mut decision_points = Vec::new();
        
        // Initialize indices
        let mut indices = ExecutionIndices {
            state_signatures: HashMap::new(),
            node_visits: HashMap::new(),
            decision_types: HashMap::new(),
            custom_events: HashMap::new(),
        };
        
        // Process trace points and build indices
        for (i, point) in trace.points.iter().enumerate() {
            // Compute state signature
            let state_signature = compressor.compute_signature(&point.state)
                .map_err(|e| HistoryError::CompressionError(e))?;
            
            // Create trace point
            let trace_point = Arc::new(point.clone());
            trace_points.push(trace_point.clone());
            
            // Add to indices
            indices.state_signatures.insert(state_signature, i);
            
            if let Some(node) = point.current_node {
                indices.node_visits.entry(node)
                    .or_insert_with(Vec::new)
                    .push(i);
            }
            
            // Create state snapshots at regular intervals
            if i % trace.config.capture_interval == 0 || point.is_decision_point {
                state_snapshots.insert(i, Arc::new(point.state.clone()));
            }
            
            // Compute state delta if not the first point
            if i > 0 {
                let prev_state = &trace.points[i - 1].state;
                let curr_state = &point.state;
                
                let delta = StateDelta::compute(prev_state, curr_state)
                    .map_err(|e| HistoryError::DeltaError(e))?;
                
                state_deltas.push(delta);
            }
            
            // Process decision points
            if point.is_decision_point {
                let decision_type = point.decision_info.as_ref()
                    .map(|info| info.decision_type.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                
                let description = point.decision_info.as_ref()
                    .map(|info| info.description.clone())
                    .unwrap_or_else(|| "Unknown decision".to_string());
                
                let importance = point.decision_info.as_ref()
                    .map(|info| info.importance)
                    .unwrap_or(0.5);
                
                let decision_point = DecisionPointEntry {
                    index: i,
                    step: point.step,
                    decision_type: decision_type.clone(),
                    description,
                    importance,
                    alternatives: Vec::new(),
                };
                
                decision_points.push(decision_point);
                
                // Add to decision type index
                indices.decision_types.entry(decision_type)
                    .or_insert_with(Vec::new)
                    .push(i);
            }
            
            // Process custom events
            if let Some(events) = &point.custom_events {
                for event in events {
                    indices.custom_events.entry(event.event_type.clone())
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }
        }
        
        Ok(Self {
            metadata,
            trace_points,
            decision_points,
            compressor,
            state_deltas,
            state_snapshots,
            indices,
        })
    }
    
    /// Get the total number of steps in the execution history
    pub fn step_count(&self) -> usize {
        self.metadata.step_count
    }
    
    /// Get trace point at specific index
    pub fn get_trace_point(&self, index: usize) -> Result<Arc<TracePoint>, HistoryError> {
        self.trace_points.get(index)
            .cloned()
            .ok_or_else(|| HistoryError::InvalidIndex(index))
    }
    
    /// Get trace point at specific step
    pub fn get_trace_point_by_step(&self, step: usize) -> Result<Arc<TracePoint>, HistoryError> {
        self.trace_points.iter()
            .find(|point| point.step == step)
            .cloned()
            .ok_or_else(|| HistoryError::InvalidIndex(step))
    }
    
    /// Get decision points in execution order
    pub fn get_decision_points(&self) -> &[DecisionPointEntry] {
        &self.decision_points
    }
    
    /// Find trace points where a specific node was visited
    pub fn find_node_visits(&self, node_id: usize) -> Vec<usize> {
        self.indices.node_visits.get(&node_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
    
    /// Find trace points with a specific decision type
    pub fn find_decision_type(&self, decision_type: &str) -> Vec<usize> {
        self.indices.decision_types.get(decision_type)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
    
    /// Find trace points with a specific custom event type
    pub fn find_custom_events(&self, event_type: &str) -> Vec<usize> {
        self.indices.custom_events.get(event_type)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
    
    /// Get algorithm state at a specific index
    pub fn get_state(&self, index: usize) -> Result<Arc<AlgorithmState>, HistoryError> {
        if index >= self.trace_points.len() {
            return Err(HistoryError::InvalidIndex(index));
        }
        
        // Check if we have a snapshot for this index
        if let Some(snapshot) = self.state_snapshots.get(&index) {
            return Ok(snapshot.clone());
        }
        
        // Find the closest snapshot before this index
        let snapshot_index = self.state_snapshots.range(..index)
            .rev()
            .next()
            .map(|(&idx, _)| idx)
            .ok_or_else(|| HistoryError::Other("No state snapshot found".to_string()))?;
        
        let base_state = self.state_snapshots.get(&snapshot_index)
            .ok_or_else(|| HistoryError::Other("State snapshot not found".to_string()))?
            .clone();
        
        // Apply deltas from the snapshot to the target index
        let mut current_state = (*base_state).clone();
        
        for i in snapshot_index..index {
            if i >= self.state_deltas.len() {
                break;
            }
            
            self.state_deltas[i].apply(&mut current_state)
                .map_err(|e| HistoryError::DeltaError(e))?;
        }
        
        Ok(Arc::new(current_state))
    }
    
    /// Get state delta between two indices
    pub fn get_state_delta(&self, from_idx: usize, to_idx: usize) -> Result<StateDelta, HistoryError> {
        if from_idx >= self.trace_points.len() || to_idx >= self.trace_points.len() {
            return Err(HistoryError::InvalidIndex(from_idx.max(to_idx)));
        }
        
        if from_idx == to_idx {
            return Ok(StateDelta::empty());
        }
        
        // Get states at both indices
        let from_state = self.get_state(from_idx)?;
        let to_state = self.get_state(to_idx)?;
        
        // Compute delta
        StateDelta::compute(&from_state, &to_state)
            .map_err(|e| HistoryError::DeltaError(e))
    }
    
    /// Create a branch from a decision point
    pub fn create_branch(&mut self, decision_point_index: usize, description: &str) -> Result<String, HistoryError> {
        // Find the decision point
        let decision_point = self.decision_points.iter_mut()
            .find(|dp| dp.index == decision_point_index)
            .ok_or_else(|| HistoryError::DecisionPointError(
                format!("Decision point not found at index {}", decision_point_index)
            ))?;
        
        // Generate branch ID
        let branch_id = format!("branch-{}-{}", decision_point_index, Utc::now().timestamp_micros());
        
        // Create alternative branch
        let branch = AlternativeBranch {
            branch_id: branch_id.clone(),
            description: description.to_string(),
            creation_time: Utc::now(),
            branch_history: None,
        };
        
        decision_point.alternatives.push(branch);
        
        Ok(branch_id)
    }
    
    /// Set branch history for an alternative branch
    pub fn set_branch_history(&mut self, branch_id: &str, history: Arc<ExecutionHistory>) -> Result<(), HistoryError> {
        // Find the branch
        for decision_point in &mut self.decision_points {
            for alternative in &mut decision_point.alternatives {
                if alternative.branch_id == branch_id {
                    alternative.branch_history = Some(history);
                    return Ok(());
                }
            }
        }
        
        Err(HistoryError::BranchCreationError(
            format!("Branch not found with ID {}", branch_id)
        ))
    }
    
    /// Export history to a serializable format
    pub fn export(&self) -> Result<SerializedHistory, HistoryError> {
        // Serialize trace points without full state data
        let trace_points = self.trace_points.iter()
            .map(|point| {
                let state_signature = self.compressor.compute_signature(&point.state)
                    .map_err(|e| HistoryError::CompressionError(e))?;
                
                Ok(SerializedTracePoint {
                    step: point.step,
                    current_node: point.current_node,
                    is_decision_point: point.is_decision_point,
                    timestamp: point.timestamp,
                    state_signature,
                })
            })
            .collect::<Result<Vec<_>, HistoryError>>()?;
        
        // Compress state deltas
        let compressed_deltas = self.compressor.compress_deltas(&self.state_deltas)
            .map_err(|e| HistoryError::CompressionError(e))?;
        
        // Serialize state snapshots
        let snapshot_indices = self.state_snapshots.keys().cloned().collect::<Vec<_>>();
        let snapshots = self.state_snapshots.values()
            .map(|s| (**s).clone())
            .collect::<Vec<_>>();
        
        let compressed_snapshots = self.compressor.compress_states(&snapshots)
            .map_err(|e| HistoryError::CompressionError(e))?;
        
        Ok(SerializedHistory {
            metadata: self.metadata.clone(),
            trace_points,
            decision_points: self.decision_points.clone(),
            compressed_deltas,
            snapshot_indices,
            compressed_snapshots,
        })
    }
    
    /// Import history from serialized format
    pub fn import(serialized: SerializedHistory) -> Result<Self, HistoryError> {
        // Create state compressor
        let mut compressor = StateCompressor::new();
        
        // Decompress state snapshots
        let snapshots = compressor.decompress_states(&serialized.compressed_snapshots)
            .map_err(|e| HistoryError::CompressionError(e))?;
        
        // Rebuild state snapshot map
        let mut state_snapshots = BTreeMap::new();
        for (i, index) in serialized.snapshot_indices.iter().enumerate() {
            if i < snapshots.len() {
                state_snapshots.insert(*index, Arc::new(snapshots[i].clone()));
            }
        }
        
        // Decompress state deltas
        let state_deltas = compressor.decompress_deltas(&serialized.compressed_deltas)
            .map_err(|e| HistoryError::CompressionError(e))?;
        
        // Rebuild trace points
        let mut trace_points = Vec::with_capacity(serialized.trace_points.len());
        let mut reconstructed_states = HashMap::new();
        
        // Reconstruct states and trace points
        for (i, serialized_point) in serialized.trace_points.iter().enumerate() {
            // Get or reconstruct state
            let state = if let Some(snapshot) = state_snapshots.get(&i) {
                // Use existing snapshot
                (**snapshot).clone()
            } else {
                // Find closest snapshot and apply deltas
                let snapshot_index = state_snapshots.range(..i)
                    .rev()
                    .next()
                    .map(|(&idx, _)| idx)
                    .ok_or_else(|| HistoryError::Other("No state snapshot found".to_string()))?;
                
                let base_state = state_snapshots.get(&snapshot_index)
                    .ok_or_else(|| HistoryError::Other("State snapshot not found".to_string()))?;
                
                let mut current_state = (**base_state).clone();
                
                for j in snapshot_index..i {
                    if j - snapshot_index >= state_deltas.len() {
                        break;
                    }
                    
                    state_deltas[j - snapshot_index].apply(&mut current_state)
                        .map_err(|e| HistoryError::DeltaError(e))?;
                }
                
                current_state
            };
            
            // Create decision info if applicable
            let decision_info = if serialized_point.is_decision_point {
                serialized.decision_points.iter()
                    .find(|dp| dp.index == i)
                    .map(|dp| crate::execution::tracer::DecisionInfo {
                        decision_type: dp.decision_type.clone(),
                        description: dp.description.clone(),
                        importance: dp.importance,
                    })
            } else {
                None
            };
            
            // Create trace point
            let trace_point = TracePoint {
                step: serialized_point.step,
                current_node: serialized_point.current_node,
                is_decision_point: serialized_point.is_decision_point,
                timestamp: serialized_point.timestamp,
                state,
                decision_info,
                custom_events: None,
            };
            
            trace_points.push(Arc::new(trace_point));
        }
        
        // Initialize indices
        let mut indices = ExecutionIndices {
            state_signatures: HashMap::new(),
            node_visits: HashMap::new(),
            decision_types: HashMap::new(),
            custom_events: HashMap::new(),
        };
        
        // Rebuild indices
        for (i, point) in trace_points.iter().enumerate() {
            // Add to state signature index
            let state_signature = compressor.compute_signature(&point.state)
                .map_err(|e| HistoryError::CompressionError(e))?;
            
            indices.state_signatures.insert(state_signature, i);
            
            // Add to node visits index
            if let Some(node) = point.current_node {
                indices.node_visits.entry(node)
                    .or_insert_with(Vec::new)
                    .push(i);
            }
            
            // Add to decision type index
            if point.is_decision_point {
                if let Some(info) = &point.decision_info {
                    indices.decision_types.entry(info.decision_type.clone())
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }
            
            // Add to custom events index
            if let Some(events) = &point.custom_events {
                for event in events {
                    indices.custom_events.entry(event.event_type.clone())
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }
        }
        
        Ok(Self {
            metadata: serialized.metadata,
            trace_points,
            decision_points: serialized.decision_points,
            compressor,
            state_deltas,
            state_snapshots,
            indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::tracer::{ExecutionTracer, ExecutionConfig};
    use crate::algorithm::state::AlgorithmState;
    
    #[test]
    fn test_execution_history_creation() {
        // Create a tracer with some trace points
        let mut tracer = ExecutionTracer::new();
        tracer.start_trace("test_algorithm").unwrap();
        
        // Add trace points
        for i in 0..10 {
            let state = AlgorithmState {
                step: i,
                open_set: vec![i, i + 1],
                closed_set: if i > 0 { vec![i - 1] } else { vec![] },
                current_node: Some(i),
                data: HashMap::new(),
            };
            
            let is_decision = i % 3 == 0;
            let decision_info = if is_decision {
                Some(crate::execution::tracer::DecisionInfo {
                    decision_type: "test_decision".to_string(),
                    description: format!("Decision at step {}", i),
                    importance: 0.5,
                })
            } else {
                None
            };
            
            tracer.add_trace_point_with_info(
                i,
                Some(i),
                state,
                is_decision,
                decision_info,
                None,
            ).unwrap();
        }
        
        tracer.end_trace().unwrap();
        
        // Create execution history
        let history = ExecutionHistory::from_tracer(&tracer).unwrap();
        
        // Check metadata
        assert_eq!(history.metadata.algorithm_name, "test_algorithm");
        assert_eq!(history.metadata.step_count, 10);
        
        // Check trace points
        assert_eq!(history.trace_points.len(), 10);
        
        // Check decision points
        assert_eq!(history.decision_points.len(), 4); // Steps 0, 3, 6, 9
        
        // Check node visits
        for i in 0..10 {
            let visits = history.find_node_visits(i);
            assert_eq!(visits.len(), 1);
            assert_eq!(visits[0], i);
        }
        
        // Check decision types
        let decisions = history.find_decision_type("test_decision");
        assert_eq!(decisions.len(), 4);
        assert_eq!(decisions, vec![0, 3, 6, 9]);
    }
    
    #[test]
    fn test_state_retrieval() {
        // Create a tracer with some trace points
        let mut tracer = ExecutionTracer::with_config(ExecutionConfig {
            capture_interval: 5,
            ..Default::default()
        });
        
        tracer.start_trace("test_algorithm").unwrap();
        
        // Add trace points
        for i in 0..20 {
            let mut state = AlgorithmState {
                step: i,
                open_set: vec![i, i + 1],
                closed_set: if i > 0 { vec![i - 1] } else { vec![] },
                current_node: Some(i),
                data: HashMap::new(),
            };
            
            // Add some data
            state.data.insert(format!("key_{}", i), format!("value_{}", i));
            
            tracer.add_trace_point(i, Some(i), state).unwrap();
        }
        
        tracer.end_trace().unwrap();
        
        // Create execution history
        let history = ExecutionHistory::from_tracer(&tracer).unwrap();
        
        // Check state snapshots
        assert!(history.state_snapshots.contains_key(&0));
        assert!(history.state_snapshots.contains_key(&5));
        assert!(history.state_snapshots.contains_key(&10));
        assert!(history.state_snapshots.contains_key(&15));
        
        // Get states at various points
        for i in 0..20 {
            let state = history.get_state(i).unwrap();
            
            // Verify state properties
            assert_eq!(state.step, i);
            assert_eq!(state.current_node, Some(i));
            assert_eq!(state.open_set, vec![i, i + 1]);
            assert_eq!(state.closed_set, if i > 0 { vec![i - 1] } else { vec![] });
            assert_eq!(state.data.get(&format!("key_{}", i)), Some(&format!("value_{}", i)));
        }
    }
    
    #[test]
    fn test_state_delta() {
        // Create a tracer with some trace points
        let mut tracer = ExecutionTracer::new();
        tracer.start_trace("test_algorithm").unwrap();
        
        // Add trace points with specific state changes
        for i in 0..5 {
            let mut state = AlgorithmState {
                step: i,
                open_set: vec![i, i + 1, i + 2],
                closed_set: if i > 0 { vec![0..i] } else { vec![] },
                current_node: Some(i),
                data: HashMap::new(),
            };
            
            // Add some data
            state.data.insert(format!("key_{}", i), format!("value_{}", i));
            
            tracer.add_trace_point(i, Some(i), state).unwrap();
        }
        
        tracer.end_trace().unwrap();
        
        // Create execution history
        let history = ExecutionHistory::from_tracer(&tracer).unwrap();
        
        // Test delta computation
        let delta_0_4 = history.get_state_delta(0, 4).unwrap();
        
        // Verify delta can transform state 0 to state 4
        let mut state_0 = (*history.get_state(0).unwrap()).clone();
        delta_0_4.apply(&mut state_0).unwrap();
        
        let state_4 = history.get_state(4).unwrap();
        
        // Verify key properties
        assert_eq!(state_0.step, state_4.step);
        assert_eq!(state_0.current_node, state_4.current_node);
        assert_eq!(state_0.open_set, state_4.open_set);
        assert_eq!(state_0.closed_set, state_4.closed_set);
        assert_eq!(state_0.data.get("key_4"), state_4.data.get("key_4"));
    }
    
    #[test]
    fn test_serialization() {
        // Create a tracer with some trace points
        let mut tracer = ExecutionTracer::new();
        tracer.start_trace("test_algorithm").unwrap();
        
        // Add trace points
        for i in 0..10 {
            let state = AlgorithmState {
                step: i,
                open_set: vec![i, i + 1],
                closed_set: if i > 0 { vec![i - 1] } else { vec![] },
                current_node: Some(i),
                data: HashMap::new(),
            };
            
            tracer.add_trace_point(i, Some(i), state).unwrap();
        }
        
        tracer.end_trace().unwrap();
        
        // Create execution history
        let history = ExecutionHistory::from_tracer(&tracer).unwrap();
        
        // Export to serialized format
        let serialized = history.export().unwrap();
        
        // Import from serialized format
        let imported = ExecutionHistory::import(serialized).unwrap();
        
        // Verify key properties
        assert_eq!(imported.metadata.algorithm_name, "test_algorithm");
        assert_eq!(imported.metadata.step_count, 10);
        assert_eq!(imported.trace_points.len(), 10);
        
        // Verify states
        for i in 0..10 {
            let original_state = history.get_state(i).unwrap();
            let imported_state = imported.get_state(i).unwrap();
            
            assert_eq!(imported_state.step, original_state.step);
            assert_eq!(imported_state.current_node, original_state.current_node);
            assert_eq!(imported_state.open_set, original_state.open_set);
            assert_eq!(imported_state.closed_set, original_state.closed_set);
        }
    }
}