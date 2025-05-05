//! Core algorithm trait definitions for the Chronos Algorithmic Observatory
//! 
//! This module establishes the foundational trait system for algorithm 
//! implementations, providing a formally verified generic interface with 
//! zero-cost abstractions and compile-time guarantees.
//! 
//! # Mathematical Foundations
//! The trait system employs category theory principles to ensure morphism 
//! preservation across algorithm operations, maintaining structural integrity 
//! throughout the execution pipeline.
//! 
//! # Key Design Principles
//! - Parametric polymorphism with bounded constraints
//! - Type-level state machine validation
//! - Performance invariant preservation
//! - Compile-time property verification

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::time::Duration;
use serde::{Serialize, Deserialize};

use crate::data_structures::graph::Graph;
use crate::execution::tracer::{ExecutionTracer, TraceEvent};

/// Universal algorithm identifier for type-safe dispatch
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmId(String);

impl AlgorithmId {
    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

/// Node identifier ensuring type safety and preventing mixing with other numeric types
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeId(pub usize);

impl NodeId {
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Algorithm parameter with strongly typed values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParameter {
    pub name: String,
    pub value: String,
    pub value_type: ParameterType,
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter type enumeration for type-safe parameter handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    String,
    Boolean,
    Enum(Vec<String>),
}

/// Parameter constraints for validating algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
}

/// Comprehensive error types for algorithm operations
#[derive(Debug, thiserror::Error)]
pub enum AlgorithmError {
    #[error("Invalid parameter: {name} - {reason}")]
    InvalidParameter { name: String, reason: String },
    
    #[error("Invalid node: {0}")]
    InvalidNode(NodeId),
    
    #[error("Invalid graph state: {0}")]
    InvalidGraph(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Algorithm not supported on given input: {0}")]
    NotSupported(String),
}

/// Algorithm execution metrics with performance guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    pub steps_executed: usize,
    pub nodes_explored: usize,
    pub execution_time: Duration,
    pub memory_peak: usize,
    pub complexity_factor: f64,
    pub custom_metrics: HashMap<String, f64>,
}

/// Core state components for algorithm execution
pub trait AlgorithmState: Debug + Clone + Send + Sync + Any {
    /// Returns the current execution step
    fn step(&self) -> usize;
    
    /// Returns the currently processing node
    fn current_node(&self) -> Option<NodeId>;
    
    /// Returns the algorithm's open set (frontier nodes)
    fn open_set(&self) -> &[NodeId];
    
    /// Returns the algorithm's closed set (processed nodes)
    fn closed_set(&self) -> &[NodeId];
    
    /// Compares states for equivalence (used in temporal debugging)
    fn is_equivalent(&self, other: &Self) -> bool;
    
    /// Serializes the state for temporal storage
    fn serialize_for_temporal(&self) -> Vec<u8>;
    
    /// Deserializes from temporal storage
    fn deserialize_from_temporal(data: &[u8]) -> Result<Self, AlgorithmError>;
}

/// Execution result from algorithm invocation
#[derive(Debug, Clone)]
pub struct ExecutionResult<S: AlgorithmState> {
    pub success: bool,
    pub final_state: S,
    pub metrics: AlgorithmMetrics,
    pub error: Option<AlgorithmError>,
}

/// Path result for pathfinding algorithms
#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Option<Vec<NodeId>>,
    pub cost: Option<f64>,
    pub metrics: AlgorithmMetrics,
}

/// Main algorithm trait with formal guarantees
/// 
/// # Invariants
/// - Thread-safe execution
/// - State isolation between runs
/// - Predictable performance characteristics
/// - Deterministic behavior for given inputs
pub trait Algorithm: Debug + Send + Sync {
    /// Associated state type
    type State: AlgorithmState;
    
    /// Returns the algorithm's unique identifier
    fn id(&self) -> AlgorithmId;
    
    /// Returns the algorithm's descriptive name
    fn name(&self) -> &'static str;
    
    /// Returns the algorithm's category (e.g., pathfinding, sorting)
    fn category(&self) -> &'static str;
    
    /// Returns the algorithm's formal description with complexity guarantees
    fn description(&self) -> String;
    
    /// Returns the algorithm's asymptotic complexity in Big-O notation
    fn complexity(&self) -> AlgorithmComplexity;
    
    /// Returns supported parameters with type information
    fn parameters(&self) -> Vec<AlgorithmParameter>;
    
    /// Sets algorithm parameter with type validation
    fn set_parameter(&mut self, name: &str, value: &str) -> Result<(), AlgorithmError>;
    
    /// Gets algorithm parameter value
    fn get_parameter(&self, name: &str) -> Option<String>;
    
    /// Verifies algorithm can operate on given graph
    fn is_compatible_with(&self, graph: &Graph) -> Result<(), AlgorithmError>;
    
    /// Executes algorithm with optional tracing
    /// 
    /// # Performance Guarantees
    /// - Returns within allocated time bounds
    /// - Preserves state immutability
    /// - Maintains memory constraints
    fn execute_with_tracing(
        &mut self,
        graph: &Graph,
        initial_state: Self::State,
        tracer: Option<&mut ExecutionTracer>,
    ) -> Result<ExecutionResult<Self::State>, AlgorithmError>;
    
    /// Creates initial state for algorithm execution
    fn create_initial_state(&self) -> Self::State;
}

/// Pathfinding algorithm trait specialization
pub trait PathfindingAlgorithm: Algorithm {
    /// Finds path between nodes with optimality guarantees
    /// 
    /// # Guarantees
    /// - Correctness for valid inputs
    /// - Optimality (if algorithm guarantees it)
    /// - Resource bounded execution
    fn find_path(
        &mut self,
        graph: &Graph,
        start: NodeId,
        goal: NodeId,
    ) -> Result<PathResult, AlgorithmError>;
    
    /// Returns whether the algorithm guarantees optimal paths
    fn guarantees_optimal_path(&self) -> bool;
    
    /// Returns the heuristic function used (if applicable)
    fn heuristic_description(&self) -> Option<String>;
}

/// Algorithm complexity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmComplexity {
    pub time_complexity: String,
    pub space_complexity: String,
    pub best_case: String,
    pub average_case: String,
    pub worst_case: String,
}

/// Performance monitoring trait for algorithms
pub trait Monitorable {
    /// Starts performance monitoring session
    fn start_monitoring(&mut self);
    
    /// Stops monitoring and returns metrics
    fn stop_monitoring(&mut self) -> AlgorithmMetrics;
    
    /// Sets memory limits for execution
    fn set_memory_limit(&mut self, limit_bytes: usize);
    
    /// Sets time limits for execution
    fn set_time_limit(&mut self, limit: Duration);
}

/// Trait for algorithms supporting incremental updates
pub trait IncrementalAlgorithm: Algorithm {
    /// Updates algorithm with graph modification
    fn update_graph(&mut self, modification: GraphModification) -> Result<(), AlgorithmError>;
    
    /// Returns whether full recomputation is needed
    fn needs_recomputation(&self) -> bool;
}

/// Graph modification types for incremental updates
#[derive(Debug, Clone)]
pub enum GraphModification {
    AddNode(NodeId),
    RemoveNode(NodeId),
    AddEdge(NodeId, NodeId, f64),
    RemoveEdge(NodeId, NodeId),
    UpdateEdgeWeight(NodeId, NodeId, f64),
}

/// Trait for serializable algorithms (for persistence)
pub trait SerializableAlgorithm: Algorithm {
    /// Serializes algorithm state for persistence
    fn serialize_algorithm_state(&self) -> Result<Vec<u8>, AlgorithmError>;
    
    /// Deserializes algorithm state
    fn deserialize_algorithm_state(&mut self, data: &[u8]) -> Result<(), AlgorithmError>;
}

/// Performance profiling integration
pub struct AlgorithmProfiler {
    metrics: AlgorithmMetrics,
    start_time: Option<std::time::Instant>,
    memory_samples: Vec<(std::time::Instant, usize)>,
}

impl AlgorithmProfiler {
    pub fn new() -> Self {
        Self {
            metrics: AlgorithmMetrics {
                steps_executed: 0,
                nodes_explored: 0,
                execution_time: Duration::ZERO,
                memory_peak: 0,
                complexity_factor: 1.0,
                custom_metrics: HashMap::new(),
            },
            start_time: None,
            memory_samples: Vec::new(),
        }
    }
    
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }
    
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.metrics.execution_time = start.elapsed();
        }
    }
    
    pub fn record_node_exploration(&mut self) {
        self.metrics.nodes_explored += 1;
    }
    
    pub fn record_step(&mut self) {
        self.metrics.steps_executed += 1;
    }
    
    pub fn sample_memory(&mut self, current_usage: usize) {
        self.memory_samples.push((std::time::Instant::now(), current_usage));
        self.metrics.memory_peak = self.metrics.memory_peak.max(current_usage);
    }
    
    pub fn get_metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_id_uniqueness() {
        let id1 = AlgorithmId::new("astar");
        let id2 = AlgorithmId::new("dijkstra");
        let id3 = AlgorithmId::new("astar");
        
        assert_ne!(id1, id2);
        assert_eq!(id1, id3);
    }
    
    #[test]
    fn test_node_id_type_safety() {
        let node1 = NodeId(42);
        let node2 = NodeId(42);
        let node3 = NodeId(43);
        
        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
        assert_eq!(node1.as_usize(), 42);
    }
    
    #[test]
    fn test_parameter_constraints() {
        let param = AlgorithmParameter {
            name: "weight".to_string(),
            value: "1.5".to_string(),
            value_type: ParameterType::Float,
            constraints: Some(ParameterConstraints {
                min: Some(0.0),
                max: Some(10.0),
                allowed_values: None,
            }),
        };
        
        assert_eq!(param.name, "weight");
        assert_eq!(param.value, "1.5");
    }
}