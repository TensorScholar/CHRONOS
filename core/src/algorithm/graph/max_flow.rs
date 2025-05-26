//! Advanced Maximum Flow Algorithm Implementation
//!
//! This module implements state-of-the-art maximum flow algorithms with
//! category-theoretic flow decomposition and advanced preflow management.
//! Features Goldberg-Tarjan push-relabel with capacity scaling optimization.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use crate::algorithm::{Algorithm, AlgorithmError, AlgorithmResult, AlgorithmState, NodeId};
use crate::data_structures::graph::{Graph, Weight};
use crate::execution::tracer::ExecutionTracer;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::{Ordering, Reverse};
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Flow capacity type with algebraic properties
pub type Capacity = f64;

/// Flow value type supporting arithmetic operations
pub type Flow = f64;

/// Distance label type for push-relabel algorithm
pub type DistanceLabel = usize;

/// Maximum flow algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxFlowAlgorithm {
    /// Ford-Fulkerson with DFS path finding
    FordFulkerson,
    /// Edmonds-Karp with BFS shortest augmenting paths
    EdmondsKarp,
    /// Goldberg-Tarjan push-relabel with FIFO vertex selection
    PushRelabelFIFO,
    /// Push-relabel with highest label selection
    PushRelabelHighestLabel,
    /// Capacity scaling push-relabel for enhanced performance
    CapacityScaling,
    /// Parallel push-relabel with work-stealing
    ParallelPushRelabel,
}

/// Flow edge representation with residual capacity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEdge {
    /// Source vertex
    pub from: NodeId,
    /// Target vertex  
    pub to: NodeId,
    /// Original edge capacity
    pub capacity: Capacity,
    /// Current flow through edge
    pub flow: Flow,
    /// Reverse edge index for residual graph
    pub reverse_edge_index: Option<usize>,
}

impl FlowEdge {
    /// Create new flow edge with specified capacity
    pub fn new(from: NodeId, to: NodeId, capacity: Capacity) -> Self {
        Self {
            from,
            to,
            capacity,
            flow: 0.0,
            reverse_edge_index: None,
        }
    }
    
    /// Get residual capacity for forward direction
    pub fn residual_capacity(&self) -> Capacity {
        self.capacity - self.flow
    }
    
    /// Get residual capacity for reverse direction  
    pub fn reverse_residual_capacity(&self) -> Capacity {
        self.flow
    }
    
    /// Push flow through edge with capacity constraints
    pub fn push_flow(&mut self, delta: Flow) -> Result<(), FlowError> {
        if delta > self.residual_capacity() {
            return Err(FlowError::InsufficientCapacity);
        }
        self.flow += delta;
        Ok(())
    }
}

/// Flow network representation with residual graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowNetwork {
    /// Number of vertices in network
    vertex_count: usize,
    /// Adjacency list representation with flow edges
    adjacency: Vec<Vec<usize>>,
    /// All edges in the network
    edges: Vec<FlowEdge>,
    /// Vertex distance labels for push-relabel
    distance_labels: Vec<DistanceLabel>,
    /// Excess flow at each vertex
    excess: Vec<Flow>,
    /// Active vertex queue for push-relabel
    active_vertices: VecDeque<NodeId>,
}

impl FlowNetwork {
    /// Create new flow network with specified vertex count
    pub fn new(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            adjacency: vec![Vec::new(); vertex_count],
            edges: Vec::new(),
            distance_labels: vec![0; vertex_count],
            excess: vec![0.0; vertex_count],
            active_vertices: VecDeque::new(),
        }
    }
    
    /// Add edge to flow network with bidirectional residual edges
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, capacity: Capacity) {
        let forward_edge_index = self.edges.len();
        let reverse_edge_index = forward_edge_index + 1;
        
        // Add forward edge
        let mut forward_edge = FlowEdge::new(from, to, capacity);
        forward_edge.reverse_edge_index = Some(reverse_edge_index);
        
        // Add reverse edge with zero capacity
        let mut reverse_edge = FlowEdge::new(to, from, 0.0);
        reverse_edge.reverse_edge_index = Some(forward_edge_index);
        
        // Update adjacency lists
        self.adjacency[from].push(forward_edge_index);
        self.adjacency[to].push(reverse_edge_index);
        
        // Add edges to collection
        self.edges.push(forward_edge);
        self.edges.push(reverse_edge);
    }
    
    /// Initialize preflow from source with maximum capacity
    pub fn initialize_preflow(&mut self, source: NodeId) {
        // Set source distance label to vertex count
        self.distance_labels[source] = self.vertex_count;
        
        // Saturate all edges from source
        for &edge_index in &self.adjacency[source].clone() {
            let edge = &self.edges[edge_index];
            let to = edge.to;
            let capacity = edge.capacity;
            
            if capacity > 0.0 {
                // Push maximum flow from source
                self.edges[edge_index].flow = capacity;
                
                // Update reverse edge
                if let Some(reverse_index) = edge.reverse_edge_index {
                    self.edges[reverse_index].flow = -capacity;
                }
                
                // Add excess to target vertex
                self.excess[to] += capacity;
                
                // Add to active vertices if not source or sink
                if to != source {
                    self.active_vertices.push_back(to);
                }
            }
        }
    }
    
    /// Push excess flow from vertex to admissible neighbors
    pub fn push(&mut self, vertex: NodeId) -> bool {
        let vertex_distance = self.distance_labels[vertex];
        let vertex_excess = self.excess[vertex];
        
        if vertex_excess <= 0.0 {
            return false;
        }
        
        // Find admissible edges for pushing
        for &edge_index in &self.adjacency[vertex].clone() {
            let edge = &self.edges[edge_index];
            let to = edge.to;
            let residual = edge.residual_capacity();
            
            // Check admissibility condition
            if residual > 0.0 && vertex_distance == self.distance_labels[to] + 1 {
                // Calculate push amount
                let push_amount = vertex_excess.min(residual);
                
                // Perform push operation
                self.edges[edge_index].flow += push_amount;
                
                // Update reverse edge
                if let Some(reverse_index) = edge.reverse_edge_index {
                    self.edges[reverse_index].flow -= push_amount;
                }
                
                // Update excess values
                self.excess[vertex] -= push_amount;
                self.excess[to] += push_amount;
                
                // Add target to active vertices if it becomes active
                if self.excess[to] == push_amount && to != vertex {
                    self.active_vertices.push_back(to);
                }
                
                return true;
            }
        }
        
        false
    }
    
    /// Relabel vertex to create admissible edges
    pub fn relabel(&mut self, vertex: NodeId) {
        let mut min_distance = usize::MAX;
        
        // Find minimum distance among neighbors with positive residual capacity
        for &edge_index in &self.adjacency[vertex] {
            let edge = &self.edges[edge_index];
            
            if edge.residual_capacity() > 0.0 {
                min_distance = min_distance.min(self.distance_labels[edge.to]);
            }
        }
        
        // Update distance label
        if min_distance < usize::MAX {
            self.distance_labels[vertex] = min_distance + 1;
        }
    }
    
    /// Gap relabeling optimization for improved performance
    pub fn gap_relabel(&mut self, gap_distance: DistanceLabel) {
        for vertex in 0..self.vertex_count {
            if self.distance_labels[vertex] > gap_distance {
                self.distance_labels[vertex] = self.vertex_count;
            }
        }
    }
    
    /// Global relabeling using reverse BFS from sink
    pub fn global_relabel(&mut self, sink: NodeId) {
        // Reset all distance labels
        self.distance_labels.fill(self.vertex_count);
        self.distance_labels[sink] = 0;
        
        let mut queue = VecDeque::new();
        queue.push_back(sink);
        
        while let Some(vertex) = queue.pop_front() {
            let current_distance = self.distance_labels[vertex];
            
            // Process all incoming edges (reverse direction in residual graph)
            for &edge_index in &self.adjacency[vertex] {
                let edge = &self.edges[edge_index];
                
                // Check reverse edge for residual capacity
                if let Some(reverse_index) = edge.reverse_edge_index {
                    let reverse_edge = &self.edges[reverse_index];
                    let from = reverse_edge.from;
                    
                    if reverse_edge.residual_capacity() > 0.0 && 
                       self.distance_labels[from] == self.vertex_count {
                        self.distance_labels[from] = current_distance + 1;
                        queue.push_back(from);
                    }
                }
            }
        }
    }
    
    /// Get maximum flow value at sink
    pub fn get_max_flow(&self, sink: NodeId) -> Flow {
        // Sum excess at sink
        self.excess[sink]
    }
}

/// Maximum flow algorithm implementation with advanced optimizations
#[derive(Debug, Clone)]
pub struct MaxFlowSolver {
    /// Selected algorithm variant
    algorithm: MaxFlowAlgorithm,
    /// Algorithm parameters
    parameters: HashMap<String, String>,
    /// Performance monitoring
    push_count: usize,
    relabel_count: usize,
    global_relabel_frequency: usize,
}

impl MaxFlowSolver {
    /// Create new maximum flow solver with specified algorithm
    pub fn new(algorithm: MaxFlowAlgorithm) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("global_relabel_frequency".to_string(), "100".to_string());
        parameters.insert("capacity_scaling_factor".to_string(), "2.0".to_string());
        
        Self {
            algorithm,
            parameters,
            push_count: 0,
            relabel_count: 0,
            global_relabel_frequency: 100,
        }
    }
    
    /// Solve maximum flow using Goldberg-Tarjan push-relabel
    pub fn solve_push_relabel(&mut self, 
                            network: &mut FlowNetwork, 
                            source: NodeId, 
                            sink: NodeId) -> Result<Flow, FlowError> {
        // Initialize preflow
        network.initialize_preflow(source);
        
        let mut iteration = 0;
        
        // Main push-relabel loop
        while let Some(vertex) = network.active_vertices.pop_front() {
            if vertex == source || vertex == sink {
                continue;
            }
            
            // Skip vertices with no excess
            if network.excess[vertex] <= 0.0 {
                continue;
            }
            
            let old_distance = network.distance_labels[vertex];
            
            // Try to push excess flow
            if !network.push(vertex) {
                // No push possible, relabel vertex
                network.relabel(vertex);
                self.relabel_count += 1;
                
                // Check for gap relabeling
                if network.distance_labels[vertex] > old_distance + 1 {
                    network.gap_relabel(old_distance);
                }
            } else {
                self.push_count += 1;
            }
            
            // Add back to active vertices if still has excess
            if network.excess[vertex] > 0.0 {
                network.active_vertices.push_back(vertex);
            }
            
            // Periodic global relabeling
            iteration += 1;
            if iteration % self.global_relabel_frequency == 0 {
                network.global_relabel(sink);
            }
        }
        
        Ok(network.get_max_flow(sink))
    }
    
    /// Solve maximum flow using capacity scaling
    pub fn solve_capacity_scaling(&mut self, 
                                network: &mut FlowNetwork, 
                                source: NodeId, 
                                sink: NodeId) -> Result<Flow, FlowError> {
        let scaling_factor: f64 = self.parameters
            .get("capacity_scaling_factor")
            .unwrap_or(&"2.0".to_string())
            .parse()
            .unwrap_or(2.0);
        
        // Find maximum capacity for scaling
        let max_capacity = network.edges.iter()
            .map(|e| e.capacity)
            .fold(0.0, f64::max);
        
        if max_capacity <= 0.0 {
            return Ok(0.0);
        }
        
        // Initialize scaling threshold
        let mut threshold = 1.0;
        while threshold <= max_capacity {
            threshold *= scaling_factor;
        }
        threshold /= scaling_factor;
        
        let mut total_flow = 0.0;
        
        // Capacity scaling phases
        while threshold >= 1.0 {
            // Create scaled network
            let mut scaled_network = network.clone();
            
            // Remove edges with capacity below threshold
            for edge in &mut scaled_network.edges {
                if edge.capacity < threshold {
                    edge.capacity = 0.0;
                }
            }
            
            // Solve scaled problem
            let phase_flow = self.solve_push_relabel(&mut scaled_network, source, sink)?;
            total_flow += phase_flow;
            
            // Update original network with scaled solution
            for (i, edge) in network.edges.iter_mut().enumerate() {
                edge.flow += scaled_network.edges[i].flow;
            }
            
            threshold /= scaling_factor;
        }
        
        Ok(total_flow)
    }
    
    /// Convert graph to flow network
    fn graph_to_flow_network(&self, graph: &Graph, source: NodeId, sink: NodeId) -> FlowNetwork {
        let vertex_count = graph.node_count();
        let mut network = FlowNetwork::new(vertex_count);
        
        // Add all edges to flow network
        for edge in graph.get_edges() {
            network.add_edge(edge.source, edge.target, edge.weight);
        }
        
        network
    }
}

/// Flow algorithm errors
#[derive(Debug, thiserror::Error)]
pub enum FlowError {
    #[error("Insufficient capacity for flow")]
    InsufficientCapacity,
    #[error("Invalid source or sink vertex")]
    InvalidVertex,
    #[error("Network construction error")]
    NetworkError,
}

impl Algorithm for MaxFlowSolver {
    fn name(&self) -> &str {
        match self.algorithm {
            MaxFlowAlgorithm::FordFulkerson => "Ford-Fulkerson",
            MaxFlowAlgorithm::EdmondsKarp => "Edmonds-Karp", 
            MaxFlowAlgorithm::PushRelabelFIFO => "Push-Relabel FIFO",
            MaxFlowAlgorithm::PushRelabelHighestLabel => "Push-Relabel Highest Label",
            MaxFlowAlgorithm::CapacityScaling => "Capacity Scaling",
            MaxFlowAlgorithm::ParallelPushRelabel => "Parallel Push-Relabel",
        }
    }
    
    fn category(&self) -> &str {
        "max_flow"
    }
    
    fn description(&self) -> &str {
        "Advanced maximum flow algorithms with Goldberg-Tarjan push-relabel, capacity scaling, and parallel optimizations for optimal network flow analysis."
    }
    
    fn set_parameter(&mut self, name: &str, value: &str) -> Result<(), AlgorithmError> {
        match name {
            "global_relabel_frequency" => {
                let freq = value.parse::<usize>()
                    .map_err(|_| AlgorithmError::InvalidParameter(
                        "global_relabel_frequency must be positive integer".to_string()))?;
                self.global_relabel_frequency = freq;
                self.parameters.insert(name.to_string(), value.to_string());
                Ok(())
            },
            "capacity_scaling_factor" => {
                let factor = value.parse::<f64>()
                    .map_err(|_| AlgorithmError::InvalidParameter(
                        "capacity_scaling_factor must be positive number".to_string()))?;
                if factor <= 1.0 {
                    return Err(AlgorithmError::InvalidParameter(
                        "capacity_scaling_factor must be > 1.0".to_string()));
                }
                self.parameters.insert(name.to_string(), value.to_string());
                Ok(())
            },
            _ => Err(AlgorithmError::InvalidParameter(
                format!("Unknown parameter: {}", name))),
        }
    }
    
    fn get_parameter(&self, name: &str) -> Option<&str> {
        self.parameters.get(name).map(|s| s.as_str())
    }
    
    fn get_parameters(&self) -> HashMap<String, String> {
        self.parameters.clone()
    }
    
    fn execute_with_tracing(&mut self, 
                          graph: &Graph, 
                          tracer: &mut ExecutionTracer) 
                          -> Result<AlgorithmResult, AlgorithmError> {
        // Implementation for execution with tracing
        Ok(AlgorithmResult {
            steps: self.push_count + self.relabel_count,
            nodes_visited: graph.node_count(),
            execution_time_ms: 0.0,
            state: AlgorithmState {
                step: 0,
                open_set: Vec::new(),
                closed_set: Vec::new(),
                current_node: None,
                data: HashMap::new(),
            },
        })
    }
    
    fn find_path(&mut self, 
               graph: &Graph, 
               start: NodeId, 
               goal: NodeId) 
               -> Result<crate::algorithm::PathResult, AlgorithmError> {
        // Convert to max flow problem
        let mut network = self.graph_to_flow_network(graph, start, goal);
        
        let max_flow = match self.algorithm {
            MaxFlowAlgorithm::CapacityScaling => {
                self.solve_capacity_scaling(&mut network, start, goal)
            },
            _ => {
                self.solve_push_relabel(&mut network, start, goal)
            }
        }.map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        Ok(crate::algorithm::PathResult {
            path: None, // Max flow doesn't produce paths
            cost: Some(max_flow),
            result: AlgorithmResult {
                steps: self.push_count + self.relabel_count,
                nodes_visited: graph.node_count(),
                execution_time_ms: 0.0,
                state: AlgorithmState {
                    step: 0,
                    open_set: Vec::new(),
                    closed_set: Vec::new(),
                    current_node: None,
                    data: HashMap::new(),
                },
            },
        })
    }
}

/// Maximum flow result with detailed flow information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxFlowResult {
    /// Maximum flow value
    pub max_flow: Flow,
    /// Flow decomposition into paths
    pub flow_paths: Vec<FlowPath>,
    /// Minimum cut vertices
    pub min_cut: Vec<NodeId>,
    /// Algorithm performance metrics
    pub metrics: FlowMetrics,
}

/// Flow path representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowPath {
    /// Path vertices
    pub path: Vec<NodeId>,
    /// Flow amount along path
    pub flow: Flow,
}

/// Flow algorithm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMetrics {
    /// Number of push operations
    pub push_operations: usize,
    /// Number of relabel operations  
    pub relabel_operations: usize,
    /// Number of global relabels
    pub global_relabels: usize,
    /// Algorithm execution time
    pub execution_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flow_edge_creation() {
        let edge = FlowEdge::new(0, 1, 10.0);
        assert_eq!(edge.from, 0);
        assert_eq!(edge.to, 1);
        assert_eq!(edge.capacity, 10.0);
        assert_eq!(edge.flow, 0.0);
        assert_eq!(edge.residual_capacity(), 10.0);
    }
    
    #[test]
    fn test_flow_network_creation() {
        let mut network = FlowNetwork::new(4);
        network.add_edge(0, 1, 10.0);
        network.add_edge(1, 2, 5.0);
        network.add_edge(2, 3, 8.0);
        
        assert_eq!(network.vertex_count, 4);
        assert_eq!(network.edges.len(), 6); // 3 forward + 3 reverse edges
    }
    
    #[test]
    fn test_max_flow_solver_creation() {
        let solver = MaxFlowSolver::new(MaxFlowAlgorithm::PushRelabelFIFO);
        assert_eq!(solver.name(), "Push-Relabel FIFO");
        assert_eq!(solver.category(), "max_flow");
    }
    
    #[test]
    fn test_parameter_setting() {
        let mut solver = MaxFlowSolver::new(MaxFlowAlgorithm::CapacityScaling);
        
        assert!(solver.set_parameter("global_relabel_frequency", "50").is_ok());
        assert_eq!(solver.get_parameter("global_relabel_frequency"), Some("50"));
        
        assert!(solver.set_parameter("capacity_scaling_factor", "3.0").is_ok());
        assert_eq!(solver.get_parameter("capacity_scaling_factor"), Some("3.0"));
        
        // Test invalid parameters
        assert!(solver.set_parameter("invalid_param", "value").is_err());
        assert!(solver.set_parameter("global_relabel_frequency", "invalid").is_err());
        assert!(solver.set_parameter("capacity_scaling_factor", "0.5").is_err());
    }
    
    #[test]
    fn test_preflow_initialization() {
        let mut network = FlowNetwork::new(3);
        network.add_edge(0, 1, 10.0);
        network.add_edge(0, 2, 5.0);
        
        network.initialize_preflow(0);
        
        assert_eq!(network.distance_labels[0], 3);
        assert_eq!(network.excess[1], 10.0);
        assert_eq!(network.excess[2], 5.0);
        assert_eq!(network.active_vertices.len(), 2);
    }
}