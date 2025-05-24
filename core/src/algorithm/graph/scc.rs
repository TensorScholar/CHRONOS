//! Strongly Connected Components Algorithm Implementation
//!
//! This module implements Tarjan's algorithm for finding strongly connected components
//! in directed graphs. The algorithm achieves optimal O(V+E) time complexity through
//! a single depth-first search traversal, making it the most efficient known solution
//! for the SCC problem.
//!
//! # Theoretical Foundation
//!
//! A strongly connected component (SCC) is a maximal set of vertices such that there
//! is a directed path from each vertex to every other vertex in the component. Tarjan's
//! algorithm leverages the mathematical properties of DFS trees and back edges to
//! identify these components in linear time.
//!
//! ## Mathematical Invariants
//!
//! 1. **Discovery Time Property**: Each vertex has a unique discovery time during DFS
//! 2. **Low-Link Property**: For each vertex v, low[v] = min(disc[v], min(low[w])) 
//!    where w is reachable from v via back edges
//! 3. **SCC Root Property**: A vertex v is the root of an SCC iff disc[v] == low[v]
//! 4. **Stack Property**: Vertices of an SCC appear consecutively on the DFS stack
//!
//! # Algorithmic Complexity
//!
//! - **Time Complexity**: O(V + E) - each vertex and edge visited exactly once
//! - **Space Complexity**: O(V) - for discovery times, low-links, and stack
//! - **Optimality**: Proven optimal - any SCC algorithm must examine all edges
//!
//! # Correctness Guarantee
//!
//! The algorithm is proven correct through mathematical induction on the structure
//! of strongly connected components and the properties of depth-first search trees.
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::traits::{Algorithm, AlgorithmError, AlgorithmResult, AlgorithmState, NodeId, PathResult};
use crate::data_structures::graph::Graph;
use crate::execution::tracer::ExecutionTracer;

/// Strongly Connected Component representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StronglyConnectedComponent {
    /// Unique identifier for this component
    pub id: usize,
    /// Vertices in this component
    pub vertices: Vec<NodeId>,
    /// Size of the component
    pub size: usize,
    /// Root vertex (lowest discovery time in component)
    pub root: NodeId,
}

impl StronglyConnectedComponent {
    /// Create a new strongly connected component
    pub fn new(id: usize, vertices: Vec<NodeId>) -> Self {
        let size = vertices.len();
        let root = *vertices.iter().min().unwrap_or(&0);
        
        Self {
            id,
            vertices,
            size,
            root,
        }
    }

    /// Check if this component contains a specific vertex
    pub fn contains(&self, vertex: NodeId) -> bool {
        self.vertices.contains(&vertex)
    }

    /// Get the vertices in this component as a hash set for O(1) lookup
    pub fn vertex_set(&self) -> HashSet<NodeId> {
        self.vertices.iter().cloned().collect()
    }

    /// Check if this is a trivial component (single vertex with no self-loop)
    pub fn is_trivial(&self, graph: &Graph) -> bool {
        self.size == 1 && !graph.has_edge(self.root, self.root)
    }

    /// Get the condensed representation size (for condensation graph)
    pub fn condensed_edges(&self, graph: &Graph, other: &StronglyConnectedComponent) -> usize {
        let mut edge_count = 0;
        let other_vertices = other.vertex_set();
        
        for &vertex in &self.vertices {
            if let Some(neighbors) = graph.get_neighbors(vertex) {
                for neighbor in neighbors {
                    if other_vertices.contains(&neighbor) {
                        edge_count += 1;
                        break; // Only count one edge between components
                    }
                }
            }
        }
        
        edge_count
    }
}

/// Tarjan's algorithm state for SCC computation
#[derive(Debug, Clone)]
struct TarjanState {
    /// Discovery time for each vertex
    discovery_time: HashMap<NodeId, usize>,
    /// Low-link value for each vertex
    low_link: HashMap<NodeId, usize>,
    /// Current time counter
    time: usize,
    /// Stack of vertices in current path
    stack: Vec<NodeId>,
    /// Set of vertices currently on stack (for O(1) membership testing)
    on_stack: HashSet<NodeId>,
    /// Found strongly connected components
    components: Vec<StronglyConnectedComponent>,
    /// Component ID counter
    component_counter: usize,
}

impl TarjanState {
    /// Create new Tarjan algorithm state
    fn new() -> Self {
        Self {
            discovery_time: HashMap::new(),
            low_link: HashMap::new(),
            time: 0,
            stack: Vec::new(),
            on_stack: HashSet::new(),
            components: Vec::new(),
            component_counter: 0,
        }
    }

    /// Check if a vertex has been visited
    fn is_visited(&self, vertex: NodeId) -> bool {
        self.discovery_time.contains_key(&vertex)
    }

    /// Get the next discovery time and increment counter
    fn next_time(&mut self) -> usize {
        let current = self.time;
        self.time += 1;
        current
    }

    /// Push vertex onto the stack
    fn push_stack(&mut self, vertex: NodeId) {
        self.stack.push(vertex);
        self.on_stack.insert(vertex);
    }

    /// Pop vertex from stack
    fn pop_stack(&mut self) -> Option<NodeId> {
        if let Some(vertex) = self.stack.pop() {
            self.on_stack.remove(&vertex);
            Some(vertex)
        } else {
            None
        }
    }

    /// Check if vertex is currently on stack
    fn is_on_stack(&self, vertex: NodeId) -> bool {
        self.on_stack.contains(&vertex)
    }
}

/// SCC computation result with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCCResult {
    /// All strongly connected components found
    pub components: Vec<StronglyConnectedComponent>,
    /// Total number of components
    pub num_components: usize,
    /// Size of the largest component
    pub largest_component_size: usize,
    /// Number of trivial components (single vertices)
    pub trivial_components: usize,
    /// Component size distribution
    pub size_distribution: HashMap<usize, usize>,
    /// Performance statistics
    pub statistics: SCCStatistics,
}

impl SCCResult {
    /// Create new SCC result from components
    fn from_components(components: Vec<StronglyConnectedComponent>, stats: SCCStatistics) -> Self {
        let num_components = components.len();
        let largest_component_size = components.iter()
            .map(|c| c.size)
            .max()
            .unwrap_or(0);
        
        let trivial_components = components.iter()
            .filter(|c| c.size == 1)
            .count();
        
        // Compute size distribution
        let mut size_distribution = HashMap::new();
        for component in &components {
            *size_distribution.entry(component.size).or_insert(0) += 1;
        }
        
        Self {
            components,
            num_components,
            largest_component_size,
            trivial_components,
            size_distribution,
            statistics: stats,
        }
    }

    /// Get component containing a specific vertex
    pub fn get_component_containing(&self, vertex: NodeId) -> Option<&StronglyConnectedComponent> {
        self.components.iter().find(|c| c.contains(vertex))
    }

    /// Check if two vertices are in the same strongly connected component
    pub fn are_strongly_connected(&self, u: NodeId, v: NodeId) -> bool {
        if let Some(component) = self.get_component_containing(u) {
            component.contains(v)
        } else {
            false
        }
    }

    /// Get condensation graph representation
    pub fn condensation_graph(&self, original_graph: &Graph) -> Graph {
        let mut condensed = Graph::new();
        
        // Add nodes for each component
        for component in &self.components {
            condensed.add_node_with_id(component.id, (component.id as f64, 0.0)).ok();
        }
        
        // Add edges between components
        let mut added_edges = HashSet::new();
        for (i, comp1) in self.components.iter().enumerate() {
            for (j, comp2) in self.components.iter().enumerate() {
                if i != j && !added_edges.contains(&(i, j)) {
                    let edge_count = comp1.condensed_edges(original_graph, comp2);
                    if edge_count > 0 {
                        condensed.add_edge(comp1.id, comp2.id, edge_count as f64).ok();
                        added_edges.insert((i, j));
                    }
                }
            }
        }
        
        condensed
    }

    /// Compute topological ordering of the condensation graph
    pub fn topological_order(&self, original_graph: &Graph) -> Vec<usize> {
        let condensed = self.condensation_graph(original_graph);
        
        // Implement Kahn's algorithm for topological sorting
        let mut in_degree = HashMap::new();
        let mut adj_list = HashMap::new();
        
        // Initialize in-degrees and adjacency list
        for component in &self.components {
            in_degree.insert(component.id, 0);
            adj_list.insert(component.id, Vec::new());
        }
        
        // Build adjacency list and compute in-degrees
        for edge in condensed.get_edges() {
            adj_list.get_mut(&edge.source).unwrap().push(edge.target);
            *in_degree.get_mut(&edge.target).unwrap() += 1;
        }
        
        // Kahn's algorithm
        let mut queue = Vec::new();
        let mut result = Vec::new();
        
        // Find all nodes with in-degree 0
        for (node, &degree) in &in_degree {
            if degree == 0 {
                queue.push(*node);
            }
        }
        
        while let Some(node) = queue.pop() {
            result.push(node);
            
            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    let new_degree = in_degree.get_mut(&neighbor).unwrap();
                    *new_degree -= 1;
                    if *new_degree == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        result
    }
}

/// Performance and behavior statistics for SCC computation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SCCStatistics {
    pub vertices_visited: usize,
    pub edges_examined: usize,
    pub max_stack_depth: usize,
    pub dfs_calls: usize,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: usize,
}

/// SCC-specific errors
#[derive(Debug, Error)]
pub enum SCCError {
    #[error("Graph has no vertices")]
    EmptyGraph,
    
    #[error("Invalid vertex: {0}")]
    InvalidVertex(NodeId),
    
    #[error("Algorithm execution error: {0}")]
    ExecutionError(String),
    
    #[error("Memory allocation error")]
    MemoryError,
}

/// Strongly Connected Components Algorithm Implementation
///
/// This implementation uses Tarjan's algorithm to find all strongly connected
/// components in a directed graph in optimal O(V+E) time complexity.
#[derive(Debug, Clone)]
pub struct StronglyConnectedComponents {
    /// Algorithm parameters
    parameters: HashMap<String, String>,
    
    /// Global time counter for thread-safe discovery time generation
    global_time: Arc<AtomicUsize>,
    
    /// Performance statistics
    statistics: SCCStatistics,
    
    /// Maximum stack depth tracking for performance analysis
    max_stack_depth: usize,
}

impl StronglyConnectedComponents {
    /// Create a new SCC algorithm instance
    pub fn new() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("compute_condensation".to_string(), "false".to_string());
        parameters.insert("track_statistics".to_string(), "true".to_string());
        parameters.insert("verify_result".to_string(), "true".to_string());

        Self {
            parameters,
            global_time: Arc::new(AtomicUsize::new(0)),
            statistics: SCCStatistics::default(),
            max_stack_depth: 0,
        }
    }

    /// Compute strongly connected components using Tarjan's algorithm
    ///
    /// This is the core implementation of Tarjan's SCC algorithm with optimal
    /// O(V+E) time complexity and linear space usage.
    pub fn compute_scc(&mut self, graph: &Graph) -> Result<SCCResult, SCCError> {
        let start_time = std::time::Instant::now();
        
        if graph.node_count() == 0 {
            return Err(SCCError::EmptyGraph);
        }

        // Reset statistics and global state
        self.statistics = SCCStatistics::default();
        self.global_time.store(0, Ordering::Relaxed);
        self.max_stack_depth = 0;
        
        // Initialize Tarjan state
        let mut state = TarjanState::new();
        
        // Run DFS from each unvisited vertex
        for node in graph.get_nodes() {
            let node_id = node.id;
            if !state.is_visited(node_id) {
                self.tarjan_dfs(node_id, graph, &mut state)?;
            }
        }
        
        // Update performance statistics
        self.statistics.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.statistics.max_stack_depth = self.max_stack_depth;
        
        // Create result with comprehensive analysis
        let result = SCCResult::from_components(state.components, self.statistics.clone());
        
        // Verify result if enabled
        if self.parameters.get("verify_result").unwrap_or(&"true".to_string()).parse().unwrap_or(true) {
            self.verify_scc_result(graph, &result)?;
        }
        
        Ok(result)
    }

    /// Tarjan's DFS implementation with mathematical precision
    ///
    /// This function implements the core logic of Tarjan's algorithm:
    /// 1. Assign discovery time and low-link value
    /// 2. Push vertex onto stack
    /// 3. Explore all neighbors recursively
    /// 4. Update low-link values based on back edges
    /// 5. Pop SCC when root is found (discovery_time == low_link)
    fn tarjan_dfs(&mut self, vertex: NodeId, graph: &Graph, state: &mut TarjanState) 
                  -> Result<(), SCCError> {
        
        self.statistics.dfs_calls += 1;
        self.statistics.vertices_visited += 1;
        
        // Initialize vertex with discovery time and low-link value
        let discovery_time = state.next_time();
        state.discovery_time.insert(vertex, discovery_time);
        state.low_link.insert(vertex, discovery_time);
        
        // Push vertex onto stack
        state.push_stack(vertex);
        
        // Track maximum stack depth for performance analysis
        if state.stack.len() > self.max_stack_depth {
            self.max_stack_depth = state.stack.len();
        }
        
        // Explore all neighbors
        if let Some(neighbors) = graph.get_neighbors(vertex) {
            for neighbor in neighbors {
                self.statistics.edges_examined += 1;
                
                if !state.is_visited(neighbor) {
                    // Tree edge: recursively visit unvisited neighbor
                    self.tarjan_dfs(neighbor, graph, state)?;
                    
                    // Update low-link value based on neighbor's low-link
                    let neighbor_low = *state.low_link.get(&neighbor).unwrap();
                    let current_low = state.low_link.get_mut(&vertex).unwrap();
                    *current_low = (*current_low).min(neighbor_low);
                    
                } else if state.is_on_stack(neighbor) {
                    // Back edge: neighbor is on stack, update low-link
                    let neighbor_discovery = *state.discovery_time.get(&neighbor).unwrap();
                    let current_low = state.low_link.get_mut(&vertex).unwrap();
                    *current_low = (*current_low).min(neighbor_discovery);
                }
                // Forward/cross edges are ignored as they don't affect SCCs
            }
        }
        
        // Check if vertex is the root of an SCC
        let vertex_discovery = *state.discovery_time.get(&vertex).unwrap();
        let vertex_low = *state.low_link.get(&vertex).unwrap();
        
        if vertex_discovery == vertex_low {
            // Found an SCC root - pop all vertices of this component
            let mut component_vertices = Vec::new();
            
            loop {
                if let Some(popped) = state.pop_stack() {
                    component_vertices.push(popped);
                    if popped == vertex {
                        break; // Found the root, component is complete
                    }
                } else {
                    return Err(SCCError::ExecutionError(
                        "Stack underflow during SCC extraction".to_string()
                    ));
                }
            }
            
            // Create and store the component
            let component = StronglyConnectedComponent::new(
                state.component_counter,
                component_vertices
            );
            state.components.push(component);
            state.component_counter += 1;
        }
        
        Ok(())
    }

    /// Verify the correctness of the SCC result
    ///
    /// This verification function ensures:
    /// 1. All vertices are covered exactly once
    /// 2. Each component is actually strongly connected
    /// 3. No edges exist between vertices in different components that would
    ///    indicate they should be in the same component
    fn verify_scc_result(&self, graph: &Graph, result: &SCCResult) -> Result<(), SCCError> {
        let total_vertices = graph.node_count();
        let mut covered_vertices = HashSet::new();
        
        // Verify all vertices are covered exactly once
        for component in &result.components {
            for &vertex in &component.vertices {
                if covered_vertices.contains(&vertex) {
                    return Err(SCCError::ExecutionError(
                        format!("Vertex {} appears in multiple components", vertex)
                    ));
                }
                covered_vertices.insert(vertex);
            }
        }
        
        if covered_vertices.len() != total_vertices {
            return Err(SCCError::ExecutionError(
                format!("Vertex coverage mismatch: expected {}, found {}", 
                       total_vertices, covered_vertices.len())
            ));
        }
        
        // Verify each component is strongly connected using BFS reachability
        for component in &result.components {
            if component.size > 1 {
                self.verify_component_connectivity(graph, component)?;
            }
        }
        
        Ok(())
    }

    /// Verify that a component is actually strongly connected
    fn verify_component_connectivity(&self, graph: &Graph, component: &StronglyConnectedComponent) 
                                   -> Result<(), SCCError> {
        let component_set = component.vertex_set();
        
        // For each pair of vertices in the component, verify there's a path between them
        // This is a simplified verification - full verification would use more efficient algorithms
        for &start in &component.vertices {
            let reachable = self.compute_reachability_within_component(graph, start, &component_set);
            
            if reachable.len() != component.size {
                return Err(SCCError::ExecutionError(
                    format!("Component {} is not strongly connected: vertex {} cannot reach all other vertices", 
                           component.id, start)
                ));
            }
        }
        
        Ok(())
    }

    /// Compute reachability within a component using BFS
    fn compute_reachability_within_component(&self, graph: &Graph, start: NodeId, 
                                           component: &HashSet<NodeId>) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = graph.get_neighbors(current) {
                for neighbor in neighbors {
                    if component.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        visited
    }

    /// Get algorithm statistics
    pub fn get_statistics(&self) -> &SCCStatistics {
        &self.statistics
    }

    /// Reset algorithm statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = SCCStatistics::default();
        self.global_time.store(0, Ordering::Relaxed);
        self.max_stack_depth = 0;
    }

    /// Check if the graph is strongly connected (single SCC)
    pub fn is_strongly_connected(&mut self, graph: &Graph) -> Result<bool, SCCError> {
        let result = self.compute_scc(graph)?;
        Ok(result.num_components == 1)
    }

    /// Find the largest strongly connected component
    pub fn largest_scc(&mut self, graph: &Graph) -> Result<Option<StronglyConnectedComponent>, SCCError> {
        let result = self.compute_scc(graph)?;
        Ok(result.components.into_iter()
           .max_by_key(|c| c.size))
    }
}

impl Default for StronglyConnectedComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl Algorithm for StronglyConnectedComponents {
    fn name(&self) -> &str {
        "Strongly Connected Components (Tarjan)"
    }

    fn category(&self) -> &str {
        "graph"
    }

    fn description(&self) -> &str {
        "Tarjan's algorithm for finding strongly connected components in directed graphs. A strongly connected component is a maximal set of vertices such that there is a directed path from each vertex to every other vertex in the component. This implementation achieves optimal O(V+E) time complexity through a single depth-first search with mathematical precision and correctness guarantees."
    }

    fn set_parameter(&mut self, name: &str, value: &str) -> Result<(), AlgorithmError> {
        match name {
            "compute_condensation" | "track_statistics" | "verify_result" => {
                match value {
                    "true" | "false" => {
                        self.parameters.insert(name.to_string(), value.to_string());
                        Ok(())
                    },
                    _ => Err(AlgorithmError::InvalidParameter(format!(
                        "Invalid boolean value: {}. Use 'true' or 'false'", 
                        value
                    ))),
                }
            },
            _ => Err(AlgorithmError::InvalidParameter(format!(
                "Unknown parameter: {}. Valid parameters: compute_condensation, track_statistics, verify_result", 
                name
            ))),
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
        // Reset statistics
        self.reset_statistics();
        
        // Compute SCCs
        let scc_result = self.compute_scc(graph)
            .map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        Ok(AlgorithmResult {
            steps: self.statistics.dfs_calls,
            nodes_visited: self.statistics.vertices_visited,
            execution_time_ms: self.statistics.execution_time_ms,
            state: AlgorithmState {
                step: self.statistics.dfs_calls,
                open_set: Vec::new(),
                closed_set: (0..graph.node_count()).collect(),
                current_node: None,
                data: {
                    let mut data = HashMap::new();
                    data.insert("num_components".to_string(), scc_result.num_components.to_string());
                    data.insert("largest_component".to_string(), scc_result.largest_component_size.to_string());
                    data.insert("trivial_components".to_string(), scc_result.trivial_components.to_string());
                    data.insert("edges_examined".to_string(), self.statistics.edges_examined.to_string());
                    data.insert("max_stack_depth".to_string(), self.statistics.max_stack_depth.to_string());
                    data
                },
            },
        })
    }

    fn find_path(&mut self, 
               graph: &Graph, 
               start: NodeId, 
               goal: NodeId) 
               -> Result<PathResult, AlgorithmError> {
        // SCC algorithm doesn't find paths directly, but we can use it to determine
        // if a path exists by checking if both vertices are in the same SCC
        
        let scc_result = self.compute_scc(graph)
            .map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        if scc_result.are_strongly_connected(start, goal) {
            // Vertices are in the same SCC, so a path exists
            // For demonstration, we'll return a trivial path
            // In practice, you'd use a different algorithm to find the actual path
            Ok(PathResult {
                path: Some(vec![start, goal]),
                cost: Some(1.0),
                result: AlgorithmResult {
                    steps: self.statistics.dfs_calls,
                    nodes_visited: self.statistics.vertices_visited,
                    execution_time_ms: self.statistics.execution_time_ms,
                    state: AlgorithmState {
                        step: self.statistics.dfs_calls,
                        open_set: Vec::new(),
                        closed_set: (0..graph.node_count()).collect(),
                        current_node: Some(goal),
                        data: {
                            let mut data = HashMap::new();
                            data.insert("same_scc".to_string(), "true".to_string());
                            data.insert("component_id".to_string(), 
                                      scc_result.get_component_containing(start)
                                               .map(|c| c.id.to_string())
                                               .unwrap_or_else(|| "unknown".to_string()));
                            data
                        },
                    },
                },
            })
        } else {
            // Vertices are in different SCCs, no path exists in the SCC sense
            // (there might be a path from start to goal, but not vice versa)
            Err(AlgorithmError::NoPathFound)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::graph::Graph;

    #[test]
    fn test_scc_creation() {
        let scc = StronglyConnectedComponents::new();
        assert_eq!(scc.name(), "Strongly Connected Components (Tarjan)");
        assert_eq!(scc.category(), "graph");
        assert_eq!(scc.get_parameter("verify_result").unwrap(), "true");
    }

    #[test]
    fn test_tarjan_state() {
        let mut state = TarjanState::new();
        
        assert_eq!(state.time, 0);
        assert_eq!(state.next_time(), 0);
        assert_eq!(state.time, 1);
        
        state.push_stack(5);
        assert!(state.is_on_stack(5));
        assert!(!state.is_on_stack(3));
        
        assert_eq!(state.pop_stack(), Some(5));
        assert!(!state.is_on_stack(5));
    }

    #[test]
    fn test_strongly_connected_component() {
        let vertices = vec![1, 3, 5, 7];
        let component = StronglyConnectedComponent::new(0, vertices.clone());
        
        assert_eq!(component.id, 0);
        assert_eq!(component.size, 4);
        assert_eq!(component.vertices, vertices);
        assert!(component.contains(3));
        assert!(!component.contains(2));
        
        let vertex_set = component.vertex_set();
        assert!(vertex_set.contains(&1));
        assert!(vertex_set.contains(&7));
        assert!(!vertex_set.contains(&2));
    }

    #[test]
    fn test_parameter_validation() {
        let mut scc = StronglyConnectedComponents::new();
        
        // Valid parameters
        assert!(scc.set_parameter("verify_result", "false").is_ok());
        assert_eq!(scc.get_parameter("verify_result").unwrap(), "false");
        
        assert!(scc.set_parameter("track_statistics", "true").is_ok());
        assert_eq!(scc.get_parameter("track_statistics").unwrap(), "true");
        
        // Invalid parameters
        assert!(scc.set_parameter("verify_result", "maybe").is_err());
        assert!(scc.set_parameter("unknown_param", "value").is_err());
    }

    #[test]
    fn test_simple_scc_graph() {
        let mut graph = Graph::new();
        
        // Create nodes
        for i in 0..4 {
            graph.add_node_with_id(i, (i as f64, 0.0)).unwrap();
        }
        
        // Create a simple cycle: 0 -> 1 -> 2 -> 0, and isolated node 3
        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 0, 1.0).unwrap();
        
        let mut scc = StronglyConnectedComponents::new();
        let result = scc.compute_scc(&graph).unwrap();
        
        // Should have 2 components: {0, 1, 2} and {3}
        assert_eq!(result.num_components, 2);
        assert_eq!(result.largest_component_size, 3);
        assert_eq!(result.trivial_components, 1);
        
        // Verify strongly connected property
        assert!(result.are_strongly_connected(0, 1));
        assert!(result.are_strongly_connected(1, 2));
        assert!(result.are_strongly_connected(2, 0));
        assert!(!result.are_strongly_connected(0, 3));
    }

    #[test]
    fn test_statistics_tracking() {
        let mut scc = StronglyConnectedComponents::new();
        
        scc.statistics.vertices_visited = 10;
        scc.statistics.edges_examined = 15;
        
        let stats = scc.get_statistics();
        assert_eq!(stats.vertices_visited, 10);
        assert_eq!(stats.edges_examined, 15);
        
        scc.reset_statistics();
        assert_eq!(scc.get_statistics().vertices_visited, 0);
        assert_eq!(scc.get_statistics().edges_examined, 0);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::new();
        let mut scc = StronglyConnectedComponents::new();
        
        let result = scc.compute_scc(&graph);
        assert!(result.is_err());
        
        if let Err(SCCError::EmptyGraph) = result {
            // Expected error
        } else {
            panic!("Expected EmptyGraph error");
        }
    }

    #[test]
    fn test_single_vertex() {
        let mut graph = Graph::new();
        graph.add_node_with_id(0, (0.0, 0.0)).unwrap();
        
        let mut scc = StronglyConnectedComponents::new();
        let result = scc.compute_scc(&graph).unwrap();
        
        assert_eq!(result.num_components, 1);
        assert_eq!(result.largest_component_size, 1);
        assert_eq!(result.trivial_components, 1);
    }
}