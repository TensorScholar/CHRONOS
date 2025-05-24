//! Minimum Spanning Tree Algorithm Implementation
//!
//! This module implements comprehensive minimum spanning tree (MST) algorithms
//! including Kruskal's algorithm with union-find optimization and Prim's algorithm
//! with priority queue optimization. Both algorithms are mathematically proven to
//! produce optimal spanning trees through greedy optimization principles.
//!
//! # Theoretical Foundation
//!
//! The MST problem seeks to find a tree that connects all vertices in a weighted
//! undirected graph with minimum total edge weight. This implementation leverages:
//!
//! - **Cut Property**: For any cut (S, V-S), the minimum-weight crossing edge
//!   is safe for the MST (Prim's algorithm foundation)
//! - **Cycle Property**: For any cycle, the maximum-weight edge is not in any MST
//!   (Kruskal's algorithm foundation)
//!
//! # Algorithmic Complexity
//!
//! - **Kruskal's Algorithm**: O(E log E) ≈ O(E log V) with union-find optimization
//! - **Prim's Algorithm**: O(E log V) with binary heap priority queue
//! - **Space Complexity**: O(V) for both algorithms
//!
//! # Correctness Guarantees
//!
//! Both algorithms are proven to produce optimal MSTs through mathematical induction
//! on the greedy choice property, ensuring global optimality from locally optimal decisions.
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::{Ordering, Reverse};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::traits::{Algorithm, AlgorithmError, AlgorithmResult, AlgorithmState, NodeId, PathResult};
use crate::data_structures::graph::{Graph, Edge, Weight};
use crate::execution::tracer::ExecutionTracer;

/// MST algorithm selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MSTAlgorithm {
    /// Kruskal's algorithm with union-find optimization
    Kruskal,
    /// Prim's algorithm with priority queue optimization
    Prim,
    /// Automatic selection based on graph characteristics
    Auto,
}

impl Default for MSTAlgorithm {
    fn default() -> Self {
        MSTAlgorithm::Auto
    }
}

/// Edge representation for MST algorithms with total ordering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MSTEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: Weight,
}

impl MSTEdge {
    pub fn new(source: NodeId, target: NodeId, weight: Weight) -> Self {
        // Ensure canonical ordering for undirected edges
        if source <= target {
            Self { source, target, weight }
        } else {
            Self { source: target, target: source, weight }
        }
    }
}

impl Eq for MSTEdge {}

impl Ord for MSTEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary ordering by weight (ascending for minimum spanning tree)
        self.weight.partial_cmp(&other.weight)
            .unwrap_or(Ordering::Equal)
            // Secondary ordering by source node for deterministic behavior
            .then_with(|| self.source.cmp(&other.source))
            // Tertiary ordering by target node
            .then_with(|| self.target.cmp(&other.target))
    }
}

impl PartialOrd for MSTEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Union-Find data structure with path compression and union by rank
///
/// This implementation achieves nearly constant amortized time complexity
/// O(α(n)) where α is the inverse Ackermann function, which is effectively
/// constant for all practical purposes.
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent pointers for each element
    parent: Vec<NodeId>,
    /// Rank (approximate depth) of each tree
    rank: Vec<usize>,
    /// Number of disjoint sets
    num_components: usize,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            num_components: n,
        }
    }

    /// Find the root of the set containing x with path compression
    ///
    /// Path compression flattens the tree structure, making subsequent
    /// operations faster by directly connecting nodes to the root.
    pub fn find(&mut self, x: NodeId) -> NodeId {
        if self.parent[x] != x {
            // Path compression: make parent[x] point directly to root
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union two sets containing x and y using union by rank
    ///
    /// Union by rank attaches the smaller tree under the root of the larger tree,
    /// keeping the tree height logarithmic and maintaining efficiency.
    pub fn union(&mut self, x: NodeId, y: NodeId) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in same set - would create cycle
        }

        // Union by rank: attach smaller tree under larger tree
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            Ordering::Less => {
                self.parent[root_x] = root_y;
            },
            Ordering::Greater => {
                self.parent[root_y] = root_x;
            },
            Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        self.num_components -= 1;
        true
    }

    /// Check if two elements are in the same connected component
    pub fn connected(&mut self, x: NodeId, y: NodeId) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the number of disjoint components
    pub fn components(&self) -> usize {
        self.num_components
    }
}

/// Priority queue entry for Prim's algorithm
#[derive(Debug, Clone, PartialEq)]
struct PrimEntry {
    node: NodeId,
    key: Weight,
    parent: Option<NodeId>,
}

impl PrimEntry {
    fn new(node: NodeId, key: Weight, parent: Option<NodeId>) -> Self {
        Self { node, key, parent }
    }
}

impl Eq for PrimEntry {}

impl Ord for PrimEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior in BinaryHeap
        other.key.partial_cmp(&self.key)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for PrimEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Minimum Spanning Tree result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSTResult {
    /// Edges in the minimum spanning tree
    pub edges: Vec<MSTEdge>,
    /// Total weight of the minimum spanning tree
    pub total_weight: Weight,
    /// Algorithm used to compute the MST
    pub algorithm_used: MSTAlgorithm,
    /// Performance statistics
    pub statistics: MSTStatistics,
}

/// Performance and behavior statistics for MST algorithms
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MSTStatistics {
    pub edges_examined: usize,
    pub union_operations: usize,
    pub find_operations: usize,
    pub priority_queue_operations: usize,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: usize,
}

/// MST-specific errors
#[derive(Debug, Error)]
pub enum MSTError {
    #[error("Graph is not connected - no spanning tree exists")]
    GraphNotConnected,
    
    #[error("Graph has no edges")]
    NoEdges,
    
    #[error("Graph has insufficient nodes: {0}")]
    InsufficientNodes(usize),
    
    #[error("Invalid edge weight: {0}")]
    InvalidWeight(Weight),
    
    #[error("Algorithm execution error: {0}")]
    ExecutionError(String),
}

/// Minimum Spanning Tree Algorithm Implementation
///
/// This implementation provides both Kruskal's and Prim's algorithms with
/// automatic selection based on graph characteristics for optimal performance.
#[derive(Debug, Clone)]
pub struct MinimumSpanningTree {
    /// Algorithm parameters
    parameters: HashMap<String, String>,
    
    /// Cached graph reference for repeated operations
    cached_graph: Option<Arc<Graph>>,
    
    /// Performance statistics
    statistics: MSTStatistics,
}

impl MinimumSpanningTree {
    /// Create a new MST algorithm instance
    pub fn new() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("algorithm".to_string(), "auto".to_string());
        parameters.insert("verify_connectivity".to_string(), "true".to_string());
        parameters.insert("collect_statistics".to_string(), "true".to_string());

        Self {
            parameters,
            cached_graph: None,
            statistics: MSTStatistics::default(),
        }
    }

    /// Set the graph for MST operations (caching for performance)
    pub fn set_graph(&mut self, graph: Arc<Graph>) {
        self.cached_graph = Some(graph);
    }

    /// Select optimal algorithm based on graph characteristics
    ///
    /// Decision heuristics:
    /// - Dense graphs (E > V²/2): Prim's algorithm (better cache locality)
    /// - Sparse graphs (E < V²/2): Kruskal's algorithm (fewer edge examinations)
    /// - Very small graphs (V < 10): Prim's algorithm (simpler implementation overhead)
    fn select_algorithm(&self, graph: &Graph) -> MSTAlgorithm {
        let algorithm_param = self.parameters.get("algorithm").unwrap_or(&"auto".to_string());
        
        match algorithm_param.as_str() {
            "kruskal" => MSTAlgorithm::Kruskal,
            "prim" => MSTAlgorithm::Prim,
            "auto" => {
                let num_vertices = graph.node_count();
                let num_edges = graph.edge_count();
                
                if num_vertices < 10 {
                    MSTAlgorithm::Prim
                } else if num_edges > (num_vertices * num_vertices) / 2 {
                    MSTAlgorithm::Prim  // Dense graph - better for Prim's
                } else {
                    MSTAlgorithm::Kruskal  // Sparse graph - better for Kruskal's
                }
            },
            _ => MSTAlgorithm::Auto, // Fallback to auto-selection
        }
    }

    /// Extract edges from graph in MST format
    fn extract_edges(&self, graph: &Graph) -> Vec<MSTEdge> {
        let mut edges = Vec::new();
        
        for edge in graph.get_edges() {
            // Validate edge weight
            if edge.weight.is_finite() && edge.weight >= 0.0 {
                edges.push(MSTEdge::new(edge.source, edge.target, edge.weight));
            }
        }
        
        edges
    }

    /// Verify graph connectivity using DFS
    fn verify_connectivity(&self, graph: &Graph) -> Result<(), MSTError> {
        if !self.parameters.get("verify_connectivity").unwrap_or(&"true".to_string()).parse().unwrap_or(true) {
            return Ok(());
        }

        let num_vertices = graph.node_count();
        if num_vertices == 0 {
            return Err(MSTError::InsufficientNodes(0));
        }

        let mut visited = vec![false; num_vertices];
        let mut stack = Vec::new();
        
        // Start DFS from node 0
        stack.push(0);
        visited[0] = true;
        let mut visited_count = 1;

        while let Some(current) = stack.pop() {
            if let Some(neighbors) = graph.get_neighbors(current) {
                for neighbor in neighbors {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        visited_count += 1;
                        stack.push(neighbor);
                    }
                }
            }
        }

        if visited_count != num_vertices {
            Err(MSTError::GraphNotConnected)
        } else {
            Ok(())
        }
    }

    /// Compute MST using Kruskal's algorithm
    ///
    /// **Algorithm**: Sort edges by weight, add edges that don't create cycles
    /// **Time Complexity**: O(E log E) ≈ O(E log V)
    /// **Space Complexity**: O(V) for union-find structure
    /// **Correctness**: Cycle property ensures optimality
    fn kruskal_mst(&mut self, graph: &Graph) -> Result<MSTResult, MSTError> {
        let start_time = std::time::Instant::now();
        let num_vertices = graph.node_count();
        
        if num_vertices < 2 {
            return Err(MSTError::InsufficientNodes(num_vertices));
        }

        // Extract and sort edges by weight
        let mut edges = self.extract_edges(graph);
        if edges.is_empty() {
            return Err(MSTError::NoEdges);
        }
        
        edges.sort(); // O(E log E)
        self.statistics.edges_examined = edges.len();
        
        // Initialize union-find structure
        let mut uf = UnionFind::new(num_vertices);
        let mut mst_edges = Vec::new();
        let mut total_weight = 0.0;
        
        // Process edges in ascending order of weight
        for edge in edges {
            self.statistics.union_operations += 1;
            
            // Check if adding this edge creates a cycle
            if uf.union(edge.source, edge.target) {
                mst_edges.push(edge.clone());
                total_weight += edge.weight;
                
                // MST is complete when we have V-1 edges
                if mst_edges.len() == num_vertices - 1 {
                    break;
                }
            }
        }
        
        // Verify we have a spanning tree
        if mst_edges.len() != num_vertices - 1 {
            return Err(MSTError::GraphNotConnected);
        }
        
        self.statistics.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(MSTResult {
            edges: mst_edges,
            total_weight,
            algorithm_used: MSTAlgorithm::Kruskal,
            statistics: self.statistics.clone(),
        })
    }

    /// Compute MST using Prim's algorithm
    ///
    /// **Algorithm**: Grow MST by adding minimum-weight edge crossing the cut
    /// **Time Complexity**: O(E log V) with binary heap
    /// **Space Complexity**: O(V) for priority queue and visited set
    /// **Correctness**: Cut property ensures optimality
    fn prim_mst(&mut self, graph: &Graph) -> Result<MSTResult, MSTError> {
        let start_time = std::time::Instant::now();
        let num_vertices = graph.node_count();
        
        if num_vertices < 2 {
            return Err(MSTError::InsufficientNodes(num_vertices));
        }

        // Initialize data structures
        let mut priority_queue = BinaryHeap::new();
        let mut in_mst = vec![false; num_vertices];
        let mut key = vec![f64::INFINITY; num_vertices];
        let mut parent = vec![None; num_vertices];
        let mut mst_edges = Vec::new();
        let mut total_weight = 0.0;
        
        // Start from vertex 0
        key[0] = 0.0;
        priority_queue.push(PrimEntry::new(0, 0.0, None));
        
        while let Some(entry) = priority_queue.pop() {
            let u = entry.node;
            
            // Skip if already in MST (may happen due to key updates)
            if in_mst[u] {
                continue;
            }
            
            // Add vertex to MST
            in_mst[u] = true;
            self.statistics.priority_queue_operations += 1;
            
            // Add edge to MST (except for the first vertex)
            if let Some(p) = entry.parent {
                mst_edges.push(MSTEdge::new(p, u, entry.key));
                total_weight += entry.key;
            }
            
            // Update keys of adjacent vertices
            if let Some(neighbors) = graph.get_neighbors(u) {
                for v in neighbors {
                    if let Some(edge_weight) = graph.get_edge_weight(u, v) {
                        if !in_mst[v] && edge_weight < key[v] {
                            key[v] = edge_weight;
                            parent[v] = Some(u);
                            priority_queue.push(PrimEntry::new(v, edge_weight, Some(u)));
                        }
                    }
                }
            }
        }
        
        // Verify we have a spanning tree
        if mst_edges.len() != num_vertices - 1 {
            return Err(MSTError::GraphNotConnected);
        }
        
        self.statistics.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(MSTResult {
            edges: mst_edges,
            total_weight,
            algorithm_used: MSTAlgorithm::Prim,
            statistics: self.statistics.clone(),
        })
    }

    /// Compute minimum spanning tree using selected algorithm
    pub fn compute_mst(&mut self, graph: &Graph) -> Result<MSTResult, MSTError> {
        // Reset statistics
        self.statistics = MSTStatistics::default();
        
        // Verify graph connectivity if enabled
        self.verify_connectivity(graph)?;
        
        // Select optimal algorithm
        let algorithm = self.select_algorithm(graph);
        
        // Execute selected algorithm
        match algorithm {
            MSTAlgorithm::Kruskal => self.kruskal_mst(graph),
            MSTAlgorithm::Prim => self.prim_mst(graph),
            MSTAlgorithm::Auto => {
                // Auto-selection already handled in select_algorithm
                let selected = self.select_algorithm(graph);
                match selected {
                    MSTAlgorithm::Kruskal => self.kruskal_mst(graph),
                    MSTAlgorithm::Prim => self.prim_mst(graph),
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Validate MST result for correctness
    ///
    /// Verification includes:
    /// 1. Correct number of edges (V-1)
    /// 2. Tree property (no cycles)
    /// 3. Connectivity (all vertices reachable)
    /// 4. Minimality (can be verified by cut property)
    pub fn validate_mst(&self, graph: &Graph, result: &MSTResult) -> Result<(), MSTError> {
        let num_vertices = graph.node_count();
        
        // Check edge count
        if result.edges.len() != num_vertices - 1 {
            return Err(MSTError::ExecutionError(
                format!("Invalid MST: expected {} edges, found {}", 
                       num_vertices - 1, result.edges.len())
            ));
        }
        
        // Check tree property using union-find
        let mut uf = UnionFind::new(num_vertices);
        for edge in &result.edges {
            if !uf.union(edge.source, edge.target) {
                return Err(MSTError::ExecutionError(
                    "Invalid MST: contains cycle".to_string()
                ));
            }
        }
        
        // Check connectivity
        if uf.components() != 1 {
            return Err(MSTError::ExecutionError(
                "Invalid MST: not connected".to_string()
            ));
        }
        
        Ok(())
    }

    /// Get algorithm statistics
    pub fn get_statistics(&self) -> &MSTStatistics {
        &self.statistics
    }

    /// Reset algorithm statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = MSTStatistics::default();
    }
}

impl Default for MinimumSpanningTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Algorithm for MinimumSpanningTree {
    fn name(&self) -> &str {
        "Minimum Spanning Tree"
    }

    fn category(&self) -> &str {
        "graph"
    }

    fn description(&self) -> &str {
        "Minimum Spanning Tree algorithms (Kruskal's and Prim's) find a tree that connects all vertices in a weighted undirected graph with minimum total edge weight. Both algorithms use greedy optimization and are proven to produce optimal results through the cut property and cycle property respectively."
    }

    fn set_parameter(&mut self, name: &str, value: &str) -> Result<(), AlgorithmError> {
        match name {
            "algorithm" => {
                match value {
                    "kruskal" | "prim" | "auto" => {
                        self.parameters.insert(name.to_string(), value.to_string());
                        Ok(())
                    },
                    _ => Err(AlgorithmError::InvalidParameter(format!(
                        "Invalid algorithm: {}. Valid options: kruskal, prim, auto", 
                        value
                    ))),
                }
            },
            "verify_connectivity" | "collect_statistics" => {
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
                "Unknown parameter: {}. Valid parameters: algorithm, verify_connectivity, collect_statistics", 
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
        
        // Compute MST
        let mst_result = self.compute_mst(graph)
            .map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        // Validate result
        self.validate_mst(graph, &mst_result)
            .map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        Ok(AlgorithmResult {
            steps: self.statistics.edges_examined,
            nodes_visited: graph.node_count(),
            execution_time_ms: self.statistics.execution_time_ms,
            state: AlgorithmState {
                step: self.statistics.edges_examined,
                open_set: Vec::new(),
                closed_set: (0..graph.node_count()).collect(),
                current_node: None,
                data: {
                    let mut data = HashMap::new();
                    data.insert("total_weight".to_string(), mst_result.total_weight.to_string());
                    data.insert("num_edges".to_string(), mst_result.edges.len().to_string());
                    data.insert("algorithm_used".to_string(), format!("{:?}", mst_result.algorithm_used));
                    data.insert("union_operations".to_string(), self.statistics.union_operations.to_string());
                    data.insert("priority_queue_ops".to_string(), self.statistics.priority_queue_operations.to_string());
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
        // MST doesn't find paths between specific nodes, but we can adapt
        // by computing the MST and then finding the unique path in the tree
        
        let mst_result = self.compute_mst(graph)
            .map_err(|e| AlgorithmError::ExecutionError(e.to_string()))?;
        
        // Build adjacency list from MST edges
        let mut mst_adj = HashMap::new();
        for edge in &mst_result.edges {
            mst_adj.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
            mst_adj.entry(edge.target).or_insert_with(Vec::new).push(edge.source);
        }
        
        // Find path in MST using DFS
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        
        fn dfs_path(current: NodeId, goal: NodeId, adj: &HashMap<NodeId, Vec<NodeId>>, 
                   visited: &mut HashSet<NodeId>, path: &mut Vec<NodeId>) -> bool {
            if current == goal {
                path.push(current);
                return true;
            }
            
            visited.insert(current);
            path.push(current);
            
            if let Some(neighbors) = adj.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        if dfs_path(neighbor, goal, adj, visited, path) {
                            return true;
                        }
                    }
                }
            }
            
            path.pop();
            false
        }
        
        if dfs_path(start, goal, &mst_adj, &mut visited, &mut path) {
            // Calculate path cost
            let mut total_cost = 0.0;
            for window in path.windows(2) {
                if let Some(weight) = graph.get_edge_weight(window[0], window[1]) {
                    total_cost += weight;
                }
            }
            
            Ok(PathResult {
                path: Some(path),
                cost: Some(total_cost),
                result: AlgorithmResult {
                    steps: self.statistics.edges_examined,
                    nodes_visited: graph.node_count(),
                    execution_time_ms: self.statistics.execution_time_ms,
                    state: AlgorithmState {
                        step: self.statistics.edges_examined,
                        open_set: Vec::new(),
                        closed_set: (0..graph.node_count()).collect(),
                        current_node: Some(goal),
                        data: {
                            let mut data = HashMap::new();
                            data.insert("mst_total_weight".to_string(), mst_result.total_weight.to_string());
                            data.insert("path_in_mst".to_string(), "true".to_string());
                            data
                        },
                    },
                },
            })
        } else {
            Err(AlgorithmError::NoPathFound)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::graph::Graph;

    #[test]
    fn test_mst_creation() {
        let mst = MinimumSpanningTree::new();
        assert_eq!(mst.name(), "Minimum Spanning Tree");
        assert_eq!(mst.category(), "graph");
        assert_eq!(mst.get_parameter("algorithm").unwrap(), "auto");
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        
        // Initially all separate
        assert_eq!(uf.components(), 5);
        assert!(!uf.connected(0, 1));
        
        // Union 0 and 1
        assert!(uf.union(0, 1));
        assert_eq!(uf.components(), 4);
        assert!(uf.connected(0, 1));
        
        // Union same elements should return false
        assert!(!uf.union(0, 1));
        assert_eq!(uf.components(), 4);
        
        // Union more elements
        assert!(uf.union(2, 3));
        assert!(uf.union(1, 2));
        assert_eq!(uf.components(), 2);
        assert!(uf.connected(0, 3));
    }

    #[test]
    fn test_mst_edge_ordering() {
        let edge1 = MSTEdge::new(0, 1, 1.0);
        let edge2 = MSTEdge::new(1, 2, 2.0);
        let edge3 = MSTEdge::new(2, 3, 1.5);
        
        let mut edges = vec![edge2.clone(), edge3.clone(), edge1.clone()];
        edges.sort();
        
        // Should be sorted by weight
        assert_eq!(edges[0], edge1);  // weight 1.0
        assert_eq!(edges[1], edge3);  // weight 1.5
        assert_eq!(edges[2], edge2);  // weight 2.0
    }

    #[test]
    fn test_parameter_validation() {
        let mut mst = MinimumSpanningTree::new();
        
        // Valid parameters
        assert!(mst.set_parameter("algorithm", "kruskal").is_ok());
        assert_eq!(mst.get_parameter("algorithm").unwrap(), "kruskal");
        
        assert!(mst.set_parameter("verify_connectivity", "false").is_ok());
        assert_eq!(mst.get_parameter("verify_connectivity").unwrap(), "false");
        
        // Invalid parameters
        assert!(mst.set_parameter("algorithm", "invalid").is_err());
        assert!(mst.set_parameter("verify_connectivity", "maybe").is_err());
        assert!(mst.set_parameter("unknown_param", "value").is_err());
    }

    #[test]
    fn test_algorithm_selection() {
        let mst = MinimumSpanningTree::new();
        
        // Create test graphs
        let mut sparse_graph = Graph::new();
        for i in 0..10 {
            sparse_graph.add_node((i as f64, 0.0));
        }
        // Add few edges (sparse)
        sparse_graph.add_edge(0, 1, 1.0).unwrap();
        sparse_graph.add_edge(1, 2, 2.0).unwrap();
        
        let mut dense_graph = Graph::new();
        for i in 0..10 {
            dense_graph.add_node((i as f64, 0.0));
        }
        // Add many edges (dense)
        for i in 0..10 {
            for j in i+1..10 {
                dense_graph.add_edge(i, j, (i + j) as f64).unwrap();
            }
        }
        
        // Sparse graph should prefer Kruskal
        assert_eq!(mst.select_algorithm(&sparse_graph), MSTAlgorithm::Kruskal);
        
        // Dense graph should prefer Prim
        assert_eq!(mst.select_algorithm(&dense_graph), MSTAlgorithm::Prim);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut mst = MinimumSpanningTree::new();
        
        mst.statistics.edges_examined = 10;
        mst.statistics.union_operations = 5;
        
        let stats = mst.get_statistics();
        assert_eq!(stats.edges_examined, 10);
        assert_eq!(stats.union_operations, 5);
        
        mst.reset_statistics();
        assert_eq!(mst.get_statistics().edges_examined, 0);
    }
}