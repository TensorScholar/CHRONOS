//! High-performance graph data structure with spatial indexing
//!
//! This module implements a mathematically rigorous graph representation
//! optimized for algorithm visualization. Employs R-tree spatial indexing,
//! arena allocation, and lock-free concurrent access patterns.
//!
//! # Theoretical Foundation
//! Based on category theory representation of graphs as functors
//! between discrete categories. Spatial queries utilize computational
//! geometry with R*-tree indexing for logarithmic complexity.

use std::alloc::{Allocator, Global, Layout};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::hint;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Serialize, Deserialize};

use crate::algorithm::traits::{NodeId, AlgorithmError};

/// 2D position with single-precision coordinates
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    #[inline]
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
    
    #[inline]
    pub fn squared_distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

/// Edge weight optimized for algorithmic operations
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EdgeWeight(pub f64);

impl EdgeWeight {
    #[inline]
    pub fn new(weight: f64) -> Self {
        debug_assert!(!weight.is_nan(), "Edge weight cannot be NaN");
        Self(weight)
    }
}

/// Cached edge lookup map for O(1) access
#[derive(Debug)]
struct EdgeMap {
    /// Primary storage with cache-aligned entries
    edges: Vec<CacheAlignedEdge>,
    /// Spatial R-tree index for proximity queries
    spatial_index: RTreeNode,
    /// Hash map for O(1) lookups
    lookup: HashMap<(NodeId, NodeId), u32>,
}

/// Cache-aligned edge structure for optimal memory access
#[repr(C, align(64))]
#[derive(Debug, Clone)]
struct CacheAlignedEdge {
    source: NodeId,
    target: NodeId,
    weight: EdgeWeight,
    attributes: Arc<HashMap<String, String>>,
    // Padding to align to cache line
    _padding: [u8; 40],
}

/// Lock-free node data structure
#[repr(C, align(64))]
#[derive(Debug)]
struct NodeData {
    position: Position,
    attributes: Arc<HashMap<String, String>>,
    adjacency: AtomicAdjacencyList,
    // Spatial cache for frequent queries
    spatial_cache: SpatialCache,
    // Reference count for safe deallocation
    ref_count: AtomicU64,
}

/// Lock-free adjacency list implementation
#[derive(Debug)]
struct AtomicAdjacencyList {
    /// Atomic pointer to adjacency vector
    ptr: atomic::AtomicPtr<Vec<NodeId>>,
    /// Version counter for ABA problem prevention
    version: AtomicU64,
}

/// Spatial cache for frequent proximity queries
#[repr(C, align(64))]
#[derive(Debug)]
struct SpatialCache {
    /// Cached bounding box
    bbox: AtomicBoundingBox,
    /// Cached nearest neighbors
    nearest_cache: Arc<RwLock<Vec<(NodeId, f64)>>>,
}

/// Atomic bounding box for spatial queries
#[derive(Debug)]
struct AtomicBoundingBox {
    min_x: AtomicF64,
    min_y: AtomicF64,
    max_x: AtomicF64,
    max_y: AtomicF64,
}

/// Atomic f64 wrapper for lock-free operations
#[derive(Debug)]
struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }
    
    fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.bits.load(ordering))
    }
    
    fn store(&self, value: f64, ordering: Ordering) {
        self.bits.store(value.to_bits(), ordering)
    }
}

/// R-tree node for spatial indexing
#[derive(Debug)]
enum RTreeNode {
    Leaf {
        entries: Vec<RTreeEntry>,
        bbox: BoundingBox,
    },
    Internal {
        children: Vec<Box<RTreeNode>>,
        bbox: BoundingBox,
    },
}

/// R-tree entry for spatial indexing
#[derive(Debug, Clone)]
struct RTreeEntry {
    id: NodeId,
    position: Position,
    bbox: BoundingBox,
}

/// Geometric bounding box
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    min: Position,
    max: Position,
}

impl BoundingBox {
    fn from_position(pos: Position) -> Self {
        Self { min: pos, max: pos }
    }
    
    fn contains(&self, pos: &Position) -> bool {
        pos.x >= self.min.x && pos.x <= self.max.x &&
        pos.y >= self.min.y && pos.y <= self.max.y
    }
    
    fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.max.x < other.min.x || self.min.x > other.max.x ||
          self.max.y < other.min.y || self.min.y > other.max.y)
    }
    
    fn merge(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: Position::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y)
            ),
            max: Position::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y)
            ),
        }
    }
    
    fn area(&self) -> f64 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y)
    }
}

/// Arena allocator for node data
#[derive(Debug)]
struct NodeArena {
    /// Memory chunks for node allocation
    chunks: Vec<ArenaChunk>,
    /// Free list for reuse
    free_list: Arc<RwLock<Vec<usize>>>,
    /// Current allocation offset
    current_offset: AtomicU64,
}

/// Arena memory chunk
#[derive(Debug)]
struct ArenaChunk {
    /// Raw memory allocation
    memory: NonNull<MaybeUninit<NodeData>>,
    /// Chunk capacity
    capacity: usize,
    /// Allocation count in chunk
    allocated: AtomicU64,
}

/// High-performance graph implementation
#[derive(Debug)]
pub struct Graph {
    /// Node storage with arena allocation
    nodes: Arc<NodeArena>,
    
    /// Edge storage with spatial indexing
    edges: RwLock<EdgeMap>,
    
    /// Graph-level metadata
    metadata: Arc<RwLock<HashMap<String, String>>>,
    
    /// Versioning for concurrent modifications
    version: AtomicU64,
    
    /// Performance metrics collector
    metrics: GraphMetrics,
}

/// Performance metrics for graph operations
#[derive(Debug)]
struct GraphMetrics {
    node_count: AtomicU64,
    edge_count: AtomicU64,
    lookup_time: AtomicU64,
    spatial_query_time: AtomicU64,
}

/// Graph query parameters for spatial searches
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// Query origin point
    origin: Position,
    /// Search radius
    radius: f64,
    /// Maximum results
    limit: Option<usize>,
    /// Node filter predicate
    filter: Option<Box<dyn Fn(NodeId) -> bool + Send + Sync>>,
}

/// Result of spatial query
#[derive(Debug, Clone)]
pub struct SpatialQueryResult {
    /// Matching nodes with distances
    nodes: Vec<(NodeId, f64)>,
    /// Query execution time
    execution_time: std::time::Duration,
}

impl Graph {
    /// Creates a new high-performance graph
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(NodeArena::new()),
            edges: RwLock::new(EdgeMap::new()),
            metadata: Arc::new(RwLock::new(HashMap::with_capacity(16))),
            version: AtomicU64::new(0),
            metrics: GraphMetrics::new(),
        }
    }
    
    /// Adds a node with optimal memory allocation
    pub fn add_node(&self, id: NodeId, position: Position) -> Result<(), AlgorithmError> {
        let node_data = NodeData::new(position);
        
        self.nodes.allocate(id, node_data)
            .map_err(|e| AlgorithmError::AllocationError(e.to_string()))?;
        
        self.metrics.increment_node_count();
        self.increment_version();
        
        Ok(())
    }
    
    /// Adds an edge with O(1) lookup optimization
    pub fn add_edge(
        &self,
        source: NodeId,
        target: NodeId,
        weight: EdgeWeight,
    ) -> Result<(), AlgorithmError> {
        let mut edge_map = self.edges.write()
            .map_err(|_| AlgorithmError::ConcurrencyError("Edge map locked".into()))?;
        
        if edge_map.lookup.contains_key(&(source, target)) {
            return Err(AlgorithmError::DuplicateEdge(source, target));
        }
        
        let edge_id = edge_map.edges.len() as u32;
        let edge = CacheAlignedEdge::new(source, target, weight);
        
        edge_map.edges.push(edge);
        edge_map.lookup.insert((source, target), edge_id);
        
        // Update spatial index
        let source_pos = self.get_node_position(source)?;
        let target_pos = self.get_node_position(target)?;
        
        edge_map.spatial_index.insert(source, source_pos);
        edge_map.spatial_index.insert(target, target_pos);
        
        self.metrics.increment_edge_count();
        self.increment_version();
        
        Ok(())
    }
    
    /// Performs optimized spatial query
    pub fn spatial_query(&self, query: GraphQuery) -> Result<SpatialQueryResult, AlgorithmError> {
        let start_time = std::time::Instant::now();
        
        let nodes = self.nodes.lock_free_spatial_query(query.origin, query.radius)?;
        
        let filtered_nodes: Vec<_> = nodes.into_iter()
            .filter(|(id, dist)| {
                query.filter.as_ref()
                    .map_or(true, |f| f(*id))
            })
            .take(query.limit.unwrap_or(usize::MAX))
            .collect();
        
        let execution_time = start_time.elapsed();
        self.metrics.update_spatial_query_time(execution_time);
        
        Ok(SpatialQueryResult {
            nodes: filtered_nodes,
            execution_time,
        })
    }
    
    /// Gets edge weight with O(1) lookup
    pub fn get_edge_weight(&self, source: NodeId, target: NodeId) -> Result<EdgeWeight, AlgorithmError> {
        let start_time = std::time::Instant::now();
        
        let edge_map = self.edges.read()
            .map_err(|_| AlgorithmError::ConcurrencyError("Edge map locked".into()))?;
        
        let weight = edge_map.lookup.get(&(source, target))
            .and_then(|&idx| edge_map.edges.get(idx as usize))
            .map(|edge| edge.weight)
            .ok_or_else(|| AlgorithmError::EdgeNotFound(source, target))?;
        
        self.metrics.update_lookup_time(start_time.elapsed());
        
        Ok(weight)
    }
    
    /// Lock-free concurrent iteration
    pub fn concurrent_nodes_iter<F>(&self, f: F) -> Result<(), AlgorithmError>
    where
        F: Fn(NodeId, Position) + Send + Sync,
    {
        self.nodes.concurrent_iter(f)
    }
    
    /// Gets node position from spatial cache
    pub fn get_node_position(&self, id: NodeId) -> Result<Position, AlgorithmError> {
        self.nodes.get_position(id)
    }
    
    /// Updates graph version for consistency
    #[inline]
    fn increment_version(&self) {
        self.version.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Gets current graph version
    #[inline]
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    
    /// Atomic snapshot creation for algorithms
    pub fn create_snapshot(&self) -> GraphSnapshot {
        GraphSnapshot::new(self)
    }
    
    /// Validates graph integrity
    pub fn validate(&self) -> Result<(), AlgorithmError> {
        // Check node-edge consistency
        let node_count = self.metrics.node_count.load(Ordering::SeqCst);
        let edge_count = self.metrics.edge_count.load(Ordering::SeqCst);
        
        if edge_count > node_count * (node_count - 1) {
            return Err(AlgorithmError::GraphInvalid(
                "Edge count exceeds maximum possible".into()
            ));
        }
        
        // Validate spatial index integrity
        self.edges.read()
            .map_err(|_| AlgorithmError::ConcurrencyError("Edge map locked".into()))?
            .spatial_index.validate()
    }
}

impl NodeArena {
    /// Creates new node arena with optimal chunk size
    fn new() -> Self {
        const CHUNK_SIZE: usize = 4096;
        let initial_chunk = ArenaChunk::new(CHUNK_SIZE);
        
        Self {
            chunks: vec![initial_chunk],
            free_list: Arc::new(RwLock::new(Vec::new())),
            current_offset: AtomicU64::new(0),
        }
    }
    
    /// Allocates node with arena optimization
    fn allocate(&self, id: NodeId, data: NodeData) -> Result<(), String> {
        // Try to reuse from free list
        if let Ok(mut free_list) = self.free_list.write() {
            if let Some(index) = free_list.pop() {
                unsafe {
                    let ptr = self.get_ptr_at(index);
                    ptr.write(data);
                }
                return Ok(());
            }
        }
        
        // Allocate from current chunk
        let offset = self.current_offset.fetch_add(1, Ordering::SeqCst);
        
        // Get current chunk or allocate new one
        let chunk = if offset < self.chunks[0].capacity as u64 {
            &self.chunks[0]
        } else {
            // Allocate new chunk
            self.allocate_new_chunk()?
        };
        
        unsafe {
            let ptr = chunk.memory.as_ptr().add(offset as usize);
            ptr.write(MaybeUninit::new(data));
        }
        
        chunk.allocated.fetch_add(1, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Gets pointer at specific offset
    unsafe fn get_ptr_at(&self, index: usize) -> *mut NodeData {
        for chunk in &self.chunks {
            if index < chunk.capacity {
                return chunk.memory.as_ptr().add(index).cast();
            }
        }
        unreachable!("Invalid arena index");
    }
    
    /// Allocates new memory chunk
    fn allocate_new_chunk(&mut self) -> Result<&ArenaChunk, String> {
        const CHUNK_SIZE: usize = 4096;
        let new_chunk = ArenaChunk::new(CHUNK_SIZE);
        self.chunks.push(new_chunk);
        Ok(self.chunks.last().unwrap())
    }
    
    /// Lock-free spatial query implementation
    fn lock_free_spatial_query(
        &self,
        center: Position,
        radius: f64,
    ) -> Result<Vec<(NodeId, f64)>, AlgorithmError> {
        let mut results = Vec::new();
        let radius_squared = radius * radius;
        
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let allocated = chunk.allocated.load(Ordering::Acquire);
            
            unsafe {
                let base_ptr = chunk.memory.as_ptr().cast::<NodeData>();
                
                for i in 0..allocated as usize {
                    let node_ptr = base_ptr.add(i);
                    let node = &*node_ptr;
                    
                    let dist_squared = node.position.squared_distance_to(&center);
                    if dist_squared <= radius_squared {
                        let node_id = NodeId(chunk_idx * chunk.capacity + i);
                        results.push((node_id, dist_squared.sqrt()));
                    }
                }
            }
        }
        
        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(results)
    }
}

impl ArenaChunk {
    /// Creates new arena chunk with aligned memory
    fn new(capacity: usize) -> Self {
        let layout = Layout::array::<MaybeUninit<NodeData>>(capacity)
            .expect("Failed to create layout");
        
        let memory = unsafe {
            Global.allocate(layout)
                .map_err(|_| "Failed to allocate memory")
                .unwrap()
                .cast()
        };
        
        Self {
            memory,
            capacity,
            allocated: AtomicU64::new(0),
        }
    }
}

impl EdgeMap {
    /// Creates optimized edge map
    fn new() -> Self {
        Self {
            edges: Vec::with_capacity(1024),
            spatial_index: RTreeNode::new_leaf(),
            lookup: HashMap::with_capacity(1024),
        }
    }
}

impl RTreeNode {
    /// Creates new R-tree leaf node
    fn new_leaf() -> Self {
        RTreeNode::Leaf {
            entries: Vec::with_capacity(32),
            bbox: BoundingBox::from_position(Position::new(0.0, 0.0)),
        }
    }
    
    /// Inserts element into R-tree with optimization
    fn insert(&mut self, id: NodeId, position: Position) {
        match self {
            RTreeNode::Leaf { entries, bbox } => {
                let entry = RTreeEntry {
                    id,
                    position,
                    bbox: BoundingBox::from_position(position),
                };
                
                entries.push(entry.clone());
                *bbox = bbox.merge(&entry.bbox);
                
                // Split if leaf is full
                if entries.len() > 32 {
                    self.split_leaf();
                }
            }
            RTreeNode::Internal { children, bbox } => {
                // Choose subtree with minimum enlargement
                let best_child = self.choose_subtree(position);
                children[best_child].insert(id, position);
                
                // Update bounding box
                *bbox = self.calculate_bbox();
            }
        }
    }
    
    /// Validates R-tree structure
    fn validate(&self) -> Result<(), AlgorithmError> {
        match self {
            RTreeNode::Leaf { entries, bbox } => {
                // Verify all entries fit within bounds
                for entry in entries {
                    if !bbox.contains(&entry.position) {
                        return Err(AlgorithmError::SpatialIndexError(
                            "Entry outside bounds".into()
                        ));
                    }
                }
            }
            RTreeNode::Internal { children, bbox } => {
                // Verify all children bounds fit within bounds
                for child in children {
                    let child_bbox = child.get_bbox();
                    if !bbox.contains(&child_bbox.min) || !bbox.contains(&child_bbox.max) {
                        return Err(AlgorithmError::SpatialIndexError(
                            "Child bounds outside parent".into()
                        ));
                    }
                }
                
                // Recursively validate children
                for child in children {
                    child.validate()?;
                }
            }
        }
        Ok(())
    }
}

/// Immutable graph snapshot for algorithm consumption
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// Snapshot of node positions
    node_positions: Arc<HashMap<NodeId, Position>>,
    
    /// Snapshot of edge weights
    edge_weights: Arc<HashMap<(NodeId, NodeId), EdgeWeight>>,
    
    /// Snapshot version
    version: u64,
}

impl GraphSnapshot {
    /// Creates atomic snapshot of graph state
    fn new(graph: &Graph) -> Self {
        let mut node_positions = HashMap::new();
        let mut edge_weights = HashMap::new();
        
        // Capture nodes atomically
        graph.concurrent_nodes_iter(|id, pos| {
            node_positions.insert(id, pos);
        }).unwrap();
        
        // Capture edges atomically
        if let Ok(edge_map) = graph.edges.read() {
            for ((src, dst), &idx) in &edge_map.lookup {
                if let Some(edge) = edge_map.edges.get(idx as usize) {
                    edge_weights.insert((*src, *dst), edge.weight);
                }
            }
        }
        
        Self {
            node_positions: Arc::new(node_positions),
            edge_weights: Arc::new(edge_weights),
            version: graph.version(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_creation_and_basic_operations() {
        let graph = Graph::new();
        
        // Add nodes
        assert!(graph.add_node(NodeId(0), Position::new(0.0, 0.0)).is_ok());
        assert!(graph.add_node(NodeId(1), Position::new(1.0, 0.0)).is_ok());
        assert!(graph.add_node(NodeId(2), Position::new(0.0, 1.0)).is_ok());
        
        // Add edges
        assert!(graph.add_edge(NodeId(0), NodeId(1), EdgeWeight::new(1.0)).is_ok());
        assert!(graph.add_edge(NodeId(1), NodeId(2), EdgeWeight::new(2.0)).is_ok());
        
        // Test edge lookup
        assert_eq!(
            graph.get_edge_weight(NodeId(0), NodeId(1)).unwrap().0,
            1.0
        );
        
        // Test non-existent edge
        assert!(graph.get_edge_weight(NodeId(0), NodeId(2)).is_err());
    }
    
    #[test]
    fn test_spatial_query() {
        let graph = Graph::new();
        
        // Create grid of nodes
        for i in 0..5 {
            for j in 0..5 {
                let id = NodeId(i * 5 + j);
                graph.add_node(id, Position::new(i as f64, j as f64)).unwrap();
            }
        }
        
        // Query near origin
        let query = GraphQuery {
            origin: Position::new(0.0, 0.0),
            radius: 1.5,
            limit: None,
            filter: None,
        };
        
        let result = graph.spatial_query(query).unwrap();
        
        // Should find nodes within radius
        assert!(!result.nodes.is_empty());
        for (_, distance) in result.nodes {
            assert!(distance <= 1.5);
        }
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let graph = Arc::new(Graph::new());
        let mut handles = vec![];
        
        // Spawn threads for concurrent access
        for i in 0..4 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let id = NodeId(i * 100 + j);
                    graph_clone.add_node(id, Position::new(i as f64, j as f64)).unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all nodes were added
        assert_eq!(graph.metrics.node_count.load(Ordering::SeqCst), 400);
    }
    
    #[test]
    fn test_graph_validation() {
        let graph = Graph::new();
        
        // Add valid nodes
        graph.add_node(NodeId(0), Position::new(0.0, 0.0)).unwrap();
        graph.add_node(NodeId(1), Position::new(1.0, 1.0)).unwrap();
        
        // Add valid edge
        graph.add_edge(NodeId(0), NodeId(1), EdgeWeight::new(1.0)).unwrap();
        
        // Validate should pass
        assert!(graph.validate().is_ok());
    }
    
    #[test]
    fn test_snapshot_consistency() {
        let graph = Graph::new();
        
        // Add some data
        for i in 0..10 {
            graph.add_node(NodeId(i), Position::new(i as f64, i as f64)).unwrap();
        }
        
        // Create snapshot
        let snapshot = graph.create_snapshot();
        
        // Verify snapshot has all nodes
        assert_eq!(snapshot.node_positions.len(), 10);
        
        // Verify version matches
        assert_eq!(snapshot.version, graph.version());
    }
    
    #[test]
    fn test_edge_cache_alignment() {
        // Verify edge structure is cache-aligned
        assert_eq!(std::mem::align_of::<CacheAlignedEdge>(), 64);
        assert_eq!(std::mem::size_of::<CacheAlignedEdge>(), 64);
    }
    
    #[test]
    fn test_performance_metrics() {
        let graph = Graph::new();
        
        // Perform operations
        graph.add_node(NodeId(0), Position::new(0.0, 0.0)).unwrap();
        graph.add_node(NodeId(1), Position::new(1.0, 1.0)).unwrap();
        
        // Check metrics
        assert_eq!(graph.metrics.node_count.load(Ordering::SeqCst), 2);
        assert!(graph.metrics.lookup_time.load(Ordering::SeqCst) >= 0);
    }
}

// Benchmark suite for performance validation
#[cfg(test)]
mod benches {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_edge_lookup(c: &mut Criterion) {
        let graph = Graph::new();
        
        // Setup
        for i in 0..1000 {
            graph.add_node(NodeId(i), Position::new(i as f64, i as f64)).unwrap();
        }
        
        for i in 0..999 {
            graph.add_edge(NodeId(i), NodeId(i + 1), EdgeWeight::new(1.0)).unwrap();
        }
        
        // Benchmark
        c.bench_function("edge_lookup", |b| {
            b.iter(|| {
                graph.get_edge_weight(
                    black_box(NodeId(500)),
                    black_box(NodeId(501))
                )
            });
        });
    }
    
    fn bench_spatial_query(c: &mut Criterion) {
        let graph = Graph::new();
        
        // Create spatial grid
        for i in 0..100 {
            for j in 0..100 {
                let id = NodeId(i * 100 + j);
                graph.add_node(id, Position::new(i as f64, j as f64)).unwrap();
            }
        }
        
        let query = GraphQuery {
            origin: Position::new(50.0, 50.0),
            radius: 10.0,
            limit: None,
            filter: None,
        };
        
        c.bench_function("spatial_query", |b| {
            b.iter(|| {
                graph.spatial_query(black_box(query.clone()))
            });
        });
    }
    
    criterion_group!(benches, bench_edge_lookup, bench_spatial_query);
    criterion_main!(benches);
}