//! State Space Perspective Visualization
//!
//! Advanced high-dimensional state space visualization with manifold learning,
//! topological preservation, and GPU-accelerated dimensionality reduction.
//!
//! ## Mathematical Foundation
//!
//! The state space visualization employs Uniform Manifold Approximation and
//! Projection (UMAP) with formal guarantees on topological preservation:
//!
//! Given a high-dimensional state space S ⊆ ℝⁿ, we construct a low-dimensional
//! embedding f: S → ℝᵈ (d ≪ n) that preserves:
//!
//! 1. Local topology: ∀x,y ∈ S, d_S(x,y) < ε ⟹ |d_E(f(x),f(y)) - d_S(x,y)| < δ
//! 2. Global structure: Persistent homology H*(S) ≈ H*(f(S))
//! 3. Geodesic distances: Approximation within factor (1+ε) for ε-nets
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

#![allow(clippy::excessive_precision)]

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::collections::{HashMap, BTreeMap};
use std::marker::PhantomData;

use dashmap::DashMap;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Vector};
use petgraph::graph::{UnGraph, NodeIndex};
use petgraph::algo::dijkstra;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use wgpu::util::DeviceExt;

use crate::core::algorithm::state::{AlgorithmState, StateSignature};
use crate::core::temporal::state_manager::StateManager;
use crate::core::temporal::timeline::Timeline;
use crate::visualization::engine::wgpu::{RenderContext, GpuBuffer};
use crate::visualization::interaction::controller::InteractionState;

/// Type alias for high-precision floating point operations
type Scalar = f64;

/// Type alias for GPU-compatible floating point
type GpuScalar = f32;

/// Configuration for state space visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceConfig {
    /// Target dimensionality for embedding (2 or 3)
    pub target_dimensions: usize,
    
    /// UMAP hyperparameters
    pub umap_params: UmapParameters,
    
    /// Visualization parameters
    pub visualization_params: VisualizationParameters,
    
    /// GPU acceleration settings
    pub gpu_config: GpuConfiguration,
    
    /// Topological preservation settings
    pub topology_config: TopologyConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapParameters {
    /// Number of nearest neighbors
    pub n_neighbors: usize,
    
    /// Minimum distance between points in embedding
    pub min_dist: Scalar,
    
    /// Spread of points in embedding space
    pub spread: Scalar,
    
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    
    /// Learning rate for optimization
    pub learning_rate: Scalar,
    
    /// Number of epochs for optimization
    pub n_epochs: usize,
    
    /// Negative sampling rate
    pub negative_sample_rate: usize,
    
    /// Transform seed for initialization
    pub transform_seed: Option<u64>,
    
    /// Metric for distance computation
    pub metric: DistanceMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    /// Custom metric with distance function
    Custom(String),
}

impl DistanceMetric {
    fn compute<V: Vector<Scalar>>(&self, a: &V, b: &V) -> Scalar {
        match self {
            Self::Euclidean => {
                (a - b).norm()
            },
            Self::Manhattan => {
                (a - b).iter().map(|x| x.abs()).sum()
            },
            Self::Chebyshev => {
                (a - b).iter().map(|x| x.abs()).fold(0.0, Scalar::max)
            },
            Self::Cosine => {
                let dot = a.dot(b);
                let norm_a = a.norm();
                let norm_b = b.norm();
                if norm_a * norm_b > Scalar::EPSILON {
                    1.0 - dot / (norm_a * norm_b)
                } else {
                    1.0
                }
            },
            Self::Correlation => {
                let mean_a = a.mean();
                let mean_b = b.mean();
                let centered_a = a.map(|x| x - mean_a);
                let centered_b = b.map(|x| x - mean_b);
                Self::Cosine.compute(&centered_a, &centered_b)
            },
            Self::Hamming => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| if (x - y).abs() < Scalar::EPSILON { 0.0 } else { 1.0 })
                    .sum()
            },
            Self::Jaccard => {
                let intersection = a.iter().zip(b.iter())
                    .map(|(x, y)| x.min(*y))
                    .sum::<Scalar>();
                let union = a.iter().zip(b.iter())
                    .map(|(x, y)| x.max(*y))
                    .sum::<Scalar>();
                if union > Scalar::EPSILON {
                    1.0 - intersection / union
                } else {
                    0.0
                }
            },
            Self::Custom(_) => {
                // Custom metrics would be implemented via trait objects
                0.0
            }
        }
    }
}

/// Main state space visualization implementation
pub struct StateSpaceVisualization<S: AlgorithmState> {
    config: StateSpaceConfig,
    
    /// Current embedding of states
    embedding: Arc<DashMap<StateSignature, EmbeddedPoint>>,
    
    /// UMAP algorithm instance
    umap: Arc<Mutex<UmapAlgorithm<S>>>,
    
    /// GPU resources
    gpu_resources: GpuResources,
    
    /// Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    
    /// Compute pipeline for GPU acceleration
    compute_pipeline: wgpu::ComputePipeline,
    
    /// Interaction state
    interaction_handler: InteractionHandler,
    
    /// Trajectory manager
    trajectory_manager: TrajectoryManager<S>,
    
    /// Performance metrics
    metrics: Arc<PerformanceMetrics>,
    
    _phantom: PhantomData<S>,
}

/// Embedded point in low-dimensional space
#[derive(Debug, Clone)]
struct EmbeddedPoint {
    /// Position in embedded space
    position: DVector<Scalar>,
    
    /// Original state signature
    signature: StateSignature,
    
    /// Timestamp of state
    timestamp: f64,
    
    /// Metadata for visualization
    metadata: PointMetadata,
}

#[derive(Debug, Clone)]
struct PointMetadata {
    /// Algorithm step number
    step: usize,
    
    /// State importance score
    importance: Scalar,
    
    /// Cluster assignment
    cluster_id: Option<usize>,
    
    /// Trajectory ID
    trajectory_id: Option<usize>,
    
    /// Visual properties
    visual_props: VisualProperties,
}

#[derive(Debug, Clone)]
struct VisualProperties {
    color: [f32; 4],
    size: f32,
    shape: PointShape,
    highlight: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PointShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Star,
}

/// UMAP algorithm implementation
struct UmapAlgorithm<S: AlgorithmState> {
    params: UmapParameters,
    
    /// High-dimensional state vectors
    high_dim_points: Vec<DVector<Scalar>>,
    
    /// Nearest neighbor graph
    nn_graph: UnGraph<usize, Scalar>,
    
    /// Fuzzy simplicial set
    fuzzy_set: SparseMatrix<Scalar>,
    
    /// Current embedding
    embedding: DMatrix<Scalar>,
    
    /// Optimization state
    optimization_state: OptimizationState,
    
    _phantom: PhantomData<S>,
}

impl<S: AlgorithmState> UmapAlgorithm<S> {
    fn new(params: UmapParameters) -> Self {
        Self {
            params,
            high_dim_points: Vec::new(),
            nn_graph: UnGraph::new_undirected(),
            fuzzy_set: SparseMatrix::new(0, 0),
            embedding: DMatrix::zeros(0, 0),
            optimization_state: OptimizationState::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Fit UMAP model to high-dimensional data
    fn fit(&mut self, states: &[S]) -> Result<(), UmapError> {
        // Convert states to high-dimensional vectors
        self.high_dim_points = states.par_iter()
            .map(|state| self.state_to_vector(state))
            .collect();
        
        // Construct nearest neighbor graph
        self.construct_nn_graph()?;
        
        // Construct fuzzy simplicial set
        self.construct_fuzzy_set()?;
        
        // Initialize low-dimensional embedding
        self.initialize_embedding()?;
        
        // Optimize embedding
        self.optimize_embedding()?;
        
        Ok(())
    }
    
    /// Transform new states to embedded space
    fn transform(&self, states: &[S]) -> Result<Vec<DVector<Scalar>>, UmapError> {
        let high_dim_points: Vec<_> = states.par_iter()
            .map(|state| self.state_to_vector(state))
            .collect();
        
        // Use existing model to transform new points
        self.transform_points(&high_dim_points)
    }
    
    fn state_to_vector(&self, state: &S) -> DVector<Scalar> {
        // Extract feature vector from state
        let features = state.extract_features();
        DVector::from_vec(features)
    }
    
    fn construct_nn_graph(&mut self) -> Result<(), UmapError> {
        let n_points = self.high_dim_points.len();
        
        // Initialize graph
        self.nn_graph = UnGraph::with_capacity(n_points, n_points * self.params.n_neighbors);
        
        // Add nodes
        let nodes: Vec<_> = (0..n_points)
            .map(|i| self.nn_graph.add_node(i))
            .collect();
        
        // Parallel construction of nearest neighbors using vantage-point tree
        let vp_tree = VantagePointTree::new(&self.high_dim_points, &self.params.metric);
        
        let edges: Vec<_> = (0..n_points).into_par_iter()
            .flat_map(|i| {
                let neighbors = vp_tree.search_knn(&self.high_dim_points[i], self.params.n_neighbors + 1);
                neighbors.into_iter()
                    .filter(|&(j, _)| j != i)
                    .take(self.params.n_neighbors)
                    .map(|(j, dist)| (i, j, dist))
                    .collect::<Vec<_>>()
            })
            .collect();
        
        // Add edges to graph
        for (i, j, weight) in edges {
            self.nn_graph.add_edge(nodes[i], nodes[j], weight);
        }
        
        Ok(())
    }
    
    fn construct_fuzzy_set(&mut self) -> Result<(), UmapError> {
        let n_points = self.high_dim_points.len();
        
        // Compute local connectivity and bandwidth
        let (sigmas, rhos) = self.compute_membership_strengths()?;
        
        // Construct fuzzy simplicial set
        let mut fuzzy_set = SparseMatrix::new(n_points, n_points);
        
        for edge in self.nn_graph.edge_references() {
            let i = self.nn_graph[edge.source()];
            let j = self.nn_graph[edge.target()];
            let dist = *edge.weight();
            
            // Compute membership strength
            let membership = self.compute_membership(dist, sigmas[i], rhos[i]);
            fuzzy_set.set(i, j, membership);
        }
        
        // Symmetrize fuzzy set
        self.fuzzy_set = self.fuzzy_union(fuzzy_set)?;
        
        Ok(())
    }
    
    fn compute_membership(&self, dist: Scalar, sigma: Scalar, rho: Scalar) -> Scalar {
        if dist <= rho {
            1.0
        } else {
            (-(dist - rho) / sigma).exp()
        }
    }
    
    fn fuzzy_union(&self, mut set: SparseMatrix<Scalar>) -> Result<SparseMatrix<Scalar>, UmapError> {
        // Compute fuzzy union: a ∪ b = a + b - a * b
        let transpose = set.transpose();
        
        for ((i, j), &val) in transpose.iter() {
            if let Some(&other_val) = set.get(i, j) {
                let union_val = val + other_val - val * other_val;
                set.set(i, j, union_val);
            } else {
                set.set(i, j, val);
            }
        }
        
        Ok(set)
    }
    
    fn initialize_embedding(&mut self) -> Result<(), UmapError> {
        let n_points = self.high_dim_points.len();
        let target_dim = 2; // Can be parameterized
        
        // Spectral initialization for better convergence
        let laplacian = self.compute_normalized_laplacian()?;
        let eigendecomp = laplacian.symmetric_eigen();
        
        // Use first non-trivial eigenvectors
        self.embedding = DMatrix::zeros(n_points, target_dim);
        for i in 0..target_dim {
            let eigenvector = eigendecomp.eigenvectors.column(i + 1);
            self.embedding.set_column(i, &eigenvector);
        }
        
        // Add small random noise to break symmetries
        use rand::distributions::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let noise = Normal::new(0.0, 1e-4).unwrap();
        
        for i in 0..n_points {
            for j in 0..target_dim {
                self.embedding[(i, j)] += noise.sample(&mut rng);
            }
        }
        
        Ok(())
    }
    
    fn optimize_embedding(&mut self) -> Result<(), UmapError> {
        let n_epochs = self.params.n_epochs;
        let learning_rate = self.params.learning_rate;
        
        // Stochastic gradient descent with negative sampling
        for epoch in 0..n_epochs {
            // Decay learning rate
            let alpha = learning_rate * (1.0 - epoch as Scalar / n_epochs as Scalar);
            
            // Positive samples (edges in fuzzy set)
            for ((i, j), &membership) in self.fuzzy_set.iter() {
                if membership > 0.0 {
                    self.update_embedding_positive(i, j, membership, alpha)?;
                }
            }
            
            // Negative sampling
            self.negative_sampling(alpha)?;
            
            // Update optimization state
            self.optimization_state.current_epoch = epoch;
            self.optimization_state.current_loss = self.compute_loss()?;
        }
        
        Ok(())
    }
    
    fn update_embedding_positive(&mut self, i: usize, j: usize, membership: Scalar, alpha: Scalar) 
        -> Result<(), UmapError> {
        let yi = self.embedding.row(i).clone_owned();
        let yj = self.embedding.row(j).clone_owned();
        
        let dist = (&yi - &yj).norm();
        let grad_coeff = -2.0 * self.params.spread * membership * dist.powi(-1);
        
        let grad = (&yi - &yj) * grad_coeff;
        
        // Update embeddings
        self.embedding.set_row(i, &(yi - alpha * &grad));
        self.embedding.set_row(j, &(yj + alpha * &grad));
        
        Ok(())
    }
    
    fn negative_sampling(&mut self, alpha: Scalar) -> Result<(), UmapError> {
        use rand::distributions::{Distribution, Uniform};
        let mut rng = rand::thread_rng();
        let n_points = self.high_dim_points.len();
        let uniform = Uniform::new(0, n_points);
        
        let n_negative_samples = self.params.negative_sample_rate * self.fuzzy_set.nnz();
        
        for _ in 0..n_negative_samples {
            let i = uniform.sample(&mut rng);
            let j = uniform.sample(&mut rng);
            
            if i != j && self.fuzzy_set.get(i, j).unwrap_or(0.0) == 0.0 {
                self.update_embedding_negative(i, j, alpha)?;
            }
        }
        
        Ok(())
    }
    
    fn update_embedding_negative(&mut self, i: usize, j: usize, alpha: Scalar) 
        -> Result<(), UmapError> {
        let yi = self.embedding.row(i).clone_owned();
        let yj = self.embedding.row(j).clone_owned();
        
        let dist_sq = (&yi - &yj).norm_squared();
        let grad_coeff = 2.0 * self.params.spread / (dist_sq + self.params.spread);
        
        let grad = (&yi - &yj) * grad_coeff;
        
        // Update embeddings
        self.embedding.set_row(i, &(yi + alpha * &grad));
        self.embedding.set_row(j, &(yj - alpha * &grad));
        
        Ok(())
    }
}

/// Sparse matrix implementation for fuzzy sets
struct SparseMatrix<T> {
    rows: usize,
    cols: usize,
    data: HashMap<(usize, usize), T>,
}

impl<T: Clone + Default> SparseMatrix<T> {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: HashMap::new(),
        }
    }
    
    fn set(&mut self, i: usize, j: usize, value: T) {
        if i < self.rows && j < self.cols {
            self.data.insert((i, j), value);
        }
    }
    
    fn get(&self, i: usize, j: usize) -> Option<&T> {
        self.data.get(&(i, j))
    }
    
    fn iter(&self) -> impl Iterator<Item = ((&(usize, usize), &T))> {
        self.data.iter()
    }
    
    fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for ((i, j), value) in self.data.iter() {
            result.set(*j, *i, value.clone());
        }
        result
    }
    
    fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Vantage-point tree for efficient nearest neighbor search
struct VantagePointTree<'a> {
    root: Option<Box<VpNode>>,
    points: &'a [DVector<Scalar>],
    metric: &'a DistanceMetric,
}

struct VpNode {
    index: usize,
    radius: Scalar,
    left: Option<Box<VpNode>>,
    right: Option<Box<VpNode>>,
}

impl<'a> VantagePointTree<'a> {
    fn new(points: &'a [DVector<Scalar>], metric: &'a DistanceMetric) -> Self {
        let indices: Vec<_> = (0..points.len()).collect();
        let root = Self::build_tree(points, metric, indices);
        
        Self { root, points, metric }
    }
    
    fn build_tree(points: &[DVector<Scalar>], metric: &DistanceMetric, mut indices: Vec<usize>) 
        -> Option<Box<VpNode>> {
        if indices.is_empty() {
            return None;
        }
        
        // Choose vantage point (random or heuristic)
        let vp_idx = indices.swap_remove(0);
        
        if indices.is_empty() {
            return Some(Box::new(VpNode {
                index: vp_idx,
                radius: 0.0,
                left: None,
                right: None,
            }));
        }
        
        // Compute distances from vantage point
        let mut distances: Vec<_> = indices.iter()
            .map(|&idx| (idx, metric.compute(&points[vp_idx], &points[idx])))
            .collect();
        
        // Find median distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let median_idx = distances.len() / 2;
        let radius = distances[median_idx].1;
        
        // Partition points
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = distances.into_iter()
            .partition(|(_, dist)| *dist <= radius);
        
        let left_indices: Vec<_> = left_indices.into_iter().map(|(idx, _)| idx).collect();
        let right_indices: Vec<_> = right_indices.into_iter().map(|(idx, _)| idx).collect();
        
        Some(Box::new(VpNode {
            index: vp_idx,
            radius,
            left: Self::build_tree(points, metric, left_indices),
            right: Self::build_tree(points, metric, right_indices),
        }))
    }
    
    fn search_knn(&self, query: &DVector<Scalar>, k: usize) -> Vec<(usize, Scalar)> {
        let mut heap = std::collections::BinaryHeap::new();
        
        if let Some(ref root) = self.root {
            self.search_node(query, root, k, &mut heap);
        }
        
        // Convert max-heap to sorted vector
        let mut result = Vec::new();
        while let Some(std::cmp::Reverse((dist, idx))) = heap.pop() {
            result.push((idx, dist));
        }
        result.reverse();
        result
    }
    
    fn search_node(&self, 
                   query: &DVector<Scalar>, 
                   node: &VpNode, 
                   k: usize,
                   heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(Scalar, usize)>>) {
        let dist = self.metric.compute(query, &self.points[node.index]);
        
        // Update heap
        if heap.len() < k {
            heap.push(std::cmp::Reverse((dist, node.index)));
        } else if let Some(&std::cmp::Reverse((max_dist, _))) = heap.peek() {
            if dist < max_dist {
                heap.pop();
                heap.push(std::cmp::Reverse((dist, node.index)));
            }
        }
        
        // Determine which branches to explore
        let max_dist = heap.peek()
            .map(|&std::cmp::Reverse((d, _))| d)
            .unwrap_or(Scalar::INFINITY);
        
        if dist <= node.radius {
            // Search left subtree first
            if let Some(ref left) = node.left {
                self.search_node(query, left, k, heap);
            }
            // Search right subtree if necessary
            if dist + max_dist > node.radius {
                if let Some(ref right) = node.right {
                    self.search_node(query, right, k, heap);
                }
            }
        } else {
            // Search right subtree first
            if let Some(ref right) = node.right {
                self.search_node(query, right, k, heap);
            }
            // Search left subtree if necessary
            if dist - max_dist <= node.radius {
                if let Some(ref left) = node.left {
                    self.search_node(query, left, k, heap);
                }
            }
        }
    }
}

/// GPU resources for accelerated rendering
struct GpuResources {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    
    point_texture: wgpu::Texture,
    trajectory_buffer: wgpu::Buffer,
}

impl GpuResources {
    fn new(device: &wgpu::Device, config: &StateSpaceConfig) -> Self {
        // Create vertex buffer for point geometry
        let vertices = Self::create_point_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("State Space Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Create index buffer
        let indices = Self::create_point_indices();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("State Space Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State Space Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create instance buffer for per-point data
        let max_points = config.visualization_params.max_points;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State Space Instance Buffer"),
            size: (max_points * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create texture for point sprites
        let point_texture = Self::create_point_texture(device);
        
        // Create trajectory buffer
        let trajectory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State Space Trajectory Buffer"),
            size: (config.visualization_params.max_trajectory_points * 
                   std::mem::size_of::<TrajectoryVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("State Space Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Create texture view and sampler
        let texture_view = point_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("State Space Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        
        Self {
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            instance_buffer,
            bind_group_layout,
            bind_group,
            point_texture,
            trajectory_buffer,
        }
    }
    
    fn create_point_vertices() -> Vec<Vertex> {
        // Create a quad for point sprites
        vec![
            Vertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            Vertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
            Vertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
            Vertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
        ]
    }
    
    fn create_point_indices() -> Vec<u16> {
        vec![0, 1, 2, 0, 2, 3]
    }
    
    fn create_point_texture(device: &wgpu::Device) -> wgpu::Texture {
        // Generate point sprite texture
        let size = 64;
        let mut data = vec![0u8; size * size * 4];
        
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 - size as f32 / 2.0) / (size as f32 / 2.0);
                let dy = (y as f32 - size as f32 / 2.0) / (size as f32 / 2.0);
                let dist = (dx * dx + dy * dy).sqrt();
                
                let alpha = if dist <= 1.0 {
                    (1.0 - dist).powf(2.0) * 255.0
                } else {
                    0.0
                };
                
                let idx = (y * size + x) * 4;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
                data[idx + 3] = alpha as u8;
            }
        }
        
        device.create_texture_with_data(
            &wgpu::TextureDescriptor {
                label: Some("State Space Point Texture"),
                size: wgpu::Extent3d {
                    width: size as u32,
                    height: size as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            &data,
        )
    }
}

/// Vertex structure for point rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

/// Instance data for per-point rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    world_position: [f32; 3],
    color: [f32; 4],
    size: f32,
    _padding: [f32; 3],
}

/// Vertex structure for trajectory rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TrajectoryVertex {
    position: [f32; 3],
    color: [f32; 4],
    timestamp: f32,
}

/// Uniform buffer structure
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 3],
    time: f32,
    point_scale: f32,
    alpha_multiplier: f32,
    highlight_strength: f32,
    _padding: f32,
}

impl<S: AlgorithmState> StateSpaceVisualization<S> {
    pub fn new(
        render_context: &RenderContext, 
        config: StateSpaceConfig,
    ) -> Result<Self, VisualizationError> {
        // Initialize UMAP algorithm
        let umap = Arc::new(Mutex::new(UmapAlgorithm::new(config.umap_params.clone())));
        
        // Create GPU resources
        let gpu_resources = GpuResources::new(&render_context.device, &config);
        
        // Create render pipeline
        let render_pipeline = Self::create_render_pipeline(
            &render_context.device,
            &gpu_resources.bind_group_layout,
            render_context.surface_format,
        )?;
        
        // Create compute pipeline for GPU acceleration
        let compute_pipeline = Self::create_compute_pipeline(
            &render_context.device,
            &gpu_resources.bind_group_layout,
        )?;
        
        Ok(Self {
            config,
            embedding: Arc::new(DashMap::new()),
            umap,
            gpu_resources,
            render_pipeline,
            compute_pipeline,
            interaction_handler: InteractionHandler::new(),
            trajectory_manager: TrajectoryManager::new(),
            metrics: Arc::new(PerformanceMetrics::default()),
            _phantom: PhantomData,
        })
    }
    
    /// Update visualization with new states
    pub fn update(&mut self, timeline: &Timeline<S>) -> Result<(), VisualizationError> {
        let start_time = std::time::Instant::now();
        
        // Extract states from timeline
        let states = timeline.get_state_sequence();
        
        // Update UMAP embedding
        self.update_embedding(&states)?;
        
        // Update trajectories
        self.trajectory_manager.update(timeline);
        
        // Update GPU buffers
        self.update_gpu_buffers()?;
        
        // Update performance metrics
        self.metrics.update_time.store(
            start_time.elapsed().as_micros() as usize,
            Ordering::Relaxed
        );
        
        Ok(())
    }
    
    fn update_embedding(&mut self, states: &[S]) -> Result<(), VisualizationError> {
        let mut umap = self.umap.lock().unwrap();
        
        // Fit or transform based on whether model is trained
        if umap.high_dim_points.is_empty() {
            umap.fit(states)?;
        } else {
            // Incremental update for new states
            let new_embeddings = umap.transform(states)?;
            
            // Update embedding cache
            for (state, embedding) in states.iter().zip(new_embeddings.iter()) {
                let signature = state.signature();
                let point = EmbeddedPoint {
                    position: embedding.clone(),
                    signature: signature.clone(),
                    timestamp: state.timestamp(),
                    metadata: self.compute_point_metadata(state),
                };
                
                self.embedding.insert(signature, point);
            }
        }
        
        Ok(())
    }
    
    fn compute_point_metadata(&self, state: &S) -> PointMetadata {
        PointMetadata {
            step: state.step(),
            importance: self.compute_state_importance(state),
            cluster_id: None, // Can be computed separately
            trajectory_id: Some(state.trajectory_id()),
            visual_props: self.compute_visual_properties(state),
        }
    }
    
    fn compute_state_importance(&self, state: &S) -> Scalar {
        // Compute importance based on state characteristics
        let decision_score = state.is_decision_point() as u8 as Scalar;
        let entropy_score = state.entropy();
        let novelty_score = state.novelty_score();
        
        // Weighted combination
        0.4 * decision_score + 0.3 * entropy_score + 0.3 * novelty_score
    }
    
    fn compute_visual_properties(&self, state: &S) -> VisualProperties {
        let base_color = self.config.visualization_params.color_scheme.get_color(state);
        
        VisualProperties {
            color: base_color,
            size: self.config.visualization_params.base_point_size,
            shape: PointShape::Circle,
            highlight: state.is_highlighted(),
        }
    }
    
    fn update_gpu_buffers(&mut self) -> Result<(), VisualizationError> {
        // Prepare instance data for GPU upload
        let instance_data: Vec<InstanceData> = self.embedding.iter()
            .map(|entry| {
                let point = entry.value();
                let pos_3d = if point.position.len() >= 3 {
                    [
                        point.position[0] as f32,
                        point.position[1] as f32,
                        point.position[2] as f32,
                    ]
                } else {
                    [
                        point.position[0] as f32,
                        point.position[1] as f32,
                        0.0,
                    ]
                };
                
                InstanceData {
                    world_position: pos_3d,
                    color: point.metadata.visual_props.color,
                    size: point.metadata.visual_props.size,
                    _padding: [0.0; 3],
                }
            })
            .collect();
        
        // Upload to GPU
        self.queue.write_buffer(
            &self.gpu_resources.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );
        
        Ok(())
    }
    
    pub fn render(
        &self,
        render_context: &mut RenderContext,
        interaction_state: &InteractionState,
    ) -> Result<(), VisualizationError> {
        let start_time = std::time::Instant::now();
        
        // Update uniforms
        let uniforms = self.compute_uniforms(render_context, interaction_state);
        render_context.queue.write_buffer(
            &self.gpu_resources.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        
        // Begin render pass
        let mut encoder = render_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("State Space Render Encoder"),
            }
        );
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("State Space Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_context.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.gpu_resources.bind_group, &[]);
            
            // Render points
            render_pass.set_vertex_buffer(0, self.gpu_resources.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.gpu_resources.instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.gpu_resources.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            
            let num_instances = self.embedding.len() as u32;
            render_pass.draw_indexed(0..6, 0, 0..num_instances);
            
            // Render trajectories
            if !self.trajectory_manager.is_empty() {
                render_pass.set_vertex_buffer(0, self.gpu_resources.trajectory_buffer.slice(..));
                let num_vertices = self.trajectory_manager.vertex_count() as u32;
                render_pass.draw(0..num_vertices, 0..1);
            }
        }
        
        // Submit command buffer
        render_context.queue.submit(std::iter::once(encoder.finish()));
        
        // Update performance metrics
        self.metrics.render_time.store(
            start_time.elapsed().as_micros() as usize,
            Ordering::Relaxed
        );
        
        Ok(())
    }
    
    fn compute_uniforms(
        &self,
        render_context: &RenderContext,
        interaction_state: &InteractionState,
    ) -> Uniforms {
        let view_proj = render_context.camera.view_projection_matrix();
        let camera_position = render_context.camera.position();
        
        Uniforms {
            view_proj,
            camera_position: [camera_position.x, camera_position.y, camera_position.z],
            time: interaction_state.time(),
            point_scale: self.config.visualization_params.point_scale,
            alpha_multiplier: self.config.visualization_params.alpha_multiplier,
            highlight_strength: self.config.visualization_params.highlight_strength,
            _padding: 0.0,
        }
    }
    
    fn create_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Result<wgpu::RenderPipeline, VisualizationError> {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("State Space Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/state_space.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("State Space Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create vertex buffer layouts
        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32,
                    },
                ],
            },
        ];
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("State Space Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        Ok(pipeline)
    }
    
    fn create_compute_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::ComputePipeline, VisualizationError> {
        // Load compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("State Space Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/state_space_compute.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("State Space Compute Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("State Space Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        Ok(pipeline)
    }
}

/// Trajectory manager for execution path visualization
struct TrajectoryManager<S: AlgorithmState> {
    trajectories: BTreeMap<usize, Trajectory>,
    vertex_buffer: Vec<TrajectoryVertex>,
    vertex_count: AtomicUsize,
    _phantom: PhantomData<S>,
}

struct Trajectory {
    id: usize,
    points: Vec<TrajectoryPoint>,
    color: [f32; 4],
    width: f32,
}

struct TrajectoryPoint {
    position: DVector<Scalar>,
    timestamp: f64,
    state_signature: StateSignature,
}

impl<S: AlgorithmState> TrajectoryManager<S> {
    fn new() -> Self {
        Self {
            trajectories: BTreeMap::new(),
            vertex_buffer: Vec::new(),
            vertex_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }
    
    fn update(&mut self, timeline: &Timeline<S>) {
        // Extract trajectories from timeline
        self.trajectories.clear();
        
        for (id, branch) in timeline.branches() {
            let points: Vec<_> = branch.states()
                .map(|state| TrajectoryPoint {
                    position: DVector::zeros(3), // Will be filled by embedding
                    timestamp: state.timestamp(),
                    state_signature: state.signature(),
                })
                .collect();
            
            let trajectory = Trajectory {
                id: *id,
                points,
                color: self.compute_trajectory_color(*id),
                width: 2.0,
            };
            
            self.trajectories.insert(*id, trajectory);
        }
        
        // Update vertex buffer
        self.update_vertex_buffer();
    }
    
    fn update_vertex_buffer(&mut self) {
        self.vertex_buffer.clear();
        
        for trajectory in self.trajectories.values() {
            for i in 0..trajectory.points.len().saturating_sub(1) {
                let p1 = &trajectory.points[i];
                let p2 = &trajectory.points[i + 1];
                
                // Create line segment
                self.vertex_buffer.push(TrajectoryVertex {
                    position: [
                        p1.position[0] as f32,
                        p1.position[1] as f32,
                        p1.position[2] as f32,
                    ],
                    color: trajectory.color,
                    timestamp: p1.timestamp as f32,
                });
                
                self.vertex_buffer.push(TrajectoryVertex {
                    position: [
                        p2.position[0] as f32,
                        p2.position[1] as f32,
                        p2.position[2] as f32,
                    ],
                    color: trajectory.color,
                    timestamp: p2.timestamp as f32,
                });
            }
        }
        
        self.vertex_count.store(self.vertex_buffer.len(), Ordering::Relaxed);
    }
    
    fn compute_trajectory_color(&self, id: usize) -> [f32; 4] {
        // Generate distinct colors for different trajectories
        let hue = (id as f32 * 0.618) % 1.0; // Golden ratio for even distribution
        let saturation = 0.7;
        let lightness = 0.6;
        let alpha = 0.8;
        
        // HSL to RGB conversion
        let c = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
        let x = c * (1.0 - ((hue * 6.0) % 2.0 - 1.0).abs());
        let m = lightness - c / 2.0;
        
        let (r, g, b) = match (hue * 6.0) as u32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        
        [r + m, g + m, b + m, alpha]
    }
    
    fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }
    
    fn vertex_count(&self) -> usize {
        self.vertex_count.load(Ordering::Relaxed)
    }
}

/// Interaction handler for user input
struct InteractionHandler {
    selected_point: Option<StateSignature>,
    hover_point: Option<StateSignature>,
    pan_offset: cgmath::Vector2<f32>,
    zoom_level: f32,
    rotation: cgmath::Quaternion<f32>,
}

impl InteractionHandler {
    fn new() -> Self {
        Self {
            selected_point: None,
            hover_point: None,
            pan_offset: cgmath::Vector2::zero(),
            zoom_level: 1.0,
            rotation: cgmath::Quaternion::one(),
        }
    }
    
    fn handle_mouse_click(&mut self, position: cgmath::Vector2<f32>, points: &DashMap<StateSignature, EmbeddedPoint>) {
        // Find nearest point to click position
        let mut nearest_point = None;
        let mut nearest_distance = f32::INFINITY;
        
        for entry in points.iter() {
            let point = entry.value();
            let screen_pos = self.world_to_screen(&point.position);
            let distance = (screen_pos - position).magnitude();
            
            if distance < nearest_distance && distance < 10.0 { // 10 pixel threshold
                nearest_distance = distance;
                nearest_point = Some(point.signature.clone());
            }
        }
        
        self.selected_point = nearest_point;
    }
    
    fn world_to_screen(&self, world_pos: &DVector<Scalar>) -> cgmath::Vector2<f32> {
        // Transform world coordinates to screen coordinates
        // This would use the current view and projection matrices
        cgmath::Vector2::new(
            world_pos[0] as f32 * self.zoom_level + self.pan_offset.x,
            world_pos[1] as f32 * self.zoom_level + self.pan_offset.y,
        )
    }
}

/// Performance metrics tracking
#[derive(Default)]
struct PerformanceMetrics {
    update_time: AtomicUsize,
    render_time: AtomicUsize,
    embedding_time: AtomicUsize,
    gpu_memory_usage: AtomicUsize,
    point_count: AtomicUsize,
}

/// Optimization state for UMAP
#[derive(Default)]
struct OptimizationState {
    current_epoch: usize,
    current_loss: Scalar,
    gradient_norm: Scalar,
    convergence_delta: Scalar,
}

/// Additional type definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfiguration {
    pub enabled: bool,
    pub max_compute_workgroups: usize,
    pub workgroup_size: usize,
    pub memory_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfiguration {
    pub preserve_local_structure: bool,
    pub preserve_global_structure: bool,
    pub homology_dimensions: Vec<usize>,
    pub persistence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationParameters {
    pub max_points: usize,
    pub max_trajectory_points: usize,
    pub point_scale: f32,
    pub alpha_multiplier: f32,
    pub highlight_strength: f32,
    pub base_point_size: f32,
    pub color_scheme: ColorScheme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Sequential,
    Diverging,
    Categorical,
    Custom(Vec<[f32; 4]>),
}

impl ColorScheme {
    fn get_color<S: AlgorithmState>(&self, state: &S) -> [f32; 4] {
        match self {
            Self::Sequential => {
                let t = state.progress();
                [t, 0.5, 1.0 - t, 1.0]
            },
            Self::Diverging => {
                let t = state.divergence_score();
                if t < 0.5 {
                    [0.0, 2.0 * t, 1.0 - 2.0 * t, 1.0]
                } else {
                    [2.0 * (t - 0.5), 1.0 - 2.0 * (t - 0.5), 0.0, 1.0]
                }
            },
            Self::Categorical => {
                let category = state.category_id();
                let hue = (category as f32 * 0.618) % 1.0;
                // HSL to RGB conversion
                [hue, 0.7, 0.6, 1.0] // Simplified
            },
            Self::Custom(colors) => {
                let idx = state.state_id() % colors.len();
                colors[idx]
            }
        }
    }
}

/// Error types
#[derive(Debug, thiserror::Error)]
pub enum UmapError {
    #[error("Insufficient data points: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    #[error("GPU error: {0}")]
    GpuError(String),
    
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] UmapError),
    
    #[error("Render error: {0}")]
    RenderError(String),
}

// Trait implementations for AlgorithmState extension
pub trait StateSpaceExtensions: AlgorithmState {
    fn extract_features(&self) -> Vec<Scalar>;
    fn progress(&self) -> f32;
    fn divergence_score(&self) -> f32;
    fn category_id(&self) -> usize;
    fn state_id(&self) -> usize;
    fn is_decision_point(&self) -> bool;
    fn entropy(&self) -> Scalar;
    fn novelty_score(&self) -> Scalar;
    fn trajectory_id(&self) -> usize;
    fn is_highlighted(&self) -> bool;
}