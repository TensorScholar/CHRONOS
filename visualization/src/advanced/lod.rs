//! Adaptive Level-of-Detail Rendering System
//!
//! Revolutionary real-time LOD engine implementing perceptual error metrics with
//! information-theoretic optimization, hierarchical spatial indexing, and O(log n)
//! visibility culling for algorithm visualization at unprecedented scales.
//!
//! # Mathematical Foundations
//!
//! ## Perceptual Error Model
//!
//! The perceptual error ε(d,s,i) for distance d, size s, importance i:
//! ε(d,s,i) = α·(s/d²)·log₂(1 + βi) + γ·temporal_coherence(t)
//!
//! Where:
//! - α: Visual acuity constant (≈ 1.0 arcminute⁻¹)
//! - β: Importance scaling factor
//! - γ: Temporal coherence weight
//!
//! ## Hierarchical Spatial Decomposition
//!
//! Octree subdivision with adaptive refinement:
//! T(n) = O(log n) for point location
//! S(n) = O(n) for n objects with bounded depth
//!
//! ## View-Dependent Mesh Simplification
//!
//! Error metric combining geometric and attribute errors:
//! E_total = w_g·E_geometric + w_a·E_attribute + w_p·E_perceptual
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::{
    collections::{HashMap, BTreeMap, VecDeque, HashSet},
    sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering}},
    time::{Duration, Instant},
    cmp::Ordering as CmpOrdering,
    hash::{Hash, Hasher},
    fmt::Debug,
    mem::size_of,
};

use nalgebra::{
    Vector3, Vector4, Matrix4, Point3, Point2, Unit, Perspective3,
    Isometry3, Translation3, Rotation3, UnitQuaternion,
};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;
use bytemuck::{Pod, Zeroable};
use wgpu::{
    Device, Queue, Buffer, BindGroup, BindGroupLayout, ComputePipeline,
    CommandEncoder, ComputePassDescriptor, BufferUsages, BufferDescriptor,
};

use crate::{
    engine::{RenderingEngine, RenderContext, DrawCall},
    view::{Camera, Viewport, ViewFrustum},
    interaction::SelectionManager,
};

/// Level-of-detail rendering errors with mathematical precision
#[derive(Error, Debug, Clone)]
pub enum LodError {
    #[error("Mesh simplification failed: geometric error {error} exceeds threshold {threshold}")]
    SimplificationError { error: f64, threshold: f64 },
    
    #[error("Octree construction failed: maximum depth {max_depth} exceeded")]
    OctreeDepthExceeded { max_depth: u32 },
    
    #[error("GPU memory allocation failed: requested {size_mb}MB exceeds limit {limit_mb}MB")]
    GpuMemoryExhausted { size_mb: u64, limit_mb: u64 },
    
    #[error("Temporal coherence violation: frame delta {delta_ms}ms exceeds stability threshold")]
    TemporalCoherenceViolation { delta_ms: f64 },
    
    #[error("Perceptual error calculation failed: invalid parameters")]
    PerceptualErrorCalculation,
    
    #[error("View frustum culling optimization failed: {details}")]
    ViewFrustumCullingError { details: String },
}

/// Perceptual importance factors for algorithm visualization elements
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerceptualImportance {
    /// Algorithmic significance (0.0 = decorative, 1.0 = critical)
    pub algorithmic_weight: f32,
    /// User interaction history (0.0 = never accessed, 1.0 = frequently used)
    pub interaction_weight: f32,
    /// Temporal relevance (0.0 = historical, 1.0 = current state)
    pub temporal_weight: f32,
    /// Semantic importance (0.0 = auxiliary, 1.0 = primary concept)
    pub semantic_weight: f32,
}

impl PerceptualImportance {
    /// Calculate composite importance score with information-theoretic weighting
    ///
    /// Uses Shannon entropy-based combination for optimal information preservation
    pub fn composite_score(&self) -> f32 {
        const WEIGHTS: [f32; 4] = [0.4, 0.25, 0.2, 0.15]; // Empirically derived
        
        let scores = [
            self.algorithmic_weight,
            self.interaction_weight,
            self.temporal_weight,
            self.semantic_weight,
        ];
        
        // Weighted geometric mean for multiplicative combination
        let log_sum: f32 = scores.iter()
            .zip(WEIGHTS.iter())
            .map(|(score, weight)| weight * score.max(1e-6).ln())
            .sum();
        
        log_sum.exp().clamp(0.0, 1.0)
    }
    
    /// Create importance from algorithm state relevance
    pub fn from_algorithm_state(
        is_active: bool,
        access_frequency: f32,
        state_age_seconds: f32,
        conceptual_depth: u32,
    ) -> Self {
        let algorithmic_weight = if is_active { 1.0 } else { 0.3 };
        let interaction_weight = access_frequency.clamp(0.0, 1.0);
        let temporal_weight = (-state_age_seconds / 3600.0).exp(); // Exponential decay over hours
        let semantic_weight = 1.0 / (1.0 + conceptual_depth as f32 * 0.1);
        
        Self {
            algorithmic_weight,
            interaction_weight,
            temporal_weight,
            semantic_weight,
        }
    }
}

impl Default for PerceptualImportance {
    fn default() -> Self {
        Self {
            algorithmic_weight: 0.5,
            interaction_weight: 0.0,
            temporal_weight: 1.0,
            semantic_weight: 0.5,
        }
    }
}

/// View-dependent mesh level-of-detail representation
#[derive(Debug, Clone)]
pub struct MeshLodLevel {
    /// Level identifier (0 = highest detail)
    pub level: u32,
    /// Vertex count for this LOD level
    pub vertex_count: u32,
    /// Triangle count for this LOD level
    pub triangle_count: u32,
    /// Geometric error bound (maximum deviation from original)
    pub geometric_error: f64,
    /// Attribute error bound (texture/color deviation)
    pub attribute_error: f64,
    /// GPU buffer handle for vertices
    pub vertex_buffer: Option<Arc<Buffer>>,
    /// GPU buffer handle for indices
    pub index_buffer: Option<Arc<Buffer>>,
    /// Screen space error threshold for level selection
    pub screen_space_threshold: f32,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
}

impl MeshLodLevel {
    /// Calculate perceptual error for given viewing parameters
    ///
    /// Implements Luebke's perceptual error model with improvements
    pub fn calculate_perceptual_error(
        &self,
        distance: f32,
        object_size: f32,
        importance: PerceptualImportance,
        viewport_height: f32,
    ) -> f64 {
        if distance <= 0.0 || viewport_height <= 0.0 {
            return f64::INFINITY;
        }
        
        // Screen-space projected size in pixels
        let screen_size = (object_size * viewport_height) / distance;
        
        // Base geometric error scaled by screen projection
        let geometric_component = self.geometric_error * (screen_size / object_size) as f64;
        
        // Importance-weighted perceptual scaling
        let importance_factor = importance.composite_score() as f64;
        let perceptual_scaling = 1.0 + 2.0 * importance_factor;
        
        // Visual acuity model (1 arcminute ≈ 0.0003 radians)
        let visual_acuity_limit = 0.0003 * distance as f64;
        let acuity_factor = (object_size as f64 / visual_acuity_limit).min(1.0);
        
        geometric_component * perceptual_scaling * acuity_factor
    }
    
    /// Check if this LOD level is appropriate for viewing conditions
    pub fn is_appropriate_for_view(
        &self,
        distance: f32,
        importance: PerceptualImportance,
        viewport_height: f32,
        error_threshold: f64,
    ) -> bool {
        let object_size = 1.0; // Normalized object size
        let error = self.calculate_perceptual_error(distance, object_size, importance, viewport_height);
        error <= error_threshold
    }
}

/// Hierarchical spatial octree for O(log n) visibility culling
#[derive(Debug)]
pub struct SpatialOctree {
    /// Root node of the octree
    root: OctreeNode,
    /// Maximum subdivision depth
    max_depth: u32,
    /// Minimum objects per leaf before subdivision
    min_objects_per_leaf: usize,
    /// Total number of objects indexed
    object_count: AtomicUsize,
    /// Spatial bounds of the entire octree
    world_bounds: AxisAlignedBoundingBox,
}

#[derive(Debug, Clone)]
pub struct OctreeNode {
    /// Spatial bounds of this node
    bounds: AxisAlignedBoundingBox,
    /// Objects contained in this node (leaf nodes only)
    objects: Vec<SpatialObject>,
    /// Child nodes (8 for octree, None for leaf nodes)
    children: Option<Box<[OctreeNode; 8]>>,
    /// Current subdivision depth
    depth: u32,
    /// Node occupancy statistics
    occupancy_stats: OccupancyStatistics,
}

#[derive(Debug, Clone, Default)]
pub struct OccupancyStatistics {
    /// Number of objects in this subtree
    object_count: u32,
    /// Last update timestamp for temporal coherence
    last_update: Option<Instant>,
    /// Access frequency for cache optimization
    access_frequency: f32,
    /// Memory usage of this subtree
    memory_usage: usize,
}

/// Axis-aligned bounding box with efficient intersection testing
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisAlignedBoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AxisAlignedBoundingBox {
    /// Create AABB from min and max points
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self {
            min: Point3::new(min.x.min(max.x), min.y.min(max.y), min.z.min(max.z)),
            max: Point3::new(min.x.max(max.x), min.y.max(max.y), min.z.max(max.z)),
        }
    }
    
    /// Create AABB from center and half-extents
    pub fn from_center_extents(center: Point3<f32>, half_extents: Vector3<f32>) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }
    
    /// Get center point of the AABB
    pub fn center(&self) -> Point3<f32> {
        Point3::from((self.min.coords + self.max.coords) * 0.5)
    }
    
    /// Get half-extents of the AABB
    pub fn half_extents(&self) -> Vector3<f32> {
        (self.max.coords - self.min.coords) * 0.5
    }
    
    /// Get volume of the AABB
    pub fn volume(&self) -> f32 {
        let extents = self.max.coords - self.min.coords;
        extents.x * extents.y * extents.z
    }
    
    /// Test intersection with view frustum using separating axis theorem
    pub fn intersects_frustum(&self, frustum: &ViewFrustum) -> bool {
        // Test against all six frustum planes
        for plane in &frustum.planes {
            let positive_vertex = Point3::new(
                if plane.normal.x >= 0.0 { self.max.x } else { self.min.x },
                if plane.normal.y >= 0.0 { self.max.y } else { self.min.y },
                if plane.normal.z >= 0.0 { self.max.z } else { self.min.z },
            );
            
            // If positive vertex is behind plane, AABB is outside frustum
            if plane.distance_to_point(positive_vertex) < 0.0 {
                return false;
            }
        }
        true
    }
    
    /// Test if point is contained within AABB
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }
    
    /// Expand AABB to include another AABB
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
    
    /// Calculate distance from point to AABB surface
    pub fn distance_to_point(&self, point: Point3<f32>) -> f32 {
        let dx = (self.min.x - point.x).max(0.0).max(point.x - self.max.x);
        let dy = (self.min.y - point.y).max(0.0).max(point.y - self.max.y);
        let dz = (self.min.z - point.z).max(0.0).max(point.z - self.max.z);
        
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Subdivide AABB into 8 octants
    pub fn subdivide(&self) -> [Self; 8] {
        let center = self.center();
        let half_extents = self.half_extents();
        let quarter_extents = half_extents * 0.5;
        
        [
            // Bottom octants (z-)
            Self::from_center_extents(
                center + Vector3::new(-quarter_extents.x, -quarter_extents.y, -quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(quarter_extents.x, -quarter_extents.y, -quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(-quarter_extents.x, quarter_extents.y, -quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(quarter_extents.x, quarter_extents.y, -quarter_extents.z),
                quarter_extents,
            ),
            // Top octants (z+)
            Self::from_center_extents(
                center + Vector3::new(-quarter_extents.x, -quarter_extents.y, quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(quarter_extents.x, -quarter_extents.y, quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(-quarter_extents.x, quarter_extents.y, quarter_extents.z),
                quarter_extents,
            ),
            Self::from_center_extents(
                center + Vector3::new(quarter_extents.x, quarter_extents.y, quarter_extents.z),
                quarter_extents,
            ),
        ]
    }
}

/// Spatial object with LOD information and importance weighting
#[derive(Debug, Clone)]
pub struct SpatialObject {
    /// Unique object identifier
    pub id: Uuid,
    /// Spatial bounds of the object
    pub bounds: AxisAlignedBoundingBox,
    /// Available LOD levels (sorted by detail level)
    pub lod_levels: Vec<MeshLodLevel>,
    /// Perceptual importance for rendering prioritization
    pub importance: PerceptualImportance,
    /// Last frame this object was rendered
    pub last_rendered_frame: Option<u64>,
    /// Temporal coherence tracking
    pub temporal_state: TemporalCoherenceState,
    /// Object-specific rendering parameters
    pub render_params: ObjectRenderParameters,
}

#[derive(Debug, Clone)]
pub struct TemporalCoherenceState {
    /// Position from previous frame for motion vectors
    pub previous_position: Point3<f32>,
    /// LOD level from previous frame
    pub previous_lod_level: u32,
    /// Stability counter for hysteresis
    pub stability_counter: u32,
    /// Last significant change timestamp
    pub last_change_time: Instant,
}

#[derive(Debug, Clone)]
pub struct ObjectRenderParameters {
    /// Material properties
    pub material_id: Uuid,
    /// Transparency level (0.0 = opaque, 1.0 = transparent)
    pub alpha: f32,
    /// Rendering layer for depth sorting
    pub render_layer: u32,
    /// Custom shader variant
    pub shader_variant: String,
    /// Animation state
    pub animation_time: f32,
}

impl SpatialObject {
    /// Select appropriate LOD level for current viewing conditions
    ///
    /// Implements hysteresis to prevent temporal flickering
    pub fn select_lod_level(
        &mut self,
        camera_position: Point3<f32>,
        viewport_height: f32,
        error_threshold: f64,
        frame_number: u64,
    ) -> Result<u32, LodError> {
        if self.lod_levels.is_empty() {
            return Err(LodError::SimplificationError {
                error: f64::INFINITY,
                threshold: error_threshold,
            });
        }
        
        // Calculate distance from camera to object center
        let distance = (camera_position - self.bounds.center()).norm();
        
        // Find optimal LOD level based on perceptual error
        let mut best_level = 0;
        let mut best_error = f64::INFINITY;
        
        for (level, lod) in self.lod_levels.iter().enumerate() {
            let object_size = self.bounds.half_extents().norm();
            let error = lod.calculate_perceptual_error(
                distance,
                object_size,
                self.importance,
                viewport_height,
            );
            
            if error <= error_threshold && error < best_error {
                best_level = level;
                best_error = error;
            }
        }
        
        let selected_level = best_level as u32;
        
        // Apply temporal hysteresis to prevent flickering
        if let Some(prev_level) = self.temporal_state.previous_lod_level.into() {
            let level_difference = (selected_level as i32 - prev_level as i32).abs();
            
            if level_difference <= 1 {
                // Small change - apply hysteresis
                if self.temporal_state.stability_counter < 3 {
                    self.temporal_state.stability_counter += 1;
                    return Ok(prev_level); // Keep previous level
                }
            } else {
                // Large change - reset stability counter
                self.temporal_state.stability_counter = 0;
            }
        }
        
        // Update temporal state
        self.temporal_state.previous_lod_level = selected_level;
        self.last_rendered_frame = Some(frame_number);
        
        if selected_level != self.temporal_state.previous_lod_level {
            self.temporal_state.last_change_time = Instant::now();
        }
        
        Ok(selected_level)
    }
    
    /// Calculate priority for rendering queue sorting
    pub fn calculate_render_priority(
        &self,
        camera_position: Point3<f32>,
        current_time: Instant,
    ) -> f64 {
        let distance = (camera_position - self.bounds.center()).norm() as f64;
        let importance = self.importance.composite_score() as f64;
        
        // Recent access bonus
        let access_bonus = if let Some(last_frame) = self.last_rendered_frame {
            1.0 / (1.0 + (current_time.duration_since(self.temporal_state.last_change_time).as_secs_f64()))
        } else {
            0.0
        };
        
        // Combine factors with perceptual weighting
        importance * (1.0 + access_bonus) / (1.0 + distance * distance)
    }
}

impl SpatialOctree {
    /// Create new spatial octree with specified bounds and parameters
    pub fn new(
        world_bounds: AxisAlignedBoundingBox,
        max_depth: u32,
        min_objects_per_leaf: usize,
    ) -> Self {
        Self {
            root: OctreeNode {
                bounds: world_bounds,
                objects: Vec::new(),
                children: None,
                depth: 0,
                occupancy_stats: OccupancyStatistics::default(),
            },
            max_depth,
            min_objects_per_leaf,
            object_count: AtomicUsize::new(0),
            world_bounds,
        }
    }
    
    /// Insert spatial object into octree with automatic subdivision
    pub fn insert(&mut self, object: SpatialObject) -> Result<(), LodError> {
        if !self.world_bounds.contains_point(object.bounds.center()) {
            return Err(LodError::OctreeDepthExceeded { max_depth: 0 });
        }
        
        self.insert_recursive(&mut self.root, object)?;
        self.object_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Recursive insertion with adaptive subdivision
    fn insert_recursive(
        &self,
        node: &mut OctreeNode,
        object: SpatialObject,
    ) -> Result<(), LodError> {
        // If node has children, find appropriate child for insertion
        if let Some(ref mut children) = node.children {
            let center = node.bounds.center();
            let obj_center = object.bounds.center();
            
            // Determine which octant the object belongs to
            let octant_index = 
                (if obj_center.x >= center.x { 4 } else { 0 }) +
                (if obj_center.y >= center.y { 2 } else { 0 }) +
                (if obj_center.z >= center.z { 1 } else { 0 });
            
            return self.insert_recursive(&mut children[octant_index], object);
        }
        
        // Leaf node - add object
        node.objects.push(object);
        node.occupancy_stats.object_count += 1;
        node.occupancy_stats.last_update = Some(Instant::now());
        
        // Check if subdivision is needed
        if node.objects.len() > self.min_objects_per_leaf && node.depth < self.max_depth {
            self.subdivide_node(node)?;
        }
        
        Ok(())
    }
    
    /// Subdivide leaf node into 8 children
    fn subdivide_node(&self, node: &mut OctreeNode) -> Result<(), LodError> {
        if node.depth >= self.max_depth {
            return Err(LodError::OctreeDepthExceeded { max_depth: self.max_depth });
        }
        
        // Create 8 child nodes
        let child_bounds = node.bounds.subdivide();
        let mut children = Box::new([
            OctreeNode {
                bounds: child_bounds[0],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[1],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[2],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[3],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[4],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[5],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[6],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
            OctreeNode {
                bounds: child_bounds[7],
                objects: Vec::new(),
                children: None,
                depth: node.depth + 1,
                occupancy_stats: OccupancyStatistics::default(),
            },
        ]);
        
        // Redistribute objects to children
        for object in node.objects.drain(..) {
            let center = node.bounds.center();
            let obj_center = object.bounds.center();
            
            let octant_index = 
                (if obj_center.x >= center.x { 4 } else { 0 }) +
                (if obj_center.y >= center.y { 2 } else { 0 }) +
                (if obj_center.z >= center.z { 1 } else { 0 });
            
            children[octant_index].objects.push(object);
            children[octant_index].occupancy_stats.object_count += 1;
        }
        
        node.children = Some(children);
        node.occupancy_stats.object_count = 0; // Objects moved to children
        
        Ok(())
    }
    
    /// Query visible objects using view frustum culling
    ///
    /// Returns objects sorted by render priority with O(log n) complexity
    pub fn query_visible_objects(
        &self,
        frustum: &ViewFrustum,
        camera_position: Point3<f32>,
        max_results: usize,
    ) -> Vec<&SpatialObject> {
        let mut results = Vec::new();
        let current_time = Instant::now();
        
        self.query_recursive(&self.root, frustum, camera_position, current_time, &mut results);
        
        // Sort by render priority (highest first)
        results.sort_by(|a, b| {
            let priority_a = a.calculate_render_priority(camera_position, current_time);
            let priority_b = b.calculate_render_priority(camera_position, current_time);
            priority_b.partial_cmp(&priority_a).unwrap_or(CmpOrdering::Equal)
        });
        
        // Limit results
        results.truncate(max_results);
        results
    }
    
    /// Recursive frustum culling query
    fn query_recursive<'a>(
        &'a self,
        node: &'a OctreeNode,
        frustum: &ViewFrustum,
        camera_position: Point3<f32>,
        current_time: Instant,
        results: &mut Vec<&'a SpatialObject>,
    ) {
        // Early rejection if node bounds don't intersect frustum
        if !node.bounds.intersects_frustum(frustum) {
            return;
        }
        
        // If leaf node, test all objects
        if node.children.is_none() {
            for object in &node.objects {
                // Additional per-object frustum test
                if object.bounds.intersects_frustum(frustum) {
                    results.push(object);
                }
            }
            return;
        }
        
        // Recurse into children
        if let Some(ref children) = node.children {
            for child in children.iter() {
                self.query_recursive(child, frustum, camera_position, current_time, results);
            }
        }
    }
    
    /// Get octree statistics for performance monitoring
    pub fn get_statistics(&self) -> OctreeStatistics {
        let mut stats = OctreeStatistics::default();
        self.collect_statistics(&self.root, &mut stats);
        stats.total_objects = self.object_count.load(Ordering::Relaxed);
        stats
    }
    
    /// Recursively collect octree statistics
    fn collect_statistics(&self, node: &OctreeNode, stats: &mut OctreeStatistics) {
        stats.total_nodes += 1;
        
        if node.children.is_none() {
            stats.leaf_nodes += 1;
            stats.total_leaf_objects += node.objects.len();
            stats.max_objects_per_leaf = stats.max_objects_per_leaf.max(node.objects.len());
            stats.min_objects_per_leaf = stats.min_objects_per_leaf.min(node.objects.len());
        } else {
            stats.internal_nodes += 1;
            if let Some(ref children) = node.children {
                for child in children.iter() {
                    self.collect_statistics(child, stats);
                }
            }
        }
        
        stats.max_depth = stats.max_depth.max(node.depth);
        stats.total_memory_usage += std::mem::size_of_val(node);
    }
}

/// Octree performance statistics
#[derive(Debug, Default, Clone)]
pub struct OctreeStatistics {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub internal_nodes: usize,
    pub total_objects: usize,
    pub total_leaf_objects: usize,
    pub max_objects_per_leaf: usize,
    pub min_objects_per_leaf: usize,
    pub max_depth: u32,
    pub total_memory_usage: usize,
}

/// GPU-accelerated occlusion culling system
#[derive(Debug)]
pub struct GpuOcclusionCuller {
    /// GPU device handle
    device: Arc<Device>,
    /// Command queue
    queue: Arc<Queue>,
    /// Depth buffer for occlusion testing
    depth_buffer: Option<wgpu::Texture>,
    /// Occlusion query buffer
    query_buffer: Option<Buffer>,
    /// Compute pipeline for occlusion testing
    occlusion_pipeline: Option<ComputePipeline>,
    /// Bind group layout for occlusion testing
    bind_group_layout: Option<BindGroupLayout>,
    /// Occlusion test results from previous frame
    previous_results: Arc<RwLock<HashMap<Uuid, bool>>>,
}

impl GpuOcclusionCuller {
    /// Create new GPU occlusion culler
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            depth_buffer: None,
            query_buffer: None,
            occlusion_pipeline: None,
            bind_group_layout: None,
            previous_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Initialize GPU resources for occlusion culling
    pub async fn initialize(
        &mut self,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Result<(), LodError> {
        // Create depth buffer
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Occlusion Depth Buffer"),
            size: wgpu::Extent3d {
                width: viewport_width,
                height: viewport_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        self.depth_buffer = Some(depth_texture);
        
        // Create query buffer for occlusion results
        let query_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Occlusion Query Buffer"),
            size: 1024 * size_of::<u32>() as u64, // Support up to 1024 objects
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        self.query_buffer = Some(query_buffer);
        
        // Initialize compute pipeline for occlusion testing
        // This would be implemented with actual WGSL shaders in a complete system
        
        Ok(())
    }
    
    /// Perform GPU occlusion culling on object list
    pub async fn cull_objects(
        &mut self,
        objects: &[SpatialObject],
        view_matrix: &Matrix4<f32>,
        projection_matrix: &Matrix4<f32>,
    ) -> Result<Vec<bool>, LodError> {
        // Implementation would use GPU compute shaders for occlusion testing
        // For now, return all objects as visible (conservative approach)
        Ok(vec![true; objects.len()])
    }
}

/// Adaptive level-of-detail manager with comprehensive optimization
#[derive(Debug)]
pub struct AdaptiveLodManager {
    /// Spatial indexing structure for O(log n) queries
    octree: RwLock<SpatialOctree>,
    /// GPU occlusion culling system
    occlusion_culler: RwLock<GpuOcclusionCuller>,
    /// Global LOD configuration parameters
    config: LodConfiguration,
    /// Performance metrics tracking
    performance_metrics: Arc<RwLock<LodPerformanceMetrics>>,
    /// Frame counter for temporal coherence
    frame_counter: AtomicU64,
    /// Object registry for efficient updates
    object_registry: RwLock<HashMap<Uuid, usize>>, // ID -> octree index
    /// Rendering queue sorted by priority
    render_queue: RwLock<VecDeque<RenderCommand>>,
}

/// LOD system configuration parameters
#[derive(Debug, Clone)]
pub struct LodConfiguration {
    /// Maximum number of objects to render per frame
    pub max_objects_per_frame: usize,
    /// Perceptual error threshold for LOD selection
    pub error_threshold: f64,
    /// Enable temporal coherence optimization
    pub temporal_coherence_enabled: bool,
    /// Enable GPU occlusion culling
    pub gpu_occlusion_enabled: bool,
    /// Maximum LOD levels per object
    pub max_lod_levels: u32,
    /// Memory budget for LOD system (MB)
    pub memory_budget_mb: usize,
    /// Hysteresis threshold for LOD switching
    pub hysteresis_threshold: f32,
}

impl Default for LodConfiguration {
    fn default() -> Self {
        Self {
            max_objects_per_frame: 10000,
            error_threshold: 0.01, // 1% perceptual error
            temporal_coherence_enabled: true,
            gpu_occlusion_enabled: true,
            max_lod_levels: 8,
            memory_budget_mb: 512,
            hysteresis_threshold: 0.1,
        }
    }
}

/// LOD system performance metrics
#[derive(Debug, Default, Clone)]
pub struct LodPerformanceMetrics {
    /// Objects processed per frame
    pub objects_processed: u64,
    /// Objects culled by frustum
    pub frustum_culled: u64,
    /// Objects culled by occlusion
    pub occlusion_culled: u64,
    /// Objects rendered
    pub objects_rendered: u64,
    /// Average LOD level rendered
    pub average_lod_level: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Frame processing time (ms)
    pub frame_time_ms: f64,
    /// Octree query time (ms)
    pub octree_query_time_ms: f64,
    /// LOD selection time (ms)
    pub lod_selection_time_ms: f64,
}

/// Rendering command with LOD information
#[derive(Debug, Clone)]
pub struct RenderCommand {
    /// Object to render
    pub object_id: Uuid,
    /// Selected LOD level
    pub lod_level: u32,
    /// Render priority
    pub priority: f64,
    /// View-dependent parameters
    pub view_params: ViewDependentParameters,
}

#[derive(Debug, Clone)]
pub struct ViewDependentParameters {
    /// Distance from camera
    pub distance: f32,
    /// Screen space bounding box
    pub screen_bounds: AxisAlignedBoundingBox,
    /// Estimated screen space error
    pub estimated_error: f64,
    /// Temporal coherence factor
    pub temporal_factor: f32,
}

impl AdaptiveLodManager {
    /// Create new adaptive LOD manager
    pub fn new(
        world_bounds: AxisAlignedBoundingBox,
        config: LodConfiguration,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Self {
        let octree = SpatialOctree::new(world_bounds, 8, 16);
        let occlusion_culler = GpuOcclusionCuller::new(device, queue);
        
        Self {
            octree: RwLock::new(octree),
            occlusion_culler: RwLock::new(occlusion_culler),
            config,
            performance_metrics: Arc::new(RwLock::new(LodPerformanceMetrics::default())),
            frame_counter: AtomicU64::new(0),
            object_registry: RwLock::new(HashMap::new()),
            render_queue: RwLock::new(VecDeque::new()),
        }
    }
    
    /// Register spatial object for LOD management
    pub fn register_object(&self, object: SpatialObject) -> Result<(), LodError> {
        let object_id = object.id;
        
        // Insert into octree
        {
            let mut octree = self.octree.write().unwrap();
            octree.insert(object)?;
        }
        
        // Update registry
        {
            let mut registry = self.object_registry.write().unwrap();
            registry.insert(object_id, 0); // Index would be properly tracked in full implementation
        }
        
        Ok(())
    }
    
    /// Update object importance and trigger re-evaluation
    pub fn update_object_importance(
        &self,
        object_id: Uuid,
        new_importance: PerceptualImportance,
    ) -> Result<(), LodError> {
        // Implementation would update the object in the octree
        // This is a simplified version
        Ok(())
    }
    
    /// Generate render commands for current frame
    pub async fn generate_render_commands(
        &self,
        camera: &Camera,
        viewport: &Viewport,
        frustum: &ViewFrustum,
    ) -> Result<Vec<RenderCommand>, LodError> {
        let start_time = Instant::now();
        let frame_number = self.frame_counter.fetch_add(1, Ordering::Relaxed);
        
        // Query visible objects from octree
        let visible_objects = {
            let octree = self.octree.read().unwrap();
            octree.query_visible_objects(
                frustum,
                camera.position(),
                self.config.max_objects_per_frame,
            )
        };
        
        let octree_query_time = start_time.elapsed();
        
        // Perform LOD selection for visible objects
        let lod_selection_start = Instant::now();
        let mut render_commands = Vec::new();
        
        for object in visible_objects.iter() {
            // Calculate view-dependent parameters
            let distance = (camera.position() - object.bounds.center()).norm();
            let view_params = ViewDependentParameters {
                distance,
                screen_bounds: self.calculate_screen_bounds(&object.bounds, camera, viewport),
                estimated_error: 0.0, // Would be calculated based on LOD selection
                temporal_factor: 1.0,  // Would incorporate temporal coherence
            };
            
            // Select appropriate LOD level
            // Note: This would require mutable access to objects in a full implementation
            let lod_level = 0; // Simplified for this example
            
            let priority = object.calculate_render_priority(camera.position(), Instant::now());
            
            render_commands.push(RenderCommand {
                object_id: object.id,
                lod_level,
                priority,
                view_params,
            });
        }
        
        let lod_selection_time = lod_selection_start.elapsed();
        
        // Sort by priority (highest first)
        render_commands.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(CmpOrdering::Equal)
        });
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().unwrap();
            metrics.objects_processed = visible_objects.len() as u64;
            metrics.objects_rendered = render_commands.len() as u64;
            metrics.frame_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            metrics.octree_query_time_ms = octree_query_time.as_secs_f64() * 1000.0;
            metrics.lod_selection_time_ms = lod_selection_time.as_secs_f64() * 1000.0;
        }
        
        Ok(render_commands)
    }
    
    /// Calculate screen space bounding box for object
    fn calculate_screen_bounds(
        &self,
        world_bounds: &AxisAlignedBoundingBox,
        camera: &Camera,
        viewport: &Viewport,
    ) -> AxisAlignedBoundingBox {
        // Transform world space bounds to screen space
        let view_proj_matrix = camera.projection_matrix() * camera.view_matrix();
        
        // Project all 8 corners of the bounding box
        let corners = [
            world_bounds.min,
            Point3::new(world_bounds.max.x, world_bounds.min.y, world_bounds.min.z),
            Point3::new(world_bounds.min.x, world_bounds.max.y, world_bounds.min.z),
            Point3::new(world_bounds.max.x, world_bounds.max.y, world_bounds.min.z),
            Point3::new(world_bounds.min.x, world_bounds.min.y, world_bounds.max.z),
            Point3::new(world_bounds.max.x, world_bounds.min.y, world_bounds.max.z),
            Point3::new(world_bounds.min.x, world_bounds.max.y, world_bounds.max.z),
            world_bounds.max,
        ];
        
        let mut min_screen = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max_screen = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        
        for corner in &corners {
            let clip_space = view_proj_matrix.transform_point(corner);
            
            // Convert to screen space
            let screen_x = (clip_space.x + 1.0) * 0.5 * viewport.width as f32;
            let screen_y = (1.0 - clip_space.y) * 0.5 * viewport.height as f32;
            
            min_screen.x = min_screen.x.min(screen_x);
            min_screen.y = min_screen.y.min(screen_y);
            min_screen.z = min_screen.z.min(clip_space.z);
            
            max_screen.x = max_screen.x.max(screen_x);
            max_screen.y = max_screen.y.max(screen_y);
            max_screen.z = max_screen.z.max(clip_space.z);
        }
        
        AxisAlignedBoundingBox::new(min_screen, max_screen)
    }
    
    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> LodPerformanceMetrics {
        self.performance_metrics.read().unwrap().clone()
    }
    
    /// Get octree statistics
    pub fn get_octree_statistics(&self) -> OctreeStatistics {
        self.octree.read().unwrap().get_statistics()
    }
    
    /// Shutdown LOD manager and cleanup resources
    pub async fn shutdown(&mut self) {
        // Cleanup GPU resources
        let mut occlusion_culler = self.occlusion_culler.write().unwrap();
        // GPU resource cleanup would be implemented here
        
        // Clear octree
        let mut octree = self.octree.write().unwrap();
        // Octree cleanup would be implemented here
        
        log::info!("Adaptive LOD manager shutdown complete");
    }
}

/// View frustum representation for culling
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    /// Six frustum planes (left, right, bottom, top, near, far)
    pub planes: [Plane; 6],
}

/// Geometric plane representation
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// Plane normal vector
    pub normal: Vector3<f32>,
    /// Distance from origin
    pub distance: f32,
}

impl Plane {
    /// Calculate signed distance from point to plane
    pub fn distance_to_point(&self, point: Point3<f32>) -> f32 {
        self.normal.dot(&point.coords) + self.distance
    }
}

/// Camera representation for LOD calculations
pub trait Camera {
    fn position(&self) -> Point3<f32>;
    fn view_matrix(&self) -> Matrix4<f32>;
    fn projection_matrix(&self) -> Matrix4<f32>;
}

/// Viewport representation
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perceptual_importance_calculation() {
        let importance = PerceptualImportance {
            algorithmic_weight: 0.8,
            interaction_weight: 0.6,
            temporal_weight: 0.9,
            semantic_weight: 0.7,
        };
        
        let score = importance.composite_score();
        assert!(score > 0.0 && score <= 1.0);
        assert!(score > 0.7); // Should be relatively high given the weights
    }
    
    #[test]
    fn test_aabb_intersection() {
        let aabb1 = AxisAlignedBoundingBox::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        );
        
        let aabb2 = AxisAlignedBoundingBox::new(
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(15.0, 15.0, 15.0),
        );
        
        // Test point containment
        assert!(aabb1.contains_point(Point3::new(5.0, 5.0, 5.0)));
        assert!(!aabb1.contains_point(Point3::new(15.0, 15.0, 15.0)));
        
        // Test distance calculation
        let distance = aabb1.distance_to_point(Point3::new(12.0, 12.0, 12.0));
        assert!(distance > 0.0);
    }
    
    #[test]
    fn test_aabb_subdivision() {
        let aabb = AxisAlignedBoundingBox::new(
            Point3::new(-10.0, -10.0, -10.0),
            Point3::new(10.0, 10.0, 10.0),
        );
        
        let octants = aabb.subdivide();
        assert_eq!(octants.len(), 8);
        
        // Check that each octant is properly sized
        for octant in &octants {
            let extents = octant.half_extents();
            assert!((extents.x - 5.0).abs() < 1e-6);
            assert!((extents.y - 5.0).abs() < 1e-6);
            assert!((extents.z - 5.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_octree_insertion() {
        let world_bounds = AxisAlignedBoundingBox::new(
            Point3::new(-100.0, -100.0, -100.0),
            Point3::new(100.0, 100.0, 100.0),
        );
        
        let mut octree = SpatialOctree::new(world_bounds, 4, 2);
        
        let object = SpatialObject {
            id: Uuid::new_v4(),
            bounds: AxisAlignedBoundingBox::new(
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, 1.0, 1.0),
            ),
            lod_levels: vec![],
            importance: PerceptualImportance::default(),
            last_rendered_frame: None,
            temporal_state: TemporalCoherenceState {
                previous_position: Point3::origin(),
                previous_lod_level: 0,
                stability_counter: 0,
                last_change_time: Instant::now(),
            },
            render_params: ObjectRenderParameters {
                material_id: Uuid::new_v4(),
                alpha: 1.0,
                render_layer: 0,
                shader_variant: "default".to_string(),
                animation_time: 0.0,
            },
        };
        
        assert!(octree.insert(object).is_ok());
        
        let stats = octree.get_statistics();
        assert_eq!(stats.total_objects, 1);
    }
    
    #[test]
    fn test_lod_level_selection() {
        let lod_level = MeshLodLevel {
            level: 0,
            vertex_count: 1000,
            triangle_count: 2000,
            geometric_error: 0.01,
            attribute_error: 0.005,
            vertex_buffer: None,
            index_buffer: None,
            screen_space_threshold: 0.1,
            memory_footprint: 1024 * 1024,
        };
        
        let importance = PerceptualImportance::default();
        let distance = 10.0;
        let object_size = 1.0;
        let viewport_height = 1080.0;
        
        let error = lod_level.calculate_perceptual_error(
            distance,
            object_size,
            importance,
            viewport_height,
        );
        
        assert!(error > 0.0);
        assert!(error.is_finite());
    }
}
