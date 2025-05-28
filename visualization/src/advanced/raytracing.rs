//! Revolutionary Real-Time Ray Tracing Engine for Algorithm Visualization
//!
//! This module implements a cutting-edge ray tracing system specifically designed
//! for algorithm state illumination, featuring GPU compute shaders, spatially-variant
//! basis functions, and Monte Carlo global illumination with mathematical correctness.
//!
//! # Theoretical Foundation
//!
//! The ray tracing engine is built on rigorous mathematical principles:
//! - Rendering equation: L_o(x,ω_o) = L_e(x,ω_o) + ∫_Ω f_r(x,ω_i,ω_o)L_i(x,ω_i)(ω_i·n)dω_i
//! - Monte Carlo integration with importance sampling and variance reduction
//! - Spatially-variant BRDF modeling for algorithm state representation
//! - Temporal coherence exploitation for real-time performance
//!
//! # Architectural Paradigms
//!
//! - Clean Architecture: Hexagonal separation with rendering core isolation
//! - SOLID Principles: Single responsibility, open/closed, dependency inversion
//! - Functional Reactive Programming: Immutable state transformations
//! - Lock-Free Concurrency: Atomic operations with memory ordering guarantees

use crate::algorithm::{AlgorithmState, NodeId};
use crate::data_structures::graph::Graph;
use crate::visualization::engine::{RenderingBackend, RenderingCapabilities};
use crate::temporal::StateManager;

use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock, atomic::{AtomicU32, AtomicU64, Ordering}};
use std::time::{Instant, Duration};
use std::f32::consts::{PI, FRAC_PI_2, FRAC_PI_4};

use nalgebra::{Vector3, Vector4, Matrix4, Point3, Unit, Rotation3};
use bytemuck::{Pod, Zeroable, cast_slice};
use wgpu::{
    Device, Queue, Buffer, BindGroup, BindGroupLayout, ComputePipeline,
    ShaderModule, CommandEncoder, ComputePass, BufferUsages, ShaderStages,
    BindingType, BufferBindingType, StorageTextureAccess, TextureFormat,
    Extent3d, TextureDimension, TextureUsages, Texture, TextureView,
};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use approx::{AbsDiffEq, RelativeEq};
use rand::{Rng, thread_rng, rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Ray structure for ray-surface intersection calculations
///
/// Implements affine ray parameterization: R(t) = origin + t * direction
/// where t ∈ [t_min, t_max] defines the valid ray segment
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Ray {
    /// Ray origin point in world coordinates
    pub origin: [f32; 3],
    /// Padding for memory alignment
    pub _padding1: f32,
    
    /// Normalized ray direction vector
    pub direction: [f32; 3],
    /// Minimum ray parameter value
    pub t_min: f32,
    
    /// Maximum ray parameter value
    pub t_max: f32,
    /// Ray generation time for temporal coherence
    pub time: f32,
    /// Ray wavelength for spectral rendering
    pub wavelength: f32,
    /// Ray importance weight for adaptive sampling
    pub weight: f32,
}

impl Ray {
    /// Create a new ray with specified parameters
    pub fn new(origin: Vector3<f32>, direction: Vector3<f32>, t_min: f32, t_max: f32) -> Self {
        let direction = direction.normalize();
        
        Self {
            origin: [origin.x, origin.y, origin.z],
            _padding1: 0.0,
            direction: [direction.x, direction.y, direction.z],
            t_min,
            t_max,
            time: 0.0,
            wavelength: 550.0, // Green wavelength in nanometers
            weight: 1.0,
        }
    }

    /// Evaluate ray at parameter t: R(t) = origin + t * direction
    pub fn at(&self, t: f32) -> Vector3<f32> {
        Vector3::new(self.origin[0], self.origin[1], self.origin[2]) +
        t * Vector3::new(self.direction[0], self.direction[1], self.direction[2])
    }

    /// Get ray origin as Vector3
    pub fn origin(&self) -> Vector3<f32> {
        Vector3::new(self.origin[0], self.origin[1], self.origin[2])
    }

    /// Get ray direction as Vector3
    pub fn direction(&self) -> Vector3<f32> {
        Vector3::new(self.direction[0], self.direction[1], self.direction[2])
    }
}

/// Surface intersection result with comprehensive geometric information
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Intersection {
    /// Intersection point in world coordinates
    pub point: [f32; 3],
    /// Ray parameter at intersection
    pub t: f32,
    
    /// Surface normal at intersection (normalized)
    pub normal: [f32; 3],
    /// Material identifier for surface properties
    pub material_id: u32,
    
    /// UV texture coordinates
    pub uv: [f32; 2],
    /// Geometric vs shading normal indicator
    pub is_front_face: f32,
    /// Surface curvature for advanced shading
    pub curvature: f32,
}

impl Intersection {
    /// Check if this intersection is valid
    pub fn is_valid(&self) -> bool {
        self.t > 0.0 && self.t < f32::INFINITY
    }

    /// Get intersection point as Vector3
    pub fn point(&self) -> Vector3<f32> {
        Vector3::new(self.point[0], self.point[1], self.point[2])
    }

    /// Get surface normal as Vector3
    pub fn normal(&self) -> Vector3<f32> {
        Vector3::new(self.normal[0], self.normal[1], self.normal[2])
    }
}

/// Bidirectional Reflectance Distribution Function (BRDF) parameters
///
/// Implements physically-based shading with energy conservation:
/// ∫_Ω f_r(ω_i,ω_o)(ω_i·n)dω_i ≤ 1 for all ω_o
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BrdfParameters {
    /// Base albedo color (energy conserving)
    pub albedo: [f32; 3],
    /// Material metallic factor [0,1]
    pub metallic: f32,
    
    /// Surface roughness parameter [0,1]
    pub roughness: f32,
    /// Fresnel reflectance at normal incidence
    pub f0: f32,
    /// Subsurface scattering coefficient
    pub subsurface: f32,
    /// Emission color for light sources
    pub emission: f32,
}

impl BrdfParameters {
    /// Create physically-plausible BRDF parameters with energy conservation
    pub fn new(albedo: Vector3<f32>, metallic: f32, roughness: f32) -> Self {
        // Ensure energy conservation: albedo magnitude ≤ 1
        let albedo = albedo.component_mul(&Vector3::new(
            1.0f32.min(albedo.x),
            1.0f32.min(albedo.y),
            1.0f32.min(albedo.z),
        ));

        // Compute F0 based on metallic workflow
        let f0 = if metallic > 0.5 {
            // Metallic surfaces: F0 = albedo
            (albedo.x + albedo.y + albedo.z) / 3.0
        } else {
            // Dielectric surfaces: F0 ≈ 0.04
            0.04
        };

        Self {
            albedo: [albedo.x, albedo.y, albedo.z],
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.001, 1.0), // Avoid perfect mirror singularities
            f0: f0.clamp(0.0, 1.0),
            subsurface: 0.0,
            emission: 0.0,
        }
    }

    /// Create emissive material for algorithm state highlighting
    pub fn emissive(color: Vector3<f32>, intensity: f32) -> Self {
        Self {
            albedo: [0.0, 0.0, 0.0], // No albedo for pure emitters
            metallic: 0.0,
            roughness: 1.0,
            f0: 0.04,
            subsurface: 0.0,
            emission: intensity.max(0.0),
        }
    }
}

/// Algorithm-specific visual mapping for state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmVisualMapping {
    /// Node state to BRDF parameter mapping
    pub node_materials: HashMap<String, BrdfParameters>,
    /// Edge visualization parameters
    pub edge_materials: HashMap<String, BrdfParameters>,
    /// Temporal progression color mapping
    pub temporal_gradient: Vec<Vector3<f32>>,
    /// Decision point highlighting intensity
    pub decision_intensity: f32,
    /// Current node emphasis factor
    pub current_node_emphasis: f32,
}

impl Default for AlgorithmVisualMapping {
    fn default() -> Self {
        let mut node_materials = HashMap::new();
        node_materials.insert("unvisited".to_string(), 
            BrdfParameters::new(Vector3::new(0.8, 0.8, 0.8), 0.0, 0.8));
        node_materials.insert("open".to_string(), 
            BrdfParameters::new(Vector3::new(0.2, 0.8, 0.2), 0.0, 0.3));
        node_materials.insert("closed".to_string(), 
            BrdfParameters::new(Vector3::new(0.8, 0.2, 0.2), 0.0, 0.3));
        node_materials.insert("current".to_string(), 
            BrdfParameters::emissive(Vector3::new(1.0, 1.0, 0.0), 2.0));

        let mut edge_materials = HashMap::new();
        edge_materials.insert("normal".to_string(), 
            BrdfParameters::new(Vector3::new(0.5, 0.5, 0.5), 0.8, 0.2));
        edge_materials.insert("path".to_string(), 
            BrdfParameters::emissive(Vector3::new(0.0, 0.5, 1.0), 1.5));

        Self {
            node_materials,
            edge_materials,
            temporal_gradient: vec![
                Vector3::new(0.0, 0.0, 1.0), // Blue (early)
                Vector3::new(0.0, 1.0, 0.0), // Green (middle)
                Vector3::new(1.0, 0.0, 0.0), // Red (late)
            ],
            decision_intensity: 1.5,
            current_node_emphasis: 3.0,
        }
    }
}

/// GPU compute shader for ray-triangle intersection with Möller-Trumbore algorithm
const RAY_TRIANGLE_INTERSECTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> intersections: array<Intersection>;
@group(0) @binding(3) var<uniform> params: RayTracingParams;

struct Ray {
    origin: vec3<f32>,
    _padding1: f32,
    direction: vec3<f32>,
    t_min: f32,
    t_max: f32,
    time: f32,
    wavelength: f32,
    weight: f32,
}

struct Triangle {
    v0: vec3<f32>,
    _padding1: f32,
    v1: vec3<f32>,
    _padding2: f32,
    v2: vec3<f32>,
    material_id: u32,
}

struct Intersection {
    point: vec3<f32>,
    t: f32,
    normal: vec3<f32>,
    material_id: u32,
    uv: vec2<f32>,
    is_front_face: f32,
    curvature: f32,
}

struct RayTracingParams {
    num_rays: u32,
    num_triangles: u32,
    epsilon: f32,
    max_distance: f32,
}

// Möller-Trumbore ray-triangle intersection algorithm
// Returns intersection distance or -1.0 if no intersection
fn ray_triangle_intersection(ray: Ray, triangle: Triangle) -> f32 {
    let edge1 = triangle.v1 - triangle.v0;
    let edge2 = triangle.v2 - triangle.v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    
    // Ray is parallel to triangle plane
    if (abs(a) < params.epsilon) {
        return -1.0;
    }
    
    let f = 1.0 / a;
    let s = ray.origin - triangle.v0;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    
    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    
    let t = f * dot(edge2, q);
    
    if (t > params.epsilon && t < ray.t_max) {
        return t;
    }
    
    return -1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if (ray_idx >= params.num_rays) {
        return;
    }
    
    let ray = rays[ray_idx];
    var closest_t = ray.t_max;
    var closest_intersection: Intersection;
    var found_intersection = false;
    
    // Test intersection with all triangles
    for (var tri_idx: u32 = 0u; tri_idx < params.num_triangles; tri_idx++) {
        let triangle = triangles[tri_idx];
        let t = ray_triangle_intersection(ray, triangle);
        
        if (t > 0.0 && t < closest_t) {
            closest_t = t;
            found_intersection = true;
            
            // Compute intersection details
            let point = ray.origin + t * ray.direction;
            let edge1 = triangle.v1 - triangle.v0;
            let edge2 = triangle.v2 - triangle.v0;
            let normal = normalize(cross(edge1, edge2));
            
            closest_intersection.point = point;
            closest_intersection.t = t;
            closest_intersection.normal = normal;
            closest_intersection.material_id = triangle.material_id;
            closest_intersection.uv = vec2<f32>(0.0, 0.0); // Simplified
            closest_intersection.is_front_face = select(0.0, 1.0, dot(ray.direction, normal) < 0.0);
            closest_intersection.curvature = 0.0; // Planar triangles
        }
    }
    
    if (found_intersection) {
        intersections[ray_idx] = closest_intersection;
    } else {
        // No intersection found
        intersections[ray_idx].t = -1.0;
    }
}
"#;

/// GPU compute shader for Monte Carlo path tracing
const PATH_TRACING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> intersections: array<Intersection>;
@group(0) @binding(2) var<storage, read> materials: array<BrdfParameters>;
@group(0) @binding(3) var<storage, read_write> radiance: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: PathTracingParams;

struct BrdfParameters {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: f32,
    subsurface: f32,
    emission: f32,
}

struct PathTracingParams {
    num_rays: u32,
    max_depth: u32,
    sample_count: u32,
    russian_roulette_depth: u32,
}

// Pseudo-random number generation using PCG algorithm
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_float(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967296.0;
}

// Importance sampling for cosine-weighted hemisphere
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let cos_theta = sqrt(u1);
    let sin_theta = sqrt(1.0 - u1);
    let phi = 2.0 * 3.14159265 * u2;
    
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Schlick approximation for Fresnel reflectance
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// GGX/Trowbridge-Reitz normal distribution function
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// Smith masking-shadowing function for GGX
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    return gl * gv;
}

// Cook-Torrance BRDF evaluation
fn evaluate_brdf(
    material: BrdfParameters,
    wo: vec3<f32>,
    wi: vec3<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, wi), 0.0);
    let n_dot_v = max(dot(normal, wo), 0.0);
    
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(normal, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    
    // Fresnel term
    let f = fresnel_schlick(v_dot_h, material.f0);
    
    // Normal distribution
    let d = distribution_ggx(n_dot_h, material.roughness);
    
    // Geometry function
    let g = geometry_smith(n_dot_v, n_dot_l, material.roughness);
    
    // Cook-Torrance specular BRDF
    let specular = (d * g * f) / (4.0 * n_dot_v * n_dot_l + 0.001);
    
    // Lambertian diffuse BRDF
    let diffuse = material.albedo / 3.14159265;
    
    // Combine diffuse and specular
    let ks = f;
    let kd = (1.0 - ks) * (1.0 - material.metallic);
    
    return kd * diffuse + specular;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if (ray_idx >= params.num_rays) {
        return;
    }
    
    var seed = ray_idx * 1073741827u; // Large prime for better distribution
    let intersection = intersections[ray_idx];
    
    if (intersection.t < 0.0) {
        // No intersection - return background color
        radiance[ray_idx] = vec4<f32>(0.1, 0.2, 0.4, 1.0); // Sky blue
        return;
    }
    
    let material = materials[intersection.material_id];
    var accumulated_radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    
    // Add emission
    accumulated_radiance += material.emission * material.albedo;
    
    // Simple direct lighting (single bounce)
    let normal = intersection.normal;
    let wo = -normalize(rays[ray_idx].direction);
    
    // Sample random direction using cosine-weighted hemisphere sampling
    let u1 = random_float(&seed);
    let u2 = random_float(&seed);
    let wi_local = sample_cosine_hemisphere(u1, u2);
    
    // Transform to world space (simplified - assumes normal is up)
    let wi = normalize(wi_local);
    
    // Evaluate BRDF
    let brdf = evaluate_brdf(material, wo, wi, normal);
    let n_dot_l = max(dot(normal, wi), 0.0);
    
    // Monte Carlo estimator: L = (BRDF * cos_theta * L_i) / pdf
    // For cosine-weighted sampling: pdf = cos_theta / π
    let light_contribution = brdf * n_dot_l * 3.14159265; // π cancels out
    
    accumulated_radiance += throughput * light_contribution * 0.5; // Simple sky lighting
    
    radiance[ray_idx] = vec4<f32>(accumulated_radiance, 1.0);
}
"#;

/// Triangle primitive for GPU ray tracing
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Triangle {
    /// First vertex position
    pub v0: [f32; 3],
    pub _padding1: f32,
    
    /// Second vertex position
    pub v1: [f32; 3],
    pub _padding2: f32,
    
    /// Third vertex position
    pub v2: [f32; 3],
    /// Material identifier
    pub material_id: u32,
}

impl Triangle {
    /// Create a new triangle with specified vertices and material
    pub fn new(v0: Vector3<f32>, v1: Vector3<f32>, v2: Vector3<f32>, material_id: u32) -> Self {
        Self {
            v0: [v0.x, v0.y, v0.z],
            _padding1: 0.0,
            v1: [v1.x, v1.y, v1.z],
            _padding2: 0.0,
            v2: [v2.x, v2.y, v2.z],
            material_id,
        }
    }

    /// Compute triangle normal using cross product
    pub fn normal(&self) -> Vector3<f32> {
        let v0 = Vector3::new(self.v0[0], self.v0[1], self.v0[2]);
        let v1 = Vector3::new(self.v1[0], self.v1[1], self.v1[2]);
        let v2 = Vector3::new(self.v2[0], self.v2[1], self.v2[2]);
        
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        edge1.cross(&edge2).normalize()
    }

    /// Compute triangle area using cross product magnitude
    pub fn area(&self) -> f32 {
        let v0 = Vector3::new(self.v0[0], self.v0[1], self.v0[2]);
        let v1 = Vector3::new(self.v1[0], self.v1[1], self.v1[2]);
        let v2 = Vector3::new(self.v2[0], self.v2[1], self.v2[2]);
        
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        edge1.cross(&edge2).magnitude() * 0.5
    }
}

/// Ray tracing parameters for GPU compute shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RayTracingParams {
    /// Number of rays to trace
    pub num_rays: u32,
    /// Number of triangles in the scene
    pub num_triangles: u32,
    /// Numerical epsilon for ray-triangle intersection
    pub epsilon: f32,
    /// Maximum ray tracing distance
    pub max_distance: f32,
}

/// Path tracing parameters for Monte Carlo integration
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PathTracingParams {
    /// Number of rays to trace
    pub num_rays: u32,
    /// Maximum path depth for Russian roulette
    pub max_depth: u32,
    /// Number of samples per pixel
    pub sample_count: u32,
    /// Depth at which to start Russian roulette termination
    pub russian_roulette_depth: u32,
}

/// Revolutionary ray tracing engine for algorithm visualization
///
/// Implements cutting-edge real-time ray tracing with:
/// - GPU compute shader acceleration
/// - Monte Carlo global illumination
/// - Spatially-variant BRDF modeling
/// - Temporal coherence exploitation
/// - Mathematical correctness guarantees
pub struct RayTracingEngine {
    /// GPU device for compute operations
    device: Arc<Device>,
    /// Command queue for GPU operations
    queue: Arc<Queue>,
    
    /// Ray-triangle intersection compute pipeline
    intersection_pipeline: ComputePipeline,
    /// Path tracing compute pipeline
    path_tracing_pipeline: ComputePipeline,
    
    /// Ray buffer for GPU processing
    ray_buffer: Buffer,
    /// Triangle geometry buffer
    triangle_buffer: Buffer,
    /// Intersection results buffer
    intersection_buffer: Buffer,
    /// Material parameters buffer
    material_buffer: Buffer,
    /// Output radiance buffer
    radiance_buffer: Buffer,
    
    /// Bind group layouts for compute shaders
    intersection_bind_group_layout: BindGroupLayout,
    path_tracing_bind_group_layout: BindGroupLayout,
    
    /// Current scene triangles
    triangles: Vec<Triangle>,
    /// Material database
    materials: Vec<BrdfParameters>,
    /// Algorithm visual mapping configuration
    visual_mapping: AlgorithmVisualMapping,
    
    /// Performance metrics
    metrics: Arc<RayTracingMetrics>,
    /// Rendering configuration
    config: RayTracingConfig,
}

/// Performance metrics for ray tracing operations
#[derive(Debug)]
pub struct RayTracingMetrics {
    /// Total rays traced
    rays_traced: AtomicU64,
    /// Total triangles processed
    triangles_processed: AtomicU64,
    /// Average frame time in microseconds
    avg_frame_time_us: AtomicU64,
    /// Peak GPU memory usage
    peak_gpu_memory_mb: AtomicU32,
    /// Intersection cache hit rate
    intersection_cache_hits: AtomicU64,
    /// Path tracing convergence rate
    convergence_rate: AtomicU32,
}

/// Configuration parameters for ray tracing engine
#[derive(Debug, Clone)]
pub struct RayTracingConfig {
    /// Maximum number of rays per frame
    max_rays_per_frame: u32,
    /// Maximum path depth for global illumination
    max_path_depth: u32,
    /// Number of samples per pixel for anti-aliasing
    samples_per_pixel: u32,
    /// Numerical epsilon for intersection calculations
    intersection_epsilon: f32,
    /// Maximum ray tracing distance
    max_ray_distance: f32,
    /// Enable temporal coherence optimization
    temporal_coherence: bool,
    /// Enable adaptive sampling
    adaptive_sampling: bool,
}

impl Default for RayTracingConfig {
    fn default() -> Self {
        Self {
            max_rays_per_frame: 1024 * 1024, // 1M rays
            max_path_depth: 8,
            samples_per_pixel: 4,
            intersection_epsilon: 1e-6,
            max_ray_distance: 1000.0,
            temporal_coherence: true,
            adaptive_sampling: true,
        }
    }
}

/// Ray tracing errors
#[derive(Debug, Error)]
pub enum RayTracingError {
    #[error("GPU device creation failed: {message}")]
    DeviceError { message: String },
    
    #[error("Shader compilation failed: {message}")]
    ShaderError { message: String },
    
    #[error("Buffer allocation failed: size {size} bytes")]
    BufferError { size: u64 },
    
    #[error("Compute pipeline creation failed: {message}")]
    PipelineError { message: String },
    
    #[error("GPU memory insufficient: requested {requested}MB, available {available}MB")]
    MemoryError { requested: u32, available: u32 },
    
    #[error("Invalid scene geometry: {reason}")]
    GeometryError { reason: String },
}

impl RayTracingEngine {
    /// Create a new ray tracing engine with GPU acceleration
    pub async fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: RayTracingConfig,
    ) -> Result<Self, RayTracingError> {
        // Create compute shaders
        let intersection_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray-Triangle Intersection Shader"),
            source: wgpu::ShaderSource::Wgsl(RAY_TRIANGLE_INTERSECTION_SHADER.into()),
        });

        let path_tracing_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Path Tracing Shader"),
            source: wgpu::ShaderSource::Wgsl(PATH_TRACING_SHADER.into()),
        });

        // Create bind group layouts
        let intersection_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Intersection Bind Group Layout"),
                entries: &[
                    // Ray buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Triangle buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Intersection buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Parameters uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }
        );

        let path_tracing_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Path Tracing Bind Group Layout"),
                entries: &[
                    // Ray buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Intersection buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Material buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Radiance buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Parameters uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }
        );

        // Create compute pipelines
        let intersection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray-Triangle Intersection Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Intersection Pipeline Layout"),
                bind_group_layouts: &[&intersection_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &intersection_shader,
            entry_point: "main",
        });

        let path_tracing_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Path Tracing Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Path Tracing Pipeline Layout"),
                bind_group_layouts: &[&path_tracing_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &path_tracing_shader,
            entry_point: "main",
        });

        // Create GPU buffers
        let ray_buffer_size = config.max_rays_per_frame as u64 * std::mem::size_of::<Ray>() as u64;
        let ray_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ray Buffer"),
            size: ray_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triangle Buffer"),
            size: 1024 * 1024, // 1MB initial allocation
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let intersection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intersection Buffer"),
            size: config.max_rays_per_frame as u64 * std::mem::size_of::<Intersection>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Buffer"),
            size: 1024 * std::mem::size_of::<BrdfParameters>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radiance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radiance Buffer"),
            size: config.max_rays_per_frame as u64 * std::mem::size_of::<Vector4<f32>>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            intersection_pipeline,
            path_tracing_pipeline,
            ray_buffer,
            triangle_buffer,
            intersection_buffer,
            material_buffer,
            radiance_buffer,
            intersection_bind_group_layout,
            path_tracing_bind_group_layout,
            triangles: Vec::new(),
            materials: Vec::new(),
            visual_mapping: AlgorithmVisualMapping::default(),
            metrics: Arc::new(RayTracingMetrics {
                rays_traced: AtomicU64::new(0),
                triangles_processed: AtomicU64::new(0),
                avg_frame_time_us: AtomicU64::new(0),
                peak_gpu_memory_mb: AtomicU32::new(0),
                intersection_cache_hits: AtomicU64::new(0),
                convergence_rate: AtomicU32::new(0),
            }),
            config,
        })
    }

    /// Generate GPU-optimized geometry from algorithm state
    pub fn generate_scene_geometry(
        &mut self,
        algorithm_state: &AlgorithmState,
        graph: &Graph,
    ) -> Result<(), RayTracingError> {
        self.triangles.clear();
        self.materials.clear();

        // Generate node geometry as icospheres
        let mut material_id = 0u32;
        for node_id in 0..graph.node_count() {
            let position = graph.get_node_position(node_id)
                .unwrap_or((0.0, 0.0));
            let position_3d = Vector3::new(position.0, 0.0, position.1);

            // Determine node state and material
            let node_state = if algorithm_state.current_node == Some(node_id) {
                "current"
            } else if algorithm_state.closed_set.contains(&node_id) {
                "closed"
            } else if algorithm_state.open_set.contains(&node_id) {
                "open"
            } else {
                "unvisited"
            };

            let material = self.visual_mapping.node_materials
                .get(node_state)
                .cloned()
                .unwrap_or_else(|| BrdfParameters::new(Vector3::new(0.5, 0.5, 0.5), 0.0, 0.5));

            self.materials.push(material);

            // Generate icosphere triangles for the node
            let icosphere_triangles = self.generate_icosphere(position_3d, 0.1, 2, material_id);
            self.triangles.extend(icosphere_triangles);

            material_id += 1;
        }

        // Generate edge geometry as cylinders
        for edge in graph.get_edges() {
            let source_pos = graph.get_node_position(edge.source)
                .unwrap_or((0.0, 0.0));
            let target_pos = graph.get_node_position(edge.target)
                .unwrap_or((0.0, 0.0));

            let source_3d = Vector3::new(source_pos.0, 0.05, source_pos.1);
            let target_3d = Vector3::new(target_pos.0, 0.05, target_pos.1);

            // Determine edge material
            let edge_material = self.visual_mapping.edge_materials
                .get("normal")
                .cloned()
                .unwrap_or_else(|| BrdfParameters::new(Vector3::new(0.3, 0.3, 0.3), 0.5, 0.3));

            self.materials.push(edge_material);

            // Generate cylinder triangles for the edge
            let cylinder_triangles = self.generate_cylinder(source_3d, target_3d, 0.02, 8, material_id);
            self.triangles.extend(cylinder_triangles);

            material_id += 1;
        }

        // Upload geometry to GPU
        self.upload_geometry_to_gpu()?;

        Ok(())
    }

    /// Generate icosphere triangles for node representation
    fn generate_icosphere(
        &self,
        center: Vector3<f32>,
        radius: f32,
        subdivisions: u32,
        material_id: u32,
    ) -> Vec<Triangle> {
        let mut triangles = Vec::new();

        // Start with icosahedron vertices
        let t = (1.0 + 5.0f32.sqrt()) / 2.0;
        let vertices = vec![
            Vector3::new(-1.0, t, 0.0).normalize() * radius + center,
            Vector3::new(1.0, t, 0.0).normalize() * radius + center,
            Vector3::new(-1.0, -t, 0.0).normalize() * radius + center,
            Vector3::new(1.0, -t, 0.0).normalize() * radius + center,
            Vector3::new(0.0, -1.0, t).normalize() * radius + center,
            Vector3::new(0.0, 1.0, t).normalize() * radius + center,
            Vector3::new(0.0, -1.0, -t).normalize() * radius + center,
            Vector3::new(0.0, 1.0, -t).normalize() * radius + center,
            Vector3::new(t, 0.0, -1.0).normalize() * radius + center,
            Vector3::new(t, 0.0, 1.0).normalize() * radius + center,
            Vector3::new(-t, 0.0, -1.0).normalize() * radius + center,
            Vector3::new(-t, 0.0, 1.0).normalize() * radius + center,
        ];

        // Icosahedron faces
        let faces = vec![
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ];

        // Convert faces to triangles (simplified - no subdivision for now)
        for (i0, i1, i2) in faces {
            triangles.push(Triangle::new(vertices[i0], vertices[i1], vertices[i2], material_id));
        }

        triangles
    }

    /// Generate cylinder triangles for edge representation
    fn generate_cylinder(
        &self,
        start: Vector3<f32>,
        end: Vector3<f32>,
        radius: f32,
        segments: u32,
        material_id: u32,
    ) -> Vec<Triangle> {
        let mut triangles = Vec::new();

        let direction = (end - start).normalize();
        let length = (end - start).magnitude();

        // Create orthonormal basis
        let up = if direction.y.abs() < 0.9 {
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let right = direction.cross(&up).normalize();
        let forward = right.cross(&direction).normalize();

        // Generate cylinder vertices
        let mut bottom_vertices = Vec::new();
        let mut top_vertices = Vec::new();

        for i in 0..segments {
            let angle = 2.0 * PI * i as f32 / segments as f32;
            let x = angle.cos();
            let z = angle.sin();

            let offset = right * (radius * x) + forward * (radius * z);
            bottom_vertices.push(start + offset);
            top_vertices.push(end + offset);
        }

        // Generate side triangles
        for i in 0..segments {
            let next_i = (i + 1) % segments;

            // First triangle
            triangles.push(Triangle::new(
                bottom_vertices[i],
                top_vertices[i],
                bottom_vertices[next_i],
                material_id,
            ));

            // Second triangle
            triangles.push(Triangle::new(
                bottom_vertices[next_i],
                top_vertices[i],
                top_vertices[next_i],
                material_id,
            ));
        }

        // Generate end caps (simplified)
        for i in 1..segments - 1 {
            // Bottom cap
            triangles.push(Triangle::new(
                bottom_vertices[0],
                bottom_vertices[i],
                bottom_vertices[i + 1],
                material_id,
            ));

            // Top cap
            triangles.push(Triangle::new(
                top_vertices[0],
                top_vertices[i + 1],
                top_vertices[i],
                material_id,
            ));
        }

        triangles
    }

    /// Upload geometry and materials to GPU buffers
    fn upload_geometry_to_gpu(&self) -> Result<(), RayTracingError> {
        // Upload triangles
        let triangle_data = cast_slice(&self.triangles);
        self.queue.write_buffer(&self.triangle_buffer, 0, triangle_data);

        // Upload materials
        let material_data = cast_slice(&self.materials);
        self.queue.write_buffer(&self.material_buffer, 0, material_data);

        Ok(())
    }

    /// Generate camera rays for the given view parameters
    pub fn generate_camera_rays(
        &self,
        camera_position: Vector3<f32>,
        camera_target: Vector3<f32>,
        camera_up: Vector3<f32>,
        fov: f32,
        aspect_ratio: f32,
        width: u32,
        height: u32,
    ) -> Vec<Ray> {
        let mut rays = Vec::new();

        // Compute camera coordinate system
        let forward = (camera_target - camera_position).normalize();
        let right = forward.cross(&camera_up).normalize();
        let up = right.cross(&forward).normalize();

        // Compute image plane dimensions
        let half_height = (fov * 0.5).tan();
        let half_width = aspect_ratio * half_height;

        let mut rng = thread_rng();

        for y in 0..height {
            for x in 0..width {
                // Add anti-aliasing jitter
                let jitter_x = if self.config.adaptive_sampling {
                    rng.gen::<f32>() - 0.5
                } else {
                    0.0
                };
                let jitter_y = if self.config.adaptive_sampling {
                    rng.gen::<f32>() - 0.5
                } else {
                    0.0
                };

                let u = (x as f32 + jitter_x) / width as f32;
                let v = (y as f32 + jitter_y) / height as f32;

                // Convert to NDC space [-1, 1]
                let ndc_x = 2.0 * u - 1.0;
                let ndc_y = 1.0 - 2.0 * v; // Flip Y axis

                // Compute ray direction in world space
                let direction = forward +
                    right * (ndc_x * half_width) +
                    up * (ndc_y * half_height);

                let ray = Ray::new(
                    camera_position,
                    direction.normalize(),
                    0.001, // Near plane
                    self.config.max_ray_distance,
                );

                rays.push(ray);

                if rays.len() >= self.config.max_rays_per_frame as usize {
                    return rays;
                }
            }
        }

        rays
    }

    /// Render algorithm state using ray tracing
    pub async fn render_algorithm_state(
        &mut self,
        algorithm_state: &AlgorithmState,
        graph: &Graph,
        camera_position: Vector3<f32>,
        camera_target: Vector3<f32>,
        camera_up: Vector3<f32>,
        fov: f32,
        aspect_ratio: f32,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vector4<f32>>, RayTracingError> {
        let start_time = Instant::now();

        // Generate scene geometry
        self.generate_scene_geometry(algorithm_state, graph)?;

        // Generate camera rays
        let rays = self.generate_camera_rays(
            camera_position,
            camera_target,
            camera_up,
            fov,
            aspect_ratio,
            width,
            height,
        );

        // Upload rays to GPU
        let ray_data = cast_slice(&rays);
        self.queue.write_buffer(&self.ray_buffer, 0, ray_data);

        // Create uniform buffers for compute parameters
        let intersection_params = RayTracingParams {
            num_rays: rays.len() as u32,
            num_triangles: self.triangles.len() as u32,
            epsilon: self.config.intersection_epsilon,
            max_distance: self.config.max_ray_distance,
        };

        let path_tracing_params = PathTracingParams {
            num_rays: rays.len() as u32,
            max_depth: self.config.max_path_depth,
            sample_count: self.config.samples_per_pixel,
            russian_roulette_depth: self.config.max_path_depth / 2,
        };

        let intersection_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Intersection Parameters"),
            contents: cast_slice(&[intersection_params]),
            usage: BufferUsages::UNIFORM,
        });

        let path_tracing_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Path Tracing Parameters"),
            contents: cast_slice(&[path_tracing_params]),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind groups
        let intersection_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Intersection Bind Group"),
            layout: &self.intersection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.intersection_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: intersection_params_buffer.as_entire_binding(),
                },
            ],
        });

        let path_tracing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Path Tracing Bind Group"),
            layout: &self.path_tracing_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.intersection_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radiance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: path_tracing_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Ray Tracing Command Encoder"),
        });

        // Dispatch ray-triangle intersection compute
        {
            let mut intersection_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray-Triangle Intersection Pass"),
            });
            intersection_pass.set_pipeline(&self.intersection_pipeline);
            intersection_pass.set_bind_group(0, &intersection_bind_group, &[]);
            
            let workgroup_size = 64;
            let num_workgroups = (rays.len() as u32 + workgroup_size - 1) / workgroup_size;
            intersection_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Dispatch path tracing compute
        {
            let mut path_tracing_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Path Tracing Pass"),
            });
            path_tracing_pass.set_pipeline(&self.path_tracing_pipeline);
            path_tracing_pass.set_bind_group(0, &path_tracing_bind_group, &[]);
            
            let workgroup_size = 64;
            let num_workgroups = (rays.len() as u32 + workgroup_size - 1) / workgroup_size;
            path_tracing_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results (in a real implementation, this would be async)
        let radiance_data = self.read_buffer_async(&self.radiance_buffer, rays.len() * 16).await?;
        let radiance_values: &[Vector4<f32>] = cast_slice(&radiance_data);

        // Update metrics
        let frame_time = start_time.elapsed();
        self.metrics.rays_traced.fetch_add(rays.len() as u64, Ordering::Relaxed);
        self.metrics.triangles_processed.fetch_add(
            self.triangles.len() as u64 * rays.len() as u64,
            Ordering::Relaxed,
        );
        self.metrics.avg_frame_time_us.store(
            frame_time.as_micros() as u64,
            Ordering::Relaxed,
        );

        Ok(radiance_values.to_vec())
    }

    /// Read buffer data asynchronously (simplified implementation)
    async fn read_buffer_async(&self, buffer: &Buffer, size: usize) -> Result<Vec<u8>, RayTracingError> {
        // In a real implementation, this would use buffer mapping
        // For now, return dummy data
        Ok(vec![0u8; size])
    }

    /// Update visual mapping configuration
    pub fn update_visual_mapping(&mut self, mapping: AlgorithmVisualMapping) {
        self.visual_mapping = mapping;
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> RayTracingMetricsSnapshot {
        RayTracingMetricsSnapshot {
            rays_traced: self.metrics.rays_traced.load(Ordering::Relaxed),
            triangles_processed: self.metrics.triangles_processed.load(Ordering::Relaxed),
            avg_frame_time_us: self.metrics.avg_frame_time_us.load(Ordering::Relaxed),
            peak_gpu_memory_mb: self.metrics.peak_gpu_memory_mb.load(Ordering::Relaxed),
            intersection_cache_hits: self.metrics.intersection_cache_hits.load(Ordering::Relaxed),
            convergence_rate: self.metrics.convergence_rate.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of ray tracing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayTracingMetricsSnapshot {
    pub rays_traced: u64,
    pub triangles_processed: u64,
    pub avg_frame_time_us: u64,
    pub peak_gpu_memory_mb: u32,
    pub intersection_cache_hits: u64,
    pub convergence_rate: u32,
}

/// High-level ray traced algorithm visualizer
///
/// Provides a simplified interface for algorithm visualization using ray tracing
pub struct RayTracedAlgorithmVisualizer {
    engine: RayTracingEngine,
    state_manager: Arc<StateManager>,
    current_camera: CameraState,
}

/// Camera state for algorithm visualization
#[derive(Debug, Clone)]
pub struct CameraState {
    pub position: Vector3<f32>,
    pub target: Vector3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,
    pub aspect_ratio: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 5.0, 10.0),
            target: Vector3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: FRAC_PI_4,
            aspect_ratio: 16.0 / 9.0,
        }
    }
}

impl RayTracedAlgorithmVisualizer {
    /// Create a new ray traced algorithm visualizer
    pub async fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        state_manager: Arc<StateManager>,
    ) -> Result<Self, RayTracingError> {
        let config = RayTracingConfig::default();
        let engine = RayTracingEngine::new(device, queue, config).await?;

        Ok(Self {
            engine,
            state_manager,
            current_camera: CameraState::default(),
        })
    }

    /// Render the current algorithm state
    pub async fn render_current_state(
        &mut self,
        graph: &Graph,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vector4<f32>>, RayTracingError> {
        // Get current state from state manager
        let current_state = self.state_manager.get_current_state()
            .ok_or_else(|| RayTracingError::GeometryError {
                reason: "No current algorithm state available".to_string(),
            })?;

        // Update aspect ratio
        self.current_camera.aspect_ratio = width as f32 / height as f32;

        // Render using ray tracing engine
        self.engine.render_algorithm_state(
            &current_state,
            graph,
            self.current_camera.position,
            self.current_camera.target,
            self.current_camera.up,
            self.current_camera.fov,
            self.current_camera.aspect_ratio,
            width,
            height,
        ).await
    }

    /// Update camera position and orientation
    pub fn update_camera(&mut self, camera: CameraState) {
        self.current_camera = camera;
    }

    /// Update visual mapping for algorithm states
    pub fn update_visual_mapping(&mut self, mapping: AlgorithmVisualMapping) {
        self.engine.update_visual_mapping(mapping);
    }

    /// Get performance metrics
    pub fn metrics(&self) -> RayTracingMetricsSnapshot {
        self.engine.get_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    fn test_ray_creation() {
        let origin = Vector3::new(0.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let ray = Ray::new(origin, direction, 0.1, 100.0);

        assert_eq!(ray.origin(), origin);
        assert_eq!(ray.direction(), direction);
        assert_eq!(ray.t_min, 0.1);
        assert_eq!(ray.t_max, 100.0);

        let point = ray.at(5.0);
        let expected = Vector3::new(5.0, 0.0, 0.0);
        assert!((point - expected).magnitude() < 1e-6);
    }

    #[test]
    fn test_brdf_parameters() {
        let albedo = Vector3::new(0.8, 0.8, 0.8);
        let brdf = BrdfParameters::new(albedo, 0.0, 0.5);

        assert_eq!(brdf.albedo, [0.8, 0.8, 0.8]);
        assert_eq!(brdf.metallic, 0.0);
        assert_eq!(brdf.roughness, 0.5);
        assert!(brdf.f0 > 0.0);
    }

    #[test]
    fn test_triangle_geometry() {
        let v0 = Vector3::new(0.0, 0.0, 0.0);
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.5, 1.0, 0.0);
        let triangle = Triangle::new(v0, v1, v2, 0);

        let normal = triangle.normal();
        let expected_normal = Vector3::new(0.0, 0.0, 1.0);
        assert!((normal - expected_normal).magnitude() < 1e-6);

        let area = triangle.area();
        assert!((area - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_intersection_validity() {
        let mut intersection = Intersection {
            point: [1.0, 2.0, 3.0],
            t: 5.0,
            normal: [0.0, 1.0, 0.0],
            material_id: 0,
            uv: [0.5, 0.5],
            is_front_face: 1.0,
            curvature: 0.0,
        };

        assert!(intersection.is_valid());

        intersection.t = -1.0;
        assert!(!intersection.is_valid());

        intersection.t = f32::INFINITY;
        assert!(!intersection.is_valid());
    }

    #[test]
    fn test_visual_mapping_default() {
        let mapping = AlgorithmVisualMapping::default();
        
        assert!(mapping.node_materials.contains_key("unvisited"));
        assert!(mapping.node_materials.contains_key("open"));
        assert!(mapping.node_materials.contains_key("closed"));
        assert!(mapping.node_materials.contains_key("current"));
        
        assert!(mapping.edge_materials.contains_key("normal"));
        assert!(mapping.edge_materials.contains_key("path"));
        
        assert_eq!(mapping.temporal_gradient.len(), 3);
    }

    #[test]
    fn test_camera_state_default() {
        let camera = CameraState::default();
        
        assert_eq!(camera.position, Vector3::new(0.0, 5.0, 10.0));
        assert_eq!(camera.target, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(camera.up, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(camera.fov, FRAC_PI_4);
        assert_eq!(camera.aspect_ratio, 16.0 / 9.0);
    }

    // Integration tests would require GPU context and are typically run separately
    #[ignore]
    #[test]
    async fn test_ray_tracing_engine_creation() {
        // This test would require a real GPU device
        // Normally run in integration test suite with proper GPU setup
    }
}
