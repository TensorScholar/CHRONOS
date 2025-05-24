//! Heatmap visualization for scalar fields and algorithm metrics
//!
//! Implements a high-performance heatmap visualization for displaying 
//! scalar fields, heuristic functions, and algorithm metrics. Supports
//! multiple interpolation modes, customizable color gradients, and
//! contour lines for enhanced visual understanding.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::Arc;
use wgpu::{Device, Queue, RenderPass, RenderPipeline, BindGroup, TextureFormat};
use glam::{Vec2, Vec4, Mat4};
use bytemuck::{Pod, Zeroable};
use log::{debug, error, info, warn};

use crate::engine::platform::RenderingPlatform;
use crate::engine::shader::{ShaderManager, ShaderStage, ShaderVariant};
use crate::engine::texture::Texture;
use crate::engine::primitives::{Vertex, VertexLayout, UniformBuffer};
use crate::view::{View, ViewError, InputEvent, ViewBounds, Renderable};

/// Data source for heatmap visualization
pub trait HeatmapDataSource: Send + Sync {
    /// Get data dimensions (width, height)
    fn dimensions(&self) -> (usize, usize);
    
    /// Get data value at position
    fn value_at(&self, x: usize, y: usize) -> Option<f32>;
    
    /// Get value range (min, max)
    fn value_range(&self) -> (f32, f32);
    
    /// Get data timestamp for animation
    fn timestamp(&self) -> u64;
    
    /// Get data label (optional)
    fn label(&self) -> Option<&str> {
        None
    }
    
    /// Get metadata (optional)
    fn metadata(&self) -> Option<&[(&str, &str)]> {
        None
    }
}

/// Interpolation modes for heatmap rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationMode {
    /// Nearest neighbor interpolation
    Nearest,
    
    /// Bilinear interpolation
    Bilinear,
    
    /// Bicubic interpolation
    Bicubic,
}

impl InterpolationMode {
    /// Get shader define for interpolation mode
    fn shader_define(&self) -> &'static str {
        match self {
            Self::Nearest => "INTERPOLATION_NEAREST",
            Self::Bilinear => "INTERPOLATION_BILINEAR",
            Self::Bicubic => "INTERPOLATION_BICUBIC",
        }
    }
}

/// Color mapping mode for heatmap
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorMapMode {
    /// Linear mapping from value range to color gradient
    Linear,
    
    /// Logarithmic mapping from value range to color gradient
    Logarithmic,
    
    /// Histogram equalization for enhanced contrast
    HistogramEqualization,
    
    /// Percentile-based mapping for outlier handling
    Percentile,
}

impl ColorMapMode {
    /// Get shader define for color mapping mode
    fn shader_define(&self) -> &'static str {
        match self {
            Self::Linear => "COLORMAP_LINEAR",
            Self::Logarithmic => "COLORMAP_LOG",
            Self::HistogramEqualization => "COLORMAP_HISTOGRAM",
            Self::Percentile => "COLORMAP_PERCENTILE",
        }
    }
}

/// Configuration for heatmap visualization
#[derive(Debug, Clone)]
pub struct HeatmapViewConfig {
    /// Color gradient for values (array of colors)
    pub color_gradient: Vec<[f32; 4]>,
    
    /// Value range mapping (min, max), None for auto-detect
    pub value_range: Option<(f32, f32)>,
    
    /// Interpolation mode
    pub interpolation: InterpolationMode,
    
    /// Color mapping mode
    pub color_map_mode: ColorMapMode,
    
    /// Show contour lines
    pub show_contours: bool,
    
    /// Contour interval, None for auto-detect
    pub contour_interval: Option<f32>,
    
    /// Contour line color
    pub contour_color: [f32; 4],
    
    /// Contour line thickness
    pub contour_thickness: f32,
    
    /// Show color legend
    pub show_legend: bool,
    
    /// Legend position
    pub legend_position: LegendPosition,
    
    /// Background color
    pub background_color: [f32; 4],
    
    /// Animation settings
    pub animation: AnimationSettings,
}

impl Default for HeatmapViewConfig {
    fn default() -> Self {
        Self {
            color_gradient: vec![
                [0.0, 0.0, 1.0, 1.0],  // Blue
                [0.0, 1.0, 1.0, 1.0],  // Cyan
                [0.0, 1.0, 0.0, 1.0],  // Green
                [1.0, 1.0, 0.0, 1.0],  // Yellow
                [1.0, 0.0, 0.0, 1.0],  // Red
            ],
            value_range: None,
            interpolation: InterpolationMode::Bilinear,
            color_map_mode: ColorMapMode::Linear,
            show_contours: false,
            contour_interval: None,
            contour_color: [0.0, 0.0, 0.0, 0.5],
            contour_thickness: 1.0,
            show_legend: true,
            legend_position: LegendPosition::BottomRight,
            background_color: [0.1, 0.1, 0.1, 1.0],
            animation: AnimationSettings::default(),
        }
    }
}

/// Position for color legend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LegendPosition {
    /// Top-right corner
    TopRight,
    
    /// Top-left corner
    TopLeft,
    
    /// Bottom-right corner
    BottomRight,
    
    /// Bottom-left corner
    BottomLeft,
    
    /// Right side
    Right,
    
    /// Left side
    Left,
}

/// Animation settings for heatmap transitions
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    /// Enable animations
    pub enable_animations: bool,
    
    /// Animation duration in seconds
    pub animation_duration: f32,
    
    /// Animation easing function
    pub easing_function: EasingFunction,
    
    /// Cross-fade between data sources
    pub cross_fade: bool,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enable_animations: true,
            animation_duration: 0.3,
            easing_function: EasingFunction::CubicInOut,
            cross_fade: true,
        }
    }
}

/// Easing functions for animations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EasingFunction {
    /// Linear interpolation
    Linear,
    
    /// Quadratic easing in
    QuadraticIn,
    
    /// Quadratic easing out
    QuadraticOut,
    
    /// Quadratic easing in and out
    QuadraticInOut,
    
    /// Cubic easing in
    CubicIn,
    
    /// Cubic easing out
    CubicOut,
    
    /// Cubic easing in and out
    CubicInOut,
    
    /// Exponential easing in
    ExponentialIn,
    
    /// Exponential easing out
    ExponentialOut,
    
    /// Exponential easing in and out
    ExponentialInOut,
}

/// Heatmap view uniform buffer data
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct HeatmapUniforms {
    /// Projection matrix
    projection: [f32; 16],
    
    /// View matrix
    view: [f32; 16],
    
    /// Viewport size (width, height, aspect_ratio, _padding)
    viewport: [f32; 4],
    
    /// Value range (min, max, range, _padding)
    value_range: [f32; 4],
    
    /// Contour parameters (interval, thickness, enabled, _padding)
    contour_params: [f32; 4],
    
    /// Contour color (rgba)
    contour_color: [f32; 4],
    
    /// Animation parameters (time, duration, easing, cross_fade)
    animation_params: [f32; 4],
    
    /// Background color (rgba)
    background_color: [f32; 4],
}

/// Heatmap visualization view
pub struct HeatmapView {
    /// View configuration
    config: HeatmapViewConfig,
    
    /// Current data source
    data_source: Option<Arc<dyn HeatmapDataSource>>,
    
    /// Previous data source (for transitions)
    prev_data_source: Option<Arc<dyn HeatmapDataSource>>,
    
    /// Rendering resources
    resources: Option<HeatmapViewResources>,
    
    /// View bounds
    bounds: ViewBounds,
    
    /// Animation state
    animation: AnimationState,
}

/// Heatmap view rendering resources
struct HeatmapViewResources {
    /// Render pipeline for heatmap
    pipeline: RenderPipeline,
    
    /// Render pipeline for contours
    contour_pipeline: Option<RenderPipeline>,
    
    /// Render pipeline for legend
    legend_pipeline: Option<RenderPipeline>,
    
    /// Vertex buffer for quad
    quad_vertex_buffer: wgpu::Buffer,
    
    /// Index buffer for quad
    quad_index_buffer: wgpu::Buffer,
    
    /// Data texture
    data_texture: Texture,
    
    /// Previous data texture (for transitions)
    prev_data_texture: Option<Texture>,
    
    /// Gradient texture
    gradient_texture: Texture,
    
    /// Uniform buffer for view parameters
    uniform_buffer: UniformBuffer<HeatmapUniforms>,
    
    /// Main bind group for rendering
    bind_group: BindGroup,
    
    /// Contour bind group
    contour_bind_group: Option<BindGroup>,
    
    /// Legend bind group
    legend_bind_group: Option<BindGroup>,
}

/// Animation state for heatmap view
#[derive(Debug, Default)]
struct AnimationState {
    /// Animation timer
    timer: f32,
    
    /// Animation start time
    start_time: f32,
    
    /// Animation in progress
    in_progress: bool,
    
    /// Animation type
    animation_type: AnimationType,
}

/// Animation type for heatmap transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AnimationType {
    /// No animation
    #[default]
    None,
    
    /// Cross-fade between data sources
    CrossFade,
    
    /// Value range transition
    ValueRange,
}

impl HeatmapView {
    /// Create a new heatmap view
    pub fn new(config: HeatmapViewConfig) -> Self {
        Self {
            config,
            data_source: None,
            prev_data_source: None,
            resources: None,
            bounds: ViewBounds::default(),
            animation: AnimationState::default(),
        }
    }
    
    /// Set data source
    pub fn set_data_source(&mut self, data_source: Arc<dyn HeatmapDataSource>) {
        // If animation is enabled, store previous data source
        if self.config.animation.enable_animations && self.data_source.is_some() {
            self.prev_data_source = self.data_source.clone();
            self.animation.start_time = self.animation.timer;
            self.animation.in_progress = true;
            self.animation.animation_type = AnimationType::CrossFade;
        }
        
        self.data_source = Some(data_source);
        
        // Update data texture
        if let Some(resources) = &mut self.resources {
            let device = resources.uniform_buffer.device();
            let queue = device.create_queue();
            
            // Update data texture
            if let Err(e) = self.update_data_texture(device, &queue) {
                error!("Failed to update data texture: {}", e);
            }
            
            // Update value range
            self.update_value_range();
            
            // Update uniforms
            self.update_uniforms();
        }
    }
    
    /// Initialize rendering resources
    fn initialize_resources(&mut self, platform: &RenderingPlatform, shader_manager: &mut ShaderManager) -> Result<(), ViewError> {
        let device = platform.device();
        let queue = platform.queue();
        
        // Create quad geometry
        let quad_vertices = [
            Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
        ];
        
        let quad_indices = [
            0, 1, 2,
            2, 3, 0,
        ];
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Heatmap Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Heatmap Quad Index Buffer"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        // Create uniform buffer
        let value_range = if let Some(data_source) = &self.data_source {
            let (min, max) = data_source.value_range();
            [min, max, max - min, 0.0]
        } else {
            [0.0, 1.0, 1.0, 0.0]
        };
        
        let uniforms = HeatmapUniforms {
            projection: Mat4::IDENTITY.to_cols_array(),
            view: Mat4::IDENTITY.to_cols_array(),
            viewport: [
                self.bounds.width as f32,
                self.bounds.height as f32,
                self.bounds.width as f32 / self.bounds.height as f32,
                0.0,
            ],
            value_range,
            contour_params: [
                self.config.contour_interval.unwrap_or(0.1),
                self.config.contour_thickness,
                if self.config.show_contours { 1.0 } else { 0.0 },
                0.0,
            ],
            contour_color: self.config.contour_color,
            animation_params: [
                0.0, // Current time
                self.config.animation.animation_duration,
                0.0, // Easing type (set in shader)
                if self.config.animation.cross_fade { 1.0 } else { 0.0 },
            ],
            background_color: self.config.background_color,
        };
        
        let uniform_buffer = UniformBuffer::new(device.clone(), "Heatmap Uniforms", uniforms);
        
        // Create initial 1x1 data texture (will be resized later)
        let initial_data = [0.0f32];
        let data_texture = Texture::from_float_data(
            device,
            queue,
            &initial_data,
            1,
            1,
            TextureFormat::R32Float,
            Some("Heatmap Data Texture"),
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )?;
        
        // Create gradient texture
        let gradient_data = self.generate_gradient_data()?;
        let gradient_texture = Texture::from_float_data(
            device,
            queue,
            &gradient_data,
            gradient_data.len() / 4,
            1,
            TextureFormat::Rgba32Float,
            Some("Heatmap Gradient Texture"),
            wgpu::TextureUsages::TEXTURE_BINDING,
        )?;
        
        // Create shader variants for heatmap
        let mut shader_variant = ShaderVariant::new("heatmap", ShaderStage::Fragment)
            .with_define("INTERPOLATION", self.config.interpolation.shader_define())
            .with_define("COLORMAP", self.config.color_map_mode.shader_define());
            
        if self.config.show_contours {
            shader_variant = shader_variant.with_define("ENABLE_CONTOURS", "1");
        }
        
        if self.config.animation.enable_animations {
            shader_variant = shader_variant.with_define("ENABLE_ANIMATION", "1");
        }
        
        let vertex_shader = shader_manager.get_shader(&ShaderVariant::new("heatmap", ShaderStage::Vertex))?;
        let fragment_shader = shader_manager.get_shader(&shader_variant)?;
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Heatmap Pipeline Layout"),
            bind_group_layouts: &[&uniform_buffer.bind_group_layout()],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Heatmap Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader.module(),
                entry_point: vertex_shader.entry_point(ShaderStage::Vertex).unwrap_or("vs_main"),
                buffers: &[VertexLayout::vertex_buffer_layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader.module(),
                entry_point: fragment_shader.entry_point(ShaderStage::Fragment).unwrap_or("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: platform.surface_format(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        
        // Create contour pipeline if needed
        let contour_pipeline = None; // TODO: Implement contour pipeline
        
        // Create legend pipeline if needed
        let legend_pipeline = None; // TODO: Implement legend pipeline
        
        // Create bind group
        let bind_group_layout = uniform_buffer.bind_group_layout();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Heatmap Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.binding_resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&data_texture.view),
                },
wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gradient_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&data_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&gradient_texture.sampler),
                },
            ],
        });
        
        let resources = HeatmapViewResources {
            pipeline,
            contour_pipeline,
            legend_pipeline,
            quad_vertex_buffer: vertex_buffer,
            quad_index_buffer: index_buffer,
            data_texture,
            prev_data_texture: None,
            gradient_texture,
            uniform_buffer,
            bind_group,
            contour_bind_group: None,
            legend_bind_group: None,
        };
        
        self.resources = Some(resources);
        
        // Update data texture if source is available
        if self.data_source.is_some() {
            self.update_data_texture(device, queue)?;
            self.update_value_range();
        }
        
        Ok(())
    }
    
    /// Update data texture with current data source
    fn update_data_texture(&mut self, device: &Device, queue: &Queue) -> Result<(), ViewError> {
        if let Some(data_source) = &self.data_source {
            let (width, height) = data_source.dimensions();
            
            // Skip if dimensions are invalid
            if width == 0 || height == 0 {
                return Err(ViewError::InvalidData("Invalid data dimensions".into()));
            }
            
            // Create float data array
            let mut data = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let value = data_source.value_at(x, y).unwrap_or(0.0);
                    data.push(value);
                }
            }
            
            // Update or create data texture
            if let Some(resources) = &mut self.resources {
                // Check if dimensions have changed
                let current_dims = resources.data_texture.dimensions();
                
                if current_dims.0 != width as u32 || current_dims.1 != height as u32 {
                    // Store previous texture for animation if needed
                    if self.config.animation.enable_animations && self.config.animation.cross_fade {
                        resources.prev_data_texture = Some(resources.data_texture.clone());
                    }
                    
                    // Create new texture with correct dimensions
                    resources.data_texture = Texture::from_float_data(
                        device,
                        queue,
                        &data,
                        width,
                        height,
                        TextureFormat::R32Float,
                        Some("Heatmap Data Texture"),
                        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    )?;
                    
                    // Update bind group
                    resources.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Heatmap Bind Group"),
                        layout: resources.uniform_buffer.bind_group_layout(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: resources.uniform_buffer.binding_resource(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&resources.data_texture.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&resources.gradient_texture.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&resources.data_texture.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&resources.gradient_texture.sampler),
                            },
                        ],
                    });
                } else {
                    // Update existing texture
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &resources.data_texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&data),
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * width as u32),
                            rows_per_image: Some(height as u32),
                        },
                        wgpu::Extent3d {
                            width: width as u32,
                            height: height as u32,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// Update value range in uniforms
    fn update_value_range(&mut self) {
        if let Some(resources) = &mut self.resources {
            if let Some(data_source) = &self.data_source {
                let (min, max) = if let Some((min, max)) = self.config.value_range {
                    (min, max)
                } else {
                    data_source.value_range()
                };
                
                let range = max - min;
                
                // If animation is enabled, use animation for value range transition
                if self.config.animation.enable_animations && self.animation.in_progress && self.animation.animation_type == AnimationType::ValueRange {
                    let progress = (self.animation.timer - self.animation.start_time) / self.config.animation.animation_duration;
                    if progress >= 1.0 {
                        // Animation finished
                        self.animation.in_progress = false;
                    } else {
                        // Animate value range
                        let mut uniforms = resources.uniform_buffer.get();
                        
                        // Linear interpolation for now, could be enhanced with easing functions
                        let old_min = uniforms.value_range[0];
                        let old_max = uniforms.value_range[1];
                        
                        uniforms.value_range[0] = old_min + (min - old_min) * progress;
                        uniforms.value_range[1] = old_max + (max - old_max) * progress;
                        uniforms.value_range[2] = uniforms.value_range[1] - uniforms.value_range[0];
                        
                        resources.uniform_buffer.update(uniforms);
                        return;
                    }
                }
                
                // Update uniform buffer
                let mut uniforms = resources.uniform_buffer.get();
                uniforms.value_range = [min, max, range, 0.0];
                resources.uniform_buffer.update(uniforms);
            }
        }
    }
    
    /// Generate gradient data for texture
    fn generate_gradient_data(&self) -> Result<Vec<f32>, ViewError> {
        if self.config.color_gradient.is_empty() {
            return Err(ViewError::InvalidConfig("Empty color gradient".into()));
        }
        
        // Generate high-resolution gradient
        let resolution = 256;
        let mut data = Vec::with_capacity(resolution * 4);
        
        for i in 0..resolution {
            let t = i as f32 / (resolution - 1) as f32;
            let color = self.sample_gradient(t);
            data.extend_from_slice(&color);
        }
        
        Ok(data)
    }
    
    /// Sample color from gradient at position t (0.0-1.0)
    fn sample_gradient(&self, t: f32) -> [f32; 4] {
        let gradient = &self.config.color_gradient;
        
        if gradient.len() == 1 {
            return gradient[0];
        }
        
        // Find segment
        let segment_count = gradient.len() - 1;
        let segment_t = t * segment_count as f32;
        let segment_idx = segment_t as usize;
        let segment_frac = segment_t - segment_idx as f32;
        
        if segment_idx >= segment_count {
            return gradient[segment_count];
        }
        
        // Interpolate colors
        let c0 = gradient[segment_idx];
        let c1 = gradient[segment_idx + 1];
        
        [
            c0[0] + (c1[0] - c0[0]) * segment_frac,
            c0[1] + (c1[1] - c0[1]) * segment_frac,
            c0[2] + (c1[2] - c0[2]) * segment_frac,
            c0[3] + (c1[3] - c0[3]) * segment_frac,
        ]
    }
    
    /// Update uniforms
    fn update_uniforms(&mut self) {
        if let Some(resources) = &mut self.resources {
            // Create projection matrix (orthographic)
            let width = self.bounds.width as f32;
            let height = self.bounds.height as f32;
            
            let left = -width / 2.0;
            let right = width / 2.0;
            let bottom = height / 2.0;
            let top = -height / 2.0;
            
            let projection = Mat4::orthographic_rh(left, right, bottom, top, -1.0, 1.0);
            
            // Create view matrix
            let view = Mat4::from_translation(Vec3::new(self.bounds.offset_x, self.bounds.offset_y, 0.0))
                * Mat4::from_scale(Vec3::new(self.bounds.scale, self.bounds.scale, 1.0));
            
            // Update uniform buffer
            let mut uniforms = resources.uniform_buffer.get();
            uniforms.projection = projection.to_cols_array();
            uniforms.view = view.to_cols_array();
            uniforms.viewport = [
                width,
                height,
                width / height,
                0.0,
            ];
            uniforms.contour_params = [
                self.config.contour_interval.unwrap_or(0.1),
                self.config.contour_thickness,
                if self.config.show_contours { 1.0 } else { 0.0 },
                0.0,
            ];
            uniforms.contour_color = self.config.contour_color;
            uniforms.animation_params[0] = self.animation.timer;
            uniforms.animation_params[1] = self.config.animation.animation_duration;
            uniforms.animation_params[3] = if self.config.animation.cross_fade { 1.0 } else { 0.0 };
            uniforms.background_color = self.config.background_color;
            
            resources.uniform_buffer.update(uniforms);
        }
    }
}

impl View for HeatmapView {
    fn name(&self) -> &str {
        "Heatmap View"
    }
    
    fn initialize(&mut self, platform: &RenderingPlatform, shader_manager: &mut ShaderManager) -> Result<(), ViewError> {
        self.initialize_resources(platform, shader_manager)
    }
    
    fn update(&mut self, dt: f32) {
        // Update animation timer
        self.animation.timer += dt;
        
        // Check if animation is finished
        if self.animation.in_progress {
            let elapsed = self.animation.timer - self.animation.start_time;
            if elapsed >= self.config.animation.animation_duration {
                self.animation.in_progress = false;
                
                // If cross-fade animation, remove previous texture
                if self.animation.animation_type == AnimationType::CrossFade {
                    if let Some(resources) = &mut self.resources {
                        resources.prev_data_texture = None;
                    }
                }
            }
        }
        
        // Update uniforms
        self.update_value_range();
        self.update_uniforms();
    }
    
    fn render<'a>(&'a self, render_pass: &mut RenderPass<'a>) -> Result<(), ViewError> {
        if let Some(resources) = &self.resources {
            // Set the pipeline
            render_pass.set_pipeline(&resources.pipeline);
            render_pass.set_bind_group(0, &resources.bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(resources.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            // Draw fullscreen quad
            render_pass.draw_indexed(0..6, 0, 0..1);
            
            // TODO: Draw contours if enabled
            
            // TODO: Draw legend if enabled
        }
        
        Ok(())
    }
    
    fn resize(&mut self, width: u32, height: u32) {
        self.bounds.width = width;
        self.bounds.height = height;
        
        // Update uniforms
        self.update_uniforms();
    }
    
    fn handle_input(&mut self, event: &InputEvent) -> bool {
        match event {
            InputEvent::Pan { delta_x, delta_y } => {
                self.bounds.offset_x += *delta_x;
                self.bounds.offset_y += *delta_y;
                self.update_uniforms();
                true
            },
            InputEvent::Zoom { delta, center_x, center_y } => {
                let old_scale = self.bounds.scale;
                self.bounds.scale *= *delta;
                
                // Clamp scale to reasonable range
                self.bounds.scale = self.bounds.scale.max(0.1).min(10.0);
                
                // Adjust offset to zoom toward cursor
                let zoom_factor = self.bounds.scale / old_scale;
                let focus_x = *center_x - self.bounds.width as f32 / 2.0 - self.bounds.offset_x;
                let focus_y = *center_y - self.bounds.height as f32 / 2.0 - self.bounds.offset_y;
                
                self.bounds.offset_x = *center_x - self.bounds.width as f32 / 2.0 - focus_x * zoom_factor;
                self.bounds.offset_y = *center_y - self.bounds.height as f32 / 2.0 - focus_y * zoom_factor;
                
                self.update_uniforms();
                true
            },
            _ => false,
        }
    }
    
    fn bounds(&self) -> &ViewBounds {
        &self.bounds
    }
    
    fn bounds_mut(&mut self) -> &mut ViewBounds {
        &mut self.bounds
    }
}

impl Drop for HeatmapView {
    fn drop(&mut self) {
        // Clean up resources if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock data source for testing
    struct MockDataSource {
        dimensions: (usize, usize),
        range: (f32, f32),
    }
    
    impl HeatmapDataSource for MockDataSource {
        fn dimensions(&self) -> (usize, usize) {
            self.dimensions
        }
        
        fn value_at(&self, x: usize, y: usize) -> Option<f32> {
            if x < self.dimensions.0 && y < self.dimensions.1 {
                // Simple gradient for testing
                let norm_x = x as f32 / (self.dimensions.0 as f32 - 1.0);
                let norm_y = y as f32 / (self.dimensions.1 as f32 - 1.0);
                let value = norm_x * norm_y;
                
                Some(self.range.0 + value * (self.range.1 - self.range.0))
            } else {
                None
            }
        }
        
        fn value_range(&self) -> (f32, f32) {
            self.range
        }
        
        fn timestamp(&self) -> u64 {
            0
        }
    }
    
    #[test]
    fn test_sample_gradient() {
        let config = HeatmapViewConfig {
            color_gradient: vec![
                [0.0, 0.0, 1.0, 1.0], // Blue
                [1.0, 0.0, 0.0, 1.0], // Red
            ],
            ..Default::default()
        };
        
        let view = HeatmapView::new(config);
        
        // Test sampling at different positions
        let c0 = view.sample_gradient(0.0);
        let c1 = view.sample_gradient(1.0);
        let c_mid = view.sample_gradient(0.5);
        
        assert_eq!(c0, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(c1, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(c_mid, [0.5, 0.0, 0.5, 1.0]);
    }
}