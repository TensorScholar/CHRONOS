//! Grid visualization view with GPU-accelerated rendering
//!
//! Implements a high-performance grid visualization with support for
//! large-scale grid data, efficient navigation, and state visualization.
//! Uses instanced rendering for optimal performance and supports
//! interactive exploration of algorithm execution on grid-based problems.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::Arc;
use wgpu::{Device, Queue, RenderPass, RenderPipeline, BindGroup, Buffer};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4, Mat4};

use crate::engine::platform::RenderingPlatform;
use crate::engine::shader::{ShaderManager, ShaderStage, ShaderVariant};
use crate::engine::primitives::{Vertex, VertexLayout, UniformBuffer};
use crate::engine::texture::Texture;
use crate::view::{View, ViewError, InputEvent, ViewBounds, Renderable};

use chronos_core::data_structures::grid::{Grid, CellState};
use chronos_core::algorithm::state::AlgorithmState;

/// Configuration for grid visualization
#[derive(Debug, Clone)]
pub struct GridViewConfig {
    /// Cell size in pixels (at zoom=1.0)
    pub cell_size: f32,
    
    /// Grid line thickness
    pub line_thickness: f32,
    
    /// Color scheme
    pub color_scheme: GridColorScheme,
    
    /// Show grid lines
    pub show_grid_lines: bool,
    
    /// Show coordinates
    pub show_coordinates: bool,
    
    /// Text size for coordinates
    pub coordinate_text_size: f32,
    
    /// Show cell values
    pub show_cell_values: bool,
    
    /// Animation settings
    pub animation: AnimationSettings,
    
    /// Level of detail settings
    pub lod: LodSettings,
}

impl Default for GridViewConfig {
    fn default() -> Self {
        Self {
            cell_size: 32.0,
            line_thickness: 1.0,
            color_scheme: GridColorScheme::default(),
            show_grid_lines: true,
            show_coordinates: true,
            coordinate_text_size: 10.0,
            show_cell_values: true,
            animation: AnimationSettings::default(),
            lod: LodSettings::default(),
        }
    }
}

/// Color scheme for grid visualization
#[derive(Debug, Clone)]
pub struct GridColorScheme {
    /// Background color
    pub background_color: [f32; 4],
    
    /// Grid line color
    pub grid_line_color: [f32; 4],
    
    /// Default cell color
    pub default_cell_color: [f32; 4],
    
    /// Wall/obstacle cell color
    pub wall_cell_color: [f32; 4],
    
    /// Start cell color
    pub start_cell_color: [f32; 4],
    
    /// Goal cell color
    pub goal_cell_color: [f32; 4],
    
    /// Path cell color
    pub path_cell_color: [f32; 4],
    
    /// Visited cell color
    pub visited_cell_color: [f32; 4],
    
    /// Frontier cell color
    pub frontier_cell_color: [f32; 4],
    
    /// Current cell color
    pub current_cell_color: [f32; 4],
    
    /// Text color
    pub text_color: [f32; 4],
}

impl Default for GridColorScheme {
    fn default() -> Self {
        Self {
            background_color: [0.1, 0.1, 0.1, 1.0],
            grid_line_color: [0.3, 0.3, 0.3, 1.0],
            default_cell_color: [0.2, 0.2, 0.2, 1.0],
            wall_cell_color: [0.5, 0.5, 0.5, 1.0],
            start_cell_color: [0.0, 0.7, 0.0, 1.0],
            goal_cell_color: [0.7, 0.0, 0.0, 1.0],
            path_cell_color: [0.7, 0.7, 0.0, 1.0],
            visited_cell_color: [0.0, 0.3, 0.6, 1.0],
            frontier_cell_color: [0.0, 0.6, 0.3, 1.0],
            current_cell_color: [0.9, 0.5, 0.0, 1.0],
            text_color: [0.9, 0.9, 0.9, 1.0],
        }
    }
}

/// Animation settings for grid transitions
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    /// Enable animations
    pub enable_animations: bool,
    
    /// Animation duration in seconds
    pub animation_duration: f32,
    
    /// Animation easing function
    pub easing_function: EasingFunction,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enable_animations: true,
            animation_duration: 0.3,
            easing_function: EasingFunction::CubicInOut,
        }
    }
}

/// Easing functions for animations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// Level of detail settings for grid rendering
#[derive(Debug, Clone)]
pub struct LodSettings {
    /// Enable level of detail
    pub enable_lod: bool,
    
    /// Maximum number of cells to render
    pub max_cells: usize,
    
    /// Scale thresholds for different detail levels
    pub scale_thresholds: [f32; 3],
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            enable_lod: true,
            max_cells: 10000,
            scale_thresholds: [0.2, 0.5, 1.0],
        }
    }
}

/// Cell instance data for GPU rendering
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct CellInstance {
    /// Position (x, y, width, height)
    position: [f32; 4],
    
    /// Color (r, g, b, a)
    color: [f32; 4],
    
    /// Cell state (as integer)
    state: u32,
    
    /// Cell flags (bitfield)
    flags: u32,
    
    /// Animation progress (0.0-1.0)
    animation: f32,
    
    /// Padding (for 16-byte alignment)
    _padding: [f32; 1],
}

/// Grid view uniform buffer data
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GridViewUniforms {
    /// Projection matrix
    projection: [f32; 16],
    
    /// View matrix
    view: [f32; 16],
    
    /// Grid parameters (cell_size, line_thickness, time, _padding)
    params: [f32; 4],
    
    /// Viewport size (width, height, aspect_ratio, _padding)
    viewport: [f32; 4],
    
    /// Grid line color
    grid_line_color: [f32; 4],
    
    /// Background color
    background_color: [f32; 4],
}

/// Grid visualization view
pub struct GridView {
    /// View configuration
    config: GridViewConfig,
    
    /// Grid data
    grid: Option<Arc<Grid>>,
    
    /// Current algorithm state
    state: Option<Arc<AlgorithmState>>,
    
    /// Rendering resources
    resources: Option<GridViewResources>,
    
    /// View bounds
    bounds: ViewBounds,
    
    /// Animation state
    animation: AnimationState,
    
    /// Grid state representation
    grid_state: GridStateRepresentation,
    
    /// Cell instance data
    cell_instances: Vec<CellInstance>,
    
    /// Level of detail state
    lod_state: LodState,
}

/// Grid view rendering resources
struct GridViewResources {
    /// Render pipeline for cells
    cell_pipeline: RenderPipeline,
    
    /// Render pipeline for grid lines
    grid_line_pipeline: Option<RenderPipeline>,
    
    /// Render pipeline for text
    text_pipeline: Option<RenderPipeline>,
    
    /// Vertex buffer for cell quad
    quad_vertex_buffer: Buffer,
    
    /// Index buffer for cell quad
    quad_index_buffer: Buffer,
    
    /// Instance buffer for cells
    instance_buffer: Buffer,
    
    /// Uniform buffer for view parameters
    uniform_buffer: UniformBuffer<GridViewUniforms>,
    
    /// Font texture for text rendering
    font_texture: Option<Texture>,
    
    /// Main bind group for rendering
    bind_group: BindGroup,
    
    /// Text rendering bind group
    text_bind_group: Option<BindGroup>,
}

/// Animation state for grid view
#[derive(Debug, Default)]
struct AnimationState {
    /// Animation timer
    timer: f32,
    
    /// Previous frame time
    last_frame_time: f32,
    
    /// Cell state animations
    cell_animations: Vec<CellAnimation>,
}

/// Cell animation data
#[derive(Debug, Clone)]
struct CellAnimation {
    /// Cell position
    position: (usize, usize),
    
    /// Start time
    start_time: f32,
    
    /// Duration
    duration: f32,
    
    /// Start state
    start_state: CellState,
    
    /// End state
    end_state: CellState,
}

/// Grid state representation for visualization
#[derive(Debug, Default)]
struct GridStateRepresentation {
    /// Start position
    start_pos: Option<(usize, usize)>,
    
    /// Goal position
    goal_pos: Option<(usize, usize)>,
    
    /// Current position
    current_pos: Option<(usize, usize)>,
    
    /// Visited cells
    visited: Vec<(usize, usize)>,
    
    /// Frontier cells
    frontier: Vec<(usize, usize)>,
    
    /// Path cells
    path: Vec<(usize, usize)>,
    
    /// Cell values
    values: Vec<((usize, usize), String)>,
}

/// Level of detail state
#[derive(Debug, Default)]
struct LodState {
    /// Current detail level (0-3)
    current_level: usize,
    
    /// Visible cell range
    visible_range: ((usize, usize), (usize, usize)),
}

impl GridView {
    /// Create a new grid view
    pub fn new(config: GridViewConfig) -> Self {
        Self {
            config,
            grid: None,
            state: None,
            resources: None,
            bounds: ViewBounds::default(),
            animation: AnimationState::default(),
            grid_state: GridStateRepresentation::default(),
            cell_instances: Vec::new(),
            lod_state: LodState::default(),
        }
    }
    
    /// Set grid data
    pub fn set_grid(&mut self, grid: Arc<Grid>) {
        self.grid = Some(grid);
        self.update_grid_state();
    }
    
    /// Set algorithm state
    pub fn set_state(&mut self, state: Arc<AlgorithmState>) {
        self.state = Some(state);
        self.update_grid_state();
    }
    
    /// Initialize rendering resources
    fn initialize_resources(&mut self, platform: &RenderingPlatform, shader_manager: &mut ShaderManager) -> Result<(), ViewError> {
        let device = platform.device();
        let queue = platform.queue();
        
        // Create cell quad geometry
        let quad_vertices = [
            Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [0.5, -0.5, 0.0], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [0.5, 0.5, 0.0], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
            Vertex { position: [-0.5, 0.5, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
        ];
        
        let quad_indices = [
            0, 1, 2,
            2, 3, 0,
        ];
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Quad Index Buffer"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        // Create instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Instance Buffer"),
            size: (std::mem::size_of::<CellInstance>() * 10000) as u64, // Initial capacity
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create uniform buffer
        let uniforms = GridViewUniforms {
            projection: Mat4::IDENTITY.to_cols_array(),
            view: Mat4::IDENTITY.to_cols_array(),
            params: [self.config.cell_size, self.config.line_thickness, 0.0, 0.0],
            viewport: [
                self.bounds.width as f32,
                self.bounds.height as f32,
                self.bounds.width as f32 / self.bounds.height as f32,
                0.0,
            ],
            grid_line_color: self.config.color_scheme.grid_line_color,
            background_color: self.config.color_scheme.background_color,
        };
        
        let uniform_buffer = UniformBuffer::new(device.clone(), "Grid View Uniforms", uniforms);
        
        // Get cell shader
        let cell_vertex_shader = shader_manager.get_shader(&ShaderVariant::new("grid_cell", ShaderStage::Vertex))?;
        let cell_fragment_shader = shader_manager.get_shader(&ShaderVariant::new("grid_cell", ShaderStage::Fragment))?;
        
        // Create cell pipeline
        let vertex_buffers = [
            // Vertex buffer layout
            VertexLayout::vertex_buffer_layout(),
            
            // Instance buffer layout
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CellInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // position
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    // color
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 4]>() as u64,
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    // state
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 8]>() as u64,
                        shader_location: 7,
                        format: wgpu::VertexFormat::Uint32,
                    },
                    // flags
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 8]>() as u64 + std::mem::size_of::<u32>() as u64,
                        shader_location: 8,
                        format: wgpu::VertexFormat::Uint32,
                    },
                    // animation
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 8]>() as u64 + std::mem::size_of::<[u32; 2]>() as u64,
                        shader_location: 9,
                        format: wgpu::VertexFormat::Float32,
                    },
                ],
            },
        ];
        
        let cell_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Cell Pipeline Layout"),
            bind_group_layouts: &[uniform_buffer.bind_group_layout()],
            push_constant_ranges: &[],
        });
        
        let cell_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Cell Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cell_vertex_shader.module(),
                entry_point: cell_vertex_shader.entry_point(ShaderStage::Vertex).unwrap_or("vs_main"),
                buffers: &vertex_buffers,
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
                module: &cell_fragment_shader.module(),
                entry_point: cell_fragment_shader.entry_point(ShaderStage::Fragment).unwrap_or("fs_main"),
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
        
        // TODO: Create grid line pipeline if show_grid_lines is true
        let grid_line_pipeline = None;
        
        // TODO: Create text pipeline if show_coordinates or show_cell_values is true
        let text_pipeline = None;
        let font_texture = None;
        let text_bind_group = None;
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid View Bind Group"),
            layout: uniform_buffer.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.binding_resource(),
                },
            ],
        });
        
        let resources = GridViewResources {
            cell_pipeline,
            grid_line_pipeline,
            text_pipeline,
            quad_vertex_buffer: vertex_buffer,
            quad_index_buffer: index_buffer,
            instance_buffer,
            uniform_buffer,
            font_texture,
            bind_group,
            text_bind_group,
        };
        
        self.resources = Some(resources);
        
        // Initialize cell instances
        self.update_cell_instances(device, queue)?;
        
        Ok(())
    }
    
    /// Update grid state representation based on grid and algorithm state
    fn update_grid_state(&mut self) {
        self.grid_state = GridStateRepresentation::default();
        
        if let Some(grid) = &self.grid {
            // Extract basic grid information
            // In a real implementation, this would be more sophisticated
            // and extract information from the algorithm state
            
            if let Some(state) = &self.state {
                // Extract information from algorithm state
                if let Some(start_str) = state.data.get("start") {
                    if let Some((x, y)) = Self::parse_position(start_str) {
                        self.grid_state.start_pos = Some((x, y));
                    }
                }
                
                if let Some(goal_str) = state.data.get("goal") {
                    if let Some((x, y)) = Self::parse_position(goal_str) {
                        self.grid_state.goal_pos = Some((x, y));
                    }
                }
                
                if let Some(current_str) = state.data.get("current") {
                    if let Some((x, y)) = Self::parse_position(current_str) {
                        self.grid_state.current_pos = Some((x, y));
                    }
                }
                
                // Extract visited cells
                if let Some(visited_str) = state.data.get("visited") {
                    self.grid_state.visited = Self::parse_position_list(visited_str);
                }
                
                // Extract frontier cells
                if let Some(frontier_str) = state.data.get("frontier") {
                    self.grid_state.frontier = Self::parse_position_list(frontier_str);
                }
                
                // Extract path cells
                if let Some(path_str) = state.data.get("path") {
                    self.grid_state.path = Self::parse_position_list(path_str);
                }
                
                // Extract cell values
                // In a real implementation, this would extract values from state
                self.grid_state.values = Vec::new();
            }
            
            // Update cell instances
            if let Some(resources) = &mut self.resources {
                let device = resources.uniform_buffer.device();
                let queue = device.create_queue();
                
                if let Err(e) = self.update_cell_instances(device, &queue) {
                    log::error!("Failed to update cell instances: {}", e);
                }
            }
        }
    }
    
    /// Update cell instances for rendering
    fn update_cell_instances(&mut self, device: &Device, queue: &Queue) -> Result<(), ViewError> {
        self.cell_instances.clear();
        
        if let Some(grid) = &self.grid {
            let width = grid.width();
            let height = grid.height();
            
            // Calculate visible range based on view bounds and zoom
            self.calculate_visible_range(width, height);
            let ((min_x, min_y), (max_x, max_y)) = self.lod_state.visible_range;
            
            // Generate cell instances
            for y in min_y..max_y.min(height) {
                for x in min_x..max_x.min(width) {
                    let cell_state = grid.get_cell(x, y).unwrap_or_default();
                    let pos_x = x as f32 * self.config.cell_size;
                    let pos_y = y as f32 * self.config.cell_size;
                    
                    let color = self.get_cell_color(x, y, cell_state);
                    let flags = self.get_cell_flags(x, y);
                    
                    let instance = CellInstance {
                        position: [pos_x, pos_y, self.config.cell_size, self.config.cell_size],
                        color,
                        state: cell_state as u32,
                        flags,
                        animation: self.get_cell_animation_progress(x, y),
                        _padding: [0.0],
                    };
                    
                    self.cell_instances.push(instance);
                }
            }
            
            // Update instance buffer if necessary
            if !self.cell_instances.is_empty() {
                let instances_size = (std::mem::size_of::<CellInstance>() * self.cell_instances.len()) as u64;
                
                if let Some(resources) = &mut self.resources {
                    // Check if buffer needs resizing
                    if instances_size > resources.instance_buffer.size() {
                        resources.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Grid Instance Buffer"),
                            size: instances_size.max(10000), // Ensure at least 10000 capacity
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                    }
                    
                    // Update buffer contents
                    queue.write_buffer(
                        &resources.instance_buffer,
                        0,
                        bytemuck::cast_slice(&self.cell_instances),
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate visible cell range based on view bounds
    fn calculate_visible_range(&mut self, grid_width: usize, grid_height: usize) {
        // Calculate cell size in screen space
        let cell_size_screen = self.config.cell_size * self.bounds.scale;
        
        // Calculate visible range
        let half_width = (self.bounds.width as f32) / 2.0;
        let half_height = (self.bounds.height as f32) / 2.0;
        
        let center_x = -self.bounds.offset_x / cell_size_screen;
        let center_y = -self.bounds.offset_y / cell_size_screen;
        
        let visible_width = (half_width / cell_size_screen).ceil() as usize + 1;
        let visible_height = (half_height / cell_size_screen).ceil() as usize + 1;
        
        let min_x = ((center_x - visible_width as f32).floor() as isize).max(0) as usize;
        let min_y = ((center_y - visible_height as f32).floor() as isize).max(0) as usize;
        let max_x = ((center_x + visible_width as f32).ceil() as usize).min(grid_width);
        let max_y = ((center_y + visible_height as f32).ceil() as usize).min(grid_height);
        
        self.lod_state.visible_range = ((min_x, min_y), (max_x, max_y));
        
        // Determine LOD level based on zoom
        if self.config.lod.enable_lod {
            let cell_count = (max_x - min_x) * (max_y - min_y);
            
            if cell_count > self.config.lod.max_cells {
                // Reduce detail level
                let ratio = (self.config.lod.max_cells as f32 / cell_count as f32).sqrt();
                let step = (1.0 / ratio).ceil() as usize;
                
                // Adjust visible range to use stride
                let new_min_x = min_x;
                let new_min_y = min_y;
                let new_max_x = min_x + ((max_x - min_x + step - 1) / step) * step;
                let new_max_y = min_y + ((max_y - min_y + step - 1) / step) * step;
                
                self.lod_state.visible_range = ((new_min_x, new_min_y), (new_max_x, new_max_y));
                self.lod_state.current_level = if self.bounds.scale < self.config.lod.scale_thresholds[0] {
                    3 // Very low detail
                } else if self.bounds.scale < self.config.lod.scale_thresholds[1] {
                    2 // Low detail
                } else if self.bounds.scale < self.config.lod.scale_thresholds[2] {
                    1 // Medium detail
                } else {
                    0 // Full detail
                };
            } else {
                self.lod_state.current_level = 0; // Full detail
            }
        }
    }
    
    /// Get cell color based on state and grid state
    fn get_cell_color(&self, x: usize, y: usize, cell_state: CellState) -> [f32; 4] {
        // Check if cell has special state in algorithm
        if let Some(current_pos) = self.grid_state.current_pos {
            if current_pos == (x, y) {
                return self.config.color_scheme.current_cell_color;
            }
        }
        
        if let Some(start_pos) = self.grid_state.start_pos {
            if start_pos == (x, y) {
                return self.config.color_scheme.start_cell_color;
            }
        }
        
        if let Some(goal_pos) = self.grid_state.goal_pos {
            if goal_pos == (x, y) {
                return self.config.color_scheme.goal_cell_color;
            }
        }
        
        if self.grid_state.path.contains(&(x, y)) {
            return self.config.color_scheme.path_cell_color;
        }
        
        if self.grid_state.frontier.contains(&(x, y)) {
            return self.config.color_scheme.frontier_cell_color;
        }
        
        if self.grid_state.visited.contains(&(x, y)) {
            return self.config.color_scheme.visited_cell_color;
        }
        
        // Use cell state for color
        match cell_state {
            CellState::Empty => self.config.color_scheme.default_cell_color,
            CellState::Wall => self.config.color_scheme.wall_cell_color,
            CellState::Start => self.config.color_scheme.start_cell_color,
            CellState::Goal => self.config.color_scheme.goal_cell_color,
            _ => self.config.color_scheme.default_cell_color,
        }
    }
    
    /// Get cell flags
    fn get_cell_flags(&self, x: usize, y: usize) -> u32 {
        let mut flags = 0;
        
        // Set flags based on cell state
        if self.grid_state.start_pos == Some((x, y)) {
            flags |= 0x01; // Start flag
        }
        
        if self.grid_state.goal_pos == Some((x, y)) {
            flags |= 0x02; // Goal flag
        }
        
        if self.grid_state.current_pos == Some((x, y)) {
            flags |= 0x04; // Current flag
        }
        
        if self.grid_state.path.contains(&(x, y)) {
            flags |= 0x08; // Path flag
        }
        
        if self.grid_state.frontier.contains(&(x, y)) {
            flags |= 0x10; // Frontier flag
        }
        
        if self.grid_state.visited.contains(&(x, y)) {
            flags |= 0x20; // Visited flag
        }
        
        if self.config.show_coordinates {
            flags |= 0x40; // Show coordinates flag
        }
        
        if self.grid_state.values.iter().any(|&((cx, cy), _)| cx == x && cy == y) {
            flags |= 0x80; // Has value flag
        }
        
        flags
    }
    
    /// Get cell animation progress
    fn get_cell_animation_progress(&self, x: usize, y: usize) -> f32 {
        if !self.config.animation.enable_animations {
            return 1.0;
        }
        
        // Find animation for this cell
        for anim in &self.animation.cell_animations {
            if anim.position == (x, y) {
                let elapsed = self.animation.timer - anim.start_time;
                if elapsed < 0.0 {
                    return 0.0;
                } else if elapsed >= anim.duration {
                    return 1.0;
                } else {
                    return elapsed / anim.duration;
                }
            }
        }
        
        1.0
    }
    
    /// Parse position from string (x,y)
    fn parse_position(pos_str: &str) -> Option<(usize, usize)> {
        let pos_str = pos_str.trim();
        if pos_str.len() < 5 || !pos_str.starts_with('(') || !pos_str.ends_with(')') {
            return None;
        }
        
        let pos_str = &pos_str[1..pos_str.len()-1];
        let parts: Vec<&str> = pos_str.split(',').collect();
        if parts.len() != 2 {
            return None;
        }
        
        let x = parts[0].trim().parse::<usize>().ok()?;
        let y = parts[1].trim().parse::<usize>().ok()?;
        
        Some((x, y))
    }
    
    /// Parse list of positions from string [(x1,y1), (x2,y2), ...]
    fn parse_position_list(list_str: &str) -> Vec<(usize, usize)> {
        let list_str = list_str.trim();
        if list_str.len() < 2 || !list_str.starts_with('[') || !list_str.ends_with(']') {
            return Vec::new();
        }
        
        let list_str = &list_str[1..list_str.len()-1];
        let parts: Vec<&str> = list_str.split("),").collect();
        
        let mut positions = Vec::new();
        for part in parts {
            let part = if part.ends_with(')') { part } else { format!("{})", part).as_str() };
            if let Some(pos) = Self::parse_position(part) {
                positions.push(pos);
            }
        }
        
        positions
    }
    
    /// Update view matrices
    fn update_view_matrices(&mut self) {
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
            uniforms.params[0] = self.config.cell_size;
            uniforms.params[1] = self.config.line_thickness;
            uniforms.params[2] = self.animation.timer;
            uniforms.viewport = [
                width,
                height,
                width / height,
                0.0,
            ];
            uniforms.grid_line_color = self.config.color_scheme.grid_line_color;
            uniforms.background_color = self.config.color_scheme.background_color;
            
            resources.uniform_buffer.update(uniforms);
        }
    }
}

impl View for GridView {
    fn name(&self) -> &str {
        "Grid View"
    }
    
    fn initialize(&mut self, platform: &RenderingPlatform, shader_manager: &mut ShaderManager) -> Result<(), ViewError> {
        self.initialize_resources(platform, shader_manager)
    }
    
    fn update(&mut self, dt: f32) {
        // Update animation timer
        self.animation.timer += dt;
        self.animation.last_frame_time = dt;
        
        // Remove completed animations
        self.animation.cell_animations.retain(|anim| {
            self.animation.timer - anim.start_time < anim.duration
        });
        
        // Update view matrices
        self.update_view_matrices();
    }
    
    fn render<'a>(&'a self, render_pass: &mut RenderPass<'a>) -> Result<(), ViewError> {
        if let Some(resources) = &self.resources {
            // Set the cell pipeline
            render_pass.set_pipeline(&resources.cell_pipeline);
            render_pass.set_bind_group(0, &resources.bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, resources.instance_buffer.slice(0..std::mem::size_of::<CellInstance>() as u64 * self.cell_instances.len() as u64));
            render_pass.set_index_buffer(resources.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            // Draw cells
            render_pass.draw_indexed(0..6, 0, 0..self.cell_instances.len() as u32);
            
            // TODO: Draw grid lines if enabled
            
            // TODO: Draw text if enabled
        }
        
        Ok(())
    }
    
    fn resize(&mut self, width: u32, height: u32) {
        self.bounds.width = width;
        self.bounds.height = height;
        
        // Update view matrices
        self.update_view_matrices();
    }
    
    fn handle_input(&mut self, event: &InputEvent) -> bool {
        match event {
            InputEvent::Pan { delta_x, delta_y } => {
                self.bounds.offset_x += *delta_x;
                self.bounds.offset_y += *delta_y;
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
                
                // Update cell instances due to LOD changes
                if let Some(grid) = &self.grid {
                    if let Some(resources) = &mut self.resources {
                        let device = resources.uniform_buffer.device();
                        let queue = device.create_queue();
                        
                        if let Err(e) = self.update_cell_instances(device, &queue) {
                            log::error!("Failed to update cell instances after zoom: {}", e);
                        }
                    }
                }
                
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

impl Drop for GridView {
    fn drop(&mut self) {
        // Clean up resources if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_position() {
        assert_eq!(GridView::parse_position("(0,0)"), Some((0, 0)));
        assert_eq!(GridView::parse_position("(10, 20)"), Some((10, 20)));
        assert_eq!(GridView::parse_position("(10,20)"), Some((10, 20)));
        assert_eq!(GridView::parse_position("( 10 , 20 )"), Some((10, 20)));
        assert_eq!(GridView::parse_position("10,20"), None);
        assert_eq!(GridView::parse_position("(10)"), None);
        assert_eq!(GridView::parse_position("(10,20,30)"), None);
        assert_eq!(GridView::parse_position("(a,b)"), None);
    }
    
    #[test]
    fn test_parse_position_list() {
        assert_eq!(GridView::parse_position_list("[(0,0), (1,1)]"), vec![(0, 0), (1, 1)]);
        assert_eq!(GridView::parse_position_list("[(0,0),(1,1),(2,2)]"), vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(GridView::parse_position_list("[]"), Vec::<(usize, usize)>::new());
        assert_eq!(GridView::parse_position_list("[(0,0)"), Vec::<(usize, usize)>::new());
        assert_eq!(GridView::parse_position_list("(0,0), (1,1)"), Vec::<(usize, usize)>::new());
    }
}