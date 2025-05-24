//! Decision Perspective Visualization
//!
//! This module implements the decision perspective visualization system, which renders
//! algorithm decision-making processes with information-theoretic importance metrics
//! and interactive counterfactual exploration capabilities.
//!
//! The implementation utilizes:
//! - Information-theoretic importance quantification for decision points
//! - Focus+context visualization with semantic zooming
//! - GPU-accelerated tree layout and rendering algorithms
//! - Multi-resolution aggregation for perceptual optimization
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>
//! All rights reserved.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use log::{debug, trace};
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::engine::{
    backend::RenderBackend,
    platform::PlatformCapabilities,
    wgpu::{
        RenderContext, 
        ShaderManager,
        TextureManager,
        BufferManager,
        DrawCommand
    },
    primitives::{
        RenderPipeline,
        VertexBuffer,
        IndexBuffer,
        UniformBuffer
    }
};

use crate::interaction::{
    controller::{InteractionController, InteractionEvent, InteractionResult},
    camera::Camera,
    selection::SelectionManager
};

use chronos_core::temporal::{
    decision::{DecisionPoint, DecisionImportance},
    branch::{BranchId, BranchManager, BranchCreationError},
    timeline::{Timeline, TimelineNavigationError}
};

use chronos_core::algorithm::{
    Algorithm, 
    AlgorithmState,
    AlgorithmError
};

/// Resolution levels for multi-resolution visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionLevel {
    /// Full detail rendering with individual decision points
    Full,
    /// Medium detail with aggregated small decision clusters
    Medium,
    /// Low detail with only critical decision points
    Low,
    /// Overview with only major algorithmic phases
    Overview,
}

/// Focus area for the decision visualization
#[derive(Debug, Clone)]
pub struct FocusArea {
    /// Center of the focus area in normalized coordinates
    center: Vec2,
    /// Radius of the focus area in normalized coordinates
    radius: f32,
    /// Importance threshold for decision points within focus
    importance_threshold: f32,
}

/// Visual encoding for decision points
#[derive(Debug, Clone)]
pub enum DecisionEncoding {
    /// Size-based encoding for importance
    Size,
    /// Color-based encoding for importance
    Color,
    /// Combined size and color encoding
    SizeAndColor,
    /// Opacity-based encoding for importance
    Opacity,
}

/// Transition animation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionType {
    /// Linear interpolation
    Linear,
    /// Exponential ease
    Exponential,
    /// Elastic movement
    Elastic,
    /// Bounce effect
    Bounce,
}

/// Decision tree layout algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeLayoutAlgorithm {
    /// Horizontal tree layout
    HorizontalTree,
    /// Radial tree layout
    RadialTree,
    /// Force-directed layout
    ForceDirected,
    /// Timeline-based layout
    Timeline,
}

/// Configuration for decision visualization
#[derive(Debug, Clone)]
pub struct DecisionVisualizationConfig {
    /// Tree layout algorithm
    layout_algorithm: TreeLayoutAlgorithm,
    /// Resolution level for detail
    resolution: ResolutionLevel,
    /// Visual encoding for decision importance
    encoding: DecisionEncoding,
    /// Focus area for detailed visualization
    focus: Option<FocusArea>,
    /// Transition animation type
    transition: TransitionType,
    /// Whether to show aggregated nodes
    show_aggregates: bool,
    /// Importance threshold for visible decisions
    importance_threshold: f32,
    /// Maximum number of visible decisions
    max_visible_decisions: usize,
}

impl Default for DecisionVisualizationConfig {
    fn default() -> Self {
        Self {
            layout_algorithm: TreeLayoutAlgorithm::HorizontalTree,
            resolution: ResolutionLevel::Medium,
            encoding: DecisionEncoding::SizeAndColor,
            focus: None,
            transition: TransitionType::Exponential,
            show_aggregates: true,
            importance_threshold: 0.05,
            max_visible_decisions: 1000,
        }
    }
}

/// Visual representation of a decision point
#[derive(Debug, Clone)]
struct DecisionNode {
    /// Unique identifier for the decision
    id: u64,
    /// Position in visualization space
    position: Vec3,
    /// Target position for animation
    target_position: Vec3,
    /// Size of the node based on importance
    size: f32,
    /// Color of the node based on type or importance
    color: Vec4,
    /// Whether the node is currently selected
    selected: bool,
    /// Whether the node is highlighted
    highlighted: bool,
    /// Opacity of the node
    opacity: f32,
    /// Parent decision ID, if any
    parent: Option<u64>,
    /// Child decision IDs
    children: Vec<u64>,
    /// Reference to the actual decision point
    decision_point: Arc<DecisionPoint>,
    /// Aggregation info if this node represents multiple decisions
    aggregation: Option<AggregationInfo>,
}

/// Information about aggregated decision nodes
#[derive(Debug, Clone)]
struct AggregationInfo {
    /// Number of decisions aggregated
    count: usize,
    /// Average importance of aggregated decisions
    average_importance: f32,
    /// Maximum importance among aggregated decisions
    max_importance: f32,
    /// IDs of aggregated decisions
    decision_ids: Vec<u64>,
    /// Bounding box of the aggregated area
    bounding_box: (Vec3, Vec3),
}

/// Decision tree visualization state
#[derive(Debug)]
struct DecisionTreeState {
    /// Mapping from decision ID to node
    nodes: HashMap<u64, DecisionNode>,
    /// Root decision node ID
    root_id: Option<u64>,
    /// Currently selected node IDs
    selected_ids: HashSet<u64>,
    /// Currently highlighted node IDs
    highlighted_ids: HashSet<u64>,
    /// Animation progress (0.0 - 1.0)
    animation_progress: f32,
    /// Whether an animation is in progress
    animating: bool,
    /// Current level of detail
    current_resolution: ResolutionLevel,
    /// Viewport center
    viewport_center: Vec2,
    /// Viewport zoom level
    viewport_zoom: f32,
}

/// Decision visualization component
pub struct DecisionVisualization {
    /// Configuration for the visualization
    config: DecisionVisualizationConfig,
    /// Current state of the decision tree
    tree_state: DecisionTreeState,
    /// Render pipeline for decision nodes
    node_pipeline: Option<RenderPipeline>,
    /// Render pipeline for decision edges
    edge_pipeline: Option<RenderPipeline>,
    /// Vertex buffer for nodes
    node_vertices: Option<VertexBuffer>,
    /// Index buffer for nodes
    node_indices: Option<IndexBuffer>,
    /// Vertex buffer for edges
    edge_vertices: Option<VertexBuffer>,
    /// Index buffer for edges
    edge_indices: Option<IndexBuffer>,
    /// Uniform buffer for transformation matrices
    transform_buffer: Option<UniformBuffer>,
    /// Uniform buffer for visualization settings
    settings_buffer: Option<UniformBuffer>,
    /// Timeline for algorithm execution
    timeline: Arc<Timeline>,
    /// Branch manager for counterfactual exploration
    branch_manager: Arc<BranchManager>,
    /// Selection manager for coordinated selection
    selection_manager: Arc<SelectionManager>,
    /// Camera for view control
    camera: Camera,
    /// Last update timestamp
    last_update: std::time::Instant,
    /// Render ready status
    render_ready: bool,
    /// Layout computation status
    layout_computed: bool,
}

impl DecisionVisualization {
    /// Create a new decision visualization
    pub fn new(
        timeline: Arc<Timeline>,
        branch_manager: Arc<BranchManager>,
        selection_manager: Arc<SelectionManager>,
        config: DecisionVisualizationConfig,
    ) -> Self {
        Self {
            config,
            tree_state: DecisionTreeState {
                nodes: HashMap::new(),
                root_id: None,
                selected_ids: HashSet::new(),
                highlighted_ids: HashSet::new(),
                animation_progress: 1.0,
                animating: false,
                current_resolution: ResolutionLevel::Medium,
                viewport_center: Vec2::ZERO,
                viewport_zoom: 1.0,
            },
            node_pipeline: None,
            edge_pipeline: None,
            node_vertices: None,
            node_indices: None,
            edge_vertices: None,
            edge_indices: None,
            transform_buffer: None,
            settings_buffer: None,
            timeline,
            branch_manager,
            selection_manager,
            camera: Camera::new(),
            last_update: std::time::Instant::now(),
            render_ready: false,
            layout_computed: false,
        }
    }

    /// Initialize rendering resources
    pub fn initialize_rendering(&mut self, context: &mut RenderContext) -> Result<(), String> {
        debug!("Initializing decision visualization rendering resources");
        
        // Create shader modules
        let shader_manager = &mut context.shader_manager;
        let node_shader = shader_manager.load_shader("decision_node", include_str!("../shaders/decision_node.wgsl"))?;
        let edge_shader = shader_manager.load_shader("decision_edge", include_str!("../shaders/decision_edge.wgsl"))?;
        
        // Create render pipelines
        let device = &context.device;
        let format = context.surface_format;

        // Node pipeline configuration
        let node_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decision Node Pipeline Layout"),
            bind_group_layouts: &[
                &context.buffer_manager.create_bind_group_layout(device, 0, true),
                &context.buffer_manager.create_bind_group_layout(device, 1, false),
            ],
            push_constant_ranges: &[],
        });

        let node_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Decision Node Pipeline"),
            layout: Some(&node_pipeline_layout),
            vertex: wgpu::VertexState {
                module: node_shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 36, // position(12) + normal(12) + uv(8) + color(4)
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // normal
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // uv
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Unorm8x4,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: node_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };

        // Edge pipeline configuration (similar to node but with different topology and shader)
        let edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decision Edge Pipeline Layout"),
            bind_group_layouts: &[
                &context.buffer_manager.create_bind_group_layout(device, 0, true),
                &context.buffer_manager.create_bind_group_layout(device, 1, false),
            ],
            push_constant_ranges: &[],
        });

        let edge_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Decision Edge Pipeline"),
            layout: Some(&edge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: edge_shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 28, // position(12) + color(4) + thickness(4) + data(8)
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Unorm8x4,
                            },
                            // thickness
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // data
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: edge_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };

        // Create pipelines
        let node_pipeline = device.create_render_pipeline(&node_pipeline_descriptor);
        let edge_pipeline = device.create_render_pipeline(&edge_pipeline_descriptor);

        // Create uniform buffers
        let transform_buffer = context.buffer_manager.create_uniform_buffer(
            device,
            &[Mat4::IDENTITY.to_cols_array_2d()[0], 
              Mat4::IDENTITY.to_cols_array_2d()[1], 
              Mat4::IDENTITY.to_cols_array_2d()[2], 
              Mat4::IDENTITY.to_cols_array_2d()[3]],
            "Decision Transform Buffer",
        )?;

        let settings_data = [
            self.config.importance_threshold,
            self.tree_state.animation_progress,
            self.config.max_visible_decisions as f32,
            self.tree_state.viewport_zoom,
        ];

        let settings_buffer = context.buffer_manager.create_uniform_buffer(
            device,
            &settings_data,
            "Decision Settings Buffer",
        )?;

        // Store resources
        self.node_pipeline = Some(RenderPipeline::new(node_pipeline));
        self.edge_pipeline = Some(RenderPipeline::new(edge_pipeline));
        self.transform_buffer = Some(transform_buffer);
        self.settings_buffer = Some(settings_buffer);

        // Allocate initial empty buffers
        self.allocate_empty_buffers(context)?;

        self.render_ready = true;
        Ok(())
    }

    /// Allocate empty vertex and index buffers
    fn allocate_empty_buffers(&mut self, context: &mut RenderContext) -> Result<(), String> {
        let device = &context.device;
        
        // Create empty node vertex buffer
        let node_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decision Node Vertex Buffer"),
            contents: &[0u8; 36], // One empty vertex
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create empty node index buffer
        let node_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decision Node Index Buffer"),
            contents: &[0u8, 0u8, 0u8, 0u8], // One empty index
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create empty edge vertex buffer
        let edge_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decision Edge Vertex Buffer"),
            contents: &[0u8; 28], // One empty vertex
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create empty edge index buffer
        let edge_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Decision Edge Index Buffer"),
            contents: &[0u8, 0u8, 0u8, 0u8], // One empty index
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        // Store buffers
        self.node_vertices = Some(VertexBuffer::new(node_vertices, 1));
        self.node_indices = Some(IndexBuffer::new(node_indices, 1));
        self.edge_vertices = Some(VertexBuffer::new(edge_vertices, 1));
        self.edge_indices = Some(IndexBuffer::new(edge_indices, 1));

        Ok(())
    }

    /// Update decision tree from timeline
    pub fn update_from_timeline(&mut self) -> Result<(), TimelineNavigationError> {
        debug!("Updating decision tree from timeline");
        
        // Get current state from timeline
        let current_state = self.timeline.current_state()?;
        let decisions = self.timeline.decision_points()?;
        
        // Clear existing nodes
        self.tree_state.nodes.clear();
        
        // Process decision points
        let mut next_id = 0;
        let mut root_id = None;
        
        for decision in &decisions {
            let id = next_id;
            next_id += 1;
            
            // Create decision node
            let importance = decision.importance();
            let node = DecisionNode {
                id,
                position: Vec3::ZERO,
                target_position: Vec3::ZERO,
                size: self.calculate_node_size(importance),
                color: self.calculate_node_color(decision),
                selected: false,
                highlighted: false,
                opacity: 1.0,
                parent: None, // Will set later
                children: Vec::new(),
                decision_point: Arc::clone(decision),
                aggregation: None,
            };
            
            // Track root
            if decision.is_root() {
                root_id = Some(id);
            }
            
            // Add to nodes map
            self.tree_state.nodes.insert(id, node);
        }
        
        // Build parent-child relationships
        for decision in &decisions {
            if let Some(parent_decision) = decision.parent() {
                // Find the nodes by decision point
                let parent_id = self.find_node_by_decision(parent_decision);
                let child_id = self.find_node_by_decision(decision);
                
                if let (Some(parent_id), Some(child_id)) = (parent_id, child_id) {
                    // Update parent reference
                    if let Some(child_node) = self.tree_state.nodes.get_mut(&child_id) {
                        child_node.parent = Some(parent_id);
                    }
                    
                    // Update children list
                    if let Some(parent_node) = self.tree_state.nodes.get_mut(&parent_id) {
                        parent_node.children.push(child_id);
                    }
                }
            }
        }
        
        // Set root ID
        self.tree_state.root_id = root_id;
        
        // Mark layout as needing computation
        self.layout_computed = false;
        
        // Start animation
        self.tree_state.animation_progress = 0.0;
        self.tree_state.animating = true;
        
        Ok(())
    }

    /// Find node ID by decision point
    fn find_node_by_decision(&self, decision: &Arc<DecisionPoint>) -> Option<u64> {
        for (id, node) in &self.tree_state.nodes {
            if Arc::ptr_eq(&node.decision_point, decision) {
                return Some(*id);
            }
        }
        None
    }

    /// Calculate node size based on importance
    fn calculate_node_size(&self, importance: DecisionImportance) -> f32 {
        // Map importance (0.0-1.0) to size (0.5-2.0)
        0.5 + (importance.value() * 1.5)
    }

    /// Calculate node color based on decision type and importance
    fn calculate_node_color(&self, decision: &Arc<DecisionPoint>) -> Vec4 {
        let importance = decision.importance().value();
        
        // Base color depends on decision type
        let base_color = match decision.category() {
            "branching" => Vec4::new(0.2, 0.6, 1.0, 1.0),
            "heuristic" => Vec4::new(0.8, 0.4, 0.0, 1.0),
            "termination" => Vec4::new(0.9, 0.1, 0.1, 1.0),
            "optimization" => Vec4::new(0.1, 0.8, 0.2, 1.0),
            _ => Vec4::new(0.6, 0.6, 0.6, 1.0),
        };
        
        // Mix with white based on importance (higher importance = more saturated)
        let white = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let saturation = 0.5 + (importance * 0.5);
        let color = base_color * saturation + white * (1.0 - saturation);
        
        // Ensure alpha is set to 1.0
        Vec4::new(color.x, color.y, color.z, 1.0)
    }

    /// Compute the tree layout
    fn compute_layout(&mut self) {
        debug!("Computing decision tree layout");
        
        if self.tree_state.nodes.is_empty() {
            debug!("No nodes to layout");
            return;
        }
        
        match self.config.layout_algorithm {
            TreeLayoutAlgorithm::HorizontalTree => self.compute_horizontal_tree_layout(),
            TreeLayoutAlgorithm::RadialTree => self.compute_radial_tree_layout(),
            TreeLayoutAlgorithm::ForceDirected => self.compute_force_directed_layout(),
            TreeLayoutAlgorithm::Timeline => self.compute_timeline_layout(),
        }
        
        self.layout_computed = true;
    }

    /// Compute horizontal tree layout
    fn compute_horizontal_tree_layout(&mut self) {
        // Get root node
        let root_id = match self.tree_state.root_id {
            Some(id) => id,
            None => return,
        };
        
        // First pass: calculate subtree sizes
        let mut subtree_sizes = HashMap::new();
        self.calculate_subtree_sizes(root_id, &mut subtree_sizes);
        
        // Second pass: assign positions
        let mut current_y = 0.0;
        self.assign_horizontal_positions(root_id, 0.0, &mut current_y, &subtree_sizes);
    }

    /// Calculate subtree sizes for layout
    fn calculate_subtree_sizes(&self, node_id: u64, sizes: &mut HashMap<u64, f32>) -> f32 {
        let node = match self.tree_state.nodes.get(&node_id) {
            Some(node) => node,
            None => return 0.0,
        };
        
        if node.children.is_empty() {
            sizes.insert(node_id, 1.0);
            return 1.0;
        }
        
        let mut subtree_size = 0.0;
        for &child_id in &node.children {
            subtree_size += self.calculate_subtree_sizes(child_id, sizes);
        }
        
        sizes.insert(node_id, subtree_size);
        subtree_size
    }

    /// Assign positions in horizontal tree layout
    fn assign_horizontal_positions(
        &mut self,
        node_id: u64,
        x: f32,
        current_y: &mut f32,
        subtree_sizes: &HashMap<u64, f32>,
    ) {
        let node = match self.tree_state.nodes.get(&node_id) {
            Some(node) => node,
            None => return,
        };
        
        let position = Vec3::new(x, *current_y, 0.0);
        
        // Set target position for animation
        if let Some(node) = self.tree_state.nodes.get_mut(&node_id) {
            node.target_position = position;
            
            // If no animation in progress, set current position too
            if !self.tree_state.animating {
                node.position = position;
            }
        }
        
        if node.children.is_empty() {
            *current_y += 1.0;
            return;
        }
        
        let start_y = *current_y;
        for &child_id in &node.children {
            self.assign_horizontal_positions(child_id, x + 1.0, current_y, subtree_sizes);
        }
        
        let end_y = *current_y;
        let center_y = (start_y + end_y) * 0.5;
        
        // Center parent node vertically over its children
        if let Some(node) = self.tree_state.nodes.get_mut(&node_id) {
            node.target_position = Vec3::new(x, center_y, 0.0);
            
            // If no animation in progress, set current position too
            if !self.tree_state.animating {
                node.position = node.target_position;
            }
        }
    }

    /// Compute radial tree layout
    fn compute_radial_tree_layout(&mut self) {
        // Get root node
        let root_id = match self.tree_state.root_id {
            Some(id) => id,
            None => return,
        };
        
        // Calculate tree depth
        let depth = self.calculate_tree_depth(root_id);
        
        // Place root at center
        if let Some(node) = self.tree_state.nodes.get_mut(&root_id) {
            node.target_position = Vec3::ZERO;
            
            // If no animation in progress, set current position too
            if !self.tree_state.animating {
                node.position = Vec3::ZERO;
            }
        }
        
        // Assign positions radially
        self.assign_radial_positions(root_id, 0.0, std::f32::consts::TAU, 1, depth);
    }

    /// Calculate tree depth
    fn calculate_tree_depth(&self, node_id: u64) -> usize {
        let node = match self.tree_state.nodes.get(&node_id) {
            Some(node) => node,
            None => return 0,
        };
        
        if node.children.is_empty() {
            return 1;
        }
        
        let mut max_depth = 0;
        for &child_id in &node.children {
            max_depth = max_depth.max(self.calculate_tree_depth(child_id));
        }
        
        max_depth + 1
    }

    /// Assign positions in radial tree layout
    fn assign_radial_positions(
        &mut self,
        node_id: u64,
        start_angle: f32,
        angle_range: f32,
        current_depth: usize,
        max_depth: usize,
    ) {
        let node = match self.tree_state.nodes.get(&node_id) {
            Some(node) => node,
            None => return,
        };
        
        if node.children.is_empty() {
            return;
        }
        
        let radius = current_depth as f32 * 1.0;
        let num_children = node.children.len();
        
        for (i, &child_id) in node.children.iter().enumerate() {
            let child_start_angle = start_angle + (angle_range * i as f32) / num_children as f32;
            let child_angle_range = angle_range / num_children as f32;
            let child_angle = child_start_angle + (child_angle_range * 0.5);
            
            let x = radius * f32::cos(child_angle);
            let y = radius * f32::sin(child_angle);
            let position = Vec3::new(x, y, 0.0);
            
            if let Some(child) = self.tree_state.nodes.get_mut(&child_id) {
                child.target_position = position;
                
                // If no animation in progress, set current position too
                if !self.tree_state.animating {
                    child.position = position;
                }
            }
            
            self.assign_radial_positions(
                child_id,
                child_start_angle,
                child_angle_range,
                current_depth + 1,
                max_depth,
            );
        }
    }

    /// Compute force-directed layout
    fn compute_force_directed_layout(&mut self) {
        // Initialize positions randomly if needed
        for (_, node) in self.tree_state.nodes.iter_mut() {
            if node.position == Vec3::ZERO && node.target_position == Vec3::ZERO {
                let x = (rand::random::<f32>() - 0.5) * 2.0;
                let y = (rand::random::<f32>() - 0.5) * 2.0;
                node.position = Vec3::new(x, y, 0.0);
                node.target_position = node.position;
            }
        }
        
        // Run force-directed algorithm with simulated annealing
        const ITERATIONS: usize = 100;
        const INITIAL_TEMPERATURE: f32 = 1.0;
        const COOLING_FACTOR: f32 = 0.95;
        
        let mut temperature = INITIAL_TEMPERATURE;
        
        for _ in 0..ITERATIONS {
            // Calculate forces
            let mut forces = HashMap::new();
            
            // Initialize with zeros
            for &id in self.tree_state.nodes.keys() {
                forces.insert(id, Vec3::ZERO);
            }
            
            // Repulsive forces between all nodes
            for (&id1, node1) in &self.tree_state.nodes {
                for (&id2, node2) in &self.tree_state.nodes {
                    if id1 == id2 {
                        continue;
                    }
                    
                    let diff = node1.position - node2.position;
                    let distance_sq = diff.length_squared().max(0.0001); // Avoid division by zero
                    let force = diff.normalize() * (1.0 / distance_sq.sqrt());
                    
                    *forces.entry(id1).or_insert(Vec3::ZERO) += force;
                }
            }
            
            // Attractive forces between connected nodes
            for (_, node) in &self.tree_state.nodes {
                if let Some(parent_id) = node.parent {
                    if let Some(parent) = self.tree_state.nodes.get(&parent_id) {
                        let diff = parent.position - node.position;
                        let distance = diff.length();
                        let force = diff.normalize() * distance * 0.1;
                        
                        *forces.entry(node.id).or_insert(Vec3::ZERO) += force;
                        *forces.entry(parent_id).or_insert(Vec3::ZERO) -= force;
                    }
                }
            }
            
            // Apply forces with temperature
            for (&id, force) in &forces {
                if let Some(node) = self.tree_state.nodes.get_mut(&id) {
                    node.target_position += *force * temperature;
                    
                    // Clamp positions to reasonable bounds
                    let max_coord = 10.0;
                    node.target_position = Vec3::new(
                        node.target_position.x.clamp(-max_coord, max_coord),
                        node.target_position.y.clamp(-max_coord, max_coord),
                        0.0,
                    );
                }
            }
            
            // Cool down
            temperature *= COOLING_FACTOR;
        }
    }

    /// Compute timeline-based layout
    fn compute_timeline_layout(&mut self) {
        // Place nodes horizontally by step number and vertically by branch
        let mut branch_indices = HashMap::new();
        let mut next_branch_index = 0;
        
        for (_, node) in self.tree_state.nodes.iter_mut() {
            let decision = &node.decision_point;
            let step = decision.step() as f32;
            
            // Get or assign branch index
            let branch_id = decision.branch_id();
            let branch_index = match branch_indices.get(&branch_id) {
                Some(&index) => index,
                None => {
                    let index = next_branch_index;
                    branch_indices.insert(branch_id, index);
                    next_branch_index += 1;
                    index
                }
            };
            
            let x = step * 0.5;
            let y = branch_index as f32 * 1.5;
            
            node.target_position = Vec3::new(x, y, 0.0);
            
            // If no animation in progress, set current position too
            if !self.tree_state.animating {
                node.position = node.target_position;
            }
        }
    }

    /// Prepare rendering data
    fn prepare_rendering_data(&mut self, context: &mut RenderContext) -> Result<(), String> {
        if !self.layout_computed {
            self.compute_layout();
        }
        
        // Skip if no nodes
        if self.tree_state.nodes.is_empty() {
            return Ok(());
        }
        
        // Prepare node vertices
        let mut node_vertices = Vec::new();
        let mut node_indices = Vec::new();
        let mut edge_vertices = Vec::new();
        let mut edge_indices = Vec::new();
        
        let mut next_node_index = 0;
        let mut next_edge_index = 0;
        
        // Generate node geometry
        for (_, node) in &self.tree_state.nodes {
            // Skip nodes below importance threshold
            if node.decision_point.importance().value() < self.config.importance_threshold {
                continue;
            }
            
            // Create node quad
            let half_size = node.size * 0.5;
            let z = 0.0;
            
            // Quad corners
            let corners = [
                // Positions           // Normals         // UVs      // Colors (RGBA)
                [node.position.x - half_size, node.position.y - half_size, z, 0.0, 0.0, 1.0, 0.0, 0.0, node.color.x as u8, node.color.y as u8, node.color.z as u8, 255],
                [node.position.x + half_size, node.position.y - half_size, z, 0.0, 0.0, 1.0, 1.0, 0.0, node.color.x as u8, node.color.y as u8, node.color.z as u8, 255],
                [node.position.x + half_size, node.position.y + half_size, z, 0.0, 0.0, 1.0, 1.0, 1.0, node.color.x as u8, node.color.y as u8, node.color.z as u8, 255],
                [node.position.x - half_size, node.position.y + half_size, z, 0.0, 0.0, 1.0, 0.0, 1.0, node.color.x as u8, node.color.y as u8, node.color.z as u8, 255],
            ];
            
            // Add corners to vertex buffer
            for corner in &corners {
                node_vertices.extend_from_slice(corner);
            }
            
            // Add indices for two triangles
            node_indices.extend_from_slice(&[
                next_node_index,
                next_node_index + 1,
                next_node_index + 2,
                next_node_index,
                next_node_index + 2,
                next_node_index + 3,
            ]);
            
            next_node_index += 4;
            
            // Generate edge geometry for connections to children
            for &child_id in &node.children {
                if let Some(child) = self.tree_state.nodes.get(&child_id) {
                    // Skip edges to nodes below importance threshold
                    if child.decision_point.importance().value() < self.config.importance_threshold {
                        continue;
                    }
                    
                    // Edge properties
                    let importance = node.decision_point.importance().value().max(
                        child.decision_point.importance().value()
                    );
                    let thickness = 0.05 + (importance * 0.1);
                    let alpha = 0.7 + (importance * 0.3);
                    
                    // Edge vertices
                    let edge = [
                        // Position 1                           // Color (RGBA)                                           // Thickness // Data
                        [node.position.x, node.position.y, z, 150, 150, 150, (alpha * 255.0) as u8, thickness, 0.0, 0.0],
                        // Position 2                           // Color (RGBA)                                           // Thickness // Data
                        [child.position.x, child.position.y, z, 150, 150, 150, (alpha * 255.0) as u8, thickness, 1.0, 0.0],
                    ];
                    
                    // Add edge vertices
                    for vertex in &edge {
                        edge_vertices.extend_from_slice(vertex);
                    }
                    
                    // Add edge indices
                    edge_indices.extend_from_slice(&[
                        next_edge_index,
                        next_edge_index + 1,
                    ]);
                    
                    next_edge_index += 2;
                }
            }
        }
        
        // Create GPU buffers
        let device = &context.device;
        let queue = &context.queue;
        
        // Update node buffers if not empty
        if !node_vertices.is_empty() {
            // Convert vectors to byte slices
            let node_vertex_bytes = unsafe { 
                std::slice::from_raw_parts(
                    node_vertices.as_ptr() as *const u8,
                    node_vertices.len() * std::mem::size_of::<f32>(),
                )
            };
            
            let node_index_bytes = unsafe {
                std::slice::from_raw_parts(
                    node_indices.as_ptr() as *const u8,
                    node_indices.len() * std::mem::size_of::<u32>(),
                )
            };
            
            // Create or resize buffers
            if let Some(buffer) = &self.node_vertices {
                if buffer.vertex_count < node_vertices.len() / 9 {
                    // Need to resize
                    let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Decision Node Vertex Buffer"),
                        contents: node_vertex_bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.node_vertices = Some(VertexBuffer::new(new_buffer, node_vertices.len() / 9));
                } else {
                    // Just update
                    queue.write_buffer(&buffer.buffer, 0, node_vertex_bytes);
                }
            }
            
            if let Some(buffer) = &self.node_indices {
                if buffer.index_count < node_indices.len() {
                    // Need to resize
                    let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Decision Node Index Buffer"),
                        contents: node_index_bytes,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.node_indices = Some(IndexBuffer::new(new_buffer, node_indices.len()));
                } else {
                    // Just update
                    queue.write_buffer(&buffer.buffer, 0, node_index_bytes);
                }
            }
        }
        
        // Update edge buffers if not empty
        if !edge_vertices.is_empty() {
            // Convert vectors to byte slices
            let edge_vertex_bytes = unsafe { 
                std::slice::from_raw_parts(
                    edge_vertices.as_ptr() as *const u8,
                    edge_vertices.len() * std::mem::size_of::<f32>(),
                )
            };
            
            let edge_index_bytes = unsafe {
                std::slice::from_raw_parts(
                    edge_indices.as_ptr() as *const u8,
                    edge_indices.len() * std::mem::size_of::<u32>(),
                )
            };
            
            // Create or resize buffers
            if let Some(buffer) = &self.edge_vertices {
                if buffer.vertex_count < edge_vertices.len() / 7 {
                    // Need to resize
                    let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Decision Edge Vertex Buffer"),
                        contents: edge_vertex_bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.edge_vertices = Some(VertexBuffer::new(new_buffer, edge_vertices.len() / 7));
                } else {
                    // Just update
                    queue.write_buffer(&buffer.buffer, 0, edge_vertex_bytes);
                }
            }
            
            if let Some(buffer) = &self.edge_indices {
                if buffer.index_count < edge_indices.len() {
                    // Need to resize
                    let new_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Decision Edge Index Buffer"),
                        contents: edge_index_bytes,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.edge_indices = Some(IndexBuffer::new(new_buffer, edge_indices.len()));
                } else {
                    // Just update
                    queue.write_buffer(&buffer.buffer, 0, edge_index_bytes);
                }
            }
        }
        
        Ok(())
    }

    /// Update animation state
    fn update_animation(&mut self, dt: f32) {
        if !self.tree_state.animating {
            return;
        }
        
        // Update animation progress
        self.tree_state.animation_progress += dt * 2.0; // 0.5 seconds for full animation
        if self.tree_state.animation_progress >= 1.0 {
            self.tree_state.animation_progress = 1.0;
            self.tree_state.animating = false;
        }
        
        // Interpolate node positions
        let t = self.tree_state.animation_progress;
        let ease_t = match self.config.transition {
            TransitionType::Linear => t,
            TransitionType::Exponential => 1.0 - (1.0 - t).powi(3),
            TransitionType::Elastic => {
                let p = 0.3;
                let s = p / 4.0;
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else {
                    let t = t - 1.0;
                    -((2.0_f32).powf(10.0 * t) * ((t - s) * (std::f32::consts::TAU / p)).sin())
                }
            },
            TransitionType::Bounce => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let f = (2.0 * t) - 2.0;
                    0.5 * f * f * f + 1.0
                }
            },
        };
        
        for (_, node) in self.tree_state.nodes.iter_mut() {
            node.position = node.position.lerp(node.target_position, ease_t);
        }
    }

    /// Update the visualization
    pub fn update(&mut self, dt: f32, context: &mut RenderContext) -> Result<(), String> {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        
        // Update animation
        self.update_animation(dt);
        
        // Initialize rendering if needed
        if !self.render_ready {
            self.initialize_rendering(context)?;
        }
        
        // Prepare rendering data
        self.prepare_rendering_data(context)?;
        
        // Update transformation matrix
        if let Some(buffer) = &self.transform_buffer {
            let view = self.camera.view_matrix();
            let proj = self.camera.projection_matrix(
                context.viewport_width as f32 / context.viewport_height as f32
            );
            let transform = proj * view;
            
            let transform_data = [
                transform.to_cols_array_2d()[0],
                transform.to_cols_array_2d()[1],
                transform.to_cols_array_2d()[2],
                transform.to_cols_array_2d()[3],
            ];
            
            context.queue.write_buffer(
                &buffer.buffer,
                0,
                bytemuck::cast_slice(&transform_data),
            );
        }
        
        // Update settings
        if let Some(buffer) = &self.settings_buffer {
            let settings_data = [
                self.config.importance_threshold,
                self.tree_state.animation_progress,
                self.config.max_visible_decisions as f32,
                self.tree_state.viewport_zoom,
            ];
            
            context.queue.write_buffer(
                &buffer.buffer,
                0,
                bytemuck::cast_slice(&settings_data),
            );
        }
        
        Ok(())
    }

    /// Render the visualization
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) -> Result<(), String> {
        if !self.render_ready {
            return Err("Rendering not initialized".to_string());
        }
        
        // Skip if no nodes
        if self.tree_state.nodes.is_empty() {
            return Ok(());
        }
        
        // Get pipelines and buffers
        let node_pipeline = self.node_pipeline.as_ref().unwrap();
        let edge_pipeline = self.edge_pipeline.as_ref().unwrap();
        let node_vertices = self.node_vertices.as_ref().unwrap();
        let node_indices = self.node_indices.as_ref().unwrap();
        let edge_vertices = self.edge_vertices.as_ref().unwrap();
        let edge_indices = self.edge_indices.as_ref().unwrap();
        
        // Draw edges
        if edge_indices.index_count > 0 {
            render_pass.set_pipeline(&edge_pipeline.pipeline);
            render_pass.set_vertex_buffer(0, edge_vertices.buffer.slice(..));
            render_pass.set_index_buffer(edge_indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..edge_indices.index_count as u32, 0, 0..1);
        }
        
        // Draw nodes
        if node_indices.index_count > 0 {
            render_pass.set_pipeline(&node_pipeline.pipeline);
            render_pass.set_vertex_buffer(0, node_vertices.buffer.slice(..));
            render_pass.set_index_buffer(node_indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..node_indices.index_count as u32, 0, 0..1);
        }
        
        Ok(())
    }

    /// Handle interaction events
    pub fn handle_interaction(
        &mut self,
        event: &InteractionEvent,
        controller: &mut InteractionController,
    ) -> InteractionResult {
        match event {
            InteractionEvent::MouseMove { x, y } => {
                // Add hover/highlight logic
                InteractionResult::Handled
            },
            InteractionEvent::MouseDown { button, x, y } => {
                // Add selection logic
                InteractionResult::Handled
            },
            InteractionEvent::MouseUp { button, x, y } => {
                // Add click handling
                InteractionResult::Handled
            },
            InteractionEvent::MouseWheel { delta } => {
                // Add zoom logic
                self.camera.zoom(*delta);
                InteractionResult::Handled
            },
            InteractionEvent::KeyDown { key } => {
                // Add keyboard navigation
                InteractionResult::Handled
            },
            InteractionEvent::KeyUp { key } => {
                InteractionResult::Ignored
            },
            InteractionEvent::Touch { .. } => {
                // Add touch handling
                InteractionResult::Handled
            },
            _ => InteractionResult::Ignored,
        }
    }

    /// Set visualization configuration
    pub fn set_config(&mut self, config: DecisionVisualizationConfig) {
        self.config = config;
        self.layout_computed = false;
    }

    /// Get current visualization configuration
    pub fn config(&self) -> &DecisionVisualizationConfig {
        &self.config
    }

    /// Set focus area
    pub fn set_focus_area(&mut self, focus: Option<FocusArea>) {
        self.config.focus = focus;
    }

    /// Set resolution level
    pub fn set_resolution(&mut self, resolution: ResolutionLevel) {
        self.config.resolution = resolution;
        self.layout_computed = false;
    }

    /// Set tree layout algorithm
    pub fn set_layout_algorithm(&mut self, algorithm: TreeLayoutAlgorithm) {
        self.config.layout_algorithm = algorithm;
        self.layout_computed = false;
    }

    /// Set visual encoding
    pub fn set_encoding(&mut self, encoding: DecisionEncoding) {
        self.config.encoding = encoding;
    }

    /// Create a branch at selected decision point
    pub fn create_branch(&mut self, modifier: Box<dyn FnOnce(&mut AlgorithmState)>) -> Result<BranchId, BranchCreationError> {
        // Get selected decision
        let selected_id = match self.tree_state.selected_ids.iter().next() {
            Some(&id) => id,
            None => return Err(BranchCreationError::NoBranchPoint),
        };
        
        let node = match self.tree_state.nodes.get(&selected_id) {
            Some(node) => node,
            None => return Err(BranchCreationError::NoBranchPoint),
        };
        
        // Create branch
        let branch_id = self.branch_manager.create_branch_at(
            &node.decision_point,
            modifier,
        )?;
        
        // Update from timeline
        self.update_from_timeline().map_err(|_| BranchCreationError::TimelineError)?;
        
        Ok(branch_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock types for testing
    struct MockTimeline {}
    impl Timeline for MockTimeline {
        fn current_state(&self) -> Result<Arc<AlgorithmState>, TimelineNavigationError> {
            unimplemented!()
        }
        
        fn decision_points(&self) -> Result<Vec<Arc<DecisionPoint>>, TimelineNavigationError> {
            unimplemented!()
        }
    }
    
    struct MockBranchManager {}
    impl BranchManager for MockBranchManager {
        fn create_branch_at(
            &self,
            decision: &Arc<DecisionPoint>,
            modifier: Box<dyn FnOnce(&mut AlgorithmState)>,
        ) -> Result<BranchId, BranchCreationError> {
            unimplemented!()
        }
    }
    
    #[test]
    fn test_decision_config_default() {
        let config = DecisionVisualizationConfig::default();
        
        assert_eq!(config.layout_algorithm, TreeLayoutAlgorithm::HorizontalTree);
        assert_eq!(config.resolution, ResolutionLevel::Medium);
        assert!(matches!(config.encoding, DecisionEncoding::SizeAndColor));
        assert!(config.focus.is_none());
        assert_eq!(config.transition, TransitionType::Exponential);
        assert!(config.show_aggregates);
        assert!(config.importance_threshold > 0.0);
        assert!(config.max_visible_decisions > 0);
    }
}