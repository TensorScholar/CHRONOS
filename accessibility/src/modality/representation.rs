//! Multi-modal representation system for algorithm behavior
//!
//! This module provides a framework for representing algorithm behavior across
//! different perceptual modalities (visual, auditory, textual) while maintaining
//! semantic equivalence and synchronization.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn, error};

use crate::algorithm::traits::{Algorithm, AlgorithmState};
use crate::temporal::timeline::Timeline;
use crate::visualization::engine::platform::PlatformCapabilities;
use crate::utils::math::{normalize};

/// Error type for modality representation
#[derive(Error, Debug)]
pub enum ModalityError {
    /// Modality not supported
    #[error("Modality not supported: {0}")]
    ModalityNotSupported(String),
    
    /// Transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),
    
    /// Synchronization error
    #[error("Synchronization error: {0}")]
    SynchronizationError(String),
    
    /// Element not found
    #[error("Element not found: {0}")]
    ElementNotFound(String),
    
    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

/// Result type for modality operations
pub type ModalityResult<T> = Result<T, ModalityError>;

/// Type of perceptual modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual representation
    Visual,
    
    /// Auditory representation
    Auditory,
    
    /// Textual representation
    Textual,
    
    /// Tactile representation
    Tactile,
}

/// Interface for perceptual modality
pub trait PerceptualModality: Send + Sync {
    /// Get the modality type
    fn modality_type(&self) -> Modality;
    
    /// Check if this modality is supported on the current platform
    fn is_supported(&self, capabilities: &PlatformCapabilities) -> bool;
    
    /// Initialize the modality
    fn initialize(&mut self, capabilities: &PlatformCapabilities) -> ModalityResult<()>;
    
    /// Shut down the modality
    fn shutdown(&mut self) -> ModalityResult<()>;
    
    /// Determine the information capacity of this modality
    fn information_capacity(&self) -> f64;
    
    /// Update the representation with new state
    fn update(&mut self, state: &AlgorithmState, timeline: Option<&Timeline>) -> ModalityResult<()>;
    
    /// Highlight a specific element
    fn highlight_element(
        &mut self, 
        element_id: &str, 
        highlight_params: Option<&HashMap<String, String>>
    ) -> ModalityResult<()>;
    
    /// Create a representation element
    fn create_element(
        &mut self,
        element_type: &str,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<RepresentationElement>;
    
    /// Get a representation element
    fn get_element(&self, element_id: &str) -> ModalityResult<RepresentationElement>;
    
    /// Update a representation element
    fn update_element(
        &mut self,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<()>;
    
    /// Remove a representation element
    fn remove_element(&mut self, element_id: &str) -> ModalityResult<()>;
    
    /// Get all representation elements
    fn get_elements(&self) -> Vec<RepresentationElement>;
    
    /// Handle interaction with the representation
    fn handle_interaction(
        &mut self,
        interaction_type: InteractionType,
        element_id: Option<&str>,
        params: Option<&HashMap<String, String>>
    ) -> ModalityResult<InteractionResult>;
}

/// Element in a representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationElement {
    /// Unique identifier
    pub id: String,
    
    /// Type of element
    pub element_type: String,
    
    /// Semantic properties of the element
    pub semantic_properties: HashMap<String, String>,
    
    /// Modality-specific properties
    pub modality_properties: HashMap<String, String>,
    
    /// Element state
    pub state: ElementState,
}

/// State of a representation element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementState {
    /// Element is active
    Active,
    
    /// Element is inactive
    Inactive,
    
    /// Element is highlighted
    Highlighted(String), // Highlight type
    
    /// Element is selected
    Selected,
    
    /// Element is focused
    Focused,
    
    /// Element is disabled
    Disabled,
}

/// Type of interaction with a representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Selection interaction
    Select,
    
    /// Focus interaction
    Focus,
    
    /// Activation interaction
    Activate,
    
    /// Hover interaction
    Hover,
    
    /// Drag interaction
    Drag,
    
    /// Zoom interaction
    Zoom,
    
    /// Navigation interaction
    Navigate,
    
    /// Custom interaction
    Custom(String),
}

/// Result of an interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionResult {
    /// Was the interaction successful
    pub success: bool,
    
    /// Result message
    pub message: Option<String>,
    
    /// Affected elements
    pub affected_elements: Vec<String>,
    
    /// Event data
    pub event_data: Option<HashMap<String, String>>,
}

/// Visual representation implementation
pub struct VisualModality {
    /// Visual elements in the representation
    elements: HashMap<String, RepresentationElement>,
    
    /// Visual capability level
    capability_level: VisualCapabilityLevel,
    
    /// Platform capabilities
    platform_capabilities: Option<PlatformCapabilities>,
    
    /// Current algorithm state
    current_state: Option<AlgorithmState>,
    
    /// Current timeline
    current_timeline: Option<Arc<Timeline>>,
    
    /// Visual settings
    settings: VisualSettings,
}

/// Visual capability level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisualCapabilityLevel {
    /// No visual capability
    None,
    
    /// Low visual capability (e.g., monochrome, low resolution)
    Low,
    
    /// Medium visual capability (e.g., color, medium resolution)
    Medium,
    
    /// High visual capability (e.g., high-resolution color, animation)
    High,
    
    /// Full visual capability (e.g., 3D, complex animation)
    Full,
}

/// Visual settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualSettings {
    /// Color scheme
    pub color_scheme: ColorScheme,
    
    /// Shape scheme
    pub shape_scheme: ShapeScheme,
    
    /// Animation settings
    pub animation: AnimationSettings,
    
    /// Contrast settings
    pub contrast: ContrastSettings,
    
    /// Size multiplier
    pub size_multiplier: f64,
}

/// Color scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default color scheme
    Default,
    
    /// High contrast color scheme
    HighContrast,
    
    /// Colorblind-friendly scheme
    ColorblindFriendly,
    
    /// Monochrome scheme
    Monochrome,
    
    /// Custom color scheme
    Custom(HashMap<String, String>),
}

/// Shape scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeScheme {
    /// Default shape scheme
    Default,
    
    /// Simplified shape scheme
    Simplified,
    
    /// Detailed shape scheme
    Detailed,
    
    /// Custom shape scheme
    Custom(HashMap<String, String>),
}

/// Animation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationSettings {
    /// Animation enabled
    pub enabled: bool,
    
    /// Animation speed
    pub speed: f64,
    
    /// Animation complexity
    pub complexity: AnimationComplexity,
}

/// Animation complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationComplexity {
    /// Simple animations
    Simple,
    
    /// Moderate complexity
    Moderate,
    
    /// Complex animations
    Complex,
}

/// Contrast settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastSettings {
    /// Contrast level
    pub level: f64,
    
    /// Use high contrast mode
    pub high_contrast_mode: bool,
    
    /// Use patterns in addition to colors
    pub use_patterns: bool,
}

impl Default for VisualSettings {
    fn default() -> Self {
        Self {
            color_scheme: ColorScheme::Default,
            shape_scheme: ShapeScheme::Default,
            animation: AnimationSettings {
                enabled: true,
                speed: 1.0,
                complexity: AnimationComplexity::Moderate,
            },
            contrast: ContrastSettings {
                level: 1.0,
                high_contrast_mode: false,
                use_patterns: false,
            },
            size_multiplier: 1.0,
        }
    }
}

impl VisualModality {
    /// Create a new visual modality
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            capability_level: VisualCapabilityLevel::None,
            platform_capabilities: None,
            current_state: None,
            current_timeline: None,
            settings: VisualSettings::default(),
        }
    }
    
    /// Detect visual capability level
    fn detect_capability_level(&self, capabilities: &PlatformCapabilities) -> VisualCapabilityLevel {
        if !capabilities.has_display {
            return VisualCapabilityLevel::None;
        }
        
        if capabilities.graphics_tier >= 3 {
            VisualCapabilityLevel::Full
        } else if capabilities.graphics_tier == 2 {
            VisualCapabilityLevel::High
        } else if capabilities.graphics_tier == 1 {
            VisualCapabilityLevel::Medium
        } else {
            VisualCapabilityLevel::Low
        }
    }
    
    /// Apply visual settings
    pub fn apply_settings(&mut self, settings: VisualSettings) -> ModalityResult<()> {
        self.settings = settings;
        
        // Update all elements with new settings
        for element in self.elements.values_mut() {
            self.apply_settings_to_element(element)?;
        }
        
        Ok(())
    }
    
    /// Apply settings to an element
    fn apply_settings_to_element(&self, element: &mut RepresentationElement) -> ModalityResult<()> {
        // Apply color scheme
        match &self.settings.color_scheme {
            ColorScheme::Default => {
                // Use default colors (already set)
            },
            ColorScheme::HighContrast => {
                // Apply high contrast colors based on element type
                let color = match element.element_type.as_str() {
                    "node" => "#FFFFFF", // White
                    "edge" => "#000000", // Black
                    "highlight" => "#FF0000", // Red
                    _ => "#888888", // Gray
                };
                element.modality_properties.insert("color".to_string(), color.to_string());
            },
            ColorScheme::ColorblindFriendly => {
                // Apply colorblind-friendly colors
                let color = match element.element_type.as_str() {
                    "node" => "#1170AA", // Blue
                    "edge" => "#000000", // Black
                    "highlight" => "#FC7D0B", // Orange
                    "visited" => "#A3ACB9", // Gray
                    "current" => "#57606C", // Dark gray
                    _ => "#5FA2CE", // Light blue
                };
                element.modality_properties.insert("color".to_string(), color.to_string());
            },
            ColorScheme::Monochrome => {
                // Apply monochrome colors with different shades
                let shade = match element.element_type.as_str() {
                    "node" => "222222", // Dark gray
                    "edge" => "000000", // Black
                    "highlight" => "888888", // Medium gray
                    "visited" => "AAAAAA", // Light gray
                    "current" => "444444", // Gray
                    _ => "666666", // Gray
                };
                element.modality_properties.insert("color".to_string(), format!("#{}", shade));
            },
            ColorScheme::Custom(scheme) => {
                // Apply custom color if available for this element type
                if let Some(color) = scheme.get(&element.element_type) {
                    element.modality_properties.insert("color".to_string(), color.clone());
                }
            },
        }
        
        // Apply shape scheme
        match &self.settings.shape_scheme {
            ShapeScheme::Default => {
                // Use default shapes (already set)
            },
            ShapeScheme::Simplified => {
                // Apply simplified shapes
                let shape = match element.element_type.as_str() {
                    "node" => "circle",
                    "edge" => "line",
                    _ => "rect",
                };
                element.modality_properties.insert("shape".to_string(), shape.to_string());
            },
            ShapeScheme::Detailed => {
                // Apply more detailed shapes
                let shape = match element.element_type.as_str() {
                    "node" => "roundedRect",
                    "edge" => "arrow",
                    _ => "polygon",
                };
                element.modality_properties.insert("shape".to_string(), shape.to_string());
            },
            ShapeScheme::Custom(scheme) => {
                // Apply custom shape if available for this element type
                if let Some(shape) = scheme.get(&element.element_type) {
                    element.modality_properties.insert("shape".to_string(), shape.clone());
                }
            },
        }
        
        // Apply size multiplier
        if let Some(size_str) = element.modality_properties.get("size") {
            if let Ok(size) = size_str.parse::<f64>() {
                let new_size = size * self.settings.size_multiplier;
                element.modality_properties.insert("size".to_string(), new_size.to_string());
            }
        }
        
        // Apply contrast settings
        if self.settings.contrast.high_contrast_mode {
            element.modality_properties.insert("high_contrast".to_string(), "true".to_string());
        } else {
            element.modality_properties.insert("high_contrast".to_string(), "false".to_string());
        }
        
        if self.settings.contrast.use_patterns {
            // Apply patterns based on element type
            let pattern = match element.element_type.as_str() {
                "node" => "solid",
                "edge" => "solid",
                "highlight" => "dashed",
                "visited" => "dotted",
                "current" => "solid",
                _ => "solid",
            };
            element.modality_properties.insert("pattern".to_string(), pattern.to_string());
        }
        
        Ok(())
    }
    
    /// Create a visual element for an algorithm component
    fn create_visual_element_for_component(
        &self,
        component_type: &str,
        component_id: &str,
        state: &AlgorithmState
    ) -> ModalityResult<RepresentationElement> {
        // Create properties for the element
        let mut semantic_properties = HashMap::new();
        let mut modality_properties = HashMap::new();
        
        // Set semantic properties
        semantic_properties.insert("component_type".to_string(), component_type.to_string());
        semantic_properties.insert("component_id".to_string(), component_id.to_string());
        semantic_properties.insert("algorithm_step".to_string(), state.step.to_string());
        
        // Set modality-specific properties
        match component_type {
            "node" => {
                modality_properties.insert("shape".to_string(), "circle".to_string());
                modality_properties.insert("size".to_string(), "10".to_string());
                
                // Determine color based on node status
                let color = if state.current_node == Some(component_id.parse().unwrap_or(0)) {
                    "#FF0000" // Red for current node
                } else if state.closed_set.contains(&component_id.parse().unwrap_or(0)) {
                    "#888888" // Gray for closed set
                } else if state.open_set.contains(&component_id.parse().unwrap_or(0)) {
                    "#00FF00" // Green for open set
                } else {
                    "#0000FF" // Blue for other nodes
                };
                
                modality_properties.insert("color".to_string(), color.to_string());
                modality_properties.insert("border_color".to_string(), "#000000".to_string());
                modality_properties.insert("border_width".to_string(), "1".to_string());
            },
            "edge" => {
                modality_properties.insert("shape".to_string(), "line".to_string());
                modality_properties.insert("color".to_string(), "#000000".to_string());
                modality_properties.insert("width".to_string(), "1".to_string());
                modality_properties.insert("style".to_string(), "solid".to_string());
            },
            "label" => {
                modality_properties.insert("font".to_string(), "Arial".to_string());
                modality_properties.insert("font_size".to_string(), "12".to_string());
                modality_properties.insert("color".to_string(), "#000000".to_string());
                modality_properties.insert("background_color".to_string(), "transparent".to_string());
            },
            _ => {
                return Err(ModalityError::UnsupportedFeature(
                    format!("Unsupported component type: {}", component_type)
                ));
            }
        }
        
        // Create element state (default to Active)
        let element_state = ElementState::Active;
        
        // Create the element
        let element = RepresentationElement {
            id: format!("{}_{}", component_type, component_id),
            element_type: component_type.to_string(),
            semantic_properties,
            modality_properties,
            state: element_state,
        };
        
        Ok(element)
    }
}

impl PerceptualModality for VisualModality {
    fn modality_type(&self) -> Modality {
        Modality::Visual
    }
    
    fn is_supported(&self, capabilities: &PlatformCapabilities) -> bool {
        capabilities.has_display
    }
    
    fn initialize(&mut self, capabilities: &PlatformCapabilities) -> ModalityResult<()> {
        self.capability_level = self.detect_capability_level(capabilities);
        self.platform_capabilities = Some(capabilities.clone());
        
        if self.capability_level == VisualCapabilityLevel::None {
            return Err(ModalityError::ModalityNotSupported(
                "Visual modality not supported on this platform".to_string()
            ));
        }
        
        info!("Visual modality initialized with capability level: {:?}", self.capability_level);
        
        Ok(())
    }
    
    fn shutdown(&mut self) -> ModalityResult<()> {
        self.elements.clear();
        self.platform_capabilities = None;
        self.current_state = None;
        self.current_timeline = None;
        
        info!("Visual modality shut down");
        
        Ok(())
    }
    
    fn information_capacity(&self) -> f64 {
        match self.capability_level {
            VisualCapabilityLevel::None => 0.0,
            VisualCapabilityLevel::Low => 0.3,
            VisualCapabilityLevel::Medium => 0.6,
            VisualCapabilityLevel::High => 0.8,
            VisualCapabilityLevel::Full => 1.0,
        }
    }
    
    fn update(&mut self, state: &AlgorithmState, timeline: Option<&Timeline>) -> ModalityResult<()> {
        self.current_state = Some(state.clone());
        self.current_timeline = timeline.map(|t| Arc::new(t.clone()));
        
        // Update or create elements for nodes
        for node_id in 0..100 { // Assuming a reasonable maximum number of nodes
            if state.open_set.contains(&node_id) || 
               state.closed_set.contains(&node_id) || 
               state.current_node == Some(node_id) {
                
                let element_id = format!("node_{}", node_id);
                
                if self.elements.contains_key(&element_id) {
                    // Update existing element
                    let mut params = HashMap::new();
                    
                    // Set color based on node status
                    let color = if state.current_node == Some(node_id) {
                        "#FF0000" // Red for current node
                    } else if state.closed_set.contains(&node_id) {
                        "#888888" // Gray for closed set
                    } else if state.open_set.contains(&node_id) {
                        "#00FF00" // Green for open set
                    } else {
                        "#0000FF" // Blue for other nodes
                    };
                    
                    params.insert("color".to_string(), color.to_string());
                    self.update_element(&element_id, &params)?;
                } else {
                    // Create new element
                    let element = self.create_visual_element_for_component(
                        "node", 
                        &node_id.to_string(), 
                        state
                    )?;
                    self.elements.insert(element_id, element);
                }
            }
        }
        
        // For a complete implementation, we would also handle edges, labels, etc.
        
        Ok(())
    }
    
    fn highlight_element(
        &mut self, 
        element_id: &str, 
        highlight_params: Option<&HashMap<String, String>>
    ) -> ModalityResult<()> {
        if let Some(element) = self.elements.get_mut(element_id) {
            // Set highlight state
            let highlight_type = highlight_params
                .and_then(|p| p.get("type"))
                .unwrap_or(&"default".to_string())
                .clone();
            
            element.state = ElementState::Highlighted(highlight_type.clone());
            
            // Set highlight properties
            let highlight_color = highlight_params
                .and_then(|p| p.get("color"))
                .unwrap_or(&"#FF0000".to_string())
                .clone();
            
            element.modality_properties.insert("highlight_color".to_string(), highlight_color);
            
            if let Some(params) = highlight_params {
                for (key, value) in params {
                    if key != "type" && key != "color" {
                        element.modality_properties.insert(format!("highlight_{}", key), value.clone());
                    }
                }
            }
            
            Ok(())
        } else {
            Err(ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
        }
    }
    
    fn create_element(
        &mut self,
        element_type: &str,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<RepresentationElement> {
        // Create semantic properties
        let mut semantic_properties = HashMap::new();
        for (key, value) in params {
            if key.starts_with("semantic_") {
                let semantic_key = key.strip_prefix("semantic_").unwrap();
                semantic_properties.insert(semantic_key.to_string(), value.clone());
            }
        }
        
        // Create modality properties
        let mut modality_properties = HashMap::new();
        for (key, value) in params {
            if !key.starts_with("semantic_") {
                modality_properties.insert(key.clone(), value.clone());
            }
        }
        
        // Create the element
        let element = RepresentationElement {
            id: element_id.to_string(),
            element_type: element_type.to_string(),
            semantic_properties,
            modality_properties,
            state: ElementState::Active,
        };
        
        // Apply settings to the element
        let mut element_with_settings = element.clone();
        self.apply_settings_to_element(&mut element_with_settings)?;
        
        // Store the element
        self.elements.insert(element_id.to_string(), element_with_settings.clone());
        
        Ok(element_with_settings)
    }
    
    fn get_element(&self, element_id: &str) -> ModalityResult<RepresentationElement> {
        self.elements.get(element_id)
            .cloned()
            .ok_or_else(|| ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
    }
    
    fn update_element(
        &mut self,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<()> {
        if let Some(element) = self.elements.get_mut(element_id) {
            // Update semantic properties
            for (key, value) in params {
                if key.starts_with("semantic_") {
                    let semantic_key = key.strip_prefix("semantic_").unwrap();
                    element.semantic_properties.insert(semantic_key.to_string(), value.clone());
                } else {
                    element.modality_properties.insert(key.clone(), value.clone());
                }
            }
            
            // Apply settings to the updated element
            self.apply_settings_to_element(element)?;
            
            Ok(())
        } else {
            Err(ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
        }
    }
    
    fn remove_element(&mut self, element_id: &str) -> ModalityResult<()> {
        if self.elements.remove(element_id).is_some() {
            Ok(())
        } else {
            Err(ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
        }
    }
    
    fn get_elements(&self) -> Vec<RepresentationElement> {
        self.elements.values().cloned().collect()
    }
    
    fn handle_interaction(
        &mut self,
        interaction_type: InteractionType,
        element_id: Option<&str>,
        params: Option<&HashMap<String, String>>
    ) -> ModalityResult<InteractionResult> {
        match interaction_type {
            InteractionType::Select => {
                if let Some(id) = element_id {
                    if let Some(element) = self.elements.get_mut(id) {
                        element.state = ElementState::Selected;
                        
                        Ok(InteractionResult {
                            success: true,
                            message: Some(format!("Element {} selected", id)),
                            affected_elements: vec![id.to_string()],
                            event_data: None,
                        })
                    } else {
                        Err(ModalityError::ElementNotFound(
                            format!("Element not found: {}", id)
                        ))
                    }
                } else {
                    // Deselect all elements
                    let mut affected_elements = Vec::new();
                    
                    for (id, element) in &mut self.elements {
                        if matches!(element.state, ElementState::Selected) {
                            element.state = ElementState::Active;
                            affected_elements.push(id.clone());
                        }
                    }
                    
                    Ok(InteractionResult {
                        success: true,
                        message: Some(format!("All elements deselected")),
                        affected_elements,
                        event_data: None,
                    })
                }
            },
            InteractionType::Focus => {
                if let Some(id) = element_id {
                    if let Some(element) = self.elements.get_mut(id) {
                        element.state = ElementState::Focused;
                        
                        // Unfocus other elements
                        let mut affected_elements = vec![id.to_string()];
                        
                        for (other_id, other_element) in &mut self.elements {
                            if other_id != id && matches!(other_element.state, ElementState::Focused) {
                                other_element.state = ElementState::Active;
                                affected_elements.push(other_id.clone());
                            }
                        }
                        
                        Ok(InteractionResult {
                            success: true,
                            message: Some(format!("Element {} focused", id)),
                            affected_elements,
                            event_data: None,
                        })
                    } else {
                        Err(ModalityError::ElementNotFound(
                            format!("Element not found: {}", id)
                        ))
                    }
                } else {
                    // Remove focus from all elements
                    let mut affected_elements = Vec::new();
                    
                    for (id, element) in &mut self.elements {
                        if matches!(element.state, ElementState::Focused) {
                            element.state = ElementState::Active;
                            affected_elements.push(id.clone());
                        }
                    }
                    
                    Ok(InteractionResult {
                        success: true,
                        message: Some(format!("Focus removed from all elements")),
                        affected_elements,
                        event_data: None,
                    })
                }
            },
            // Implement other interaction types as needed
            _ => {
                Err(ModalityError::UnsupportedFeature(
                    format!("Interaction type not implemented: {:?}", interaction_type)
                ))
            }
        }
    }
}

/// Auditory representation implementation
pub struct AuditoryModality {
    /// Auditory elements in the representation
    elements: HashMap<String, RepresentationElement>,
    
    /// Auditory capability level
    capability_level: AuditoryCapabilityLevel,
    
    /// Platform capabilities
    platform_capabilities: Option<PlatformCapabilities>,
    
    /// Current algorithm state
    current_state: Option<AlgorithmState>,
    
    /// Current timeline
    current_timeline: Option<Arc<Timeline>>,
    
    /// Auditory settings
    settings: AuditorySettings,
    
    /// Sound generation system
    sound_system: Option<Box<dyn SoundSystem>>,
}

/// Auditory capability level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditoryCapabilityLevel {
    /// No auditory capability
    None,
    
    /// Low auditory capability (e.g., basic beeps)
    Low,
    
    /// Medium auditory capability (e.g., simple sounds)
    Medium,
    
    /// High auditory capability (e.g., complex sounds, stereo)
    High,
    
    /// Full auditory capability (e.g., spatial audio, complex synthesis)
    Full,
}

/// Auditory settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorySettings {
    /// Volume level
    pub volume: f64,
    
    /// Enable spatial audio
    pub spatial_audio: bool,
    
    /// Background sounds enabled
    pub background_sounds: bool,
    
    /// Sound complexity
    pub complexity: SoundComplexity,
    
    /// Speech synthesis enabled
    pub speech_synthesis: bool,
    
    /// Speech rate
    pub speech_rate: f64,
}

/// Sound complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoundComplexity {
    /// Simple sounds (e.g., basic tones)
    Simple,
    
    /// Moderate complexity (e.g., musical notes)
    Moderate,
    
    /// Complex sounds (e.g., realistic instruments)
    Complex,
}

impl Default for AuditorySettings {
    fn default() -> Self {
        Self {
            volume: 0.8,
            spatial_audio: true,
            background_sounds: true,
            complexity: SoundComplexity::Moderate,
            speech_synthesis: true,
            speech_rate: 1.0,
        }
    }
}

/// Sound system interface
pub trait SoundSystem: Send + Sync {
    /// Initialize the sound system
    fn initialize(&mut self) -> ModalityResult<()>;
    
    /// Shut down the sound system
    fn shutdown(&mut self) -> ModalityResult<()>;
    
    /// Play a sound
    fn play_sound(
        &self,
        sound_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<()>;
    
    /// Stop a sound
    fn stop_sound(&self, sound_id: &str) -> ModalityResult<()>;
    
    /// Speak text
    fn speak(
        &self,
        text: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<()>;
    
    /// Stop speaking
    fn stop_speaking(&self) -> ModalityResult<()>;
}

impl AuditoryModality {
    /// Create a new auditory modality
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            capability_level: AuditoryCapabilityLevel::None,
            platform_capabilities: None,
            current_state: None,
            current_timeline: None,
            settings: AuditorySettings::default(),
            sound_system: None,
        }
    }
    
    /// Detect auditory capability level
    fn detect_capability_level(&self, capabilities: &PlatformCapabilities) -> AuditoryCapabilityLevel {
        if !capabilities.has_audio {
            return AuditoryCapabilityLevel::None;
        }
        
        if capabilities.has_spatial_audio && capabilities.has_speech_synthesis {
            AuditoryCapabilityLevel::Full
        } else if capabilities.has_speech_synthesis {
            AuditoryCapabilityLevel::High
        } else if capabilities.has_stereo_audio {
            AuditoryCapabilityLevel::Medium
        } else {
            AuditoryCapabilityLevel::Low
        }
    }
    
    /// Apply auditory settings
    pub fn apply_settings(&mut self, settings: AuditorySettings) -> ModalityResult<()> {
        self.settings = settings;
        
        // Apply settings to sound system
        if let Some(sound_system) = &self.sound_system {
            // Update global volume
            let mut params = HashMap::new();
            params.insert("volume".to_string(), self.settings.volume.to_string());
            sound_system.play_sound("settings_update", &params)?;
        }
        
        Ok(())
    }
    
    /// Create an auditory element for an algorithm component
    fn create_auditory_element_for_component(
        &self,
        component_type: &str,
        component_id: &str,
        state: &AlgorithmState
    ) -> ModalityResult<RepresentationElement> {
        // Create properties for the element
        let mut semantic_properties = HashMap::new();
        let mut modality_properties = HashMap::new();
        
        // Set semantic properties
        semantic_properties.insert("component_type".to_string(), component_type.to_string());
        semantic_properties.insert("component_id".to_string(), component_id.to_string());
        semantic_properties.insert("algorithm_step".to_string(), state.step.to_string());
        
        // Set modality-specific properties based on component type
        match component_type {
            "node" => {
                // Determine pitch based on node status
                let pitch = if state.current_node == Some(component_id.parse().unwrap_or(0)) {
                    "high" // High pitch for current node
                } else if state.closed_set.contains(&component_id.parse().unwrap_or(0)) {
                    "low" // Low pitch for closed set
                } else if state.open_set.contains(&component_id.parse().unwrap_or(0)) {
                    "medium" // Medium pitch for open set
                } else {
                    "very_low" // Very low pitch for other nodes
                };
                
                modality_properties.insert("sound_type".to_string(), "tone".to_string());
                modality_properties.insert("pitch".to_string(), pitch.to_string());
                modality_properties.insert("duration".to_string(), "0.2".to_string());
                
                // Add spatial position if supported
                if self.settings.spatial_audio && 
                   self.capability_level >= AuditoryCapabilityLevel::High {
                    // Arbitrary spatial positioning based on node ID for this example
                    let id_num = component_id.parse::<usize>().unwrap_or(0);
                    let x = ((id_num % 10) as f64 / 10.0) * 2.0 - 1.0; // Range: -1.0 to 1.0
                    let y = ((id_num / 10) as f64 / 10.0) * 2.0 - 1.0; // Range: -1.0 to 1.0
                    
                    modality_properties.insert("spatial_x".to_string(), x.to_string());
                    modality_properties.insert("spatial_y".to_string(), y.to_string());
                }
            },
            "edge" => {
                modality_properties.insert("sound_type".to_string(), "sweep".to_string());
                modality_properties.insert("start_pitch".to_string(), "low".to_string());
                modality_properties.insert("end_pitch".to_string(), "medium".to_string());
                modality_properties.insert("duration".to_string(), "0.5".to_string());
            },
            "state_change" => {
                modality_properties.insert("sound_type".to_string(), "effect".to_string());
                modality_properties.insert("effect_type".to_string(), "state_transition".to_string());
                modality_properties.insert("duration".to_string(), "0.3".to_string());
            },
            _ => {
                return Err(ModalityError::UnsupportedFeature(
                    format!("Unsupported component type for auditory modality: {}", component_type)
                ));
            }
        }
        
        // Create element state (default to Active)
        let element_state = ElementState::Active;
        
        // Create the element
        let element = RepresentationElement {
            id: format!("audio_{}_{}", component_type, component_id),
            element_type: component_type.to_string(),
            semantic_properties,
            modality_properties,
            state: element_state,
        };
        
        Ok(element)
    }
    
    /// Sonify algorithm state
    fn sonify_state(&self, state: &AlgorithmState) -> ModalityResult<()> {
        if let Some(sound_system) = &self.sound_system {
            // Sonify current node if present
            if let Some(current_node) = state.current_node {
                let mut params = HashMap::new();
                params.insert("sound_type".to_string(), "tone".to_string());
                params.insert("pitch".to_string(), "high".to_string());
                params.insert("duration".to_string(), "0.3".to_string());
                
                if self.settings.spatial_audio {
                    // Position sound based on node ID
                    let x = ((current_node % 10) as f64 / 10.0) * 2.0 - 1.0; // Range: -1.0 to 1.0
                    let y = ((current_node / 10) as f64 / 10.0) * 2.0 - 1.0; // Range: -1.0 to 1.0
                    
                    params.insert("spatial_x".to_string(), x.to_string());
                    params.insert("spatial_y".to_string(), y.to_string());
                }
                
                sound_system.play_sound(&format!("node_{}", current_node), &params)?;
                
                // Speak node description if speech synthesis is enabled
                if self.settings.speech_synthesis {
                    let text = format!("Current node: {}", current_node);
                    let mut speech_params = HashMap::new();
                    speech_params.insert("rate".to_string(), self.settings.speech_rate.to_string());
                    speech_params.insert("volume".to_string(), self.settings.volume.to_string());
                    
                    sound_system.speak(&text, &speech_params)?;
                }
            }
            
            // Sonify state change
            let mut state_params = HashMap::new();
            state_params.insert("sound_type".to_string(), "effect".to_string());
            state_params.insert("effect_type".to_string(), "state_transition".to_string());
            state_params.insert("step".to_string(), state.step.to_string());
            
            sound_system.play_sound("state_change", &state_params)?;
        }
        
        Ok(())
    }
}

impl PerceptualModality for AuditoryModality {
    fn modality_type(&self) -> Modality {
        Modality::Auditory
    }
    
    fn is_supported(&self, capabilities: &PlatformCapabilities) -> bool {
        capabilities.has_audio
    }
    
    fn initialize(&mut self, capabilities: &PlatformCapabilities) -> ModalityResult<()> {
        self.capability_level = self.detect_capability_level(capabilities);
        self.platform_capabilities = Some(capabilities.clone());
        
        if self.capability_level == AuditoryCapabilityLevel::None {
            return Err(ModalityError::ModalityNotSupported(
                "Auditory modality not supported on this platform".to_string()
            ));
        }
        
        // Initialize sound system (simplified - would be platform-specific)
        // self.sound_system = Some(Box::new(DefaultSoundSystem::new(capabilities)));
        // self.sound_system.as_mut().unwrap().initialize()?;
        
        info!("Auditory modality initialized with capability level: {:?}", self.capability_level);
        
        Ok(())
    }
    
    fn shutdown(&mut self) -> ModalityResult<()> {
        if let Some(sound_system) = &mut self.sound_system {
            sound_system.shutdown()?;
        }
        
        self.elements.clear();
        self.platform_capabilities = None;
        self.current_state = None;
        self.current_timeline = None;
        self.sound_system = None;
        
        info!("Auditory modality shut down");
        
        Ok(())
    }
    
    fn information_capacity(&self) -> f64 {
        match self.capability_level {
            AuditoryCapabilityLevel::None => 0.0,
            AuditoryCapabilityLevel::Low => 0.2,
            AuditoryCapabilityLevel::Medium => 0.4,
            AuditoryCapabilityLevel::High => 0.6,
            AuditoryCapabilityLevel::Full => 0.7,
        }
    }
    
    fn update(&mut self, state: &AlgorithmState, timeline: Option<&Timeline>) -> ModalityResult<()> {
        self.current_state = Some(state.clone());
        self.current_timeline = timeline.map(|t| Arc::new(t.clone()));
        
        // Sonify the updated state
        self.sonify_state(state)?;
        
        // For a complete implementation, we would manage auditory elements
        // Similar to the visual modality implementation
        
        Ok(())
    }
    
    // Implementation of other methods would be similar to VisualModality
    // For brevity, they are not fully implemented here
    
    fn highlight_element(
        &mut self, 
        element_id: &str, 
        highlight_params: Option<&HashMap<String, String>>
    ) -> ModalityResult<()> {
        // Implementation would be similar to VisualModality
        Err(ModalityError::UnsupportedFeature("Not implemented".to_string()))
    }
    
    fn create_element(
        &mut self,
        element_type: &str,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<RepresentationElement> {
        // Implementation would be similar to VisualModality
        Err(ModalityError::UnsupportedFeature("Not implemented".to_string()))
    }
    
    fn get_element(&self, element_id: &str) -> ModalityResult<RepresentationElement> {
        self.elements.get(element_id)
            .cloned()
            .ok_or_else(|| ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
    }
    
    fn update_element(
        &mut self,
        element_id: &str,
        params: &HashMap<String, String>
    ) -> ModalityResult<()> {
        // Implementation would be similar to VisualModality
        Err(ModalityError::UnsupportedFeature("Not implemented".to_string()))
    }
    
    fn remove_element(&mut self, element_id: &str) -> ModalityResult<()> {
        if self.elements.remove(element_id).is_some() {
            Ok(())
        } else {
            Err(ModalityError::ElementNotFound(
                format!("Element not found: {}", element_id)
            ))
        }
    }
    
    fn get_elements(&self) -> Vec<RepresentationElement> {
        self.elements.values().cloned().collect()
    }
    
    fn handle_interaction(
        &mut self,
        interaction_type: InteractionType,
        element_id: Option<&str>,
        params: Option<&HashMap<String, String>>
    ) -> ModalityResult<InteractionResult> {
        // Implementation would be similar to VisualModality
        Err(ModalityError::UnsupportedFeature("Not implemented".to_string()))
    }
}

/// Multi-modal representation manager
pub struct MultiModalRepresentation {
    /// Available modalities
    modalities: HashMap<Modality, Box<dyn PerceptualModality>>,
    
    /// Active modalities
    active_modalities: HashSet<Modality>,
    
    /// Platform capabilities
    platform_capabilities: PlatformCapabilities,
    
    /// Synchronization manager
    synchronization: ModalitySynchronization,
    
    /// Current algorithm state
    current_state: Option<AlgorithmState>,
    
    /// Current timeline
    current_timeline: Option<Arc<Timeline>>,
}

/// Modality synchronization manager
struct ModalitySynchronization {
    /// Synchronization lock
    lock: Mutex<()>,
    
    /// Last update time
    last_update: std::time::Instant,
    
    /// Pending updates by modality
    pending_updates: HashMap<Modality, Vec<SynchronizationEvent>>,
}

/// Synchronization event
#[derive(Debug, Clone)]
struct SynchronizationEvent {
    /// Event type
    event_type: SynchronizationEventType,
    
    /// Event timestamp
    timestamp: std::time::Instant,
    
    /// Event parameters
    params: HashMap<String, String>,
}

/// Synchronization event type
#[derive(Debug, Clone)]
enum SynchronizationEventType {
    /// State update event
    StateUpdate,
    
    /// Element highlight event
    ElementHighlight,
    
    /// Element update event
    ElementUpdate,
    
    /// Element creation event
    ElementCreation,
    
    /// Element removal event
    ElementRemoval,
    
    /// Interaction event
    Interaction,
}

impl MultiModalRepresentation {
    /// Create a new multi-modal representation
    pub fn new(platform_capabilities: PlatformCapabilities) -> Self {
        Self {
            modalities: HashMap::new(),
            active_modalities: HashSet::new(),
            platform_capabilities,
            synchronization: ModalitySynchronization {
                lock: Mutex::new(()),
                last_update: std::time::Instant::now(),
                pending_updates: HashMap::new(),
            },
            current_state: None,
            current_timeline: None,
        }
    }
    
    /// Register a modality
    pub fn register_modality(&mut self, modality: Box<dyn PerceptualModality>) -> ModalityResult<()> {
        let modality_type = modality.modality_type();
        
        if !modality.is_supported(&self.platform_capabilities) {
            return Err(ModalityError::ModalityNotSupported(
                format!("Modality {:?} not supported on this platform", modality_type)
            ));
        }
        
        self.modalities.insert(modality_type, modality);
        
        Ok(())
    }
    
    /// Activate a modality
    pub fn activate_modality(&mut self, modality_type: Modality) -> ModalityResult<()> {
        if let Some(modality) = self.modalities.get_mut(&modality_type) {
            modality.initialize(&self.platform_capabilities)?;
            self.active_modalities.insert(modality_type);
            
            // Update the modality with current state if available
            if let Some(state) = &self.current_state {
                let timeline = self.current_timeline.as_deref();
                modality.update(state, timeline)?;
            }
            
            Ok(())
        } else {
            Err(ModalityError::ModalityNotSupported(
                format!("Modality {:?} not registered", modality_type)
            ))
        }
    }
    
    /// Deactivate a modality
    pub fn deactivate_modality(&mut self, modality_type: Modality) -> ModalityResult<()> {
        if let Some(modality) = self.modalities.get_mut(&modality_type) {
            modality.shutdown()?;
            self.active_modalities.remove(&modality_type);
            
            Ok(())
        } else {
            Err(ModalityError::ModalityNotSupported(
                format!("Modality {:?} not registered", modality_type)
            ))
        }
    }
    
    /// Update all active modalities with new state
    pub fn update(&mut self, state: &AlgorithmState, timeline: Option<&Timeline>) -> ModalityResult<()> {
        // Store current state and timeline
        self.current_state = Some(state.clone());
        self.current_timeline = timeline.map(|t| Arc::new(t.clone()));
        
        // Create synchronization event
        let event = SynchronizationEvent {
            event_type: SynchronizationEventType::StateUpdate,
            timestamp: std::time::Instant::now(),
            params: HashMap::new(),
        };
        
        // Acquire synchronization lock
        let _lock = self.synchronization.lock.lock().unwrap();
        
        // Update each active modality
        for modality_type in &self.active_modalities {
            if let Some(modality) = self.modalities.get_mut(modality_type) {
                modality.update(state, timeline)?;
                
                // Record update event
                if !self.synchronization.pending_updates.contains_key(modality_type) {
                    self.synchronization.pending_updates.insert(*modality_type, Vec::new());
                }
                
                if let Some(events) = self.synchronization.pending_updates.get_mut(modality_type) {
                    events.push(event.clone());
                }
            }
        }
        
        self.synchronization.last_update = std::time::Instant::now();
        
        Ok(())
    }
    
    /// Highlight an element across all active modalities
    pub fn highlight_element(
        &mut self, 
        element_id: &str, 
        highlight_params: Option<&HashMap<String, String>>
    ) -> ModalityResult<()> {
        // Create synchronization event
        let mut event_params = HashMap::new();
        event_params.insert("element_id".to_string(), element_id.to_string());
        
        if let Some(params) = highlight_params {
            for (key, value) in params {
                event_params.insert(format!("highlight_{}", key), value.clone());
            }
        }
        
        let event = SynchronizationEvent {
            event_type: SynchronizationEventType::ElementHighlight,
            timestamp: std::time::Instant::now(),
            params: event_params,
        };
        
        // Acquire synchronization lock
        let _lock = self.synchronization.lock.lock().unwrap();
        
        // Highlight the element in each active modality
        for modality_type in &self.active_modalities {
            if let Some(modality) = self.modalities.get_mut(modality_type) {
                modality.highlight_element(element_id, highlight_params)?;
                
                // Record highlight event
                if !self.synchronization.pending_updates.contains_key(modality_type) {
                    self.synchronization.pending_updates.insert(*modality_type, Vec::new());
                }
                
                if let Some(events) = self.synchronization.pending_updates.get_mut(modality_type) {
                    events.push(event.clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the information capacity for each active modality
    pub fn get_information_capacities(&self) -> HashMap<Modality, f64> {
        let mut capacities = HashMap::new();
        
        for modality_type in &self.active_modalities {
            if let Some(modality) = self.modalities.get(modality_type) {
                capacities.insert(*modality_type, modality.information_capacity());
            }
        }
        
        capacities
    }
    
    /// Get the most suitable modality for a given information type
    pub fn get_suitable_modality(&self, information_type: &str) -> Option<Modality> {
        // Calculate suitability scores for each active modality
        let mut scores = HashMap::new();
        
        for modality_type in &self.active_modalities {
            let score = match (information_type, modality_type) {
                // Spatial information is best presented visually
                ("spatial", &Modality::Visual) => 0.9,
                ("spatial", &Modality::Auditory) => 0.4,
                ("spatial", &Modality::Textual) => 0.3,
                ("spatial", &Modality::Tactile) => 0.5,
                
                // Temporal information is well-suited for auditory modality
                ("temporal", &Modality::Visual) => 0.6,
                ("temporal", &Modality::Auditory) => 0.8,
                ("temporal", &Modality::Textual) => 0.5,
                ("temporal", &Modality::Tactile) => 0.4,
                
                // Categorical information works well in most modalities
                ("categorical", &Modality::Visual) => 0.7,
                ("categorical", &Modality::Auditory) => 0.7,
                ("categorical", &Modality::Textual) => 0.8,
                ("categorical", &Modality::Tactile) => 0.6,
                
                // Quantitative information is best presented visually or textually
                ("quantitative", &Modality::Visual) => 0.8,
                ("quantitative", &Modality::Auditory) => 0.5,
                ("quantitative", &Modality::Textual) => 0.9,
                ("quantitative", &Modality::Tactile) => 0.3,
                
                // Relational information is best presented visually
                ("relational", &Modality::Visual) => 0.9,
                ("relational", &Modality::Auditory) => 0.4,
                ("relational", &Modality::Textual) => 0.6,
                ("relational", &Modality::Tactile) => 0.3,
                
                // Default to medium suitability
                (_, _) => 0.5,
            };
            
            // Adjust score by modality information capacity
            if let Some(modality) = self.modalities.get(modality_type) {
                let capacity = modality.information_capacity();
                scores.insert(*modality_type, score * capacity);
            }
        }
        
        // Find the modality with the highest score
        scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(modality, _)| modality)
    }
    
    /// Transform an element from one modality to another
    pub fn transform_element(
        &self,
        element_id: &str,
        source_modality: Modality,
        target_modality: Modality
    ) -> ModalityResult<RepresentationElement> {
        // Get the element from the source modality
        let source_element = if let Some(modality) = self.modalities.get(&source_modality) {
            modality.get_element(element_id)?
        } else {
            return Err(ModalityError::ModalityNotSupported(
                format!("Source modality {:?} not registered", source_modality)
            ));
        };
        
        // Create parameters for the target modality
        let mut params = HashMap::new();
        
        // Copy semantic properties
        for (key, value) in &source_element.semantic_properties {
            params.insert(format!("semantic_{}", key), value.clone());
        }
        
        // Transform modality-specific properties based on element type
        match (source_modality, target_modality, source_element.element_type.as_str()) {
            // Visual node to auditory node
            (Modality::Visual, Modality::Auditory, "node") => {
                let color = source_element.modality_properties.get("color")
                    .unwrap_or(&"#0000FF".to_string()).to_string();
                
                let pitch = match color.as_str() {
                    "#FF0000" => "high", // Red -> high pitch
                    "#00FF00" => "medium", // Green -> medium pitch
                    "#888888" => "low", // Gray -> low pitch
                    _ => "medium", // Default to medium pitch
                };
                
                params.insert("sound_type".to_string(), "tone".to_string());
                params.insert("pitch".to_string(), pitch.to_string());
                params.insert("duration".to_string(), "0.2".to_string());
                
                // Transform spatial position if available
                if let (Some(x), Some(y)) = (
                    source_element.modality_properties.get("x"),
                    source_element.modality_properties.get("y")
                ) {
                    params.insert("spatial_x".to_string(), x.clone());
                    params.insert("spatial_y".to_string(), y.clone());
                }
            },
            
            // Visual node to textual node
            (Modality::Visual, Modality::Textual, "node") => {
                let color = source_element.modality_properties.get("color")
                    .unwrap_or(&"#0000FF".to_string()).to_string();
                
                let status = match color.as_str() {
                    "#FF0000" => "current", // Red -> current
                    "#00FF00" => "open", // Green -> open
                    "#888888" => "closed", // Gray -> closed
                    _ => "unknown", // Default to unknown
                };
                
                params.insert("text_description".to_string(), 
                             format!("Node {} ({})", element_id.replace("node_", ""), status));
                params.insert("format".to_string(), "plain".to_string());
            },
            
            // More transformations would be defined here...
            
            _ => {
                return Err(ModalityError::TransformationError(
                    format!("Transformation from {:?} to {:?} for element type {} not supported",
                           source_modality, target_modality, source_element.element_type)
                ));
            }
        }
        
        // Create the element in the target modality
        if let Some(modality) = self.modalities.get(&target_modality) {
            modality.get_element(element_id)
                .or_else(|_| {
                    // Element doesn't exist yet in target modality, create it
                    if let Some(modality) = self.modalities.get_mut(&target_modality) {
                        modality.create_element(&source_element.element_type, element_id, &params)
                    } else {
                        Err(ModalityError::ModalityNotSupported(
                            format!("Target modality {:?} not registered", target_modality)
                        ))
                    }
                })
        } else {
            Err(ModalityError::ModalityNotSupported(
                format!("Target modality {:?} not registered", target_modality)
            ))
        }
    }
}

/// Unit tests for the multi-modal representation system
#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test platform capabilities
    struct TestPlatformCapabilities {
        has_display: bool,
        has_audio: bool,
    }
    
    impl PlatformCapabilities {
        /// Create mock platform capabilities for testing
        fn mock(has_display: bool, has_audio: bool) -> Self {
            Self {
                has_display,
                has_audio,
                graphics_tier: if has_display { 2 } else { 0 },
                has_stereo_audio: has_audio,
                has_spatial_audio: has_audio,
                has_speech_synthesis: has_audio,
                // Other fields would be populated here...
            }
        }
    }
    
    /// Test visual modality capabilities
    #[test]
    fn test_visual_modality_capabilities() {
        let capabilities = PlatformCapabilities::mock(true, false);
        let mut modality = VisualModality::new();
        
        assert!(modality.is_supported(&capabilities));
        assert!(modality.initialize(&capabilities).is_ok());
        assert_eq!(modality.modality_type(), Modality::Visual);
        assert!(modality.information_capacity() > 0.0);
    }
    
    /// Test auditory modality capabilities
    #[test]
    fn test_auditory_modality_capabilities() {
        let capabilities = PlatformCapabilities::mock(false, true);
        let mut modality = AuditoryModality::new();
        
        assert!(modality.is_supported(&capabilities));
        // Cannot test initialize without sound system implementation
        assert_eq!(modality.modality_type(), Modality::Auditory);
        assert!(modality.information_capacity() > 0.0);
    }
    
    /// Test multi-modal representation
    #[test]
    fn test_multimodal_representation() {
        let capabilities = PlatformCapabilities::mock(true, true);
        let mut representation = MultiModalRepresentation::new(capabilities.clone());
        
        // Register modalities
        let visual_modality = Box::new(VisualModality::new());
        assert!(representation.register_modality(visual_modality).is_ok());
        
        // Activate visual modality
        assert!(representation.activate_modality(Modality::Visual).is_ok());
        
        // Check information capacities
        let capacities = representation.get_information_capacities();
        assert!(capacities.contains_key(&Modality::Visual));
        assert!(capacities[&Modality::Visual] > 0.0);
        
        // Test suitable modality selection
        let spatial_modality = representation.get_suitable_modality("spatial");
        assert_eq!(spatial_modality, Some(Modality::Visual));
    }
    
    /// Test element creation and transformation
    #[test]
    fn test_element_creation_and_transformation() {
        // TODO: Implement comprehensive tests for element creation and transformation
    }
}