//! # Dynamic Interface Adaptation System
//!
//! This module implements an advanced cognitive adaptation system that dynamically
//! adjusts interfaces based on user cognitive profiles, interaction patterns, and
//! cognitive load measurements. The system employs a mixed-initiative approach that
//! balances automated adaptations with user control to create personalized interfaces
//! that optimize cognitive accessibility while maintaining predictability.
//!
//! ## Theoretical Foundation
//!
//! The adaptation system is grounded in several theoretical frameworks:
//!
//! 1. **Cognitive Fit Theory** (Vessey, 1991): Optimizes the match between task
//!    characteristics, user cognitive traits, and interface representation.
//!
//! 2. **Adaptive User Interface Theory** (Lavie & Meyer, 2010): Balances
//!    adaptation benefits against disruption costs through principled adaptation
//!    decisions.
//!
//! 3. **Progressive Disclosure** (Nielsen, 2006): Manages interface complexity
//!    through gradual revelation of advanced functionality based on user expertise.
//!
//! 4. **Mixed-Initiative Interaction** (Horvitz, 1999): Coordinates user and system
//!    initiatives to optimize cognitive burden and user control.
//!
//! ## Key Components
//!
//! * `UserModel`: Encapsulates user cognitive traits, preferences, and interaction history
//! * `InterfaceAdapter`: Transforms interfaces based on cognitive profiles and load metrics
//! * `AdaptationManager`: Orchestrates adaptation decisions and timing
//! * `AdaptationStrategy`: Defines specific adaptation techniques for different scenarios
//!
//! ## Copyright
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algorithm::state::AlgorithmState;
use crate::cognitive::load::{
    CognitiveLoadEstimator, CognitiveLoadEstimate, CognitiveProfile, 
    ComplexityMetrics, InteractionMetrics, ComplexityTransformation,
    CognitiveLoadError
};
use crate::education::progressive::ExpertiseLevel;
use crate::modality::representation::{ModalityRepresentation, RepresentationError};

/// Adaptation error types
#[derive(Error, Debug)]
pub enum AdaptationError {
    /// Error when adaptation fails due to invalid parameters
    #[error("Invalid adaptation parameters: {0}")]
    InvalidParameters(String),
    
    /// Error when adaptation strategy is not applicable
    #[error("Adaptation strategy not applicable: {0}")]
    StrategyNotApplicable(String),
    
    /// Error when adaptation fails due to resource constraints
    #[error("Resource constraint violation: {0}")]
    ResourceConstraint(String),
    
    /// Error propagated from cognitive load estimation
    #[error("Cognitive load error: {0}")]
    CognitiveLoadError(#[from] CognitiveLoadError),
    
    /// Error in the underlying representation
    #[error("Representation error: {0}")]
    RepresentationError(#[from] RepresentationError),
    
    /// Generic adaptation error
    #[error("Adaptation error: {0}")]
    Generic(String),
}

/// Result type for adaptation operations
pub type AdaptationResult<T> = Result<T, AdaptationError>;

/// Interface element type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterfaceElementType {
    /// Visual element (controls, displays, etc.)
    Visual,
    /// Auditory element (sounds, narration, etc.)
    Auditory,
    /// Tactile element (haptic feedback, etc.)
    Tactile,
    /// Multimodal element (combines multiple modalities)
    Multimodal,
}

/// Interface element importance
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ElementImportance {
    /// Critical elements (must always be present)
    Critical,
    /// High importance elements (should be present for most users)
    High,
    /// Medium importance elements (useful but not essential)
    Medium,
    /// Low importance elements (can be hidden for most users)
    Low,
    /// Optional elements (only shown on demand)
    Optional,
}

impl ElementImportance {
    /// Convert to numeric value for calculations
    pub fn to_value(&self) -> f64 {
        match self {
            ElementImportance::Critical => 1.0,
            ElementImportance::High => 0.8,
            ElementImportance::Medium => 0.6,
            ElementImportance::Low => 0.4,
            ElementImportance::Optional => 0.2,
        }
    }
    
    /// Create from numeric value
    pub fn from_value(value: f64) -> Self {
        if value >= 0.9 {
            ElementImportance::Critical
        } else if value >= 0.7 {
            ElementImportance::High
        } else if value >= 0.5 {
            ElementImportance::Medium
        } else if value >= 0.3 {
            ElementImportance::Low
        } else {
            ElementImportance::Optional
        }
    }
}

/// Interface element representing a UI component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceElement {
    /// Unique element identifier
    pub id: String,
    /// Element type
    pub element_type: InterfaceElementType,
    /// Element importance
    pub importance: ElementImportance,
    /// Element complexity (0.0-1.0)
    pub complexity: f64,
    /// Task association (what task this element supports)
    pub task: Option<String>,
    /// Expertise level required for this element
    pub required_expertise: Option<ExpertiseLevel>,
    /// Dependencies on other elements
    pub dependencies: HashSet<String>,
    /// Element properties
    pub properties: HashMap<String, String>,
}

impl InterfaceElement {
    /// Create a new interface element
    pub fn new(id: &str, element_type: InterfaceElementType, importance: ElementImportance) -> Self {
        Self {
            id: id.to_string(),
            element_type,
            importance,
            complexity: 0.5, // Default medium complexity
            task: None,
            required_expertise: None,
            dependencies: HashSet::new(),
            properties: HashMap::new(),
        }
    }
    
    /// Check if this element should be visible based on user expertise
    pub fn is_visible_for_expertise(&self, expertise: &ExpertiseLevel) -> bool {
        if let Some(required) = &self.required_expertise {
            match (required, expertise) {
                (ExpertiseLevel::Beginner, _) => true,
                (ExpertiseLevel::Intermediate, ExpertiseLevel::Beginner) => false,
                (ExpertiseLevel::Intermediate, _) => true,
                (ExpertiseLevel::Advanced, ExpertiseLevel::Expert) => true,
                (ExpertiseLevel::Advanced, ExpertiseLevel::Advanced) => true,
                (ExpertiseLevel::Advanced, _) => false,
                (ExpertiseLevel::Expert, ExpertiseLevel::Expert) => true,
                (ExpertiseLevel::Expert, _) => false,
            }
        } else {
            true // No expertise requirement specified
        }
    }
}

/// Interface layout representing a collection of elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceLayout {
    /// Interface identifier
    pub id: String,
    /// Layout elements
    pub elements: HashMap<String, InterfaceElement>,
    /// Layout complexity (0.0-1.0)
    pub complexity: f64,
    /// Layout structure (hierarchical organization)
    pub structure: InterfaceStructure,
    /// Layout properties
    pub properties: HashMap<String, String>,
}

impl InterfaceLayout {
    /// Create a new interface layout
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            elements: HashMap::new(),
            complexity: 0.5, // Default medium complexity
            structure: InterfaceStructure::Flat, // Default flat structure
            properties: HashMap::new(),
        }
    }
    
    /// Add an element to the layout
    pub fn add_element(&mut self, element: InterfaceElement) {
        self.elements.insert(element.id.clone(), element);
        
        // Recalculate complexity
        self.update_complexity();
    }
    
    /// Remove an element from the layout
    pub fn remove_element(&mut self, element_id: &str) -> Option<InterfaceElement> {
        let element = self.elements.remove(element_id);
        
        // Recalculate complexity if element was removed
        if element.is_some() {
            self.update_complexity();
        }
        
        element
    }
    
    /// Update layout complexity based on elements
    fn update_complexity(&mut self) {
        if self.elements.is_empty() {
            self.complexity = 0.0;
            return;
        }
        
        // Calculate average element complexity
        let avg_complexity: f64 = self.elements.values()
            .map(|e| e.complexity)
            .sum::<f64>() / self.elements.len() as f64;
        
        // Consider element count in complexity (more elements = higher complexity)
        let count_factor = (self.elements.len() as f64).sqrt() / 10.0; // Square root to dampen effect
        
        // Consider structural complexity
        let structure_factor = match self.structure {
            InterfaceStructure::Flat => 0.8,
            InterfaceStructure::Hierarchical(_) => 1.0,
            InterfaceStructure::Network => 1.2,
        };
        
        // Calculate overall complexity
        self.complexity = (avg_complexity * count_factor * structure_factor).min(1.0);
    }
    
    /// Filter elements based on a predicate
    pub fn filter_elements<F>(&self, predicate: F) -> InterfaceLayout 
    where
        F: Fn(&InterfaceElement) -> bool,
    {
        let mut filtered = self.clone();
        
        // Remove elements that don't satisfy the predicate
        filtered.elements.retain(|_, element| predicate(element));
        
        // Recalculate complexity
        filtered.update_complexity();
        
        filtered
    }
    
    /// Get visible elements for a specific expertise level
    pub fn get_visible_elements(&self, expertise: &ExpertiseLevel) -> InterfaceLayout {
        self.filter_elements(|element| element.is_visible_for_expertise(expertise))
    }
}

/// Interface structure type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterfaceStructure {
    /// Flat structure (all elements at same level)
    Flat,
    /// Hierarchical structure (elements organized in tree)
    Hierarchical(HashMap<String, Vec<String>>), // Parent -> Children mapping
    /// Network structure (elements with arbitrary connections)
    Network,
}

/// Cognitive trait type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveTrait {
    /// Verbal ability
    VerbalAbility,
    /// Spatial ability
    SpatialAbility,
    /// Working memory capacity
    WorkingMemory,
    /// Attention control
    AttentionControl,
    /// Processing speed
    ProcessingSpeed,
    /// Cognitive flexibility
    CognitiveFlexibility,
}

/// Cognitive trait level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraitLevel {
    /// Very low level
    VeryLow,
    /// Low level
    Low,
    /// Average level
    Average,
    /// High level
    High,
    /// Very high level
    VeryHigh,
}

impl TraitLevel {
    /// Convert to numeric value for calculations
    pub fn to_value(&self) -> f64 {
        match self {
            TraitLevel::VeryLow => 0.1,
            TraitLevel::Low => 0.3,
            TraitLevel::Average => 0.5,
            TraitLevel::High => 0.7,
            TraitLevel::VeryHigh => 0.9,
        }
    }
    
    /// Create from numeric value
    pub fn from_value(value: f64) -> Self {
        if value < 0.2 {
            TraitLevel::VeryLow
        } else if value < 0.4 {
            TraitLevel::Low
        } else if value < 0.6 {
            TraitLevel::Average
        } else if value < 0.8 {
            TraitLevel::High
        } else {
            TraitLevel::VeryHigh
        }
    }
}

/// User interaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
    /// Event type
    pub event_type: String,
    /// Target element ID
    pub target_element: Option<String>,
    /// Duration of interaction (if applicable)
    pub duration: Option<Duration>,
    /// Success/failure indicator (if applicable)
    pub success: Option<bool>,
    /// Additional event data
    pub data: HashMap<String, String>,
}

/// User model for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserModel {
    /// User identifier
    pub user_id: String,
    /// Cognitive profile
    pub cognitive_profile: CognitiveProfile,
    /// Cognitive traits
    pub cognitive_traits: HashMap<CognitiveTrait, TraitLevel>,
    /// Interface preferences
    pub interface_preferences: HashMap<String, String>,
    /// Interaction history
    #[serde(skip)]
    pub interaction_history: VecDeque<InteractionEvent>,
    /// Model confidence (0.0-1.0)
    pub confidence: f64,
    /// Last update timestamp
    #[serde(skip)]
    pub last_update: std::time::SystemTime,
}

impl UserModel {
    /// Create a new user model with default values
    pub fn new(user_id: &str) -> Self {
        let mut traits = HashMap::new();
        traits.insert(CognitiveTrait::VerbalAbility, TraitLevel::Average);
        traits.insert(CognitiveTrait::SpatialAbility, TraitLevel::Average);
        traits.insert(CognitiveTrait::WorkingMemory, TraitLevel::Average);
        traits.insert(CognitiveTrait::AttentionControl, TraitLevel::Average);
        traits.insert(CognitiveTrait::ProcessingSpeed, TraitLevel::Average);
        traits.insert(CognitiveTrait::CognitiveFlexibility, TraitLevel::Average);
        
        Self {
            user_id: user_id.to_string(),
            cognitive_profile: CognitiveProfile::default(),
            cognitive_traits: traits,
            interface_preferences: HashMap::new(),
            interaction_history: VecDeque::with_capacity(100), // Keep 100 most recent interactions
            confidence: 0.5, // Initial confidence is medium
            last_update: std::time::SystemTime::now(),
        }
    }
    
    /// Add an interaction event to history
    pub fn add_interaction(&mut self, event: InteractionEvent) {
        // Keep history at maximum capacity
        if self.interaction_history.len() >= 100 {
            self.interaction_history.pop_front();
        }
        
        self.interaction_history.push_back(event);
        self.last_update = std::time::SystemTime::now();
    }
    
    /// Update model based on interaction events
    pub fn update_from_interactions(&mut self) {
        if self.interaction_history.is_empty() {
            return;
        }
        
        // Update traits based on interaction patterns
        self.update_traits_from_interactions();
        
        // Update preferences based on successful interactions
        self.update_preferences_from_interactions();
        
        // Increase confidence with more interaction data
        self.update_confidence();
        
        self.last_update = std::time::SystemTime::now();
    }
    
    /// Update cognitive traits based on interaction patterns
    fn update_traits_from_interactions(&mut self) {
        // This is a simplified implementation - a real system would
        // use more sophisticated analysis of interaction patterns
        
        // Count successful vs. failed interactions
        let (success_count, fail_count) = self.interaction_history.iter()
            .filter_map(|e| e.success)
            .fold((0, 0), |(s, f), success| {
                if success { (s + 1, f) } else { (s, f + 1) }
            });
        
        // Calculate success rate
        let total = success_count + fail_count;
        let success_rate = if total > 0 {
            success_count as f64 / total as f64
        } else {
            0.5 // Default if no success/fail data
        };
        
        // Update processing speed based on interaction durations
        let avg_duration = self.interaction_history.iter()
            .filter_map(|e| e.duration)
            .map(|d| d.as_secs_f64())
            .collect::<Vec<_>>();
        
        if !avg_duration.is_empty() {
            let avg = avg_duration.iter().sum::<f64>() / avg_duration.len() as f64;
            
            // Assuming shorter durations indicate higher processing speed
            // Normalize to 0-1 range assuming 5s is very slow, 0.5s is very fast
            let speed_value = ((5.0 - avg) / 4.5).clamp(0.0, 1.0);
            
            let speed_level = TraitLevel::from_value(speed_value);
            self.cognitive_traits.insert(CognitiveTrait::ProcessingSpeed, speed_level);
        }
        
        // Update working memory based on task complexity and success rate
        // This is a placeholder - real implementation would be more sophisticated
        if success_rate > 0.8 {
            // High success rate might indicate good working memory
            let current = self.cognitive_traits.get(&CognitiveTrait::WorkingMemory)
                .copied()
                .unwrap_or(TraitLevel::Average);
            
            // Gradually increase trait level
            let new_value = (current.to_value() + 0.05).min(0.9);
            self.cognitive_traits.insert(
                CognitiveTrait::WorkingMemory, 
                TraitLevel::from_value(new_value)
            );
        } else if success_rate < 0.4 {
            // Low success rate might indicate working memory limitations
            let current = self.cognitive_traits.get(&CognitiveTrait::WorkingMemory)
                .copied()
                .unwrap_or(TraitLevel::Average);
            
            // Gradually decrease trait level
            let new_value = (current.to_value() - 0.05).max(0.1);
            self.cognitive_traits.insert(
                CognitiveTrait::WorkingMemory, 
                TraitLevel::from_value(new_value)
            );
        }
    }
    
    /// Update preferences based on interaction patterns
    fn update_preferences_from_interactions(&mut self) {
        // Count element interactions
        let mut element_counts = HashMap::new();
        
        for event in &self.interaction_history {
            if let Some(element_id) = &event.target_element {
                let count = element_counts.entry(element_id.clone()).or_insert(0);
                *count += 1;
            }
        }
        
        // Find most frequently used elements
        if !element_counts.is_empty() {
            let max_element = element_counts.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(element, _)| element.clone())
                .unwrap();
            
            // Update preferences to prioritize frequently used elements
            self.interface_preferences.insert("prioritize_element".to_string(), max_element);
        }
        
        // Analyze successful interactions for modality preferences
        let modality_successes = self.interaction_history.iter()
            .filter_map(|e| {
                if let (Some(true), Some(element_id)) = (e.success, &e.target_element) {
                    // Extract modality from element ID (assuming format like "visual_element_1")
                    if element_id.starts_with("visual_") {
                        Some("visual")
                    } else if element_id.starts_with("auditory_") {
                        Some("auditory")
                    } else if element_id.starts_with("tactile_") {
                        Some("tactile")
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        
        // Count successes by modality
        let mut modality_counts = HashMap::new();
        for modality in modality_successes {
            let count = modality_counts.entry(modality).or_insert(0);
            *count += 1;
        }
        
        // Find most successful modality
        if !modality_counts.is_empty() {
            let max_modality = modality_counts.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&modality, _)| modality)
                .unwrap();
            
            // Update preferred modality in cognitive profile
            self.cognitive_profile.preferred_modality = max_modality.to_string();
        }
    }
    
    /// Update model confidence based on data quantity
    fn update_confidence(&mut self) {
        // More interaction data increases confidence
        let interaction_count = self.interaction_history.len() as f64;
        
        // Sigmoid function for confidence: 50 interactions -> 0.8 confidence
        let new_confidence = 1.0 / (1.0 + (-0.1 * (interaction_count - 30.0)).exp());
        
        // Ensure confidence stays in [0.1, 0.95] range and only increases gradually
        self.confidence = (self.confidence * 0.8 + new_confidence * 0.2).clamp(0.1, 0.95);
    }
    
    /// Get trait value as normalized float
    pub fn get_trait_value(&self, trait_type: CognitiveTrait) -> f64 {
        self.cognitive_traits.get(&trait_type)
            .copied()
            .unwrap_or(TraitLevel::Average)
            .to_value()
    }
    
    /// Get interface preference
    pub fn get_preference(&self, key: &str) -> Option<&str> {
        self.interface_preferences.get(key).map(|s| s.as_str())
    }
    
    /// Set interface preference
    pub fn set_preference(&mut self, key: &str, value: &str) {
        self.interface_preferences.insert(key.to_string(), value.to_string());
    }
}

/// Adaptation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Show/hide elements
    Visibility,
    /// Reorder elements
    Ordering,
    /// Change presentation style
    Style,
    /// Change modality
    Modality,
    /// Change information density
    Density,
    /// Change interaction method
    Interaction,
}

/// Adaptation effect on user experience
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationEffect {
    /// Cognitive load reduction (0.0-1.0)
    pub load_reduction: f64,
    /// Usability impact (-1.0 to 1.0, positive is better)
    pub usability_impact: f64,
    /// Learning curve impact (-1.0 to 1.0, negative means harder to learn)
    pub learning_impact: f64,
    /// User satisfaction impact (-1.0 to 1.0, positive is better)
    pub satisfaction_impact: f64,
    /// Adaptation disruption (0.0-1.0, higher is more disruptive)
    pub disruption: f64,
}

impl AdaptationEffect {
    /// Calculate overall effect score (higher is better)
    pub fn overall_score(&self) -> f64 {
        // Weighted sum of components
        (self.load_reduction * 0.3) +
        (self.usability_impact * 0.2) +
        (self.learning_impact * 0.2) +
        (self.satisfaction_impact * 0.2) -
        (self.disruption * 0.1)
    }
}

/// Interface adaptation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationAction {
    /// Adaptation type
    pub adaptation_type: AdaptationType,
    /// Target elements
    pub target_elements: Vec<String>,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Estimated effects
    pub estimated_effects: AdaptationEffect,
    /// Explanation for the adaptation
    pub explanation: String,
    /// Revert action information (to undo this adaptation)
    pub revert_action: Option<Box<AdaptationAction>>,
}

impl AdaptationAction {
    /// Create a new adaptation action
    pub fn new(adaptation_type: AdaptationType, 
              target_elements: Vec<String>, 
              effects: AdaptationEffect) -> Self {
        Self {
            adaptation_type,
            target_elements,
            parameters: HashMap::new(),
            estimated_effects: effects,
            explanation: String::new(),
            revert_action: None,
        }
    }
    
    /// Set a parameter
    pub fn with_parameter(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Set explanation
    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = explanation.to_string();
        self
    }
    
    /// Set revert action
    pub fn with_revert(mut self, revert: AdaptationAction) -> Self {
        self.revert_action = Some(Box::new(revert));
        self
    }
}

/// Adaptation history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationHistoryEntry {
    /// Timestamp of adaptation
    pub timestamp: std::time::SystemTime,
    /// Adaptation action
    pub action: AdaptationAction,
    /// Was adaptation successful
    pub success: bool,
    /// User response (-1.0 to 1.0, positive means user liked it)
    pub user_response: Option<f64>,
    /// Pre-adaptation cognitive load
    pub pre_load: Option<f64>,
    /// Post-adaptation cognitive load
    pub post_load: Option<f64>,
}

/// Adaptation timing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationTiming {
    /// Immediate adaptation
    Immediate,
    /// Adaptation at next logical breakpoint
    Breakpoint,
    /// Adaptation after delay
    Delayed(Duration),
    /// Adaptation on user request
    OnRequest,
}

/// Adaptation notification level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NotificationLevel {
    /// No notification of adaptation
    None,
    /// Subtle indication of adaptation
    Subtle,
    /// Clear notification of adaptation
    Clear,
    /// Request confirmation before adaptation
    Confirmation,
}

/// Adaptation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Threshold for triggering adaptation (0.0-1.0)
    pub adaptation_threshold: f64,
    /// Minimum cognitive load reduction to justify adaptation
    pub min_load_reduction: f64,
    /// Maximum acceptable disruption
    pub max_disruption: f64,
    /// Adaptation timing strategy
    pub timing: AdaptationTiming,
    /// Notification level
    pub notification: NotificationLevel,
    /// Minimum time between adaptations
    pub min_interval: Duration,
    /// Whether to enable/disable specific adaptation types
    pub enabled_adaptations: HashSet<AdaptationType>,
    /// User override preferences
    pub user_overrides: HashMap<String, String>,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        let mut enabled = HashSet::new();
        enabled.insert(AdaptationType::Visibility);
        enabled.insert(AdaptationType::Ordering);
        enabled.insert(AdaptationType::Style);
        enabled.insert(AdaptationType::Modality);
        enabled.insert(AdaptationType::Density);
        enabled.insert(AdaptationType::Interaction);
        
        Self {
            adaptation_threshold: 0.7, // Adapt when cognitive load > 70%
            min_load_reduction: 0.1, // At least 10% load reduction to justify adaptation
            max_disruption: 0.5, // No more than 50% disruption
            timing: AdaptationTiming::Breakpoint,
            notification: NotificationLevel::Subtle,
            min_interval: Duration::from_secs(60), // At least 1 minute between adaptations
            enabled_adaptations: enabled,
            user_overrides: HashMap::new(),
        }
    }
}

/// Interface adaptation strategy
pub trait AdaptationStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;
    
    /// Strategy description
    fn description(&self) -> &str;
    
    /// Check if strategy is applicable to current state
    fn is_applicable(&self, 
                   user_model: &UserModel, 
                   current_layout: &InterfaceLayout,
                   load_estimate: &CognitiveLoadEstimate) -> bool;
    
    /// Generate adaptation actions
    fn generate_actions(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      load_estimate: &CognitiveLoadEstimate) -> AdaptationResult<Vec<AdaptationAction>>;
    
    /// Apply actions to the interface layout
    fn apply_actions(&self,
                   current_layout: &InterfaceLayout,
                   actions: &[AdaptationAction]) -> AdaptationResult<InterfaceLayout>;
    
    /// Estimate effects of adaptation
    fn estimate_effects(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      actions: &[AdaptationAction]) -> AdaptationResult<AdaptationEffect>;
}

/// Visibility adaptation strategy
pub struct VisibilityStrategy;

impl AdaptationStrategy for VisibilityStrategy {
    fn name(&self) -> &str {
        "Visibility Adaptation"
    }
    
    fn description(&self) -> &str {
        "Adapts interface by showing or hiding elements based on importance and user needs"
    }
    
    fn is_applicable(&self, 
                   user_model: &UserModel, 
                   current_layout: &InterfaceLayout,
                   load_estimate: &CognitiveLoadEstimate) -> bool {
        // Applicable if there are more than minimal elements
        // and cognitive load is high
        current_layout.elements.len() > 3 && 
        load_estimate.overall_load > user_model.cognitive_profile.load_tolerance
    }
    
    fn generate_actions(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      load_estimate: &CognitiveLoadEstimate) -> AdaptationResult<Vec<AdaptationAction>> {
        // This is a simplified implementation - a real system would
        // use more sophisticated analysis and selection
        
        let mut actions = Vec::new();
        
        // Sort elements by importance
        let mut elements: Vec<(&String, &InterfaceElement)> = 
            current_layout.elements.iter().collect();
        
        elements.sort_by(|a, b| b.1.importance.cmp(&a.1.importance));
        
        // If cognitive load is high, hide low importance elements
        if load_estimate.overall_load > user_model.cognitive_profile.load_tolerance {
            // Get elements to hide (low importance elements)
            let elements_to_hide: Vec<String> = elements.iter()
                .filter(|(_, element)| {
                    element.importance == ElementImportance::Low ||
                    element.importance == ElementImportance::Optional
                })
                .map(|(id, _)| (*id).clone())
                .collect();
            
            if !elements_to_hide.is_empty() {
                // Create action to hide elements
                let effects = AdaptationEffect {
load_reduction: 0.15, // Estimated load reduction from hiding elements
                    usability_impact: -0.05, // Slight negative impact on usability
                    learning_impact: 0.0, // Neutral learning impact
                    satisfaction_impact: 0.1, // Slight positive impact (less cluttered)
                    disruption: 0.3, // Moderate disruption from hiding elements
                };
                
                let hide_action = AdaptationAction::new(
                    AdaptationType::Visibility,
                    elements_to_hide.clone(),
                    effects
                )
                .with_parameter("visibility", "hidden")
                .with_explanation("Hiding low importance elements to reduce cognitive load");
                
                // Create revert action (to show elements again)
                let revert_effects = AdaptationEffect {
                    load_reduction: -0.15, // Negative load reduction (increases load)
                    usability_impact: 0.05, // Slight positive usability impact
                    learning_impact: 0.0, // Neutral learning impact
                    satisfaction_impact: -0.1, // Slight negative impact (more cluttered)
                    disruption: 0.3, // Moderate disruption from showing elements
                };
                
                let revert_action = AdaptationAction::new(
                    AdaptationType::Visibility,
                    elements_to_hide,
                    revert_effects
                )
                .with_parameter("visibility", "visible")
                .with_explanation("Restoring visibility of previously hidden elements");
                
                actions.push(hide_action.with_revert(revert_action));
            }
        } else if load_estimate.overall_load < 0.5 * user_model.cognitive_profile.load_tolerance {
            // If cognitive load is very low, show more elements
            
            // Get currently hidden elements (assuming they're stored in a property)
            let hidden_elements_str = current_layout.properties
                .get("hidden_elements")
                .cloned()
                .unwrap_or_else(|| String::new());
            
            let elements_to_show: Vec<String> = hidden_elements_str
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            
            if !elements_to_show.is_empty() {
                let effects = AdaptationEffect {
                    load_reduction: -0.1, // Negative load reduction (increases load)
                    usability_impact: 0.1, // Positive usability impact
                    learning_impact: 0.05, // Slight positive learning impact
                    satisfaction_impact: 0.0, // Neutral satisfaction impact
                    disruption: 0.2, // Some disruption from showing elements
                };
                
                let show_action = AdaptationAction::new(
                    AdaptationType::Visibility,
                    elements_to_show.clone(),
                    effects
                )
                .with_parameter("visibility", "visible")
                .with_explanation("Showing more elements due to low cognitive load");
                
                // Create revert action (to hide elements again)
                let revert_effects = AdaptationEffect {
                    load_reduction: 0.1, // Positive load reduction
                    usability_impact: -0.1, // Negative usability impact
                    learning_impact: -0.05, // Slight negative learning impact
                    satisfaction_impact: 0.0, // Neutral satisfaction impact
                    disruption: 0.2, // Some disruption from hiding elements
                };
                
                let revert_action = AdaptationAction::new(
                    AdaptationType::Visibility,
                    elements_to_show,
                    revert_effects
                )
                .with_parameter("visibility", "hidden")
                .with_explanation("Hiding elements to manage cognitive load");
                
                actions.push(show_action.with_revert(revert_action));
            }
        }
        
        Ok(actions)
    }
    
    fn apply_actions(&self,
                   current_layout: &InterfaceLayout,
                   actions: &[AdaptationAction]) -> AdaptationResult<InterfaceLayout> {
        let mut adapted_layout = current_layout.clone();
        
        for action in actions {
            if action.adaptation_type != AdaptationType::Visibility {
                continue; // Skip non-visibility actions
            }
            
            let visibility = action.parameters.get("visibility")
                .cloned()
                .unwrap_or_else(|| "visible".to_string());
            
            if visibility == "hidden" {
                // Hide elements
                let mut hidden_elements = adapted_layout.properties
                    .get("hidden_elements")
                    .cloned()
                    .unwrap_or_else(|| String::new());
                
                for element_id in &action.target_elements {
                    // Actually remove element from layout
                    adapted_layout.elements.remove(element_id);
                    
                    // Add to hidden elements list
                    if !hidden_elements.is_empty() {
                        hidden_elements.push(',');
                    }
                    hidden_elements.push_str(element_id);
                }
                
                adapted_layout.properties.insert("hidden_elements".to_string(), hidden_elements);
            } else if visibility == "visible" {
                // Show elements
                // This is a placeholder - would need to retrieve element details from elsewhere
                let mut hidden_elements = adapted_layout.properties
                    .get("hidden_elements")
                    .cloned()
                    .unwrap_or_else(|| String::new());
                
                // Update hidden elements list
                let remaining_hidden: Vec<String> = hidden_elements.split(',')
                    .filter(|s| !s.is_empty() && !action.target_elements.contains(&s.to_string()))
                    .map(|s| s.to_string())
                    .collect();
                
                adapted_layout.properties.insert(
                    "hidden_elements".to_string(), 
                    remaining_hidden.join(",")
                );
            }
        }
        
        // Recalculate layout complexity
        adapted_layout.update_complexity();
        
        Ok(adapted_layout)
    }
    
    fn estimate_effects(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      actions: &[AdaptationAction]) -> AdaptationResult<AdaptationEffect> {
        // Combine effects from all actions
        let mut combined_effect = AdaptationEffect {
            load_reduction: 0.0,
            usability_impact: 0.0,
            learning_impact: 0.0,
            satisfaction_impact: 0.0,
            disruption: 0.0,
        };
        
        for action in actions {
            // Only consider visibility actions
            if action.adaptation_type != AdaptationType::Visibility {
                continue;
            }
            
            // Add effects, applying user trait modifiers
            
            // Working memory affects load reduction
            let working_memory = user_model.get_trait_value(CognitiveTrait::WorkingMemory);
            let adjusted_load_reduction = action.estimated_effects.load_reduction * 
                                         (2.0 - working_memory); // Lower working memory = higher benefit
            
            // Cognitive flexibility affects disruption
            let cognitive_flexibility = user_model.get_trait_value(CognitiveTrait::CognitiveFlexibility);
            let adjusted_disruption = action.estimated_effects.disruption * 
                                     (2.0 - cognitive_flexibility); // Lower flexibility = higher disruption
            
            combined_effect.load_reduction += adjusted_load_reduction;
            combined_effect.usability_impact += action.estimated_effects.usability_impact;
            combined_effect.learning_impact += action.estimated_effects.learning_impact;
            combined_effect.satisfaction_impact += action.estimated_effects.satisfaction_impact;
            combined_effect.disruption += adjusted_disruption;
        }
        
        // Apply diminishing returns for multiple actions
        if actions.len() > 1 {
            let action_count = actions.len() as f64;
            combined_effect.load_reduction = combined_effect.load_reduction * 
                                            (1.0 + (action_count - 1.0) * 0.5) / action_count;
            combined_effect.disruption = combined_effect.disruption * 
                                        (1.0 + (action_count - 1.0) * 0.3) / action_count;
        }
        
        // Ensure values stay in valid ranges
        combined_effect.load_reduction = combined_effect.load_reduction.clamp(0.0, 1.0);
        combined_effect.usability_impact = combined_effect.usability_impact.clamp(-1.0, 1.0);
        combined_effect.learning_impact = combined_effect.learning_impact.clamp(-1.0, 1.0);
        combined_effect.satisfaction_impact = combined_effect.satisfaction_impact.clamp(-1.0, 1.0);
        combined_effect.disruption = combined_effect.disruption.clamp(0.0, 1.0);
        
        Ok(combined_effect)
    }
}

/// Style adaptation strategy
pub struct StyleStrategy;

impl AdaptationStrategy for StyleStrategy {
    fn name(&self) -> &str {
        "Style Adaptation"
    }
    
    fn description(&self) -> &str {
        "Adapts interface presentation style based on user preferences and cognitive load"
    }
    
    fn is_applicable(&self, 
                   user_model: &UserModel, 
                   current_layout: &InterfaceLayout,
                   load_estimate: &CognitiveLoadEstimate) -> bool {
        // Style adaptation is broadly applicable
        true
    }
    
    fn generate_actions(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      load_estimate: &CognitiveLoadEstimate) -> AdaptationResult<Vec<AdaptationAction>> {
        let mut actions = Vec::new();
        
        // Get current style
        let current_style = current_layout.properties
            .get("style")
            .cloned()
            .unwrap_or_else(|| "standard".to_string());
        
        // If cognitive load is high, simplify style
        if load_estimate.overall_load > user_model.cognitive_profile.load_tolerance {
            // Only adapt if not already using simplified style
            if current_style != "simplified" {
                let effects = AdaptationEffect {
                    load_reduction: 0.1, // Moderate load reduction
                    usability_impact: 0.05, // Slight usability improvement
                    learning_impact: -0.05, // Slight negative learning impact
                    satisfaction_impact: 0.0, // Neutral satisfaction
                    disruption: 0.2, // Moderate disruption
                };
                
                let action = AdaptationAction::new(
                    AdaptationType::Style,
                    vec!["layout".to_string()], // Target entire layout
                    effects
                )
                .with_parameter("style", "simplified")
                .with_explanation("Simplifying interface style to reduce cognitive load");
                
                // Create revert action
                let revert_effects = AdaptationEffect {
                    load_reduction: -0.1, // Negative load reduction
                    usability_impact: -0.05, // Slight usability reduction
                    learning_impact: 0.05, // Slight positive learning impact
                    satisfaction_impact: 0.0, // Neutral satisfaction
                    disruption: 0.2, // Moderate disruption
                };
                
                let revert_action = AdaptationAction::new(
                    AdaptationType::Style,
                    vec!["layout".to_string()],
                    revert_effects
                )
                .with_parameter("style", &current_style)
                .with_explanation("Restoring previous interface style");
                
                actions.push(action.with_revert(revert_action));
            }
        } else if load_estimate.overall_load < 0.5 * user_model.cognitive_profile.load_tolerance {
            // If cognitive load is very low, can use more elaborate style
            
            // Check user preference for style
            let preferred_style = user_model.get_preference("preferred_style")
                .unwrap_or("standard");
            
            // Only adapt if current style is not the preferred style
            if current_style != preferred_style && preferred_style != "simplified" {
                let effects = AdaptationEffect {
                    load_reduction: -0.05, // Slight increase in load
                    usability_impact: 0.1, // Positive usability impact
                    learning_impact: 0.1, // Positive learning impact
                    satisfaction_impact: 0.15, // Positive satisfaction impact
                    disruption: 0.25, // Moderate disruption
                };
                
                let action = AdaptationAction::new(
                    AdaptationType::Style,
                    vec!["layout".to_string()],
                    effects
                )
                .with_parameter("style", preferred_style)
                .with_explanation("Applying preferred interface style due to low cognitive load");
                
                // Create revert action
                let revert_effects = AdaptationEffect {
                    load_reduction: 0.05, // Slight reduction in load
                    usability_impact: -0.1, // Negative usability impact
                    learning_impact: -0.1, // Negative learning impact
                    satisfaction_impact: -0.15, // Negative satisfaction impact
                    disruption: 0.25, // Moderate disruption
                };
                
                let revert_action = AdaptationAction::new(
                    AdaptationType::Style,
                    vec!["layout".to_string()],
                    revert_effects
                )
                .with_parameter("style", &current_style)
                .with_explanation("Restoring previous interface style");
                
                actions.push(action.with_revert(revert_action));
            }
        }
        
        Ok(actions)
    }
    
    fn apply_actions(&self,
                   current_layout: &InterfaceLayout,
                   actions: &[AdaptationAction]) -> AdaptationResult<InterfaceLayout> {
        let mut adapted_layout = current_layout.clone();
        
        for action in actions {
            if action.adaptation_type != AdaptationType::Style {
                continue; // Skip non-style actions
            }
            
            if let Some(style) = action.parameters.get("style") {
                adapted_layout.properties.insert("style".to_string(), style.clone());
                
                // Additional style-specific adaptations could be applied here
                // For example, changing color scheme, font sizes, spacing, etc.
            }
        }
        
        Ok(adapted_layout)
    }
    
    fn estimate_effects(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      actions: &[AdaptationAction]) -> AdaptationResult<AdaptationEffect> {
        // Similar to visibility strategy, but with style-specific adjustments
        let mut combined_effect = AdaptationEffect {
            load_reduction: 0.0,
            usability_impact: 0.0,
            learning_impact: 0.0,
            satisfaction_impact: 0.0,
            disruption: 0.0,
        };
        
        for action in actions {
            if action.adaptation_type != AdaptationType::Style {
                continue;
            }
            
            // Apply user trait modifiers
            let visual_preference = user_model.cognitive_profile.preferred_modality == "visual";
            let style_multiplier = if visual_preference { 1.2 } else { 0.8 };
            
            combined_effect.load_reduction += action.estimated_effects.load_reduction;
            combined_effect.usability_impact += action.estimated_effects.usability_impact * style_multiplier;
            combined_effect.learning_impact += action.estimated_effects.learning_impact;
            combined_effect.satisfaction_impact += action.estimated_effects.satisfaction_impact * style_multiplier;
            combined_effect.disruption += action.estimated_effects.disruption;
        }
        
        // Normalize values
        combined_effect.load_reduction = combined_effect.load_reduction.clamp(0.0, 1.0);
        combined_effect.usability_impact = combined_effect.usability_impact.clamp(-1.0, 1.0);
        combined_effect.learning_impact = combined_effect.learning_impact.clamp(-1.0, 1.0);
        combined_effect.satisfaction_impact = combined_effect.satisfaction_impact.clamp(-1.0, 1.0);
        combined_effect.disruption = combined_effect.disruption.clamp(0.0, 1.0);
        
        Ok(combined_effect)
    }
}

/// Modality adaptation strategy
pub struct ModalityStrategy;

impl AdaptationStrategy for ModalityStrategy {
    fn name(&self) -> &str {
        "Modality Adaptation"
    }
    
    fn description(&self) -> &str {
        "Adapts interface by changing presentation modality based on user preferences and needs"
    }
    
    fn is_applicable(&self, 
                   user_model: &UserModel, 
                   current_layout: &InterfaceLayout,
                   load_estimate: &CognitiveLoadEstimate) -> bool {
        // Applicable if we have elements that can use different modalities
        current_layout.elements.values().any(|e| {
            e.element_type == InterfaceElementType::Visual || 
            e.element_type == InterfaceElementType::Auditory ||
            e.element_type == InterfaceElementType::Multimodal
        })
    }
    
    fn generate_actions(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      load_estimate: &CognitiveLoadEstimate) -> AdaptationResult<Vec<AdaptationAction>> {
        let mut actions = Vec::new();
        
        // Get user's preferred modality
        let preferred_modality = &user_model.cognitive_profile.preferred_modality;
        
        // Current primary modality (from layout properties)
        let current_modality = current_layout.properties
            .get("primary_modality")
            .cloned()
            .unwrap_or_else(|| "visual".to_string());
        
        // If preferred modality differs from current and cognitive load is high
        if &current_modality != preferred_modality && 
           load_estimate.overall_load > 0.7 * user_model.cognitive_profile.load_tolerance {
            
            // Find elements that can be adapted to preferred modality
            let adaptable_elements: Vec<String> = current_layout.elements.values()
                .filter(|e| {
                    match (&e.element_type, preferred_modality.as_str()) {
                        (InterfaceElementType::Visual, "auditory") |
                        (InterfaceElementType::Auditory, "visual") |
                        (InterfaceElementType::Multimodal, _) => true,
                        _ => false
                    }
                })
                .map(|e| e.id.clone())
                .collect();
            
            if !adaptable_elements.is_empty() {
                let effects = AdaptationEffect {
                    load_reduction: 0.2, // Significant load reduction using preferred modality
                    usability_impact: 0.15, // Positive usability impact
                    learning_impact: 0.1, // Positive learning impact
                    satisfaction_impact: 0.2, // Positive satisfaction impact
                    disruption: 0.4, // Significant disruption
                };
                
                let action = AdaptationAction::new(
                    AdaptationType::Modality,
                    adaptable_elements.clone(),
                    effects
                )
                .with_parameter("modality", preferred_modality)
                .with_explanation(format!("Changing to {} modality based on your preferences", preferred_modality));
                
                // Create revert action
                let revert_effects = AdaptationEffect {
                    load_reduction: -0.2, // Negative load reduction
                    usability_impact: -0.15, // Negative usability impact
                    learning_impact: -0.1, // Negative learning impact
                    satisfaction_impact: -0.2, // Negative satisfaction impact
                    disruption: 0.4, // Significant disruption
                };
                
                let revert_action = AdaptationAction::new(
                    AdaptationType::Modality,
                    adaptable_elements,
                    revert_effects
                )
                .with_parameter("modality", &current_modality)
                .with_explanation("Restoring previous modality");
                
                actions.push(action.with_revert(revert_action));
            }
        }
        
        Ok(actions)
    }
    
    fn apply_actions(&self,
                   current_layout: &InterfaceLayout,
                   actions: &[AdaptationAction]) -> AdaptationResult<InterfaceLayout> {
        let mut adapted_layout = current_layout.clone();
        
        for action in actions {
            if action.adaptation_type != AdaptationType::Modality {
                continue; // Skip non-modality actions
            }
            
            if let Some(modality) = action.parameters.get("modality") {
                // Update layout primary modality
                adapted_layout.properties.insert("primary_modality".to_string(), modality.clone());
                
                // Update individual elements
                for element_id in &action.target_elements {
                    if let Some(element) = adapted_layout.elements.get_mut(element_id) {
                        // Set element to use this modality
                        element.properties.insert("modality".to_string(), modality.clone());
                        
                        // Update element type if needed
                        match modality.as_str() {
                            "visual" => {
                                element.element_type = InterfaceElementType::Visual;
                            },
                            "auditory" => {
                                element.element_type = InterfaceElementType::Auditory;
                            },
                            "tactile" => {
                                element.element_type = InterfaceElementType::Tactile;
                            },
                            "multimodal" => {
                                element.element_type = InterfaceElementType::Multimodal;
                            },
                            _ => {
                                // Unknown modality
                            },
                        }
                    }
                }
            }
        }
        
        Ok(adapted_layout)
    }
    
    fn estimate_effects(&self,
                      user_model: &UserModel,
                      current_layout: &InterfaceLayout,
                      actions: &[AdaptationAction]) -> AdaptationResult<AdaptationEffect> {
        let mut combined_effect = AdaptationEffect {
            load_reduction: 0.0,
            usability_impact: 0.0,
            learning_impact: 0.0,
            satisfaction_impact: 0.0,
            disruption: 0.0,
        };
        
        for action in actions {
            if action.adaptation_type != AdaptationType::Modality {
                continue;
            }
            
            // Check if modality matches preference
            if let Some(modality) = action.parameters.get("modality") {
                let preference_match = modality == &user_model.cognitive_profile.preferred_modality;
                let multiplier = if preference_match { 1.5 } else { 0.7 };
                
                // Modality changes have potentially large effects
                combined_effect.load_reduction += action.estimated_effects.load_reduction * multiplier;
                combined_effect.usability_impact += action.estimated_effects.usability_impact * multiplier;
                combined_effect.learning_impact += action.estimated_effects.learning_impact;
                combined_effect.satisfaction_impact += action.estimated_effects.satisfaction_impact * multiplier;
                combined_effect.disruption += action.estimated_effects.disruption;
            }
        }
        
        // Normalize values
        combined_effect.load_reduction = combined_effect.load_reduction.clamp(0.0, 1.0);
        combined_effect.usability_impact = combined_effect.usability_impact.clamp(-1.0, 1.0);
        combined_effect.learning_impact = combined_effect.learning_impact.clamp(-1.0, 1.0);
        combined_effect.satisfaction_impact = combined_effect.satisfaction_impact.clamp(-1.0, 1.0);
        combined_effect.disruption = combined_effect.disruption.clamp(0.0, 1.0);
        
        Ok(combined_effect)
    }
}

/// Adaptation manager for coordinating interface adaptations
pub struct AdaptationManager {
    /// User model
    user_model: Arc<RwLock<UserModel>>,
    /// Cognitive load estimator
    load_estimator: Arc<Mutex<CognitiveLoadEstimator>>,
    /// Adaptation strategies
    strategies: Vec<Box<dyn AdaptationStrategy>>,
    /// Adaptation parameters
    parameters: AdaptationParameters,
    /// Current interface layout
    current_layout: Arc<RwLock<InterfaceLayout>>,
    /// Adaptation history
    history: Vec<AdaptationHistoryEntry>,
    /// Last adaptation timestamp
    last_adaptation: Instant,
    /// Pending adaptations
    pending_adaptations: Vec<AdaptationAction>,
}

impl AdaptationManager {
    /// Create a new adaptation manager
    pub fn new(
        user_model: Arc<RwLock<UserModel>>,
        load_estimator: Arc<Mutex<CognitiveLoadEstimator>>,
        current_layout: Arc<RwLock<InterfaceLayout>>,
    ) -> Self {
        let mut strategies: Vec<Box<dyn AdaptationStrategy>> = Vec::new();
        strategies.push(Box::new(VisibilityStrategy));
        strategies.push(Box::new(StyleStrategy));
        strategies.push(Box::new(ModalityStrategy));
        
        Self {
            user_model,
            load_estimator,
            strategies,
            parameters: AdaptationParameters::default(),
            current_layout,
            history: Vec::new(),
            last_adaptation: Instant::now(),
            pending_adaptations: Vec::new(),
        }
    }
    
    /// Add a new adaptation strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn AdaptationStrategy>) {
        self.strategies.push(strategy);
    }
    
    /// Set adaptation parameters
    pub fn set_parameters(&mut self, parameters: AdaptationParameters) {
        self.parameters = parameters;
    }
    
    /// Get the current adaptation parameters
    pub fn get_parameters(&self) -> &AdaptationParameters {
        &self.parameters
    }
    
    /// Update adaptation based on current cognitive load
    pub fn update(&mut self, load_estimate: &CognitiveLoadEstimate) -> AdaptationResult<()> {
        // Check if it's too soon for another adaptation
        if self.last_adaptation.elapsed() < self.parameters.min_interval {
            return Ok(());
        }
        
        // Check if load is above threshold
        if load_estimate.overall_load <= self.parameters.adaptation_threshold {
            // No adaptation needed
            return Ok(());
        }
        
        // Get current user model and layout
        let user_model = self.user_model.read().unwrap().clone();
        let current_layout = self.current_layout.read().unwrap().clone();
        
        // Generate candidate adaptations from all strategies
        let mut candidate_actions = Vec::new();
        
        for strategy in &self.strategies {
            // Skip disabled adaptation types
            let strategy_name = strategy.name().to_lowercase();
            let adaptation_type = if strategy_name.contains("visibility") {
                AdaptationType::Visibility
            } else if strategy_name.contains("style") {
                AdaptationType::Style
            } else if strategy_name.contains("modality") {
                AdaptationType::Modality
            } else if strategy_name.contains("dens") {
                AdaptationType::Density
            } else if strategy_name.contains("order") {
                AdaptationType::Ordering
            } else {
                AdaptationType::Interaction
            };
            
            if !self.parameters.enabled_adaptations.contains(&adaptation_type) {
                continue;
            }
            
            // Check if strategy is applicable
            if !strategy.is_applicable(&user_model, &current_layout, load_estimate) {
                continue;
            }
            
            // Generate actions
            if let Ok(actions) = strategy.generate_actions(&user_model, &current_layout, load_estimate) {
                candidate_actions.extend(actions);
            }
        }
        
        // No candidate actions
        if candidate_actions.is_empty() {
            return Ok(());
        }
        
        // Filter actions by minimum load reduction and maximum disruption
        let filtered_actions: Vec<AdaptationAction> = candidate_actions.into_iter()
            .filter(|action| {
                action.estimated_effects.load_reduction >= self.parameters.min_load_reduction &&
                action.estimated_effects.disruption <= self.parameters.max_disruption
            })
            .collect();
        
        // No actions pass filtering
        if filtered_actions.is_empty() {
            return Ok(());
        }
        
        // Select best actions based on overall score
        let best_action = filtered_actions.into_iter()
            .max_by(|a, b| {
                a.estimated_effects.overall_score()
                    .partial_cmp(&b.estimated_effects.overall_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        
        // Apply or schedule adaptation based on timing strategy
        match self.parameters.timing {
            AdaptationTiming::Immediate => {
                self.apply_adaptation(vec![best_action], load_estimate.overall_load)?;
            },
            AdaptationTiming::Breakpoint => {
                // Store for later application at breakpoint
                self.pending_adaptations.push(best_action);
            },
            AdaptationTiming::Delayed(duration) => {
                // Schedule for later application
                self.pending_adaptations.push(best_action);
                
                // In a real implementation, this would start a timer
                // For simplicity, we're just storing it
            },
            AdaptationTiming::OnRequest => {
                // Store for application when requested
                self.pending_adaptations.push(best_action);
            },
        }
        
        Ok(())
    }
    
    /// Apply scheduled adaptations at a breakpoint
    pub fn apply_at_breakpoint(&mut self, load: f64) -> AdaptationResult<()> {
        if self.pending_adaptations.is_empty() {
            return Ok(());
        }
        
        let actions = std::mem::take(&mut self.pending_adaptations);
        self.apply_adaptation(actions, load)
    }
    
    /// Apply adaptations on user request
    pub fn apply_on_request(&mut self, load: f64) -> AdaptationResult<()> {
        if self.pending_adaptations.is_empty() {
            return Ok(());
        }
        
        let actions = std::mem::take(&mut self.pending_adaptations);
        self.apply_adaptation(actions, load)
    }
    
    /// Apply adaptations immediately
    fn apply_adaptation(&mut self, actions: Vec<AdaptationAction>, pre_load: f64) -> AdaptationResult<()> {
        // Get current user model and layout
        let user_model = self.user_model.read().unwrap().clone();
        let current_layout = self.current_layout.read().unwrap().clone();
        
        // Group actions by strategy
        let mut action_by_strategy: HashMap<&str, Vec<AdaptationAction>> = HashMap::new();
        
        for action in &actions {
            let strategy_name = match action.adaptation_type {
                AdaptationType::Visibility => "Visibility Adaptation",
                AdaptationType::Style => "Style Adaptation",
                AdaptationType::Modality => "Modality Adaptation",
                AdaptationType::Density => "Density Adaptation",
                AdaptationType::Ordering => "Ordering Adaptation",
                AdaptationType::Interaction => "Interaction Adaptation",
            };
            
            action_by_strategy.entry(strategy_name)
                .or_insert_with(Vec::new)
                .push(action.clone());
        }
        
        // Apply actions using appropriate strategies
        let mut adapted_layout = current_layout.clone();
        
        for (strategy_name, strategy_actions) in action_by_strategy {
            let strategy = self.strategies.iter()
                .find(|s| s.name() == strategy_name);
            
            if let Some(strategy) = strategy {
                if let Ok(new_layout) = strategy.apply_actions(&adapted_layout, &strategy_actions) {
                    adapted_layout = new_layout;
                }
            }
        }
        
        // Update current layout
        *self.current_layout.write().unwrap() = adapted_layout;
        
        // Record adaptations in history
        let timestamp = std::time::SystemTime::now();
        
        for action in actions {
            let entry = AdaptationHistoryEntry {
                timestamp,
                action,
                success: true,
                user_response: None,
                pre_load: Some(pre_load),
                post_load: None, // Will be updated later
            };
            
            self.history.push(entry);
        }
        
        self.last_adaptation = Instant::now();
        
        Ok(())
    }
    
    /// Provide feedback on adaptation
    pub fn provide_feedback(&mut self, action_id: usize, response: f64, post_load: f64) -> AdaptationResult<()> {
        if action_id >= self.history.len() {
            return Err(AdaptationError::InvalidParameters(
                format!("Invalid action ID: {}", action_id)));
        }
        
        let entry = &mut self.history[action_id];
        entry.user_response = Some(response);
        entry.post_load = Some(post_load);
        
        // Update user model based on feedback
        let mut user_model = self.user_model.write().unwrap();
        
        // If user liked adaptation (positive response)
        if response > 0.0 {
            // Increase threshold for triggering adaptations
            self.parameters.adaptation_threshold += 0.02;
            self.parameters.adaptation_threshold = self.parameters.adaptation_threshold.min(0.9);
            
            // Record preference for this adaptation type
            let adaptation_type = format!("{:?}", entry.action.adaptation_type).to_lowercase();
            user_model.set_preference(&format!("prefers_{}", adaptation_type), "true");
            
            // If modality adaptation, update preferred modality
            if entry.action.adaptation_type == AdaptationType::Modality {
                if let Some(modality) = entry.action.parameters.get("modality") {
                    user_model.cognitive_profile.preferred_modality = modality.clone();
                }
            }
        } else if response < 0.0 {
            // Decrease threshold for triggering adaptations
            self.parameters.adaptation_threshold -= 0.02;
            self.parameters.adaptation_threshold = self.parameters.adaptation_threshold.max(0.5);
            
            // Record dislike for this adaptation type
            let adaptation_type = format!("{:?}", entry.action.adaptation_type).to_lowercase();
            user_model.set_preference(&format!("prefers_{}", adaptation_type), "false");
        }
        
        // Update min_load_reduction based on actual load reduction
        if let (Some(pre), Some(post)) = (entry.pre_load, entry.post_load) {
            let actual_reduction = pre - post;
            
            // If actual reduction was less than expected
            if actual_reduction < entry.action.estimated_effects.load_reduction * 0.5 {
                // Increase minimum required reduction
                self.parameters.min_load_reduction += 0.02;
                self.parameters.min_load_reduction = self.parameters.min_load_reduction.min(0.3);
            } else {
                // Actual reduction was good, can be more aggressive
                self.parameters.min_load_reduction -= 0.01;
                self.parameters.min_load_reduction = self.parameters.min_load_reduction.max(0.05);
            }
        }
        
        Ok(())
    }
    
    /// Revert the most recent adaptation
    pub fn revert_last_adaptation(&mut self) -> AdaptationResult<()> {
        if self.history.is_empty() {
            return Err(AdaptationError::InvalidParameters(
                "No adaptations to revert".to_string()));
        }
        
        let last_entry = self.history.pop().unwrap();
        
        if let Some(revert_action) = last_entry.action.revert_action {
            // Apply revert action
            let current_layout = self.current_layout.read().unwrap().clone();
            
            // Find the appropriate strategy
            let strategy_name = match revert_action.adaptation_type {
                AdaptationType::Visibility => "Visibility Adaptation",
                AdaptationType::Style => "Style Adaptation",
                AdaptationType::Modality => "Modality Adaptation",
                AdaptationType::Density => "Density Adaptation",
                AdaptationType::Ordering => "Ordering Adaptation",
                AdaptationType::Interaction => "Interaction Adaptation",
            };
            
            let strategy = self.strategies.iter()
                .find(|s| s.name() == strategy_name);
            
            if let Some(strategy) = strategy {
                if let Ok(reverted_layout) = strategy.apply_actions(&current_layout, &[*revert_action]) {
                    *self.current_layout.write().unwrap() = reverted_layout;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get adaptation history
    pub fn get_history(&self) -> &[AdaptationHistoryEntry] {
        &self.history
    }
    
    /// Clear adaptation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    
    /// Get pending adaptations
    pub fn get_pending_adaptations(&self) -> &[AdaptationAction] {
        &self.pending_adaptations
    }
    
    /// Clear pending adaptations
    pub fn clear_pending_adaptations(&mut self) {
        self.pending_adaptations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_element_importance() {
        // Test conversion to/from numeric values
        assert_eq!(ElementImportance::from_value(ElementImportance::Critical.to_value()), 
                 ElementImportance::Critical);
        assert_eq!(ElementImportance::from_value(ElementImportance::High.to_value()), 
                 ElementImportance::High);
        assert_eq!(ElementImportance::from_value(ElementImportance::Medium.to_value()), 
                 ElementImportance::Medium);
        assert_eq!(ElementImportance::from_value(ElementImportance::Low.to_value()), 
                 ElementImportance::Low);
        assert_eq!(ElementImportance::from_value(ElementImportance::Optional.to_value()), 
                 ElementImportance::Optional);
        
        // Test ordering
        assert!(ElementImportance::Critical > ElementImportance::High);
        assert!(ElementImportance::High > ElementImportance::Medium);
        assert!(ElementImportance::Medium > ElementImportance::Low);
        assert!(ElementImportance::Low > ElementImportance::Optional);
    }
    
    #[test]
    fn test_interface_element() {
        let element = InterfaceElement::new("button1", InterfaceElementType::Visual, ElementImportance::High);
        
        // Test expertise visibility
        assert!(element.is_visible_for_expertise(&ExpertiseLevel::Beginner));
        assert!(element.is_visible_for_expertise(&ExpertiseLevel::Intermediate));
        assert!(element.is_visible_for_expertise(&ExpertiseLevel::Advanced));
        assert!(element.is_visible_for_expertise(&ExpertiseLevel::Expert));
        
        // Test with expertise requirement
        let mut element_with_req = element.clone();
        element_with_req.required_expertise = Some(ExpertiseLevel::Advanced);
        
        assert!(!element_with_req.is_visible_for_expertise(&ExpertiseLevel::Beginner));
        assert!(!element_with_req.is_visible_for_expertise(&ExpertiseLevel::Intermediate));
        assert!(element_with_req.is_visible_for_expertise(&ExpertiseLevel::Advanced));
        assert!(element_with_req.is_visible_for_expertise(&ExpertiseLevel::Expert));
    }
    
    #[test]
    fn test_interface_layout() {
        let mut layout = InterfaceLayout::new("main_layout");
        
        // Add elements
        let element1 = InterfaceElement::new("button1", InterfaceElementType::Visual, ElementImportance::High);
        let element2 = InterfaceElement::new("label1", InterfaceElementType::Visual, ElementImportance::Medium);
        
        layout.add_element(element1);
        layout.add_element(element2);
        
        // Test complexity calculation
        assert!(layout.complexity > 0.0);
        
        // Test element removal
        let removed = layout.remove_element("button1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "button1");
        assert_eq!(layout.elements.len(), 1);
        
        // Test filtering
        let mut layout2 = InterfaceLayout::new("test_layout");
        let element3 = InterfaceElement::new("button2", InterfaceElementType::Visual, ElementImportance::High);
        let mut element4 = InterfaceElement::new("label2", InterfaceElementType::Visual, ElementImportance::Low);
        element4.required_expertise = Some(ExpertiseLevel::Advanced);
        
        layout2.add_element(element3);
        layout2.add_element(element4);
        
        // Filter by importance
        let filtered = layout2.filter_elements(|e| e.importance == ElementImportance::High);
        assert_eq!(filtered.elements.len(), 1);
        assert!(filtered.elements.contains_key("button2"));
        
        // Get visible elements for expertise
        let visible = layout2.get_visible_elements(&ExpertiseLevel::Beginner);
        assert_eq!(visible.elements.len(), 1);
        assert!(visible.elements.contains_key("button2"));
    }
    
    #[test]
    fn test_user_model() {
        let mut model = UserModel::new("user1");
        
        // Test trait access
        assert_eq!(model.get_trait_value(CognitiveTrait::WorkingMemory), 0.5); // Default Average
        
        // Test preference access
        assert_eq!(model.get_preference("non_existent"), None);
        
        model.set_preference("test_pref", "value");
        assert_eq!(model.get_preference("test_pref"), Some("value"));
        
        // Test interaction history
        let event = InteractionEvent {
            timestamp: std::time::SystemTime::now(),
            event_type: "click".to_string(),
            target_element: Some("button1".to_string()),
            duration: Some(Duration::from_millis(500)),
            success: Some(true),
            data: HashMap::new(),
        };
        
        model.add_interaction(event);
        assert_eq!(model.interaction_history.len(), 1);
        
        // Test model update
        model.update_from_interactions();
        assert!(model.confidence > 0.1); // Should have some confidence now
    }
    
    #[test]
    fn test_adaptation_effect() {
        let effect = AdaptationEffect {
            load_reduction: 0.2,
            usability_impact: 0.1,
            learning_impact: -0.05,
            satisfaction_impact: 0.15,
            disruption: 0.3,
        };
        
        let score = effect.overall_score();
        assert!(score > 0.0); // Overall positive effect
        
        // Test with more disruption
        let effect2 = AdaptationEffect {
            load_reduction: 0.2,
            usability_impact: 0.1,
            learning_impact: -0.05,
            satisfaction_impact: 0.15,
            disruption: 0.8, // High disruption
        };
        
        let score2 = effect2.overall_score();
        assert!(score > score2); // First effect should be better
    }
}
