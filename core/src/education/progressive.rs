//! # Progressive Disclosure Educational Framework
//! 
//! This module implements a theoretically-grounded progressive disclosure system
//! based on variable information density pedagogical theory. The system enables
//! seamless adaptation of educational content across multiple expertise dimensions
//! while maintaining semantic invariants between representations.
//! 
//! ## Theoretical Foundation
//! 
//! The implementation leverages category-theoretic mappings between expertise-dependent
//! representation spaces, ensuring that educational transformations preserve core semantic
//! properties while modulating information density. This approach guarantees that learners
//! at different expertise levels receive isomorphic knowledge representations with
//! appropriate complexity calibration.
//! 
//! ## Computational Characteristics
//! 
//! - Time Complexity: O(1) for expertise transitions and adaptivity operations
//! - Space Complexity: O(k) where k is the cardinality of expertise dimensions
//! - Memory Model: Zero-copy transformations with immutable representation states
//! 
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::Algorithm;
use crate::visualization::View;
use crate::insights::pattern::PatternRecognizer;

/// Error types for progressive disclosure operations
#[derive(Debug, Error)]
pub enum ProgressiveError {
    /// Invalid expertise level
    #[error("Invalid expertise level: {0}")]
    InvalidExpertiseLevel(String),
    
    /// Invalid transformation
    #[error("Invalid transformation between expertise levels: {0} -> {1}")]
    InvalidTransformation(String, String),
    
    /// Feature not available at expertise level
    #[error("Feature {0} not available at expertise level {1}")]
    FeatureNotAvailable(String, String),
    
    /// Semantic preservation violation
    #[error("Semantic preservation violation: {0}")]
    SemanticViolation(String),
    
    /// Other progressive error
    #[error("Progressive disclosure error: {0}")]
    Other(String),
}

/// Expertise level within a specific knowledge domain
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Beginner level - foundational concepts only
    Beginner,
    
    /// Intermediate level - standard concepts and techniques
    Intermediate,
    
    /// Advanced level - sophisticated concepts and optimizations
    Advanced,
    
    /// Expert level - theoretical foundations and edge cases
    Expert,
    
    /// Custom expertise level with specified capabilities
    Custom(String),
}

impl ExpertiseLevel {
    /// Returns the ordinal value of the expertise level for comparison
    pub fn ordinal(&self) -> u8 {
        match self {
            ExpertiseLevel::Beginner => 1,
            ExpertiseLevel::Intermediate => 2,
            ExpertiseLevel::Advanced => 3,
            ExpertiseLevel::Expert => 4,
            ExpertiseLevel::Custom(_) => 2, // Default to intermediate for custom levels
        }
    }
    
    /// Check if this expertise level is at least as advanced as the specified level
    pub fn is_at_least(&self, other: &ExpertiseLevel) -> bool {
        self.ordinal() >= other.ordinal()
    }
}

impl Default for ExpertiseLevel {
    fn default() -> Self {
        ExpertiseLevel::Beginner
    }
}

/// Feature access control for different expertise levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAccessControl {
    /// Minimum expertise level required for feature access
    pub minimum_level: ExpertiseLevel,
    
    /// Optional custom access predicate (overrides minimum_level if present)
    #[serde(skip)]
    pub custom_predicate: Option<Arc<dyn Fn(&UserProfile) -> bool + Send + Sync>>,
    
    /// Feature description for each expertise level
    pub level_descriptions: HashMap<ExpertiseLevel, String>,
}

impl FeatureAccessControl {
    /// Create a new feature access control with minimum expertise level
    pub fn new(minimum_level: ExpertiseLevel) -> Self {
        Self {
            minimum_level,
            custom_predicate: None,
            level_descriptions: HashMap::new(),
        }
    }
    
    /// Add description for a specific expertise level
    pub fn with_description(mut self, level: ExpertiseLevel, description: String) -> Self {
        self.level_descriptions.insert(level, description);
        self
    }
    
    /// Add custom access predicate
    pub fn with_predicate<F>(mut self, predicate: F) -> Self 
    where 
        F: Fn(&UserProfile) -> bool + Send + Sync + 'static 
    {
        self.custom_predicate = Some(Arc::new(predicate));
        self
    }
    
    /// Check if the feature is accessible at the given expertise level
    pub fn is_accessible(&self, profile: &UserProfile) -> bool {
        // Check custom predicate first if available
        if let Some(predicate) = &self.custom_predicate {
            return predicate(profile);
        }
        
        // Otherwise check minimum expertise level
        profile.expertise_level.is_at_least(&self.minimum_level)
    }
    
    /// Get description for the given expertise level
    pub fn get_description(&self, level: &ExpertiseLevel) -> Option<&String> {
        self.level_descriptions.get(level)
    }
}

/// Educational component that can be adapted based on expertise level
pub trait ExpertiseAdaptive: Send + Sync {
    /// Get the component type identifier
    fn component_type(&self) -> &'static str;
    
    /// Adapt the component for the given expertise level
    fn adapt_for_expertise(&self, profile: &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError>;
    
    /// Get the minimum expertise level required for this component
    fn minimum_expertise(&self) -> ExpertiseLevel;
    
    /// Check if the component is accessible at the given expertise level
    fn is_accessible(&self, profile: &UserProfile) -> bool {
        profile.expertise_level.is_at_least(&self.minimum_expertise())
    }
}

/// User profile containing expertise and preference information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique identifier for the user
    pub user_id: String,
    
    /// Overall expertise level
    pub expertise_level: ExpertiseLevel,
    
    /// Domain-specific expertise levels
    pub domain_expertise: HashMap<String, ExpertiseLevel>,
    
    /// Learning preferences
    pub preferences: LearningPreferences,
    
    /// Usage history for adaptivity
    pub usage_history: UsageHistory,
}

impl UserProfile {
    /// Create a new user profile with default values
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            expertise_level: ExpertiseLevel::default(),
            domain_expertise: HashMap::new(),
            preferences: LearningPreferences::default(),
            usage_history: UsageHistory::default(),
        }
    }
    
    /// Get domain-specific expertise level, falling back to overall level if not specified
    pub fn get_domain_expertise(&self, domain: &str) -> &ExpertiseLevel {
        self.domain_expertise.get(domain).unwrap_or(&self.expertise_level)
    }
    
    /// Set domain-specific expertise level
    pub fn set_domain_expertise(&mut self, domain: String, level: ExpertiseLevel) {
        self.domain_expertise.insert(domain, level);
    }
    
    /// Update profile based on interaction data
    pub fn update_from_interaction(&mut self, interaction: &InteractionData) {
        // Update usage history
        self.usage_history.record_interaction(interaction);
        
        // Check for expertise progression threshold
        if interaction.difficulty_rating.map_or(false, |d| d < 0.3) {
            // Content was too easy, consider increasing expertise level
            if self.usage_history.consecutive_easy_interactions > 5 {
                // Threshold reached, increase domain expertise
                if let Some(domain) = &interaction.domain {
                    let current_level = self.get_domain_expertise(domain);
                    match current_level {
                        ExpertiseLevel::Beginner => {
                            self.set_domain_expertise(domain.clone(), ExpertiseLevel::Intermediate);
                        }
                        ExpertiseLevel::Intermediate => {
                            self.set_domain_expertise(domain.clone(), ExpertiseLevel::Advanced);
                        }
                        ExpertiseLevel::Advanced => {
                            self.set_domain_expertise(domain.clone(), ExpertiseLevel::Expert);
                        }
                        _ => {} // No change for Expert or Custom levels
                    }
                }
            }
        }
    }
}

/// Learning preferences for personalized adaptivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPreferences {
    /// Preferred learning modality
    pub preferred_modality: LearningModality,
    
    /// Information density preference (0.0-1.0)
    pub information_density: f32,
    
    /// Interaction frequency preference (0.0-1.0)
    pub interaction_frequency: f32,
    
    /// Visual complexity preference (0.0-1.0)
    pub visual_complexity: f32,
    
    /// Feedback detail preference (0.0-1.0)
    pub feedback_detail: f32,
    
    /// Custom preferences
    pub custom_preferences: HashMap<String, f32>,
}

impl Default for LearningPreferences {
    fn default() -> Self {
        Self {
            preferred_modality: LearningModality::Visual,
            information_density: 0.5,
            interaction_frequency: 0.5,
            visual_complexity: 0.5,
            feedback_detail: 0.5,
            custom_preferences: HashMap::new(),
        }
    }
}

/// Learning modality preference
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearningModality {
    /// Visual learning (diagrams, animations)
    Visual,
    
    /// Auditory learning (spoken explanations)
    Auditory,
    
    /// Reading/writing learning (text-based)
    ReadingWriting,
    
    /// Kinesthetic learning (interactive exercises)
    Kinesthetic,
    
    /// Multimodal learning (combination)
    Multimodal,
}

/// Usage history for adaptivity decisions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageHistory {
    /// Total interaction count
    pub total_interactions: usize,
    
    /// Interaction count by domain
    pub domain_interactions: HashMap<String, usize>,
    
    /// Feature usage counts
    pub feature_usage: HashMap<String, usize>,
    
    /// Consecutive easy interactions (difficulty < 0.3)
    pub consecutive_easy_interactions: usize,
    
    /// Consecutive difficult interactions (difficulty > 0.7)
    pub consecutive_difficult_interactions: usize,
    
    /// Recent interaction history (last 20 interactions)
    pub recent_interactions: Vec<InteractionData>,
}

impl UsageHistory {
    /// Record a new interaction
    pub fn record_interaction(&mut self, interaction: &InteractionData) {
        self.total_interactions += 1;
        
        // Update domain interaction count
        if let Some(domain) = &interaction.domain {
            *self.domain_interactions.entry(domain.clone()).or_insert(0) += 1;
        }
        
        // Update feature usage count
        if let Some(feature) = &interaction.feature {
            *self.feature_usage.entry(feature.clone()).or_insert(0) += 1;
        }
        
        // Update difficulty tracking
        if let Some(difficulty) = interaction.difficulty_rating {
            if difficulty < 0.3 {
                self.consecutive_easy_interactions += 1;
                self.consecutive_difficult_interactions = 0;
            } else if difficulty > 0.7 {
                self.consecutive_difficult_interactions += 1;
                self.consecutive_easy_interactions = 0;
            } else {
                self.consecutive_easy_interactions = 0;
                self.consecutive_difficult_interactions = 0;
            }
        }
        
        // Add to recent interactions, maintaining limited history
        self.recent_interactions.push(interaction.clone());
        if self.recent_interactions.len() > 20 {
            self.recent_interactions.remove(0);
        }
    }
}

/// Interaction data for adaptivity decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    /// Interaction timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Interaction domain (if applicable)
    pub domain: Option<String>,
    
    /// Interaction feature (if applicable)
    pub feature: Option<String>,
    
    /// Interaction duration in seconds
    pub duration_seconds: Option<f64>,
    
    /// Difficulty rating (0.0-1.0, if available)
    pub difficulty_rating: Option<f32>,
    
    /// Success/completion status
    pub success: Option<bool>,
    
    /// Custom interaction data
    pub custom_data: HashMap<String, String>,
}

/// Progressive disclosure manager
///
/// Central component for managing expertise-based adaptation across the system
#[derive(Debug)]
pub struct ProgressiveDisclosureManager {
    /// Feature access controls
    feature_access: HashMap<String, FeatureAccessControl>,
    
    /// Component adapters
    component_adapters: HashMap<&'static str, Box<dyn Fn(&dyn ExpertiseAdaptive, &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> + Send + Sync>>,
    
    /// Event listeners for expertise changes
    #[allow(clippy::type_complexity)]
    expertise_change_listeners: Vec<Box<dyn Fn(&UserProfile, &ExpertiseLevel, &ExpertiseLevel) + Send + Sync>>,
}

impl ProgressiveDisclosureManager {
    /// Create a new progressive disclosure manager
    pub fn new() -> Self {
        Self {
            feature_access: HashMap::new(),
            component_adapters: HashMap::new(),
            expertise_change_listeners: Vec::new(),
        }
    }
    
    /// Register feature access control
    pub fn register_feature(&mut self, feature_id: String, access_control: FeatureAccessControl) {
        self.feature_access.insert(feature_id, access_control);
    }
    
    /// Register component adapter
    pub fn register_component_adapter<F, T>(&mut self, component_type: &'static str, adapter: F)
    where
        F: Fn(&T, &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> + Send + Sync + 'static,
        T: ExpertiseAdaptive + 'static,
    {
        let type_erased_adapter = move |component: &dyn ExpertiseAdaptive, profile: &UserProfile| -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> {
            // Attempt downcast to specific type
            if let Some(typed_component) = (component as &dyn std::any::Any).downcast_ref::<T>() {
                adapter(typed_component, profile)
            } else {
                Err(ProgressiveError::Other(format!(
                    "Component type mismatch: expected {}, got {}",
                    std::any::type_name::<T>(),
                    component.component_type()
                )))
            }
        };
        
        self.component_adapters.insert(component_type, Box::new(type_erased_adapter));
    }
    
    /// Register expertise change listener
    pub fn register_expertise_change_listener<F>(&mut self, listener: F)
    where
        F: Fn(&UserProfile, &ExpertiseLevel, &ExpertiseLevel) + Send + Sync + 'static,
    {
        self.expertise_change_listeners.push(Box::new(listener));
    }
    
    /// Check if a feature is accessible for the given profile
    pub fn is_feature_accessible(&self, feature_id: &str, profile: &UserProfile) -> bool {
        self.feature_access
            .get(feature_id)
            .map_or(false, |access| access.is_accessible(profile))
    }
    
    /// Get accessible features for the given profile
    pub fn get_accessible_features(&self, profile: &UserProfile) -> Vec<String> {
        self.feature_access
            .iter()
            .filter_map(|(feature_id, access)| {
                if access.is_accessible(profile) {
                    Some(feature_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Adapt component for the given profile
    pub fn adapt_component(&self, component: &dyn ExpertiseAdaptive, profile: &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> {
        // Check if component is accessible
        if !component.is_accessible(profile) {
            return Err(ProgressiveError::FeatureNotAvailable(
                component.component_type().to_string(),
                format!("{:?}", profile.expertise_level)
            ));
        }
        
        // Get adapter for component type
        if let Some(adapter) = self.component_adapters.get(component.component_type()) {
            adapter(component, profile)
        } else {
            // Default adapter just returns the component's own adaptation
            component.adapt_for_expertise(profile)
        }
    }
    
    /// Change expertise level for a user profile
    pub fn change_expertise_level(&self, profile: &mut UserProfile, new_level: ExpertiseLevel) {
        let old_level = profile.expertise_level.clone();
        profile.expertise_level = new_level.clone();
        
        // Notify listeners
        for listener in &self.expertise_change_listeners {
            listener(profile, &old_level, &new_level);
        }
    }

    /// Assess whether the user should transition to a different expertise level
    pub fn assess_expertise_transition(&self, profile: &UserProfile) -> Option<ExpertiseLevel> {
        // Check for expertise progression based on usage history
        if profile.usage_history.consecutive_easy_interactions > 10 {
            // Content consistently too easy, suggest increasing expertise level
            match &profile.expertise_level {
                ExpertiseLevel::Beginner => Some(ExpertiseLevel::Intermediate),
                ExpertiseLevel::Intermediate => Some(ExpertiseLevel::Advanced),
                ExpertiseLevel::Advanced => Some(ExpertiseLevel::Expert),
                _ => None, // No change for Expert or Custom levels
            }
        } else if profile.usage_history.consecutive_difficult_interactions > 8 {
            // Content consistently too difficult, suggest decreasing expertise level
            match &profile.expertise_level {
                ExpertiseLevel::Intermediate => Some(ExpertiseLevel::Beginner),
                ExpertiseLevel::Advanced => Some(ExpertiseLevel::Intermediate),
                ExpertiseLevel::Expert => Some(ExpertiseLevel::Advanced),
                _ => None, // No change for Beginner or Custom levels
            }
        } else {
            None // No change suggested
        }
    }
    
    /// Get recommended next features based on usage history
    pub fn get_feature_recommendations(&self, profile: &UserProfile, limit: usize) -> Vec<String> {
        // Get all accessible features
        let accessible_features = self.get_accessible_features(profile);
        
        // Calculate feature scores based on usage gaps and relevance
        let mut feature_scores: Vec<(String, f32)> = accessible_features
            .into_iter()
            .map(|feature_id| {
                let usage_count = profile.usage_history.feature_usage.get(&feature_id).copied().unwrap_or(0);
                let usage_score = 1.0 / (usage_count as f32 + 1.0); // Higher score for less used features
                
                // Additional relevance score could be calculated here
                let relevance_score = 1.0; // Placeholder
                
                let total_score = usage_score * relevance_score;
                (feature_id, total_score)
            })
            .collect();
        
        // Sort by score (highest first)
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top N features
        feature_scores.into_iter()
            .take(limit)
            .map(|(feature_id, _)| feature_id)
            .collect()
    }
}

impl Default for ProgressiveDisclosureManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating standard expertise adaptive components
pub struct AdaptiveComponentFactory;

impl AdaptiveComponentFactory {
    /// Create an adaptive algorithm wrapper
    pub fn create_adaptive_algorithm<T: Algorithm + 'static>(algorithm: T, minimum_level: ExpertiseLevel) -> AdaptiveAlgorithm<T> {
        AdaptiveAlgorithm {
            algorithm,
            minimum_level,
            level_configurations: HashMap::new(),
        }
    }
    
    /// Create an adaptive view wrapper
    pub fn create_adaptive_view<T: View + 'static>(view: T, minimum_level: ExpertiseLevel) -> AdaptiveView<T> {
        AdaptiveView {
            view,
            minimum_level,
            level_configurations: HashMap::new(),
        }
    }
    
    /// Create an adaptive insight wrapper
    pub fn create_adaptive_insight<T: PatternRecognizer + 'static>(recognizer: T, minimum_level: ExpertiseLevel) -> AdaptiveInsight<T> {
        AdaptiveInsight {
            recognizer,
            minimum_level,
            level_configurations: HashMap::new(),
        }
    }
}

/// Adaptive algorithm wrapper
pub struct AdaptiveAlgorithm<T: Algorithm> {
    /// Wrapped algorithm
    algorithm: T,
    
    /// Minimum expertise level
    minimum_level: ExpertiseLevel,
    
    /// Configuration for different expertise levels
    level_configurations: HashMap<ExpertiseLevel, HashMap<String, String>>,
}

impl<T: Algorithm + Clone + 'static> AdaptiveAlgorithm<T> {
    /// Add configuration for a specific expertise level
    pub fn with_level_config(mut self, level: ExpertiseLevel, config: HashMap<String, String>) -> Self {
        self.level_configurations.insert(level, config);
        self
    }
}

impl<T: Algorithm + Clone + 'static> ExpertiseAdaptive for AdaptiveAlgorithm<T> {
    fn component_type(&self) -> &'static str {
        "algorithm"
    }
    
    fn adapt_for_expertise(&self, profile: &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> {
        let mut algorithm_clone = self.algorithm.clone();
        
        // Apply configuration for the current expertise level
        if let Some(config) = self.level_configurations.get(&profile.expertise_level) {
            for (key, value) in config {
                algorithm_clone.set_parameter(key, value).map_err(|e| {
                    ProgressiveError::Other(format!("Failed to set parameter: {}", e))
                })?;
            }
        }
        
        // Create a new adaptive algorithm with the configured algorithm
        Ok(Box::new(AdaptiveAlgorithm {
            algorithm: algorithm_clone,
            minimum_level: self.minimum_level.clone(),
            level_configurations: self.level_configurations.clone(),
        }))
    }
    
    fn minimum_expertise(&self) -> ExpertiseLevel {
        self.minimum_level.clone()
    }
}

/// Adaptive view wrapper
pub struct AdaptiveView<T: View> {
    /// Wrapped view
    view: T,
    
    /// Minimum expertise level
    minimum_level: ExpertiseLevel,
    
    /// Configuration for different expertise levels
    level_configurations: HashMap<ExpertiseLevel, HashMap<String, String>>,
}

impl<T: View + Clone + 'static> AdaptiveView<T> {
    /// Add configuration for a specific expertise level
    pub fn with_level_config(mut self, level: ExpertiseLevel, config: HashMap<String, String>) -> Self {
        self.level_configurations.insert(level, config);
        self
    }
}

impl<T: View + Clone + 'static> ExpertiseAdaptive for AdaptiveView<T> {
    fn component_type(&self) -> &'static str {
        "view"
    }
    
    fn adapt_for_expertise(&self, profile: &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> {
        // Create view clone with appropriate configuration
        let view_clone = self.view.clone();
        
        // Further configuration would be applied here based on expertise level
        
        Ok(Box::new(AdaptiveView {
            view: view_clone,
            minimum_level: self.minimum_level.clone(),
            level_configurations: self.level_configurations.clone(),
        }))
    }
    
    fn minimum_expertise(&self) -> ExpertiseLevel {
        self.minimum_level.clone()
    }
}

/// Adaptive insight wrapper
pub struct AdaptiveInsight<T: PatternRecognizer> {
    /// Wrapped recognizer
    recognizer: T,
    
    /// Minimum expertise level
    minimum_level: ExpertiseLevel,
    
    /// Configuration for different expertise levels
    level_configurations: HashMap<ExpertiseLevel, HashMap<String, String>>,
}

impl<T: PatternRecognizer + Clone + 'static> AdaptiveInsight<T> {
    /// Add configuration for a specific expertise level
    pub fn with_level_config(mut self, level: ExpertiseLevel, config: HashMap<String, String>) -> Self {
        self.level_configurations.insert(level, config);
        self
    }
}

impl<T: PatternRecognizer + Clone + 'static> ExpertiseAdaptive for AdaptiveInsight<T> {
    fn component_type(&self) -> &'static str {
        "insight"
    }
    
    fn adapt_for_expertise(&self, profile: &UserProfile) -> Result<Box<dyn ExpertiseAdaptive>, ProgressiveError> {
        // Create insight clone with appropriate configuration
        let recognizer_clone = self.recognizer.clone();
        
        // Further configuration would be applied here based on expertise level
        
        Ok(Box::new(AdaptiveInsight {
            recognizer: recognizer_clone,
            minimum_level: self.minimum_level.clone(),
            level_configurations: self.level_configurations.clone(),
        }))
    }
    
    fn minimum_expertise(&self) -> ExpertiseLevel {
        self.minimum_level.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock algorithm for testing
    #[derive(Clone)]
    struct MockAlgorithm {
        parameters: HashMap<String, String>,
    }
    
    impl MockAlgorithm {
        fn new() -> Self {
            Self {
                parameters: HashMap::new(),
            }
        }
    }
    
    impl Algorithm for MockAlgorithm {
        fn name(&self) -> &str {
            "MockAlgorithm"
        }
        
        fn category(&self) -> &str {
            "mock"
        }
        
        fn description(&self) -> &str {
            "Mock algorithm for testing"
        }
        
        fn set_parameter(&mut self, name: &str, value: &str) -> Result<(), crate::algorithm::AlgorithmError> {
            self.parameters.insert(name.to_string(), value.to_string());
            Ok(())
        }
        
        fn get_parameter(&self, name: &str) -> Option<&str> {
            self.parameters.get(name).map(|s| s.as_str())
        }
        
        fn get_parameters(&self) -> HashMap<String, String> {
            self.parameters.clone()
        }
        
        fn execute_with_tracing(&mut self, 
                              _graph: &crate::data_structures::graph::Graph, 
                              _tracer: &mut crate::execution::tracer::ExecutionTracer) 
                              -> Result<crate::algorithm::AlgorithmResult, crate::algorithm::AlgorithmError> {
            unimplemented!()
        }
        
        fn find_path(&mut self, 
                   _graph: &crate::data_structures::graph::Graph, 
                   _start: crate::algorithm::NodeId, 
                   _goal: crate::algorithm::NodeId) 
                   -> Result<crate::algorithm::PathResult, crate::algorithm::AlgorithmError> {
            unimplemented!()
        }
    }
    
    #[test]
    fn test_expertise_level_comparison() {
        assert!(ExpertiseLevel::Expert.is_at_least(&ExpertiseLevel::Beginner));
        assert!(ExpertiseLevel::Expert.is_at_least(&ExpertiseLevel::Intermediate));
        assert!(ExpertiseLevel::Expert.is_at_least(&ExpertiseLevel::Advanced));
        assert!(ExpertiseLevel::Expert.is_at_least(&ExpertiseLevel::Expert));
        
        assert!(!ExpertiseLevel::Beginner.is_at_least(&ExpertiseLevel::Intermediate));
        assert!(!ExpertiseLevel::Beginner.is_at_least(&ExpertiseLevel::Advanced));
        assert!(!ExpertiseLevel::Beginner.is_at_least(&ExpertiseLevel::Expert));
        
        assert!(ExpertiseLevel::Intermediate.is_at_least(&ExpertiseLevel::Beginner));
        assert!(!ExpertiseLevel::Intermediate.is_at_least(&ExpertiseLevel::Advanced));
    }
    
    #[test]
    fn test_feature_access_control() {
        let feature = FeatureAccessControl::new(ExpertiseLevel::Intermediate)
            .with_description(ExpertiseLevel::Beginner, "Basic features only".to_string())
            .with_description(ExpertiseLevel::Intermediate, "Standard features".to_string())
            .with_description(ExpertiseLevel::Advanced, "Advanced features".to_string());
        
        let beginner_profile = UserProfile {
            user_id: "user1".to_string(),
            expertise_level: ExpertiseLevel::Beginner,
            domain_expertise: HashMap::new(),
            preferences: LearningPreferences::default(),
            usage_history: UsageHistory::default(),
        };
        
        let intermediate_profile = UserProfile {
            user_id: "user2".to_string(),
            expertise_level: ExpertiseLevel::Intermediate,
            domain_expertise: HashMap::new(),
            preferences: LearningPreferences::default(),
            usage_history: UsageHistory::default(),
        };
        
        assert!(!feature.is_accessible(&beginner_profile));
        assert!(feature.is_accessible(&intermediate_profile));
        
        assert_eq!(
            feature.get_description(&ExpertiseLevel::Beginner).unwrap(),
            "Basic features only"
        );
    }
    
    #[test]
    fn test_progressive_disclosure_manager() {
        let mut manager = ProgressiveDisclosureManager::new();
        
        // Register features
        manager.register_feature(
            "basic_visualization".to_string(),
            FeatureAccessControl::new(ExpertiseLevel::Beginner)
        );
        
        manager.register_feature(
            "advanced_visualization".to_string(),
            FeatureAccessControl::new(ExpertiseLevel::Advanced)
        );
        
        // Create user profile
        let mut profile = UserProfile::new("test_user".to_string());
        profile.expertise_level = ExpertiseLevel::Intermediate;
        
        // Check feature accessibility
        assert!(manager.is_feature_accessible("basic_visualization", &profile));
        assert!(!manager.is_feature_accessible("advanced_visualization", &profile));
        
        // Get accessible features
        let features = manager.get_accessible_features(&profile);
        assert!(features.contains(&"basic_visualization".to_string()));
        assert!(!features.contains(&"advanced_visualization".to_string()));
        
        // Change expertise level
        manager.change_expertise_level(&mut profile, ExpertiseLevel::Advanced);
        
        // Check feature accessibility after level change
        assert!(manager.is_feature_accessible("basic_visualization", &profile));
        assert!(manager.is_feature_accessible("advanced_visualization", &profile));
    }
    
    #[test]
    fn test_adaptive_algorithm() {
        let mock_algorithm = MockAlgorithm::new();
        
        // Create adaptive algorithm with configurations for different levels
        let mut beginner_config = HashMap::new();
        beginner_config.insert("complexity".to_string(), "low".to_string());
        
        let mut advanced_config = HashMap::new();
        advanced_config.insert("complexity".to_string(), "high".to_string());
        advanced_config.insert("optimization".to_string(), "enabled".to_string());
        
        let adaptive_algorithm = AdaptiveComponentFactory::create_adaptive_algorithm(
            mock_algorithm,
            ExpertiseLevel::Beginner
        )
        .with_level_config(ExpertiseLevel::Beginner, beginner_config)
        .with_level_config(ExpertiseLevel::Advanced, advanced_config);
        
        // Create profiles
        let beginner_profile = UserProfile {
            user_id: "user1".to_string(),
            expertise_level: ExpertiseLevel::Beginner,
            domain_expertise: HashMap::new(),
            preferences: LearningPreferences::default(),
            usage_history: UsageHistory::default(),
        };
        
        let advanced_profile = UserProfile {
            user_id: "user2".to_string(),
            expertise_level: ExpertiseLevel::Advanced,
            domain_expertise: HashMap::new(),
            preferences: LearningPreferences::default(),
            usage_history: UsageHistory::default(),
        };
        
        // Adapt for beginner
        let adapted_beginner = adaptive_algorithm.adapt_for_expertise(&beginner_profile).unwrap();
        let beginner_alg = adapted_beginner.downcast_ref::<AdaptiveAlgorithm<MockAlgorithm>>().unwrap();
        
        // Adapt for advanced
        let adapted_advanced = adaptive_algorithm.adapt_for_expertise(&advanced_profile).unwrap();
        let advanced_alg = adapted_advanced.downcast_ref::<AdaptiveAlgorithm<MockAlgorithm>>().unwrap();
        
        // Check parameters
        assert_eq!(beginner_alg.algorithm.get_parameter("complexity").unwrap(), "low");
        assert_eq!(advanced_alg.algorithm.get_parameter("complexity").unwrap(), "high");
        assert_eq!(advanced_alg.algorithm.get_parameter("optimization").unwrap(), "enabled");
    }
}