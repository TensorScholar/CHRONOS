//! # Cognitive Load Estimation and Management System
//!
//! This module implements an information-theoretic approach to cognitive load
//! estimation, quantification, and management for algorithm visualizations.
//! The system employs multi-factorial cognitive models based on established
//! research in cognitive psychology and human-computer interaction to provide
//! adaptive visualization transformations that optimize information transfer
//! while minimizing cognitive burden.
//!
//! ## Theoretical Foundation
//!
//! The implementation is based on three complementary theoretical frameworks:
//!
//! 1. **Cognitive Load Theory** (Sweller, 1988; 2011): Distinguishes between
//!    intrinsic, extraneous, and germane cognitive load, with particular focus
//!    on reducing extraneous load while preserving germane load.
//!
//! 2. **Information Theory** (Shannon, 1948; MacKay, 2003): Quantifies information
//!    density and entropy in visualizations as objective measures of complexity.
//!
//! 3. **Perceptual Processing Models** (Ware, 2012; Healey & Enns, 2012): Accounts
//!    for the cognitive effort required to process different visual channels and
//!    the limitations of human visual processing.
//!
//! ## Key Components
//!
//! * `CognitiveLoadEstimator`: Multi-factor model for estimating cognitive load
//! * `ComplexityMetrics`: Information-theoretic quantification of visualization complexity
//! * `AdaptationThreshold`: Dynamic threshold management for intervention decisions
//! * `ComplexityTransformer`: Transformation framework for complexity reduction
//!
//! ## Copyright
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algorithm::state::AlgorithmState;
use crate::education::progressive::ExpertiseLevel;
use crate::modality::representation::{ModalityRepresentation, RepresentationError};

/// Cognitive load estimation error types
#[derive(Error, Debug)]
pub enum CognitiveLoadError {
    /// Error when insufficient data is available for estimation
    #[error("Insufficient data for cognitive load estimation: {0}")]
    InsufficientData(String),

    /// Error when model parameters are invalid
    #[error("Invalid model parameters: {0}")]
    InvalidParameters(String),

    /// Error when transformation fails
    #[error("Transformation error: {0}")]
    TransformationError(String),

    /// Error in the underlying representation
    #[error("Representation error: {0}")]
    RepresentationError(#[from] RepresentationError),
}

/// Result type for cognitive load operations
pub type CognitiveLoadResult<T> = Result<T, CognitiveLoadError>;

/// Cognitive load component types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveLoadComponent {
    /// Intrinsic load from the inherent complexity of the material
    Intrinsic,
    /// Extraneous load from presentation format and interaction design
    Extraneous,
    /// Germane load from learning processes and schema construction
    Germane,
}

/// Information channel in visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InformationChannel {
    /// Position in 2D or 3D space
    Position,
    /// Color hue
    Color,
    /// Shape of visual elements
    Shape,
    /// Size of visual elements
    Size,
    /// Orientation of visual elements
    Orientation,
    /// Texture pattern
    Texture,
    /// Motion and animation
    Motion,
    /// Stereoscopic depth (if applicable)
    Depth,
    /// Connection between elements (edge visibility)
    Connection,
    /// Text and labels
    Text,
}

/// User interaction pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionPattern {
    /// Rapid navigation (suggests searching)
    RapidNavigation,
    /// Repeated viewing of the same element (suggests confusion)
    RepeatedViewing,
    /// Long pauses (suggests processing)
    LongPauses,
    /// Erratic movements (suggests uncertainty)
    ErraticMovement,
    /// Methodical exploration (suggests structured learning)
    MethodicalExploration,
    /// Abandoned interaction (suggests frustration)
    AbandonedInteraction,
}

/// Cognitive profile of a user
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CognitiveProfile {
    /// User expertise level
    pub expertise_level: ExpertiseLevel,
    /// Known cognitive characteristics (e.g., "high_spatial_ability")
    pub characteristics: HashSet<String>,
    /// Preferred learning modality (visual, auditory, kinesthetic)
    pub preferred_modality: String,
    /// Historical cognitive load tolerance
    pub load_tolerance: f64,
    /// Adaptation preferences
    pub adaptation_preferences: HashMap<String, String>,
}

impl Default for CognitiveProfile {
    fn default() -> Self {
        Self {
            expertise_level: ExpertiseLevel::Beginner,
            characteristics: HashSet::new(),
            preferred_modality: "visual".to_string(),
            load_tolerance: 0.65, // Default tolerance threshold (0.0-1.0)
            adaptation_preferences: HashMap::new(),
        }
    }
}

/// Visualization complexity metrics using information theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Information entropy (bits) - overall uncertainty in the visualization
    pub entropy: f64,
    /// Information density (bits per unit area)
    pub density: f64,
    /// Channel utilization (bits per channel)
    pub channel_utilization: HashMap<InformationChannel, f64>,
    /// Redundancy measure (0.0-1.0) - higher means more redundant encoding
    pub redundancy: f64,
    /// Visual clutter score (0.0-1.0)
    pub clutter: f64,
    /// Semantic complexity (number of distinct concepts)
    pub semantic_complexity: usize,
}

impl ComplexityMetrics {
    /// Create a new instance with default values
    pub fn new() -> Self {
        Self {
            entropy: 0.0,
            density: 0.0,
            channel_utilization: HashMap::new(),
            redundancy: 0.0,
            clutter: 0.0,
            semantic_complexity: 0,
        }
    }

    /// Calculate overall complexity score (0.0-1.0)
    pub fn overall_complexity(&self) -> f64 {
        // Weighted combination of different metrics
        let channel_avg = if self.channel_utilization.is_empty() {
            0.0
        } else {
            self.channel_utilization.values().sum::<f64>() / self.channel_utilization.len() as f64
        };

        // Entropy normalized by logarithm of semantic complexity (avoiding division by zero)
        let normalized_entropy = if self.semantic_complexity > 1 {
            self.entropy / (self.semantic_complexity as f64).log2()
        } else {
            self.entropy
        };

        // Weighted combination
        let weighted_score = 0.3 * normalized_entropy + 
                             0.2 * self.density + 
                             0.2 * channel_avg + 
                             0.1 * self.redundancy + 
                             0.2 * self.clutter;
        
        // Clamp to [0.0, 1.0] range
        weighted_score.min(1.0).max(0.0)
    }
}

/// Interaction metrics for cognitive load estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMetrics {
    /// Detected interaction patterns with confidence scores
    pub patterns: HashMap<InteractionPattern, f64>,
    /// Average response time (seconds)
    pub avg_response_time: f64,
    /// Response time variability (standard deviation)
    pub response_time_variability: f64,
    /// Error rate in interactions
    pub error_rate: f64,
    /// Attention switching frequency (switches per minute)
    pub attention_switches: f64,
    /// Backtracking frequency (returns to previous states)
    pub backtracking_frequency: f64,
}

impl Default for InteractionMetrics {
    fn default() -> Self {
        Self {
            patterns: HashMap::new(),
            avg_response_time: 0.0,
            response_time_variability: 0.0,
            error_rate: 0.0,
            attention_switches: 0.0,
            backtracking_frequency: 0.0,
        }
    }
}

/// Cognitive load estimate with component breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadEstimate {
    /// Overall cognitive load (0.0-1.0)
    pub overall_load: f64,
    /// Component breakdown
    pub components: HashMap<CognitiveLoadComponent, f64>,
    /// Confidence in the estimate (0.0-1.0)
    pub confidence: f64,
    /// Timestamp of the estimate
    pub timestamp: std::time::SystemTime,
    /// Source metrics that informed this estimate
    pub source_metrics: HashMap<String, f64>,
}

impl CognitiveLoadEstimate {
    /// Create a new estimate with current timestamp
    pub fn new(overall_load: f64, confidence: f64) -> Self {
        Self {
            overall_load,
            components: HashMap::new(),
            confidence,
            timestamp: std::time::SystemTime::now(),
            source_metrics: HashMap::new(),
        }
    }

    /// Check if the load exceeds a specified threshold
    pub fn exceeds_threshold(&self, threshold: f64) -> bool {
        self.overall_load > threshold
    }

    /// Get the dominant component of cognitive load
    pub fn dominant_component(&self) -> Option<CognitiveLoadComponent> {
        self.components.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| *k)
    }
}

/// Complexity transformation operation type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplexityTransformation {
    /// Reduce information density by filtering non-essential elements
    FilterNonEssential {
        /// Importance threshold below which elements are filtered
        importance_threshold: f64,
    },
    /// Aggregate similar elements to reduce visual complexity
    Aggregate {
        /// Similarity threshold for aggregation
        similarity_threshold: f64,
        /// Maximum group size for aggregation
        max_group_size: usize,
    },
    /// Highlight essential elements to guide attention
    HighlightEssential {
        /// Number of elements to highlight
        count: usize,
    },
    /// Progressive disclosure to reveal information gradually
    ProgressiveDisclosure {
        /// Number of elements to show initially
        initial_elements: usize,
        /// Criteria for progressive element revelation
        revelation_criteria: String,
    },
    /// Reduce channel utilization by reassigning information to fewer channels
    ReduceChannels {
        /// Maximum number of channels to use
        max_channels: usize,
    },
    /// Switch to a more abstract representation
    AbstractRepresentation {
        /// Level of abstraction (0.0-1.0)
        abstraction_level: f64,
    },
}

/// Strategy for combining multiple transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformationStrategy {
    /// Apply transformations sequentially
    Sequential,
    /// Apply transformations in parallel and merge results
    Parallel,
    /// Apply transformations conditionally based on metrics
    Conditional,
    /// Apply the most effective transformation based on simulation
    Adaptive,
}

/// Parameters for cognitive load estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationParameters {
    /// Weights for different complexity metrics
    pub complexity_weights: HashMap<String, f64>,
    /// Weights for different interaction metrics
    pub interaction_weights: HashMap<String, f64>,
    /// Baseline cognitive load for expertise levels
    pub expertise_baselines: HashMap<ExpertiseLevel, f64>,
    /// Parameters for component estimation
    pub component_parameters: HashMap<CognitiveLoadComponent, HashMap<String, f64>>,
    /// Temporal window size for trend analysis (seconds)
    pub temporal_window: f64,
}

impl Default for EstimationParameters {
    fn default() -> Self {
        let mut complexity_weights = HashMap::new();
        complexity_weights.insert("entropy".to_string(), 0.3);
        complexity_weights.insert("density".to_string(), 0.2);
        complexity_weights.insert("channel_utilization".to_string(), 0.2);
        complexity_weights.insert("redundancy".to_string(), 0.1);
        complexity_weights.insert("clutter".to_string(), 0.2);
        
        let mut interaction_weights = HashMap::new();
        interaction_weights.insert("response_time".to_string(), 0.3);
        interaction_weights.insert("error_rate".to_string(), 0.2);
        interaction_weights.insert("attention_switches".to_string(), 0.2);
        interaction_weights.insert("backtracking".to_string(), 0.3);
        
        let mut expertise_baselines = HashMap::new();
        expertise_baselines.insert(ExpertiseLevel::Beginner, 0.4);
        expertise_baselines.insert(ExpertiseLevel::Intermediate, 0.6);
        expertise_baselines.insert(ExpertiseLevel::Advanced, 0.8);
        expertise_baselines.insert(ExpertiseLevel::Expert, 0.9);
        
        let mut intrinsic_params = HashMap::new();
        intrinsic_params.insert("semantic_weight".to_string(), 0.7);
        intrinsic_params.insert("algorithmic_complexity_weight".to_string(), 0.3);
        
        let mut extraneous_params = HashMap::new();
        extraneous_params.insert("visual_clutter_weight".to_string(), 0.4);
        extraneous_params.insert("channel_overload_weight".to_string(), 0.3);
        extraneous_params.insert("interaction_complexity_weight".to_string(), 0.3);
        
        let mut germane_params = HashMap::new();
        germane_params.insert("schema_construction_weight".to_string(), 0.5);
        germane_params.insert("elaborative_processing_weight".to_string(), 0.5);
        
        let mut component_parameters = HashMap::new();
        component_parameters.insert(CognitiveLoadComponent::Intrinsic, intrinsic_params);
        component_parameters.insert(CognitiveLoadComponent::Extraneous, extraneous_params);
        component_parameters.insert(CognitiveLoadComponent::Germane, germane_params);
        
        Self {
            complexity_weights,
            interaction_weights,
            expertise_baselines,
            component_parameters,
            temporal_window: 60.0, // 1 minute window for trend analysis
        }
    }
}

/// History of cognitive load estimates for trend analysis
#[derive(Debug, Clone)]
pub struct CognitiveLoadHistory {
    /// Historical estimates with timestamps
    estimates: Vec<CognitiveLoadEstimate>,
    /// Maximum history size
    max_size: usize,
}

impl CognitiveLoadHistory {
    /// Create a new history with specified maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            estimates: Vec::with_capacity(max_size),
            max_size,
        }
    }
    
    /// Add a new estimate to the history
    pub fn add_estimate(&mut self, estimate: CognitiveLoadEstimate) {
        if self.estimates.len() >= self.max_size {
            self.estimates.remove(0);
        }
        self.estimates.push(estimate);
    }
    
    /// Calculate linear trend over the historical window
    pub fn calculate_trend(&self) -> f64 {
        if self.estimates.len() < 2 {
            return 0.0;
        }
        
        // Simple linear regression for trend
        let n = self.estimates.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        
        for (i, estimate) in self.estimates.iter().enumerate() {
            let x = i as f64;
            let y = estimate.overall_load;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        slope
    }
    
    /// Calculate moving average of cognitive load
    pub fn moving_average(&self, window_size: usize) -> Option<f64> {
        if self.estimates.is_empty() {
            return None;
        }
        
        let effective_window = std::cmp::min(window_size, self.estimates.len());
        let sum: f64 = self.estimates.iter()
            .rev()
            .take(effective_window)
            .map(|e| e.overall_load)
            .sum();
            
        Some(sum / effective_window as f64)
    }
}

/// Transformation result with metrics
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Original complexity metrics
    pub original_metrics: ComplexityMetrics,
    /// Transformed complexity metrics
    pub transformed_metrics: ComplexityMetrics,
    /// Reduction in cognitive load (estimated)
    pub load_reduction: f64,
    /// Information preserved (0.0-1.0)
    pub information_preserved: f64,
    /// Applied transformations
    pub applied_transformations: Vec<ComplexityTransformation>,
}

/// Multi-factor cognitive load estimator
pub struct CognitiveLoadEstimator {
    /// Estimation parameters
    parameters: EstimationParameters,
    /// Cognitive profile of the user
    profile: CognitiveProfile,
    /// History of cognitive load estimates
    history: CognitiveLoadHistory,
    /// Current transformation pipeline
    transformation_pipeline: Vec<ComplexityTransformation>,
    /// Transformation strategy
    transformation_strategy: TransformationStrategy,
    /// Last update timestamp
    last_update: Instant,
}

impl CognitiveLoadEstimator {
    /// Create a new cognitive load estimator with default parameters
    pub fn new(profile: CognitiveProfile) -> Self {
        Self {
            parameters: EstimationParameters::default(),
            profile,
            history: CognitiveLoadHistory::new(100), // Keep 100 historical estimates
            transformation_pipeline: Vec::new(),
            transformation_strategy: TransformationStrategy::Sequential,
            last_update: Instant::now(),
        }
    }
    
    /// Create with custom parameters
    pub fn with_parameters(profile: CognitiveProfile, parameters: EstimationParameters) -> Self {
        Self {
            parameters,
            profile,
            history: CognitiveLoadHistory::new(100),
            transformation_pipeline: Vec::new(),
            transformation_strategy: TransformationStrategy::Sequential,
            last_update: Instant::now(),
        }
    }
    
    /// Estimate cognitive load from complexity metrics
    pub fn estimate_from_complexity(&self, metrics: &ComplexityMetrics) -> CognitiveLoadResult<CognitiveLoadEstimate> {
        // Calculate overall complexity score
        let complexity_score = metrics.overall_complexity();
        
        // Calculate overall cognitive load based on complexity and user expertise
        let expertise_factor = self.parameters.expertise_baselines
            .get(&self.profile.expertise_level)
            .copied()
            .unwrap_or(0.5);
        
        // Higher expertise allows handling more complexity
        let overall_load = complexity_score * (1.0 / expertise_factor);
        let clamped_load = overall_load.min(1.0).max(0.0);
        
        // Create base estimate
        let mut estimate = CognitiveLoadEstimate::new(clamped_load, 0.7); // 0.7 confidence when using only complexity
        
        // Calculate component breakdown
        let intrinsic_load = self.calculate_intrinsic_load(metrics);
        let extraneous_load = self.calculate_extraneous_load(metrics);
        let germane_load = self.calculate_germane_load(metrics);
        
        estimate.components.insert(CognitiveLoadComponent::Intrinsic, intrinsic_load);
        estimate.components.insert(CognitiveLoadComponent::Extraneous, extraneous_load);
        estimate.components.insert(CognitiveLoadComponent::Germane, germane_load);
        
        // Add source metrics
        estimate.source_metrics.insert("complexity_score".to_string(), complexity_score);
        estimate.source_metrics.insert("expertise_factor".to_string(), expertise_factor);
        
        Ok(estimate)
    }
    
    /// Estimate cognitive load from interaction metrics
    pub fn estimate_from_interaction(&self, metrics: &InteractionMetrics) -> CognitiveLoadResult<CognitiveLoadEstimate> {
        // Calculate cognitive load from interaction patterns
        let mut pattern_load = 0.0;
        let mut pattern_count = 0;
        
        // Patterns that indicate high cognitive load
        let high_load_patterns = [
            InteractionPattern::RepeatedViewing,
            InteractionPattern::LongPauses,
            InteractionPattern::ErraticMovement,
            InteractionPattern::AbandonedInteraction,
        ];
        
        for pattern in &high_load_patterns {
            if let Some(confidence) = metrics.patterns.get(pattern) {
                pattern_load += confidence;
                pattern_count += 1;
            }
        }
        
        let pattern_score = if pattern_count > 0 {
            pattern_load / pattern_count as f64
        } else {
            0.5 // Default if no patterns detected
        };
        
        // Response time impact (normalized to 0-1 range, assuming 5s is high cognitive load)
        let response_time_score = (metrics.avg_response_time / 5.0).min(1.0);
        
        // Error rate impact
        let error_score = metrics.error_rate.min(1.0);
        
        // Combined interaction score
        let interaction_score = 0.4 * pattern_score + 
                               0.3 * response_time_score + 
                               0.3 * error_score;
        
        // Adjust for expertise
        let expertise_factor = self.parameters.expertise_baselines
            .get(&self.profile.expertise_level)
            .copied()
            .unwrap_or(0.5);
        
        let overall_load = interaction_score * (1.0 / expertise_factor);
        let clamped_load = overall_load.min(1.0).max(0.0);
        
        // Create estimate
        let mut estimate = CognitiveLoadEstimate::new(clamped_load, 0.6); // 0.6 confidence for interaction-based
        
        // Component breakdown (simplified for interaction metrics)
        estimate.components.insert(CognitiveLoadComponent::Intrinsic, 0.3 * clamped_load);
        estimate.components.insert(CognitiveLoadComponent::Extraneous, 0.6 * clamped_load);
        estimate.components.insert(CognitiveLoadComponent::Germane, 0.1 * clamped_load);
        
        // Add source metrics
        estimate.source_metrics.insert("pattern_score".to_string(), pattern_score);
        estimate.source_metrics.insert("response_time_score".to_string(), response_time_score);
        estimate.source_metrics.insert("error_score".to_string(), error_score);
        
        Ok(estimate)
    }
    
    /// Estimate cognitive load from both complexity and interaction metrics
    pub fn estimate_combined(&mut self, 
                           complexity: &ComplexityMetrics, 
                           interaction: &InteractionMetrics) -> CognitiveLoadResult<CognitiveLoadEstimate> {
        // Get individual estimates
        let complexity_estimate = self.estimate_from_complexity(complexity)?;
        let interaction_estimate = self.estimate_from_interaction(interaction)?;
        
        // Combined load with weighted average (higher weight to interaction as it's more direct)
        let combined_load = 0.4 * complexity_estimate.overall_load + 
                           0.6 * interaction_estimate.overall_load;
        
        // Create combined estimate with higher confidence
        let mut estimate = CognitiveLoadEstimate::new(combined_load, 0.85);
        
        // Combine component breakdown
        for component in [CognitiveLoadComponent::Intrinsic, 
                          CognitiveLoadComponent::Extraneous, 
                          CognitiveLoadComponent::Germane] {
            let complexity_component = complexity_estimate.components.get(&component).copied().unwrap_or(0.0);
            let interaction_component = interaction_estimate.components.get(&component).copied().unwrap_or(0.0);
            
            let combined_component = 0.4 * complexity_component + 0.6 * interaction_component;
            estimate.components.insert(component, combined_component);
        }
        
        // Combine source metrics
        estimate.source_metrics.extend(complexity_estimate.source_metrics);
        estimate.source_metrics.extend(interaction_estimate.source_metrics);
        
        // Add to history
        self.history.add_estimate(estimate.clone());
        self.last_update = Instant::now();
        
        Ok(estimate)
    }
    
    /// Calculate information-theoretic complexity metrics for a visualization
    pub fn calculate_complexity_metrics(&self, representation: &ModalityRepresentation) -> CognitiveLoadResult<ComplexityMetrics> {
        // This is a complex calculation that depends on the specific visualization
        // Here we provide a simplified implementation
        
        let mut metrics = ComplexityMetrics::new();
        
        match representation {
            ModalityRepresentation::Visual(visual) => {
                // Calculate visual complexity metrics
                
                // Number of visual elements as a proxy for semantic complexity
                metrics.semantic_complexity = visual.element_count.unwrap_or(0);
                
                // Information entropy based on distribution of visual properties
                metrics.entropy = visual.estimated_entropy.unwrap_or(0.0);
                
                // Information density based on visual space utilization
                let area = visual.viewport_area.unwrap_or(1.0);
                metrics.density = (metrics.semantic_complexity as f64) / area;
                
                // Channel utilization
                if let Some(channels) = &visual.channel_utilization {
                    for (channel_name, utilization) in channels {
                        if let Ok(channel) = parse_channel(channel_name) {
                            metrics.channel_utilization.insert(channel, *utilization);
                        }
                    }
                }
                
                // Visual clutter estimate
                metrics.clutter = visual.clutter_estimate.unwrap_or(0.0);
                
                // Redundancy estimate
                metrics.redundancy = visual.redundancy_estimate.unwrap_or(0.0);
            },
            ModalityRepresentation::Auditory(auditory) => {
                // Calculate auditory complexity metrics
                
                // Number of sound elements as semantic complexity
                metrics.semantic_complexity = auditory.element_count.unwrap_or(0);
                
                // Information entropy based on sound distribution
                metrics.entropy = auditory.estimated_entropy.unwrap_or(0.0);
                
                // Information density based on temporal density
                let duration = auditory.duration_seconds.unwrap_or(1.0);
                metrics.density = (metrics.semantic_complexity as f64) / duration;
                
                // Simplified channel utilization for auditory representation
                metrics.channel_utilization.insert(InformationChannel::Position, 
                    auditory.spatial_utilization.unwrap_or(0.0));
                metrics.channel_utilization.insert(InformationChannel::Color, 
                    auditory.timbral_utilization.unwrap_or(0.0));
                
                // Clutter equivalent for auditory (overlapping sounds)
                metrics.clutter = auditory.overlap_estimate.unwrap_or(0.0);
                
                // Redundancy estimate
                metrics.redundancy = auditory.redundancy_estimate.unwrap_or(0.0);
            },
            ModalityRepresentation::Tactile(tactile) => {
                // Calculate tactile complexity metrics
                
                // Number of tactile elements as semantic complexity
                metrics.semantic_complexity = tactile.element_count.unwrap_or(0);
                
                // Information entropy for tactile representation
                metrics.entropy = tactile.estimated_entropy.unwrap_or(0.0);
                
                // Information density based on spatial density
                let area = tactile.surface_area.unwrap_or(1.0);
                metrics.density = (metrics.semantic_complexity as f64) / area;
                
                // Simplified channel utilization for tactile
                metrics.channel_utilization.insert(InformationChannel::Position, 
                    tactile.spatial_utilization.unwrap_or(0.0));
                metrics.channel_utilization.insert(InformationChannel::Texture, 
                    tactile.texture_utilization.unwrap_or(0.0));
                
                // Clutter equivalent (overlapping tactile elements)
                metrics.clutter = tactile.overlap_estimate.unwrap_or(0.0);
                
                // Redundancy estimate
                metrics.redundancy = tactile.redundancy_estimate.unwrap_or(0.0);
            },
            ModalityRepresentation::Multimodal(multimodal) => {
                // For multimodal, we take the weighted average of the component representations
                let mut component_metrics = Vec::new();
                let mut weights = Vec::new();
                
                for (representation, weight) in &multimodal.components {
                    if let Ok(metrics) = self.calculate_complexity_metrics(representation) {
                        component_metrics.push(metrics);
                        weights.push(*weight);
                    }
                }
                
                if component_metrics.is_empty() {
                    return Err(CognitiveLoadError::InsufficientData(
                        "No valid component representations in multimodal representation".to_string()));
                }
                
                // Calculate weighted averages
                let total_weight: f64 = weights.iter().sum();
                
                // Semantic complexity uses maximum rather than average
                metrics.semantic_complexity = component_metrics.iter()
                    .map(|m| m.semantic_complexity)
                    .max()
                    .unwrap_or(0);
                
                // Other metrics use weighted average
                for (i, component) in component_metrics.iter().enumerate() {
                    let normalized_weight = weights[i] / total_weight;
                    
                    metrics.entropy += component.entropy * normalized_weight;
                    metrics.density += component.density * normalized_weight;
                    metrics.redundancy += component.redundancy * normalized_weight;
                    metrics.clutter += component.clutter * normalized_weight;
                    
                    for (channel, utilization) in &component.channel_utilization {
                        let current = metrics.channel_utilization
                            .entry(*channel)
                            .or_insert(0.0);
                        *current += utilization * normalized_weight;
                    }
                }
            },
        }
        
        Ok(metrics)
    }
    
    /// Get appropriate transformations for the given load estimate
    pub fn get_transformations(&self, 
                             estimate: &CognitiveLoadEstimate,
                             metrics: &ComplexityMetrics) -> Vec<ComplexityTransformation> {
        // Don't transform if load is acceptable
        if estimate.overall_load <= self.profile.load_tolerance {
            return Vec::new();
        }
        
        let mut transformations = Vec::new();
        
        // Determine which component contributes most to cognitive load
        let dominant_component = estimate.dominant_component()
            .unwrap_or(CognitiveLoadComponent::Extraneous);
        
        // Transformations based on the dominant component
        match dominant_component {
            CognitiveLoadComponent::Intrinsic => {
                // Intrinsic load is inherent to the material, so we need abstraction
                transformations.push(ComplexityTransformation::AbstractRepresentation {
                    abstraction_level: 0.7,
                });
            },
            CognitiveLoadComponent::Extraneous => {
                // Extraneous load can be reduced by improving presentation
                
                // If many channels are overutilized
                if metrics.channel_utilization.values().filter(|&&u| u > 0.7).count() > 2 {
                    transformations.push(ComplexityTransformation::ReduceChannels {
                        max_channels: 3, // Limit to three primary information channels
                    });
                }
                
                // If visual clutter is high
                if metrics.clutter > 0.6 {
                    transformations.push(ComplexityTransformation::FilterNonEssential {
                        importance_threshold: 0.3, // Filter elements below 30% importance
                    });
                }
                
                // If semantic complexity is high
                if metrics.semantic_complexity > 20 {
                    transformations.push(ComplexityTransformation::Aggregate {
                        similarity_threshold: 0.7,
                        max_group_size: 5,
                    });
                }
            },
            CognitiveLoadComponent::Germane => {
                // Germane load is productive, but we can help focus attention
                transformations.push(ComplexityTransformation::HighlightEssential {
                    count: 5, // Highlight the 5 most important elements
                });
                
                // Progressive disclosure can help manage germane load
                transformations.push(ComplexityTransformation::ProgressiveDisclosure {
                    initial_elements: 10,
                    revelation_criteria: "importance".to_string(),
                });
            },
        }
        
        // Limit the number of simultaneous transformations to avoid confusion
        if transformations.len() > 2 {
            transformations.truncate(2);
        }
        
        transformations
    }
    
    /// Apply transformations to reduce cognitive load
    pub fn apply_transformations(&self, 
                               representation: &ModalityRepresentation,
                               transformations: &[ComplexityTransformation]) 
        -> CognitiveLoadResult<(ModalityRepresentation, TransformationResult)> {
        
        if transformations.is_empty() {
            return Err(CognitiveLoadError::InsufficientData("No transformations provided".to_string()));
        }
        
        // Calculate original complexity metrics
        let original_metrics = self.calculate_complexity_metrics(representation)?;
        
        // Apply transformations according to strategy
        let mut transformed = representation.clone();
        
        match self.transformation_strategy {
            TransformationStrategy::Sequential => {
                // Apply transformations in sequence
                for transformation in transformations {
                    transformed = self.apply_single_transformation(&transformed, transformation)?;
                }
            },
            TransformationStrategy::Parallel => {
                // Apply transformations in parallel and merge results
                // This is a simplified implementation - actual parallel application
                // would be more complex and require more sophisticated merging
                let mut transformed_versions = Vec::new();
                
                for transformation in transformations {
                    if let Ok(transformed_version) = self.apply_single_transformation(representation, transformation) {
                        transformed_versions.push(transformed_version);
                    }
                }
                
                if transformed_versions.is_empty() {
                    return Err(CognitiveLoadError::TransformationError(
                        "All parallel transformations failed".to_string()));
                }
                
                // Merge transformed versions (simplified)
                transformed = transformed_versions[0].clone();
            },
            TransformationStrategy::Conditional => {
                // Apply transformations conditionally
                // Simplified implementation - would be more sophisticated in practice
                for transformation in transformations {
                    if self.should_apply_transformation(&transformed, transformation) {
                        transformed = self.apply_single_transformation(&transformed, transformation)?;
                    }
                }
            },
            TransformationStrategy::Adaptive => {
                // Apply the most effective transformation
                let mut best_transformation = None;
                let mut best_score = f64::INFINITY;
                
                for transformation in transformations {
                    // Simulate application
                    if let Ok(simulated) = self.apply_single_transformation(representation, transformation) {
                        if let Ok(metrics) = self.calculate_complexity_metrics(&simulated) {
                            let score = metrics.overall_complexity();
                            if score < best_score {
                                best_score = score;
                                best_transformation = Some(transformation);
                            }
                        }
                    }
                }
                
                if let Some(transformation) = best_transformation {
                    transformed = self.apply_single_transformation(representation, transformation)?;
                } else {
                    return Err(CognitiveLoadError::TransformationError(
                        "No valid transformation found".to_string()));
                }
            },
        }
        
        // Calculate transformed metrics
        let transformed_metrics = self.calculate_complexity_metrics(&transformed)?;
        
        // Calculate load reduction and information preservation
        let original_complexity = original_metrics.overall_complexity();
        let transformed_complexity = transformed_metrics.overall_complexity();
        
        let load_reduction = (original_complexity - transformed_complexity)
            .max(0.0); // Ensure non-negative
            
        // Information preservation is the ratio of essential information retained
        // This is a simplified calculation - a more sophisticated approach would be used in practice
        let information_preserved = 1.0 - (load_reduction * 0.5); // Assuming 50% of reduction affects essential information
        
        let result = TransformationResult {
            original_metrics,
            transformed_metrics,
            load_reduction,
            information_preserved,
            applied_transformations: transformations.to_vec(),
        };
        
        Ok((transformed, result))
    }
    
    /// Apply a single transformation
    fn apply_single_transformation(&self, 
                                 representation: &ModalityRepresentation,
                                 transformation: &ComplexityTransformation) 
        -> CognitiveLoadResult<ModalityRepresentation> {
        
        // This is a placeholder implementation - actual transformation would be more complex
        let mut transformed = representation.clone();
        
        // Apply transformation based on type
        match transformation {
            ComplexityTransformation::FilterNonEssential { importance_threshold } => {
                // Filter non-essential elements based on importance
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Filter visual elements
                        visual.filter_by_importance = Some(*importance_threshold);
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Filter auditory elements
                        auditory.filter_by_importance = Some(*importance_threshold);
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Filter tactile elements
                        tactile.filter_by_importance = Some(*importance_threshold);
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::FilterNonEssential { 
                                    importance_threshold: *importance_threshold 
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
            ComplexityTransformation::Aggregate { similarity_threshold, max_group_size } => {
                // Aggregate similar elements
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Aggregate visual elements
                        visual.aggregation_threshold = Some(*similarity_threshold);
                        visual.max_group_size = Some(*max_group_size);
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Aggregate auditory elements
                        auditory.aggregation_threshold = Some(*similarity_threshold);
                        auditory.max_group_size = Some(*max_group_size);
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Aggregate tactile elements
                        tactile.aggregation_threshold = Some(*similarity_threshold);
                        tactile.max_group_size = Some(*max_group_size);
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::Aggregate { 
                                    similarity_threshold: *similarity_threshold,
                                    max_group_size: *max_group_size,
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
            ComplexityTransformation::HighlightEssential { count } => {
                // Highlight essential elements
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Highlight visual elements
                        visual.highlight_count = Some(*count);
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Highlight auditory elements
                        auditory.highlight_count = Some(*count);
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Highlight tactile elements
                        tactile.highlight_count = Some(*count);
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::HighlightEssential { 
                                    count: *count,
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
            ComplexityTransformation::ProgressiveDisclosure { initial_elements, revelation_criteria } => {
                // Progressive disclosure
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Progressive disclosure for visual
                        visual.initial_elements = Some(*initial_elements);
                        visual.revelation_criteria = Some(revelation_criteria.clone());
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Progressive disclosure for auditory
                        auditory.initial_elements = Some(*initial_elements);
                        auditory.revelation_criteria = Some(revelation_criteria.clone());
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Progressive disclosure for tactile
                        tactile.initial_elements = Some(*initial_elements);
                        tactile.revelation_criteria = Some(revelation_criteria.clone());
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::ProgressiveDisclosure { 
                                    initial_elements: *initial_elements,
                                    revelation_criteria: revelation_criteria.clone(),
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
            ComplexityTransformation::ReduceChannels { max_channels } => {
                // Reduce information channels
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Reduce visual channels
                        visual.max_channels = Some(*max_channels);
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Reduce auditory channels
                        auditory.max_channels = Some(*max_channels);
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Reduce tactile channels
                        tactile.max_channels = Some(*max_channels);
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::ReduceChannels { 
                                    max_channels: *max_channels,
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
            ComplexityTransformation::AbstractRepresentation { abstraction_level } => {
                // Abstract representation
                match &mut transformed {
                    ModalityRepresentation::Visual(visual) => {
                        // Abstract visual representation
                        visual.abstraction_level = Some(*abstraction_level);
                    },
                    ModalityRepresentation::Auditory(auditory) => {
                        // Abstract auditory representation
                        auditory.abstraction_level = Some(*abstraction_level);
                    },
                    ModalityRepresentation::Tactile(tactile) => {
                        // Abstract tactile representation
                        tactile.abstraction_level = Some(*abstraction_level);
                    },
                    ModalityRepresentation::Multimodal(multimodal) => {
                        // Apply to each component
                        for (component, _) in &mut multimodal.components {
                            if let Ok(transformed_component) = self.apply_single_transformation(
                                component, 
                                &ComplexityTransformation::AbstractRepresentation { 
                                    abstraction_level: *abstraction_level,
                                }) {
                                *component = transformed_component;
                            }
                        }
                    },
                }
            },
        }
        
        Ok(transformed)
    }
    
    /// Determine if a transformation should be applied based on current state
    fn should_apply_transformation(&self, 
                                 representation: &ModalityRepresentation,
                                 transformation: &ComplexityTransformation) -> bool {
        // This is a placeholder implementation - actual determination would be more sophisticated
        
        // Calculate metrics to inform decision
        let metrics = match self.calculate_complexity_metrics(representation) {
            Ok(m) => m,
            Err(_) => return false, // If we can't calculate metrics, don't apply
        };
        
        match transformation {
            ComplexityTransformation::FilterNonEssential { importance_threshold: _ } => {
                // Apply filtering if clutter is high
                metrics.clutter > 0.6
            },
            ComplexityTransformation::Aggregate { similarity_threshold: _, max_group_size: _ } => {
                // Apply aggregation if semantic complexity is high
                metrics.semantic_complexity > 20
            },
            ComplexityTransformation::HighlightEssential { count: _ } => {
                // Apply highlighting if density is high
                metrics.density > 0.7
            },
            ComplexityTransformation::ProgressiveDisclosure { initial_elements: _, revelation_criteria: _ } => {
                // Apply progressive disclosure if overall complexity is high
                metrics.overall_complexity() > 0.8
            },
            ComplexityTransformation::ReduceChannels { max_channels: _ } => {
                // Apply channel reduction if multiple channels are overutilized
                metrics.channel_utilization.values().filter(|&&u| u > 0.7).count() > 2
            },
            ComplexityTransformation::AbstractRepresentation { abstraction_level: _ } => {
                // Apply abstraction if entropy is high
                metrics.entropy > 4.0 // 4 bits is approximately 16 distinct states
            },
        }
    }
    
    /// Calculate intrinsic cognitive load component
    fn calculate_intrinsic_load(&self, metrics: &ComplexityMetrics) -> f64 {
        // Get parameters for intrinsic load calculation
        let params = self.parameters.component_parameters
            .get(&CognitiveLoadComponent::Intrinsic)
            .cloned()
            .unwrap_or_else(HashMap::new);
        
        let semantic_weight = params.get("semantic_weight").copied().unwrap_or(0.7);
        let complexity_weight = params.get("algorithmic_complexity_weight").copied().unwrap_or(0.3);
        
        // Semantic complexity normalized (assuming max of 100 elements)
        let normalized_semantic = (metrics.semantic_complexity as f64 / 100.0).min(1.0);
        
        // Algorithmic complexity from entropy (normalized assuming max of 10 bits)
        let normalized_entropy = (metrics.entropy / 10.0).min(1.0);
        
        // Weighted combination
        let intrinsic_load = semantic_weight * normalized_semantic + 
                            complexity_weight * normalized_entropy;
        
        // Apply expertise adjustment (higher expertise reduces intrinsic load)
        let expertise_factor = match self.profile.expertise_level {
            ExpertiseLevel::Beginner => 1.0,
            ExpertiseLevel::Intermediate => 0.8,
            ExpertiseLevel::Advanced => 0.6,
            ExpertiseLevel::Expert => 0.4,
        };
        
        (intrinsic_load * expertise_factor).min(1.0)
    }
    
    /// Calculate extraneous cognitive load component
    fn calculate_extraneous_load(&self, metrics: &ComplexityMetrics) -> f64 {
        // Get parameters for extraneous load calculation
        let params = self.parameters.component_parameters
            .get(&CognitiveLoadComponent::Extraneous)
            .cloned()
            .unwrap_or_else(HashMap::new);
        
        let clutter_weight = params.get("visual_clutter_weight").copied().unwrap_or(0.4);
        let channel_weight = params.get("channel_overload_weight").copied().unwrap_or(0.3);
        let interaction_weight = params.get("interaction_complexity_weight").copied().unwrap_or(0.3);
        
        // Visual clutter directly contributes to extraneous load
        let clutter_load = metrics.clutter;
        
        // Channel overload - average utilization above 0.7 threshold
        let channel_overload = metrics.channel_utilization.values()
            .map(|&u| (u - 0.7).max(0.0))
            .sum::<f64>();
        let normalized_channel_overload = (channel_overload / 
            metrics.channel_utilization.len().max(1) as f64).min(1.0);
        
        // Interaction complexity (placeholder - would be calculated from actual interaction data)
        let interaction_complexity = metrics.density * 0.5 + metrics.redundancy * 0.5;
        
        // Weighted combination
        let extraneous_load = clutter_weight * clutter_load + 
                             channel_weight * normalized_channel_overload + 
                             interaction_weight * interaction_complexity;
        
        // Extraneous load is not directly affected by expertise
        extraneous_load.min(1.0)
    }
    
    /// Calculate germane cognitive load component
    fn calculate_germane_load(&self, metrics: &ComplexityMetrics) -> f64 {
        // Get parameters for germane load calculation
        let params = self.parameters.component_parameters
            .get(&CognitiveLoadComponent::Germane)
            .cloned()
            .unwrap_or_else(HashMap::new);
        
        let schema_weight = params.get("schema_construction_weight").copied().unwrap_or(0.5);
        let elaboration_weight = params.get("elaborative_processing_weight").copied().unwrap_or(0.5);
        
        // Schema construction - related to semantic complexity and redundancy
        // Higher redundancy aids schema construction
        let schema_load = (metrics.semantic_complexity as f64 / 100.0) * (1.0 + metrics.redundancy);
        let normalized_schema = schema_load.min(1.0);
        
        // Elaborative processing - related to information density and entropy
        // Higher entropy requires more elaborative processing
        let elaboration_load = metrics.entropy * 0.1 + metrics.density * 0.5;
        let normalized_elaboration = elaboration_load.min(1.0);
        
        // Weighted combination
        let germane_load = schema_weight * normalized_schema + 
                          elaboration_weight * normalized_elaboration;
        
        // Apply expertise adjustment (higher expertise increases germane load capacity)
        let expertise_factor = match self.profile.expertise_level {
            ExpertiseLevel::Beginner => 0.5,
            ExpertiseLevel::Intermediate => 0.7,
            ExpertiseLevel::Advanced => 0.9,
            ExpertiseLevel::Expert => 1.0,
        };
        
        (germane_load * expertise_factor).min(1.0)
    }
    
    /// Update the cognitive profile based on interaction data
    pub fn update_profile(&mut self, 
                        estimate: &CognitiveLoadEstimate,
                        interaction: &InteractionMetrics) {
        // Update load tolerance based on historical performance
        // If the user consistently handles higher loads, increase tolerance
        let trend = self.history.calculate_trend();
        if trend < 0.0 && self.history.estimates.len() > 10 {
            // Decreasing load trend with sufficient history
            self.profile.load_tolerance = (self.profile.load_tolerance + 0.05).min(0.9);
        } else if trend > 0.0 && estimate.exceeds_threshold(self.profile.load_tolerance) {
            // Increasing load trend and currently over threshold
            self.profile.load_tolerance = (self.profile.load_tolerance - 0.05).max(0.3);
        }
        
        // Update preferred modality based on performance
        // This is a placeholder - actual update would be more sophisticated
        if interaction.error_rate < 0.1 && interaction.avg_response_time < 2.0 {
            // Good performance with current modality - no change needed
        }
        
        // Log the profile update
        debug!("Updated cognitive profile: load_tolerance={}", self.profile.load_tolerance);
    }
    
    /// Set transformation strategy
    pub fn set_transformation_strategy(&mut self, strategy: TransformationStrategy) {
        self.transformation_strategy = strategy;
    }
    
    /// Set transformation pipeline
    pub fn set_transformation_pipeline(&mut self, pipeline: Vec<ComplexityTransformation>) {
        self.transformation_pipeline = pipeline;
    }
    
    /// Get the current cognitive profile
    pub fn get_profile(&self) -> &CognitiveProfile {
        &self.profile
    }
    
    /// Get mutable reference to the current cognitive profile
    pub fn get_profile_mut(&mut self) -> &mut CognitiveProfile {
        &mut self.profile
    }
    
    /// Get the current load history
    pub fn get_history(&self) -> &CognitiveLoadHistory {
        &self.history
    }
    
    /// Check if the cognitive load estimator needs updating
    pub fn needs_update(&self, max_interval: Duration) -> bool {
        self.last_update.elapsed() > max_interval
    }
}

/// Parse channel name to enum
fn parse_channel(name: &str) -> Result<InformationChannel, String> {
    match name.to_lowercase().as_str() {
        "position" => Ok(InformationChannel::Position),
        "color" => Ok(InformationChannel::Color),
        "shape" => Ok(InformationChannel::Shape),
        "size" => Ok(InformationChannel::Size),
        "orientation" => Ok(InformationChannel::Orientation),
        "texture" => Ok(InformationChannel::Texture),
        "motion" => Ok(InformationChannel::Motion),
        "depth" => Ok(InformationChannel::Depth),
        "connection" => Ok(InformationChannel::Connection),
        "text" => Ok(InformationChannel::Text),
        _ => Err(format!("Unknown channel: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complexity_metrics_overall_complexity() {
        let mut metrics = ComplexityMetrics::new();
        metrics.entropy = 3.0;
        metrics.density = 0.5;
        metrics.clutter = 0.7;
        metrics.semantic_complexity = 20;
        
        let mut channels = HashMap::new();
        channels.insert(InformationChannel::Color, 0.8);
        channels.insert(InformationChannel::Shape, 0.6);
        metrics.channel_utilization = channels;
        
        let complexity = metrics.overall_complexity();
        assert!(complexity > 0.0 && complexity <= 1.0);
    }
    
    #[test]
    fn test_cognitive_load_estimate() {
        let mut estimate = CognitiveLoadEstimate::new(0.7, 0.8);
        estimate.components.insert(CognitiveLoadComponent::Intrinsic, 0.4);
        estimate.components.insert(CognitiveLoadComponent::Extraneous, 0.5);
        estimate.components.insert(CognitiveLoadComponent::Germane, 0.3);
        
        assert!(estimate.exceeds_threshold(0.6));
        assert!(!estimate.exceeds_threshold(0.8));
        
        assert_eq!(estimate.dominant_component(), Some(CognitiveLoadComponent::Extraneous));
    }
    
    #[test]
    fn test_cognitive_load_history() {
        let mut history = CognitiveLoadHistory::new(3);
        
        let estimate1 = CognitiveLoadEstimate::new(0.5, 0.8);
        let estimate2 = CognitiveLoadEstimate::new(0.6, 0.8);
        let estimate3 = CognitiveLoadEstimate::new(0.7, 0.8);
        let estimate4 = CognitiveLoadEstimate::new(0.8, 0.8);
        
        history.add_estimate(estimate1);
        history.add_estimate(estimate2);
        history.add_estimate(estimate3);
        
        // Test that history maintains max size
        assert_eq!(history.estimates.len(), 3);
        
        // Adding one more should remove the oldest
        history.add_estimate(estimate4);
        assert_eq!(history.estimates.len(), 3);
        assert_eq!(history.estimates[0].overall_load, 0.6);
        
        // Test trend calculation
        let trend = history.calculate_trend();
        assert!(trend > 0.0); // Should be positive (increasing load)
        
        // Test moving average
        let avg = history.moving_average(2).unwrap();
        assert!((avg - 0.75).abs() < 0.0001); // Average of last two estimates (0.7 and 0.8)
    }
}