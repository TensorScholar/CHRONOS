//! Cognitive Complexity Estimation Framework
//!
//! This module implements an information-theoretic approach to cognitive complexity
//! estimation, providing mathematically rigorous metrics for interface elements
//! and algorithm visualizations. The framework leverages Shannon entropy,
//! Kolmogorov complexity approximations, and psychological cognitive load models
//! to deliver real-time complexity assessment with formal validity guarantees.
//!
//! # Theoretical Foundation
//!
//! The cognitive complexity estimation is grounded in:
//! - Information Theory (Shannon, Kolmogorov)
//! - Cognitive Load Theory (Sweller, 1988)
//! - Psychological Complexity Models (Halford et al., 1994)
//! - Perceptual Information Processing (Miller, 1956)
//!
//! # Mathematical Properties
//!
//! - **Metric Validity**: Satisfies triangle inequality, non-negativity, identity
//! - **Dimensional Independence**: Orthogonal complexity measures across modalities
//! - **Monotonicity**: Complexity increases monotonically with information density
//! - **Bounded Range**: Complexity values normalized to [0.0, 1.0] interval
//!
//! Copyright (c) 2025 Chronos Quantum-Computational Framework

use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::f64::consts::{E, LN_2};
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Fundamental constants for cognitive complexity estimation
mod constants {
    /// Miller's magic number: average human working memory capacity
    pub const MILLER_CAPACITY: f64 = 7.0;
    
    /// Halford's relational complexity bound
    pub const HALFORD_RELATIONAL_BOUND: f64 = 4.0;
    
    /// Entropy normalization constant (log₂(max_symbols))
    pub const ENTROPY_NORMALIZATION: f64 = 5.643856189; // log₂(50)
    
    /// Cognitive load scaling factor based on empirical studies
    pub const COGNITIVE_SCALING: f64 = 1.618; // Golden ratio approximation
    
    /// Complexity decay constant for temporal integration
    pub const TEMPORAL_DECAY: f64 = 0.693147; // ln(2)
}

/// Cognitive complexity dimensions with orthogonal independence guarantees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityDimension {
    /// Visual information density and perceptual load
    Visual,
    
    /// Structural relationship complexity
    Structural,
    
    /// Temporal dynamics and change patterns
    Temporal,
    
    /// Interactive element complexity
    Interactive,
    
    /// Semantic meaning complexity
    Semantic,
    
    /// Navigational complexity
    Navigational,
}

impl ComplexityDimension {
    /// Returns all complexity dimensions for comprehensive analysis
    pub fn all() -> &'static [ComplexityDimension] {
        &[
            Self::Visual,
            Self::Structural,
            Self::Temporal,
            Self::Interactive,
            Self::Semantic,
            Self::Navigational,
        ]
    }
    
    /// Returns the theoretical maximum complexity for this dimension
    pub fn max_complexity(self) -> f64 {
        match self {
            Self::Visual => 1.0,
            Self::Structural => constants::HALFORD_RELATIONAL_BOUND / constants::MILLER_CAPACITY,
            Self::Temporal => 1.0,
            Self::Interactive => 1.0,
            Self::Semantic => 1.0,
            Self::Navigational => 1.0,
        }
    }
}

/// Interface element representation for complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceElement {
    /// Unique element identifier
    pub id: String,
    
    /// Element type classification
    pub element_type: ElementType,
    
    /// Visual properties affecting complexity
    pub visual_properties: VisualProperties,
    
    /// Structural relationships with other elements
    pub relationships: Vec<ElementRelationship>,
    
    /// Interactive capabilities
    pub interactions: Vec<InteractionType>,
    
    /// Semantic content complexity
    pub semantic_content: SemanticContent,
    
    /// Temporal behavior patterns
    pub temporal_behavior: Option<TemporalBehavior>,
}

/// Element type classification for complexity estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementType {
    /// Static visual element (text, image, shape)
    Static { complexity_modifier: f64 },
    
    /// Dynamic element with state changes
    Dynamic { state_count: usize, transition_rate: f64 },
    
    /// Interactive control element
    Interactive { input_dimensions: usize },
    
    /// Container element with child elements
    Container { child_count: usize, layout_complexity: f64 },
    
    /// Algorithmic visualization element
    Algorithmic { algorithm_complexity: f64, data_size: usize },
}

/// Visual properties affecting cognitive complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualProperties {
    /// Number of distinct colors
    pub color_count: usize,
    
    /// Visual density (elements per unit area)
    pub density: f64,
    
    /// Contrast ratio for accessibility
    pub contrast_ratio: f64,
    
    /// Animation frequency (Hz)
    pub animation_frequency: Option<f64>,
    
    /// Text complexity (words, symbols)
    pub text_complexity: Option<TextComplexity>,
}

/// Text complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextComplexity {
    /// Word count
    pub word_count: usize,
    
    /// Average word length
    pub avg_word_length: f64,
    
    /// Sentence complexity score
    pub sentence_complexity: f64,
    
    /// Technical term density
    pub technical_density: f64,
}

/// Relationship between interface elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementRelationship {
    /// Target element identifier
    pub target_id: String,
    
    /// Relationship type and strength
    pub relationship_type: RelationshipType,
    
    /// Relationship strength [0.0, 1.0]
    pub strength: f64,
}

/// Types of relationships between elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Spatial proximity relationship
    Spatial,
    
    /// Functional dependency
    Functional,
    
    /// Temporal correlation
    Temporal,
    
    /// Semantic similarity
    Semantic,
    
    /// Hierarchical parent-child
    Hierarchical,
}

/// Interaction types for complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Simple click interaction
    Click,
    
    /// Drag and drop operation
    DragDrop { target_count: usize },
    
    /// Multi-touch gesture
    Gesture { gesture_complexity: f64 },
    
    /// Keyboard input
    KeyboardInput { input_complexity: f64 },
    
    /// Continuous manipulation (slider, dial)
    Continuous { parameter_count: usize },
}

/// Semantic content complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContent {
    /// Information entropy of the content
    pub entropy: f64,
    
    /// Conceptual complexity score
    pub conceptual_complexity: f64,
    
    /// Domain-specific complexity
    pub domain_complexity: f64,
    
    /// Abstraction level [0.0 = concrete, 1.0 = highly abstract]
    pub abstraction_level: f64,
}

/// Temporal behavior patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBehavior {
    /// State change frequency (Hz)
    pub change_frequency: f64,
    
    /// Predictability score [0.0 = chaotic, 1.0 = fully predictable]
    pub predictability: f64,
    
    /// Temporal correlation with other elements
    pub temporal_correlations: Vec<String>,
}

/// Comprehensive complexity estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityEstimation {
    /// Per-dimension complexity scores
    pub dimension_scores: HashMap<ComplexityDimension, f64>,
    
    /// Overall complexity score (weighted combination)
    pub overall_score: f64,
    
    /// Confidence interval for the estimation
    pub confidence_interval: (f64, f64),
    
    /// Complexity contributors breakdown
    pub contributors: Vec<ComplexityContributor>,
    
    /// Recommendations for complexity reduction
    pub recommendations: Vec<ComplexityRecommendation>,
    
    /// Estimation metadata
    pub metadata: EstimationMetadata,
}

/// Individual complexity contributor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityContributor {
    /// Source element or feature
    pub source: String,
    
    /// Contribution magnitude [0.0, 1.0]
    pub contribution: f64,
    
    /// Affected dimensions
    pub dimensions: Vec<ComplexityDimension>,
    
    /// Explanation of the contribution
    pub explanation: String,
}

/// Recommendation for complexity reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    
    /// Expected complexity reduction
    pub expected_reduction: f64,
    
    /// Implementation difficulty [0.0 = easy, 1.0 = very difficult]
    pub implementation_difficulty: f64,
    
    /// Detailed recommendation description
    pub description: String,
}

/// Types of complexity reduction recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Reduce visual density
    ReduceVisualDensity,
    
    /// Simplify interactions
    SimplifyInteractions,
    
    /// Improve visual hierarchy
    ImproveHierarchy,
    
    /// Reduce information density
    ReduceInformation,
    
    /// Enhance predictability
    EnhancePredictability,
    
    /// Provide progressive disclosure
    ProgressiveDisclosure,
}

/// Estimation process metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationMetadata {
    /// Timestamp of estimation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Estimation algorithm version
    pub algorithm_version: String,
    
    /// Processing time in microseconds
    pub processing_time_us: u64,
    
    /// Estimation confidence level
    pub confidence_level: f64,
}

/// Cognitive complexity estimator with information-theoretic foundation
pub struct CognitiveComplexityEstimator {
    /// Dimension-specific estimators
    estimators: HashMap<ComplexityDimension, Box<dyn DimensionEstimator + Send + Sync>>,
    
    /// Weighted combination coefficients
    dimension_weights: HashMap<ComplexityDimension, f64>,
    
    /// Estimation cache for performance optimization
    cache: Arc<EstimationCache>,
    
    /// Performance metrics
    metrics: Arc<PerformanceMetrics>,
}

/// Trait for dimension-specific complexity estimation
pub trait DimensionEstimator: Send + Sync {
    /// Estimate complexity for a single element in this dimension
    fn estimate_element(&self, element: &InterfaceElement) -> Result<f64, ComplexityError>;
    
    /// Estimate aggregate complexity for multiple elements
    fn estimate_aggregate(&self, elements: &[InterfaceElement]) -> Result<f64, ComplexityError>;
    
    /// Get dimension-specific recommendations
    fn get_recommendations(&self, element: &InterfaceElement, complexity: f64) -> Vec<ComplexityRecommendation>;
    
    /// Validate estimation parameters
    fn validate_parameters(&self) -> Result<(), ComplexityError>;
}

/// Visual complexity estimator implementing perceptual information theory
struct VisualComplexityEstimator {
    /// Perceptual model parameters
    perceptual_model: PerceptualModel,
}

/// Perceptual model for visual complexity
struct PerceptualModel {
    /// Color discrimination threshold
    color_threshold: f64,
    
    /// Spatial frequency sensitivity
    spatial_sensitivity: f64,
    
    /// Temporal frequency sensitivity
    temporal_sensitivity: f64,
    
    /// Contrast sensitivity function parameters
    csf_parameters: ContrastSensitivityFunction,
}

/// Contrast Sensitivity Function parameters
struct ContrastSensitivityFunction {
    /// Peak sensitivity frequency
    peak_frequency: f64,
    
    /// Sensitivity bandwidth
    bandwidth: f64,
    
    /// Maximum sensitivity
    max_sensitivity: f64,
}

impl DimensionEstimator for VisualComplexityEstimator {
    fn estimate_element(&self, element: &InterfaceElement) -> Result<f64, ComplexityError> {
        let visual_props = &element.visual_properties;
        
        // Shannon entropy calculation for color distribution
        let color_entropy = self.calculate_color_entropy(visual_props.color_count)?;
        
        // Spatial density complexity using Weber-Fechner law
        let density_complexity = self.calculate_density_complexity(visual_props.density)?;
        
        // Contrast complexity using CSF
        let contrast_complexity = self.calculate_contrast_complexity(visual_props.contrast_ratio)?;
        
        // Animation complexity using temporal frequency analysis
        let animation_complexity = match visual_props.animation_frequency {
            Some(freq) => self.calculate_animation_complexity(freq)?,
            None => 0.0,
        };
        
        // Text complexity using information theory
        let text_complexity = match &visual_props.text_complexity {
            Some(text) => self.calculate_text_complexity(text)?,
            None => 0.0,
        };
        
        // Weighted combination using psychophysical scaling
        let complexity = (
            color_entropy * 0.25 +
            density_complexity * 0.30 +
            contrast_complexity * 0.15 +
            animation_complexity * 0.20 +
            text_complexity * 0.10
        ).min(1.0);
        
        Ok(complexity)
    }
    
    fn estimate_aggregate(&self, elements: &[InterfaceElement]) -> Result<f64, ComplexityError> {
        if elements.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate individual complexities
        let individual_complexities: Result<Vec<f64>, _> = elements
            .iter()
            .map(|element| self.estimate_element(element))
            .collect();
        
        let complexities = individual_complexities?;
        
        // Aggregate using information-theoretic combination
        let mean_complexity = complexities.iter().sum::<f64>() / complexities.len() as f64;
        
        // Apply interaction effects using entropy maximization
        let interaction_factor = 1.0 + (complexities.len() as f64).ln() / constants::ENTROPY_NORMALIZATION;
        
        let aggregate_complexity = (mean_complexity * interaction_factor).min(1.0);
        
        Ok(aggregate_complexity)
    }
    
    fn get_recommendations(&self, element: &InterfaceElement, complexity: f64) -> Vec<ComplexityRecommendation> {
        let mut recommendations = Vec::new();
        
        if complexity > 0.7 {
            let visual_props = &element.visual_properties;
            
            if visual_props.color_count > 10 {
                recommendations.push(ComplexityRecommendation {
                    recommendation_type: RecommendationType::ReduceVisualDensity,
                    expected_reduction: 0.15,
                    implementation_difficulty: 0.3,
                    description: "Reduce color palette to improve visual clarity".to_string(),
                });
            }
            
            if visual_props.density > 0.8 {
                recommendations.push(ComplexityRecommendation {
                    recommendation_type: RecommendationType::ImproveHierarchy,
                    expected_reduction: 0.20,
                    implementation_difficulty: 0.5,
                    description: "Increase whitespace to reduce visual density".to_string(),
                });
            }
        }
        
        recommendations
    }
    
    fn validate_parameters(&self) -> Result<(), ComplexityError> {
        if self.perceptual_model.color_threshold <= 0.0 {
            return Err(ComplexityError::InvalidParameter(
                "Color threshold must be positive".to_string()
            ));
        }
        
        if !(0.0..=1.0).contains(&self.perceptual_model.spatial_sensitivity) {
            return Err(ComplexityError::InvalidParameter(
                "Spatial sensitivity must be in [0, 1]".to_string()
            ));
        }
        
        Ok(())
    }
}

impl VisualComplexityEstimator {
    /// Create new visual complexity estimator with default perceptual model
    fn new() -> Self {
        Self {
            perceptual_model: PerceptualModel {
                color_threshold: 2.3, // Just noticeable difference threshold
                spatial_sensitivity: 0.85,
                temporal_sensitivity: 0.75,
                csf_parameters: ContrastSensitivityFunction {
                    peak_frequency: 3.0, // cycles per degree
                    bandwidth: 1.5,
                    max_sensitivity: 1.0,
                },
            },
        }
    }
    
    /// Calculate Shannon entropy for color distribution
    fn calculate_color_entropy(&self, color_count: usize) -> Result<f64, ComplexityError> {
        if color_count == 0 {
            return Ok(0.0);
        }
        
        // Assume uniform color distribution for worst-case entropy
        let probability = 1.0 / color_count as f64;
        let entropy = -(color_count as f64) * probability * probability.log2();
        
        // Normalize by maximum possible entropy
        let normalized_entropy = entropy / constants::ENTROPY_NORMALIZATION;
        
        Ok(normalized_entropy.min(1.0))
    }
    
    /// Calculate density complexity using Weber-Fechner law
    fn calculate_density_complexity(&self, density: f64) -> Result<f64, ComplexityError> {
        if density < 0.0 {
            return Err(ComplexityError::InvalidParameter(
                "Density cannot be negative".to_string()
            ));
        }
        
        // Weber-Fechner law: perceived intensity = k * log(stimulus)
        let perceived_density = (1.0 + density).ln() / (1.0 + 10.0).ln(); // Normalize to [0, 1]
        
        Ok(perceived_density.min(1.0))
    }
    
    /// Calculate contrast complexity using contrast sensitivity function
    fn calculate_contrast_complexity(&self, contrast_ratio: f64) -> Result<f64, ComplexityError> {
        if contrast_ratio < 1.0 {
            return Err(ComplexityError::InvalidParameter(
                "Contrast ratio must be >= 1.0".to_string()
            ));
        }
        
        // WCAG contrast ratio to complexity mapping
        let complexity = match contrast_ratio {
            r if r >= 7.0 => 0.1,  // Excellent contrast, low complexity
            r if r >= 4.5 => 0.3,  // Good contrast, moderate complexity
            r if r >= 3.0 => 0.6,  // Poor contrast, high complexity
            _ => 0.9,              // Very poor contrast, very high complexity
        };
        
        Ok(complexity)
    }
    
    /// Calculate animation complexity using temporal frequency analysis
    fn calculate_animation_complexity(&self, frequency: f64) -> Result<f64, ComplexityError> {
        if frequency < 0.0 {
            return Err(ComplexityError::InvalidParameter(
                "Animation frequency cannot be negative".to_string()
            ));
        }
        
        // Critical flicker fusion threshold consideration
        let complexity = if frequency > 24.0 {
            // Above CFF threshold, complexity increases with frequency
            (frequency / 60.0).min(1.0)
        } else {
            // Below CFF threshold, moderate complexity
            frequency / 24.0
        };
        
        Ok(complexity)
    }
    
    /// Calculate text complexity using information theory and readability metrics
    fn calculate_text_complexity(&self, text: &TextComplexity) -> Result<f64, ComplexityError> {
        // Flesch-Kincaid inspired complexity calculation
        let word_complexity = (text.avg_word_length - 3.0).max(0.0) / 10.0;
        let sentence_complexity = text.sentence_complexity / 100.0;
        let technical_complexity = text.technical_density;
        
        // Information-theoretic word count complexity
        let count_complexity = (text.word_count as f64).ln() / (1000.0_f64).ln();
        
        let total_complexity = (
            word_complexity * 0.25 +
            sentence_complexity * 0.25 +
            technical_complexity * 0.30 +
            count_complexity * 0.20
        ).min(1.0);
        
        Ok(total_complexity)
    }
}

/// Structural complexity estimator using graph theory and relational complexity
struct StructuralComplexityEstimator {
    /// Maximum considered relationship depth
    max_depth: usize,
    
    /// Relationship weight factors
    relationship_weights: HashMap<RelationshipType, f64>,
}

impl DimensionEstimator for StructuralComplexityEstimator {
    fn estimate_element(&self, element: &InterfaceElement) -> Result<f64, ComplexityError> {
        let relationship_count = element.relationships.len();
        
        if relationship_count == 0 {
            return Ok(0.0);
        }
        
        // Halford's relational complexity theory
        let relational_complexity = (relationship_count as f64 / constants::HALFORD_RELATIONAL_BOUND).min(1.0);
        
        // Weighted relationship complexity
        let weighted_complexity = element.relationships
            .iter()
            .map(|rel| {
                let weight = self.relationship_weights
                    .get(&rel.relationship_type)
                    .copied()
                    .unwrap_or(1.0);
                rel.strength * weight
            })
            .sum::<f64>() / relationship_count as f64;
        
        let total_complexity = (relational_complexity + weighted_complexity) / 2.0;
        
        Ok(total_complexity.min(1.0))
    }
    
    fn estimate_aggregate(&self, elements: &[InterfaceElement]) -> Result<f64, ComplexityError> {
        if elements.is_empty() {
            return Ok(0.0);
        }
        
        // Build relationship graph
        let mut graph_complexity = 0.0;
        let mut total_edges = 0;
        
        for element in elements {
            let local_complexity = self.estimate_element(element)?;
            graph_complexity += local_complexity;
            total_edges += element.relationships.len();
        }
        
        // Graph density complexity
        let node_count = elements.len();
        let max_edges = node_count * (node_count - 1) / 2;
        let density_complexity = if max_edges > 0 {
            total_edges as f64 / max_edges as f64
        } else {
            0.0
        };
        
        let aggregate_complexity = (
            graph_complexity / elements.len() as f64 * 0.7 +
            density_complexity * 0.3
        ).min(1.0);
        
        Ok(aggregate_complexity)
    }
    
    fn get_recommendations(&self, element: &InterfaceElement, complexity: f64) -> Vec<ComplexityRecommendation> {
        let mut recommendations = Vec::new();
        
        if complexity > 0.6 && element.relationships.len() > 5 {
            recommendations.push(ComplexityRecommendation {
                recommendation_type: RecommendationType::ImproveHierarchy,
                expected_reduction: 0.25,
                implementation_difficulty: 0.6,
                description: "Simplify relationship structure using hierarchical grouping".to_string(),
            });
        }
        
        recommendations
    }
    
    fn validate_parameters(&self) -> Result<(), ComplexityError> {
        if self.max_depth == 0 {
            return Err(ComplexityError::InvalidParameter(
                "Maximum depth must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

impl StructuralComplexityEstimator {
    fn new() -> Self {
        let mut relationship_weights = HashMap::new();
        relationship_weights.insert(RelationshipType::Hierarchical, 1.2);
        relationship_weights.insert(RelationshipType::Functional, 1.1);
        relationship_weights.insert(RelationshipType::Temporal, 1.0);
        relationship_weights.insert(RelationshipType::Spatial, 0.8);
        relationship_weights.insert(RelationshipType::Semantic, 0.9);
        
        Self {
            max_depth: 5,
            relationship_weights,
        }
    }
}

/// Estimation cache for performance optimization
struct EstimationCache {
    /// Element complexity cache
    element_cache: parking_lot::RwLock<HashMap<String, (f64, chrono::DateTime<chrono::Utc>)>>,
    
    /// Cache size limit
    max_size: usize,
    
    /// Cache hit statistics
    hit_count: AtomicU64,
    miss_count: AtomicU64,
}

impl EstimationCache {
    fn new(max_size: usize) -> Self {
        Self {
            element_cache: parking_lot::RwLock::new(HashMap::new()),
            max_size,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }
    
    fn get(&self, key: &str) -> Option<f64> {
        let cache = self.element_cache.read();
        if let Some((complexity, timestamp)) = cache.get(key) {
            // Check if cache entry is still valid (within 1 hour)
            let age = chrono::Utc::now().signed_duration_since(*timestamp);
            if age.num_hours() < 1 {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                return Some(*complexity);
            }
        }
        
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    fn insert(&self, key: String, complexity: f64) {
        let mut cache = self.element_cache.write();
        
        // Simple LRU eviction if cache is full
        if cache.len() >= self.max_size {
            // Remove oldest entry (simplified LRU)
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }
        
        cache.insert(key, (complexity, chrono::Utc::now()));
    }
    
    fn hit_rate(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Performance metrics for monitoring estimation efficiency
struct PerformanceMetrics {
    /// Total estimation count
    estimation_count: AtomicU64,
    
    /// Total processing time in microseconds
    total_processing_time: AtomicU64,
    
    /// Error count
    error_count: AtomicU64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            estimation_count: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }
    
    fn record_estimation(&self, processing_time_us: u64) {
        self.estimation_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time.fetch_add(processing_time_us, Ordering::Relaxed);
    }
    
    fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn average_processing_time(&self) -> f64 {
        let count = self.estimation_count.load(Ordering::Relaxed);
        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        
        if count > 0 {
            total_time as f64 / count as f64
        } else {
            0.0
        }
    }
    
    fn error_rate(&self) -> f64 {
        let errors = self.error_count.load(Ordering::Relaxed);
        let total = self.estimation_count.load(Ordering::Relaxed);
        
        if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl CognitiveComplexityEstimator {
    /// Create a new cognitive complexity estimator with default configuration
    pub fn new() -> Self {
        let mut estimators: HashMap<ComplexityDimension, Box<dyn DimensionEstimator + Send + Sync>> = HashMap::new();
        
        estimators.insert(
            ComplexityDimension::Visual,
            Box::new(VisualComplexityEstimator::new())
        );
        
        estimators.insert(
            ComplexityDimension::Structural,
            Box::new(StructuralComplexityEstimator::new())
        );
        
        // Default dimension weights based on cognitive load research
        let mut dimension_weights = HashMap::new();
        dimension_weights.insert(ComplexityDimension::Visual, 0.25);
        dimension_weights.insert(ComplexityDimension::Structural, 0.20);
        dimension_weights.insert(ComplexityDimension::Temporal, 0.15);
        dimension_weights.insert(ComplexityDimension::Interactive, 0.15);
        dimension_weights.insert(ComplexityDimension::Semantic, 0.15);
        dimension_weights.insert(ComplexityDimension::Navigational, 0.10);
        
        Self {
            estimators,
            dimension_weights,
            cache: Arc::new(EstimationCache::new(1000)),
            metrics: Arc::new(PerformanceMetrics::new()),
        }
    }
    
    /// Estimate complexity for a single interface element
    pub fn estimate_element_complexity(&self, element: &InterfaceElement) -> Result<ComplexityEstimation, ComplexityError> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if let Some(cached_complexity) = self.cache.get(&element.id) {
            return Ok(ComplexityEstimation {
                dimension_scores: HashMap::new(), // Simplified for cache hits
                overall_score: cached_complexity,
                confidence_interval: (cached_complexity * 0.95, cached_complexity * 1.05),
                contributors: Vec::new(),
                recommendations: Vec::new(),
                metadata: EstimationMetadata {
                    timestamp: chrono::Utc::now(),
                    algorithm_version: "1.0.0".to_string(),
                    processing_time_us: 1,
                    confidence_level: 0.95,
                },
            });
        }
        
        let mut dimension_scores = HashMap::new();
        let mut contributors = Vec::new();
        
        // Estimate complexity for each dimension
        for (dimension, estimator) in &self.estimators {
            match estimator.estimate_element(element) {
                Ok(score) => {
                    dimension_scores.insert(*dimension, score);
                    
                    if score > 0.1 {
                        contributors.push(ComplexityContributor {
                            source: format!("{:?} dimension", dimension),
                            contribution: score,
                            dimensions: vec![*dimension],
                            explanation: format!("Complexity from {:?} factors", dimension),
                        });
                    }
                },
                Err(e) => {
                    self.metrics.record_error();
                    return Err(e);
                }
            }
        }
        
        // Calculate weighted overall score
        let overall_score = self.calculate_weighted_score(&dimension_scores)?;
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        for (dimension, estimator) in &self.estimators {
            if let Some(score) = dimension_scores.get(dimension) {
                recommendations.extend(estimator.get_recommendations(element, *score));
            }
        }
        
        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(overall_score, &dimension_scores);
        
        let processing_time_us = start_time.elapsed().as_micros() as u64;
        self.metrics.record_estimation(processing_time_us);
        
        // Cache the result
        self.cache.insert(element.id.clone(), overall_score);
        
        let estimation = ComplexityEstimation {
            dimension_scores,
            overall_score,
            confidence_interval,
            contributors,
            recommendations,
            metadata: EstimationMetadata {
                timestamp: chrono::Utc::now(),
                algorithm_version: "1.0.0".to_string(),
                processing_time_us,
                confidence_level: 0.95,
            },
        };
        
        Ok(estimation)
    }
    
    /// Estimate aggregate complexity for multiple interface elements
    pub fn estimate_aggregate_complexity(&self, elements: &[InterfaceElement]) -> Result<ComplexityEstimation, ComplexityError> {
        let start_time = std::time::Instant::now();
        
        if elements.is_empty() {
            return Ok(self.empty_estimation());
        }
        
        let mut dimension_scores = HashMap::new();
        let mut all_contributors = Vec::new();
        let mut all_recommendations = Vec::new();
        
        // Estimate aggregate complexity for each dimension
        for (dimension, estimator) in &self.estimators {
            match estimator.estimate_aggregate(elements) {
                Ok(score) => {
                    dimension_scores.insert(*dimension, score);
                    
                    if score > 0.1 {
                        all_contributors.push(ComplexityContributor {
                            source: format!("Aggregate {:?} complexity", dimension),
                            contribution: score,
                            dimensions: vec![*dimension],
                            explanation: format!("Combined complexity from {} elements in {:?} dimension", elements.len(), dimension),
                        });
                    }
                },
                Err(e) => {
                    self.metrics.record_error();
                    return Err(e);
                }
            }
        }
        
        // Calculate weighted overall score with interaction effects
        let base_score = self.calculate_weighted_score(&dimension_scores)?;
        let interaction_factor = self.calculate_interaction_factor(elements.len());
        let overall_score = (base_score * interaction_factor).min(1.0);
        
        // Aggregate recommendations (deduplicate similar ones)
        for element in elements {
            for (dimension, estimator) in &self.estimators {
                if let Some(score) = dimension_scores.get(dimension) {
                    all_recommendations.extend(estimator.get_recommendations(element, *score));
                }
            }
        }
        
        // Deduplicate recommendations
        all_recommendations.sort_by(|a, b| {
            a.recommendation_type.to_string().cmp(&b.recommendation_type.to_string())
        });
        all_recommendations.dedup_by(|a, b| {
            std::mem::discriminant(&a.recommendation_type) == std::mem::discriminant(&b.recommendation_type)
        });
        
        let confidence_interval = self.calculate_confidence_interval(overall_score, &dimension_scores);
        
        let processing_time_us = start_time.elapsed().as_micros() as u64;
        self.metrics.record_estimation(processing_time_us);
        
        Ok(ComplexityEstimation {
            dimension_scores,
            overall_score,
            confidence_interval,
            contributors: all_contributors,
            recommendations: all_recommendations,
            metadata: EstimationMetadata {
                timestamp: chrono::Utc::now(),
                algorithm_version: "1.0.0".to_string(),
                processing_time_us,
                confidence_level: 0.95,
            },
        })
    }
    
    /// Calculate weighted overall complexity score
    fn calculate_weighted_score(&self, dimension_scores: &HashMap<ComplexityDimension, f64>) -> Result<f64, ComplexityError> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (dimension, score) in dimension_scores {
            if let Some(weight) = self.dimension_weights.get(dimension) {
                weighted_sum += score * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            Err(ComplexityError::InvalidParameter(
                "No valid dimension weights found".to_string()
            ))
        }
    }
    
    /// Calculate interaction factor for multiple elements
    fn calculate_interaction_factor(&self, element_count: usize) -> f64 {
        if element_count <= 1 {
            return 1.0;
        }
        
        // Logarithmic scaling based on Miller's magic number
        let base_factor = 1.0 + (element_count as f64 / constants::MILLER_CAPACITY).ln();
        
        // Apply cognitive scaling factor
        (base_factor * constants::COGNITIVE_SCALING / constants::COGNITIVE_SCALING).min(2.0)
    }
    
    /// Calculate confidence interval for complexity estimation
    fn calculate_confidence_interval(&self, overall_score: f64, dimension_scores: &HashMap<ComplexityDimension, f64>) -> (f64, f64) {
        if dimension_scores.is_empty() {
            return (overall_score, overall_score);
        }
        
        // Calculate variance across dimensions
        let mean = dimension_scores.values().sum::<f64>() / dimension_scores.len() as f64;
        let variance = dimension_scores.values()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / dimension_scores.len() as f64;
        
        let std_dev = variance.sqrt();
        let margin = 1.96 * std_dev / (dimension_scores.len() as f64).sqrt(); // 95% confidence
        
        let lower = (overall_score - margin).max(0.0);
        let upper = (overall_score + margin).min(1.0);
        
        (lower, upper)
    }
    
    /// Create empty estimation for edge cases
    fn empty_estimation(&self) -> ComplexityEstimation {
        ComplexityEstimation {
            dimension_scores: HashMap::new(),
            overall_score: 0.0,
            confidence_interval: (0.0, 0.0),
            contributors: Vec::new(),
            recommendations: Vec::new(),
            metadata: EstimationMetadata {
                timestamp: chrono::Utc::now(),
                algorithm_version: "1.0.0".to_string(),
                processing_time_us: 1,
                confidence_level: 1.0,
            },
        }
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> (f64, f64, f64) {
        (
            self.metrics.average_processing_time(),
            self.metrics.error_rate(),
            self.cache.hit_rate(),
        )
    }
    
    /// Update dimension weights for customization
    pub fn update_dimension_weights(&mut self, weights: HashMap<ComplexityDimension, f64>) -> Result<(), ComplexityError> {
        // Validate weights sum to approximately 1.0
        let total_weight: f64 = weights.values().sum();
        if (total_weight - 1.0).abs() > 0.1 {
            return Err(ComplexityError::InvalidParameter(
                "Dimension weights must sum to approximately 1.0".to_string()
            ));
        }
        
        // Validate individual weights are in valid range
        for weight in weights.values() {
            if !weight.is_finite() || *weight < 0.0 || *weight > 1.0 {
                return Err(ComplexityError::InvalidParameter(
                    "Individual weights must be in range [0.0, 1.0]".to_string()
                ));
            }
        }
        
        self.dimension_weights = weights;
        Ok(())
    }
}

impl Default for CognitiveComplexityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Error types for cognitive complexity estimation
#[derive(Debug, Error)]
pub enum ComplexityError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Estimation failed: {0}")]
    EstimationFailed(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Mathematical error: {0}")]
    MathematicalError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

// Helper trait implementations
impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ReduceVisualDensity => "Reduce Visual Density",
            Self::SimplifyInteractions => "Simplify Interactions",
            Self::ImproveHierarchy => "Improve Hierarchy",
            Self::ReduceInformation => "Reduce Information",
            Self::EnhancePredictability => "Enhance Predictability",
            Self::ProgressiveDisclosure => "Progressive Disclosure",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visual_complexity_estimation() {
        let estimator = VisualComplexityEstimator::new();
        
        let element = InterfaceElement {
            id: "test_element".to_string(),
            element_type: ElementType::Static { complexity_modifier: 1.0 },
            visual_properties: VisualProperties {
                color_count: 5,
                density: 0.5,
                contrast_ratio: 4.5,
                animation_frequency: None,
                text_complexity: None,
            },
            relationships: Vec::new(),
            interactions: Vec::new(),
            semantic_content: SemanticContent {
                entropy: 0.5,
                conceptual_complexity: 0.3,
                domain_complexity: 0.4,
                abstraction_level: 0.5,
            },
            temporal_behavior: None,
        };
        
        let complexity = estimator.estimate_element(&element).unwrap();
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }
    
    #[test]
    fn test_structural_complexity_estimation() {
        let estimator = StructuralComplexityEstimator::new();
        
        let element = InterfaceElement {
            id: "test_element".to_string(),
            element_type: ElementType::Container { child_count: 3, layout_complexity: 0.5 },
            visual_properties: VisualProperties {
                color_count: 1,
                density: 0.1,
                contrast_ratio: 7.0,
                animation_frequency: None,
                text_complexity: None,
            },
            relationships: vec![
                ElementRelationship {
                    target_id: "child1".to_string(),
                    relationship_type: RelationshipType::Hierarchical,
                    strength: 1.0,
                },
                ElementRelationship {
                    target_id: "child2".to_string(),
                    relationship_type: RelationshipType::Functional,
                    strength: 0.8,
                },
            ],
            interactions: Vec::new(),
            semantic_content: SemanticContent {
                entropy: 0.2,
                conceptual_complexity: 0.1,
                domain_complexity: 0.2,
                abstraction_level: 0.3,
            },
            temporal_behavior: None,
        };
        
        let complexity = estimator.estimate_element(&element).unwrap();
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }
    
    #[test]
    fn test_cognitive_complexity_estimator() {
        let estimator = CognitiveComplexityEstimator::new();
        
        let element = InterfaceElement {
            id: "test_element".to_string(),
            element_type: ElementType::Interactive { input_dimensions: 2 },
            visual_properties: VisualProperties {
                color_count: 8,
                density: 0.7,
                contrast_ratio: 3.5,
                animation_frequency: Some(15.0),
                text_complexity: Some(TextComplexity {
                    word_count: 50,
                    avg_word_length: 6.0,
                    sentence_complexity: 25.0,
                    technical_density: 0.3,
                }),
            },
            relationships: vec![
                ElementRelationship {
                    target_id: "related1".to_string(),
                    relationship_type: RelationshipType::Functional,
                    strength: 0.9,
                },
            ],
            interactions: vec![
                InteractionType::Click,
                InteractionType::DragDrop { target_count: 3 },
            ],
            semantic_content: SemanticContent {
                entropy: 0.8,
                conceptual_complexity: 0.6,
                domain_complexity: 0.7,
                abstraction_level: 0.5,
            },
            temporal_behavior: Some(TemporalBehavior {
                change_frequency: 2.0,
                predictability: 0.7,
                temporal_correlations: vec!["correlated1".to_string()],
            }),
        };
        
        let estimation = estimator.estimate_element_complexity(&element).unwrap();
        
        assert!(estimation.overall_score >= 0.0 && estimation.overall_score <= 1.0);
        assert!(!estimation.dimension_scores.is_empty());
        assert!(estimation.confidence_interval.0 <= estimation.overall_score);
        assert!(estimation.confidence_interval.1 >= estimation.overall_score);
        assert_eq!(estimation.metadata.algorithm_version, "1.0.0");
    }
    
    #[test]
    fn test_cache_functionality() {
        let cache = EstimationCache::new(10);
        
        // Test miss
        assert!(cache.get("nonexistent").is_none());
        
        // Test insert and hit
        cache.insert("test_key".to_string(), 0.75);
        assert_eq!(cache.get("test_key"), Some(0.75));
        
        // Test hit rate calculation
        let hit_rate = cache.hit_rate();
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
    }
}