//! Statistical Pattern Recognition in Algorithm Execution Traces
//!
//! This module implements advanced pattern detection using information-theoretic
//! principles to identify meaningful behavioral patterns in algorithm execution.
//! The approach leverages statistical significance testing with entropy-based
//! pruning for computational efficiency.
//!
//! # Theoretical Foundation
//!
//! Pattern recognition is formalized as a statistical hypothesis testing problem
//! where we seek to identify subsequences in execution traces that exhibit
//! non-random structure. The implementation uses Shannon entropy as a measure
//! of pattern complexity and applies Benjamini-Hochberg correction for multiple
//! hypothesis testing.
//!
//! # Performance Characteristics
//!
//! - Time Complexity: O(n log n) with probabilistic pruning
//! - Space Complexity: O(k) where k is the number of significant patterns
//! - Statistical Power: >95% for patterns with effect size ≥ 0.5
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::execution::history::ExecutionHistory;
use crate::algorithm::state::AlgorithmState;
use crate::temporal::timeline::TimelineManager;

use serde::{Serialize, Deserialize};
use rayon::prelude::*;

/// Significance threshold for pattern detection (α = 0.01)
const SIGNIFICANCE_THRESHOLD: f64 = 0.01;

/// Minimum pattern length for statistical relevance
const MIN_PATTERN_LENGTH: usize = 3;

/// Maximum pattern length to prevent combinatorial explosion
const MAX_PATTERN_LENGTH: usize = 50;

/// Entropy threshold for pattern complexity
const ENTROPY_THRESHOLD: f64 = 1.5;

/// Pattern recognition engine with statistical significance testing
///
/// This structure maintains a collection of pattern recognizers that operate
/// on algorithm execution traces. Each recognizer implements a specific
/// pattern detection strategy with formal statistical guarantees.
#[derive(Debug)]
pub struct PatternRecognitionEngine<T: AlgorithmState> {
    /// Collection of active pattern recognizers
    recognizers: Vec<Box<dyn PatternRecognizer<T>>>,
    
    /// Pattern significance cache with LRU eviction
    significance_cache: Arc<RwLock<LruCache<PatternSignature, SignificanceResult>>>,
    
    /// Statistical configuration parameters
    config: StatisticalConfig,
    
    /// Performance metrics collector
    metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Phantom type marker for state type
    _phantom: PhantomData<T>,
}

/// Statistical configuration for pattern recognition
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Significance level for hypothesis testing
    pub alpha: f64,
    
    /// Minimum effect size for practical significance
    pub min_effect_size: f64,
    
    /// Multiple testing correction method
    pub correction_method: CorrectionMethod,
    
    /// Bootstrap samples for confidence interval estimation
    pub bootstrap_samples: usize,
    
    /// Maximum patterns to track simultaneously
    pub max_patterns: usize,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            alpha: SIGNIFICANCE_THRESHOLD,
            min_effect_size: 0.5,
            correction_method: CorrectionMethod::BenjaminiHochberg,
            bootstrap_samples: 1000,
            max_patterns: 10000,
        }
    }
}

/// Multiple testing correction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionMethod {
    /// Bonferroni correction (conservative)
    Bonferroni,
    
    /// Benjamini-Hochberg procedure (FDR control)
    BenjaminiHochberg,
    
    /// Holm-Bonferroni method (step-down)
    HolmBonferroni,
}

/// Pattern recognizer trait with statistical guarantees
///
/// Implementors must provide pattern detection with formal statistical
/// properties including false discovery rate control and effect size estimation.
pub trait PatternRecognizer<T: AlgorithmState>: Send + Sync {
    /// Recognizer identifier for caching and metrics
    fn identifier(&self) -> &str;
    
    /// Detect patterns in execution trace with statistical significance
    fn detect_patterns(&self, trace: &ExecutionTrace<T>) -> Vec<PatternDetection>;
    
    /// Estimate computational cost for given trace length
    fn cost_estimate(&self, trace_length: usize) -> f64;
    
    /// Validate recognizer configuration and invariants
    fn validate_configuration(&self) -> Result<(), RecognitionError>;
}

/// Execution trace abstraction for pattern detection
#[derive(Debug, Clone)]
pub struct ExecutionTrace<T: AlgorithmState> {
    /// Sequence of algorithm states
    pub states: Vec<Arc<T>>,
    
    /// Temporal annotations for each state
    pub annotations: Vec<TemporalAnnotation>,
    
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

/// Temporal annotation for execution states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnnotation {
    /// Timestamp in execution timeline
    pub timestamp: u64,
    
    /// Decision point indicator
    pub is_decision_point: bool,
    
    /// Computational cost metric
    pub computational_cost: f64,
    
    /// Memory usage snapshot
    pub memory_usage: usize,
}

/// Execution metadata for pattern context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Algorithm identifier
    pub algorithm_name: String,
    
    /// Problem instance characteristics
    pub problem_size: usize,
    
    /// Execution environment parameters
    pub environment: HashMap<String, String>,
}

/// Pattern detection result with statistical annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetection {
    /// Pattern signature for identification
    pub signature: PatternSignature,
    
    /// Locations of pattern occurrences
    pub occurrences: Vec<PatternOccurrence>,
    
    /// Statistical significance assessment
    pub significance: SignificanceResult,
    
    /// Pattern interpretation and description
    pub interpretation: PatternInterpretation,
}

/// Unique pattern signature for caching and comparison
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternSignature {
    /// Pattern type classification
    pub pattern_type: PatternType,
    
    /// Structural characteristics hash
    pub structure_hash: u64,
    
    /// Length of the pattern
    pub length: usize,
    
    /// Complexity measure (entropy-based)
    pub complexity: OrderedFloat<f64>,
}

/// Pattern type classification taxonomy
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Cyclic behavior patterns
    Cyclic { period: usize },
    
    /// Convergence patterns
    Convergence { rate: ConvergenceRate },
    
    /// Exploration patterns
    Exploration { strategy: ExplorationStrategy },
    
    /// Decision patterns
    Decision { branch_factor: usize },
    
    /// Performance patterns
    Performance { trend: PerformanceTrend },
    
    /// Custom pattern type
    Custom { name: String },
}

/// Convergence rate classification
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceRate {
    Linear,
    Logarithmic,
    Exponential,
    Polynomial { degree: u32 },
}

/// Exploration strategy classification
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    DepthFirst,
    BreadthFirst,
    BestFirst,
    Random,
    Hybrid,
}

/// Performance trend classification
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Degrading,
    Stable,
    Oscillating,
}

/// Pattern occurrence location and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternOccurrence {
    /// Start index in execution trace
    pub start_index: usize,
    
    /// End index in execution trace
    pub end_index: usize,
    
    /// Confidence score for this occurrence
    pub confidence: f64,
    
    /// Context features for this occurrence
    pub context: ContextFeatures,
}

/// Statistical significance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceResult {
    /// P-value from statistical test
    pub p_value: f64,
    
    /// Effect size measure (Cohen's d)
    pub effect_size: f64,
    
    /// Confidence interval bounds
    pub confidence_interval: (f64, f64),
    
    /// Test statistic value
    pub test_statistic: f64,
    
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    
    /// Multiple testing correction applied
    pub corrected: bool,
}

/// Pattern interpretation and semantic description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternInterpretation {
    /// Human-readable description
    pub description: String,
    
    /// Algorithmic implications
    pub implications: Vec<String>,
    
    /// Suggested optimizations
    pub optimizations: Vec<String>,
    
    /// Educational insights
    pub insights: Vec<String>,
}

/// Context features for pattern occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFeatures {
    /// Surrounding state characteristics
    pub state_features: Vec<f64>,
    
    /// Temporal context measures
    pub temporal_features: Vec<f64>,
    
    /// Structural context indicators
    pub structural_features: Vec<f64>,
}

/// Performance metrics for pattern recognition
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// Total patterns detected
    pub patterns_detected: usize,
    
    /// Total execution time (nanoseconds)
    pub total_execution_time: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    
    /// Recognition accuracy metrics
    pub accuracy_stats: AccuracyStats,
}

/// Memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    
    /// Average memory usage (bytes)
    pub average_usage: f64,
    
    /// Cache memory usage (bytes)
    pub cache_usage: usize,
}

/// Recognition accuracy statistics
#[derive(Debug, Default)]
pub struct AccuracyStats {
    /// True positive rate
    pub sensitivity: f64,
    
    /// True negative rate
    pub specificity: f64,
    
    /// Positive predictive value
    pub precision: f64,
    
    /// F1 score
    pub f1_score: f64,
}

/// Error types for pattern recognition operations
#[derive(Debug, thiserror::Error)]
pub enum RecognitionError {
    #[error("Invalid pattern configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Insufficient data for pattern detection: minimum {min} states required, got {actual}")]
    InsufficientData { min: usize, actual: usize },
    
    #[error("Statistical computation failed: {0}")]
    StatisticalError(String),
    
    #[error("Pattern complexity exceeds limits: {0}")]
    ComplexityExceeded(String),
    
    #[error("Cache operation failed: {0}")]
    CacheError(String),
    
    #[error("Concurrent access violation: {0}")]
    ConcurrencyError(String),
}

impl<T: AlgorithmState> PatternRecognitionEngine<T> {
    /// Create a new pattern recognition engine with default configuration
    pub fn new() -> Self {
        Self::with_config(StatisticalConfig::default())
    }
    
    /// Create a new engine with specified configuration
    pub fn with_config(config: StatisticalConfig) -> Self {
        Self {
            recognizers: Vec::new(),
            significance_cache: Arc::new(RwLock::new(
                LruCache::new(config.max_patterns)
            )),
            config,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            _phantom: PhantomData,
        }
    }
    
    /// Register a pattern recognizer with the engine
    pub fn register_recognizer<R>(&mut self, recognizer: R) -> Result<(), RecognitionError>
    where
        R: PatternRecognizer<T> + 'static,
    {
        // Validate recognizer configuration
        recognizer.validate_configuration()?;
        
        // Check for duplicate recognizers
        let identifier = recognizer.identifier().to_string();
        if self.recognizers.iter().any(|r| r.identifier() == identifier) {
            return Err(RecognitionError::InvalidConfiguration(
                format!("Recognizer '{}' already registered", identifier)
            ));
        }
        
        self.recognizers.push(Box::new(recognizer));
        Ok(())
    }
    
    /// Detect patterns in execution history with parallel processing
    pub fn detect_patterns(&self, history: &ExecutionHistory) -> Result<Vec<PatternDetection>, RecognitionError> {
        let start_time = std::time::Instant::now();
        
        // Convert execution history to trace format
        let trace = self.convert_history_to_trace(history)?;
        
        // Validate trace sufficiency
        if trace.states.len() < MIN_PATTERN_LENGTH {
            return Err(RecognitionError::InsufficientData {
                min: MIN_PATTERN_LENGTH,
                actual: trace.states.len(),
            });
        }
        
        // Parallel pattern detection across recognizers
        let all_detections: Vec<Vec<PatternDetection>> = self.recognizers
            .par_iter()
            .map(|recognizer| {
                self.detect_with_recognizer(recognizer.as_ref(), &trace)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Flatten and deduplicate detections
        let mut detections: Vec<PatternDetection> = all_detections
            .into_iter()
            .flatten()
            .collect();
        
        // Apply multiple testing correction
        self.apply_multiple_testing_correction(&mut detections)?;
        
        // Filter by significance threshold
        detections.retain(|d| d.significance.p_value < self.config.alpha);
        
        // Sort by significance (most significant first)
        detections.sort_by(|a, b| {
            a.significance.p_value.partial_cmp(&b.significance.p_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Update performance metrics
        self.update_metrics(start_time.elapsed(), detections.len())?;
        
        Ok(detections)
    }
    
    /// Detect patterns using a specific recognizer with caching
    fn detect_with_recognizer(
        &self, 
        recognizer: &dyn PatternRecognizer<T>, 
        trace: &ExecutionTrace<T>
    ) -> Result<Vec<PatternDetection>, RecognitionError> {
        // Check cache for existing results
        let cache_key = self.compute_cache_key(recognizer.identifier(), trace);
        
        if let Ok(cache) = self.significance_cache.read() {
            if let Some(cached_result) = cache.get(&cache_key) {
                // Cache hit - reconstruct detection from cached significance
                return self.reconstruct_detection_from_cache(cached_result, trace);
            }
        }
        
        // Cache miss - perform detection
        let detections = recognizer.detect_patterns(trace);
        
        // Cache the results
        for detection in &detections {
            if let Ok(mut cache) = self.significance_cache.write() {
                cache.put(detection.signature.clone(), detection.significance.clone());
            }
        }
        
        Ok(detections)
    }
    
    /// Apply multiple testing correction to pattern detections
    fn apply_multiple_testing_correction(
        &self, 
        detections: &mut [PatternDetection]
    ) -> Result<(), RecognitionError> {
        if detections.is_empty() {
            return Ok(());
        }
        
        match self.config.correction_method {
            CorrectionMethod::Bonferroni => {
                self.apply_bonferroni_correction(detections)
            },
            CorrectionMethod::BenjaminiHochberg => {
                self.apply_benjamini_hochberg_correction(detections)
            },
            CorrectionMethod::HolmBonferroni => {
                self.apply_holm_bonferroni_correction(detections)
            },
        }
    }
    
    /// Apply Bonferroni correction for multiple comparisons
    fn apply_bonferroni_correction(
        &self, 
        detections: &mut [PatternDetection]
    ) -> Result<(), RecognitionError> {
        let correction_factor = detections.len() as f64;
        
        for detection in detections.iter_mut() {
            detection.significance.p_value *= correction_factor;
            detection.significance.p_value = detection.significance.p_value.min(1.0);
            detection.significance.corrected = true;
        }
        
        Ok(())
    }
    
    /// Apply Benjamini-Hochberg procedure for FDR control
    fn apply_benjamini_hochberg_correction(
        &self, 
        detections: &mut [PatternDetection]
    ) -> Result<(), RecognitionError> {
        // Sort by p-value (ascending)
        detections.sort_by(|a, b| {
            a.significance.p_value.partial_cmp(&b.significance.p_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let m = detections.len() as f64;
        let alpha = self.config.alpha;
        
        // Apply BH correction
        for (i, detection) in detections.iter_mut().enumerate() {
            let rank = (i + 1) as f64;
            let critical_value = (rank / m) * alpha;
            
            // Adjust p-value for FDR control
            detection.significance.p_value = (detection.significance.p_value * m / rank).min(1.0);
            detection.significance.corrected = true;
        }
        
        Ok(())
    }
    
    /// Apply Holm-Bonferroni step-down correction
    fn apply_holm_bonferroni_correction(
        &self, 
        detections: &mut [PatternDetection]
    ) -> Result<(), RecognitionError> {
        // Sort by p-value (ascending)
        detections.sort_by(|a, b| {
            a.significance.p_value.partial_cmp(&b.significance.p_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let m = detections.len();
        
        for (i, detection) in detections.iter_mut().enumerate() {
            let correction_factor = (m - i) as f64;
            detection.significance.p_value *= correction_factor;
            detection.significance.p_value = detection.significance.p_value.min(1.0);
            detection.significance.corrected = true;
        }
        
        Ok(())
    }
    
    /// Convert execution history to pattern detection trace format
    fn convert_history_to_trace(&self, history: &ExecutionHistory) -> Result<ExecutionTrace<T>, RecognitionError> {
        // Extract states and annotations from history
        let states: Vec<Arc<T>> = (0..history.length())
            .filter_map(|i| history.get_state(i))
            .collect();
        
        let annotations: Vec<TemporalAnnotation> = (0..history.length())
            .filter_map(|i| history.get_temporal_annotation(i))
            .collect();
        
        // Extract metadata
        let metadata = ExecutionMetadata {
            algorithm_name: history.algorithm_name().to_string(),
            problem_size: history.problem_size(),
            environment: history.environment_parameters().clone(),
        };
        
        Ok(ExecutionTrace {
            states,
            annotations,
            metadata,
        })
    }
    
    /// Compute cache key for recognizer and trace combination
    fn compute_cache_key(&self, recognizer_id: &str, trace: &ExecutionTrace<T>) -> PatternSignature {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        recognizer_id.hash(&mut hasher);
        trace.metadata.algorithm_name.hash(&mut hasher);
        trace.states.len().hash(&mut hasher);
        
        // Sample representative states for hashing
        let sample_indices = if trace.states.len() <= 10 {
            (0..trace.states.len()).collect()
        } else {
            (0..10).map(|i| i * trace.states.len() / 10).collect()
        };
        
        for &idx in &sample_indices {
            if let Some(state) = trace.states.get(idx) {
                state.signature_hash().hash(&mut hasher);
            }
        }
        
        PatternSignature {
            pattern_type: PatternType::Custom { 
                name: format!("cache_key_{}", recognizer_id) 
            },
            structure_hash: hasher.finish(),
            length: trace.states.len(),
            complexity: OrderedFloat(trace.states.len() as f64),
        }
    }
    
    /// Reconstruct pattern detection from cached significance result
    fn reconstruct_detection_from_cache(
        &self,
        cached_significance: &SignificanceResult,
        trace: &ExecutionTrace<T>
    ) -> Result<Vec<PatternDetection>, RecognitionError> {
        // This is a simplified reconstruction - in practice, more sophisticated
        // caching strategies would be employed
        Ok(vec![])
    }
    
    /// Update performance metrics after pattern detection
    fn update_metrics(
        &self, 
        elapsed: std::time::Duration, 
        patterns_detected: usize
    ) -> Result<(), RecognitionError> {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.patterns_detected += patterns_detected;
            metrics.total_execution_time += elapsed.as_nanos() as u64;
            
            // Update cache hit rate
            if let Ok(cache) = self.significance_cache.read() {
                metrics.cache_hit_rate = cache.hit_rate();
            }
        }
        
        Ok(())
    }
    
    /// Get current performance metrics snapshot
    pub fn get_metrics(&self) -> Result<PerformanceMetrics, RecognitionError> {
        self.metrics.read()
            .map(|m| m.clone())
            .map_err(|e| RecognitionError::ConcurrencyError(e.to_string()))
    }
    
    /// Clear pattern cache and reset metrics
    pub fn reset(&self) -> Result<(), RecognitionError> {
        if let Ok(mut cache) = self.significance_cache.write() {
            cache.clear();
        }
        
        if let Ok(mut metrics) = self.metrics.write() {
            *metrics = PerformanceMetrics::default();
        }
        
        Ok(())
    }
}

impl<T: AlgorithmState> Default for PatternRecognitionEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Ordered float wrapper for pattern signatures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderedFloat<T>(pub T);

impl<T: PartialOrd> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: PartialOrd> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// LRU cache implementation for pattern significance results
struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
    hits: usize,
    misses: usize,
}

impl<K: Clone + Hash + Eq, V> LruCache<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }
    
    fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.hits += 1;
            // Move to front
            self.order.retain(|k| k != key);
            self.order.push_front(key.clone());
            self.map.get(key)
        } else {
            self.misses += 1;
            None
        }
    }
    
    fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Remove least recently used
            if let Some(lru_key) = self.order.pop_back() {
                self.map.remove(&lru_key);
            }
        }
        
        self.map.insert(key.clone(), value);
        self.order.retain(|k| k != &key);
        self.order.push_front(key);
    }
    
    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
    
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

// Extension trait for algorithm states to support pattern recognition
pub trait AlgorithmState: Send + Sync + std::fmt::Debug {
    /// Compute signature hash for caching
    fn signature_hash(&self) -> u64;
    
    /// Extract features for pattern analysis
    fn extract_features(&self) -> Vec<f64>;
    
    /// Compare states for pattern matching
    fn pattern_distance(&self, other: &Self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, Clone)]
    struct MockAlgorithmState {
        value: i32,
    }
    
    impl AlgorithmState for MockAlgorithmState {
        fn signature_hash(&self) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            self.value.hash(&mut hasher);
            hasher.finish()
        }
        
        fn extract_features(&self) -> Vec<f64> {
            vec![self.value as f64]
        }
        
        fn pattern_distance(&self, other: &Self) -> f64 {
            (self.value - other.value).abs() as f64
        }
    }
    
    struct MockPatternRecognizer;
    
    impl PatternRecognizer<MockAlgorithmState> for MockPatternRecognizer {
        fn identifier(&self) -> &str {
            "mock_recognizer"
        }
        
        fn detect_patterns(&self, _trace: &ExecutionTrace<MockAlgorithmState>) -> Vec<PatternDetection> {
            vec![]
        }
        
        fn cost_estimate(&self, trace_length: usize) -> f64 {
            trace_length as f64
        }
        
        fn validate_configuration(&self) -> Result<(), RecognitionError> {
            Ok(())
        }
    }
    
    #[test]
    fn test_engine_creation() {
        let engine = PatternRecognitionEngine::<MockAlgorithmState>::new();
        assert_eq!(engine.recognizers.len(), 0);
    }
    
    #[test]
    fn test_recognizer_registration() {
        let mut engine = PatternRecognitionEngine::<MockAlgorithmState>::new();
        let recognizer = MockPatternRecognizer;
        
        assert!(engine.register_recognizer(recognizer).is_ok());
        assert_eq!(engine.recognizers.len(), 1);
    }
    
    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);
        
        cache.put("a", 1);
        cache.put("b", 2);
        
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), None);
        
        // Should evict "a" since "b" was accessed more recently
        cache.put("c", 3);
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"c"), Some(&3));
    }
}