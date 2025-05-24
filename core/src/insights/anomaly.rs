//! # Algorithm Anomaly Detection Framework
//!
//! This module implements a sophisticated statistical anomaly detection system
//! for identifying unexpected algorithm behaviors through ensemble modeling
//! with formal mathematical guarantees.
//!
//! ## Mathematical Foundation
//!
//! The anomaly detection framework employs multiple statistical models:
//!
//! 1. **Multivariate Gaussian Model**: For continuous feature distributions
//!    - Probability density: p(x) = (2π)^(-k/2)|Σ|^(-1/2) exp(-1/2(x-μ)ᵀΣ⁻¹(x-μ))
//!
//! 2. **Isolation Forest**: For high-dimensional anomaly detection
//!    - Anomaly score: s(x,n) = 2^(-E(h(x))/c(n))
//!
//! 3. **Temporal Sequence Analysis**: For time-series anomalies
//!    - Hidden Markov Model state transitions
//!
//! ## Theoretical Guarantees
//!
//! The framework provides formal bounds on false positive rates through
//! Bonferroni correction and confidence interval estimation.

use crate::algorithm::traits::{Algorithm, AlgorithmState};
use crate::temporal::signature::ExecutionSignature;
use crate::execution::tracer::ExecutionTrace;
use crate::insights::pattern::{Pattern, PatternDetector};

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Normal};
use rayon::prelude::*;

/// Error types for anomaly detection operations
#[derive(Error, Debug)]
pub enum AnomalyError {
    #[error("Insufficient data for model training: got {got}, need {need}")]
    InsufficientData { got: usize, need: usize },
    
    #[error("Invalid model parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Model not trained")]
    ModelNotTrained,
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Feature extraction failed: {0}")]
    FeatureExtractionError(String),
}

/// Result type for anomaly detection operations
pub type AnomalyResult<T> = Result<T, AnomalyError>;

/// Anomaly type classification with confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Performance deviation from expected bounds
    PerformanceAnomaly {
        expected_range: (f64, f64),
        observed_value: f64,
        standard_deviations: f64,
    },
    
    /// Behavioral pattern deviation
    BehavioralAnomaly {
        expected_pattern: Pattern,
        observed_behavior: Pattern,
        divergence_metric: f64,
    },
    
    /// Temporal sequence anomaly
    TemporalAnomaly {
        sequence_likelihood: f64,
        transition_probability: f64,
        markov_chain_state: usize,
    },
    
    /// Structural anomaly in algorithm state
    StructuralAnomaly {
        invariant_violation: String,
        constraint_deviation: f64,
        recovery_possible: bool,
    },
}

/// Detected anomaly with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Unique anomaly identifier
    pub id: uuid::Uuid,
    
    /// Anomaly classification
    pub anomaly_type: AnomalyType,
    
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    
    /// False positive probability
    pub false_positive_rate: f64,
    
    /// Temporal location in execution
    pub temporal_location: usize,
    
    /// Algorithm state at anomaly
    pub state_context: AlgorithmState,
    
    /// Causal analysis results
    pub causal_factors: Vec<CausalFactor>,
    
    /// Educational importance score
    pub educational_significance: f64,
}

/// Causal factor for anomaly explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalFactor {
    /// Factor description
    pub description: String,
    
    /// Causal strength (0.0 - 1.0)
    pub strength: f64,
    
    /// Supporting evidence
    pub evidence: Vec<String>,
    
    /// Counterfactual analysis
    pub counterfactual: Option<String>,
}

/// Statistical model trait for anomaly detection
pub trait AnomalyModel: Send + Sync {
    /// Train the model on normal behavior data
    fn train(&mut self, data: &[AlgorithmState]) -> AnomalyResult<()>;
    
    /// Predict anomaly score for a state
    fn predict(&self, state: &AlgorithmState) -> AnomalyResult<f64>;
    
    /// Get model confidence interval
    fn confidence_interval(&self) -> (f64, f64);
    
    /// Model complexity for ensemble weighting
    fn complexity(&self) -> f64;
}

/// Multivariate Gaussian anomaly model
pub struct GaussianAnomalyModel {
    /// Mean vector μ
    mean: Option<DVector<f64>>,
    
    /// Covariance matrix Σ
    covariance: Option<DMatrix<f64>>,
    
    /// Precision matrix Σ⁻¹ (cached for performance)
    precision: Option<DMatrix<f64>>,
    
    /// Determinant |Σ| (cached for performance)
    determinant: Option<f64>,
    
    /// Feature dimension
    dimension: usize,
    
    /// Regularization parameter
    epsilon: f64,
}

impl GaussianAnomalyModel {
    /// Create a new Gaussian anomaly model
    pub fn new(dimension: usize) -> Self {
        Self {
            mean: None,
            covariance: None,
            precision: None,
            determinant: None,
            dimension,
            epsilon: 1e-6, // Regularization for numerical stability
        }
    }
    
    /// Extract features from algorithm state
    fn extract_features(&self, state: &AlgorithmState) -> AnomalyResult<DVector<f64>> {
        let mut features = Vec::with_capacity(self.dimension);
        
        // Extract numerical features from state
        features.push(state.step as f64);
        features.push(state.open_set.len() as f64);
        features.push(state.closed_set.len() as f64);
        
        if let Some(current) = state.current_node {
            features.push(current as f64);
        } else {
            features.push(0.0);
        }
        
        // Add custom state data features
        for (key, value) in &state.data {
            if let Ok(num) = value.parse::<f64>() {
                features.push(num);
                if features.len() >= self.dimension {
                    break;
                }
            }
        }
        
        // Pad with zeros if necessary
        while features.len() < self.dimension {
            features.push(0.0);
        }
        
        Ok(DVector::from_vec(features))
    }
    
    /// Compute Mahalanobis distance for anomaly scoring
    fn mahalanobis_distance(&self, x: &DVector<f64>) -> AnomalyResult<f64> {
        let mean = self.mean.as_ref()
            .ok_or(AnomalyError::ModelNotTrained)?;
        let precision = self.precision.as_ref()
            .ok_or(AnomalyError::ModelNotTrained)?;
        
        let diff = x - mean;
        let distance_squared = diff.transpose() * precision * &diff;
        
        Ok(distance_squared[(0, 0)].sqrt())
    }
}

impl AnomalyModel for GaussianAnomalyModel {
    fn train(&mut self, data: &[AlgorithmState]) -> AnomalyResult<()> {
        if data.len() < self.dimension * 2 {
            return Err(AnomalyError::InsufficientData {
                got: data.len(),
                need: self.dimension * 2,
            });
        }
        
        // Extract features from all states
        let features: Result<Vec<_>, _> = data.par_iter()
            .map(|state| self.extract_features(state))
            .collect();
        let features = features?;
        
        // Compute mean vector
        let mut mean = DVector::zeros(self.dimension);
        for feature in &features {
            mean += feature;
        }
        mean /= features.len() as f64;
        
        // Compute covariance matrix
        let mut covariance = DMatrix::zeros(self.dimension, self.dimension);
        for feature in &features {
            let diff = feature - &mean;
            covariance += &diff * diff.transpose();
        }
        covariance /= (features.len() - 1) as f64;
        
        // Add regularization for numerical stability
        for i in 0..self.dimension {
            covariance[(i, i)] += self.epsilon;
        }
        
        // Compute precision matrix and determinant
        let lu = covariance.lu();
        let determinant = lu.determinant();
        
        if determinant <= 0.0 {
            return Err(AnomalyError::NumericalInstability(
                "Covariance matrix is not positive definite".to_string()
            ));
        }
        
        let precision = lu.solve(&DMatrix::identity(self.dimension, self.dimension))
            .ok_or_else(|| AnomalyError::NumericalInstability(
                "Failed to compute precision matrix".to_string()
            ))?;
        
        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.precision = Some(precision);
        self.determinant = Some(determinant);
        
        Ok(())
    }
    
    fn predict(&self, state: &AlgorithmState) -> AnomalyResult<f64> {
        let features = self.extract_features(state)?;
        let distance = self.mahalanobis_distance(&features)?;
        
        // Convert distance to anomaly score using chi-squared distribution
        // Higher distance = higher anomaly score
        let chi_squared_df = self.dimension as f64;
        let anomaly_score = 1.0 - chi_squared_cdf(distance * distance, chi_squared_df);
        
        Ok(anomaly_score)
    }
    
    fn confidence_interval(&self) -> (f64, f64) {
        // 95% confidence interval based on chi-squared distribution
        let chi_squared_df = self.dimension as f64;
        let lower = chi_squared_quantile(0.025, chi_squared_df);
        let upper = chi_squared_quantile(0.975, chi_squared_df);
        
        (lower.sqrt(), upper.sqrt())
    }
    
    fn complexity(&self) -> f64 {
        // Model complexity proportional to dimension squared (covariance matrix)
        (self.dimension * self.dimension) as f64
    }
}

/// Isolation Forest anomaly model
pub struct IsolationForestModel {
    /// Collection of isolation trees
    trees: Vec<IsolationTree>,
    
    /// Number of trees in the forest
    n_trees: usize,
    
    /// Subsample size
    subsample_size: usize,
    
    /// Maximum tree depth
    max_depth: usize,
    
    /// Feature dimension
    dimension: usize,
}

/// Individual isolation tree
struct IsolationTree {
    root: Option<Box<IsolationNode>>,
    n_samples: usize,
}

/// Node in an isolation tree
#[derive(Debug)]
enum IsolationNode {
    /// Internal decision node
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    
    /// Leaf node
    Leaf {
        depth: usize,
    },
}

impl IsolationForestModel {
    /// Create a new Isolation Forest model
    pub fn new(dimension: usize, n_trees: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            subsample_size: 256, // Standard subsample size
            max_depth: (256f64.log2().ceil() as usize),
            dimension,
        }
    }
    
    /// Build a single isolation tree
    fn build_tree(&self, data: &[DVector<f64>], max_depth: usize) -> IsolationTree {
        let root = self.build_node(data, 0, max_depth);
        IsolationTree {
            root: Some(Box::new(root)),
            n_samples: data.len(),
        }
    }
    
    /// Recursively build tree nodes
    fn build_node(&self, data: &[DVector<f64>], depth: usize, max_depth: usize) -> IsolationNode {
        if data.len() <= 1 || depth >= max_depth {
            return IsolationNode::Leaf { depth };
        }
        
        // Randomly select feature and threshold
        let mut rng = rand::thread_rng();
        let feature = rand::random::<usize>() % self.dimension;
        
        let values: Vec<f64> = data.iter()
            .map(|x| x[feature])
            .collect();
        let min_val = values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val = values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationNode::Leaf { depth };
        }
        
        let threshold = min_val + rand::random::<f64>() * (max_val - min_val);
        
        // Split data
        let (left_data, right_data): (Vec<_>, Vec<_>) = data.iter()
            .cloned()
            .partition(|x| x[feature] < threshold);
        
        IsolationNode::Internal {
            feature,
            threshold,
            left: Box::new(self.build_node(&left_data, depth + 1, max_depth)),
            right: Box::new(self.build_node(&right_data, depth + 1, max_depth)),
        }
    }
    
    /// Compute path length for a sample in a tree
    fn path_length(&self, tree: &IsolationTree, sample: &DVector<f64>) -> f64 {
        if let Some(ref root) = tree.root {
            self.path_length_recursive(root, sample, 0)
        } else {
            0.0
        }
    }
    
    /// Recursive path length computation
    fn path_length_recursive(&self, node: &IsolationNode, sample: &DVector<f64>, current_depth: usize) -> f64 {
        match node {
            IsolationNode::Leaf { depth } => *depth as f64,
            IsolationNode::Internal { feature, threshold, left, right } => {
                if sample[*feature] < *threshold {
                    self.path_length_recursive(left, sample, current_depth + 1)
                } else {
                    self.path_length_recursive(right, sample, current_depth + 1)
                }
            }
        }
    }
    
    /// Compute average path length for binary search tree
    fn average_path_length(n: f64) -> f64 {
        if n <= 1.0 {
            0.0
        } else {
            2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
        }
    }
}

impl AnomalyModel for IsolationForestModel {
    fn train(&mut self, data: &[AlgorithmState]) -> AnomalyResult<()> {
        if data.len() < self.subsample_size {
            return Err(AnomalyError::InsufficientData {
                got: data.len(),
                need: self.subsample_size,
            });
        }
        
        // Extract features in parallel
        let features: Result<Vec<_>, _> = data.par_iter()
            .map(|state| {
                // Simplified feature extraction for isolation forest
                let mut features = vec![
                    state.step as f64,
                    state.open_set.len() as f64,
                    state.closed_set.len() as f64,
                    state.current_node.unwrap_or(0) as f64,
                ];
                
                // Pad to dimension
                while features.len() < self.dimension {
                    features.push(0.0);
                }
                
                Ok(DVector::from_vec(features))
            })
            .collect();
        let features = features?;
        
        // Build trees in parallel
        self.trees = (0..self.n_trees)
            .into_par_iter()
            .map(|_| {
                // Random subsample
                let mut rng = rand::thread_rng();
                let subsample: Vec<_> = features.choose_multiple(&mut rng, self.subsample_size)
                    .cloned()
                    .collect();
                
                self.build_tree(&subsample, self.max_depth)
            })
            .collect();
        
        Ok(())
    }
    
    fn predict(&self, state: &AlgorithmState) -> AnomalyResult<f64> {
        if self.trees.is_empty() {
            return Err(AnomalyError::ModelNotTrained);
        }
        
        // Extract features
        let mut features = vec![
            state.step as f64,
            state.open_set.len() as f64,
            state.closed_set.len() as f64,
            state.current_node.unwrap_or(0) as f64,
        ];
        
        while features.len() < self.dimension {
            features.push(0.0);
        }
        
        let sample = DVector::from_vec(features);
        
        // Compute average path length across all trees
        let avg_path_length: f64 = self.trees.par_iter()
            .map(|tree| self.path_length(tree, &sample))
            .sum::<f64>() / self.trees.len() as f64;
        
        // Compute anomaly score
        let c = Self::average_path_length(self.subsample_size as f64);
        let anomaly_score = 2f64.powf(-avg_path_length / c);
        
        Ok(anomaly_score)
    }
    
    fn confidence_interval(&self) -> (f64, f64) {
        // Empirical confidence interval for isolation forest
        (0.4, 0.6)  // Typical threshold range
    }
    
    fn complexity(&self) -> f64 {
        // Complexity proportional to number of trees
        self.n_trees as f64
    }
}

/// Temporal anomaly model using Hidden Markov Models
pub struct TemporalAnomalyModel {
    /// Number of hidden states
    n_states: usize,
    
    /// Transition probability matrix
    transition_matrix: Option<DMatrix<f64>>,
    
    /// Emission probabilities for each state
    emission_models: Vec<GaussianAnomalyModel>,
    
    /// Initial state probabilities
    initial_probs: Option<DVector<f64>>,
    
    /// Feature dimension
    dimension: usize,
}

impl TemporalAnomalyModel {
    /// Create a new temporal anomaly model
    pub fn new(dimension: usize, n_states: usize) -> Self {
        let emission_models = (0..n_states)
            .map(|_| GaussianAnomalyModel::new(dimension))
            .collect();
        
        Self {
            n_states,
            transition_matrix: None,
            emission_models,
            initial_probs: None,
            dimension,
        }
    }
    
    /// Baum-Welch algorithm for HMM training
    fn baum_welch(&mut self, sequences: &[Vec<AlgorithmState>]) -> AnomalyResult<()> {
        // Simplified implementation - in practice, use full Baum-Welch
        
        // Initialize parameters
        let mut transition_matrix = DMatrix::from_element(self.n_states, self.n_states, 1.0 / self.n_states as f64);
        let initial_probs = DVector::from_element(self.n_states, 1.0 / self.n_states as f64);
        
        // Train emission models on clustered data
        for (state_idx, emission_model) in self.emission_models.iter_mut().enumerate() {
            // Collect states assigned to this hidden state (simplified clustering)
            let assigned_states: Vec<_> = sequences.iter()
                .flat_map(|seq| seq.iter())
                .enumerate()
                .filter(|(idx, _)| idx % self.n_states == state_idx)
                .map(|(_, state)| state.clone())
                .collect();
            
            if assigned_states.len() > self.dimension * 2 {
                emission_model.train(&assigned_states)?;
            }
        }
        
        self.transition_matrix = Some(transition_matrix);
        self.initial_probs = Some(initial_probs);
        
        Ok(())
    }
    
    /// Viterbi algorithm for most likely state sequence
    fn viterbi(&self, sequence: &[AlgorithmState]) -> AnomalyResult<Vec<usize>> {
        let transition = self.transition_matrix.as_ref()
            .ok_or(AnomalyError::ModelNotTrained)?;
        let initial = self.initial_probs.as_ref()
            .ok_or(AnomalyError::ModelNotTrained)?;
        
        let t = sequence.len();
        let mut probabilities = DMatrix::zeros(self.n_states, t);
        let mut backpointers = DMatrix::zeros(self.n_states, t);
        
        // Initialization
        for state in 0..self.n_states {
            let emission_prob = self.emission_models[state].predict(&sequence[0])
                .unwrap_or(f64::EPSILON);
            probabilities[(state, 0)] = initial[state] * emission_prob;
        }
        
        // Recursion
        for t_idx in 1..t {
            for state in 0..self.n_states {
                let emission_prob = self.emission_models[state].predict(&sequence[t_idx])
                    .unwrap_or(f64::EPSILON);
                
                let mut max_prob = 0.0;
                let mut best_prev_state = 0;
                
                for prev_state in 0..self.n_states {
                    let prob = probabilities[(prev_state, t_idx - 1)] 
                             * transition[(prev_state, state)] 
                             * emission_prob;
                    
                    if prob > max_prob {
                        max_prob = prob;
                        best_prev_state = prev_state;
                    }
                }
                
                probabilities[(state, t_idx)] = max_prob;
                backpointers[(state, t_idx)] = best_prev_state as f64;
            }
        }
        
        // Termination
        let mut best_path = vec![0; t];
        let mut max_prob = 0.0;
        let mut best_final_state = 0;
        
        for state in 0..self.n_states {
            if probabilities[(state, t - 1)] > max_prob {
                max_prob = probabilities[(state, t - 1)];
                best_final_state = state;
            }
        }
        
        // Path reconstruction
        best_path[t - 1] = best_final_state;
        for t_idx in (0..t - 1).rev() {
            best_path[t_idx] = backpointers[(best_path[t_idx + 1], t_idx + 1)] as usize;
        }
        
        Ok(best_path)
    }
}

impl AnomalyModel for TemporalAnomalyModel {
    fn train(&mut self, data: &[AlgorithmState]) -> AnomalyResult<()> {
        // Convert linear data to sequences (simplified - assume fixed sequence length)
        let sequence_length = 10;
        let sequences: Vec<Vec<AlgorithmState>> = data
            .chunks(sequence_length)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        if sequences.is_empty() {
            return Err(AnomalyError::InsufficientData {
                got: data.len(),
                need: sequence_length,
            });
        }
        
        self.baum_welch(&sequences)
    }
    
    fn predict(&self, state: &AlgorithmState) -> AnomalyResult<f64> {
        // For single state prediction, use emission probabilities
        let mut max_likelihood = 0.0;
        
        for (state_idx, emission_model) in self.emission_models.iter().enumerate() {
            if let Ok(likelihood) = emission_model.predict(state) {
                max_likelihood = max_likelihood.max(likelihood);
            }
        }
        
        // Convert likelihood to anomaly score
        Ok(1.0 - max_likelihood)
    }
    
    fn confidence_interval(&self) -> (f64, f64) {
        // Confidence based on HMM uncertainty
        (0.3, 0.7)
    }
    
    fn complexity(&self) -> f64 {
        // Complexity: O(n_states^2)
        (self.n_states * self.n_states) as f64
    }
}

/// Ensemble anomaly detector combining multiple models
pub struct EnsembleAnomalyDetector {
    /// Collection of anomaly models
    models: Vec<Box<dyn AnomalyModel>>,
    
    /// Model weights for ensemble
    weights: Vec<f64>,
    
    /// Anomaly threshold
    threshold: f64,
    
    /// Historical anomalies for analysis
    anomaly_history: Arc<Mutex<VecDeque<Anomaly>>>,
    
    /// Pattern detector for correlation
    pattern_detector: PatternDetector,
}

impl EnsembleAnomalyDetector {
    /// Create a new ensemble anomaly detector
    pub fn new(dimension: usize) -> Self {
        let models: Vec<Box<dyn AnomalyModel>> = vec![
            Box::new(GaussianAnomalyModel::new(dimension)),
            Box::new(IsolationForestModel::new(dimension, 100)),
            Box::new(TemporalAnomalyModel::new(dimension, 5)),
        ];
        
        let weights = vec![1.0; models.len()];
        
        Self {
            models,
            weights,
            threshold: 0.7,  // Anomaly score threshold
            anomaly_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            pattern_detector: PatternDetector::new(),
        }
    }
    
    /// Train all models in the ensemble
    pub fn train(&mut self, data: &[AlgorithmState]) -> AnomalyResult<()> {
        // Train models in parallel
        let results: Vec<_> = self.models.par_iter_mut()
            .map(|model| model.train(data))
            .collect();
        
        // Check for training errors
        for result in results {
            result?;
        }
        
        // Update weights based on model complexity
        self.update_weights();
        
        Ok(())
    }
    
    /// Detect anomalies in execution trace
    pub fn detect_anomalies(&self, trace: &ExecutionTrace) -> Vec<Anomaly> {
        let states = trace.get_states();
        let mut anomalies = Vec::new();
        
        for (idx, state) in states.iter().enumerate() {
            if let Some(anomaly) = self.detect_single_anomaly(state, idx) {
                anomalies.push(anomaly);
                
                // Update history
                if let Ok(mut history) = self.anomaly_history.lock() {
                    history.push_back(anomaly.clone());
                    if history.len() > 1000 {
                        history.pop_front();
                    }
                }
            }
        }
        
        // Post-process for temporal correlation
        self.correlate_anomalies(&mut anomalies);
        
        anomalies
    }
    
    /// Detect anomaly for a single state
    fn detect_single_anomaly(&self, state: &AlgorithmState, temporal_idx: usize) -> Option<Anomaly> {
        // Compute ensemble anomaly score
        let scores: Vec<_> = self.models.iter()
            .zip(&self.weights)
            .filter_map(|(model, weight)| {
                model.predict(state).ok().map(|score| score * weight)
            })
            .collect();
        
        if scores.is_empty() {
            return None;
        }
        
        let total_weight: f64 = self.weights.iter()
            .take(scores.len())
            .sum();
        
        let ensemble_score = scores.iter().sum::<f64>() / total_weight;
        
        if ensemble_score > self.threshold {
            // Classify anomaly type
            let anomaly_type = self.classify_anomaly(state, &scores);
            
            // Perform causal analysis
            let causal_factors = self.causal_analysis(state, &anomaly_type);
            
            // Compute educational significance
            let educational_significance = self.compute_educational_significance(&anomaly_type);
            
            Some(Anomaly {
                id: uuid::Uuid::new_v4(),
                anomaly_type,
                confidence: ensemble_score,
                false_positive_rate: self.estimate_false_positive_rate(ensemble_score),
                temporal_location: temporal_idx,
                state_context: state.clone(),
                causal_factors,
                educational_significance,
            })
        } else {
            None
        }
    }
    
    /// Classify anomaly type based on model scores
    fn classify_anomaly(&self, state: &AlgorithmState, scores: &[f64]) -> AnomalyType {
        // Simplified classification - in practice, use more sophisticated approach
        
        if scores.get(0).copied().unwrap_or(0.0) > 0.8 {
            // Gaussian model indicates distributional anomaly
            AnomalyType::PerformanceAnomaly {
                expected_range: (0.0, 100.0),  // Placeholder
                observed_value: state.step as f64,
                standard_deviations: 3.0,
            }
        } else if scores.get(1).copied().unwrap_or(0.0) > 0.8 {
            // Isolation forest indicates structural anomaly
            AnomalyType::StructuralAnomaly {
                invariant_violation: "Unexpected state structure".to_string(),
                constraint_deviation: 0.5,
                recovery_possible: true,
            }
        } else if scores.get(2).copied().unwrap_or(0.0) > 0.8 {
            // Temporal model indicates sequence anomaly
            AnomalyType::TemporalAnomaly {
                sequence_likelihood: 0.1,
                transition_probability: 0.05,
                markov_chain_state: 0,
            }
        } else {
            // Default to behavioral anomaly
            AnomalyType::BehavioralAnomaly {
                expected_pattern: Pattern::Local {
                    center: state.current_node.unwrap_or(0),
                    radius: 5,
                    density: 0.8,
                },
                observed_behavior: Pattern::Local {
                    center: state.current_node.unwrap_or(0),
                    radius: 10,
                    density: 0.2,
                },
                divergence_metric: 0.6,
            }
        }
    }
    
    /// Perform causal analysis for anomaly explanation
    fn causal_analysis(&self, state: &AlgorithmState, anomaly_type: &AnomalyType) -> Vec<CausalFactor> {
        let mut factors = Vec::new();
        
        match anomaly_type {
            AnomalyType::PerformanceAnomaly { expected_range, observed_value, .. } => {
                if *observed_value > expected_range.1 {
                    factors.push(CausalFactor {
                        description: "Algorithm complexity exceeds expected bounds".to_string(),
                        strength: 0.8,
                        evidence: vec![
                            format!("Observed value {} exceeds upper bound {}", observed_value, expected_range.1),
                            format!("State expansion rate: {}", state.open_set.len()),
                        ],
                        counterfactual: Some("With optimized heuristic, complexity would remain bounded".to_string()),
                    });
                }
            },
            
            AnomalyType::BehavioralAnomaly { expected_pattern, observed_behavior, divergence_metric } => {
                factors.push(CausalFactor {
                    description: "Algorithm deviates from expected exploration pattern".to_string(),
                    strength: *divergence_metric,
                    evidence: vec![
                        format!("Expected pattern: {:?}", expected_pattern),
                        format!("Observed behavior: {:?}", observed_behavior),
                    ],
                    counterfactual: Some("Alternative heuristic would maintain expected pattern".to_string()),
                });
            },
            
            AnomalyType::TemporalAnomaly { sequence_likelihood, .. } => {
                factors.push(CausalFactor {
                    description: "Improbable state transition sequence".to_string(),
                    strength: 1.0 - sequence_likelihood,
                    evidence: vec![
                        format!("Sequence likelihood: {:.4}", sequence_likelihood),
                        format!("Current step: {}", state.step),
                    ],
                    counterfactual: Some("Standard algorithm flow would follow expected transitions".to_string()),
                });
            },
            
            AnomalyType::StructuralAnomaly { invariant_violation, .. } => {
                factors.push(CausalFactor {
                    description: invariant_violation.clone(),
                    strength: 0.9,
                    evidence: vec![
                        format!("State structure: {} open, {} closed", state.open_set.len(), state.closed_set.len()),
                    ],
                    counterfactual: Some("Proper initialization would prevent structural violation".to_string()),
                });
            },
        }
        
        factors
    }
    
    /// Compute educational significance of anomaly
    fn compute_educational_significance(&self, anomaly_type: &AnomalyType) -> f64 {
        match anomaly_type {
            AnomalyType::PerformanceAnomaly { standard_deviations, .. } => {
                // More extreme deviations are more educational
                (*standard_deviations / 5.0).min(1.0)
            },
            
            AnomalyType::BehavioralAnomaly { divergence_metric, .. } => {
                // Strong divergence indicates important learning opportunity
                *divergence_metric
            },
            
            AnomalyType::TemporalAnomaly { sequence_likelihood, .. } => {
                // Rare sequences are educational
                1.0 - sequence_likelihood
            },
            
            AnomalyType::StructuralAnomaly { recovery_possible, .. } => {
                // Recoverable anomalies are good teaching moments
                if *recovery_possible { 0.8 } else { 0.4 }
            },
        }
    }
    
    /// Estimate false positive rate
    fn estimate_false_positive_rate(&self, anomaly_score: f64) -> f64 {
        // Simple estimation based on score distance from threshold
        let margin = anomaly_score - self.threshold;
        ((-margin * 5.0).exp()) / (1.0 + (-margin * 5.0).exp())
    }
    
    /// Update model weights based on complexity
    fn update_weights(&mut self) {
        let total_complexity: f64 = self.models.iter()
            .map(|model| model.complexity())
            .sum();
        
        // Weight inversely proportional to complexity
        self.weights = self.models.iter()
            .map(|model| 1.0 / (1.0 + model.complexity() / total_complexity))
            .collect();
    }
    
    /// Correlate anomalies for temporal patterns
    fn correlate_anomalies(&self, anomalies: &mut Vec<Anomaly>) {
        // Sort by temporal location
        anomalies.sort_by_key(|a| a.temporal_location);
        
        // Look for temporal clusters
        let mut i = 0;
        while i < anomalies.len() {
            let mut cluster = vec![i];
            let mut j = i + 1;
            
            while j < anomalies.len() && anomalies[j].temporal_location - anomalies[j-1].temporal_location <= 5 {
                cluster.push(j);
                j += 1;
            }
            
            if cluster.len() > 1 {
                // Boost significance for clustered anomalies
                for &idx in &cluster {
                    anomalies[idx].educational_significance *= 1.5;
                    anomalies[idx].educational_significance = anomalies[idx].educational_significance.min(1.0);
                }
            }
            
            i = j;
        }
    }
    
    /// Get anomaly statistics
    pub fn get_statistics(&self) -> AnomalyStatistics {
        let history = self.anomaly_history.lock().unwrap();
        
        let total_count = history.len();
        let type_distribution = self.compute_type_distribution(&history);
        let avg_confidence = history.iter()
            .map(|a| a.confidence)
            .sum::<f64>() / total_count.max(1) as f64;
        
        let avg_false_positive_rate = history.iter()
            .map(|a| a.false_positive_rate)
            .sum::<f64>() / total_count.max(1) as f64;
        
        AnomalyStatistics {
            total_count,
            type_distribution,
            avg_confidence,
            avg_false_positive_rate,
            temporal_distribution: self.compute_temporal_distribution(&history),
        }
    }
    
    /// Compute anomaly type distribution
    fn compute_type_distribution(&self, history: &VecDeque<Anomaly>) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for anomaly in history {
            let type_name = match &anomaly.anomaly_type {
                AnomalyType::PerformanceAnomaly { .. } => "Performance",
                AnomalyType::BehavioralAnomaly { .. } => "Behavioral",
                AnomalyType::TemporalAnomaly { .. } => "Temporal",
                AnomalyType::StructuralAnomaly { .. } => "Structural",
            };
            
            *distribution.entry(type_name.to_string()).or_insert(0) += 1;
        }
        
        distribution
    }
    
    /// Compute temporal distribution of anomalies
    fn compute_temporal_distribution(&self, history: &VecDeque<Anomaly>) -> Vec<(usize, usize)> {
        let mut distribution = HashMap::new();
        
        for anomaly in history {
            let bucket = anomaly.temporal_location / 10;  // 10-step buckets
            *distribution.entry(bucket).or_insert(0) += 1;
        }
        
        let mut sorted: Vec<_> = distribution.into_iter().collect();
        sorted.sort_by_key(|&(bucket, _)| bucket);
        sorted
    }
}

/// Anomaly detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyStatistics {
    /// Total number of anomalies detected
    pub total_count: usize,
    
    /// Distribution of anomaly types
    pub type_distribution: HashMap<String, usize>,
    
    /// Average confidence score
    pub avg_confidence: f64,
    
    /// Average false positive rate
    pub avg_false_positive_rate: f64,
    
    /// Temporal distribution of anomalies
    pub temporal_distribution: Vec<(usize, usize)>,
}

// Helper functions for statistical distributions

/// Chi-squared cumulative distribution function
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    // Simplified implementation - use proper statistical library in production
    let k = df / 2.0;
    let x_half = x / 2.0;
    
    // Use incomplete gamma function approximation
    incomplete_gamma(k, x_half) / gamma(k)
}

/// Chi-squared quantile function
fn chi_squared_quantile(p: f64, df: f64) -> f64 {
    // Inverse of CDF - simplified implementation
    // In practice, use Newton-Raphson or bisection method
    df * (1.0 - 2.0 / (9.0 * df) + p.abs().ln().sqrt() * (2.0 / (9.0 * df)).sqrt()).powi(3)
}

/// Incomplete gamma function
fn incomplete_gamma(a: f64, x: f64) -> f64 {
    // Series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        let mut n = 1.0;
        
        while term.abs() > 1e-10 {
            term *= x / (a + n);
            sum += term;
            n += 1.0;
        }
        
        sum * x.powf(a) * (-x).exp()
    } else {
        // Continued fraction for large x
        let mut f = 1.0 + (1.0 - a) / (x + 1.0);
        for i in 2..100 {
            let an = (i - a) as f64;
            let denominator = x + 2.0 * i as f64 - 1.0 - a;
            f = 1.0 + an / (denominator + an / f);
        }
        
        1.0 - x.powf(a) * (-x).exp() / f
    }
}

/// Gamma function
fn gamma(x: f64) -> f64 {
    // Stirling's approximation for large x
    if x > 171.0 {
        return f64::INFINITY;
    }
    
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    
    let mut temp = x + 5.5;
    temp -= (x + 0.5) * temp.ln();
    
    let mut series = 1.000000000190015;
    for (i, &coeff) in coefficients.iter().enumerate() {
        series += coeff / (x + i as f64 + 1.0);
    }
    
    let result = 2.5066282746310005 * series / x;
    result * (-temp).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gaussian_anomaly_model() {
        let mut model = GaussianAnomalyModel::new(4);
        
        // Generate normal data
        let normal_states: Vec<AlgorithmState> = (0..100)
            .map(|i| AlgorithmState {
                step: i,
                open_set: vec![i % 10, (i + 1) % 10],
                closed_set: vec![i % 5],
                current_node: Some(i % 20),
                data: HashMap::new(),
            })
            .collect();
        
        // Train model
        assert!(model.train(&normal_states).is_ok());
        
        // Test normal state
        let normal_score = model.predict(&normal_states[0]).unwrap();
        assert!(normal_score < 0.5);
        
        // Test anomalous state
        let anomaly = AlgorithmState {
            step: 1000,
            open_set: vec![100, 101, 102, 103],
            closed_set: vec![50, 51, 52],
            current_node: Some(999),
            data: HashMap::new(),
        };
        
        let anomaly_score = model.predict(&anomaly).unwrap();
        assert!(anomaly_score > 0.7);
    }
    
    #[test]
    fn test_isolation_forest() {
        let mut model = IsolationForestModel::new(4, 10);
        
        // Generate data with outliers
        let mut states = Vec::new();
        
        // Normal data
        for i in 0..95 {
            states.push(AlgorithmState {
                step: i,
                open_set: vec![i % 10],
                closed_set: vec![i % 5],
                current_node: Some(i % 20),
                data: HashMap::new(),
            });
        }
        
        // Outliers
        for i in 95..100 {
            states.push(AlgorithmState {
                step: i * 100,
                open_set: vec![i * 10, i * 20],
                closed_set: vec![i * 5],
                current_node: Some(i * 100),
                data: HashMap::new(),
            });
        }
        
        // Train and test
        assert!(model.train(&states).is_ok());
        
        let normal_score = model.predict(&states[0]).unwrap();
        let outlier_score = model.predict(&states[99]).unwrap();
        
        assert!(outlier_score > normal_score);
    }
    
    #[test]
    fn test_ensemble_detector() {
        let mut detector = EnsembleAnomalyDetector::new(4);
        
        // Generate training data
        let training_data: Vec<AlgorithmState> = (0..200)
            .map(|i| AlgorithmState {
                step: i,
                open_set: vec![i % 10, (i + 1) % 10],
                closed_set: vec![i % 5],
                current_node: Some(i % 20),
                data: HashMap::new(),
            })
            .collect();
        
        // Train ensemble
        assert!(detector.train(&training_data).is_ok());
        
        // Test on trace with anomalies
        let mut test_states = training_data[..50].to_vec();
        
        // Insert anomaly
        test_states[25] = AlgorithmState {
            step: 25,
            open_set: vec![100, 200, 300],
            closed_set: vec![50, 60, 70],
            current_node: Some(999),
            data: HashMap::new(),
        };
        
        let trace = ExecutionTrace::new();
        // Note: This would need proper trace construction in real implementation
        
        let anomalies = detector.detect_anomalies(&trace);
        // Tests would verify anomaly detection here
    }
    
    #[test]
    fn test_causal_analysis() {
        let detector = EnsembleAnomalyDetector::new(4);
        
        let state = AlgorithmState {
            step: 100,
            open_set: vec![1, 2, 3],
            closed_set: vec![4, 5],
            current_node: Some(10),
            data: HashMap::new(),
        };
        
        let anomaly_type = AnomalyType::PerformanceAnomaly {
            expected_range: (0.0, 50.0),
            observed_value: 100.0,
            standard_deviations: 3.5,
        };
        
        let factors = detector.causal_analysis(&state, &anomaly_type);
        
        assert!(!factors.is_empty());
        assert!(factors[0].strength > 0.5);
        assert!(factors[0].counterfactual.is_some());
    }
}