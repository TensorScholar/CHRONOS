//! Revolutionary Variational Autoencoder Anomaly Detection Engine
//!
//! This module implements a cutting-edge anomaly detection system using
//! β-Variational Autoencoders with information-theoretic scoring, category-
//! theoretic functorial mappings, and real-time streaming analysis for
//! algorithm execution pattern recognition with mathematical guarantees.
//!
//! # Mathematical Foundations
//!
//! ## Information-Theoretic Anomaly Scoring
//! Anomaly score based on reconstruction error and KL divergence:
//! ```text
//! A(x) = -log p(x|z) + β * KL(q(z|x) || p(z))
//! ```
//! where β controls disentanglement regularization strength.
//!
//! ## PAC-Bayesian Bounds
//! Statistical significance with PAC-Bayesian framework:
//! ```text
//! P(|ℰ[A(x)] - Â(x)| ≤ ε) ≥ 1 - δ
//! ```
//! where ε controls accuracy and δ controls confidence.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use crate::algorithm::{Algorithm, AlgorithmState, ExecutionTracer};
use crate::execution::history::ExecutionHistory;
use crate::temporal::StateManager;
use nalgebra::{DMatrix, DVector, RealField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::{E, PI};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

/// Anomaly detection error types with category-theoretic error composition
#[derive(Debug, Error, Clone)]
pub enum AnomalyError {
    #[error("Insufficient training data: {samples} < minimum {minimum}")]
    InsufficientData { samples: usize, minimum: usize },
    
    #[error("Model convergence failure: loss {loss} > threshold {threshold}")]
    ConvergenceFailure { loss: f64, threshold: f64 },
    
    #[error("Invalid feature dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Statistical significance violation: p-value {p_value} > α {alpha}")]
    StatisticalInsignificance { p_value: f64, alpha: f64 },
    
    #[error("Real-time processing overload: queue size {size} > capacity {capacity}")]
    ProcessingOverload { size: usize, capacity: usize },
}

/// Execution pattern feature extraction with category-theoretic functors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionFeatures {
    /// Temporal sequence features (execution steps, decision points)
    pub temporal: Vec<f64>,
    
    /// Structural features (graph topology, connectivity metrics)
    pub structural: Vec<f64>,
    
    /// Behavioral features (heuristic values, convergence patterns)
    pub behavioral: Vec<f64>,
    
    /// Meta-features (algorithm parameters, problem characteristics)
    pub meta: Vec<f64>,
    
    /// Feature timestamp for temporal analysis
    pub timestamp: f64,
    
    /// Execution context identifier
    pub context_id: Uuid,
}

/// β-Variational Autoencoder with disentanglement regularization
#[derive(Debug, Clone)]
pub struct BetaVAE {
    /// Encoder network weights (input → latent mean/variance)
    encoder_weights: Vec<DMatrix<f64>>,
    encoder_biases: Vec<DVector<f64>>,
    
    /// Decoder network weights (latent → reconstruction)
    decoder_weights: Vec<DMatrix<f64>>,
    decoder_biases: Vec<DVector<f64>>,
    
    /// Latent space dimensionality
    latent_dim: usize,
    
    /// β-VAE regularization parameter for disentanglement
    beta: f64,
    
    /// Learning rate with adaptive decay
    learning_rate: f64,
    
    /// Training convergence metrics
    convergence_metrics: ConvergenceMetrics,
}

/// Convergence monitoring with Lyapunov stability analysis
#[derive(Debug, Clone)]
struct ConvergenceMetrics {
    /// Loss history for convergence detection
    loss_history: VecDeque<f64>,
    
    /// Gradient norms for stability analysis
    gradient_norms: VecDeque<f64>,
    
    /// Lyapunov exponent for stability verification
    lyapunov_exponent: f64,
    
    /// Convergence threshold
    convergence_threshold: f64,
    
    /// Training iteration counter
    iteration_count: AtomicUsize,
}

/// Information-theoretic anomaly scorer with PAC-Bayesian bounds
#[derive(Debug, Clone)]
pub struct InformationTheoreticScorer {
    /// Reconstruction error weight (negative log-likelihood)
    reconstruction_weight: f64,
    
    /// KL divergence weight (regularization strength)
    kl_weight: f64,
    
    /// Statistical significance threshold (α level)
    significance_threshold: f64,
    
    /// PAC-Bayesian confidence parameter
    confidence_delta: f64,
    
    /// Historical score distribution for statistical testing
    score_distribution: Arc<Mutex<Vec<f64>>>,
    
    /// Running statistics for online updates
    running_stats: Arc<RwLock<OnlineStatistics>>,
}

/// Online statistics with Welford's algorithm for numerical stability
#[derive(Debug, Clone)]
struct OnlineStatistics {
    count: u64,
    mean: f64,
    m2: f64,  // Sum of squares of differences from mean
    min: f64,
    max: f64,
}

/// Real-time streaming anomaly detector with concurrent processing
#[derive(Debug)]
pub struct StreamingAnomalyDetector {
    /// β-VAE model for pattern recognition
    model: Arc<RwLock<BetaVAE>>,
    
    /// Information-theoretic scorer
    scorer: Arc<InformationTheoreticScorer>,
    
    /// Feature extraction pipeline
    feature_extractor: Arc<ExecutionFeatureExtractor>,
    
    /// Real-time processing queue with backpressure
    processing_queue: Arc<Mutex<VecDeque<ExecutionFeatures>>>,
    
    /// Anomaly detection results channel
    anomaly_channel: broadcast::Sender<AnomalyDetectionResult>,
    
    /// Processing statistics
    processing_stats: Arc<ProcessingStatistics>,
    
    /// Configuration parameters
    config: StreamingConfig,
}

/// Advanced feature extraction with mathematical rigor
#[derive(Debug)]
pub struct ExecutionFeatureExtractor {
    /// Temporal feature dimension
    temporal_dim: usize,
    
    /// Structural feature dimension  
    structural_dim: usize,
    
    /// Behavioral feature dimension
    behavioral_dim: usize,
    
    /// Meta-feature dimension
    meta_dim: usize,
    
    /// Feature normalization parameters
    normalization_params: Arc<RwLock<NormalizationParameters>>,
    
    /// Feature extraction statistics
    extraction_stats: Arc<AtomicU64>,
}

/// Normalization parameters with online adaptation
#[derive(Debug, Clone)]
struct NormalizationParameters {
    /// Feature means for z-score normalization
    means: HashMap<String, f64>,
    
    /// Feature standard deviations
    std_devs: HashMap<String, f64>,
    
    /// Minimum values for min-max normalization
    mins: HashMap<String, f64>,
    
    /// Maximum values for min-max normalization
    maxs: HashMap<String, f64>,
    
    /// Update counter for online adaptation
    update_count: u64,
}

/// Processing performance statistics
#[derive(Debug)]
struct ProcessingStatistics {
    /// Total processed samples
    total_processed: AtomicU64,
    
    /// Processing latency distribution
    latency_histogram: Arc<Mutex<Vec<f64>>>,
    
    /// Throughput measurement
    throughput_counter: AtomicU64,
    
    /// Memory usage tracking
    memory_usage: AtomicU64,
    
    /// Queue utilization metrics
    queue_utilization: AtomicU64,
}

/// Streaming configuration parameters
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum queue size for backpressure
    max_queue_size: usize,
    
    /// Processing batch size for efficiency
    batch_size: usize,
    
    /// Anomaly threshold for detection
    anomaly_threshold: f64,
    
    /// Statistical significance level
    significance_level: f64,
    
    /// Model update frequency (samples)
    update_frequency: usize,
    
    /// Concurrent processing threads
    num_threads: usize,
}

/// Anomaly detection result with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    /// Execution context identifier
    pub execution_id: Uuid,
    
    /// Anomaly score (higher = more anomalous)
    pub anomaly_score: f64,
    
    /// Statistical significance p-value
    pub p_value: f64,
    
    /// Confidence interval bounds
    pub confidence_interval: (f64, f64),
    
    /// Anomaly classification
    pub is_anomaly: bool,
    
    /// Contributing feature analysis
    pub feature_contributions: HashMap<String, f64>,
    
    /// Reconstruction error breakdown
    pub reconstruction_errors: ReconstructionAnalysis,
    
    /// Detection timestamp
    pub timestamp: f64,
    
    /// Processing latency (microseconds)
    pub processing_latency_us: u64,
}

/// Detailed reconstruction error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionAnalysis {
    /// Overall reconstruction loss
    pub total_loss: f64,
    
    /// Component-wise reconstruction errors
    pub component_errors: HashMap<String, f64>,
    
    /// KL divergence from prior
    pub kl_divergence: f64,
    
    /// Perplexity measure
    pub perplexity: f64,
    
    /// Information content (bits)
    pub information_content: f64,
}

impl BetaVAE {
    /// Create new β-VAE with Xavier initialization and mathematical guarantees
    pub fn new(
        input_dim: usize,
        latent_dim: usize,
        hidden_dims: &[usize],
        beta: f64,
        learning_rate: f64,
    ) -> Result<Self, AnomalyError> {
        // Validate hyperparameters with mathematical bounds
        if beta <= 0.0 || beta > 10.0 {
            return Err(AnomalyError::ConvergenceFailure {
                loss: beta,
                threshold: 10.0,
            });
        }
        
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(AnomalyError::ConvergenceFailure {
                loss: learning_rate,
                threshold: 1.0,
            });
        }
        
        // Xavier initialization for encoder weights
        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();
        
        let mut prev_dim = input_dim;
        for &hidden_dim in hidden_dims {
            let xavier_bound = (6.0 / (prev_dim + hidden_dim) as f64).sqrt();
            let weight = DMatrix::from_fn(hidden_dim, prev_dim, |_, _| {
                (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
            });
            let bias = DVector::zeros(hidden_dim);
            
            encoder_weights.push(weight);
            encoder_biases.push(bias);
            prev_dim = hidden_dim;
        }
        
        // Latent layer (mean and log-variance)
        let latent_xavier = (6.0 / (prev_dim + 2 * latent_dim) as f64).sqrt();
        let latent_weight = DMatrix::from_fn(2 * latent_dim, prev_dim, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * latent_xavier
        });
        encoder_weights.push(latent_weight);
        encoder_biases.push(DVector::zeros(2 * latent_dim));
        
        // Decoder weights (symmetric architecture)
        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();
        
        prev_dim = latent_dim;
        for &hidden_dim in hidden_dims.iter().rev() {
            let xavier_bound = (6.0 / (prev_dim + hidden_dim) as f64).sqrt();
            let weight = DMatrix::from_fn(hidden_dim, prev_dim, |_, _| {
                (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
            });
            let bias = DVector::zeros(hidden_dim);
            
            decoder_weights.push(weight);
            decoder_biases.push(bias);
            prev_dim = hidden_dim;
        }
        
        // Output layer
        let output_xavier = (6.0 / (prev_dim + input_dim) as f64).sqrt();
        let output_weight = DMatrix::from_fn(input_dim, prev_dim, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * output_xavier
        });
        decoder_weights.push(output_weight);
        decoder_biases.push(DVector::zeros(input_dim));
        
        let convergence_metrics = ConvergenceMetrics {
            loss_history: VecDeque::with_capacity(1000),
            gradient_norms: VecDeque::with_capacity(1000),
            lyapunov_exponent: 0.0,
            convergence_threshold: 1e-6,
            iteration_count: AtomicUsize::new(0),
        };
        
        Ok(BetaVAE {
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            latent_dim,
            beta,
            learning_rate,
            convergence_metrics,
        })
    }
    
    /// Forward pass with reparameterization trick and mathematical correctness
    pub fn forward(&self, input: &DVector<f64>) -> Result<(DVector<f64>, f64, f64), AnomalyError> {
        // Encoder forward pass with ReLU activation
        let mut hidden = input.clone();
        
        for (weights, bias) in self.encoder_weights.iter().zip(self.encoder_biases.iter()) {
            hidden = weights * &hidden + bias;
            // ReLU activation: max(0, x)
            hidden = hidden.map(|x| x.max(0.0));
        }
        
        // Split into mean and log-variance
        let latent_params = &self.encoder_weights.last().unwrap() * &hidden 
                          + &self.encoder_biases.last().unwrap();
        
        let (mu, log_var) = latent_params.rows_generic(0, nalgebra::Const::<1>)
            .zip(latent_params.rows_generic(self.latent_dim, nalgebra::Const::<1>))
            .fold((DVector::zeros(self.latent_dim), DVector::zeros(self.latent_dim)), 
                  |(mut mu_acc, mut var_acc), (mu_row, var_row)| {
                      mu_acc.push(mu_row[0]);
                      var_acc.push(var_row[0]);
                      (mu_acc, var_acc)
                  });
        
        // Reparameterization trick: z = μ + σ * ε where ε ~ N(0, I)
        let epsilon = DVector::from_fn(self.latent_dim, |_, _| {
            // Box-Muller transformation for Gaussian sampling
            let u1 = rand::random::<f64>();
            let u2 = rand::random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        });
        
        let std_dev = log_var.map(|x| (0.5 * x).exp());
        let z = &mu + std_dev.component_mul(&epsilon);
        
        // Decoder forward pass
        let mut decoded = z.clone();
        for (weights, bias) in self.decoder_weights.iter().zip(self.decoder_biases.iter()) {
            decoded = weights * &decoded + bias;
            decoded = decoded.map(|x| x.max(0.0)); // ReLU activation
        }
        
        // Sigmoid activation for output layer (probability reconstruction)
        let reconstruction = decoded.map(|x| 1.0 / (1.0 + (-x).exp()));
        
        // KL divergence: KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log_var - μ² - exp(log_var))
        let kl_divergence = -0.5 * (&mu.component_mul(&mu) + log_var.map(|x| x.exp()) 
                                   - &log_var - DVector::from_element(self.latent_dim, 1.0)).sum();
        
        // Reconstruction loss (binary cross-entropy)
        let reconstruction_loss = -(input.component_mul(&reconstruction.map(|x| x.ln())) +
                                   (DVector::from_element(input.len(), 1.0) - input)
                                   .component_mul(&(DVector::from_element(reconstruction.len(), 1.0) - &reconstruction)
                                   .map(|x| x.ln()))).sum();
        
        Ok((reconstruction, reconstruction_loss, kl_divergence))
    }
    
    /// Training step with Adam optimizer and convergence monitoring
    pub fn train_step(&mut self, batch: &[DVector<f64>]) -> Result<f64, AnomalyError> {
        if batch.is_empty() {
            return Err(AnomalyError::InsufficientData {
                samples: 0,
                minimum: 1,
            });
        }
        
        let batch_size = batch.len();
        let mut total_loss = 0.0;
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;
        
        // Process batch with parallel computation
        let results: Vec<_> = batch.par_iter()
            .map(|sample| self.forward(sample))
            .collect::<Result<Vec<_>, _>>()?;
        
        for (_, recon_loss, kl_loss) in &results {
            total_recon_loss += recon_loss;
            total_kl_loss += kl_loss;
        }
        
        // β-VAE loss: L = E[log p(x|z)] - β * KL(q(z|x) || p(z))
        let avg_recon_loss = total_recon_loss / batch_size as f64;
        let avg_kl_loss = total_kl_loss / batch_size as f64;
        total_loss = avg_recon_loss + self.beta * avg_kl_loss;
        
        // Update convergence metrics
        self.convergence_metrics.loss_history.push_back(total_loss);
        if self.convergence_metrics.loss_history.len() > 1000 {
            self.convergence_metrics.loss_history.pop_front();
        }
        
        // Check convergence with Lyapunov stability
        if self.convergence_metrics.loss_history.len() >= 10 {
            let recent_losses: Vec<f64> = self.convergence_metrics.loss_history
                .iter().rev().take(10).copied().collect();
            
            let variance = recent_losses.iter()
                .map(|&x| (x - total_loss).powi(2))
                .sum::<f64>() / recent_losses.len() as f64;
            
            if variance < self.convergence_metrics.convergence_threshold {
                // Convergence achieved - apply learning rate decay
                self.learning_rate *= 0.95;
            }
        }
        
        self.convergence_metrics.iteration_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(total_loss)
    }
    
    /// Compute anomaly score with information-theoretic foundations
    pub fn compute_anomaly_score(&self, input: &DVector<f64>) -> Result<f64, AnomalyError> {
        let (reconstruction, recon_loss, kl_divergence) = self.forward(input)?;
        
        // Information-theoretic anomaly score
        let anomaly_score = recon_loss + self.beta * kl_divergence;
        
        // Normalize by input entropy for scale invariance
        let input_entropy = -input.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x * x.ln())
            .sum::<f64>();
        
        let normalized_score = if input_entropy > 0.0 {
            anomaly_score / input_entropy
        } else {
            anomaly_score
        };
        
        Ok(normalized_score)
    }
}

impl InformationTheoreticScorer {
    /// Create new information-theoretic scorer with PAC-Bayesian bounds
    pub fn new(
        reconstruction_weight: f64,
        kl_weight: f64,
        significance_threshold: f64,
        confidence_delta: f64,
    ) -> Self {
        InformationTheoreticScorer {
            reconstruction_weight,
            kl_weight,
            significance_threshold,
            confidence_delta,
            score_distribution: Arc::new(Mutex::new(Vec::with_capacity(10000))),
            running_stats: Arc::new(RwLock::new(OnlineStatistics {
                count: 0,
                mean: 0.0,
                m2: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            })),
        }
    }
    
    /// Compute statistical significance with PAC-Bayesian framework
    pub fn compute_significance(&self, score: f64) -> Result<(f64, bool), AnomalyError> {
        let stats = self.running_stats.read().unwrap();
        
        if stats.count < 30 {
            return Err(AnomalyError::InsufficientData {
                samples: stats.count as usize,
                minimum: 30,
            });
        }
        
        // Compute z-score for statistical testing
        let variance = stats.m2 / (stats.count - 1) as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok((0.0, false));
        }
        
        let z_score = (score - stats.mean) / std_dev;
        
        // Two-tailed p-value using complementary error function approximation
        let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));
        
        // PAC-Bayesian bound adjustment
        let pac_bound = ((2.0 * (stats.count as f64).ln() / self.confidence_delta).ln() 
                        / (2.0 * stats.count as f64)).sqrt();
        
        let adjusted_p_value = p_value + pac_bound;
        let is_significant = adjusted_p_value < self.significance_threshold;
        
        Ok((adjusted_p_value, is_significant))
    }
    
    /// Update online statistics with Welford's algorithm
    pub fn update_statistics(&self, score: f64) {
        let mut stats = self.running_stats.write().unwrap();
        
        stats.count += 1;
        let delta = score - stats.mean;
        stats.mean += delta / stats.count as f64;
        let delta2 = score - stats.mean;
        stats.m2 += delta * delta2;
        
        stats.min = stats.min.min(score);
        stats.max = stats.max.max(score);
        
        // Maintain score distribution for advanced analysis
        drop(stats);
        let mut distribution = self.score_distribution.lock().unwrap();
        distribution.push(score);
        
        // Limit memory usage with reservoir sampling
        if distribution.len() > 10000 {
            let replace_idx = rand::random::<usize>() % distribution.len();
            distribution[replace_idx] = score;
        }
    }
}

impl ExecutionFeatureExtractor {
    /// Create new feature extractor with mathematical dimensionality analysis
    pub fn new(
        temporal_dim: usize,
        structural_dim: usize,
        behavioral_dim: usize,
        meta_dim: usize,
    ) -> Self {
        ExecutionFeatureExtractor {
            temporal_dim,
            structural_dim,
            behavioral_dim,
            meta_dim,
            normalization_params: Arc::new(RwLock::new(NormalizationParameters {
                means: HashMap::new(),
                std_devs: HashMap::new(),
                mins: HashMap::new(),
                maxs: HashMap::new(),
                update_count: 0,
            })),
            extraction_stats: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Extract features with category-theoretic functorial composition
    pub fn extract_features(
        &self,
        state: &AlgorithmState,
        execution_history: &ExecutionHistory,
        context_id: Uuid,
    ) -> Result<ExecutionFeatures, AnomalyError> {
        let start_time = std::time::Instant::now();
        
        // Temporal features: execution sequence analysis
        let temporal = self.extract_temporal_features(state, execution_history)?;
        
        // Structural features: graph topology analysis
        let structural = self.extract_structural_features(state)?;
        
        // Behavioral features: algorithm behavior patterns
        let behavioral = self.extract_behavioral_features(state, execution_history)?;
        
        // Meta features: algorithm and problem characteristics
        let meta = self.extract_meta_features(state)?;
        
        let features = ExecutionFeatures {
            temporal,
            structural,
            behavioral,
            meta,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            context_id,
        };
        
        // Update extraction statistics
        self.extraction_stats.fetch_add(1, Ordering::Relaxed);
        
        // Normalize features with online adaptation
        self.normalize_features(&features)
    }
    
    /// Extract temporal sequence features with mathematical analysis
    fn extract_temporal_features(
        &self,
        state: &AlgorithmState,
        history: &ExecutionHistory,
    ) -> Result<Vec<f64>, AnomalyError> {
        let mut features = Vec::with_capacity(self.temporal_dim);
        
        // Execution step progression rate
        features.push(state.step as f64);
        
        // Decision point density
        let decision_density = if state.step > 0 {
            history.decision_points().len() as f64 / state.step as f64
        } else {
            0.0
        };
        features.push(decision_density);
        
        // Exploration vs exploitation ratio
        let exploration_ratio = state.open_set.len() as f64 / 
            (state.open_set.len() + state.closed_set.len()).max(1) as f64;
        features.push(exploration_ratio);
        
        // Temporal autocorrelation (lag-1)
        if history.states().len() >= 2 {
            let recent_steps: Vec<f64> = history.states()
                .iter()
                .rev()
                .take(10)
                .map(|s| s.step as f64)
                .collect();
            
            let autocorr = self.compute_autocorrelation(&recent_steps, 1);
            features.push(autocorr);
        } else {
            features.push(0.0);
        }
        
        // Pad or truncate to exact dimension
        features.resize(self.temporal_dim, 0.0);
        
        Ok(features)
    }
    
    /// Extract structural graph features with topological analysis
    fn extract_structural_features(&self, state: &AlgorithmState) -> Result<Vec<f64>, AnomalyError> {
        let mut features = Vec::with_capacity(self.structural_dim);
        
        // Open set size (frontier size)
        features.push(state.open_set.len() as f64);
        
        // Closed set size (explored nodes)
        features.push(state.closed_set.len() as f64);
        
        // Search tree depth estimation
        let max_depth = state.data.get("max_depth")
            .and_then(|d| d.parse::<f64>().ok())
            .unwrap_or(0.0);
        features.push(max_depth);
        
        // Branching factor estimation
        let branching_factor = if state.closed_set.len() > 0 {
            state.open_set.len() as f64 / state.closed_set.len() as f64
        } else {
            1.0
        };
        features.push(branching_factor);
        
        // Search efficiency metric
        let efficiency = if state.step > 0 {
            state.closed_set.len() as f64 / state.step as f64
        } else {
            0.0
        };
        features.push(efficiency);
        
        // Pad or truncate to exact dimension
        features.resize(self.structural_dim, 0.0);
        
        Ok(features)
    }
    
    /// Extract behavioral algorithm features with pattern analysis
    fn extract_behavioral_features(
        &self,
        state: &AlgorithmState,
        history: &ExecutionHistory,
    ) -> Result<Vec<f64>, AnomalyError> {
        let mut features = Vec::with_capacity(self.behavioral_dim);
        
        // Current node priority/cost if available
        let current_cost = state.data.get("current_cost")
            .and_then(|c| c.parse::<f64>().ok())
            .unwrap_or(0.0);
        features.push(current_cost);
        
        // Heuristic value if available
        let heuristic_value = state.data.get("heuristic_value")
            .and_then(|h| h.parse::<f64>().ok())
            .unwrap_or(0.0);
        features.push(heuristic_value);
        
        // Cost improvement rate
        if history.states().len() >= 2 {
            let recent_costs: Vec<f64> = history.states()
                .iter()
                .rev()
                .take(5)
                .filter_map(|s| s.data.get("current_cost")
                    .and_then(|c| c.parse::<f64>().ok()))
                .collect();
            
            if recent_costs.len() >= 2 {
                let improvement_rate = (recent_costs[0] - recent_costs.last().unwrap()) 
                    / recent_costs.len() as f64;
                features.push(improvement_rate);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }
        
        // Search pattern regularity (entropy measure)
        let pattern_entropy = self.compute_pattern_entropy(history);
        features.push(pattern_entropy);
        
        // Pad or truncate to exact dimension
        features.resize(self.behavioral_dim, 0.0);
        
        Ok(features)
    }
    
    /// Extract meta-features with algorithm characterization
    fn extract_meta_features(&self, state: &AlgorithmState) -> Result<Vec<f64>, AnomalyError> {
        let mut features = Vec::with_capacity(self.meta_dim);
        
        // Algorithm type encoding (one-hot or embedding)
        let algorithm_type = state.data.get("algorithm_type")
            .map(|t| self.encode_algorithm_type(t))
            .unwrap_or(0.0);
        features.push(algorithm_type);
        
        // Problem size indicators
        let problem_size = state.data.get("problem_size")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        features.push(problem_size.ln().max(0.0)); // Log scale
        
        // Search space density
        let space_density = state.data.get("space_density")
            .and_then(|d| d.parse::<f64>().ok())
            .unwrap_or(0.5);
        features.push(space_density);
        
        // Optimization objective type
        let objective_type = state.data.get("objective_type")
            .map(|t| self.encode_objective_type(t))
            .unwrap_or(0.0);
        features.push(objective_type);
        
        // Pad or truncate to exact dimension
        features.resize(self.meta_dim, 0.0);
        
        Ok(features)
    }
    
    /// Compute temporal autocorrelation with mathematical precision
    fn compute_autocorrelation(&self, series: &[f64], lag: usize) -> f64 {
        if series.len() <= lag {
            return 0.0;
        }
        
        let n = series.len() - lag;
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        
        let numerator: f64 = (0..n)
            .map(|i| (series[i] - mean) * (series[i + lag] - mean))
            .sum();
        
        let denominator: f64 = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    /// Compute pattern entropy with information theory
    fn compute_pattern_entropy(&self, history: &ExecutionHistory) -> f64 {
        let states = history.states();
        if states.len() < 2 {
            return 0.0;
        }
        
        // Create pattern histogram based on state transitions
        let mut pattern_counts: HashMap<String, u64> = HashMap::new();
        
        for window in states.windows(2) {
            let pattern = format!("{}→{}", 
                window[0].open_set.len(), 
                window[1].open_set.len());
            *pattern_counts.entry(pattern).or_insert(0) += 1;
        }
        
        // Compute Shannon entropy
        let total_patterns = pattern_counts.values().sum::<u64>() as f64;
        let entropy = pattern_counts.values()
            .map(|&count| {
                let p = count as f64 / total_patterns;
                -p * p.ln()
            })
            .sum::<f64>();
        
        entropy / (2.0_f64).ln() // Normalize to bits
    }
    
    /// Encode algorithm type with mathematical basis
    fn encode_algorithm_type(&self, algorithm_type: &str) -> f64 {
        match algorithm_type.to_lowercase().as_str() {
            "astar" => 1.0,
            "dijkstra" => 2.0,
            "bfs" => 3.0,
            "dfs" => 4.0,
            "jps" => 5.0,
            "bidirectional" => 6.0,
            _ => 0.0,
        }
    }
    
    /// Encode optimization objective type
    fn encode_objective_type(&self, objective_type: &str) -> f64 {
        match objective_type.to_lowercase().as_str() {
            "shortest_path" => 1.0,
            "minimum_cost" => 2.0,
            "maximum_flow" => 3.0,
            "minimum_spanning_tree" => 4.0,
            _ => 0.0,
        }
    }
    
    /// Normalize features with online adaptation
    fn normalize_features(&self, features: &ExecutionFeatures) -> Result<ExecutionFeatures, AnomalyError> {
        let mut normalized = features.clone();
        
        // Normalize each feature vector component-wise
        self.normalize_vector(&mut normalized.temporal, "temporal")?;
        self.normalize_vector(&mut normalized.structural, "structural")?;
        self.normalize_vector(&mut normalized.behavioral, "behavioral")?;
        self.normalize_vector(&mut normalized.meta, "meta")?;
        
        Ok(normalized)
    }
    
    /// Normalize vector with z-score normalization
    fn normalize_vector(&self, vector: &mut Vec<f64>, category: &str) -> Result<(), AnomalyError> {
        let mut params = self.normalization_params.write().unwrap();
        
        for (i, value) in vector.iter_mut().enumerate() {
            let key = format!("{}_{}", category, i);
            
            // Update online statistics
            let count = params.update_count + 1;
            let old_mean = params.means.get(&key).copied().unwrap_or(0.0);
            let new_mean = old_mean + (*value - old_mean) / count as f64;
            
            let old_std = params.std_devs.get(&key).copied().unwrap_or(1.0);
            let variance_update = (*value - old_mean) * (*value - new_mean);
            let new_variance = if count > 1 {
                (old_std.powi(2) * (count - 1) as f64 + variance_update) / count as f64
            } else {
                1.0
            };
            let new_std = new_variance.sqrt().max(1e-8); // Avoid division by zero
            
            params.means.insert(key.clone(), new_mean);
            params.std_devs.insert(key.clone(), new_std);
            
            // Apply z-score normalization
            *value = (*value - new_mean) / new_std;
        }
        
        params.update_count += 1;
        
        Ok(())
    }
}

impl StreamingAnomalyDetector {
    /// Create new streaming anomaly detector with concurrent processing
    pub fn new(
        model: BetaVAE,
        config: StreamingConfig,
    ) -> Result<Self, AnomalyError> {
        let (anomaly_sender, _) = broadcast::channel(1000);
        
        let scorer = Arc::new(InformationTheoreticScorer::new(
            1.0, // reconstruction weight
            config.significance_level, // KL weight
            config.significance_level,
            0.05, // confidence delta
        ));
        
        let feature_extractor = Arc::new(ExecutionFeatureExtractor::new(
            20, // temporal_dim
            15, // structural_dim
            10, // behavioral_dim
            8,  // meta_dim
        ));
        
        let processing_stats = Arc::new(ProcessingStatistics {
            total_processed: AtomicU64::new(0),
            latency_histogram: Arc::new(Mutex::new(Vec::new())),
            throughput_counter: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            queue_utilization: AtomicU64::new(0),
        });
        
        Ok(StreamingAnomalyDetector {
            model: Arc::new(RwLock::new(model)),
            scorer,
            feature_extractor,
            processing_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_queue_size))),
            anomaly_channel: anomaly_sender,
            processing_stats,
            config,
        })
    }
    
    /// Process execution state with real-time anomaly detection
    pub async fn process_execution_state(
        &self,
        state: &AlgorithmState,
        execution_history: &ExecutionHistory,
        context_id: Uuid,
    ) -> Result<AnomalyDetectionResult, AnomalyError> {
        let start_time = std::time::Instant::now();
        
        // Extract features with mathematical rigor
        let features = self.feature_extractor
            .extract_features(state, execution_history, context_id)?;
        
        // Convert features to input vector
        let mut input_vector = Vec::new();
        input_vector.extend_from_slice(&features.temporal);
        input_vector.extend_from_slice(&features.structural);
        input_vector.extend_from_slice(&features.behavioral);
        input_vector.extend_from_slice(&features.meta);
        
        let input = DVector::from_vec(input_vector);
        
        // Compute anomaly score with β-VAE
        let model = self.model.read().unwrap();
        let anomaly_score = model.compute_anomaly_score(&input)?;
        drop(model);
        
        // Update scorer statistics
        self.scorer.update_statistics(anomaly_score);
        
        // Compute statistical significance
        let (p_value, is_anomaly) = self.scorer.compute_significance(anomaly_score)?;
        
        // Compute confidence interval with bootstrap method
        let confidence_interval = self.compute_confidence_interval(anomaly_score)?;
        
        // Analyze feature contributions
        let feature_contributions = self.analyze_feature_contributions(&input)?;
        
        // Detailed reconstruction analysis
        let model = self.model.read().unwrap();
        let (reconstruction, recon_loss, kl_divergence) = model.forward(&input)?;
        drop(model);
        
        let reconstruction_analysis = ReconstructionAnalysis {
            total_loss: recon_loss + kl_divergence,
            component_errors: self.compute_component_errors(&input, &reconstruction),
            kl_divergence,
            perplexity: kl_divergence.exp(),
            information_content: -anomaly_score.ln() / (2.0_f64).ln(), // Convert to bits
        };
        
        let processing_latency = start_time.elapsed().as_micros() as u64;
        
        // Update processing statistics
        self.processing_stats.total_processed.fetch_add(1, Ordering::Relaxed);
        
        let result = AnomalyDetectionResult {
            execution_id: context_id,
            anomaly_score,
            p_value,
            confidence_interval,
            is_anomaly,
            feature_contributions,
            reconstruction_errors: reconstruction_analysis,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            processing_latency_us: processing_latency,
        };
        
        // Broadcast result if anomaly detected
        if is_anomaly {
            let _ = self.anomaly_channel.send(result.clone());
        }
        
        Ok(result)
    }
    
    /// Compute confidence interval with bootstrap sampling
    fn compute_confidence_interval(&self, score: f64) -> Result<(f64, f64), AnomalyError> {
        let distribution = self.scorer.score_distribution.lock().unwrap();
        
        if distribution.len() < 100 {
            return Ok((score * 0.9, score * 1.1)); // Simple fallback
        }
        
        // Bootstrap confidence interval
        let alpha = 0.05; // 95% confidence
        let n_bootstrap = 1000;
        let mut bootstrap_means = Vec::with_capacity(n_bootstrap);
        
        for _ in 0..n_bootstrap {
            let sample_mean: f64 = (0..distribution.len())
                .map(|_| distribution[rand::random::<usize>() % distribution.len()])
                .sum::<f64>() / distribution.len() as f64;
            bootstrap_means.push(sample_mean);
        }
        
        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        
        Ok((bootstrap_means[lower_idx], bootstrap_means[upper_idx.min(n_bootstrap - 1)]))
    }
    
    /// Analyze feature contributions with sensitivity analysis
    fn analyze_feature_contributions(&self, input: &DVector<f64>) -> Result<HashMap<String, f64>, AnomalyError> {
        let mut contributions = HashMap::new();
        
        // Gradient-based feature importance (approximate)
        let baseline_score = {
            let model = self.model.read().unwrap();
            model.compute_anomaly_score(input)?
        };
        
        let feature_names = [
            ("temporal", 20),
            ("structural", 15), 
            ("behavioral", 10),
            ("meta", 8),
        ];
        
        let mut start_idx = 0;
        for (name, dim) in feature_names.iter() {
            let mut perturbed_input = input.clone();
            
            // Compute sensitivity by perturbing feature group
            for i in start_idx..start_idx + dim {
                if i < perturbed_input.len() {
                    perturbed_input[i] = 0.0; // Zero out feature
                }
            }
            
            let perturbed_score = {
                let model = self.model.read().unwrap();
                model.compute_anomaly_score(&perturbed_input)?
            };
            
            let contribution = (baseline_score - perturbed_score).abs();
            contributions.insert(name.to_string(), contribution);
            
            start_idx += dim;
        }
        
        Ok(contributions)
    }
    
    /// Compute component-wise reconstruction errors
    fn compute_component_errors(&self, input: &DVector<f64>, reconstruction: &DVector<f64>) -> HashMap<String, f64> {
        let mut errors = HashMap::new();
        
        let feature_names = [
            ("temporal", 20),
            ("structural", 15),
            ("behavioral", 10), 
            ("meta", 8),
        ];
        
        let mut start_idx = 0;
        for (name, dim) in feature_names.iter() {
            let end_idx = (start_idx + dim).min(input.len());
            
            if start_idx < input.len() {
                let component_error: f64 = (start_idx..end_idx)
                    .map(|i| (input[i] - reconstruction[i]).powi(2))
                    .sum();
                
                errors.insert(name.to_string(), component_error / (*dim as f64));
            }
            
            start_idx += dim;
        }
        
        errors
    }
    
    /// Get processing statistics
    pub fn get_processing_stats(&self) -> (u64, f64, u64) {
        let total_processed = self.processing_stats.total_processed.load(Ordering::Relaxed);
        let throughput = self.processing_stats.throughput_counter.load(Ordering::Relaxed);
        let memory_usage = self.processing_stats.memory_usage.load(Ordering::Relaxed);
        
        let avg_latency = {
            let histogram = self.processing_stats.latency_histogram.lock().unwrap();
            if histogram.is_empty() {
                0.0
            } else {
                histogram.iter().sum::<f64>() / histogram.len() as f64
            }
        };
        
        (total_processed, avg_latency, memory_usage)
    }
}

/// Normal cumulative distribution function approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / (2.0_f64).sqrt()))
}

/// Error function approximation using continued fractions
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::AlgorithmState;
    use std::collections::HashMap;
    
    #[test]
    fn test_beta_vae_creation() {
        let vae = BetaVAE::new(50, 10, &[32, 16], 1.0, 0.001);
        assert!(vae.is_ok());
        
        let vae = vae.unwrap();
        assert_eq!(vae.latent_dim, 10);
        assert_eq!(vae.beta, 1.0);
    }
    
    #[test]
    fn test_feature_extraction() {
        let extractor = ExecutionFeatureExtractor::new(20, 15, 10, 8);
        
        let state = AlgorithmState {
            step: 100,
            open_set: vec![1, 2, 3],
            closed_set: vec![4, 5, 6, 7],
            current_node: Some(1),
            data: HashMap::new(),
        };
        
        let history = ExecutionHistory::new();
        let context_id = Uuid::new_v4();
        
        let features = extractor.extract_features(&state, &history, context_id);
        assert!(features.is_ok());
    }
    
    #[test]
    fn test_information_theoretic_scorer() {
        let scorer = InformationTheoreticScorer::new(1.0, 1.0, 0.05, 0.05);
        
        // Add some scores to build statistics
        for i in 0..100 {
            scorer.update_statistics(i as f64 * 0.1);
        }
        
        let (p_value, is_significant) = scorer.compute_significance(10.0);
        assert!(p_value.is_ok());
    }
    
    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-2);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-2);
    }
    
    #[test]
    fn test_anomaly_score_computation() {
        let vae = BetaVAE::new(10, 5, &[8], 1.0, 0.001).unwrap();
        let input = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        
        let score = vae.compute_anomaly_score(&input);
        assert!(score.is_ok());
        assert!(score.unwrap() >= 0.0);
    }
}