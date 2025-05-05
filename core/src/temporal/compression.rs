//! State Compression Engine
//!
//! This module implements an advanced compression system for algorithm states,
//! utilizing information-theoretic principles to achieve optimal compression
//! ratios while maintaining perfect reconstruction capability.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::mem::{size_of, transmute};
use std::marker::PhantomData;

use serde::{Serialize, Deserialize};
use thiserror::Error;
use rayon::prelude::*;
use bitvec::prelude::*;
use ordered_float::OrderedFloat;

use crate::algorithm::state::{AlgorithmState, StateChecksum};
use crate::temporal::delta::StateDelta;

/// Compression errors with diagnostic context
#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("Invalid compression level: {0} (valid range: 0-9)")]
    InvalidCompressionLevel(u8),
    
    #[error("Decompression failed: state checksum mismatch")]
    ChecksumMismatch,
    
    #[error("Context model overflow at position {0}")]
    ContextOverflow(usize),
    
    #[error("Entropy encoding exceeded maximum size")]
    EntropyOverflow,
    
    #[error("Parallel compression error: {0}")]
    ParallelError(String),
}

/// Compression statistics for monitoring
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub entropy_bits: f64,
    pub prediction_accuracy: f64,
    pub compression_time_ms: f64,
}

/// Compressed state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedState {
    compressed_data: BitVec,
    original_checksum: StateChecksum,
    metadata: CompressionMetadata,
    statistics: CompressionStats,
}

/// Compression metadata for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressionMetadata {
    compression_level: u8,
    context_model_id: u32,
    probability_scale: u32,
    chunk_boundaries: Vec<usize>,
}

/// Compression engine configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub compression_level: u8,
    pub enable_parallel: bool,
    pub context_size: usize,
    pub prediction_order: usize,
    pub entropy_model: EntropyModel,
}

/// Information-theoretic compression engine
pub struct StateCompressor {
    config: CompressionConfig,
    arithmetic_coder: ArithmeticCoder,
    context_model: Arc<ContextModel>,
    prediction_model: PredictionModel,
    entropy_analyzer: EntropyAnalyzer,
}

/// Arithmetic coding implementation for entropy compression
struct ArithmeticCoder {
    /// Probability distribution cache
    prob_cache: HashMap<u32, ProbabilityDistribution>,
    /// Current encoding precision
    precision: u32,
    /// Frequency tables
    frequency_tables: HashMap<u32, FrequencyTable>,
}

/// Context model for adaptive prediction
#[derive(Debug)]
struct ContextModel {
    /// Context size in bits
    context_size: usize,
    /// Context state transitions
    state_transitions: HashMap<u64, TransitionProbabilities>,
    /// Learning rate for adaptation
    learning_rate: f64,
    /// Total observation count
    total_observations: u64,
}

/// Prediction model for state element forecasting
struct PredictionModel {
    /// PPM (Prediction by Partial Matching) models
    ppm_models: Vec<PPMModel>,
    /// Neural predictor for complex patterns
    neural_predictor: NeuralPredictor,
    /// Mixing weights for ensemble
    mixing_weights: Vec<f64>,
}

/// Entropy analyzer for optimal encoding
struct EntropyAnalyzer {
    /// Entropy computation cache
    entropy_cache: HashMap<StateType, f64>,
    /// Kolmogorov complexity estimates
    complexity_estimates: HashMap<StateType, f64>,
    /// Compressibility measure
    compressibility_score: HashMap<StateType, f64>,
}

/// Probability distribution for symbols
#[derive(Debug, Clone)]
struct ProbabilityDistribution {
    /// Symbol frequencies
    frequencies: Vec<u32>,
    /// Total frequency
    total: u32,
    /// Cumulative frequencies for fast lookup
    cumulative: Vec<u32>,
}

/// Frequency table for arithmetic coding
#[derive(Debug, Clone)]
struct FrequencyTable {
    /// Symbol counts
    counts: HashMap<u32, u32>,
    /// Total count
    total: u32,
    /// Escape probability
    escape_prob: f64,
}

/// Context transition probabilities
#[derive(Debug, Clone)]
struct TransitionProbabilities {
    /// Next symbol probabilities
    symbol_probs: HashMap<u32, f64>,
    /// Conditional entropy
    conditional_entropy: f64,
    /// Context entropy
    context_entropy: f64,
}

/// PPM (Prediction by Partial Matching) model
struct PPMModel {
    /// Model order
    order: usize,
    /// Context trie
    context_trie: ContextTrie,
    /// Escape estimation
    escape_estimator: EscapeEstimator,
}

/// Neural prediction network
struct NeuralPredictor {
    /// Network weights (simplified for demonstration)
    weights: Vec<Vec<f64>>,
    /// Activation function
    activation: ActivationFunction,
    /// Training optimizer
    optimizer: AdamOptimizer,
}

impl StateCompressor {
    /// Create new state compressor with advanced configuration
    pub fn new(config: CompressionConfig) -> Result<Self, CompressionError> {
        // Validate compression level
        if config.compression_level > 9 {
            return Err(CompressionError::InvalidCompressionLevel(config.compression_level));
        }
        
        // Initialize arithmetic coder with precision based on level
        let precision = 1u32 << (12 + config.compression_level as u32);
        let arithmetic_coder = ArithmeticCoder::new(precision);
        
        // Initialize context model
        let context_model = Arc::new(ContextModel::new(config.context_size));
        
        // Create prediction model ensemble
        let prediction_model = PredictionModel::new(config.prediction_order);
        
        // Initialize entropy analyzer
        let entropy_analyzer = EntropyAnalyzer::new();
        
        Ok(Self {
            config,
            arithmetic_coder,
            context_model,
            prediction_model,
            entropy_analyzer,
        })
    }
    
    /// Compress algorithm state using information-theoretic methods
    pub fn compress(&self, state: &AlgorithmState) -> Result<CompressedState, CompressionError> {
        let start_time = std::time::Instant::now();
        
        // Analyze state entropy and compressibility
        let entropy_analysis = self.entropy_analyzer.analyze_state(state);
        
        // Select optimal compression strategy based on entropy
        let strategy = self.select_compression_strategy(&entropy_analysis);
        
        // Preprocess state for compression
        let preprocessed = self.preprocess_state(state)?;
        
        // Apply parallel compression if enabled
        let compressed_chunks = if self.config.enable_parallel {
            self.parallel_compress(&preprocessed)?
        } else {
            self.sequential_compress(&preprocessed)?
        };
        
        // Merge compressed chunks
        let compressed_data = self.merge_chunks(&compressed_chunks);
        
        // Generate compression metadata
        let metadata = self.generate_metadata(&compressed_chunks);
        
        // Compute statistics
        let statistics = CompressionStats {
            original_size: size_of_val(state),
            compressed_size: compressed_data.len() / 8,
            compression_ratio: size_of_val(state) as f64 / (compressed_data.len() / 8) as f64,
            entropy_bits: entropy_analysis.total_entropy,
            prediction_accuracy: self.prediction_model.accuracy(),
            compression_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        };
        
        Ok(CompressedState {
            compressed_data,
            original_checksum: state.checksum(),
            metadata,
            statistics,
        })
    }
    
    /// Decompress state with perfect reconstruction guarantee
    pub fn decompress(&self, compressed: &CompressedState) -> Result<AlgorithmState, CompressionError> {
        // Restore context model state
        self.restore_context_model(&compressed.metadata);
        
        // Decompress data chunks
        let decompressed_chunks = self.decompress_chunks(
            &compressed.compressed_data,
            &compressed.metadata.chunk_boundaries,
        )?;
        
        // Reconstruct state from chunks
        let reconstructed_state = self.reconstruct_state(&decompressed_chunks)?;
        
        // Verify checksum for integrity
        if reconstructed_state.checksum() != compressed.original_checksum {
            return Err(CompressionError::ChecksumMismatch);
        }
        
        Ok(reconstructed_state)
    }
    
    /// Preprocess state for optimal compression
    fn preprocess_state(&self, state: &AlgorithmState) -> Result<Vec<u8>, CompressionError> {
        let mut buffer = Vec::with_capacity(size_of_val(state));
        
        // Encode step number using Elias gamma coding for optimal integer compression
        self.encode_integer(state.step as u64, &mut buffer);
        
        // Encode open set using delta encoding
        self.encode_delta_set(&state.open_set, &mut buffer);
        
        // Encode closed set using run-length encoding
        self.encode_rle_set(&state.closed_set, &mut buffer);
        
        // Encode current node with context prediction
        if let Some(node) = state.current_node {
            self.encode_with_prediction(node as u64, &mut buffer);
        }
        
        // Encode metadata with adaptive dictionary
        if let Some(metadata) = &state.metadata {
            self.encode_metadata(metadata, &mut buffer);
        }
        
        Ok(buffer)
    }
    
    /// Parallel compression for large states
    fn parallel_compress(&self, data: &[u8]) -> Result<Vec<BitVec>, CompressionError> {
        // Determine optimal chunk size based on CPU count
        let chunk_size = (data.len() / rayon::current_num_threads()).max(1024);
        
        // Process chunks in parallel
        let chunks: Vec<BitVec> = data
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut context = self.context_model.create_local_context();
                self.compress_chunk(chunk, &mut context)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CompressionError::ParallelError(e.to_string()))?;
        
        Ok(chunks)
    }
    
    /// Compress data chunk using arithmetic coding
    fn compress_chunk(
        &self,
        chunk: &[u8],
        context: &mut LocalContext,
    ) -> Result<BitVec, String> {
        let mut output = BitVec::new();
        
        // Initialize arithmetic coding state
        let mut low: u64 = 0;
        let mut high: u64 = (1u64 << 32) - 1;
        let mut pending_bits = 0;
        
        for &byte in chunk {
            // Get context-based probability distribution
            let distribution = context.get_distribution(byte);
            
            // Update arithmetic coding interval
            let range = high - low + 1;
            high = low + (range * distribution.cumulative[byte as usize] as u64 
                         / distribution.total as u64) - 1;
            low = low + (range * distribution.cumulative[byte as usize - 1] as u64 
                        / distribution.total as u64);
            
            // Output bits when intervals converge
            while (high ^ low) & 0x80000000 == 0 {
                output.push((high & 0x80000000) != 0);
                
                // Shift out MSB and process pending bits
                low <<= 1;
                high = (high << 1) | 1;
                
                for _ in 0..pending_bits {
                    output.push((high & 0x80000000) == 0);
                }
                pending_bits = 0;
            }
            
            // Handle underflow
            while (low & 0x40000000) != 0 && (high & 0x40000000) == 0 {
                pending_bits += 1;
                low = (low ^ 0x40000000) << 1;
                high = ((high ^ 0x40000000) << 1) | 1;
            }
            
            // Update context
            context.update(byte);
        }
        
        // Flush remaining bits
        output.push((low & 0x40000000) != 0);
        for _ in 0..pending_bits + 1 {
            output.push((output[output.len() - 1]) == false);
        }
        
        Ok(output)
    }
    
    /// Entropy analysis for compression strategy
    fn analyze_entropy(&self, data: &[u8]) -> EntropyMetrics {
        let mut symbol_counts = HashMap::new();
        let mut total_symbols = 0;
        
        // Count symbol frequencies
        for &byte in data {
            *symbol_counts.entry(byte).or_insert(0) += 1;
            total_symbols += 1;
        }
        
        // Calculate Shannon entropy
        let mut entropy = 0.0;
        for &count in symbol_counts.values() {
            let probability = count as f64 / total_symbols as f64;
            entropy -= probability * probability.log2();
        }
        
        // Estimate Kolmogorov complexity (simplified)
        let kolmogorov_estimate = self.estimate_kolmogorov_complexity(data);
        
        EntropyMetrics {
            shannon_entropy: entropy,
            kolmogorov_complexity: kolmogorov_estimate,
            symbol_distribution: symbol_counts,
            compressibility_ratio: entropy / 8.0,
        }
    }
    
    /// Prediction-based encoding for context-aware compression
    fn encode_with_prediction(&self, value: u64, buffer: &mut Vec<u8>) {
        // Get prediction from ensemble model
        let prediction = self.prediction_model.predict();
        
        // Calculate residual (actual - predicted)
        let residual = value as i64 - prediction as i64;
        
        // Encode residual using optimal integer coding
        if residual >= 0 {
            self.encode_positive_integer(residual as u64, buffer);
        } else {
            self.encode_negative_integer(residual.abs() as u64, buffer);
        }
        
        // Update prediction model
        self.prediction_model.update(value);
    }
    
    /// Delta encoding for ordered sequences
    fn encode_delta_set(&self, set: &[u32], buffer: &mut Vec<u8>) {
        if set.is_empty() {
            return;
        }
        
        // Encode first element directly
        self.encode_integer(set[0] as u64, buffer);
        
        // Encode deltas for remaining elements
        for i in 1..set.len() {
            let delta = set[i] - set[i-1];
            self.encode_positive_integer(delta as u64, buffer);
        }
    }
    
    /// Kolmogorov complexity estimation using LZW
    fn estimate_kolmogorov_complexity(&self, data: &[u8]) -> f64 {
        let mut dictionary = HashMap::new();
        let mut next_code = 256u32;
        let mut compressed_size = 0;
        
        // Initialize dictionary with single bytes
        for i in 0..256 {
            dictionary.insert(vec![i as u8], i as u32);
        }
        
        let mut current = Vec::new();
        for &byte in data {
            let mut next = current.clone();
            next.push(byte);
            
            if dictionary.contains_key(&next) {
                current = next;
            } else {
                compressed_size += 12; // Assume 12-bit codes
                dictionary.insert(next, next_code);
                next_code += 1;
                current = vec![byte];
            }
        }
        
        if !current.is_empty() {
            compressed_size += 12;
        }
        
        compressed_size as f64 / 8.0 // Convert to bytes
    }
}

impl ArithmeticCoder {
    fn new(precision: u32) -> Self {
        Self {
            prob_cache: HashMap::new(),
            precision,
            frequency_tables: HashMap::new(),
        }
    }
    
    fn update_frequency_table(&mut self, context: u32, symbol: u32) {
        let table = self.frequency_tables.entry(context).or_insert_with(|| {
            FrequencyTable {
                counts: HashMap::new(),
                total: 0,
                escape_prob: 0.1,
            }
        });
        
        *table.counts.entry(symbol).or_insert(0) += 1;
        table.total += 1;
        
        // Adapt escape probability
        table.escape_prob = table.escape_prob * 0.99;
    }
}

impl ContextModel {
    fn new(context_size: usize) -> Self {
        Self {
            context_size,
            state_transitions: HashMap::new(),
            learning_rate: 0.01,
            total_observations: 0,
        }
    }
    
    fn get_context_hash(&self, data: &[u8]) -> u64 {
        let mut hash = 5381u64;
        for &byte in data.iter().take(self.context_size) {
            hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u64);
        }
        hash
    }
    
    fn update_transitions(&mut self, context_hash: u64, symbol: u32) {
        let transitions = self.state_transitions.entry(context_hash).or_insert_with(|| {
            TransitionProbabilities {
                symbol_probs: HashMap::new(),
                conditional_entropy: 0.0,
                context_entropy: 0.0,
            }
        });
        
        // Update symbol probability with learning rate
        let current_prob = transitions.symbol_probs.entry(symbol).or_insert(0.0);
        *current_prob = *current_prob * (1.0 - self.learning_rate) + self.learning_rate;
        
        // Recompute conditional entropy
        transitions.conditional_entropy = self.compute_conditional_entropy(&transitions.symbol_probs);
        
        self.total_observations += 1;
    }
}

impl PredictionModel {
    fn new(order: usize) -> Self {
        // Create ensemble of PPM models with different orders
        let ppm_models = (1..=order)
            .map(|o| PPMModel::new(o))
            .collect();
        
        // Initialize neural predictor
        let neural_predictor = NeuralPredictor::new();
        
        // Initialize mixing weights (uniform initially)
        let mixing_weights = vec![1.0 / (order + 1) as f64; order + 1];
        
        Self {
            ppm_models,
            neural_predictor,
            mixing_weights,
        }
    }
    
    fn predict(&self) -> u32 {
        let mut prediction = 0.0;
        
        // Combine PPM predictions
        for (i, model) in self.ppm_models.iter().enumerate() {
            prediction += self.mixing_weights[i] * model.predict() as f64;
        }
        
        // Add neural network prediction
        prediction += self.mixing_weights.last().unwrap() * self.neural_predictor.predict() as f64;
        
        prediction.round() as u32
    }
    
    fn update(&mut self, actual: u64) {
        let prediction = self.predict();
        let error = (actual as f64 - prediction as f64).abs();
        
        // Adapt mixing weights based on prediction error
        for (i, model) in self.ppm_models.iter_mut().enumerate() {
            let model_error = (actual as f64 - model.predict() as f64).abs();
            self.mixing_weights[i] *= 0.99 + 0.01 * (1.0 / (1.0 + model_error));
        }
        
        // Update neural predictor
        self.neural_predictor.train(actual);
        
        // Normalize weights
        let sum: f64 = self.mixing_weights.iter().sum();
        for weight in &mut self.mixing_weights {
            *weight /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::traits::NodeId;
    
    #[test]
    fn test_compression_ratio() {
        let config = CompressionConfig {
            compression_level: 6,
            enable_parallel: false,
            context_size: 8,
            prediction_order: 4,
            entropy_model: EntropyModel::Adaptive,
        };
        
        let compressor = StateCompressor::new(config).unwrap();
        
        // Create test state
        let state = AlgorithmState {
            step: 100,
            open_set: (0..1000).collect(),
            closed_set: (1000..2000).collect(),
            current_node: Some(NodeId(500)),
            metadata: None,
        };
        
        // Compress state
        let compressed = compressor.compress(&state).unwrap();
        
        // Verify compression ratio
        assert!(compressed.statistics.compression_ratio > 5.0);
        
        // Decompress and verify
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(state, decompressed);
    }
    
    #[test]
    fn test_parallel_compression() {
        let config = CompressionConfig {
            compression_level: 9,
            enable_parallel: true,
            context_size: 16,
            prediction_order: 8,
            entropy_model: EntropyModel::Adaptive,
        };
        
        let compressor = StateCompressor::new(config).unwrap();
        
        // Create large state
        let state = AlgorithmState {
            step: 10000,
            open_set: (0..10000).collect(),
            closed_set: (10000..20000).collect(),
            current_node: Some(NodeId(5000)),
            metadata: None,
        };
        
        // Measure compression time
        let start = std::time::Instant::now();
        let compressed = compressor.compress(&state).unwrap();
        let compression_time = start.elapsed();
        
        // Verify compression efficiency
        assert!(compressed.statistics.compression_ratio > 10.0);
        assert!(compressed.statistics.compression_time_ms < 1000.0);
        
        println!("Parallel compression ratio: {:.2}x in {:.2}ms", 
            compressed.statistics.compression_ratio,
            compression_time.as_secs_f64() * 1000.0
        );
    }
    
    #[test]
    fn test_entropy_analysis() {
        let config = CompressionConfig {
            compression_level: 7,
            enable_parallel: false,
            context_size: 12,
            prediction_order: 6,
            entropy_model: EntropyModel::Optimal,
        };
        
        let compressor = StateCompressor::new(config).unwrap();
        
        // Test with high-entropy data
        let high_entropy: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let entropy_high = compressor.analyze_entropy(&high_entropy);
        
        // Test with low-entropy data
        let low_entropy: Vec<u8> = vec![1; 256];
        let entropy_low = compressor.analyze_entropy(&low_entropy);
        
        assert!(entropy_high.shannon_entropy > 7.5);
        assert!(entropy_low.shannon_entropy < 0.1);
    }
    
    #[test]
    fn test_prediction_model_adaptation() {
        let config = CompressionConfig {
            compression_level: 8,
            enable_parallel: false,
            context_size: 10,
            prediction_order: 5,
            entropy_model: EntropyModel::Predictive,
        };
        
        let mut compressor = StateCompressor::new(config).unwrap();
        
        // Train on sequence
        let sequence: Vec<u64> = (0..100).map(|i| i * 2).collect();
        for &value in &sequence {
            compressor.prediction_model.update(value);
        }
        
        // Test prediction accuracy
        let prediction = compressor.prediction_model.predict();
        assert!((prediction as i64 - 200).abs() < 5);
    }
}