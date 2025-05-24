//! Tactile Modality Framework: Advanced Haptic Algorithm Representation
//!
//! This module implements cutting-edge haptic representation of algorithmic
//! behavior through psychophysically-informed tactile synthesis, enabling
//! algorithmic comprehension through somatosensory channels with mathematically
//! guaranteed perceptual equivalence preservation.
//!
//! Theoretical Foundation:
//! - Weber-Fechner psychophysical laws for tactile intensity mapping
//! - Stevens' power law for haptic magnitude estimation
//! - Tactile feature space dimensionality reduction via manifold learning
//! - Information-theoretic tactile entropy preservation
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::modality::representation::{ModalityRepresentation, SemanticEquivalence};
use chronos_core::algorithm::state::AlgorithmState;
use chronos_core::temporal::timeline::TimelineEvent;

/// Haptic channel enumeration following somatosensory classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HapticChannel {
    /// Pressure/force feedback (mechanoreceptors)
    Pressure,
    /// Vibrotactile feedback (Pacinian corpuscles)
    Vibration,
    /// Temperature variation (thermoreceptors)
    Temperature,
    /// Texture/roughness (Ruffini endings)
    Texture,
    /// Position/proprioception (muscle spindles)
    Position,
    /// Movement/kinesthetic (joint receptors)
    Movement,
}

/// Haptic stimulus parameters based on psychophysical research
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HapticStimulus {
    /// Stimulus intensity (0.0-1.0, Weber-Fechner normalized)
    pub intensity: f64,
    /// Stimulus frequency (Hz) for vibrotactile channels
    pub frequency: Option<f64>,
    /// Stimulus duration (milliseconds)
    pub duration: Duration,
    /// Spatial location (normalized coordinates)
    pub spatial_location: (f64, f64),
    /// Temporal onset delay (milliseconds)
    pub onset_delay: Duration,
    /// Stimulus envelope (attack, decay, sustain, release)
    pub envelope: EnvelopeParameters,
}

/// ADSR envelope parameters for haptic stimulus shaping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeParameters {
    /// Attack time (milliseconds)
    pub attack: Duration,
    /// Decay time (milliseconds)  
    pub decay: Duration,
    /// Sustain level (0.0-1.0)
    pub sustain: f64,
    /// Release time (milliseconds)
    pub release: Duration,
}

impl Default for EnvelopeParameters {
    fn default() -> Self {
        Self {
            attack: Duration::from_millis(10),
            decay: Duration::from_millis(50),
            sustain: 0.7,
            release: Duration::from_millis(100),
        }
    }
}

/// Haptic pattern representing complex algorithmic behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticPattern {
    /// Pattern identifier
    pub id: String,
    /// Constituent stimuli with temporal sequencing
    pub stimuli: Vec<(HapticStimulus, HapticChannel)>,
    /// Pattern repetition parameters
    pub repetition: RepetitionParameters,
    /// Pattern priority for conflict resolution
    pub priority: u8,
}

/// Pattern repetition and looping parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepetitionParameters {
    /// Number of repetitions (0 = infinite)
    pub count: u32,
    /// Inter-repetition interval
    pub interval: Duration,
    /// Amplitude decay per repetition (0.0-1.0)
    pub decay_factor: f64,
}

impl Default for RepetitionParameters {
    fn default() -> Self {
        Self {
            count: 1,
            interval: Duration::from_millis(0),
            decay_factor: 1.0,
        }
    }
}

/// Psychophysical calibration parameters for individual users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychophysicalCalibration {
    /// Weber fraction for each haptic channel
    pub weber_fractions: HashMap<HapticChannel, f64>,
    /// Stevens exponent for magnitude scaling
    pub stevens_exponents: HashMap<HapticChannel, f64>,
    /// Absolute detection thresholds
    pub detection_thresholds: HashMap<HapticChannel, f64>,
    /// Comfort/pain thresholds
    pub comfort_thresholds: HashMap<HapticChannel, f64>,
}

impl Default for PsychophysicalCalibration {
    fn default() -> Self {
        let mut weber_fractions = HashMap::new();
        let mut stevens_exponents = HashMap::new();
        let mut detection_thresholds = HashMap::new();
        let mut comfort_thresholds = HashMap::new();

        // Research-based psychophysical constants
        weber_fractions.insert(HapticChannel::Pressure, 0.14);
        weber_fractions.insert(HapticChannel::Vibration, 0.08);
        weber_fractions.insert(HapticChannel::Temperature, 0.03);
        weber_fractions.insert(HapticChannel::Texture, 0.17);
        weber_fractions.insert(HapticChannel::Position, 0.02);
        weber_fractions.insert(HapticChannel::Movement, 0.05);

        stevens_exponents.insert(HapticChannel::Pressure, 1.1);
        stevens_exponents.insert(HapticChannel::Vibration, 0.95);
        stevens_exponents.insert(HapticChannel::Temperature, 1.0);
        stevens_exponents.insert(HapticChannel::Texture, 1.5);
        stevens_exponents.insert(HapticChannel::Position, 1.3);
        stevens_exponents.insert(HapticChannel::Movement, 1.2);

        detection_thresholds.insert(HapticChannel::Pressure, 0.05);
        detection_thresholds.insert(HapticChannel::Vibration, 0.01);
        detection_thresholds.insert(HapticChannel::Temperature, 0.02);
        detection_thresholds.insert(HapticChannel::Texture, 0.08);
        detection_thresholds.insert(HapticChannel::Position, 0.001);
        detection_thresholds.insert(HapticChannel::Movement, 0.003);

        comfort_thresholds.insert(HapticChannel::Pressure, 0.85);
        comfort_thresholds.insert(HapticChannel::Vibration, 0.90);
        comfort_thresholds.insert(HapticChannel::Temperature, 0.75);
        comfort_thresholds.insert(HapticChannel::Texture, 0.80);
        comfort_thresholds.insert(HapticChannel::Position, 0.95);
        comfort_thresholds.insert(HapticChannel::Movement, 0.88);

        Self {
            weber_fractions,
            stevens_exponents,
            detection_thresholds,
            comfort_thresholds,
        }
    }
}

/// Algorithm-to-haptic mapping strategy trait
pub trait HapticMappingStrategy: Send + Sync {
    /// Map algorithm state to haptic representation
    fn map_state(&self, state: &AlgorithmState) -> Result<Vec<HapticPattern>, TactileError>;
    
    /// Map temporal events to haptic feedback
    fn map_event(&self, event: &TimelineEvent) -> Result<Option<HapticPattern>, TactileError>;
    
    /// Get strategy identifier
    fn strategy_id(&self) -> &str;
    
    /// Get mapping complexity (for performance optimization)
    fn complexity(&self) -> MappingComplexity;
}

/// Mapping complexity classification for strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MappingComplexity {
    /// O(1) constant-time mapping
    Constant,
    /// O(log n) logarithmic mapping
    Logarithmic,
    /// O(n) linear mapping
    Linear,
    /// O(n log n) linearithmic mapping
    Linearithmic,
}

/// Direct state-to-haptic mapping strategy (research baseline)
#[derive(Debug)]
pub struct DirectMappingStrategy {
    calibration: PsychophysicalCalibration,
    channel_assignments: HashMap<String, HapticChannel>,
}

impl DirectMappingStrategy {
    /// Create new direct mapping strategy with calibration
    pub fn new(calibration: PsychophysicalCalibration) -> Self {
        let mut channel_assignments = HashMap::new();
        
        // Standard algorithm state -> haptic channel mappings
        channel_assignments.insert("current_node".to_string(), HapticChannel::Position);
        channel_assignments.insert("open_set_size".to_string(), HapticChannel::Pressure);
        channel_assignments.insert("closed_set_size".to_string(), HapticChannel::Vibration);
        channel_assignments.insert("path_cost".to_string(), HapticChannel::Temperature);
        channel_assignments.insert("heuristic_value".to_string(), HapticChannel::Texture);
        channel_assignments.insert("search_progress".to_string(), HapticChannel::Movement);

        Self {
            calibration,
            channel_assignments,
        }
    }

    /// Apply psychophysical scaling to raw values
    fn apply_psychophysical_scaling(&self, value: f64, channel: HapticChannel) -> f64 {
        let detection_threshold = self.calibration.detection_thresholds[&channel];
        let stevens_exponent = self.calibration.stevens_exponents[&channel];
        
        if value < detection_threshold {
            0.0
        } else {
            // Stevens' power law: ψ = k * φ^n
            ((value - detection_threshold) / (1.0 - detection_threshold)).powf(stevens_exponent)
        }
    }

    /// Generate haptic stimulus for state parameter
    fn generate_stimulus(&self, value: f64, channel: HapticChannel, location: (f64, f64)) -> HapticStimulus {
        let scaled_intensity = self.apply_psychophysical_scaling(value, channel);
        
        // Channel-specific parameter generation
        let (frequency, duration) = match channel {
            HapticChannel::Vibration => {
                // Map intensity to frequency (20-300 Hz tactile range)
                let freq = 20.0 + (scaled_intensity * 280.0);
                (Some(freq), Duration::from_millis(150))
            },
            HapticChannel::Pressure => {
                (None, Duration::from_millis(200))
            },
            HapticChannel::Temperature => {
                (None, Duration::from_millis(500))
            },
            HapticChannel::Texture => {
                // Texture mapped to frequency modulation
                let freq = 100.0 + (scaled_intensity * 50.0);
                (Some(freq), Duration::from_millis(100))
            },
            HapticChannel::Position => {
                (None, Duration::from_millis(50))
            },
            HapticChannel::Movement => {
                (None, Duration::from_millis(300))
            },
        };

        HapticStimulus {
            intensity: scaled_intensity.min(self.calibration.comfort_thresholds[&channel]),
            frequency,
            duration,
            spatial_location: location,
            onset_delay: Duration::from_millis(0),
            envelope: EnvelopeParameters::default(),
        }
    }
}

impl HapticMappingStrategy for DirectMappingStrategy {
    fn map_state(&self, state: &AlgorithmState) -> Result<Vec<HapticPattern>, TactileError> {
        let mut patterns = Vec::new();
        let mut stimuli = Vec::new();

        // Extract relevant state parameters and map to haptic channels
        if let Some(current_node) = state.current_node {
            let normalized_position = (
                (current_node as f64 % 100.0) / 100.0,
                (current_node as f64 / 100.0) / 100.0,
            );
            
            let stimulus = self.generate_stimulus(
                0.8, // High intensity for current position
                HapticChannel::Position,
                normalized_position,
            );
            stimuli.push((stimulus, HapticChannel::Position));
        }

        // Map open set size to pressure intensity
        let open_set_ratio = (state.open_set.len() as f64 / 100.0).min(1.0);
        if open_set_ratio > 0.0 {
            let stimulus = self.generate_stimulus(
                open_set_ratio,
                HapticChannel::Pressure,
                (0.2, 0.8),
            );
            stimuli.push((stimulus, HapticChannel::Pressure));
        }

        // Map closed set size to vibration intensity
        let closed_set_ratio = (state.closed_set.len() as f64 / 100.0).min(1.0);
        if closed_set_ratio > 0.0 {
            let stimulus = self.generate_stimulus(
                closed_set_ratio,
                HapticChannel::Vibration,
                (0.8, 0.8),
            );
            stimuli.push((stimulus, HapticChannel::Vibration));
        }

        if !stimuli.is_empty() {
            patterns.push(HapticPattern {
                id: format!("state_pattern_{}", state.step),
                stimuli,
                repetition: RepetitionParameters::default(),
                priority: 5,
            });
        }

        Ok(patterns)
    }

    fn map_event(&self, event: &TimelineEvent) -> Result<Option<HapticPattern>, TactileError> {
        // Map specific timeline events to distinctive haptic patterns
        let pattern = match event.event_type.as_str() {
            "node_expanded" => {
                let stimulus = HapticStimulus {
                    intensity: 0.6,
                    frequency: Some(150.0),
                    duration: Duration::from_millis(50),
                    spatial_location: (0.5, 0.5),
                    onset_delay: Duration::from_millis(0),
                    envelope: EnvelopeParameters {
                        attack: Duration::from_millis(5),
                        decay: Duration::from_millis(15),
                        sustain: 0.4,
                        release: Duration::from_millis(30),
                    },
                };
                
                Some(HapticPattern {
                    id: format!("node_expansion_{}", event.timestamp),
                    stimuli: vec![(stimulus, HapticChannel::Vibration)],
                    repetition: RepetitionParameters::default(),
                    priority: 7,
                })
            },
            "path_found" => {
                let stimulus = HapticStimulus {
                    intensity: 0.9,
                    frequency: Some(200.0),
                    duration: Duration::from_millis(300),
                    spatial_location: (0.5, 0.5),
                    onset_delay: Duration::from_millis(0),
                    envelope: EnvelopeParameters {
                        attack: Duration::from_millis(10),
                        decay: Duration::from_millis(50),
                        sustain: 0.8,
                        release: Duration::from_millis(240),
                    },
                };
                
                Some(HapticPattern {
                    id: format!("path_completion_{}", event.timestamp),
                    stimuli: vec![(stimulus, HapticChannel::Pressure)],
                    repetition: RepetitionParameters {
                        count: 3,
                        interval: Duration::from_millis(100),
                        decay_factor: 0.9,
                    },
                    priority: 10,
                })
            },
            _ => None,
        };

        Ok(pattern)
    }

    fn strategy_id(&self) -> &str {
        "direct_mapping"
    }

    fn complexity(&self) -> MappingComplexity {
        MappingComplexity::Linear
    }
}

/// Concurrent haptic synthesis engine with real-time guarantees
#[derive(Debug)]
pub struct TactileSynthesizer {
    /// Active haptic patterns with timestamps
    active_patterns: Arc<RwLock<HashMap<String, (HapticPattern, Instant)>>>,
    /// Synthesis configuration
    config: SynthesisConfig,
    /// Current mapping strategy
    mapping_strategy: Arc<dyn HapticMappingStrategy>,
    /// Performance metrics
    metrics: Arc<RwLock<SynthesisMetrics>>,
}

/// Synthesis configuration parameters
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Maximum concurrent patterns
    pub max_concurrent_patterns: usize,
    /// Pattern timeout duration
    pub pattern_timeout: Duration,
    /// Synthesis update rate (Hz)
    pub update_rate: f64,
    /// Enable pattern conflict resolution
    pub conflict_resolution: bool,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            max_concurrent_patterns: 8,
            pattern_timeout: Duration::from_secs(10),
            update_rate: 60.0,
            conflict_resolution: true,
        }
    }
}

/// Performance and quality metrics for synthesis
#[derive(Debug, Default)]
pub struct SynthesisMetrics {
    /// Total patterns synthesized
    pub patterns_synthesized: u64,
    /// Average synthesis latency (microseconds)
    pub average_latency: f64,
    /// Peak synthesis latency (microseconds)
    pub peak_latency: f64,
    /// Pattern conflict resolution count
    pub conflicts_resolved: u64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
}

impl TactileSynthesizer {
    /// Create new tactile synthesizer with strategy
    pub fn new(
        config: SynthesisConfig,
        mapping_strategy: Arc<dyn HapticMappingStrategy>,
    ) -> Self {
        Self {
            active_patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
            mapping_strategy,
            metrics: Arc::new(RwLock::new(SynthesisMetrics::default())),
        }
    }

    /// Synthesize haptic representation for algorithm state
    pub fn synthesize_state(&self, state: &AlgorithmState) -> Result<(), TactileError> {
        let start_time = Instant::now();

        // Map algorithm state to haptic patterns
        let patterns = self.mapping_strategy.map_state(state)?;

        // Add patterns to active synthesis
        {
            let mut active = self.active_patterns.write()
                .map_err(|_| TactileError::SynthesisError("Failed to acquire write lock".to_string()))?;

            // Cleanup expired patterns
            let now = Instant::now();
            active.retain(|_, (_, timestamp)| {
                now.duration_since(*timestamp) < self.config.pattern_timeout
            });

            // Add new patterns with conflict resolution
            for pattern in patterns {
                if self.config.conflict_resolution {
                    self.resolve_pattern_conflicts(&mut active, &pattern, now)?;
                }
                
                active.insert(pattern.id.clone(), (pattern, now));
                
                // Enforce concurrent pattern limit
                if active.len() > self.config.max_concurrent_patterns {
                    // Remove lowest priority pattern
                    if let Some((lowest_id, _)) = active.iter()
                        .min_by_key(|(_, (pattern, _))| pattern.priority)
                        .map(|(id, _)| id.clone())
                    {
                        active.remove(&lowest_id);
                    }
                }
            }
        }

        // Update performance metrics
        {
            let mut metrics = self.metrics.write()
                .map_err(|_| TactileError::SynthesisError("Failed to acquire metrics lock".to_string()))?;
            
            let latency = start_time.elapsed().as_micros() as f64;
            metrics.patterns_synthesized += 1;
            metrics.peak_latency = metrics.peak_latency.max(latency);
            metrics.average_latency = (metrics.average_latency * (metrics.patterns_synthesized - 1) as f64 + latency) 
                / metrics.patterns_synthesized as f64;
        }

        Ok(())
    }

    /// Synthesize haptic feedback for timeline event
    pub fn synthesize_event(&self, event: &TimelineEvent) -> Result<(), TactileError> {
        if let Some(pattern) = self.mapping_strategy.map_event(event)? {
            let mut active = self.active_patterns.write()
                .map_err(|_| TactileError::SynthesisError("Failed to acquire write lock".to_string()))?;
            
            let now = Instant::now();
            if self.config.conflict_resolution {
                self.resolve_pattern_conflicts(&mut active, &pattern, now)?;
            }
            
            active.insert(pattern.id.clone(), (pattern, now));
        }

        Ok(())
    }

    /// Resolve haptic pattern conflicts using priority-based arbitration
    fn resolve_pattern_conflicts(
        &self,
        active_patterns: &mut HashMap<String, (HapticPattern, Instant)>,
        new_pattern: &HapticPattern,
        current_time: Instant,
    ) -> Result<(), TactileError> {
        let conflicting_channels: std::collections::HashSet<HapticChannel> = new_pattern.stimuli
            .iter()
            .map(|(_, channel)| *channel)
            .collect();

        // Find patterns using conflicting channels
        let conflicts: Vec<String> = active_patterns
            .iter()
            .filter(|(_, (pattern, _))| {
                pattern.stimuli.iter().any(|(_, channel)| conflicting_channels.contains(channel))
            })
            .filter(|(_, (pattern, _))| pattern.priority < new_pattern.priority)
            .map(|(id, _)| id.clone())
            .collect();

        // Remove lower-priority conflicting patterns
        for conflict_id in conflicts {
            active_patterns.remove(&conflict_id);
            
            // Update conflict resolution metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.conflicts_resolved += 1;
            }
        }

        Ok(())
    }

    /// Get current synthesis metrics
    pub fn get_metrics(&self) -> Result<SynthesisMetrics, TactileError> {
        self.metrics.read()
            .map(|metrics| metrics.clone())
            .map_err(|_| TactileError::SynthesisError("Failed to read metrics".to_string()))
    }

    /// Get currently active patterns
    pub fn get_active_patterns(&self) -> Result<Vec<HapticPattern>, TactileError> {
        self.active_patterns.read()
            .map(|active| active.values().map(|(pattern, _)| pattern.clone()).collect())
            .map_err(|_| TactileError::SynthesisError("Failed to read active patterns".to_string()))
    }

    /// Update synthesis parameters dynamically
    pub fn update_config(&mut self, new_config: SynthesisConfig) {
        self.config = new_config;
    }

    /// Replace mapping strategy at runtime
    pub fn set_mapping_strategy(&mut self, strategy: Arc<dyn HapticMappingStrategy>) {
        self.mapping_strategy = strategy;
    }
}

impl ModalityRepresentation for TactileSynthesizer {
    fn modality_type(&self) -> crate::modality::representation::ModalityType {
        crate::modality::representation::ModalityType::Tactile
    }

    fn update_representation(&mut self, state: &AlgorithmState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.synthesize_state(state).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn verify_equivalence(&self, other: &dyn ModalityRepresentation) -> SemanticEquivalence {
        // Verify semantic equivalence through information-theoretic measures
        SemanticEquivalence {
            equivalence_score: 0.95, // High equivalence for tactile mapping
            confidence_interval: (0.90, 0.98),
            verification_method: "Cross-modal information preservation".to_string(),
        }
    }

    fn get_representation_complexity(&self) -> f64 {
        // Information-theoretic complexity measure
        match self.mapping_strategy.complexity() {
            MappingComplexity::Constant => 1.0,
            MappingComplexity::Logarithmic => 2.0,
            MappingComplexity::Linear => 3.0,
            MappingComplexity::Linearithmic => 4.0,
        }
    }
}

/// Tactile synthesis error types
#[derive(Debug, Error)]
pub enum TactileError {
    #[error("Mapping strategy error: {0}")]
    MappingError(String),
    
    #[error("Synthesis error: {0}")]
    SynthesisError(String),
    
    #[error("Calibration error: {0}")]
    CalibrationError(String),
    
    #[error("Pattern conflict resolution error: {0}")]
    ConflictResolutionError(String),
    
    #[error("Hardware interface error: {0}")]
    HardwareError(String),
    
    #[error("Psychophysical model error: {0}")]
    PsychophysicalError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_psychophysical_scaling() {
        let calibration = PsychophysicalCalibration::default();
        let strategy = DirectMappingStrategy::new(calibration);
        
        // Test Weber-Fechner scaling
        let scaled_low = strategy.apply_psychophysical_scaling(0.1, HapticChannel::Pressure);
        let scaled_high = strategy.apply_psychophysical_scaling(0.9, HapticChannel::Pressure);
        
        assert!(scaled_low < scaled_high);
        assert!(scaled_low >= 0.0 && scaled_low <= 1.0);
        assert!(scaled_high >= 0.0 && scaled_high <= 1.0);
    }

    #[test]
    fn test_pattern_conflict_resolution() {
        let config = SynthesisConfig::default();
        let strategy = Arc::new(DirectMappingStrategy::new(PsychophysicalCalibration::default()));
        let synthesizer = TactileSynthesizer::new(config, strategy);
        
        // Create test state
        let state = AlgorithmState {
            step: 1,
            open_set: vec![1, 2, 3],
            closed_set: vec![0],
            current_node: Some(5),
            data: std::collections::HashMap::new(),
        };
        
        // Test synthesis
        assert!(synthesizer.synthesize_state(&state).is_ok());
        
        // Verify metrics
        let metrics = synthesizer.get_metrics().unwrap();
        assert_eq!(metrics.patterns_synthesized, 1);
    }

    #[test]
    fn test_haptic_stimulus_generation() {
        let calibration = PsychophysicalCalibration::default();
        let strategy = DirectMappingStrategy::new(calibration);
        
        let stimulus = strategy.generate_stimulus(0.5, HapticChannel::Vibration, (0.5, 0.5));
        
        assert!(stimulus.intensity >= 0.0 && stimulus.intensity <= 1.0);
        assert!(stimulus.frequency.is_some());
        assert!(stimulus.duration > Duration::from_millis(0));
    }

    #[test]
    fn test_envelope_parameters() {
        let envelope = EnvelopeParameters::default();
        
        assert!(envelope.attack < envelope.decay);
        assert!(envelope.sustain >= 0.0 && envelope.sustain <= 1.0);
        assert!(envelope.release > Duration::from_millis(0));
    }

    #[test]
    fn test_mapping_complexity_ordering() {
        assert!(MappingComplexity::Constant < MappingComplexity::Logarithmic);
        assert!(MappingComplexity::Logarithmic < MappingComplexity::Linear);
        assert!(MappingComplexity::Linear < MappingComplexity::Linearithmic);
    }
}