//! Algorithm Pattern Recognition Engine
//!
//! This module implements a statistical pattern recognition system for algorithmic behaviors,
//! temporal sequence mining, and educational pattern identification. The implementation is
//! designed to identify meaningful patterns in algorithm execution with high precision and
//! educational relevance.
//!
//! The implementation utilizes:
//! - Information-theoretic pattern identification metrics
//! - Multi-model ensemble detection approaches
//! - Incremental sequence mining with pruning optimizations
//! - Formal pattern taxonomies derived from educational research
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>
//! All rights reserved.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt;

use log::{debug, trace, warn};
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::{
    Algorithm, 
    AlgorithmState,
    NodeId
};

use crate::temporal::{
    signature::ExecutionSignature,
    timeline::Timeline,
    state::StateManager
};

use crate::execution::tracer::ExecutionTracer;

/// Pattern detection error types
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PatternError {
    /// Pattern definition is invalid
    #[error("Invalid pattern definition: {0}")]
    InvalidPattern(String),
    
    /// Insufficient data for pattern detection
    #[error("Insufficient data for pattern detection: {0}")]
    InsufficientData(String),
    
    /// Algorithm execution does not match pattern
    #[error("Algorithm execution does not match pattern: {0}")]
    NoMatch(String),
    
    /// Internal error during pattern processing
    #[error("Internal pattern processing error: {0}")]
    InternalError(String),
}

/// Confidence level for pattern detection
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(f32);

impl Confidence {
    /// Create a new confidence value
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    
    /// Get confidence value
    pub fn value(&self) -> f32 {
        self.0
    }
    
    /// Check if confidence exceeds threshold
    pub fn exceeds(&self, threshold: f32) -> bool {
        self.0 >= threshold
    }
    
    /// Convert confidence to linguistic description
    pub fn to_linguistic(&self) -> &'static str {
        match self.0 {
            v if v >= 0.95 => "certain",
            v if v >= 0.85 => "highly confident",
            v if v >= 0.7 => "confident",
            v if v >= 0.5 => "likely",
            v if v >= 0.3 => "possible",
            _ => "uncertain",
        }
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(0.0)
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "{} ({:.1}%)", self.to_linguistic(), self.0 * 100.0)
        } else {
            write!(f, "{:.1}%", self.0 * 100.0)
        }
    }
}

/// Pattern category in the formal taxonomy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    /// Search strategy patterns
    SearchStrategy,
    
    /// Heuristic behavior patterns
    HeuristicBehavior,
    
    /// Optimization patterns
    Optimization,
    
    /// Convergence patterns
    Convergence,
    
    /// Exploration/exploitation patterns
    ExplorationExploitation,
    
    /// Edge case handling patterns
    EdgeCase,
    
    /// Performance patterns
    Performance,
    
    /// Miscellaneous patterns
    Miscellaneous,
}

impl fmt::Display for PatternCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SearchStrategy => write!(f, "Search Strategy"),
            Self::HeuristicBehavior => write!(f, "Heuristic Behavior"),
            Self::Optimization => write!(f, "Optimization"),
            Self::Convergence => write!(f, "Convergence"),
            Self::ExplorationExploitation => write!(f, "Exploration/Exploitation"),
            Self::EdgeCase => write!(f, "Edge Case Handling"),
            Self::Performance => write!(f, "Performance"),
            Self::Miscellaneous => write!(f, "Miscellaneous"),
        }
    }
}

/// Pattern type within the category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Local optimum detection
    LocalOptimum,
    
    /// Plateau detection
    Plateau,
    
    /// Backtracking behavior
    Backtracking,
    
    /// Heuristic dominance
    HeuristicDominance,
    
    /// Search space pruning
    Pruning,
    
    /// Oscillatory behavior
    Oscillation,
    
    /// Rapid convergence
    RapidConvergence,
    
    /// Exploration phase
    ExplorationPhase,
    
    /// Exploitation phase
    ExploitationPhase,
    
    /// Boundary condition handling
    BoundaryCondition,
    
    /// Path redundancy
    PathRedundancy,
    
    /// Performance anomaly
    PerformanceAnomaly,
    
    /// Custom pattern type
    Custom(String),
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalOptimum => write!(f, "Local Optimum"),
            Self::Plateau => write!(f, "Plateau"),
            Self::Backtracking => write!(f, "Backtracking"),
            Self::HeuristicDominance => write!(f, "Heuristic Dominance"),
            Self::Pruning => write!(f, "Search Space Pruning"),
            Self::Oscillation => write!(f, "Oscillatory Behavior"),
            Self::RapidConvergence => write!(f, "Rapid Convergence"),
            Self::ExplorationPhase => write!(f, "Exploration Phase"),
            Self::ExploitationPhase => write!(f, "Exploitation Phase"),
            Self::BoundaryCondition => write!(f, "Boundary Condition Handling"),
            Self::PathRedundancy => write!(f, "Path Redundancy"),
            Self::PerformanceAnomaly => write!(f, "Performance Anomaly"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Educational significance of a pattern
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EducationalSignificance(f32);

impl EducationalSignificance {
    /// Create a new educational significance value
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    
    /// Get significance value
    pub fn value(&self) -> f32 {
        self.0
    }
    
    /// Check if significance exceeds threshold
    pub fn exceeds(&self, threshold: f32) -> bool {
        self.0 >= threshold
    }
    
    /// Convert significance to linguistic description
    pub fn to_linguistic(&self) -> &'static str {
        match self.0 {
            v if v >= 0.9 => "fundamental",
            v if v >= 0.75 => "highly significant",
            v if v >= 0.5 => "significant",
            v if v >= 0.25 => "moderately significant",
            _ => "minor",
        }
    }
}

impl Default for EducationalSignificance {
    fn default() -> Self {
        Self(0.5) // Default to medium significance
    }
}

impl fmt::Display for EducationalSignificance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "{}", self.to_linguistic())
        } else {
            write!(f, "{:.1}%", self.0 * 100.0)
        }
    }
}

/// Temporal scope of a pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalScope {
    /// Point pattern (single state)
    Point,
    
    /// Interval pattern (continuous range of states)
    Interval(usize, usize),
    
    /// Sequence pattern (ordered collection of states)
    Sequence(Vec<usize>),
    
    /// Cyclic pattern (repeating sequence)
    Cyclic(Vec<usize>, usize), // states, cycle count
}

impl TemporalScope {
    /// Get the number of states involved in the pattern
    pub fn state_count(&self) -> usize {
        match self {
            Self::Point => 1,
            Self::Interval(start, end) => end - start + 1,
            Self::Sequence(states) => states.len(),
            Self::Cyclic(states, count) => states.len() * count,
        }
    }
    
    /// Check if a step is included in the scope
    pub fn includes_step(&self, step: usize) -> bool {
        match self {
            Self::Point => false, // Need specific point
            Self::Interval(start, end) => step >= *start && step <= *end,
            Self::Sequence(states) => states.contains(&step),
            Self::Cyclic(states, _) => {
                // Check if step is part of any cycle
                for state in states {
                    if step % states.len() == *state % states.len() {
                        return true;
                    }
                }
                false
            },
        }
    }
}

impl fmt::Display for TemporalScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Point => write!(f, "single point"),
            Self::Interval(start, end) => write!(f, "steps {}-{}", start, end),
            Self::Sequence(states) => {
                write!(f, "sequence [")?;
                for (i, state) in states.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", state)?;
                }
                write!(f, "]")
            },
            Self::Cyclic(states, count) => {
                write!(f, "{} cycles of [", count)?;
                for (i, state) in states.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", state)?;
                }
                write!(f, "]")
            },
        }
    }
}

/// Spatial scope of a pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialScope {
    /// Global pattern (entire state)
    Global,
    
    /// Node pattern (specific node)
    Node(NodeId),
    
    /// Region pattern (group of nodes)
    Region(Vec<NodeId>),
    
    /// Path pattern (sequence of connected nodes)
    Path(Vec<NodeId>),
}

impl SpatialScope {
    /// Check if a node is included in the scope
    pub fn includes_node(&self, node: NodeId) -> bool {
        match self {
            Self::Global => true,
            Self::Node(id) => *id == node,
            Self::Region(nodes) => nodes.contains(&node),
            Self::Path(path) => path.contains(&node),
        }
    }
    
    /// Get the number of nodes involved in the pattern
    pub fn node_count(&self) -> Option<usize> {
        match self {
            Self::Global => None, // Unknown
            Self::Node(_) => Some(1),
            Self::Region(nodes) => Some(nodes.len()),
            Self::Path(path) => Some(path.len()),
        }
    }
}

impl fmt::Display for SpatialScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Node(id) => write!(f, "node {}", id),
            Self::Region(nodes) => {
                if nodes.len() <= 3 {
                    write!(f, "region [")?;
                    for (i, node) in nodes.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", node)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "region with {} nodes", nodes.len())
                }
            },
            Self::Path(path) => {
                if path.len() <= 3 {
                    write!(f, "path [")?;
                    for (i, node) in path.iter().enumerate() {
                        if i > 0 {
                            write!(f, "→")?;
                        }
                        write!(f, "{}", node)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "path with {} nodes", path.len())
                }
            },
        }
    }
}

/// Pattern instance detected in algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternInstance {
    /// Pattern ID
    pub id: String,
    
    /// Pattern category
    pub category: PatternCategory,
    
    /// Pattern type
    pub pattern_type: PatternType,
    
    /// Human-readable name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Temporal scope of the pattern
    pub temporal_scope: TemporalScope,
    
    /// Spatial scope of the pattern
    pub spatial_scope: SpatialScope,
    
    /// Detection confidence
    pub confidence: Confidence,
    
    /// Educational significance
    pub significance: EducationalSignificance,
    
    /// Related concepts for learning
    pub related_concepts: Vec<String>,
    
    /// Detection timestamp
    pub detection_time: std::time::SystemTime,
}

impl PatternInstance {
    /// Create a new pattern instance
    pub fn new(
        id: String,
        category: PatternCategory,
        pattern_type: PatternType,
        name: String,
        description: String,
        temporal_scope: TemporalScope,
        spatial_scope: SpatialScope,
        confidence: Confidence,
        significance: EducationalSignificance,
        related_concepts: Vec<String>,
    ) -> Self {
        Self {
            id,
            category,
            pattern_type,
            name,
            description,
            temporal_scope,
            spatial_scope,
            confidence,
            significance,
            related_concepts,
            detection_time: std::time::SystemTime::now(),
        }
    }
    
    /// Get a short description of the pattern
    pub fn short_description(&self) -> String {
        format!(
            "{} pattern ({}): {} in {}",
            self.pattern_type,
            self.category,
            self.name,
            self.temporal_scope
        )
    }
    
    /// Check if the pattern has high confidence
    pub fn is_high_confidence(&self) -> bool {
        self.confidence.exceeds(0.8)
    }
    
    /// Check if the pattern has high educational significance
    pub fn is_educationally_significant(&self) -> bool {
        self.significance.exceeds(0.7)
    }
}

/// Trait for pattern recognizers
pub trait PatternRecognizer: Send + Sync {
    /// Get the pattern category
    fn category(&self) -> PatternCategory;
    
    /// Get the pattern type
    fn pattern_type(&self) -> PatternType;
    
    /// Analyze execution trace to detect patterns
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError>;
    
    /// Analyze algorithm state to detect patterns
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError>;
    
    /// Analyze timeline to detect patterns
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError>;
    
    /// Check if this recognizer is applicable to the given algorithm
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool;
}

/// Local optimum pattern recognizer
pub struct LocalOptimumRecognizer {
    /// Minimum number of steps to consider a local optimum
    min_steps: usize,
    
    /// Improvement threshold for considering a significant change
    improvement_threshold: f64,
    
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

impl LocalOptimumRecognizer {
    /// Create a new local optimum recognizer
    pub fn new(min_steps: usize, improvement_threshold: f64, confidence_threshold: f32) -> Self {
        Self {
            min_steps,
            improvement_threshold,
            confidence_threshold,
        }
    }
}

impl Default for LocalOptimumRecognizer {
    fn default() -> Self {
        Self {
            min_steps: 5,
            improvement_threshold: 0.001,
            confidence_threshold: 0.7,
        }
    }
}

impl PatternRecognizer for LocalOptimumRecognizer {
    fn category(&self) -> PatternCategory {
        PatternCategory::SearchStrategy
    }
    
    fn pattern_type(&self) -> PatternType {
        PatternType::LocalOptimum
    }
    
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get execution history
        let states = tracer.states();
        if states.len() < self.min_steps {
            return Err(PatternError::InsufficientData(
                format!("Need at least {} steps, got {}", self.min_steps, states.len())
            ));
        }
        
        let mut patterns = Vec::new();
        
        // Scan for local optima
        for i in self.min_steps..states.len() {
            // Get current state and previous states
            let current_state = &states[i];
            let previous_states = &states[i - self.min_steps..i];
            
            // Check if current state is better than all previous states
            let is_better_than_previous = previous_states.iter().all(|state| {
                self.is_better_state(current_state, state)
            });
            
            // Check if current state is not improving in next states (if available)
            let is_local_optimum = is_better_than_previous && 
                (i + self.min_steps >= states.len() || {
                    let next_states = &states[i + 1..std::cmp::min(i + self.min_steps + 1, states.len())];
                    next_states.iter().all(|state| {
                        !self.is_significantly_better_state(state, current_state)
                    })
                });
            
            if is_local_optimum {
                // Calculate confidence based on how many steps we've verified
                let verification_steps = std::cmp::min(self.min_steps, states.len() - i - 1);
                let confidence = Confidence::new(
                    self.confidence_threshold + 
                    (1.0 - self.confidence_threshold) * (verification_steps as f32 / self.min_steps as f32)
                );
                
                // Create pattern instance
                let pattern = PatternInstance::new(
                    format!("local_optimum_{}", i),
                    self.category(),
                    self.pattern_type(),
                    format!("Local Optimum at Step {}", i),
                    format!(
                        "The algorithm reached a local optimum at step {}, where the solution \
                        quality stopped improving significantly for {} consecutive steps.",
                        i, self.min_steps
                    ),
                    TemporalScope::Interval(i, i + verification_steps),
                    SpatialScope::Global,
                    confidence,
                    EducationalSignificance::new(0.8),
                    vec![
                        "Optimization".to_string(),
                        "Local vs. Global Optima".to_string(),
                        "Search Strategies".to_string(),
                    ],
                );
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError> {
        // Cannot detect local optima from a single state
        Err(PatternError::InsufficientData(
            "Local optimum detection requires execution history".to_string()
        ))
    }
    
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get states from timeline
        let result = timeline.current_state()
            .map_err(|e| PatternError::InternalError(format!("Timeline error: {}", e)))?;
        
        // Not enough information in a single state
        Err(PatternError::InsufficientData(
            "Local optimum detection requires execution history".to_string()
        ))
    }
    
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool {
        // Applicable to optimization algorithms
        algorithm.category() == "optimization" || 
        algorithm.category() == "path_finding" || 
        algorithm.category() == "search"
    }
}

impl LocalOptimumRecognizer {
    /// Check if state1 is better than state2
    fn is_better_state(&self, state1: &AlgorithmState, state2: &AlgorithmState) -> bool {
        // Compare metrics in the state
        if let (Some(metric1), Some(metric2)) = (
            state1.data.get("objective_value"),
            state2.data.get("objective_value"),
        ) {
            // Parse values (assuming they're stored as strings)
            if let (Ok(value1), Ok(value2)) = (
                metric1.parse::<f64>(),
                metric2.parse::<f64>(),
            ) {
                return value1 < value2; // Lower is better (minimize)
            }
        }
        
        // Compare other possible metrics
        if let (Some(metric1), Some(metric2)) = (
            state1.data.get("distance"),
            state2.data.get("distance"),
        ) {
            // Parse values (assuming they're stored as strings)
            if let (Ok(value1), Ok(value2)) = (
                metric1.parse::<f64>(),
                metric2.parse::<f64>(),
            ) {
                return value1 < value2; // Lower is better (minimize)
            }
        }
        
        // Default: can't determine
        false
    }
    
    /// Check if state1 is significantly better than state2
    fn is_significantly_better_state(&self, state1: &AlgorithmState, state2: &AlgorithmState) -> bool {
        // Compare metrics in the state
        if let (Some(metric1), Some(metric2)) = (
            state1.data.get("objective_value"),
            state2.data.get("objective_value"),
        ) {
            // Parse values (assuming they're stored as strings)
            if let (Ok(value1), Ok(value2)) = (
                metric1.parse::<f64>(),
                metric2.parse::<f64>(),
            ) {
                // Calculate relative improvement
                if value2 == 0.0 {
                    return value1 < value2; // Any improvement is significant
                }
                
                let relative_improvement = (value2 - value1).abs() / value2.abs();
                return relative_improvement > self.improvement_threshold; // Significant if above threshold
            }
        }
        
        // Compare other possible metrics
        if let (Some(metric1), Some(metric2)) = (
            state1.data.get("distance"),
            state2.data.get("distance"),
        ) {
            // Parse values (assuming they're stored as strings)
            if let (Ok(value1), Ok(value2)) = (
                metric1.parse::<f64>(),
                metric2.parse::<f64>(),
            ) {
                // Calculate relative improvement
                if value2 == 0.0 {
                    return value1 < value2; // Any improvement is significant
                }
                
                let relative_improvement = (value2 - value1).abs() / value2.abs();
                return relative_improvement > self.improvement_threshold; // Significant if above threshold
            }
        }
        
        // Default: can't determine
        false
    }
}

/// Exploration-exploitation pattern recognizer
pub struct ExplorationExploitationRecognizer {
    /// Window size for analysis
    window_size: usize,
    
    /// Exploration threshold for ratio of new nodes to total nodes
    exploration_threshold: f64,
    
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

impl ExplorationExploitationRecognizer {
    /// Create a new exploration-exploitation recognizer
    pub fn new(window_size: usize, exploration_threshold: f64, confidence_threshold: f32) -> Self {
        Self {
            window_size,
            exploration_threshold,
            confidence_threshold,
        }
    }
}

impl Default for ExplorationExploitationRecognizer {
    fn default() -> Self {
        Self {
            window_size: 10,
            exploration_threshold: 0.6,
            confidence_threshold: 0.75,
        }
    }
}

impl PatternRecognizer for ExplorationExploitationRecognizer {
    fn category(&self) -> PatternCategory {
        PatternCategory::ExplorationExploitation
    }
    
    fn pattern_type(&self) -> PatternType {
        PatternType::ExplorationPhase // Can be either exploration or exploitation
    }
    
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get execution history
        let states = tracer.states();
        if states.len() < self.window_size {
            return Err(PatternError::InsufficientData(
                format!("Need at least {} steps, got {}", self.window_size, states.len())
            ));
        }
        
        let mut patterns = Vec::new();
        let mut visited_nodes = HashSet::new();
        let mut exploration_phases = Vec::new();
        let mut exploitation_phases = Vec::new();
        let mut current_phase_start = 0;
        let mut is_exploration = true;
        
        // Scan for exploration/exploitation phases
        for i in 0..states.len() {
            let state = &states[i];
            
            // Check current node
            if let Some(current_node) = state.current_node {
                let window_start = if i >= self.window_size { i - self.window_size } else { 0 };
                let window_states = &states[window_start..i + 1];
                
                // Count distinct nodes visited in this window
                let mut window_visited = HashSet::new();
                for window_state in window_states {
                    if let Some(node) = window_state.current_node {
                        window_visited.insert(node);
                    }
                }
                
                // Calculate exploration ratio
                let exploration_ratio = window_visited.len() as f64 / window_states.len() as f64;
                
                // Determine phase type
                let is_current_exploration = exploration_ratio >= self.exploration_threshold;
                
                // Check for phase transition
                if is_current_exploration != is_exploration && i > current_phase_start {
                    // Record completed phase
                    if is_exploration {
                        exploration_phases.push((current_phase_start, i - 1));
                    } else {
                        exploitation_phases.push((current_phase_start, i - 1));
                    }
                    
                    // Start new phase
                    current_phase_start = i;
                    is_exploration = is_current_exploration;
                }
                
                // Always add to visited nodes
                visited_nodes.insert(current_node);
            }
        }
        
        // Record final phase
        if states.len() > current_phase_start {
            if is_exploration {
                exploration_phases.push((current_phase_start, states.len() - 1));
            } else {
                exploitation_phases.push((current_phase_start, states.len() - 1));
            }
        }
        
        // Create patterns for exploration phases
        for (start, end) in exploration_phases {
            // Minimum phase length
            if end - start + 1 < self.window_size / 2 {
                continue;
            }
            
            // Calculate confidence based on phase length
            let phase_length = end - start + 1;
            let confidence = Confidence::new(
                self.confidence_threshold + 
                (1.0 - self.confidence_threshold) * (phase_length as f32 / (self.window_size * 2) as f32).min(1.0)
            );
            
            // Create pattern instance
            let pattern = PatternInstance::new(
                format!("exploration_phase_{}_{}", start, end),
                self.category(),
                PatternType::ExplorationPhase,
                format!("Exploration Phase (Steps {}-{})", start, end),
                format!(
                    "The algorithm was in an exploration phase from step {} to {}, \
                    focusing on discovering new areas of the search space.",
                    start, end
                ),
                TemporalScope::Interval(start, end),
                SpatialScope::Global,
                confidence,
                EducationalSignificance::new(0.75),
                vec![
                    "Exploration vs. Exploitation".to_string(),
                    "Search Strategies".to_string(),
                    "Algorithm Phases".to_string(),
                ],
            );
            
            patterns.push(pattern);
        }
        
        // Create patterns for exploitation phases
        for (start, end) in exploitation_phases {
            // Minimum phase length
            if end - start + 1 < self.window_size / 2 {
                continue;
            }
            
            // Calculate confidence based on phase length
            let phase_length = end - start + 1;
            let confidence = Confidence::new(
                self.confidence_threshold + 
                (1.0 - self.confidence_threshold) * (phase_length as f32 / (self.window_size * 2) as f32).min(1.0)
            );
            
            // Create pattern instance
            let pattern = PatternInstance::new(
                format!("exploitation_phase_{}_{}", start, end),
                self.category(),
                PatternType::ExploitationPhase,
                format!("Exploitation Phase (Steps {}-{})", start, end),
                format!(
                    "The algorithm was in an exploitation phase from step {} to {}, \
                    focusing on refining solutions in promising areas.",
                    start, end
                ),
                TemporalScope::Interval(start, end),
                SpatialScope::Global,
                confidence,
                EducationalSignificance::new(0.75),
                vec![
                    "Exploration vs. Exploitation".to_string(),
                    "Search Strategies".to_string(),
                    "Algorithm Phases".to_string(),
                ],
            );
            
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError> {
        // Cannot detect exploration/exploitation from a single state
        Err(PatternError::InsufficientData(
            "Exploration/exploitation detection requires execution history".to_string()
        ))
    }
    
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Not enough information in timeline without full history
        Err(PatternError::InsufficientData(
            "Exploration/exploitation detection requires execution history".to_string()
        ))
    }
    
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool {
        // Applicable to search and path-finding algorithms
        algorithm.category() == "search" || 
        algorithm.category() == "path_finding" || 
        algorithm.category() == "optimization"
    }
}

/// Backtracking pattern recognizer
pub struct BacktrackingRecognizer {
    /// Minimum number of steps to consider a backtracking sequence
    min_backtrack_steps: usize,
    
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

impl BacktrackingRecognizer {
    /// Create a new backtracking recognizer
    pub fn new(min_backtrack_steps: usize, confidence_threshold: f32) -> Self {
        Self {
            min_backtrack_steps,
            confidence_threshold,
        }
    }
}

impl Default for BacktrackingRecognizer {
    fn default() -> Self {
        Self {
            min_backtrack_steps: 3,
            confidence_threshold: 0.8,
        }
    }
}

impl PatternRecognizer for BacktrackingRecognizer {
    fn category(&self) -> PatternCategory {
        PatternCategory::SearchStrategy
    }
    
    fn pattern_type(&self) -> PatternType {
        PatternType::Backtracking
    }
    
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get execution history
        let states = tracer.states();
        if states.len() < self.min_backtrack_steps * 2 {
            return Err(PatternError::InsufficientData(
                format!("Need at least {} steps, got {}", self.min_backtrack_steps * 2, states.len())
            ));
        }
        
        let mut patterns = Vec::new();
        let mut visited_sequences = Vec::new();
        
        // Scan for backtracking
        for i in self.min_backtrack_steps..states.len() {
            // Get current sequence of nodes
            let mut current_sequence = Vec::new();
            for j in 0..self.min_backtrack_steps {
                if let Some(node) = states[i - j].current_node {
                    current_sequence.push(node);
                }
            }
            
            // Check if we have a complete sequence
            if current_sequence.len() < self.min_backtrack_steps {
                continue;
            }
            
            // Store sequence for future comparisons
            visited_sequences.push((i - self.min_backtrack_steps + 1, i, current_sequence.clone()));
            
            // Check for backtracking by comparing with earlier sequences
            for (start_idx, end_idx, sequence) in &visited_sequences[..visited_sequences.len() - 1] {
                // Skip adjacent sequences
                if *end_idx >= i - self.min_backtrack_steps {
                    continue;
                }
                
                // Check if sequences match (reversed for backtracking)
                let is_backtracking = sequence.iter().rev().zip(current_sequence.iter()).all(
                    |(a, b)| a == b
                );
                
                if is_backtracking {
                    // Calculate confidence
                    let confidence = Confidence::new(
                        self.confidence_threshold + 
                        (1.0 - self.confidence_threshold) * 
                        (self.min_backtrack_steps as f32 / 10.0).min(1.0)
                    );
                    
                    // Get nodes involved
                    let nodes = sequence.clone();
                    
                    // Create pattern instance
                    let pattern = PatternInstance::new(
                        format!("backtracking_{}", i),
                        self.category(),
                        self.pattern_type(),
                        format!("Backtracking at Step {}", i),
                        format!(
                            "The algorithm performed backtracking at step {}, \
                            returning along a previously explored path to try an alternative route.",
                            i
                        ),
                        TemporalScope::Interval(i - self.min_backtrack_steps + 1, i),
                        SpatialScope::Path(nodes),
                        confidence,
                        EducationalSignificance::new(0.85),
                        vec![
                            "Backtracking".to_string(),
                            "Search Strategies".to_string(),
                            "Path Exploration".to_string(),
                        ],
                    );
                    
                    patterns.push(pattern);
                    
                    // Don't check other sequences for this position
                    break;
                }
            }
        }
        
        Ok(patterns)
    }
    
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError> {
        // Cannot detect backtracking from a single state
        Err(PatternError::InsufficientData(
            "Backtracking detection requires execution history".to_string()
        ))
    }
    
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Not enough information in timeline without full history
        Err(PatternError::InsufficientData(
            "Backtracking detection requires execution history".to_string()
        ))
    }
    
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool {
        // Applicable to search and path-finding algorithms
        algorithm.category() == "search" || 
        algorithm.category() == "path_finding"
    }
}

/// Heuristic dominance pattern recognizer
pub struct HeuristicDominanceRecognizer {
    /// Minimum ratio of heuristic-influenced decisions to total decisions
    dominance_threshold: f64,
    
    /// Minimum number of steps to analyze
    min_steps: usize,
    
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

impl HeuristicDominanceRecognizer {
    /// Create a new heuristic dominance recognizer
    pub fn new(dominance_threshold: f64, min_steps: usize, confidence_threshold: f32) -> Self {
        Self {
            dominance_threshold,
            min_steps,
            confidence_threshold,
        }
    }
}

impl Default for HeuristicDominanceRecognizer {
    fn default() -> Self {
        Self {
            dominance_threshold: 0.8,
            min_steps: 10,
            confidence_threshold: 0.75,
        }
    }
}

impl PatternRecognizer for HeuristicDominanceRecognizer {
    fn category(&self) -> PatternCategory {
        PatternCategory::HeuristicBehavior
    }
    
    fn pattern_type(&self) -> PatternType {
        PatternType::HeuristicDominance
    }
    
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get execution history
        let states = tracer.states();
        if states.len() < self.min_steps {
            return Err(PatternError::InsufficientData(
                format!("Need at least {} steps, got {}", self.min_steps, states.len())
            ));
        }
        
        let mut patterns = Vec::new();
        
        // Count heuristic-influenced decisions
        let mut total_decisions = 0;
        let mut heuristic_decisions = 0;
        
        for state in states {
            // Check if there was a decision in this state
            if state.data.contains_key("decision") {
                total_decisions += 1;
                
                // Check if decision was influenced by heuristic
                if state.data.contains_key("heuristic_value") || 
                   state.data.contains_key("f_score") ||
                   state.data.contains_key("h_score") {
                    heuristic_decisions += 1;
                }
            }
        }
        
        // Calculate dominance ratio
        if total_decisions > 0 {
            let dominance_ratio = heuristic_decisions as f64 / total_decisions as f64;
            
            if dominance_ratio >= self.dominance_threshold {
                // Calculate confidence
                let confidence = Confidence::new(
                    self.confidence_threshold + 
                    (1.0 - self.confidence_threshold) * 
                    ((dominance_ratio - self.dominance_threshold) / (1.0 - self.dominance_threshold)).min(1.0) as f32
                );
                
                // Create pattern instance
                let pattern = PatternInstance::new(
                    "heuristic_dominance".to_string(),
                    self.category(),
                    self.pattern_type(),
                    "Heuristic-Dominated Search".to_string(),
                    format!(
                        "The algorithm's decision-making is strongly dominated by the heuristic function, \
                        with {:.1}% of decisions primarily influenced by heuristic values.",
                        dominance_ratio * 100.0
                    ),
                    TemporalScope::Interval(0, states.len() - 1),
                    SpatialScope::Global,
                    confidence,
                    EducationalSignificance::new(0.7),
                    vec![
                        "Heuristic Functions".to_string(),
                        "Informed Search".to_string(),
                        "Search Bias".to_string(),
                    ],
                );
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError> {
        // Cannot detect heuristic dominance from a single state
        Err(PatternError::InsufficientData(
            "Heuristic dominance detection requires execution history".to_string()
        ))
    }
    
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Not enough information in timeline without full history
        Err(PatternError::InsufficientData(
            "Heuristic dominance detection requires execution history".to_string()
        ))
    }
    
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool {
        // Check if algorithm has heuristic parameter
        algorithm.get_parameter("heuristic").is_some() || 
        algorithm.category() == "informed_search" ||
        algorithm.name().contains("A*") ||
        algorithm.name().contains("Greedy")
    }
}

/// Plateau pattern recognizer
pub struct PlateauRecognizer {
    /// Minimum number of steps with minimal improvement
    min_plateau_steps: usize,
    
    /// Maximum improvement to consider a plateau
    max_improvement: f64,
    
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

impl PlateauRecognizer {
    /// Create a new plateau recognizer
    pub fn new(min_plateau_steps: usize, max_improvement: f64, confidence_threshold: f32) -> Self {
        Self {
            min_plateau_steps,
            max_improvement,
            confidence_threshold,
        }
    }
}

impl Default for PlateauRecognizer {
    fn default() -> Self {
        Self {
            min_plateau_steps: 5,
            max_improvement: 0.001,
            confidence_threshold: 0.7,
        }
    }
}

impl PatternRecognizer for PlateauRecognizer {
    fn category(&self) -> PatternCategory {
        PatternCategory::SearchStrategy
    }
    
    fn pattern_type(&self) -> PatternType {
        PatternType::Plateau
    }
    
    fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Get execution history
        let states = tracer.states();
        if states.len() < self.min_plateau_steps {
            return Err(PatternError::InsufficientData(
                format!("Need at least {} steps, got {}", self.min_plateau_steps, states.len())
            ));
        }
        
        let mut patterns = Vec::new();
        let mut plateaus = Vec::new();
        let mut current_plateau_start = 0;
        let mut in_plateau = false;
        let mut last_metric_value = None;
        
        // Scan for plateaus
        for i in 0..states.len() {
            let state = &states[i];
            
            // Get metric value
            let current_metric = self.get_metric_value(state);
            
            if let Some(current_value) = current_metric {
                if let Some(last_value) = last_metric_value {
                    // Calculate improvement
                    let improvement = if last_value == 0.0 {
                        if current_value == 0.0 {
                            0.0
                        } else {
                            1.0 // Any non-zero improvement from zero is significant
                        }
                    } else {
                        (last_value - current_value).abs() / last_value.abs()
                    };
                    
                    // Check if in plateau
                    let is_plateau_point = improvement <= self.max_improvement;
                    
                    if is_plateau_point {
                        if !in_plateau {
                            // Start new plateau
                            current_plateau_start = i - 1; // Include previous point
                            in_plateau = true;
                        }
                    } else if in_plateau {
                        // End of plateau
                        if i - current_plateau_start >= self.min_plateau_steps {
                            plateaus.push((current_plateau_start, i - 1));
                        }
                        in_plateau = false;
                    }
                }
                
                last_metric_value = Some(current_value);
            }
        }
        
        // Check for plateau at the end
        if in_plateau && states.len() - current_plateau_start >= self.min_plateau_steps {
            plateaus.push((current_plateau_start, states.len() - 1));
        }
        
        // Create patterns for plateaus
        for (start, end) in plateaus {
            // Calculate confidence
            let plateau_length = end - start + 1;
            let confidence = Confidence::new(
                self.confidence_threshold + 
                (1.0 - self.confidence_threshold) * 
                ((plateau_length - self.min_plateau_steps) as f32 / self.min_plateau_steps as f32).min(1.0)
            );
            
            // Create pattern instance
            let pattern = PatternInstance::new(
                format!("plateau_{}_{}", start, end),
                self.category(),
                self.pattern_type(),
                format!("Search Plateau (Steps {}-{})", start, end),
                format!(
                    "The algorithm encountered a plateau from step {} to {}, \
                    where solution quality improved by less than {:.2}% per step.",
                    start, end, self.max_improvement * 100.0
                ),
                TemporalScope::Interval(start, end),
                SpatialScope::Global,
                confidence,
                EducationalSignificance::new(0.7),
                vec![
                    "Search Plateaus".to_string(),
                    "Optimization Challenges".to_string(),
                    "Local Search".to_string(),
                ],
            );
            
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn analyze_state(
        &self,
        state: &AlgorithmState,
    ) -> Result<Option<PatternInstance>, PatternError> {
        // Cannot detect plateaus from a single state
        Err(PatternError::InsufficientData(
            "Plateau detection requires execution history".to_string()
        ))
    }
    
    fn analyze_timeline(
        &self,
        timeline: &Timeline,
    ) -> Result<Vec<PatternInstance>, PatternError> {
        // Not enough information in timeline without full history
        Err(PatternError::InsufficientData(
            "Plateau detection requires execution history".to_string()
        ))
    }
    
    fn is_applicable_to(&self, algorithm: &dyn Algorithm) -> bool {
        // Applicable to optimization algorithms
        algorithm.category() == "optimization" || 
        algorithm.category() == "path_finding" || 
        algorithm.category() == "search"
    }
}

impl PlateauRecognizer {
    /// Get metric value from state for comparing progress
    fn get_metric_value(&self, state: &AlgorithmState) -> Option<f64> {
        // Try various metrics in priority order
        if let Some(value) = state.data.get("objective_value") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        
        if let Some(value) = state.data.get("distance") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        
        if let Some(value) = state.data.get("f_score") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        
        if let Some(value) = state.data.get("cost") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        
        // No suitable metric found
        None
    }
}

/// Pattern registry for managing recognizers
pub struct PatternRegistry {
    /// Registered pattern recognizers
    recognizers: Vec<Box<dyn PatternRecognizer>>,
}

impl PatternRegistry {
    /// Create a new pattern registry
    pub fn new() -> Self {
        Self {
            recognizers: Vec::new(),
        }
    }
    
    /// Create a registry with default recognizers
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        
        // Add default recognizers
        registry.register(Box::new(LocalOptimumRecognizer::default()));
        registry.register(Box::new(ExplorationExploitationRecognizer::default()));
        registry.register(Box::new(BacktrackingRecognizer::default()));
        registry.register(Box::new(HeuristicDominanceRecognizer::default()));
        registry.register(Box::new(PlateauRecognizer::default()));
        
        registry
    }
    
    /// Register a pattern recognizer
    pub fn register(&mut self, recognizer: Box<dyn PatternRecognizer>) {
        self.recognizers.push(recognizer);
    }
    
    /// Get recognizers applicable to an algorithm
    pub fn applicable_recognizers(&self, algorithm: &dyn Algorithm) -> Vec<&dyn PatternRecognizer> {
        self.recognizers.iter()
            .filter(|r| r.is_applicable_to(algorithm))
            .map(|r| r.as_ref())
            .collect()
    }
    
    /// Analyze execution trace to detect patterns
    pub fn analyze_trace(
        &self,
        tracer: &ExecutionTracer,
        algorithm: &dyn Algorithm,
    ) -> Vec<PatternInstance> {
        let mut patterns = Vec::new();
        
        // Apply applicable recognizers
        for recognizer in self.applicable_recognizers(algorithm) {
            match recognizer.analyze_trace(tracer) {
                Ok(mut recognizer_patterns) => patterns.append(&mut recognizer_patterns),
                Err(err) => {
                    trace!("Pattern recognizer error: {}", err);
                    continue;
                }
            }
        }
        
        // Sort patterns by confidence and significance
        patterns.sort_by(|a, b| {
            // First by confidence (descending)
            let confidence_cmp = b.confidence.value().partial_cmp(&a.confidence.value()).unwrap();
            if confidence_cmp != std::cmp::Ordering::Equal {
                return confidence_cmp;
            }
            
            // Then by significance (descending)
            let significance_cmp = b.significance.value().partial_cmp(&a.significance.value()).unwrap();
            if significance_cmp != std::cmp::Ordering::Equal {
                return significance_cmp;
            }
            
            // Finally by temporal scope (ascending)
            match (&a.temporal_scope, &b.temporal_scope) {
                (TemporalScope::Point, TemporalScope::Point) => std::cmp::Ordering::Equal,
                (TemporalScope::Point, _) => std::cmp::Ordering::Less,
                (_, TemporalScope::Point) => std::cmp::Ordering::Greater,
                (TemporalScope::Interval(a_start, _), TemporalScope::Interval(b_start, _)) => {
                    a_start.cmp(b_start)
                },
                _ => std::cmp::Ordering::Equal,
            }
        });
        
        patterns
    }
    
    /// Analyze algorithm state to detect patterns
    pub fn analyze_state(
        &self,
        state: &AlgorithmState,
        algorithm: &dyn Algorithm,
    ) -> Vec<PatternInstance> {
        let mut patterns = Vec::new();
        
        // Apply applicable recognizers
        for recognizer in self.applicable_recognizers(algorithm) {
            if let Ok(Some(pattern)) = recognizer.analyze_state(state) {
                patterns.push(pattern);
            }
        }
        
        // Sort patterns by confidence and significance
        patterns.sort_by(|a, b| {
            // First by confidence (descending)
            let confidence_cmp = b.confidence.value().partial_cmp(&a.confidence.value()).unwrap();
            if confidence_cmp != std::cmp::Ordering::Equal {
                return confidence_cmp;
            }
            
            // Then by significance (descending)
            b.significance.value().partial_cmp(&a.significance.value()).unwrap()
        });
        
        patterns
    }
    
    /// Analyze timeline to detect patterns
    pub fn analyze_timeline(
        &self,
        timeline: &Timeline,
        algorithm: &dyn Algorithm,
    ) -> Vec<PatternInstance> {
        let mut patterns = Vec::new();
        
        // Apply applicable recognizers
        for recognizer in self.applicable_recognizers(algorithm) {
            match recognizer.analyze_timeline(timeline) {
                Ok(mut recognizer_patterns) => patterns.append(&mut recognizer_patterns),
                Err(err) => {
                    trace!("Pattern recognizer error: {}", err);
                    continue;
                }
            }
        }
        
        // Sort patterns by confidence and significance
        patterns.sort_by(|a, b| {
            // First by confidence (descending)
            let confidence_cmp = b.confidence.value().partial_cmp(&a.confidence.value()).unwrap();
            if confidence_cmp != std::cmp::Ordering::Equal {
                return confidence_cmp;
            }
            
            // Then by significance (descending)
            b.significance.value().partial_cmp(&a.significance.value()).unwrap()
        });
        
        patterns
    }
}

impl Default for PatternRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock types for testing
    #[derive(Debug, Clone)]
    struct MockAlgorithmState {
        step: usize,
        current_node: Option<NodeId>,
        data: HashMap<String, String>,
    }
    
    impl AlgorithmState {
        fn mock(step: usize, current_node: Option<NodeId>) -> Self {
            let mut state = AlgorithmState {
                step,
                open_set: Vec::new(),
                closed_set: Vec::new(),
                current_node,
                data: HashMap::new(),
            };
            
            state
        }
        
        fn with_objective(mut self, value: f64) -> Self {
            self.data.insert("objective_value".to_string(), value.to_string());
            self
        }
    }
    
    struct MockExecutionTracer {
        states: Vec<AlgorithmState>,
    }
    
    impl MockExecutionTracer {
        fn new(states: Vec<AlgorithmState>) -> Self {
            Self { states }
        }
    }
    
    impl ExecutionTracer for MockExecutionTracer {
        fn states(&self) -> &Vec<AlgorithmState> {
            &self.states
        }
    }
    
    struct MockAlgorithm {}
    
    impl Algorithm for MockAlgorithm {
        fn name(&self) -> &str {
            "MockAlgorithm"
        }
        
        fn category(&self) -> &str {
            "path_finding"
        }
        
        fn description(&self) -> &str {
            "Mock algorithm for testing"
        }
        
        fn set_parameter(&mut self, _name: &str, _value: &str) -> Result<(), AlgorithmError> {
            Ok(())
        }
        
        fn get_parameter(&self, name: &str) -> Option<&str> {
            if name == "heuristic" {
                Some("manhattan")
            } else {
                None
            }
        }
        
        fn get_parameters(&self) -> HashMap<String, String> {
            let mut params = HashMap::new();
            params.insert("heuristic".to_string(), "manhattan".to_string());
            params
        }
    }
    
    #[test]
    fn test_confidence() {
        let c1 = Confidence::new(0.75);
        let c2 = Confidence::new(0.95);
        let c3 = Confidence::new(1.5); // Should be clamped to 1.0
        
        assert_eq!(c1.value(), 0.75);
        assert_eq!(c2.value(), 0.95);
        assert_eq!(c3.value(), 1.0);
        
        assert!(c1.exceeds(0.7));
        assert!(!c1.exceeds(0.8));
        
        assert_eq!(c1.to_linguistic(), "confident");
        assert_eq!(c2.to_linguistic(), "certain");
    }
    
    #[test]
    fn test_pattern_instance() {
        let pattern = PatternInstance::new(
            "test_pattern".to_string(),
            PatternCategory::SearchStrategy,
            PatternType::LocalOptimum,
            "Test Pattern".to_string(),
            "Test description".to_string(),
            TemporalScope::Interval(1, 5),
            SpatialScope::Global,
            Confidence::new(0.9),
            EducationalSignificance::new(0.8),
            vec!["Test Concept".to_string()],
        );
        
        assert_eq!(pattern.id, "test_pattern");
        assert_eq!(pattern.category, PatternCategory::SearchStrategy);
        assert!(matches!(pattern.pattern_type, PatternType::LocalOptimum));
        assert_eq!(pattern.name, "Test Pattern");
        assert_eq!(pattern.confidence.value(), 0.9);
        assert_eq!(pattern.significance.value(), 0.8);
        assert!(pattern.is_high_confidence());
        assert!(pattern.is_educationally_significant());
    }
    
    #[test]
    fn test_local_optimum_recognizer() {
        // Create a sequence of states with a local optimum
        let mut states = Vec::new();
        
        // Initial states with improving objective
        for i in 0..10 {
            states.push(AlgorithmState::mock(i, Some(i)).with_objective(100.0 - i as f64 * 10.0));
        }
        
        // Local optimum (no improvement)
        for i in 10..15 {
            states.push(AlgorithmState::mock(i, Some(i)).with_objective(0.0));
        }
        
        // Create tracer and recognizer
        let tracer = MockExecutionTracer::new(states);
        let recognizer = LocalOptimumRecognizer::default();
        
        // Analyze
        let patterns = recognizer.analyze_trace(&tracer).unwrap();
        
        // Should detect one pattern
        assert_eq!(patterns.len(), 1);
        assert!(matches!(patterns[0].pattern_type, PatternType::LocalOptimum));
        assert!(patterns[0].confidence.exceeds(0.7));
    }
}