//! Context-aware algorithm insight selection and presentation system
//!
//! This module provides a sophisticated framework for selecting, prioritizing, and
//! presenting algorithm insights based on contextual relevance, user expertise, and
//! educational value. It uses information-theoretic methods to maximize the educational
//! impact of insights while adapting to the user's expertise level.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};

use crate::algorithm::traits::{Algorithm, AlgorithmState};
use crate::education::progressive::{ExpertiseLevel, ProgressiveDisclosure};
use crate::insights::pattern::{Pattern, PatternMatch, PatternRepository};
use crate::insights::anomaly::{Anomaly, AnomalyMatch, AnomalyDetector};
use crate::insights::explanation::{Explanation, ExplanationGenerator};
use crate::temporal::timeline::{Timeline};
use crate::utils::math::{normalize, weighted_sum};

/// Insight presentation context containing all relevant contextual information
#[derive(Debug, Clone)]
pub struct PresentationContext {
    /// Current algorithm state
    pub current_state: AlgorithmState,
    
    /// Timeline containing execution history
    pub timeline: Option<Arc<Timeline>>,
    
    /// User expertise level
    pub expertise_level: ExpertiseLevel,
    
    /// Previously presented insights
    pub previous_insights: Vec<PresentedInsight>,
    
    /// Current algorithm being executed
    pub algorithm: String,
    
    /// Educational focus areas for the current session
    pub focus_areas: Vec<String>,
    
    /// Time spent viewing the current state (ms)
    pub view_time_ms: u64,
    
    /// User interaction count with the current state
    pub interaction_count: u32,
    
    /// Interaction history with chronos
    pub interaction_history: InteractionHistory,
}

/// Record of user interaction with the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHistory {
    /// Total time spent with the system (seconds)
    pub total_time_s: u64,
    
    /// Number of algorithms explored
    pub algorithms_explored: u32,
    
    /// Number of insights interacted with
    pub insights_interacted: u32,
    
    /// Expertise progression rates by area
    pub expertise_progression: HashMap<String, f64>,
    
    /// Average rating of insights (0.0-1.0)
    pub average_insight_rating: f64,
    
    /// Areas of demonstrated interest
    pub interest_areas: HashMap<String, f64>,
}

/// Types of insights that can be presented
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InsightType {
    /// Pattern-based insight
    Pattern,
    
    /// Anomaly-based insight
    Anomaly,
    
    /// Performance-based insight
    Performance,
    
    /// Decision-based insight
    Decision,
    
    /// Educational concept insight
    Concept,
    
    /// Recommendation for exploration
    Recommendation,
}

/// A prioritized insight ready for presentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentedInsight {
    /// Unique identifier for the insight
    pub id: String,
    
    /// Type of insight
    pub insight_type: InsightType,
    
    /// Title of the insight
    pub title: String,
    
    /// Short summary of the insight
    pub summary: String,
    
    /// Detailed explanation at the current expertise level
    pub explanation: Explanation,
    
    /// Relevance score (0.0-1.0)
    pub relevance: f64,
    
    /// Novelty score (0.0-1.0)
    pub novelty: f64,
    
    /// Educational value score (0.0-1.0)
    pub educational_value: f64,
    
    /// Overall priority score (0.0-1.0)
    pub priority: f64,
    
    /// Visualization references for the insight
    pub visualization_refs: Vec<VisualizationReference>,
    
    /// State references for the insight
    pub state_refs: Vec<StateReference>,
    
    /// Related concepts
    pub related_concepts: Vec<String>,
    
    /// Timestamp when the insight was generated
    pub timestamp: std::time::SystemTime,
    
    /// Has this insight been seen by the user
    pub seen: bool,
    
    /// Has this insight been interacted with
    pub interacted: bool,
    
    /// User rating if provided (0.0-1.0)
    pub user_rating: Option<f64>,
}

/// Reference to a visualization element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationReference {
    /// Type of visualization element
    pub ref_type: String,
    
    /// Identifier for the element
    pub element_id: String,
    
    /// Highlight parameters
    pub highlight_params: HashMap<String, String>,
}

/// Reference to a state element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateReference {
    /// Timeline step index
    pub step_index: usize,
    
    /// Branch ID if applicable
    pub branch_id: Option<String>,
    
    /// State component path
    pub component_path: String,
}

/// Interface for insight generation and prioritization
pub trait InsightPresenter: Send + Sync {
    /// Generate insights for the current context
    fn generate_insights(&self, context: &PresentationContext) -> Vec<PresentedInsight>;
    
    /// Prioritize insights for presentation
    fn prioritize_insights(
        &self,
        insights: Vec<PresentedInsight>,
        context: &PresentationContext
    ) -> Vec<PresentedInsight>;
    
    /// Get the top N insights for presentation
    fn get_top_insights(
        &self,
        context: &PresentationContext,
        count: usize
    ) -> Vec<PresentedInsight>;
    
    /// Record user interaction with an insight
    fn record_interaction(
        &self,
        insight_id: &str,
        interaction_type: InsightInteraction,
        rating: Option<f64>
    );
}

/// Types of insight interactions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InsightInteraction {
    /// Insight was viewed
    View,
    
    /// Insight was expanded
    Expand,
    
    /// Insight was dismissed
    Dismiss,
    
    /// Insight was rated
    Rate,
    
    /// Insight was applied
    Apply,
    
    /// Insight was shared
    Share,
}

/// Implementation of the insight presenter
pub struct InsightPresenterImpl {
    /// Pattern repository for pattern matching
    pattern_repository: Arc<PatternRepository>,
    
    /// Anomaly detector for anomaly detection
    anomaly_detector: Arc<AnomalyDetector>,
    
    /// Explanation generator for generating explanations
    explanation_generator: Arc<ExplanationGenerator>,
    
    /// Progressive disclosure system for expertise adaptation
    progressive_disclosure: Arc<ProgressiveDisclosure>,
    
    /// Cache of recently generated insights
    insight_cache: HashMap<String, Vec<PresentedInsight>>,
    
    /// Relevance factors for insight prioritization
    relevance_factors: RelevanceFactors,
    
    /// Interaction history for users
    interaction_histories: HashMap<String, InteractionHistory>,
}

/// Factors used for calculating insight relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelevanceFactors {
    /// Weight for educational value
    educational_value_weight: f64,
    
    /// Weight for novelty
    novelty_weight: f64,
    
    /// Weight for relevance to current state
    relevance_weight: f64,
    
    /// Weight for recency
    recency_weight: f64,
    
    /// Weight for user interest alignment
    interest_weight: f64,
    
    /// Base priority values for insight types
    base_priorities: HashMap<InsightType, f64>,
}

impl Default for RelevanceFactors {
    fn default() -> Self {
        let mut base_priorities = HashMap::new();
        base_priorities.insert(InsightType::Anomaly, 0.9);
        base_priorities.insert(InsightType::Pattern, 0.8);
        base_priorities.insert(InsightType::Decision, 0.7);
        base_priorities.insert(InsightType::Performance, 0.6);
        base_priorities.insert(InsightType::Concept, 0.5);
        base_priorities.insert(InsightType::Recommendation, 0.4);
        
        Self {
            educational_value_weight: 0.5,
            novelty_weight: 0.3,
            relevance_weight: 0.7,
            recency_weight: 0.2,
            interest_weight: 0.4,
            base_priorities,
        }
    }
}

impl InsightPresenterImpl {
    /// Create a new insight presenter
    pub fn new(
        pattern_repository: Arc<PatternRepository>,
        anomaly_detector: Arc<AnomalyDetector>,
        explanation_generator: Arc<ExplanationGenerator>,
        progressive_disclosure: Arc<ProgressiveDisclosure>
    ) -> Self {
        Self {
            pattern_repository,
            anomaly_detector,
            explanation_generator,
            progressive_disclosure,
            insight_cache: HashMap::new(),
            relevance_factors: RelevanceFactors::default(),
            interaction_histories: HashMap::new(),
        }
    }
    
    /// Check cache for existing insights
    fn check_cache(&self, context_key: &str) -> Option<Vec<PresentedInsight>> {
        self.insight_cache.get(context_key).cloned()
    }
    
    /// Update cache with new insights
    fn update_cache(&mut self, context_key: &str, insights: Vec<PresentedInsight>) {
        self.insight_cache.insert(context_key.to_string(), insights);
    }
    
    /// Generate context key for caching
    fn generate_context_key(&self, context: &PresentationContext) -> String {
        format!("{}:{}:{}",
            context.algorithm,
            context.current_state.step,
            context.expertise_level
        )
    }
    
    /// Generate pattern-based insights
    fn generate_pattern_insights(
        &self,
        context: &PresentationContext
    ) -> Vec<PresentedInsight> {
        let mut insights = Vec::new();
        
        // Get patterns from the repository that match the current algorithm
        let patterns = self.pattern_repository.get_patterns_for_algorithm(&context.algorithm);
        
        // Match patterns against the current state and timeline
        for pattern in patterns {
            if let Some(matches) = self.pattern_repository.match_pattern(
                &pattern,
                &context.current_state,
                context.timeline.as_ref()
            ) {
                for pattern_match in matches {
                    // Generate explanation for this pattern at the appropriate expertise level
                    let explanation = self.explanation_generator.generate_explanation(
                        &pattern_match,
                        context.expertise_level.clone()
                    );
                    
                    // Create insight from pattern match
                    let insight = self.create_pattern_insight(pattern_match, explanation);
                    insights.push(insight);
                }
            }
        }
        
        insights
    }
    
    /// Generate anomaly-based insights
    fn generate_anomaly_insights(
        &self,
        context: &PresentationContext
    ) -> Vec<PresentedInsight> {
        let mut insights = Vec::new();
        
        // Detect anomalies in the current state and timeline
        if let Some(timeline) = &context.timeline {
            let anomalies = self.anomaly_detector.detect_anomalies(
                &context.current_state,
                timeline.as_ref()
            );
            
            for anomaly_match in anomalies {
                // Generate explanation for this anomaly at the appropriate expertise level
                let explanation = self.explanation_generator.generate_explanation(
                    &anomaly_match,
                    context.expertise_level.clone()
                );
                
                // Create insight from anomaly match
                let insight = self.create_anomaly_insight(anomaly_match, explanation);
                insights.push(insight);
            }
        }
        
        insights
    }
    
    /// Create a pattern-based insight
    fn create_pattern_insight(
        &self,
        pattern_match: PatternMatch,
        explanation: Explanation
    ) -> PresentedInsight {
        PresentedInsight {
            id: format!("pattern-{}-{}", pattern_match.pattern_id, pattern_match.match_id),
            insight_type: InsightType::Pattern,
            title: pattern_match.title.clone(),
            summary: pattern_match.summary.clone(),
            explanation,
            relevance: pattern_match.confidence,
            novelty: 0.8, // Initial value, will be refined during prioritization
            educational_value: pattern_match.educational_value,
            priority: 0.0, // Will be calculated during prioritization
            visualization_refs: pattern_match.visualization_refs,
            state_refs: pattern_match.state_refs,
            related_concepts: pattern_match.related_concepts,
            timestamp: std::time::SystemTime::now(),
            seen: false,
            interacted: false,
            user_rating: None,
        }
    }
    
    /// Create an anomaly-based insight
    fn create_anomaly_insight(
        &self,
        anomaly_match: AnomalyMatch,
        explanation: Explanation
    ) -> PresentedInsight {
        PresentedInsight {
            id: format!("anomaly-{}-{}", anomaly_match.anomaly_id, anomaly_match.match_id),
            insight_type: InsightType::Anomaly,
            title: anomaly_match.title.clone(),
            summary: anomaly_match.summary.clone(),
            explanation,
            relevance: anomaly_match.confidence,
            novelty: 0.9, // Anomalies are inherently novel
            educational_value: anomaly_match.educational_value,
            priority: 0.0, // Will be calculated during prioritization
            visualization_refs: anomaly_match.visualization_refs,
            state_refs: anomaly_match.state_refs,
            related_concepts: anomaly_match.related_concepts,
            timestamp: std::time::SystemTime::now(),
            seen: false,
            interacted: false,
            user_rating: None,
        }
    }
    
    /// Calculate insight priority based on multiple factors
    fn calculate_priority(
        &self,
        insight: &PresentedInsight,
        context: &PresentationContext
    ) -> f64 {
        // Get base priority for this insight type
        let base_priority = self.relevance_factors.base_priorities
            .get(&insight.insight_type)
            .cloned()
            .unwrap_or(0.5);
        
        // Calculate novelty relative to previous insights
        let novelty = self.calculate_novelty(insight, &context.previous_insights);
        
        // Calculate relevance to current context
        let relevance = self.calculate_relevance(insight, context);
        
        // Calculate educational value based on user's expertise and session focus
        let educational_value = self.calculate_educational_value(insight, context);
        
        // Calculate interest alignment with user's demonstrated interests
        let interest_alignment = self.calculate_interest_alignment(insight, &context.interaction_history);
        
        // Calculate recency factor if this is a repeated insight
        let recency = self.calculate_recency(insight, &context.previous_insights);
        
        // Combine factors with weights
        let priority = weighted_sum(&[
            (base_priority, 0.2),
            (novelty, self.relevance_factors.novelty_weight),
            (relevance, self.relevance_factors.relevance_weight),
            (educational_value, self.relevance_factors.educational_value_weight),
            (interest_alignment, self.relevance_factors.interest_weight),
            (recency, self.relevance_factors.recency_weight),
        ]);
        
        normalize(priority, 0.0, 1.0)
    }
    
    /// Calculate novelty relative to previous insights
    fn calculate_novelty(
        &self,
        insight: &PresentedInsight,
        previous_insights: &[PresentedInsight]
    ) -> f64 {
        // Check if this exact insight has been seen before
        if previous_insights.iter().any(|i| i.id == insight.id) {
            return 0.2; // Very low novelty for repeated insights
        }
        
        // Check similarity to previous insights of the same type
        let similar_insights: Vec<&PresentedInsight> = previous_insights.iter()
            .filter(|i| i.insight_type == insight.insight_type)
            .collect();
        
        if similar_insights.is_empty() {
            return 0.9; // High novelty for first insight of this type
        }
        
        // Check concept overlap
        let concept_overlap: f64 = similar_insights.iter()
            .map(|i| {
                let common_concepts = i.related_concepts.iter()
                    .filter(|c| insight.related_concepts.contains(c))
                    .count();
                let total_concepts = i.related_concepts.len().max(1);
                common_concepts as f64 / total_concepts as f64
            })
            .sum::<f64>() / similar_insights.len() as f64;
        
        // Inverse of overlap gives novelty
        1.0 - concept_overlap.min(0.8) // Cap the reduction to preserve some novelty
    }
    
    /// Calculate relevance to current context
    fn calculate_relevance(
        &self,
        insight: &PresentedInsight,
        context: &PresentationContext
    ) -> f64 {
        // Base relevance from the insight itself
        let base_relevance = insight.relevance;
        
        // Focus area alignment
        let focus_alignment = if !context.focus_areas.is_empty() {
            let common_concepts = insight.related_concepts.iter()
                .filter(|c| context.focus_areas.contains(c))
                .count();
            let total_focus_areas = context.focus_areas.len().max(1);
            (common_concepts as f64 / total_focus_areas as f64).min(1.0)
        } else {
            0.5 // Neutral if no focus areas specified
        };
        
        // State relevance - how directly this relates to current state
        let state_relevance = if insight.state_refs.iter()
            .any(|r| r.step_index == context.current_state.step) {
            1.0 // Directly references current state
        } else {
            0.7 // References other states
        };
        
        // Interaction-based relevance
        let interaction_relevance = if context.interaction_count > 5 {
            0.8 // User is actively exploring this state
        } else if context.view_time_ms > 5000 {
            0.7 // User is studying this state
        } else {
            0.5 // Neutral for brief viewing
        };
        
        // Combine relevance factors
        weighted_sum(&[
            (base_relevance, 0.4),
            (focus_alignment, 0.3),
            (state_relevance, 0.2),
            (interaction_relevance, 0.1),
        ])
    }
    
    /// Calculate educational value based on user profile
    fn calculate_educational_value(
        &self,
        insight: &PresentedInsight,
        context: &PresentationContext
    ) -> f64 {
        // Base educational value from the insight
        let base_value = insight.educational_value;
        
        // Expertise-appropriate value
        let expertise_value = match context.expertise_level {
            ExpertiseLevel::Beginner => {
                // For beginners, basic concepts have higher value
                if insight.insight_type == InsightType::Concept {
                    0.9
                } else {
                    0.7
                }
            },
            ExpertiseLevel::Intermediate => {
                // For intermediate users, patterns have higher value
                if insight.insight_type == InsightType::Pattern {
                    0.9
                } else {
                    0.7
                }
            },
            ExpertiseLevel::Advanced => {
                // For advanced users, anomalies and performance insights have higher value
                if insight.insight_type == InsightType::Anomaly || 
                   insight.insight_type == InsightType::Performance {
                    0.9
                } else {
                    0.7
                }
            },
            ExpertiseLevel::Expert => {
                // For experts, rare patterns and anomalies have higher value
                if (insight.insight_type == InsightType::Anomaly || 
                    insight.insight_type == InsightType::Pattern) &&
                    insight.novelty > 0.8 {
                    0.9
                } else {
                    0.6
                }
            }
        };
        
        // Combine educational value factors
        weighted_sum(&[
            (base_value, 0.6),
            (expertise_value, 0.4),
        ])
    }
    
    /// Calculate interest alignment with user profile
    fn calculate_interest_alignment(
        &self,
        insight: &PresentedInsight,
        history: &InteractionHistory
    ) -> f64 {
        if history.interest_areas.is_empty() {
            return 0.5; // Neutral if no history
        }
        
        // Calculate concept alignment with interest areas
        let mut alignment_sum = 0.0;
        let mut alignment_count = 0;
        
        for concept in &insight.related_concepts {
            if let Some(interest_level) = history.interest_areas.get(concept) {
                alignment_sum += interest_level;
                alignment_count += 1;
            }
        }
        
        if alignment_count > 0 {
            alignment_sum / alignment_count as f64
        } else {
            0.5 // Neutral if no overlap
        }
    }
    
    /// Calculate recency factor for previously seen insights
    fn calculate_recency(
        &self,
        insight: &PresentedInsight,
        previous_insights: &[PresentedInsight]
    ) -> f64 {
        // Find this insight in previous insights if it exists
        if let Some(previous) = previous_insights.iter().find(|i| i.id == insight.id) {
            // Calculate time since last seen
            if let Ok(duration) = previous.timestamp.elapsed() {
                let hours = duration.as_secs() as f64 / 3600.0;
                
                // Recency curve - higher value for insights not seen recently
                // Logarithmic curve that approaches 1.0 as time increases
                let recency = (1.0 - (-0.1 * hours).exp()).min(1.0);
                
                // If user interacted positively, increase recency value
                if previous.interacted {
                    if let Some(rating) = previous.user_rating {
                        if rating > 0.7 {
                            // Good rating increases recency value (worth showing again sooner)
                            return recency * 0.8;
                        } else if rating < 0.3 {
                            // Poor rating decreases recency value (don't show again soon)
                            return recency * 0.2;
                        }
                    }
                }
                
                recency * 0.5 // Default modification if no rating
            } else {
                0.5 // Default if can't determine elapsed time
            }
        } else {
            1.0 // Max recency for never-before-seen insights
        }
    }
}

impl InsightPresenter for InsightPresenterImpl {
    fn generate_insights(&self, context: &PresentationContext) -> Vec<PresentedInsight> {
        // Generate context key for caching
        let context_key = self.generate_context_key(context);
        
        // Check cache first
        if let Some(cached_insights) = self.check_cache(&context_key) {
            debug!("Using cached insights for context {}", context_key);
            return cached_insights;
        }
        
        info!("Generating insights for context {}", context_key);
        
        // Generate insights from different sources
        let mut insights = Vec::new();
        
        // Add pattern-based insights
        let pattern_insights = self.generate_pattern_insights(context);
        insights.extend(pattern_insights);
        
        // Add anomaly-based insights
        let anomaly_insights = self.generate_anomaly_insights(context);
        insights.extend(anomaly_insights);
        
        // Add additional insight types as needed...
        
        // Update cache with generated insights
        if let Some(this_mut) = (self as *const Self as *mut Self).as_mut() {
            this_mut.update_cache(&context_key, insights.clone());
        }
        
        insights
    }
    
    fn prioritize_insights(
        &self,
        mut insights: Vec<PresentedInsight>,
        context: &PresentationContext
    ) -> Vec<PresentedInsight> {
        info!("Prioritizing {} insights", insights.len());
        
        // Calculate priority for each insight
        for insight in &mut insights {
            let priority = self.calculate_priority(&insight, context);
            insight.priority = priority;
        }
        
        // Sort by priority (descending)
        insights.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(Ordering::Equal));
        
        insights
    }
    
    fn get_top_insights(
        &self,
        context: &PresentationContext,
        count: usize
    ) -> Vec<PresentedInsight> {
        // Generate all insights
        let all_insights = self.generate_insights(context);
        
        // Prioritize insights
        let prioritized_insights = self.prioritize_insights(all_insights, context);
        
        // Return top N insights
        prioritized_insights.into_iter().take(count).collect()
    }
    
    fn record_interaction(
        &self,
        insight_id: &str,
        interaction_type: InsightInteraction,
        rating: Option<f64>
    ) {
        debug!("Recording interaction {:?} for insight {}", interaction_type, insight_id);
        
        // Note: In a real implementation, this would update the insight and user history
        // in a persistent store. For this implementation, we'll log the interaction only.
        match interaction_type {
            InsightInteraction::View => {
                debug!("Insight {} viewed", insight_id);
            },
            InsightInteraction::Expand => {
                debug!("Insight {} expanded", insight_id);
            },
            InsightInteraction::Dismiss => {
                debug!("Insight {} dismissed", insight_id);
            },
            InsightInteraction::Rate => {
                if let Some(r) = rating {
                    debug!("Insight {} rated {}", insight_id, r);
                }
            },
            InsightInteraction::Apply => {
                debug!("Insight {} applied", insight_id);
            },
            InsightInteraction::Share => {
                debug!("Insight {} shared", insight_id);
            }
        }
    }
}

/// Unit tests for the insight presentation system
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::education::progressive::ExpertiseLevel;
    
    /// Test insight priority calculation
    #[test]
    fn test_calculate_priority() {
        // TODO: Implement comprehensive tests for priority calculation
    }
    
    /// Test novelty calculation
    #[test]
    fn test_calculate_novelty() {
        // TODO: Implement comprehensive tests for novelty calculation
    }
    
    /// Test relevance calculation
    #[test]
    fn test_calculate_relevance() {
        // TODO: Implement comprehensive tests for relevance calculation
    }
    
    /// Test educational value calculation
    #[test]
    fn test_calculate_educational_value() {
        // TODO: Implement comprehensive tests for educational value calculation
    }
    
    /// Test interest alignment calculation
    #[test]
    fn test_calculate_interest_alignment() {
        // TODO: Implement comprehensive tests for interest alignment calculation
    }
    
    /// Test recency calculation
    #[test]
    fn test_calculate_recency() {
        // TODO: Implement comprehensive tests for recency calculation
    }
    
    /// Test insight generation and prioritization
    #[test]
    fn test_generate_and_prioritize_insights() {
        // TODO: Implement comprehensive tests for insight generation and prioritization
    }
}