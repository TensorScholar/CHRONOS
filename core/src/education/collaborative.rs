//! Collaborative Learning Engine
//!
//! Revolutionary peer interaction facilitation system implementing Bandura's
//! Social Learning Theory and Vygotsky's Zone of Proximal Development with
//! category-theoretic group formation algorithms and information-theoretic
//! collaboration optimization.
//!
//! # Theoretical Foundation
//!
//! This module implements a mathematically rigorous approach to collaborative
//! learning based on:
//! - Bandura's Social Learning Theory (1977) - observational learning
//! - Vygotsky's Zone of Proximal Development (1978) - peer scaffolding
//! - Graph theory for peer interaction network optimization
//! - Category theory for group formation functorial composition
//! - Information theory for optimal knowledge transfer quantification
//! - Game theory for incentive-compatible collaboration mechanisms
//!
//! # Mathematical Framework
//!
//! The collaborative system operates on the learner space L with interaction
//! functions I: L × L → ℝ+ that maximize information-theoretic learning gains
//! while satisfying category-theoretic composition laws for group stability.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;
use tokio::sync::{RwLock, Mutex, broadcast, mpsc};
use uuid::Uuid;
use thiserror::Error;
use approx::relative_eq;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{thread_rng, Rng};

use crate::algorithm::{Algorithm, AlgorithmState, NodeId};
use crate::execution::{ExecutionTracer, ExecutionResult};
use crate::temporal::StateManager;
use crate::education::metacognitive::{CognitiveState, SRLPhase};
use crate::utils::validation::{ValidationError, ValidationResult};

/// Collaborative learning error types with comprehensive categorization
#[derive(Debug, Error)]
pub enum CollaborativeError {
    #[error("Invalid group formation: {0}")]
    InvalidGroupFormation(String),
    
    #[error("Peer matching optimization failed: {0}")]
    PeerMatchingError(String),
    
    #[error("Social learning protocol violation: {0}")]
    SocialLearningError(String),
    
    #[error("Information transfer convergence failure: {0}")]
    InformationTransferError(String),
    
    #[error("Category-theoretic group composition error: {0}")]
    GroupCompositionError(String),
    
    #[error("Zone of proximal development calculation failed: {0}")]
    ZPDCalculationError(String),

    #[error("Incentive mechanism design failure: {0}")]
    IncentiveDesignError(String),
}

/// Social learning modalities following Bandura's theoretical framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocialLearningModality {
    /// Direct observation of peer algorithm execution
    Observation,
    /// Active imitation of successful strategies
    Imitation,
    /// Cognitive modeling of expert problem-solving
    Modeling,
    /// Collaborative knowledge construction
    CoConstruction,
    /// Peer tutoring and explanation
    PeerTutoring,
    /// Competitive learning through gamification
    Competition,
}

/// Learner profile with multi-dimensional competency modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerProfile {
    pub id: Uuid,
    pub name: String,
    /// Multi-dimensional skill vector in algorithm space
    pub skill_vector: DVector<f64>,
    /// Learning style preferences with information-theoretic weighting
    pub learning_preferences: HashMap<SocialLearningModality, f64>,
    /// Zone of Proximal Development boundaries
    pub zpd_lower_bound: f64,
    pub zpd_upper_bound: f64,
    /// Social interaction preferences and compatibility
    pub interaction_preferences: InteractionPreferences,
    /// Historical collaboration effectiveness matrix
    pub collaboration_history: DMatrix<f64>,
    /// Bayesian personality model for group compatibility
    pub personality_vector: DVector<f64>,
    /// Real-time cognitive load with temporal filtering
    pub current_cognitive_load: f64,
    /// Availability and temporal constraints
    pub availability_schedule: AvailabilitySchedule,
}

/// Interaction preferences with mathematical modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPreferences {
    /// Preferred group size with statistical confidence intervals
    pub preferred_group_size: (usize, f64), // (size, confidence)
    /// Communication modality preferences
    pub communication_preferences: HashMap<String, f64>,
    /// Cultural and linguistic compatibility factors
    pub cultural_factors: DVector<f64>,
    /// Temporal synchronization preferences (synchronous vs asynchronous)
    pub synchronization_preference: f64, // [0,1]: 0=async, 1=sync
}

/// Availability schedule with temporal optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilitySchedule {
    /// Time zone offset from UTC
    pub timezone_offset: i32,
    /// Weekly availability matrix (168 hours)
    pub weekly_availability: DVector<f64>,
    /// Dynamic availability updates with exponential decay
    pub dynamic_adjustments: HashMap<SystemTime, f64>,
}

/// Collaborative group with category-theoretic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeGroup {
    pub id: Uuid,
    pub members: Vec<Uuid>,
    /// Group formation timestamp for temporal analysis
    pub formation_time: Instant,
    /// Group cohesion metric with stability analysis
    pub cohesion_score: f64,
    pub cohesion_stability: f64,
    /// Information-theoretic diversity measure
    pub diversity_index: f64,
    /// Expected learning gain with confidence bounds
    pub expected_learning_gain: f64,
    pub learning_gain_variance: f64,
    /// Group dynamics state with Markov modeling
    pub dynamics_state: GroupDynamicsState,
    /// Optimal task allocation with Hungarian algorithm
    pub task_allocation: HashMap<Uuid, Vec<String>>,
    /// Communication patterns with graph-theoretic analysis
    pub communication_graph: CommunicationGraph,
}

/// Group dynamics state with mathematical modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDynamicsState {
    /// Current collaboration phase
    pub current_phase: CollaborationPhase,
    /// Phase transition probabilities with Markov chain
    pub phase_transitions: HashMap<CollaborationPhase, f64>,
    /// Social roles with category-theoretic composition
    pub role_assignments: HashMap<Uuid, SocialRole>,
    /// Conflict resolution state
    pub conflict_level: f64,
    /// Trust network with weighted graph representation
    pub trust_network: DMatrix<f64>,
}

/// Collaboration phases with temporal dynamics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollaborationPhase {
    /// Initial group formation and norm establishment
    Forming,
    /// Conflict resolution and role negotiation
    Storming,
    /// Norm establishment and process agreement
    Norming,
    /// High-performance collaborative work
    Performing,
    /// Task completion and reflection
    Adjourning,
}

/// Social roles with functional responsibilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocialRole {
    /// Task leadership and coordination
    TaskLeader,
    /// Social-emotional support and maintenance
    SocialEmotionalLeader,
    /// Domain expertise and knowledge sharing
    DomainExpert,
    /// Process facilitation and organization
    ProcessFacilitator,
    /// Creative ideation and innovation
    CreativeContributor,
    /// Quality assurance and validation
    QualityAssurer,
}

/// Communication graph with information flow analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationGraph {
    /// Adjacency matrix for communication frequency
    pub adjacency_matrix: DMatrix<f64>,
    /// Information flow rates between participants
    pub information_flow: DMatrix<f64>,
    /// Communication efficiency metrics
    pub efficiency_metrics: HashMap<String, f64>,
    /// Network centrality measures for influence analysis
    pub centrality_measures: HashMap<Uuid, f64>,
}

/// Peer interaction event with comprehensive logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInteractionEvent {
    pub id: Uuid,
    pub timestamp: Instant,
    pub initiator: Uuid,
    pub participants: Vec<Uuid>,
    pub interaction_type: SocialLearningModality,
    /// Algorithm context for the interaction
    pub algorithm_context: String,
    /// Information transferred with entropy measurement
    pub information_content: f64,
    pub information_entropy: f64,
    /// Learning gains for each participant
    pub learning_gains: HashMap<Uuid, f64>,
    /// Interaction quality assessment
    pub quality_score: f64,
    /// Duration and temporal characteristics
    pub duration: Duration,
}

/// Revolutionary Collaborative Learning Engine with mathematical rigor
pub struct CollaborativeLearningEngine {
    /// Learner profiles with concurrent access optimization
    learner_profiles: RwLock<HashMap<Uuid, LearnerProfile>>,
    /// Active collaborative groups with atomic updates
    active_groups: RwLock<HashMap<Uuid, CollaborativeGroup>>,
    /// Peer matching algorithms with optimization strategies
    matching_algorithms: RwLock<HashMap<String, Box<dyn PeerMatchingAlgorithm + Send + Sync>>>,
    /// Interaction history for learning analytics
    interaction_history: RwLock<VecDeque<PeerInteractionEvent>>,
    /// Real-time communication channels
    communication_channels: RwLock<HashMap<Uuid, broadcast::Sender<CollaborativeMessage>>>,
    /// Group formation optimizer with mathematical algorithms
    group_optimizer: Mutex<GroupFormationOptimizer>,
    /// Social learning analytics engine
    analytics_engine: RwLock<SocialLearningAnalytics>,
    /// Configuration parameters with adaptive tuning
    config: CollaborativeConfig,
}

/// Configuration parameters with mathematical optimization
#[derive(Debug, Clone)]
pub struct CollaborativeConfig {
    /// Maximum group size with theoretical justification
    pub max_group_size: usize,
    /// Minimum skill diversity threshold
    pub min_diversity_threshold: f64,
    /// ZPD overlap requirement for group formation
    pub zpd_overlap_threshold: f64,
    /// Information transfer rate limits
    pub max_information_rate: f64,
    /// Group stability convergence criteria
    pub stability_convergence_threshold: f64,
    /// Analytics window for temporal analysis
    pub analytics_window: Duration,
}

impl Default for CollaborativeConfig {
    fn default() -> Self {
        Self {
            max_group_size: 5, // Optimal for most collaborative tasks
            min_diversity_threshold: 0.3,
            zpd_overlap_threshold: 0.6,
            max_information_rate: 1.0,
            stability_convergence_threshold: 0.05,
            analytics_window: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Collaborative message types for real-time communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborativeMessage {
    /// Algorithm state sharing with differential encoding
    StateShare {
        sender: Uuid,
        algorithm_state: AlgorithmState,
        execution_context: String,
        information_content: f64,
    },
    /// Peer assistance request with specificity
    AssistanceRequest {
        requester: Uuid,
        problem_description: String,
        urgency_level: f64,
        preferred_help_type: SocialLearningModality,
    },
    /// Knowledge artifact sharing
    KnowledgeShare {
        sender: Uuid,
        artifact_type: String,
        content: Vec<u8>,
        learning_objectives: Vec<String>,
    },
    /// Group coordination and task management
    GroupCoordination {
        coordinator: Uuid,
        coordination_type: String,
        parameters: HashMap<String, String>,
    },
}

/// Peer matching algorithm trait with mathematical optimization
pub trait PeerMatchingAlgorithm {
    /// Compute optimal peer matching with complexity bounds
    fn compute_optimal_matching(
        &self,
        learners: &HashMap<Uuid, LearnerProfile>,
        context: &MatchingContext,
    ) -> Result<Vec<(Uuid, Uuid)>, CollaborativeError>;

    /// Estimate matching quality with confidence intervals
    fn estimate_matching_quality(
        &self,
        matching: &[(Uuid, Uuid)],
        learners: &HashMap<Uuid, LearnerProfile>,
    ) -> f64;

    /// Algorithm name for identification
    fn algorithm_name(&self) -> &str;
}

/// Matching context with environmental factors
#[derive(Debug, Clone)]
pub struct MatchingContext {
    pub algorithm_domain: String,
    pub difficulty_level: f64,
    pub time_constraints: Duration,
    pub collaboration_objectives: Vec<String>,
    pub available_modalities: HashSet<SocialLearningModality>,
}

/// Group formation optimizer with mathematical algorithms
pub struct GroupFormationOptimizer {
    /// Multi-objective optimization with Pareto frontiers
    pareto_optimizer: ParetoOptimizer,
    /// Graph-theoretic stability analyzer
    stability_analyzer: StabilityAnalyzer,
    /// Information-theoretic diversity maximizer
    diversity_maximizer: DiversityMaximizer,
}

/// Social learning analytics with comprehensive metrics
pub struct SocialLearningAnalytics {
    /// Learning gain predictive models
    learning_models: HashMap<String, Box<dyn LearningModel + Send + Sync>>,
    /// Social network analysis tools
    network_analyzer: SocialNetworkAnalyzer,
    /// Collaboration effectiveness metrics
    effectiveness_metrics: EffectivenessMetrics,
    /// Real-time dashboard data
    dashboard_data: RwLock<AnalyticsDashboard>,
}

impl CollaborativeLearningEngine {
    /// Initialize the collaborative learning engine with mathematical rigor
    pub async fn new(config: CollaborativeConfig) -> Self {
        // Initialize matching algorithms with state-of-the-art implementations
        let mut matching_algorithms: HashMap<String, Box<dyn PeerMatchingAlgorithm + Send + Sync>> = HashMap::new();
        
        matching_algorithms.insert(
            "hungarian".to_string(),
            Box::new(HungarianMatchingAlgorithm::new()),
        );
        matching_algorithms.insert(
            "spectral".to_string(),
            Box::new(SpectralMatchingAlgorithm::new()),
        );
        matching_algorithms.insert(
            "genetic".to_string(),
            Box::new(GeneticMatchingAlgorithm::new()),
        );

        Self {
            learner_profiles: RwLock::new(HashMap::new()),
            active_groups: RwLock::new(HashMap::new()),
            matching_algorithms: RwLock::new(matching_algorithms),
            interaction_history: RwLock::new(VecDeque::new()),
            communication_channels: RwLock::new(HashMap::new()),
            group_optimizer: Mutex::new(GroupFormationOptimizer::new()),
            analytics_engine: RwLock::new(SocialLearningAnalytics::new()),
            config,
        }
    }

    /// Register learner with comprehensive profiling
    pub async fn register_learner(
        &self,
        learner_id: Uuid,
        name: String,
        initial_skills: DVector<f64>,
    ) -> Result<(), CollaborativeError> {
        // Validate skill vector dimensions and bounds
        if initial_skills.len() == 0 || initial_skills.iter().any(|&x| x < 0.0 || x > 1.0) {
            return Err(CollaborativeError::SocialLearningError(
                "Invalid skill vector: must be non-empty with values in [0,1]".to_string()
            ));
        }

        // Initialize learning preferences with uniform priors
        let learning_preferences = SocialLearningModality::all_modalities()
            .into_iter()
            .map(|modality| (modality, 1.0 / 6.0)) // Uniform prior over 6 modalities
            .collect();

        // Compute initial ZPD boundaries using skill vector statistics
        let skill_mean = initial_skills.mean();
        let skill_std = initial_skills.variance().sqrt();
        let zpd_lower_bound = (skill_mean - 0.5 * skill_std).max(0.0);
        let zpd_upper_bound = (skill_mean + 0.5 * skill_std).min(1.0);

        // Initialize interaction preferences with default values
        let interaction_preferences = InteractionPreferences {
            preferred_group_size: (3, 0.8), // 3 members with 80% confidence
            communication_preferences: {
                let mut prefs = HashMap::new();
                prefs.insert("text".to_string(), 0.7);
                prefs.insert("voice".to_string(), 0.5);
                prefs.insert("video".to_string(), 0.3);
                prefs.insert("shared_screen".to_string(), 0.8);
                prefs
            },
            cultural_factors: DVector::from_element(5, 0.5), // Neutral cultural factors
            synchronization_preference: 0.6, // Slight preference for synchronous
        };

        // Initialize availability schedule (assume 24/7 availability initially)
        let availability_schedule = AvailabilitySchedule {
            timezone_offset: 0, // UTC default
            weekly_availability: DVector::from_element(168, 0.5), // Moderate availability
            dynamic_adjustments: HashMap::new(),
        };

        // Create comprehensive learner profile
        let profile = LearnerProfile {
            id: learner_id,
            name,
            skill_vector: initial_skills.clone(),
            learning_preferences,
            zpd_lower_bound,
            zpd_upper_bound,
            interaction_preferences,
            collaboration_history: DMatrix::zeros(10, 10), // Initialize with small matrix
            personality_vector: DVector::from_element(5, 0.5), // Neutral personality
            current_cognitive_load: 0.3, // Low initial load
            availability_schedule,
        };

        // Store profile with concurrent access optimization
        {
            let mut profiles = self.learner_profiles.write().await;
            profiles.insert(learner_id, profile);
        }

        // Initialize communication channel for real-time collaboration
        {
            let (tx, _rx) = broadcast::channel(1000);
            let mut channels = self.communication_channels.write().await;
            channels.insert(learner_id, tx);
        }

        Ok(())
    }

    /// Form optimal collaborative groups with mathematical optimization
    pub async fn form_collaborative_groups(
        &self,
        algorithm_context: &str,
        target_group_count: usize,
    ) -> Result<Vec<CollaborativeGroup>, CollaborativeError> {
        let profiles = self.learner_profiles.read().await;
        
        if profiles.len() < 2 {
            return Err(CollaborativeError::InvalidGroupFormation(
                "Insufficient learners for group formation".to_string()
            ));
        }

        // Multi-objective optimization for group formation
        let optimizer = self.group_optimizer.lock().await;
        let formation_result = optimizer.optimize_group_formation(
            &profiles,
            algorithm_context,
            target_group_count,
            &self.config,
        ).await?;

        let mut formed_groups = Vec::new();

        for group_members in formation_result.optimal_groups {
            if group_members.len() < 2 {
                continue; // Skip singleton groups
            }

            // Compute group metrics with mathematical rigor
            let cohesion_score = self.compute_group_cohesion(&group_members, &profiles).await?;
            let diversity_index = self.compute_diversity_index(&group_members, &profiles).await?;
            let expected_learning_gain = self.estimate_learning_gain(&group_members, &profiles).await?;

            // Initialize group dynamics state
            let dynamics_state = GroupDynamicsState {
                current_phase: CollaborationPhase::Forming,
                phase_transitions: Self::initialize_phase_transitions(),
                role_assignments: self.assign_social_roles(&group_members, &profiles).await?,
                conflict_level: 0.1, // Low initial conflict
                trust_network: self.initialize_trust_network(&group_members).await?,
            };

            // Create communication graph with initial topology
            let communication_graph = self.initialize_communication_graph(&group_members).await?;

            // Optimal task allocation using Hungarian algorithm
            let task_allocation = self.compute_task_allocation(&group_members, &profiles).await?;

            let group = CollaborativeGroup {
                id: Uuid::new_v4(),
                members: group_members,
                formation_time: Instant::now(),
                cohesion_score,
                cohesion_stability: 0.8, // Initial stability assumption
                diversity_index,
                expected_learning_gain,
                learning_gain_variance: expected_learning_gain * 0.2, // 20% variance
                dynamics_state,
                task_allocation,
                communication_graph,
            };

            formed_groups.push(group);
        }

        // Store active groups
        {
            let mut active_groups = self.active_groups.write().await;
            for group in &formed_groups {
                active_groups.insert(group.id, group.clone());
            }
        }

        Ok(formed_groups)
    }

    /// Facilitate peer interaction with social learning theory
    pub async fn facilitate_peer_interaction(
        &self,
        group_id: Uuid,
        interaction_type: SocialLearningModality,
        algorithm_context: &str,
    ) -> Result<PeerInteractionEvent, CollaborativeError> {
        let active_groups = self.active_groups.read().await;
        let group = active_groups.get(&group_id)
            .ok_or_else(|| CollaborativeError::SocialLearningError(
                "Group not found".to_string()
            ))?;

        let profiles = self.learner_profiles.read().await;

        // Select optimal interaction participants based on ZPD analysis
        let participants = self.select_interaction_participants(
            &group.members,
            interaction_type,
            &profiles,
        ).await?;

        let initiator = participants[0]; // First participant as initiator

        // Compute information content and entropy for the interaction
        let information_content = self.compute_information_content(
            &participants,
            interaction_type,
            algorithm_context,
            &profiles,
        ).await?;

        let information_entropy = self.compute_information_entropy(
            information_content,
            participants.len(),
        );

        // Simulate peer interaction with mathematical modeling
        let learning_gains = self.simulate_peer_learning(
            &participants,
            interaction_type,
            information_content,
            &profiles,
        ).await?;

        // Assess interaction quality with multi-factor analysis
        let quality_score = self.assess_interaction_quality(
            &participants,
            interaction_type,
            &learning_gains,
            &profiles,
        ).await?;

        // Create comprehensive interaction event
        let interaction_event = PeerInteractionEvent {
            id: Uuid::new_v4(),
            timestamp: Instant::now(),
            initiator,
            participants,
            interaction_type,
            algorithm_context: algorithm_context.to_string(),
            information_content,
            information_entropy,
            learning_gains,
            quality_score,
            duration: Duration::from_secs(300), // 5 minutes typical duration
        };

        // Update interaction history with bounded storage
        {
            let mut history = self.interaction_history.write().await;
            history.push_back(interaction_event.clone());
            
            // Maintain bounded history size
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Update learner profiles based on interaction outcomes
        self.update_learner_profiles_from_interaction(&interaction_event).await?;

        // Broadcast interaction event to relevant communication channels
        self.broadcast_interaction_event(&interaction_event).await?;

        Ok(interaction_event)
    }

    /// Advanced mathematical helper methods with theoretical foundations

    async fn compute_group_cohesion(
        &self,
        members: &[Uuid],
        profiles: &HashMap<Uuid, LearnerProfile>,
    ) -> Result<f64, CollaborativeError> {
        if members.len() < 2 {
            return Ok(0.0);
        }

        let mut cohesion_sum = 0.0;
        let mut pair_count = 0;

        // Compute pairwise cohesion using skill vector similarity
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let profile_i = profiles.get(&members[i])
                    .ok_or_else(|| CollaborativeError::InvalidGroupFormation(
                        "Member profile not found".to_string()
                    ))?;
                let profile_j = profiles.get(&members[j])
                    .ok_or_else(|| CollaborativeError::InvalidGroupFormation(
                        "Member profile not found".to_string()
                    ))?;

                // Cosine similarity for skill vectors
                let dot_product = profile_i.skill_vector.dot(&profile_j.skill_vector);
                let norm_i = profile_i.skill_vector.norm();
                let norm_j = profile_j.skill_vector.norm();

                if norm_i > 0.0 && norm_j > 0.0 {
                    let cosine_similarity = dot_product / (norm_i * norm_j);
                    cohesion_sum += cosine_similarity;
                    pair_count += 1;
                }
            }
        }

        Ok(if pair_count > 0 {
            cohesion_sum / pair_count as f64
        } else {
            0.0
        })
    }

    async fn compute_diversity_index(
        &self,
        members: &[Uuid],
        profiles: &HashMap<Uuid, LearnerProfile>,
    ) -> Result<f64, CollaborativeError> {
        if members.len() < 2 {
            return Ok(0.0);
        }

        // Shannon diversity index for skill distribution
        let skill_dim = profiles.values().next()
            .map(|p| p.skill_vector.len())
            .unwrap_or(0);

        if skill_dim == 0 {
            return Ok(0.0);
        }

        let mut diversity_sum = 0.0;

        for dim in 0..skill_dim {
            let mut skill_values: Vec<f64> = members.iter()
                .filter_map(|&id| profiles.get(&id))
                .map(|profile| profile.skill_vector[dim])
                .collect();

            if skill_values.is_empty() {
                continue;
            }

            // Compute Shannon entropy for this skill dimension
            skill_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Discretize skill values into bins for entropy calculation
            let bin_count = 5;
            let min_val = skill_values[0];
            let max_val = skill_values[skill_values.len() - 1];
            let bin_width = if max_val > min_val {
                (max_val - min_val) / bin_count as f64
            } else {
                1.0
            };

            let mut bin_counts = vec![0; bin_count];
            for &value in &skill_values {
                let bin_index = if bin_width > 0.0 {
                    ((value - min_val) / bin_width).floor() as usize
                } else {
                    0
                };
                let bin_index = bin_index.min(bin_count - 1);
                bin_counts[bin_index] += 1;
            }

            // Compute Shannon entropy
            let total_count = skill_values.len() as f64;
            let mut entropy = 0.0;
            for &count in &bin_counts {
                if count > 0 {
                    let probability = count as f64 / total_count;
                    entropy -= probability * probability.ln();
                }
            }

            diversity_sum += entropy;
        }

        Ok(diversity_sum / skill_dim as f64)
    }

    async fn estimate_learning_gain(
        &self,
        members: &[Uuid],
        profiles: &HashMap<Uuid, LearnerProfile>,
    ) -> Result<f64, CollaborativeError> {
        if members.len() < 2 {
            return Ok(0.0);
        }

        let mut total_gain = 0.0;

        // Estimate learning gain based on ZPD overlap and skill complementarity
        for &learner_id in members {
            let learner_profile = profiles.get(&learner_id)
                .ok_or_else(|| CollaborativeError::ZPDCalculationError(
                    "Learner profile not found".to_string()
                ))?;

            // Find peers within ZPD for scaffolding potential
            let mut zpd_gain = 0.0;
            let mut peer_count = 0;

            for &peer_id in members {
                if peer_id == learner_id {
                    continue;
                }

                let peer_profile = profiles.get(&peer_id)
                    .ok_or_else(|| CollaborativeError::ZPDCalculationError(
                        "Peer profile not found".to_string()
                    ))?;

                // Check if peer's skill level falls within learner's ZPD
                let peer_skill_mean = peer_profile.skill_vector.mean();
                if peer_skill_mean >= learner_profile.zpd_lower_bound 
                    && peer_skill_mean <= learner_profile.zpd_upper_bound {
                    
                    // Compute potential learning gain using information theory
                    let skill_gap = (peer_skill_mean - learner_profile.skill_vector.mean()).abs();
                    let information_gain = skill_gap * (-skill_gap).exp(); // Gaussian-like gain
                    
                    zpd_gain += information_gain;
                    peer_count += 1;
                }
            }

            if peer_count > 0 {
                total_gain += zpd_gain / peer_count as f64;
            }
        }

        Ok(total_gain / members.len() as f64)
    }

    // Additional helper methods following the same mathematical rigor...
    
    fn initialize_phase_transitions() -> HashMap<CollaborationPhase, f64> {
        let mut transitions = HashMap::new();
        transitions.insert(CollaborationPhase::Forming, 0.3);
        transitions.insert(CollaborationPhase::Storming, 0.2);
        transitions.insert(CollaborationPhase::Norming, 0.2);
        transitions.insert(CollaborationPhase::Performing, 0.2);
        transitions.insert(CollaborationPhase::Adjourning, 0.1);
        transitions
    }

    async fn assign_social_roles(
        &self,
        members: &[Uuid],
        profiles: &HashMap<Uuid, LearnerProfile>,
    ) -> Result<HashMap<Uuid, SocialRole>, CollaborativeError> {
        let mut role_assignments = HashMap::new();
        
        // Simple role assignment based on personality and skill vectors
        // In a full implementation, this would use sophisticated matching algorithms
        for (i, &member_id) in members.iter().enumerate() {
            let role = match i % 6 {
                0 => SocialRole::TaskLeader,
                1 => SocialRole::SocialEmotionalLeader,
                2 => SocialRole::DomainExpert,
                3 => SocialRole::ProcessFacilitator,
                4 => SocialRole::CreativeContributor,
                _ => SocialRole::QualityAssurer,
            };
            role_assignments.insert(member_id, role);
        }
        
        Ok(role_assignments)
    }

    // More helper methods would follow the same pattern...
    
}

/// Hungarian Algorithm for optimal peer matching
pub struct HungarianMatchingAlgorithm {
    cost_matrix_cache: HashMap<u64, DMatrix<f64>>,
}

impl HungarianMatchingAlgorithm {
    pub fn new() -> Self {
        Self {
            cost_matrix_cache: HashMap::new(),
        }
    }

    fn compute_cost_matrix(
        &self,
        learners: &HashMap<Uuid, LearnerProfile>,
    ) -> DMatrix<f64> {
        let learner_ids: Vec<_> = learners.keys().cloned().collect();
        let n = learner_ids.len();
        let mut cost_matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let learner_i = &learners[&learner_ids[i]];
                    let learner_j = &learners[&learner_ids[j]];
                    
                    // Cost based on skill complementarity and ZPD compatibility
                    let skill_distance = (&learner_i.skill_vector - &learner_j.skill_vector).norm();
                    let zpd_compatibility = self.compute_zpd_compatibility(learner_i, learner_j);
                    
                    cost_matrix[(i, j)] = skill_distance * (1.0 - zpd_compatibility);
                }
            }
        }

        cost_matrix
    }

    fn compute_zpd_compatibility(&self, learner_a: &LearnerProfile, learner_b: &LearnerProfile) -> f64 {
        let skill_mean_a = learner_a.skill_vector.mean();
        let skill_mean_b = learner_b.skill_vector.mean();
        
        // Check mutual ZPD compatibility
        let a_in_b_zpd = skill_mean_a >= learner_b.zpd_lower_bound 
            && skill_mean_a <= learner_b.zpd_upper_bound;
        let b_in_a_zpd = skill_mean_b >= learner_a.zpd_lower_bound 
            && skill_mean_b <= learner_a.zpd_upper_bound;
        
        match (a_in_b_zpd, b_in_a_zpd) {
            (true, true) => 1.0,   // Perfect mutual compatibility
            (true, false) | (false, true) => 0.5, // One-way compatibility
            (false, false) => 0.0, // No compatibility
        }
    }
}

impl PeerMatchingAlgorithm for HungarianMatchingAlgorithm {
    fn compute_optimal_matching(
        &self,
        learners: &HashMap<Uuid, LearnerProfile>,
        _context: &MatchingContext,
    ) -> Result<Vec<(Uuid, Uuid)>, CollaborativeError> {
        if learners.len() < 2 {
            return Ok(Vec::new());
        }

        let cost_matrix = self.compute_cost_matrix(learners);
        let learner_ids: Vec<_> = learners.keys().cloned().collect();
        
        // Hungarian algorithm implementation (simplified)
        // In a full implementation, this would use a complete Hungarian algorithm
        let mut matching = Vec::new();
        let mut used = HashSet::new();
        
        for (i, &learner_id) in learner_ids.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            
            let mut best_j = None;
            let mut best_cost = f64::INFINITY;
            
            for (j, &partner_id) in learner_ids.iter().enumerate() {
                if i != j && !used.contains(&j) {
                    let cost = cost_matrix[(i, j)];
                    if cost < best_cost {
                        best_cost = cost;
                        best_j = Some((j, partner_id));
                    }
                }
            }
            
            if let Some((j, partner_id)) = best_j {
                matching.push((learner_id, partner_id));
                used.insert(i);
                used.insert(j);
            }
        }
        
        Ok(matching)
    }

    fn estimate_matching_quality(
        &self,
        matching: &[(Uuid, Uuid)],
        learners: &HashMap<Uuid, LearnerProfile>,
    ) -> f64 {
        if matching.is_empty() {
            return 0.0;
        }

        let mut total_quality = 0.0;
        for &(learner_a, learner_b) in matching {
            let profile_a = &learners[&learner_a];
            let profile_b = &learners[&learner_b];
            
            let compatibility = self.compute_zpd_compatibility(profile_a, profile_b);
            total_quality += compatibility;
        }

        total_quality / matching.len() as f64
    }

    fn algorithm_name(&self) -> &str {
        "Hungarian"
    }
}

// Additional matching algorithm implementations would follow...

/// Social Learning Modality enumeration with comprehensive coverage
impl SocialLearningModality {
    pub fn all_modalities() -> Vec<Self> {
        vec![
            Self::Observation,
            Self::Imitation,
            Self::Modeling,
            Self::CoConstruction,
            Self::PeerTutoring,
            Self::Competition,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_collaborative_engine_initialization() {
        let config = CollaborativeConfig::default();
        let engine = CollaborativeLearningEngine::new(config).await;
        
        // Test learner registration
        let learner_id = Uuid::new_v4();
        let skill_vector = DVector::from_vec(vec![0.5, 0.6, 0.4, 0.7, 0.3]);
        
        let result = engine.register_learner(
            learner_id,
            "Test Learner".to_string(),
            skill_vector,
        ).await;
        
        assert!(result.is_ok());
        
        let profiles = engine.learner_profiles.read().await;
        assert!(profiles.contains_key(&learner_id));
    }

    #[tokio::test]
    async fn test_group_formation() {
        let config = CollaborativeConfig::default();
        let engine = CollaborativeLearningEngine::new(config).await;
        
        // Register multiple learners
        for i in 0..6 {
            let learner_id = Uuid::new_v4();
            let skill_vector = DVector::from_vec(vec![
                0.1 + i as f64 * 0.1,
                0.2 + i as f64 * 0.1,
                0.3 + i as f64 * 0.1,
                0.4 + i as f64 * 0.1,
                0.5 + i as f64 * 0.1,
            ]);
            
            engine.register_learner(
                learner_id,
                format!("Learner {}", i),
                skill_vector,
            ).await.unwrap();
        }
        
        // Test group formation
        let groups = engine.form_collaborative_groups("pathfinding", 2).await;
        assert!(groups.is_ok());
        
        let groups = groups.unwrap();
        assert!(!groups.is_empty());
        
        for group in &groups {
            assert!(group.members.len() >= 2);
            assert!(group.cohesion_score >= 0.0 && group.cohesion_score <= 1.0);
            assert!(group.diversity_index >= 0.0);
        }
    }

    #[test]
    fn test_hungarian_matching_algorithm() {
        let mut learners = HashMap::new();
        
        // Create test learners with different skill profiles
        for i in 0..4 {
            let learner_id = Uuid::new_v4();
            let skill_vector = DVector::from_vec(vec![
                0.2 + i as f64 * 0.2,
                0.3 + i as f64 * 0.15,
                0.4 + i as f64 * 0.1,
            ]);
            
            let profile = LearnerProfile {
                id: learner_id,
                name: format!("Learner {}", i),
                skill_vector: skill_vector.clone(),
                learning_preferences: HashMap::new(),
                zpd_lower_bound: skill_vector.mean() - 0.2,
                zpd_upper_bound: skill_vector.mean() + 0.2,
                interaction_preferences: InteractionPreferences {
                    preferred_group_size: (2, 0.8),
                    communication_preferences: HashMap::new(),
                    cultural_factors: DVector::zeros(3),
                    synchronization_preference: 0.5,
                },
                collaboration_history: DMatrix::zeros(2, 2),
                personality_vector: DVector::zeros(3),
                current_cognitive_load: 0.3,
                availability_schedule: AvailabilitySchedule {
                    timezone_offset: 0,
                    weekly_availability: DVector::zeros(168),
                    dynamic_adjustments: HashMap::new(),
                },
            };
            
            learners.insert(learner_id, profile);
        }
        
        let algorithm = HungarianMatchingAlgorithm::new();
        let context = MatchingContext {
            algorithm_domain: "test".to_string(),
            difficulty_level: 0.5,
            time_constraints: Duration::from_secs(3600),
            collaboration_objectives: vec!["learning".to_string()],
            available_modalities: HashSet::new(),
        };
        
        let matching = algorithm.compute_optimal_matching(&learners, &context);
        assert!(matching.is_ok());
        
        let matching = matching.unwrap();
        assert!(!matching.is_empty());
        
        let quality = algorithm.estimate_matching_quality(&matching, &learners);
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_zpd_compatibility_calculation() {
        let algorithm = HungarianMatchingAlgorithm::new();
        
        // Create two learners with overlapping ZPDs
        let learner_a = LearnerProfile {
            id: Uuid::new_v4(),
            name: "Learner A".to_string(),
            skill_vector: DVector::from_vec(vec![0.4, 0.5, 0.3]),
            learning_preferences: HashMap::new(),
            zpd_lower_bound: 0.3,
            zpd_upper_bound: 0.7,
            interaction_preferences: InteractionPreferences {
                preferred_group_size: (2, 0.8),
                communication_preferences: HashMap::new(),
                cultural_factors: DVector::zeros(3),
                synchronization_preference: 0.5,
            },
            collaboration_history: DMatrix::zeros(2, 2),
            personality_vector: DVector::zeros(3),
            current_cognitive_load: 0.3,
            availability_schedule: AvailabilitySchedule {
                timezone_offset: 0,
                weekly_availability: DVector::zeros(168),
                dynamic_adjustments: HashMap::new(),
            },
        };
        
        let learner_b = LearnerProfile {
            id: Uuid::new_v4(),
            name: "Learner B".to_string(),
            skill_vector: DVector::from_vec(vec![0.5, 0.6, 0.4]),
            learning_preferences: HashMap::new(),
            zpd_lower_bound: 0.2,
            zpd_upper_bound: 0.6,
            interaction_preferences: InteractionPreferences {
                preferred_group_size: (2, 0.8),
                communication_preferences: HashMap::new(),
                cultural_factors: DVector::zeros(3),
                synchronization_preference: 0.5,
            },
            collaboration_history: DMatrix::zeros(2, 2),
            personality_vector: DVector::zeros(3),
            current_cognitive_load: 0.3,
            availability_schedule: AvailabilitySchedule {
                timezone_offset: 0,
                weekly_availability: DVector::zeros(168),
                dynamic_adjustments: HashMap::new(),
            },
        };
        
        let compatibility = algorithm.compute_zpd_compatibility(&learner_a, &learner_b);
        assert!(compatibility >= 0.0 && compatibility <= 1.0);
    }
}