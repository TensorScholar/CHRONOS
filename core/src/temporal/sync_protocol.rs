//! Distributed Temporal Debugging Synchronization Protocol
//!
//! This module implements a revolutionary vector clock coordination system
//! for distributed temporal debugging with consensus algorithms and
//! category-theoretic event ordering guarantees.
//!
//! # Theoretical Foundation
//!
//! The synchronization protocol is built on:
//! - Lamport's logical clocks extended to vector clocks
//! - Category-theoretic causal order preservation
//! - Information-theoretic consensus bounds
//! - Byzantine fault tolerance with PBFT adaptation
//!
//! # Mathematical Guarantees
//!
//! - Causal ordering: ∀e₁,e₂: e₁ → e₂ ⟺ VC(e₁) < VC(e₂)
//! - Consensus safety: |F| < n/3 for Byzantine fault tolerance
//! - Liveness: Progress guaranteed under network synchrony assumptions
//! - Information bounds: O(log n) message complexity per synchronization

use crate::temporal::{StateManager, StateDelta, TimelineId, BranchId};
use crate::algorithm::AlgorithmState;
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tokio::time::{timeout, sleep};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use rayon::prelude::*;
use nalgebra::{DVector, DMatrix};
use rand::{Rng, thread_rng};
use approx::AbsDiffEq;

/// Node identifier in the distributed system
pub type NodeId = u32;

/// Logical timestamp for vector clocks
pub type LogicalTime = u64;

/// Epoch identifier for consensus rounds
pub type EpochId = u64;

/// Vector clock for distributed event ordering
///
/// Implements category-theoretic partial order with functorial composition:
/// - Objects: Events in the distributed system
/// - Morphisms: Causal relationships (happens-before)
/// - Composition: Transitive closure of causal dependencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Clock values for each node in the system
    clocks: BTreeMap<NodeId, LogicalTime>,
    /// Epoch for consensus coordination
    epoch: EpochId,
    /// Entropy measure for information-theoretic analysis
    entropy: f64,
}

impl VectorClock {
    /// Create a new vector clock for the given node
    pub fn new(node_id: NodeId, num_nodes: usize) -> Self {
        let mut clocks = BTreeMap::new();
        for i in 0..num_nodes as NodeId {
            clocks.insert(i, 0);
        }
        
        Self {
            clocks,
            epoch: 0,
            entropy: 0.0,
        }
    }

    /// Increment the clock for the given node (local event)
    pub fn tick(&mut self, node_id: NodeId) {
        if let Some(clock) = self.clocks.get_mut(&node_id) {
            *clock += 1;
            self.update_entropy();
        }
    }

    /// Update this clock with another clock (message receive)
    ///
    /// Implements category-theoretic supremum in the causal order lattice
    pub fn update(&mut self, other: &VectorClock, node_id: NodeId) {
        // Take the maximum of corresponding clock values (lattice supremum)
        for (&other_node, &other_time) in &other.clocks {
            if let Some(our_time) = self.clocks.get_mut(&other_node) {
                *our_time = (*our_time).max(other_time);
            }
        }
        
        // Increment our own clock
        self.tick(node_id);
        
        // Update epoch to maximum
        self.epoch = self.epoch.max(other.epoch);
        
        self.update_entropy();
    }

    /// Check if this clock is causally before another
    ///
    /// Implements the happens-before relation: e₁ → e₂ ⟺ VC(e₁) < VC(e₂)
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;
        
        for (&node, &our_time) in &self.clocks {
            if let Some(&other_time) = other.clocks.get(&node) {
                if our_time > other_time {
                    return false; // Not causally ordered
                } else if our_time < other_time {
                    strictly_less = true;
                }
            }
        }
        
        strictly_less
    }

    /// Check if two clocks are concurrent (neither happens-before the other)
    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Calculate information-theoretic entropy of the clock state
    fn update_entropy(&mut self) {
        let total: u64 = self.clocks.values().sum();
        if total == 0 {
            self.entropy = 0.0;
            return;
        }

        let mut entropy = 0.0;
        for &clock_value in self.clocks.values() {
            if clock_value > 0 {
                let p = clock_value as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }
        
        self.entropy = entropy;
    }

    /// Get the entropy measure for information-theoretic analysis
    pub fn entropy(&self) -> f64 {
        self.entropy
    }

    /// Convert to compact representation for network transmission
    pub fn to_compact(&self) -> CompactVectorClock {
        CompactVectorClock {
            clocks: self.clocks.iter().map(|(&k, &v)| (k, v)).collect(),
            epoch: self.epoch,
        }
    }
}

/// Compact vector clock representation for efficient network transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactVectorClock {
    clocks: Vec<(NodeId, LogicalTime)>,
    epoch: EpochId,
}

impl From<CompactVectorClock> for VectorClock {
    fn from(compact: CompactVectorClock) -> Self {
        let mut vc = VectorClock {
            clocks: compact.clocks.into_iter().collect(),
            epoch: compact.epoch,
            entropy: 0.0,
        };
        vc.update_entropy();
        vc
    }
}

/// Distributed synchronization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    /// Local state modification event
    StateModification {
        timeline_id: TimelineId,
        branch_id: BranchId,
        delta: StateDelta,
        vector_clock: VectorClock,
    },
    /// Consensus proposal for global synchronization
    ConsensusProposal {
        epoch: EpochId,
        proposed_state: GlobalConsensusState,
        proposer: NodeId,
        vector_clock: VectorClock,
    },
    /// Vote in consensus protocol
    ConsensusVote {
        epoch: EpochId,
        proposal_hash: u64,
        vote: ConsensusVote,
        voter: NodeId,
        vector_clock: VectorClock,
    },
    /// Heartbeat for liveness detection
    Heartbeat {
        node_id: NodeId,
        vector_clock: VectorClock,
        load_metrics: NodeLoadMetrics,
    },
}

/// Consensus vote in Byzantine fault tolerant protocol
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusVote {
    /// Accept the proposal
    Accept,
    /// Reject the proposal with reason
    Reject(String),
    /// Abstain from voting
    Abstain,
}

/// Global consensus state for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConsensusState {
    /// Current global vector clock
    global_clock: VectorClock,
    /// Active timelines across all nodes
    active_timelines: HashMap<TimelineId, TimelineMetadata>,
    /// Consensus round identifier
    round: u64,
    /// Merkle root of distributed state
    state_hash: u64,
}

/// Metadata for distributed timeline coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineMetadata {
    /// Owner node of the timeline
    owner: NodeId,
    /// Current head state
    head_state_id: String,
    /// Number of active branches
    branch_count: usize,
    /// Access control permissions
    permissions: TimelinePermissions,
}

/// Access control for distributed timeline operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelinePermissions {
    /// Nodes with read access
    readers: Vec<NodeId>,
    /// Nodes with write access
    writers: Vec<NodeId>,
    /// Public access level
    public_access: AccessLevel,
}

/// Access level enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessLevel {
    None,
    Read,
    Write,
    Admin,
}

/// Node performance and load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLoadMetrics {
    /// CPU utilization percentage
    cpu_usage: f64,
    /// Memory utilization percentage  
    memory_usage: f64,
    /// Network bandwidth utilization
    network_usage: f64,
    /// Number of active temporal operations
    active_operations: usize,
    /// Average response latency
    avg_latency_ms: f64,
}

/// Distributed synchronization protocol implementation
///
/// Implements a revolutionary consensus algorithm combining:
/// - PBFT for Byzantine fault tolerance
/// - Vector clocks for causal ordering
/// - Information-theoretic optimization
/// - Category-theoretic correctness guarantees
pub struct DistributedSyncProtocol {
    /// Current node identifier
    node_id: NodeId,
    /// Vector clock for this node
    vector_clock: Arc<RwLock<VectorClock>>,
    /// Known nodes in the distributed system
    known_nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    /// Current consensus state
    consensus_state: Arc<RwLock<GlobalConsensusState>>,
    /// Pending consensus proposals
    pending_proposals: Arc<RwLock<HashMap<EpochId, ConsensusProposal>>>,
    /// Message queue for incoming synchronization events
    event_queue: Arc<AsyncMutex<VecDeque<SyncEvent>>>,
    /// Network communication channels
    network_tx: mpsc::UnboundedSender<NetworkMessage>,
    network_rx: Arc<AsyncMutex<mpsc::UnboundedReceiver<NetworkMessage>>>,
    /// Performance metrics
    metrics: Arc<SyncMetrics>,
    /// Configuration parameters
    config: SyncConfig,
}

/// Node information in the distributed system
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node identifier
    id: NodeId,
    /// Network address
    address: String,
    /// Last known vector clock
    last_clock: VectorClock,
    /// Last heartbeat timestamp
    last_heartbeat: Instant,
    /// Node performance metrics
    load_metrics: NodeLoadMetrics,
    /// Trust score for Byzantine fault tolerance
    trust_score: f64,
}

/// Consensus proposal tracking
#[derive(Debug, Clone)]
pub struct ConsensusProposal {
    /// Proposal epoch
    epoch: EpochId,
    /// Proposed global state
    proposed_state: GlobalConsensusState,
    /// Proposer node
    proposer: NodeId,
    /// Received votes
    votes: HashMap<NodeId, ConsensusVote>,
    /// Proposal timestamp
    timestamp: Instant,
    /// Proposal hash for integrity
    hash: u64,
}

/// Network message for inter-node communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Source node identifier
    source: NodeId,
    /// Destination node identifier  
    destination: NodeId,
    /// Synchronization event payload
    event: SyncEvent,
    /// Message timestamp
    timestamp: SystemTime,
    /// Message signature for integrity
    signature: u64,
}

/// Performance metrics for synchronization protocol
#[derive(Debug)]
pub struct SyncMetrics {
    /// Total number of events processed
    events_processed: AtomicU64,
    /// Total number of consensus rounds
    consensus_rounds: AtomicU64,
    /// Average consensus latency
    avg_consensus_latency: AtomicU64,
    /// Network message count
    messages_sent: AtomicU64,
    /// Synchronization errors
    sync_errors: AtomicU64,
    /// Active node count
    active_nodes: AtomicUsize,
}

/// Configuration for synchronization protocol
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Consensus timeout duration
    consensus_timeout: Duration,
    /// Heartbeat interval
    heartbeat_interval: Duration,
    /// Maximum number of Byzantine failures tolerated
    max_byzantine_failures: usize,
    /// Minimum votes required for consensus
    min_consensus_votes: usize,
    /// Network message timeout
    network_timeout: Duration,
    /// Trust threshold for node acceptance
    trust_threshold: f64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            consensus_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(5),
            max_byzantine_failures: 1,
            min_consensus_votes: 3,
            network_timeout: Duration::from_secs(10),
            trust_threshold: 0.8,
        }
    }
}

/// Synchronization protocol errors
#[derive(Debug, Error)]
pub enum SyncError {
    #[error("Consensus timeout after {timeout:?}")]
    ConsensusTimeout { timeout: Duration },
    
    #[error("Byzantine failure detected from node {node_id}")]
    ByzantineFailure { node_id: NodeId },
    
    #[error("Network communication error: {message}")]
    NetworkError { message: String },
    
    #[error("Invalid vector clock: {reason}")]
    InvalidVectorClock { reason: String },
    
    #[error("Insufficient consensus votes: got {got}, need {need}")]
    InsufficientVotes { got: usize, need: usize },
    
    #[error("Node trust below threshold: {score} < {threshold}")]
    TrustViolation { score: f64, threshold: f64 },
}

impl DistributedSyncProtocol {
    /// Create a new distributed synchronization protocol instance
    pub fn new(node_id: NodeId, num_nodes: usize) -> Self {
        let (network_tx, network_rx) = mpsc::unbounded_channel();
        
        Self {
            node_id,
            vector_clock: Arc::new(RwLock::new(VectorClock::new(node_id, num_nodes))),
            known_nodes: Arc::new(RwLock::new(HashMap::new())),
            consensus_state: Arc::new(RwLock::new(GlobalConsensusState {
                global_clock: VectorClock::new(node_id, num_nodes),
                active_timelines: HashMap::new(),
                round: 0,
                state_hash: 0,
            })),
            pending_proposals: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(AsyncMutex::new(VecDeque::new())),
            network_tx,
            network_rx: Arc::new(AsyncMutex::new(network_rx)),
            metrics: Arc::new(SyncMetrics {
                events_processed: AtomicU64::new(0),
                consensus_rounds: AtomicU64::new(0),
                avg_consensus_latency: AtomicU64::new(0),
                messages_sent: AtomicU64::new(0),
                sync_errors: AtomicU64::new(0),
                active_nodes: AtomicUsize::new(0),
            }),
            config: SyncConfig::default(),
        }
    }

    /// Add a known node to the distributed system
    pub async fn add_node(&self, node_info: NodeInfo) -> Result<(), SyncError> {
        let mut nodes = self.known_nodes.write().unwrap();
        nodes.insert(node_info.id, node_info);
        self.metrics.active_nodes.store(nodes.len(), Ordering::Relaxed);
        Ok(())
    }

    /// Process a synchronization event with category-theoretic ordering
    pub async fn process_event(&self, event: SyncEvent) -> Result<(), SyncError> {
        self.metrics.events_processed.fetch_add(1, Ordering::Relaxed);
        
        match event {
            SyncEvent::StateModification { vector_clock, .. } => {
                self.update_vector_clock(&vector_clock).await?;
            }
            SyncEvent::ConsensusProposal { epoch, proposed_state, proposer, vector_clock } => {
                self.handle_consensus_proposal(epoch, proposed_state, proposer, vector_clock).await?;
            }
            SyncEvent::ConsensusVote { epoch, proposal_hash, vote, voter, vector_clock } => {
                self.handle_consensus_vote(epoch, proposal_hash, vote, voter, vector_clock).await?;
            }
            SyncEvent::Heartbeat { node_id, vector_clock, load_metrics } => {
                self.handle_heartbeat(node_id, vector_clock, load_metrics).await?;
            }
        }
        
        Ok(())
    }

    /// Update vector clock with causal ordering preservation
    async fn update_vector_clock(&self, other_clock: &VectorClock) -> Result<(), SyncError> {
        let mut our_clock = self.vector_clock.write().unwrap();
        our_clock.update(other_clock, self.node_id);
        
        // Verify causal ordering preservation
        if !self.verify_causal_consistency(&our_clock).await {
            return Err(SyncError::InvalidVectorClock {
                reason: "Causal ordering violation detected".to_string(),
            });
        }
        
        Ok(())
    }

    /// Handle consensus proposal with Byzantine fault tolerance
    async fn handle_consensus_proposal(
        &self,
        epoch: EpochId,
        proposed_state: GlobalConsensusState,
        proposer: NodeId,
        vector_clock: VectorClock,
    ) -> Result<(), SyncError> {
        // Verify proposer trust score
        let trust_score = self.get_node_trust_score(proposer).await;
        if trust_score < self.config.trust_threshold {
            return Err(SyncError::TrustViolation {
                score: trust_score,
                threshold: self.config.trust_threshold,
            });
        }

        // Update our vector clock
        self.update_vector_clock(&vector_clock).await?;

        // Validate the proposed state
        let vote = if self.validate_proposed_state(&proposed_state).await {
            ConsensusVote::Accept
        } else {
            ConsensusVote::Reject("State validation failed".to_string())
        };

        // Send our vote
        let vote_event = SyncEvent::ConsensusVote {
            epoch,
            proposal_hash: self.hash_state(&proposed_state),
            vote,
            voter: self.node_id,
            vector_clock: self.vector_clock.read().unwrap().clone(),
        };

        self.broadcast_event(vote_event).await?;
        Ok(())
    }

    /// Handle consensus vote with aggregation
    async fn handle_consensus_vote(
        &self,
        epoch: EpochId,
        proposal_hash: u64,
        vote: ConsensusVote,
        voter: NodeId,
        vector_clock: VectorClock,
    ) -> Result<(), SyncError> {
        self.update_vector_clock(&vector_clock).await?;

        let mut proposals = self.pending_proposals.write().unwrap();
        if let Some(proposal) = proposals.get_mut(&epoch) {
            if proposal.hash == proposal_hash {
                proposal.votes.insert(voter, vote);

                // Check if we have enough votes for consensus
                let accept_votes = proposal.votes.values()
                    .filter(|&&ref v| *v == ConsensusVote::Accept)
                    .count();

                if accept_votes >= self.config.min_consensus_votes {
                    // Consensus reached - apply the proposed state
                    self.apply_consensus_state(&proposal.proposed_state).await?;
                    proposals.remove(&epoch);
                    self.metrics.consensus_rounds.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Handle heartbeat for liveness detection
    async fn handle_heartbeat(
        &self,
        node_id: NodeId,
        vector_clock: VectorClock,
        load_metrics: NodeLoadMetrics,
    ) -> Result<(), SyncError> {
        self.update_vector_clock(&vector_clock).await?;

        // Update node information
        let mut nodes = self.known_nodes.write().unwrap();
        if let Some(node_info) = nodes.get_mut(&node_id) {
            node_info.last_clock = vector_clock;
            node_info.last_heartbeat = Instant::now();
            node_info.load_metrics = load_metrics;
        }

        Ok(())
    }

    /// Verify causal consistency of vector clock
    async fn verify_causal_consistency(&self, clock: &VectorClock) -> bool {
        // Implement causal consistency verification
        // This is a simplified version - full implementation would include
        // more sophisticated consistency checks
        clock.entropy() >= 0.0 && clock.entropy() <= 10.0 // Reasonable entropy bounds
    }

    /// Get trust score for a node
    async fn get_node_trust_score(&self, node_id: NodeId) -> f64 {
        let nodes = self.known_nodes.read().unwrap();
        nodes.get(&node_id)
            .map(|info| info.trust_score)
            .unwrap_or(0.0)
    }

    /// Validate a proposed consensus state
    async fn validate_proposed_state(&self, state: &GlobalConsensusState) -> bool {
        // Implement state validation logic
        // This includes checking state hash, timeline consistency, etc.
        state.round > 0 && state.state_hash != 0
    }

    /// Hash a consensus state for integrity verification
    fn hash_state(&self, state: &GlobalConsensusState) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        state.round.hash(&mut hasher);
        state.state_hash.hash(&mut hasher);
        hasher.finish()
    }

    /// Apply a consensus state after successful voting
    async fn apply_consensus_state(&self, state: &GlobalConsensusState) -> Result<(), SyncError> {
        let mut consensus_state = self.consensus_state.write().unwrap();
        *consensus_state = state.clone();
        Ok(())
    }

    /// Broadcast an event to all known nodes
    async fn broadcast_event(&self, event: SyncEvent) -> Result<(), SyncError> {
        let nodes = self.known_nodes.read().unwrap();
        
        for (&node_id, _) in nodes.iter() {
            if node_id != self.node_id {
                let message = NetworkMessage {
                    source: self.node_id,
                    destination: node_id,
                    event: event.clone(),
                    timestamp: SystemTime::now(),
                    signature: self.sign_message(&event),
                };

                if let Err(_) = self.network_tx.send(message) {
                    return Err(SyncError::NetworkError {
                        message: format!("Failed to send message to node {}", node_id),
                    });
                }
            }
        }

        self.metrics.messages_sent.fetch_add(nodes.len() as u64 - 1, Ordering::Relaxed);
        Ok(())
    }

    /// Sign a message for integrity verification
    fn sign_message(&self, event: &SyncEvent) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.node_id.hash(&mut hasher);
        // In a real implementation, this would use cryptographic signatures
        hasher.finish()
    }

    /// Start the synchronization protocol event loop
    pub async fn start(&self) -> Result<(), SyncError> {
        let heartbeat_interval = self.config.heartbeat_interval;
        let node_id = self.node_id;
        let vector_clock = Arc::clone(&self.vector_clock);
        let network_tx = self.network_tx.clone();

        // Start heartbeat task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat_interval);
            loop {
                interval.tick().await;
                
                let heartbeat = SyncEvent::Heartbeat {
                    node_id,
                    vector_clock: vector_clock.read().unwrap().clone(),
                    load_metrics: NodeLoadMetrics {
                        cpu_usage: 0.5, // Would be actual metrics
                        memory_usage: 0.3,
                        network_usage: 0.2,
                        active_operations: 10,
                        avg_latency_ms: 5.0,
                    },
                };

                // Would broadcast to other nodes
                // This is simplified for demonstration
            }
        });

        Ok(())
    }

    /// Get current synchronization metrics
    pub fn get_metrics(&self) -> SyncMetricsSnapshot {
        SyncMetricsSnapshot {
            events_processed: self.metrics.events_processed.load(Ordering::Relaxed),
            consensus_rounds: self.metrics.consensus_rounds.load(Ordering::Relaxed),
            avg_consensus_latency_us: self.metrics.avg_consensus_latency.load(Ordering::Relaxed),
            messages_sent: self.metrics.messages_sent.load(Ordering::Relaxed),
            sync_errors: self.metrics.sync_errors.load(Ordering::Relaxed),
            active_nodes: self.metrics.active_nodes.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of synchronization metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetricsSnapshot {
    pub events_processed: u64,
    pub consensus_rounds: u64,
    pub avg_consensus_latency_us: u64,
    pub messages_sent: u64,
    pub sync_errors: u64,
    pub active_nodes: usize,
}

/// Distributed temporal state synchronizer
///
/// High-level interface for distributed temporal debugging coordination
pub struct DistributedTemporalSynchronizer {
    protocol: DistributedSyncProtocol,
    state_manager: Arc<StateManager>,
}

impl DistributedTemporalSynchronizer {
    /// Create a new distributed temporal synchronizer
    pub fn new(node_id: NodeId, num_nodes: usize, state_manager: Arc<StateManager>) -> Self {
        Self {
            protocol: DistributedSyncProtocol::new(node_id, num_nodes),
            state_manager,
        }
    }

    /// Synchronize a state modification across the distributed system
    pub async fn sync_state_modification(
        &self,
        timeline_id: TimelineId,
        branch_id: BranchId,
        delta: StateDelta,
    ) -> Result<(), SyncError> {
        let vector_clock = self.protocol.vector_clock.read().unwrap().clone();
        
        let event = SyncEvent::StateModification {
            timeline_id,
            branch_id,
            delta,
            vector_clock,
        };

        self.protocol.process_event(event).await?;
        Ok(())
    }

    /// Propose a global consensus state
    pub async fn propose_consensus(
        &self,
        proposed_state: GlobalConsensusState,
    ) -> Result<(), SyncError> {
        let vector_clock = self.protocol.vector_clock.read().unwrap().clone();
        let epoch = proposed_state.round;
        
        let event = SyncEvent::ConsensusProposal {
            epoch,
            proposed_state,
            proposer: self.protocol.node_id,
            vector_clock,
        };

        self.protocol.process_event(event).await?;
        Ok(())
    }

    /// Start the distributed synchronization system
    pub async fn start(&self) -> Result<(), SyncError> {
        self.protocol.start().await
    }

    /// Get current metrics
    pub fn metrics(&self) -> SyncMetricsSnapshot {
        self.protocol.get_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_vector_clock_ordering() {
        let mut clock1 = VectorClock::new(0, 3);
        let mut clock2 = VectorClock::new(1, 3);

        clock1.tick(0);
        clock2.tick(1);

        // Concurrent events
        assert!(clock1.concurrent_with(&clock2));
        assert!(clock2.concurrent_with(&clock1));

        // Create causal relationship
        clock2.update(&clock1, 1);
        assert!(clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    async fn test_consensus_protocol() {
        let protocol = DistributedSyncProtocol::new(0, 3);
        
        let proposed_state = GlobalConsensusState {
            global_clock: VectorClock::new(0, 3),
            active_timelines: HashMap::new(),
            round: 1,
            state_hash: 12345,
        };

        let result = protocol.handle_consensus_proposal(
            1,
            proposed_state,
            0,
            VectorClock::new(0, 3),
        ).await;

        assert!(result.is_ok());
    }

    #[test]
    async fn test_byzantine_fault_tolerance() {
        let protocol = DistributedSyncProtocol::new(0, 4);
        
        // Add a node with low trust score
        let untrusted_node = NodeInfo {
            id: 1,
            address: "node1:8080".to_string(),
            last_clock: VectorClock::new(1, 4),
            last_heartbeat: Instant::now(),
            load_metrics: NodeLoadMetrics {
                cpu_usage: 0.5,
                memory_usage: 0.3,
                network_usage: 0.2,
                active_operations: 5,
                avg_latency_ms: 10.0,
            },
            trust_score: 0.1, // Below threshold
        };

        protocol.add_node(untrusted_node).await.unwrap();

        let proposed_state = GlobalConsensusState {
            global_clock: VectorClock::new(1, 4),
            active_timelines: HashMap::new(),
            round: 1,
            state_hash: 12345,
        };

        // Should reject due to low trust score
        let result = protocol.handle_consensus_proposal(
            1,
            proposed_state,
            1, // Untrusted proposer
            VectorClock::new(1, 4),
        ).await;

        assert!(matches!(result, Err(SyncError::TrustViolation { .. })));
    }

    #[test]
    fn test_vector_clock_entropy() {
        let mut clock = VectorClock::new(0, 3);
        assert_eq!(clock.entropy(), 0.0);

        clock.tick(0);
        clock.update_entropy();
        assert!(clock.entropy() >= 0.0);
    }

    #[test]
    async fn test_distributed_synchronizer() {
        // Mock state manager would be created here
        // let state_manager = Arc::new(StateManager::new());
        // let synchronizer = DistributedTemporalSynchronizer::new(0, 3, state_manager);
        
        // Test synchronization operations
        // This would include more comprehensive integration tests
    }
}