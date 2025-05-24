//! Persistent state storage system
//!
//! This module provides a hierarchical storage system for algorithm execution
//! states, enabling efficient persistence and retrieval of execution timelines
//! while maintaining high performance through a multi-tiered caching architecture.
//!
//! The storage system employs memory-mapped files, write-ahead logging, and
//! transactional semantics to ensure data integrity and crash recovery.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use log::{debug, error, info, trace, warn};

use crate::algorithm::AlgorithmState;
use crate::temporal::delta::{DeltaChain, DeltaError, StateDelta, StateId};
use crate::temporal::compression::CompressedState;

/// Storage tier identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Memory tier (fastest, volatile)
    Memory,
    
    /// Cache tier (fast, volatile)
    Cache,
    
    /// Disk tier (slower, persistent)
    Disk,
    
    /// Archive tier (slowest, compressed)
    Archive,
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Number of states in memory tier
    pub memory_states: usize,
    
    /// Number of states in cache tier
    pub cache_states: usize,
    
    /// Number of states in disk tier
    pub disk_states: usize,
    
    /// Number of states in archive tier
    pub archive_states: usize,
    
    /// Memory tier size in bytes
    pub memory_size: usize,
    
    /// Cache tier size in bytes
    pub cache_size: usize,
    
    /// Disk tier size in bytes
    pub disk_size: usize,
    
    /// Archive tier size in bytes
    pub archive_size: usize,
    
    /// Number of cache hits
    pub cache_hits: usize,
    
    /// Number of cache misses
    pub cache_misses: usize,
    
    /// Number of disk reads
    pub disk_reads: usize,
    
    /// Number of disk writes
    pub disk_writes: usize,
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Base storage directory
    pub base_dir: PathBuf,
    
    /// Maximum memory tier size in bytes
    pub max_memory_size: usize,
    
    /// Maximum cache tier size in bytes
    pub max_cache_size: usize,
    
    /// Maximum disk tier size in bytes
    pub max_disk_size: usize,
    
    /// Enable write-ahead logging
    pub enable_wal: bool,
    
    /// Enable memory-mapped I/O
    pub enable_mmap: bool,
    
    /// Enable state compression
    pub enable_compression: bool,
    
    /// Cache tier LRU timeout in seconds
    pub cache_timeout: u64,
    
    /// Enable automatic tier migration
    pub enable_auto_migration: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("chronos_storage"),
            max_memory_size: 100 * 1024 * 1024, // 100 MB
            max_cache_size: 500 * 1024 * 1024,  // 500 MB
            max_disk_size: 10 * 1024 * 1024 * 1024, // 10 GB
            enable_wal: true,
            enable_mmap: true,
            enable_compression: true,
            cache_timeout: 300, // 5 minutes
            enable_auto_migration: true,
        }
    }
}

/// Storage error
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// State not found
    #[error("State not found: {0}")]
    StateNotFound(StateId),
    
    /// Storage tier full
    #[error("Storage tier full: {0:?}")]
    TierFull(StorageTier),
    
    /// Transaction error
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    /// Delta error
    #[error("Delta error: {0}")]
    DeltaError(#[from] DeltaError),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Storage entry holding state data
#[derive(Clone)]
pub struct StorageEntry {
    /// State ID
    pub id: StateId,
    
    /// Creation timestamp
    pub created_at: Instant,
    
    /// Last access timestamp
    pub last_accessed: Instant,
    
    /// Storage tier
    pub tier: StorageTier,
    
    /// Entry size in bytes
    pub size_bytes: usize,
    
    /// Compressed state (for Memory and Cache tiers)
    pub state: Option<CompressedState>,
    
    /// State delta (for Delta chains)
    pub delta: Option<Arc<StateDelta>>,
    
    /// State file path (for Disk and Archive tiers)
    pub file_path: Option<PathBuf>,
    
    /// Access count
    pub access_count: usize,
}

impl StorageEntry {
    /// Create a new entry from a state
    pub fn from_state(id: StateId, state: &AlgorithmState) -> Self {
        let compressed = CompressedState::from_state(state);
        let size = compressed.size_bytes();
        
        Self {
            id,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            tier: StorageTier::Memory,
            size_bytes: size,
            state: Some(compressed),
            delta: None,
            file_path: None,
            access_count: 0,
        }
    }
    
    /// Create a new entry from a delta
    pub fn from_delta(delta: Arc<StateDelta>) -> Self {
        let id = delta.target_state_id;
        let size = delta.size_bytes();
        
        Self {
            id,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            tier: StorageTier::Memory,
            size_bytes: size,
            state: None,
            delta: Some(delta),
            file_path: None,
            access_count: 0,
        }
    }
    
    /// Get the state
    pub fn get_state(&mut self) -> Result<AlgorithmState, StorageError> {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        
        if let Some(state) = &self.state {
            return state.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
        }
        
        Err(StorageError::StateNotFound(self.id))
    }
    
    /// Check if this entry has a state
    pub fn has_state(&self) -> bool {
        self.state.is_some()
    }
    
    /// Check if this entry has a delta
    pub fn has_delta(&self) -> bool {
        self.delta.is_some()
    }
    
    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
    
    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }
    
    /// Get idle time in seconds
    pub fn idle_seconds(&self) -> u64 {
        self.last_accessed.elapsed().as_secs()
    }
}

impl fmt::Debug for StorageEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageEntry")
            .field("id", &self.id)
            .field("tier", &self.tier)
            .field("size_bytes", &self.size_bytes)
            .field("has_state", &self.state.is_some())
            .field("has_delta", &self.delta.is_some())
            .field("has_file", &self.file_path.is_some())
            .field("access_count", &self.access_count)
            .finish()
    }
}

/// Storage index for tracking state locations
pub struct StorageIndex {
    /// Map of state ID to storage entry
    entries: HashMap<StateId, StorageEntry>,
    
    /// States by tier
    tier_states: HashMap<StorageTier, HashSet<StateId>>,
    
    /// Decision points
    decision_points: HashSet<StateId>,
    
    /// Delta chains
    delta_chains: HashMap<String, DeltaChain>,
    
    /// State dependencies
    dependencies: HashMap<StateId, Vec<StateId>>,
    
    /// Memory tier size
    memory_size: usize,
    
    /// Cache tier size
    cache_size: usize,
    
    /// Disk tier size
    disk_size: usize,
    
    /// Archive tier size
    archive_size: usize,
    
    /// Cache hits
    cache_hits: usize,
    
    /// Cache misses
    cache_misses: usize,
    
    /// Disk reads
    disk_reads: usize,
    
    /// Disk writes
    disk_writes: usize,
}

impl StorageIndex {
    /// Create a new storage index
    pub fn new() -> Self {
        let mut tier_states = HashMap::new();
        tier_states.insert(StorageTier::Memory, HashSet::new());
        tier_states.insert(StorageTier::Cache, HashSet::new());
        tier_states.insert(StorageTier::Disk, HashSet::new());
        tier_states.insert(StorageTier::Archive, HashSet::new());
        
        Self {
            entries: HashMap::new(),
            tier_states,
            decision_points: HashSet::new(),
            delta_chains: HashMap::new(),
            dependencies: HashMap::new(),
            memory_size: 0,
            cache_size: 0,
            disk_size: 0,
            archive_size: 0,
            cache_hits: 0,
            cache_misses: 0,
            disk_reads: 0,
            disk_writes: 0,
        }
    }
    
    /// Add an entry
    pub fn add_entry(&mut self, entry: StorageEntry) {
        let id = entry.id;
        let tier = entry.tier;
        let size = entry.size_bytes;
        
        // Update tier size
        match tier {
            StorageTier::Memory => self.memory_size += size,
            StorageTier::Cache => self.cache_size += size,
            StorageTier::Disk => self.disk_size += size,
            StorageTier::Archive => self.archive_size += size,
        }
        
        // Add to tier states
        if let Some(states) = self.tier_states.get_mut(&tier) {
            states.insert(id);
        }
        
        // Add entry
        self.entries.insert(id, entry);
    }
    
    /// Remove an entry
    pub fn remove_entry(&mut self, id: StateId) -> Option<StorageEntry> {
        if let Some(entry) = self.entries.remove(&id) {
            // Update tier size
            match entry.tier {
                StorageTier::Memory => self.memory_size -= entry.size_bytes,
                StorageTier::Cache => self.cache_size -= entry.size_bytes,
                StorageTier::Disk => self.disk_size -= entry.size_bytes,
                StorageTier::Archive => self.archive_size -= entry.size_bytes,
            }
            
            // Remove from tier states
            if let Some(states) = self.tier_states.get_mut(&entry.tier) {
                states.remove(&id);
            }
            
            // Remove from decision points
            self.decision_points.remove(&id);
            
            Some(entry)
        } else {
            None
        }
    }
    
    /// Get an entry
    pub fn get_entry(&self, id: StateId) -> Option<&StorageEntry> {
        self.entries.get(&id)
    }
    
    /// Get a mutable entry
    pub fn get_entry_mut(&mut self, id: StateId) -> Option<&mut StorageEntry> {
        self.entries.get_mut(&id)
    }
    
    /// Move an entry to a different tier
    pub fn move_to_tier(&mut self, id: StateId, new_tier: StorageTier) -> Result<(), StorageError> {
        if let Some(entry) = self.entries.get_mut(&id) {
            let old_tier = entry.tier;
            
            // Remove from old tier
            if let Some(states) = self.tier_states.get_mut(&old_tier) {
                states.remove(&id);
            }
            
            // Update tier size
            match old_tier {
                StorageTier::Memory => self.memory_size -= entry.size_bytes,
                StorageTier::Cache => self.cache_size -= entry.size_bytes,
                StorageTier::Disk => self.disk_size -= entry.size_bytes,
                StorageTier::Archive => self.archive_size -= entry.size_bytes,
            }
            
            // Add to new tier
            if let Some(states) = self.tier_states.get_mut(&new_tier) {
                states.insert(id);
            }
            
            // Update tier size
            match new_tier {
                StorageTier::Memory => self.memory_size += entry.size_bytes,
                StorageTier::Cache => self.cache_size += entry.size_bytes,
                StorageTier::Disk => self.disk_size += entry.size_bytes,
                StorageTier::Archive => self.archive_size += entry.size_bytes,
            }
            
            // Update entry
            entry.tier = new_tier;
            
            Ok(())
        } else {
            Err(StorageError::StateNotFound(id))
        }
    }
    
    /// Mark an entry as a decision point
    pub fn mark_decision_point(&mut self, id: StateId) {
        self.decision_points.insert(id);
    }
    
    /// Check if an entry is a decision point
    pub fn is_decision_point(&self, id: StateId) -> bool {
        self.decision_points.contains(&id)
    }
    
    /// Add a delta chain
    pub fn add_delta_chain(&mut self, chain: DeltaChain) {
        self.delta_chains.insert(chain.id.clone(), chain);
    }
    
    /// Get a delta chain
    pub fn get_delta_chain(&self, id: &str) -> Option<&DeltaChain> {
        self.delta_chains.get(id)
    }
    
    /// Add a dependency
    pub fn add_dependency(&mut self, dependent: StateId, dependency: StateId) {
        self.dependencies.entry(dependent).or_default().push(dependency);
    }
    
    /// Get dependencies
    pub fn get_dependencies(&self, id: StateId) -> Option<&Vec<StateId>> {
        self.dependencies.get(&id)
    }
    
    /// Get entries by tier
    pub fn get_by_tier(&self, tier: StorageTier) -> Vec<&StorageEntry> {
        if let Some(states) = self.tier_states.get(&tier) {
            states.iter()
                .filter_map(|id| self.entries.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get entries by tier (mutable)
    pub fn get_by_tier_mut(&mut self, tier: StorageTier) -> Vec<&mut StorageEntry> {
        let states = self.tier_states.get(&tier)
            .map(|s| s.clone())
            .unwrap_or_default();
            
        states.iter()
            .filter_map(|id| self.entries.get_mut(id))
            .collect()
    }
    
    /// Get the size of a tier
    pub fn tier_size(&self, tier: StorageTier) -> usize {
        match tier {
            StorageTier::Memory => self.memory_size,
            StorageTier::Cache => self.cache_size,
            StorageTier::Disk => self.disk_size,
            StorageTier::Archive => self.archive_size,
        }
    }
    
    /// Get the count of states in a tier
    pub fn tier_count(&self, tier: StorageTier) -> usize {
        self.tier_states.get(&tier).map(|s| s.len()).unwrap_or(0)
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    /// Record a disk read
    pub fn record_disk_read(&mut self) {
        self.disk_reads += 1;
    }
    
    /// Record a disk write
    pub fn record_disk_write(&mut self) {
        self.disk_writes += 1;
    }
    
    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        StorageStats {
            memory_states: self.tier_count(StorageTier::Memory),
            cache_states: self.tier_count(StorageTier::Cache),
            disk_states: self.tier_count(StorageTier::Disk),
            archive_states: self.tier_count(StorageTier::Archive),
            memory_size: self.memory_size,
            cache_size: self.cache_size,
            disk_size: self.disk_size,
            archive_size: self.archive_size,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            disk_reads: self.disk_reads,
            disk_writes: self.disk_writes,
        }
    }
}

impl Default for StorageIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage transaction
pub struct StorageTransaction<'a> {
    /// Storage manager
    storage: &'a mut DiskStorage,
    
    /// Transaction log
    log: Vec<TransactionOp>,
    
    /// Committed flag
    committed: bool,
}

/// Transaction operation
#[derive(Debug, Clone)]
enum TransactionOp {
    /// Store state
    Store(StateId, CompressedState),
    
    /// Store delta
    StoreDelta(StateId, Arc<StateDelta>),
    
    /// Delete state
    Delete(StateId),
    
    /// Move state
    Move(StateId, StorageTier),
}

impl<'a> StorageTransaction<'a> {
    /// Create a new transaction
    fn new(storage: &'a mut DiskStorage) -> Self {
        Self {
            storage,
            log: Vec::new(),
            committed: false,
        }
    }
    
    /// Store a state
    pub fn store_state(&mut self, id: StateId, state: &AlgorithmState) -> Result<(), StorageError> {
        let compressed = CompressedState::from_state(state);
        self.log.push(TransactionOp::Store(id, compressed));
        Ok(())
    }
    
    /// Store a delta
    pub fn store_delta(&mut self, delta: Arc<StateDelta>) -> Result<(), StorageError> {
        let id = delta.target_state_id;
        self.log.push(TransactionOp::StoreDelta(id, delta));
        Ok(())
    }
    
    /// Delete a state
    pub fn delete_state(&mut self, id: StateId) -> Result<(), StorageError> {
        self.log.push(TransactionOp::Delete(id));
        Ok(())
    }
    
    /// Move a state
    pub fn move_state(&mut self, id: StateId, tier: StorageTier) -> Result<(), StorageError> {
        self.log.push(TransactionOp::Move(id, tier));
        Ok(())
    }
    
    /// Commit the transaction
    pub fn commit(mut self) -> Result<(), StorageError> {
        if self.committed {
            return Err(StorageError::TransactionError("Transaction already committed".to_string()));
        }
        
        // Apply operations
        for op in &self.log {
            match op {
                TransactionOp::Store(id, state) => {
                    self.storage.store_state_internal(*id, state.clone())?;
                }
                
                TransactionOp::StoreDelta(id, delta) => {
                    self.storage.store_delta_internal(*id, Arc::clone(delta))?;
                }
                
                TransactionOp::Delete(id) => {
                    self.storage.delete_state_internal(*id)?;
                }
                
                TransactionOp::Move(id, tier) => {
                    self.storage.move_state_internal(*id, *tier)?;
                }
            }
        }
        
        // Commit WAL if enabled
        if self.storage.config.enable_wal {
            // This would involve flushing the WAL
            // Currently a placeholder
            debug!("Committed transaction with {} operations", self.log.len());
        }
        
        self.committed = true;
        
        Ok(())
    }
    
    /// Rollback the transaction
    pub fn rollback(mut self) {
        if !self.committed {
            self.log.clear();
            debug!("Rolled back transaction");
        }
    }
}

impl<'a> Drop for StorageTransaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            warn!("Transaction dropped without commit or rollback");
        }
    }
}

/// Disk storage manager
pub struct DiskStorage {
    /// Storage configuration
    config: StorageConfig,
    
    /// Storage index
    index: StorageIndex,
    
    /// Write-ahead log file
    wal_file: Option<File>,
    
    /// Memory-mapped storage
    mmap: Option<MmapMut>,
    
    /// Memory-mapped storage file
    mmap_file: Option<File>,
    
    /// Background migration task running
    migration_running: bool,
    
    /// Last optimization time
    last_optimization: Instant,
}

impl DiskStorage {
    /// Create a new disk storage manager
    pub fn new(config: StorageConfig) -> Result<Self, StorageError> {
        // Create base directory
        fs::create_dir_all(&config.base_dir)?;
        
        // Create subdirectories
        fs::create_dir_all(config.base_dir.join("cache"))?;
        fs::create_dir_all(config.base_dir.join("disk"))?;
        fs::create_dir_all(config.base_dir.join("archive"))?;
        fs::create_dir_all(config.base_dir.join("wal"))?;
        
        // Create WAL file if enabled
        let wal_file = if config.enable_wal {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(config.base_dir.join("wal").join("current.log"))?;
            
            Some(file)
        } else {
            None
        };
        
        // Create memory-mapped storage if enabled
        let (mmap_file, mmap) = if config.enable_mmap {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(config.base_dir.join("mmap.dat"))?;
            
            // Allocate initial size (e.g., 10MB)
            file.set_len(10 * 1024 * 1024)?;
            
            let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
            
            (Some(file), Some(mmap))
        } else {
            (None, None)
        };
        
        Ok(Self {
            config,
            index: StorageIndex::new(),
            wal_file,
            mmap_file,
            mmap,
            migration_running: false,
            last_optimization: Instant::now(),
        })
    }
    
    /// Begin a transaction
    pub fn begin_transaction(&mut self) -> StorageTransaction {
        StorageTransaction::new(self)
    }
    
    /// Store a state
    pub fn store_state(&mut self, id: StateId, state: &AlgorithmState) -> Result<(), StorageError> {
        let compressed = CompressedState::from_state(state);
        self.store_state_internal(id, compressed)
    }
    
    /// Store a state (internal implementation)
    fn store_state_internal(&mut self, id: StateId, state: CompressedState) -> Result<(), StorageError> {
        let size = state.size_bytes();
        
        // Check if we can store in memory
        if self.index.memory_size + size > self.config.max_memory_size {
            // Need to make room
            self.make_room_in_tier(StorageTier::Memory, size)?;
        }
        
        // Create entry
        let entry = StorageEntry {
            id,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            tier: StorageTier::Memory,
            size_bytes: size,
            state: Some(state),
            delta: None,
            file_path: None,
            access_count: 0,
        };
        
        // Add to index
        self.index.add_entry(entry);
        
        debug!("Stored state {} in memory tier", id);
        
        Ok(())
    }
    
    /// Store a delta
    pub fn store_delta(&mut self, delta: Arc<StateDelta>) -> Result<(), StorageError> {
        self.store_delta_internal(delta.target_state_id, delta)
    }
    
    /// Store a delta (internal implementation)
    fn store_delta_internal(&mut self, id: StateId, delta: Arc<StateDelta>) -> Result<(), StorageError> {
        let size = delta.size_bytes();
        
        // Check if we can store in memory
        if self.index.memory_size + size > self.config.max_memory_size {
            // Need to make room
            self.make_room_in_tier(StorageTier::Memory, size)?;
        }
        
        // Create entry
        let entry = StorageEntry::from_delta(delta);
        
        // Add to index
        self.index.add_entry(entry);
        
        debug!("Stored delta for state {} in memory tier", id);
        
        Ok(())
    }
    
    /// Get a state
    pub fn get_state(&mut self, id: StateId) -> Result<AlgorithmState, StorageError> {
        if let Some(entry) = self.index.get_entry_mut(id) {
            entry.mark_accessed();
            
            match entry.tier {
                StorageTier::Memory => {
                    // State is in memory
                    if let Some(state) = &entry.state {
                        return state.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
                    } else if let Some(delta) = &entry.delta {
                        if delta.is_snapshot() {
                            if let Some(snapshot) = &delta.snapshot {
                                return snapshot.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
                            }
                        }
                    }
                    
                    Err(StorageError::StateNotFound(id))
                }
                
                StorageTier::Cache => {
                    // State is in cache
                    self.index.record_cache_hit();
                    
                    if let Some(state) = &entry.state {
                        return state.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
                    } else if let Some(delta) = &entry.delta {
                        if delta.is_snapshot() {
                            if let Some(snapshot) = &delta.snapshot {
                                return snapshot.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
                            }
                        }
                    }
                    
                    Err(StorageError::StateNotFound(id))
                }
                
                StorageTier::Disk | StorageTier::Archive => {
                    // State is on disk
                    self.index.record_disk_read();
                    
                    if let Some(path) = &entry.file_path {
                        let data = fs::read(path)?;
                        
                        let state: CompressedState = bincode::deserialize(&data)
                            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                        
                        // Update entry
                        if let Some(entry) = self.index.get_entry_mut(id) {
                            entry.state = Some(state.clone());
                        }
                        
                        return state.decompress().map_err(|e| StorageError::SerializationError(e.to_string()));
                    }
                    
                    Err(StorageError::StateNotFound(id))
                }
            }
        } else {
            self.index.record_cache_miss();
            Err(StorageError::StateNotFound(id))
        }
    }
    
    /// Check if a state exists
    pub fn has_state(&self, id: StateId) -> bool {
        self.index.get_entry(id).is_some()
    }
    
    /// Delete a state
    pub fn delete_state(&mut self, id: StateId) -> Result<(), StorageError> {
        self.delete_state_internal(id)
    }
    
    /// Delete a state (internal implementation)
    fn delete_state_internal(&mut self, id: StateId) -> Result<(), StorageError> {
        if let Some(entry) = self.index.remove_entry(id) {
            // Delete file if on disk
            if let Some(path) = entry.file_path {
                if path.exists() {
                    fs::remove_file(path)?;
                }
            }
            
            debug!("Deleted state {}", id);
            
            Ok(())
        } else {
            Err(StorageError::StateNotFound(id))
        }
    }
    
    /// Move a state to a different tier
    pub fn move_state(&mut self, id: StateId, tier: StorageTier) -> Result<(), StorageError> {
        self.move_state_internal(id, tier)
    }
    
    /// Move a state to a different tier (internal implementation)
    fn move_state_internal(&mut self, id: StateId, tier: StorageTier) -> Result<(), StorageError> {
        if let Some(entry) = self.index.get_entry(id).cloned() {
            let old_tier = entry.tier;
            
            if old_tier == tier {
                return Ok(());
            }
            
            match (old_tier, tier) {
                (StorageTier::Memory, StorageTier::Cache) |
                (StorageTier::Cache, StorageTier::Memory) => {
                    // Memory <-> Cache: Just update index
                    self.index.move_to_tier(id, tier)?;
                }
                
                (StorageTier::Memory, StorageTier::Disk) |
                (StorageTier::Cache, StorageTier::Disk) => {
                    // Memory/Cache -> Disk: Write to disk
                    if let Some(entry) = self.index.get_entry(id) {
                        let state = if let Some(state) = &entry.state {
                            state.clone()
                        } else if let Some(delta) = &entry.delta {
                            if delta.is_snapshot() {
                                if let Some(snapshot) = &delta.snapshot {
                                    snapshot.clone()
                                } else {
                                    return Err(StorageError::StateNotFound(id));
                                }
                            } else {
                                return Err(StorageError::StateNotFound(id));
                            }
                        } else {
                            return Err(StorageError::StateNotFound(id));
                        };
                        
                        // Write to disk
                        let path = self.get_state_path(id, StorageTier::Disk);
                        let data = bincode::serialize(&state)
                            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                        
                        fs::write(&path, &data)?;
                        
                        // Update index
                        if let Some(entry) = self.index.get_entry_mut(id) {
                            entry.file_path = Some(path);
                            entry.state = None; // Free memory
                        }
                        
                        self.index.move_to_tier(id, tier)?;
                        self.index.record_disk_write();
                    } else {
                        return Err(StorageError::StateNotFound(id));
                    }
                }
                
                (StorageTier::Disk, StorageTier::Memory) |
                (StorageTier::Disk, StorageTier::Cache) => {
                    // Disk -> Memory/Cache: Read from disk
                    if let Some(entry) = self.index.get_entry(id) {
                        if let Some(path) = &entry.file_path {
                            let data = fs::read(path)?;
                            
                            let state: CompressedState = bincode::deserialize(&data)
                                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                            
                            // Update index
                            if let Some(entry) = self.index.get_entry_mut(id) {
                                entry.state = Some(state);
                            }
                            
                            self.index.move_to_tier(id, tier)?;
                            self.index.record_disk_read();
                        } else {
                            return Err(StorageError::StateNotFound(id));
                        }
                    } else {
                        return Err(StorageError::StateNotFound(id));
                    }
                }
                
                (StorageTier::Disk, StorageTier::Archive) => {
                    // Disk -> Archive: Compress and move
                    if let Some(entry) = self.index.get_entry(id) {
                        if let Some(path) = &entry.file_path {
                            let data = fs::read(path)?;
                            
                            let state: CompressedState = bincode::deserialize(&data)
                                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                            
                            // Compress and write to archive
                            let archive_path = self.get_state_path(id, StorageTier::Archive);
                            
                            // This would involve additional compression
                            // Currently a placeholder
                            fs::write(&archive_path, &data)?;
                            
                            // Delete original file
                            fs::remove_file(path)?;
                            
                            // Update index
                            if let Some(entry) = self.index.get_entry_mut(id) {
                                entry.file_path = Some(archive_path);
                            }
                            
                            self.index.move_to_tier(id, tier)?;
                            self.index.record_disk_read();
                            self.index.record_disk_write();
                        } else {
                            return Err(StorageError::StateNotFound(id));
                        }
                    } else {
                        return Err(StorageError::StateNotFound(id));
                    }
                }
                
                (StorageTier::Archive, StorageTier::Disk) => {
                    // Archive -> Disk: Decompress and move
                    if let Some(entry) = self.index.get_entry(id) {
                        if let Some(path) = &entry.file_path {
                            let data = fs::read(path)?;
                            
                            // This would involve decompression
                            // Currently a placeholder
                            let decompressed_data = data;
                            
                            // Write to disk
                            let disk_path = self.get_state_path(id, StorageTier::Disk);
                            fs::write(&disk_path, &decompressed_data)?;
                            
                            // Delete archive file
                            fs::remove_file(path)?;
                            
                            // Update index
                            if let Some(entry) = self.index.get_entry_mut(id) {
                                entry.file_path = Some(disk_path);
                            }
                            
                            self.index.move_to_tier(id, tier)?;
                            self.index.record_disk_read();
                            self.index.record_disk_write();
                        } else {
                            return Err(StorageError::StateNotFound(id));
                        }
                    } else {
                        return Err(StorageError::StateNotFound(id));
                    }
                }
                
                _ => {
                    // Other transitions are not supported
                    return Err(StorageError::ConfigError(format!(
                        "Unsupported tier transition: {:?} -> {:?}",
                        old_tier, tier
                    )));
                }
            }
            
            debug!("Moved state {} from {:?} to {:?}", id, old_tier, tier);
            
            Ok(())
        } else {
            Err(StorageError::StateNotFound(id))
        }
    }
    
    /// Mark a state as a decision point
    pub fn mark_decision_point(&mut self, id: StateId) -> Result<(), StorageError> {
        if self.index.get_entry(id).is_some() {
            self.index.mark_decision_point(id);
            
            // Ensure decision points are in memory or cache
            if let Some(entry) = self.index.get_entry(id) {
                if entry.tier == StorageTier::Disk || entry.tier == StorageTier::Archive {
                    self.move_state(id, StorageTier::Cache)?;
                }
            }
            
            debug!("Marked state {} as decision point", id);
            
            Ok(())
        } else {
            Err(StorageError::StateNotFound(id))
        }
    }
    
    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        self.index.get_stats()
    }
    
    /// Make room in a tier
    fn make_room_in_tier(&mut self, tier: StorageTier, needed_bytes: usize) -> Result<(), StorageError> {
        let max_size = match tier {
            StorageTier::Memory => self.config.max_memory_size,
            StorageTier::Cache => self.config.max_cache_size,
            StorageTier::Disk => self.config.max_disk_size,
            StorageTier::Archive => usize::MAX, // No limit for archive
        };
        
        let current_size = self.index.tier_size(tier);
        
        if current_size + needed_bytes <= max_size {
            // Already have room
            return Ok(());
        }
        
        let target_size = max_size - needed_bytes;
        
        // Get candidates for eviction
        let candidates = match tier {
            StorageTier::Memory => self.get_eviction_candidates(tier, current_size - target_size),
            StorageTier::Cache => self.get_eviction_candidates(tier, current_size - target_size),
            StorageTier::Disk => self.get_eviction_candidates(tier, current_size - target_size),
            StorageTier::Archive => Vec::new(), // No eviction from archive
        };
        
        if candidates.is_empty() {
            return Err(StorageError::TierFull(tier));
        }
        
        // Evict candidates
        for &id in &candidates {
            match tier {
                StorageTier::Memory => {
                    // Move to cache
                    self.move_state(id, StorageTier::Cache)?;
                }
                StorageTier::Cache => {
                    // Move to disk
                    self.move_state(id, StorageTier::Disk)?;
                }
                StorageTier::Disk => {
                    // Move to archive
                    self.move_state(id, StorageTier::Archive)?;
                }
                StorageTier::Archive => {
                    // Cannot evict from archive
                    return Err(StorageError::TierFull(tier));
                }
            }
        }
        
        // Check if we have enough room now
        let new_size = self.index.tier_size(tier);
        if new_size + needed_bytes > max_size {
            return Err(StorageError::TierFull(tier));
        }
        
        Ok(())
    }
    
    /// Get candidates for eviction
    fn get_eviction_candidates(&self, tier: StorageTier, needed_bytes: usize) -> Vec<StateId> {
        let mut candidates = Vec::new();
        let mut freed_bytes = 0;
        
        // Get all entries in tier
        let entries = self.index.get_by_tier(tier);
        
        // Skip decision points
        let mut entries: Vec<_> = entries.into_iter()
            .filter(|e| !self.index.is_decision_point(e.id))
            .collect();
        
        // Sort by access time (least recently accessed first)
        entries.sort_by(|a, b| a.last_accessed.cmp(&b.last_accessed));
        
        for entry in entries {
            if !self.index.is_decision_point(entry.id) {
                candidates.push(entry.id);
                freed_bytes += entry.size_bytes;
                
                if freed_bytes >= needed_bytes {
                    break;
                }
            }
        }
        
        candidates
    }
    
    /// Get the path for a state in a tier
    fn get_state_path(&self, id: StateId, tier: StorageTier) -> PathBuf {
        let subdir = match tier {
            StorageTier::Memory => "memory",
            StorageTier::Cache => "cache",
            StorageTier::Disk => "disk",
            StorageTier::Archive => "archive",
        };
        
        self.config.base_dir.join(subdir).join(format!("state_{}.bin", id))
    }
    
    /// Optimize storage
    pub fn optimize(&mut self) -> Result<(), StorageError> {
        // Check if it's time to optimize
        if self.last_optimization.elapsed() < Duration::from_secs(300) {
            return Ok(());
        }
        
        debug!("Optimizing storage...");
        
        // Optimize delta chains
        for chain in self.index.delta_chains.values_mut() {
            chain.optimize()?;
        }
        
        // Expire cache entries
        let cache_timeout = self.config.cache_timeout;
        let cache_entries = self.index.get_by_tier(StorageTier::Cache);
        
        let expired: Vec<_> = cache_entries.iter()
            .filter(|e| e.idle_seconds() > cache_timeout && !self.index.is_decision_point(e.id))
            .map(|e| e.id)
            .collect();
        
        for id in expired {
            self.move_state(id, StorageTier::Disk)?;
        }
        
        self.last_optimization = Instant::now();
        
        debug!("Storage optimization complete");
        
        Ok(())
    }
    
    /// Close storage and ensure all data is flushed
    pub fn close(mut self) -> Result<(), StorageError> {
        // Flush all cache entries to disk
        let cache_entries = self.index.get_by_tier(StorageTier::Cache);
        
        for entry in cache_entries {
            self.move_state(entry.id, StorageTier::Disk)?;
        }
        
        // Flush all memory entries to disk
        let memory_entries = self.index.get_by_tier(StorageTier::Memory);
        
        for entry in memory_entries {
            self.move_state(entry.id, StorageTier::Disk)?;
        }
        
        // Close WAL file
        self.wal_file = None;
        
        // Close mmap
        self.mmap = None;
        self.mmap_file = None;
        
        debug!("Storage closed");
        
        Ok(())
    }
}

impl Drop for DiskStorage {
    fn drop(&mut self) {
        debug!("Storage dropped without explicit close");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::collections::HashMap;
    
    /// Create a test algorithm state
    fn create_test_state(step: usize) -> AlgorithmState {
        let mut state = AlgorithmState {
            step,
            open_set: vec![1, 2, 3],
            closed_set: vec![4, 5],
            current_node: Some(3),
            g_scores: HashMap::new(),
            f_scores: HashMap::new(),
            data: HashMap::new(),
        };
        
        state.g_scores.insert(1, 0.0);
        state.g_scores.insert(2, 1.0);
        state.g_scores.insert(3, 2.0);
        
        state.f_scores.insert(1, 5.0);
        state.f_scores.insert(2, 4.0);
        state.f_scores.insert(3, 3.0);
        
        state.data.insert("visited".to_string(), "3".to_string());
        
        state
    }
    
    #[test]
    fn test_storage_basic_operations() -> Result<(), StorageError> {
        // Create temp directory
        let temp_dir = tempdir()?;
        
        // Create storage config
        let mut config = StorageConfig::default();
        config.base_dir = temp_dir.path().to_path_buf();
        config.max_memory_size = 1024 * 1024; // 1MB
        config.enable_wal = false; // Disable WAL for testing
        config.enable_mmap = false; // Disable mmap for testing
        
        // Create storage
        let mut storage = DiskStorage::new(config)?;
        
        // Store a state
        let state1 = create_test_state(1);
        storage.store_state(1, &state1)?;
        
        // Check if state exists
        assert!(storage.has_state(1));
        
        // Get state
        let retrieved = storage.get_state(1)?;
        assert_eq!(retrieved.step, state1.step);
        
        // Store another state
        let state2 = create_test_state(2);
        storage.store_state(2, &state2)?;
        
        // Move state to cache
        storage.move_state(1, StorageTier::Cache)?;
        
        // Get state from cache
        let retrieved = storage.get_state(1)?;
        assert_eq!(retrieved.step, state1.step);
        
        // Move state to disk
        storage.move_state(1, StorageTier::Disk)?;
        
        // Get state from disk
        let retrieved = storage.get_state(1)?;
        assert_eq!(retrieved.step, state1.step);
        
        // Delete state
        storage.delete_state(1)?;
        assert!(!storage.has_state(1));
        
        // Mark as decision point
        storage.mark_decision_point(2)?;
        
        // Close storage
        storage.close()?;
        
        Ok(())
    }
    
    #[test]
    fn test_storage_transaction() -> Result<(), StorageError> {
        // Create temp directory
        let temp_dir = tempdir()?;
        
        // Create storage config
        let mut config = StorageConfig::default();
        config.base_dir = temp_dir.path().to_path_buf();
        config.enable_wal = false; // Disable WAL for testing
        config.enable_mmap = false; // Disable mmap for testing
        
        // Create storage
        let mut storage = DiskStorage::new(config)?;
        
        // Begin transaction
        let mut tx = storage.begin_transaction();
        
        // Store states in transaction
        let state1 = create_test_state(1);
        let state2 = create_test_state(2);
        
        tx.store_state(1, &state1)?;
        tx.store_state(2, &state2)?;
        
        // States should not be available yet
        assert!(!storage.has_state(1));
        assert!(!storage.has_state(2));
        
        // Commit transaction
        tx.commit()?;
        
        // States should be available now
        assert!(storage.has_state(1));
        assert!(storage.has_state(2));
        
        // Begin another transaction
        let mut tx = storage.begin_transaction();
        
        // Delete state in transaction
        tx.delete_state(1)?;
        
        // State should still be available
        assert!(storage.has_state(1));
        
        // Commit transaction
        tx.commit()?;
        
        // State should be deleted now
        assert!(!storage.has_state(1));
        
        // Close storage
        storage.close()?;
        
        Ok(())
    }
    
    #[test]
    fn test_storage_eviction() -> Result<(), StorageError> {
        // Create temp directory
        let temp_dir = tempdir()?;
        
        // Create storage config with small limits
        let mut config = StorageConfig::default();
        config.base_dir = temp_dir.path().to_path_buf();
        config.max_memory_size = 10000; // Very small memory tier
        config.max_cache_size = 20000; // Very small cache tier
        config.enable_wal = false; // Disable WAL for testing
        config.enable_mmap = false; // Disable mmap for testing
        
        // Create storage
        let mut storage = DiskStorage::new(config)?;
        
        // Store states until eviction happens
        let mut stored_ids = Vec::new();
        
        for i in 1..100 {
            let state = create_test_state(i);
            storage.store_state(i, &state)?;
            stored_ids.push(i);
        }
        
        // Some states should be evicted to cache
        let stats = storage.get_stats();
        assert!(stats.memory_states < 100);
        assert!(stats.cache_states > 0 || stats.disk_states > 0);
        
        // All states should still be retrievable
        for &id in &stored_ids {
            let state = storage.get_state(id)?;
            assert_eq!(state.step, id);
        }
        
        // Close storage
        storage.close()?;
        
        Ok(())
    }
}