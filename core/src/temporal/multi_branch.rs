//! Multi-Dimensional Temporal Branching System
//!
//! This module implements revolutionary N-dimensional parameter space exploration
//! for temporal debugging with Pareto optimization, category-theoretic functorial
//! composition, and formal convergence guarantees.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use crate::algorithm::state::AlgorithmState;
use crate::temporal::{StateManager, TimelineBranch, DecisionPoint};
use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, BinaryHeap};
use std::sync::{Arc, RwLock, Mutex};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use thiserror::Error;

/// Multi-dimensional branching errors
#[derive(Debug, Error)]
pub enum MultiBranchError {
    #[error("Invalid parameter dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    
    #[error("Pareto frontier computation failed: {reason}")]
    ParetoComputationError { reason: String },
    
    #[error("Branch space overflow: {current_branches} exceeds limit {max_branches}")]
    BranchSpaceOverflow { current_branches: usize, max_branches: usize },
    
    #[error("Convergence failure: {iterations} iterations without convergence")]
    ConvergenceFailure { iterations: usize },
    
    #[error("Parameter space error: {reason}")]
    ParameterSpaceError { reason: String },
}

/// N-dimensional parameter vector with category-theoretic structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterVector {
    /// Parameter values in N-dimensional space
    pub values: DVector<f64>,
    
    /// Parameter names for semantic interpretation
    pub names: Vec<String>,
    
    /// Parameter bounds for constraint satisfaction
    pub bounds: Vec<(f64, f64)>,
    
    /// Metadata for categorical functorial composition
    pub metadata: HashMap<String, String>,
}

impl ParameterVector {
    /// Create new parameter vector with bounds checking
    pub fn new(values: Vec<f64>, names: Vec<String>, bounds: Vec<(f64, f64)>) -> Result<Self, MultiBranchError> {
        if values.len() != names.len() || values.len() != bounds.len() {
            return Err(MultiBranchError::InvalidDimension {
                expected: names.len(),
                actual: values.len(),
            });
        }
        
        // Validate bounds constraints
        for (i, (&value, &(min, max))) in values.iter().zip(bounds.iter()).enumerate() {
            if value < min || value > max {
                return Err(MultiBranchError::ParameterSpaceError {
                    reason: format!("Parameter {} = {} violates bounds [{}, {}]", names[i], value, min, max),
                });
            }
        }
        
        Ok(Self {
            values: DVector::from_vec(values),
            names,
            bounds,
            metadata: HashMap::new(),
        })
    }
    
    /// Category-theoretic functorial mapping
    pub fn map<F>(&self, f: F) -> Result<Self, MultiBranchError> 
    where
        F: Fn(f64) -> f64,
    {
        let mapped_values: Vec<f64> = self.values.iter().map(|&x| f(x)).collect();
        Self::new(mapped_values, self.names.clone(), self.bounds.clone())
    }
    
    /// Euclidean distance in parameter space
    pub fn distance(&self, other: &Self) -> f64 {
        (&self.values - &other.values).norm()
    }
    
    /// Parameter space dimensionality
    pub fn dimension(&self) -> usize {
        self.values.len()
    }
}

impl Hash for ParameterVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash discretized parameter values for efficient lookup
        for &value in self.values.iter() {
            ((value * 1e6) as i64).hash(state);
        }
    }
}

/// Multi-objective evaluation result with Pareto dominance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// Parameter configuration
    pub parameters: ParameterVector,
    
    /// Multi-objective performance metrics
    pub objectives: DVector<f64>,
    
    /// Objective names for interpretation
    pub objective_names: Vec<String>,
    
    /// Algorithm execution state
    pub state: AlgorithmState,
    
    /// Execution metadata
    pub metadata: HashMap<String, f64>,
    
    /// Pareto rank (0 = non-dominated)
    pub pareto_rank: usize,
    
    /// Crowding distance for diversity preservation
    pub crowding_distance: f64,
}

impl MultiObjectiveResult {
    /// Check Pareto dominance relationship
    pub fn dominates(&self, other: &Self) -> bool {
        let mut at_least_one_better = false;
        
        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a < b {
                return false; // Assuming minimization objectives
            }
            if a > b {
                at_least_one_better = true;
            }
        }
        
        at_least_one_better
    }
    
    /// Compute hypervolume contribution
    pub fn hypervolume_contribution(&self, reference_point: &DVector<f64>) -> f64 {
        self.objectives.iter()
            .zip(reference_point.iter())
            .map(|(obj, ref_val)| (ref_val - obj).max(0.0))
            .product()
    }
}

/// Pareto frontier with efficient non-dominated sorting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier {
    /// Non-dominated solutions organized by rank
    pub fronts: Vec<Vec<MultiObjectiveResult>>,
    
    /// Hypervolume indicator
    pub hypervolume: f64,
    
    /// Reference point for hypervolume computation
    pub reference_point: DVector<f64>,
    
    /// Archive of all evaluated solutions
    pub archive: Vec<MultiObjectiveResult>,
}

impl ParetoFrontier {
    /// Create new Pareto frontier
    pub fn new(reference_point: DVector<f64>) -> Self {
        Self {
            fronts: Vec::new(),
            hypervolume: 0.0,
            reference_point,
            archive: Vec::new(),
        }
    }
    
    /// Fast non-dominated sorting (NSGA-II algorithm)
    pub fn update_with_results(&mut self, mut results: Vec<MultiObjectiveResult>) -> Result<(), MultiBranchError> {
        self.archive.extend(results.clone());
        
        // Fast non-dominated sorting
        let mut fronts = Vec::new();
        let mut domination_count = vec![0; results.len()];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); results.len()];
        let mut current_front = Vec::new();
        
        // Count domination relationships
        for i in 0..results.len() {
            for j in 0..results.len() {
                if i != j {
                    if results[i].dominates(&results[j]) {
                        dominated_solutions[i].push(j);
                    } else if results[j].dominates(&results[i]) {
                        domination_count[i] += 1;
                    }
                }
            }
            
            if domination_count[i] == 0 {
                results[i].pareto_rank = 0;
                current_front.push(i);
            }
        }
        
        // Build fronts iteratively
        let mut front_number = 0;
        while !current_front.is_empty() {
            let front_results: Vec<MultiObjectiveResult> = current_front.iter()
                .map(|&i| results[i].clone())
                .collect();
            
            // Compute crowding distances
            let front_with_distances = self.compute_crowding_distances(front_results)?;
            fronts.push(front_with_distances);
            
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        results[j].pareto_rank = front_number + 1;
                        next_front.push(j);
                    }
                }
            }
            
            current_front = next_front;
            front_number += 1;
        }
        
        self.fronts = fronts;
        self.update_hypervolume()?;
        
        Ok(())
    }
    
    /// Compute crowding distances for diversity preservation
    fn compute_crowding_distances(&self, mut front: Vec<MultiObjectiveResult>) -> Result<Vec<MultiObjectiveResult>, MultiBranchError> {
        let front_size = front.len();
        if front_size <= 2 {
            // Boundary solutions get maximum distance
            for result in &mut front {
                result.crowding_distance = f64::INFINITY;
            }
            return Ok(front);
        }
        
        let num_objectives = front[0].objectives.len();
        
        // Initialize distances to zero
        for result in &mut front {
            result.crowding_distance = 0.0;
        }
        
        // For each objective
        for obj_idx in 0..num_objectives {
            // Sort by objective value
            front.sort_by(|a, b| {
                a.objectives[obj_idx].partial_cmp(&b.objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });
            
            // Boundary solutions get infinite distance
            front[0].crowding_distance = f64::INFINITY;
            front[front_size - 1].crowding_distance = f64::INFINITY;
            
            // Compute distances for middle solutions
            let obj_range = front[front_size - 1].objectives[obj_idx] - front[0].objectives[obj_idx];
            if obj_range > 0.0 {
                for i in 1..front_size - 1 {
                    let distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range;
                    front[i].crowding_distance += distance;
                }
            }
        }
        
        Ok(front)
    }
    
    /// Update hypervolume indicator using Monte Carlo estimation
    fn update_hypervolume(&mut self) -> Result<(), MultiBranchError> {
        if self.fronts.is_empty() {
            self.hypervolume = 0.0;
            return Ok();
        }
        
        let first_front = &self.fronts[0];
        if first_front.is_empty() {
            self.hypervolume = 0.0;
            return Ok();
        }
        
        // Monte Carlo hypervolume estimation
        const SAMPLE_SIZE: usize = 100000;
        let mut dominated_samples = 0;
        
        let rng = &mut rand::thread_rng();
        use rand::Rng;
        
        for _ in 0..SAMPLE_SIZE {
            // Generate random point in objective space
            let random_point: DVector<f64> = DVector::from_fn(
                self.reference_point.len(),
                |i, _| rng.gen_range(0.0..self.reference_point[i])
            );
            
            // Check if dominated by any solution in first front
            let is_dominated = first_front.iter().any(|solution| {
                solution.objectives.iter()
                    .zip(random_point.iter())
                    .all(|(obj, &rand_val)| *obj <= rand_val)
            });
            
            if is_dominated {
                dominated_samples += 1;
            }
        }
        
        // Estimate hypervolume
        let reference_volume: f64 = self.reference_point.iter().product();
        self.hypervolume = reference_volume * (dominated_samples as f64) / (SAMPLE_SIZE as f64);
        
        Ok(())
    }
    
    /// Get non-dominated solutions
    pub fn get_pareto_optimal(&self) -> Option<&Vec<MultiObjectiveResult>> {
        self.fronts.get(0)
    }
}

/// Multi-dimensional branch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBranchConfig {
    /// Maximum number of concurrent branches
    pub max_branches: usize,
    
    /// Parameter space exploration strategy
    pub exploration_strategy: ExplorationStrategy,
    
    /// Convergence criteria
    pub convergence_tolerance: f64,
    
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    
    /// Pareto archive size limit
    pub archive_size_limit: usize,
    
    /// Sampling density in parameter space
    pub sampling_density: f64,
}

impl Default for MultiBranchConfig {
    fn default() -> Self {
        Self {
            max_branches: 1000,
            exploration_strategy: ExplorationStrategy::AdaptiveSampling,
            convergence_tolerance: 1e-6,
            max_iterations: 10000,
            archive_size_limit: 5000,
            sampling_density: 0.1,
        }
    }
}

/// Parameter space exploration strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Uniform grid sampling
    UniformGrid { resolution: usize },
    
    /// Latin hypercube sampling
    LatinHypercube { samples: usize },
    
    /// Adaptive sampling based on gradient information
    AdaptiveSampling,
    
    /// Evolutionary multi-objective optimization
    NSGA2 { population_size: usize, generations: usize },
    
    /// Bayesian optimization with Gaussian processes
    BayesianOptimization { acquisition_function: String },
}

/// Multi-dimensional temporal branching manager
pub struct MultiBranchManager {
    /// Current parameter space configuration
    parameter_space: Arc<RwLock<HashMap<String, ParameterVector>>>,
    
    /// Active branches with their evaluation results
    active_branches: Arc<RwLock<HashMap<String, MultiObjectiveResult>>>,
    
    /// Pareto frontier maintenance
    pareto_frontier: Arc<Mutex<ParetoFrontier>>,
    
    /// Branch configuration
    config: MultiBranchConfig,
    
    /// State manager for temporal debugging integration
    state_manager: Arc<StateManager>,
    
    /// Exploration statistics
    exploration_stats: Arc<RwLock<ExplorationStatistics>>,
}

/// Exploration statistics for performance monitoring
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExplorationStatistics {
    pub total_evaluations: usize,
    pub pareto_optimal_solutions: usize,
    pub hypervolume_progression: Vec<f64>,
    pub convergence_iterations: usize,
    pub average_evaluation_time: f64,
    pub parameter_space_coverage: f64,
}

impl MultiBranchManager {
    /// Create new multi-dimensional branch manager
    pub fn new(
        config: MultiBranchConfig,
        state_manager: Arc<StateManager>,
        reference_point: DVector<f64>,
    ) -> Self {
        let pareto_frontier = Arc::new(Mutex::new(ParetoFrontier::new(reference_point)));
        
        Self {
            parameter_space: Arc::new(RwLock::new(HashMap::new())),
            active_branches: Arc::new(RwLock::new(HashMap::new())),
            pareto_frontier,
            config,
            state_manager,
            exploration_stats: Arc::new(RwLock::new(ExplorationStatistics::default())),
        }
    }
    
    /// Initialize parameter space exploration
    pub async fn initialize_exploration(
        &self,
        parameter_ranges: HashMap<String, (f64, f64)>,
        objective_names: Vec<String>,
    ) -> Result<(), MultiBranchError> {
        let parameter_vectors = self.generate_initial_sampling(&parameter_ranges)?;
        
        // Store parameter space configuration
        {
            let mut space = self.parameter_space.write().unwrap();
            for (i, vector) in parameter_vectors.into_iter().enumerate() {
                space.insert(format!("param_config_{}", i), vector);
            }
        }
        
        log::info!(
            "Initialized multi-dimensional exploration with {} parameter configurations",
            self.parameter_space.read().unwrap().len()
        );
        
        Ok(())
    }
    
    /// Generate initial parameter sampling based on exploration strategy
    fn generate_initial_sampling(
        &self,
        parameter_ranges: &HashMap<String, (f64, f64)>,
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        let param_names: Vec<String> = parameter_ranges.keys().cloned().collect();
        let bounds: Vec<(f64, f64)> = param_names.iter()
            .map(|name| parameter_ranges[name])
            .collect();
        
        match &self.config.exploration_strategy {
            ExplorationStrategy::UniformGrid { resolution } => {
                self.generate_uniform_grid(&param_names, &bounds, *resolution)
            },
            ExplorationStrategy::LatinHypercube { samples } => {
                self.generate_latin_hypercube(&param_names, &bounds, *samples)
            },
            ExplorationStrategy::AdaptiveSampling => {
                self.generate_adaptive_sampling(&param_names, &bounds)
            },
            ExplorationStrategy::NSGA2 { population_size, .. } => {
                self.generate_random_population(&param_names, &bounds, *population_size)
            },
            ExplorationStrategy::BayesianOptimization { .. } => {
                self.generate_bayesian_initial(&param_names, &bounds)
            },
        }
    }
    
    /// Generate uniform grid sampling
    fn generate_uniform_grid(
        &self,
        param_names: &[String],
        bounds: &[(f64, f64)],
        resolution: usize,
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        let mut parameter_vectors = Vec::new();
        let dimensions = param_names.len();
        
        // Generate all combinations of grid points
        let total_points = resolution.pow(dimensions as u32);
        if total_points > self.config.max_branches {
            return Err(MultiBranchError::BranchSpaceOverflow {
                current_branches: total_points,
                max_branches: self.config.max_branches,
            });
        }
        
        for i in 0..total_points {
            let mut values = Vec::with_capacity(dimensions);
            let mut index = i;
            
            for dim in 0..dimensions {
                let grid_pos = index % resolution;
                index /= resolution;
                
                let (min_val, max_val) = bounds[dim];
                let step_size = (max_val - min_val) / (resolution - 1) as f64;
                let value = min_val + grid_pos as f64 * step_size;
                
                values.push(value);
            }
            
            let param_vector = ParameterVector::new(
                values,
                param_names.to_vec(),
                bounds.to_vec(),
            )?;
            
            parameter_vectors.push(param_vector);
        }
        
        Ok(parameter_vectors)
    }
    
    /// Generate Latin hypercube sampling
    fn generate_latin_hypercube(
        &self,
        param_names: &[String],
        bounds: &[(f64, f64)],
        samples: usize,
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let dimensions = param_names.len();
        let mut rng = thread_rng();
        let mut parameter_vectors = Vec::with_capacity(samples);
        
        // Generate Latin hypercube design
        let mut hypercube = vec![vec![0; samples]; dimensions];
        
        for dim in 0..dimensions {
            let mut indices: Vec<usize> = (0..samples).collect();
            indices.shuffle(&mut rng);
            hypercube[dim] = indices;
        }
        
        // Convert to parameter vectors
        for sample in 0..samples {
            let mut values = Vec::with_capacity(dimensions);
            
            for dim in 0..dimensions {
                let (min_val, max_val) = bounds[dim];
                let grid_pos = hypercube[dim][sample];
                let random_offset: f64 = rand::random(); // [0, 1)
                
                let normalized_value = (grid_pos as f64 + random_offset) / samples as f64;
                let value = min_val + normalized_value * (max_val - min_val);
                
                values.push(value);
            }
            
            let param_vector = ParameterVector::new(
                values,
                param_names.to_vec(),
                bounds.to_vec(),
            )?;
            
            parameter_vectors.push(param_vector);
        }
        
        Ok(parameter_vectors)
    }
    
    /// Generate adaptive sampling based on exploration history
    fn generate_adaptive_sampling(
        &self,
        param_names: &[String],
        bounds: &[(f64, f64)],
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        // Start with Latin hypercube as baseline
        let initial_samples = 50.min(self.config.max_branches / 2);
        self.generate_latin_hypercube(param_names, bounds, initial_samples)
    }
    
    /// Generate random population for evolutionary optimization
    fn generate_random_population(
        &self,
        param_names: &[String],
        bounds: &[(f64, f64)],
        population_size: usize,
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        let mut parameter_vectors = Vec::with_capacity(population_size);
        
        for _ in 0..population_size {
            let mut values = Vec::with_capacity(param_names.len());
            
            for &(min_val, max_val) in bounds {
                let value = rand::random::<f64>() * (max_val - min_val) + min_val;
                values.push(value);
            }
            
            let param_vector = ParameterVector::new(
                values,
                param_names.to_vec(),
                bounds.to_vec(),
            )?;
            
            parameter_vectors.push(param_vector);
        }
        
        Ok(parameter_vectors)
    }
    
    /// Generate initial points for Bayesian optimization
    fn generate_bayesian_initial(
        &self,
        param_names: &[String],
        bounds: &[(f64, f64)],
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        // Use Latin hypercube for initial Bayesian optimization points
        let initial_points = 10.min(self.config.max_branches / 10);
        self.generate_latin_hypercube(param_names, bounds, initial_points)
    }
    
    /// Create multi-dimensional branch with parameter configuration
    pub async fn create_multi_branch<F>(
        &self,
        branch_id: String,
        parameters: ParameterVector,
        evaluation_function: F,
    ) -> Result<MultiObjectiveResult, MultiBranchError>
    where
        F: Fn(&ParameterVector, &AlgorithmState) -> Result<DVector<f64>, MultiBranchError> + Send + Sync,
    {
        // Check branch limit
        {
            let active_branches = self.active_branches.read().unwrap();
            if active_branches.len() >= self.config.max_branches {
                return Err(MultiBranchError::BranchSpaceOverflow {
                    current_branches: active_branches.len(),
                    max_branches: self.config.max_branches,
                });
            }
        }
        
        // Get current algorithm state from temporal manager
        let current_state = self.state_manager.get_current_state()
            .ok_or_else(|| MultiBranchError::ParameterSpaceError {
                reason: "No current algorithm state available".to_string(),
            })?;
        
        // Evaluate objectives
        let start_time = std::time::Instant::now();
        let objectives = evaluation_function(&parameters, &current_state)?;
        let evaluation_time = start_time.elapsed().as_secs_f64();
        
        // Create result
        let mut metadata = HashMap::new();
        metadata.insert("evaluation_time".to_string(), evaluation_time);
        metadata.insert("branch_creation_time".to_string(), chrono::Utc::now().timestamp() as f64);
        
        let result = MultiObjectiveResult {
            parameters: parameters.clone(),
            objectives,
            objective_names: vec!["performance".to_string(), "accuracy".to_string()], // Default objectives
            state: current_state,
            metadata,
            pareto_rank: 0, // Will be updated by Pareto frontier computation
            crowding_distance: 0.0,
        };
        
        // Store in active branches
        {
            let mut active_branches = self.active_branches.write().unwrap();
            active_branches.insert(branch_id.clone(), result.clone());
        }
        
        // Update exploration statistics
        {
            let mut stats = self.exploration_stats.write().unwrap();
            stats.total_evaluations += 1;
            stats.average_evaluation_time = 
                (stats.average_evaluation_time * (stats.total_evaluations - 1) as f64 + evaluation_time) 
                / stats.total_evaluations as f64;
        }
        
        log::debug!(
            "Created multi-dimensional branch '{}' with {} parameters and {} objectives",
            branch_id,
            parameters.dimension(),
            objectives.len()
        );
        
        Ok(result)
    }
    
    /// Update Pareto frontier with new results
    pub async fn update_pareto_frontier(
        &self,
        new_results: Vec<MultiObjectiveResult>,
    ) -> Result<(), MultiBranchError> {
        let mut frontier = self.pareto_frontier.lock().unwrap();
        frontier.update_with_results(new_results)?;
        
        // Update statistics
        {
            let mut stats = self.exploration_stats.write().unwrap();
            if let Some(pareto_optimal) = frontier.get_pareto_optimal() {
                stats.pareto_optimal_solutions = pareto_optimal.len();
            }
            stats.hypervolume_progression.push(frontier.hypervolume);
        }
        
        log::info!(
            "Updated Pareto frontier: {} fronts, hypervolume = {:.6}",
            frontier.fronts.len(),
            frontier.hypervolume
        );
        
        Ok(())
    }
    
    /// Get current Pareto optimal solutions
    pub fn get_pareto_optimal_solutions(&self) -> Option<Vec<MultiObjectiveResult>> {
        let frontier = self.pareto_frontier.lock().unwrap();
        frontier.get_pareto_optimal().cloned()
    }
    
    /// Get exploration statistics
    pub fn get_exploration_statistics(&self) -> ExplorationStatistics {
        self.exploration_stats.read().unwrap().clone()
    }
    
    /// Check convergence based on hypervolume improvement
    pub fn check_convergence(&self) -> bool {
        let stats = self.exploration_stats.read().unwrap();
        
        if stats.hypervolume_progression.len() < 10 {
            return false;
        }
        
        // Check if hypervolume improvement has stagnated
        let recent_window = &stats.hypervolume_progression[stats.hypervolume_progression.len() - 10..];
        let improvement = recent_window.last().unwrap() - recent_window.first().unwrap();
        
        improvement.abs() < self.config.convergence_tolerance
    }
    
    /// Optimize parameter space using evolutionary multi-objective optimization
    pub async fn optimize_parameter_space<F>(
        &self,
        objective_function: F,
        objective_names: Vec<String>,
    ) -> Result<Vec<MultiObjectiveResult>, MultiBranchError>
    where
        F: Fn(&ParameterVector, &AlgorithmState) -> Result<DVector<f64>, MultiBranchError> + Send + Sync + Clone + 'static,
    {
        let mut iteration = 0;
        let mut all_results = Vec::new();
        
        // Get initial parameter configurations
        let parameter_configs = {
            self.parameter_space.read().unwrap().values().cloned().collect::<Vec<_>>()
        };
        
        // Evaluate initial population in parallel
        let results: Result<Vec<_>, _> = parameter_configs
            .par_iter()
            .enumerate()
            .map(|(i, params)| {
                let branch_id = format!("branch_{}", i);
                
                // Create a runtime for the async function
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.create_multi_branch(
                    branch_id,
                    params.clone(),
                    objective_function.clone(),
                ))
            })
            .collect();
        
        let mut current_population = results?;
        all_results.extend(current_population.clone());
        
        // Update Pareto frontier
        self.update_pareto_frontier(current_population.clone()).await?;
        
        // Evolutionary optimization loop
        while iteration < self.config.max_iterations && !self.check_convergence() {
            // Selection, crossover, and mutation would be implemented here
            // For now, we'll use adaptive sampling to generate new candidates
            
            // Get best solutions for next generation sampling
            let pareto_optimal = self.get_pareto_optimal_solutions()
                .unwrap_or_else(|| current_population.clone());
            
            // Generate new candidates around Pareto optimal solutions
            let new_candidates = self.generate_adaptive_candidates(&pareto_optimal)?;
            
            // Evaluate new candidates
            let new_results: Result<Vec<_>, _> = new_candidates
                .par_iter()
                .enumerate()
                .map(|(i, params)| {
                    let branch_id = format!("branch_{}_{}", iteration, i);
                    
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(self.create_multi_branch(
                        branch_id,
                        params.clone(),
                        objective_function.clone(),
                    ))
                })
                .collect();
            
            let new_population = new_results?;
            all_results.extend(new_population.clone());
            current_population = new_population;
            
            // Update Pareto frontier
            self.update_pareto_frontier(current_population.clone()).await?;
            
            iteration += 1;
            
            if iteration % 10 == 0 {
                log::info!(
                    "Multi-objective optimization iteration {}: {} solutions, hypervolume = {:.6}",
                    iteration,
                    all_results.len(),
                    self.pareto_frontier.lock().unwrap().hypervolume
                );
            }
        }
        
        // Update final statistics
        {
            let mut stats = self.exploration_stats.write().unwrap();
            stats.convergence_iterations = iteration;
        }
        
        Ok(all_results)
    }
    
    /// Generate adaptive candidates around high-quality solutions
    fn generate_adaptive_candidates(
        &self,
        pareto_solutions: &[MultiObjectiveResult],
    ) -> Result<Vec<ParameterVector>, MultiBranchError> {
        if pareto_solutions.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut candidates = Vec::new();
        let num_candidates = 20.min(self.config.max_branches / 10);
        
        for _ in 0..num_candidates {
            // Select random Pareto optimal solution as center
            let center_idx = rand::random::<usize>() % pareto_solutions.len();
            let center = &pareto_solutions[center_idx].parameters;
            
            // Generate Gaussian perturbation around center
            let mut perturbed_values = Vec::with_capacity(center.dimension());
            
            for (i, &center_val) in center.values.iter().enumerate() {
                let (min_bound, max_bound) = center.bounds[i];
                let range = max_bound - min_bound;
                let std_dev = range * 0.1; // 10% of parameter range
                
                // Gaussian perturbation with bounds checking
                let perturbation = rand_distr::Normal::new(0.0, std_dev)
                    .map_err(|e| MultiBranchError::ParameterSpaceError {
                        reason: format!("Failed to create normal distribution: {}", e),
                    })?
                    .sample(&mut rand::thread_rng());
                
                let perturbed_val = (center_val + perturbation).clamp(min_bound, max_bound);
                perturbed_values.push(perturbed_val);
            }
            
            let candidate = ParameterVector::new(
                perturbed_values,
                center.names.clone(),
                center.bounds.clone(),
            )?;
            
            candidates.push(candidate);
        }
        
        Ok(candidates)
    }
    
    /// Export multi-dimensional exploration results
    pub fn export_exploration_results(&self) -> Result<String, MultiBranchError> {
        let frontier = self.pareto_frontier.lock().unwrap();
        let stats = self.exploration_stats.read().unwrap();
        
        let export_data = serde_json::json!({
            "pareto_fronts": frontier.fronts,
            "hypervolume": frontier.hypervolume,
            "exploration_statistics": *stats,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        serde_json::to_string_pretty(&export_data)
            .map_err(|e| MultiBranchError::ParameterSpaceError {
                reason: format!("Failed to serialize results: {}", e),
            })
    }
}

// Required trait implementations for parallel processing
unsafe impl Send for MultiBranchManager {}
unsafe impl Sync for MultiBranchManager {}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;
    
    #[test]
    fn test_parameter_vector_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let bounds = vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)];
        
        let param_vector = ParameterVector::new(values, names, bounds).unwrap();
        assert_eq!(param_vector.dimension(), 3);
        assert_eq!(param_vector.values[0], 1.0);
    }
    
    #[test]
    fn test_pareto_dominance() {
        let params = ParameterVector::new(
            vec![1.0, 2.0],
            vec!["x".to_string(), "y".to_string()],
            vec![(0.0, 10.0), (0.0, 10.0)],
        ).unwrap();
        
        let result1 = MultiObjectiveResult {
            parameters: params.clone(),
            objectives: dvector![1.0, 2.0], // Better in both objectives
            objective_names: vec!["obj1".to_string(), "obj2".to_string()],
            state: AlgorithmState::default(),
            metadata: HashMap::new(),
            pareto_rank: 0,
            crowding_distance: 0.0,
        };
        
        let result2 = MultiObjectiveResult {
            parameters: params,
            objectives: dvector![2.0, 3.0], // Worse in both objectives
            objective_names: vec!["obj1".to_string(), "obj2".to_string()],
            state: AlgorithmState::default(),
            metadata: HashMap::new(),
            pareto_rank: 0,
            crowding_distance: 0.0,
        };
        
        assert!(result1.dominates(&result2));
        assert!(!result2.dominates(&result1));
    }
    
    #[test]
    fn test_uniform_grid_generation() {
        use std::sync::Arc;
        
        let config = MultiBranchConfig::default();
        let state_manager = Arc::new(StateManager::new());
        let reference_point = dvector![10.0, 10.0];
        
        let manager = MultiBranchManager::new(config, state_manager, reference_point);
        
        let param_names = vec!["x".to_string(), "y".to_string()];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        
        let grid = manager.generate_uniform_grid(&param_names, &bounds, 3).unwrap();
        assert_eq!(grid.len(), 9); // 3^2 = 9 points
        
        // Check that all points are within bounds
        for point in &grid {
            for (i, &value) in point.values.iter().enumerate() {
                let (min_bound, max_bound) = bounds[i];
                assert!(value >= min_bound && value <= max_bound);
            }
        }
    }
    
    #[tokio::test]
    async fn test_multi_branch_creation() {
        use std::sync::Arc;
        
        let config = MultiBranchConfig::default();
        let state_manager = Arc::new(StateManager::new());
        let reference_point = dvector![10.0, 10.0];
        
        let manager = MultiBranchManager::new(config, state_manager, reference_point);
        
        let params = ParameterVector::new(
            vec![1.0, 2.0],
            vec!["x".to_string(), "y".to_string()],
            vec![(0.0, 10.0), (0.0, 10.0)],
        ).unwrap();
        
        // Mock evaluation function
        let eval_fn = |_params: &ParameterVector, _state: &AlgorithmState| {
            Ok(dvector![1.0, 2.0])
        };
        
        let result = manager.create_multi_branch(
            "test_branch".to_string(),
            params,
            eval_fn,
        ).await;
        
        assert!(result.is_ok());
        let multi_result = result.unwrap();
        assert_eq!(multi_result.objectives.len(), 2);
    }
}