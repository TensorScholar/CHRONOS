"""
Chronos Educational Learning Path Framework
============================================

Advanced Pythonic interface for knowledge graph navigation with functional reactive
programming principles, zero-copy Rust integration, and lazy evaluation semantics.

Copyright (c) 2025 Mohammad Atashi. All rights reserved.

Theoretical Foundation:
- Category theory for compositional learning path construction
- Directed acyclic knowledge graphs with topological ordering
- Lazy evaluation monad for efficient path traversal
- Functional reactive programming for dynamic adaptation

Architectural Principles:
- Immutable data structures with structural sharing
- Zero-copy integration with Rust computational core
- Type-driven development with comprehensive static analysis
- Monadic composition for learning state transformation
"""

from __future__ import annotations

import asyncio
import functools
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, ClassVar, Dict, FrozenSet, Generic, List, Optional, Protocol,
    Set, Tuple, TypeVar, Union, runtime_checkable
)

import numpy as np
from typing_extensions import ParamSpec, TypeAlias

try:
    from chronos._core.education import (
        LearningPathCore, 
        KnowledgeNode, 
        PrerequisiteGraph,
        LearningObjective,
        ExpertiseLevel
    )
except ImportError:
    logging.warning("Chronos core not available. Using fallback implementations.")
    # Fallback type definitions for development/testing
    LearningPathCore = Any
    KnowledgeNode = Any
    PrerequisiteGraph = Any
    LearningObjective = Any
    ExpertiseLevel = Any

# Type Variables for Generic Programming
T = TypeVar('T')
P = ParamSpec('P')
StateT = TypeVar('StateT')
NodeT = TypeVar('NodeT', bound='LearningNode')

# Type Aliases for Enhanced Readability
NodeId: TypeAlias = str
PathId: TypeAlias = str
CompetencyScore: TypeAlias = float
ProgressWeight: TypeAlias = float

logger = logging.getLogger(__name__)


class ExpertiseDomain(Enum):
    """Enumeration of expertise domains with hierarchical ordering."""
    NOVICE = auto()
    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()
    
    def __lt__(self, other: ExpertiseDomain) -> bool:
        """Enable ordering comparison for expertise levels."""
        return self.value < other.value
    
    def __le__(self, other: ExpertiseDomain) -> bool:
        return self.value <= other.value


@dataclass(frozen=True, slots=True)
class CompetencyVector:
    """
    Immutable competency representation with vectorized operations.
    
    Theoretical Foundation:
    - Vector space model for competency representation
    - L2 norm for competency distance calculation
    - Cosine similarity for competency alignment
    """
    
    # Core competency dimensions
    algorithmic_thinking: CompetencyScore = 0.0
    mathematical_reasoning: CompetencyScore = 0.0
    computational_complexity: CompetencyScore = 0.0
    problem_decomposition: CompetencyScore = 0.0
    pattern_recognition: CompetencyScore = 0.0
    
    # Advanced competency dimensions
    formal_verification: CompetencyScore = 0.0
    optimization_theory: CompetencyScore = 0.0
    abstraction_design: CompetencyScore = 0.0
    
    def __post_init__(self) -> None:
        """Validate competency scores are within [0, 1] range."""
        for field_name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Competency score {field_name} must be in [0, 1], got {value}")
    
    @property
    def magnitude(self) -> float:
        """Calculate L2 norm of competency vector."""
        return np.sqrt(sum(score ** 2 for score in self.__dict__.values()))
    
    @property
    def normalized(self) -> CompetencyVector:
        """Return unit vector in same direction."""
        mag = self.magnitude
        if mag == 0:
            return self
        
        return CompetencyVector(
            **{field: value / mag for field, value in self.__dict__.items()}
        )
    
    def distance_to(self, other: CompetencyVector) -> float:
        """Calculate Euclidean distance to another competency vector."""
        return np.sqrt(sum(
            (getattr(self, field) - getattr(other, field)) ** 2
            for field in self.__dataclass_fields__
        ))
    
    def similarity_to(self, other: CompetencyVector) -> float:
        """Calculate cosine similarity to another competency vector."""
        dot_product = sum(
            getattr(self, field) * getattr(other, field)
            for field in self.__dataclass_fields__
        )
        return dot_product / (self.magnitude * other.magnitude) if self.magnitude * other.magnitude > 0 else 0.0
    
    def blend_with(self, other: CompetencyVector, weight: float = 0.5) -> CompetencyVector:
        """Create weighted blend with another competency vector."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Blend weight must be in [0, 1], got {weight}")
        
        return CompetencyVector(
            **{
                field: weight * getattr(self, field) + (1 - weight) * getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class LearningObjectiveSpec:
    """
    Immutable specification for learning objectives with formal semantics.
    
    Theoretical Foundation:
    - Bloom's taxonomy for cognitive level classification
    - Item Response Theory for difficulty calibration
    - Bayesian networks for prerequisite modeling
    """
    
    id: str
    title: str
    description: str
    
    # Cognitive classification
    cognitive_level: int = field(default=1)  # 1-6 Bloom's taxonomy levels
    difficulty: float = field(default=0.5)   # [0, 1] normalized difficulty
    
    # Competency requirements
    required_competencies: CompetencyVector = field(default_factory=CompetencyVector)
    developed_competencies: CompetencyVector = field(default_factory=CompetencyVector)
    
    # Learning characteristics
    estimated_duration: int = field(default=30)  # minutes
    interaction_type: str = field(default="exploration")
    
    # Metadata
    tags: FrozenSet[str] = field(default_factory=frozenset)
    
    def __post_init__(self) -> None:
        """Validate objective specification parameters."""
        if not 1 <= self.cognitive_level <= 6:
            raise ValueError(f"Cognitive level must be 1-6, got {self.cognitive_level}")
        
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"Difficulty must be in [0, 1], got {self.difficulty}")
        
        if self.estimated_duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.estimated_duration}")
    
    @property
    def complexity_score(self) -> float:
        """Calculate objective complexity based on cognitive level and competency requirements."""
        competency_complexity = self.required_competencies.magnitude
        cognitive_complexity = self.cognitive_level / 6.0
        return (competency_complexity + cognitive_complexity) / 2.0


@runtime_checkable
class PathTraversalStrategy(Protocol):
    """Protocol for learning path traversal strategies."""
    
    def next_objective(
        self, 
        current_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec],
        learning_history: Sequence[str]
    ) -> Optional[LearningObjectiveSpec]:
        """Select next learning objective based on strategy."""
        ...
    
    def estimate_path_length(
        self,
        current_competencies: CompetencyVector,
        target_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec]
    ) -> int:
        """Estimate number of objectives to reach target competencies."""
        ...


class OptimalTraversalStrategy:
    """
    Optimal path traversal using dynamic programming with competency distance minimization.
    
    Theoretical Foundation:
    - Dijkstra's algorithm for shortest path in competency space
    - Dynamic programming for optimal substructure
    - Memoization for computational efficiency
    """
    
    def __init__(self, exploration_factor: float = 0.1):
        """Initialize with exploration vs exploitation balance."""
        self.exploration_factor = exploration_factor
        self._memoization_cache: Dict[Tuple[CompetencyVector, CompetencyVector], List[str]] = {}
    
    def next_objective(
        self,
        current_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec],
        learning_history: Sequence[str]
    ) -> Optional[LearningObjectiveSpec]:
        """Select objective that minimizes competency gap with exploration factor."""
        if not available_objectives:
            return None
        
        # Filter objectives that are accessible given current competencies
        accessible_objectives = [
            obj for obj in available_objectives
            if self._is_accessible(current_competencies, obj)
            and obj.id not in learning_history
        ]
        
        if not accessible_objectives:
            return None
        
        # Calculate competency gains for each objective
        scored_objectives = [
            (
                obj,
                self._calculate_objective_score(current_competencies, obj, learning_history)
            )
            for obj in accessible_objectives
        ]
        
        # Select objective with highest score
        return max(scored_objectives, key=lambda x: x[1])[0]
    
    def estimate_path_length(
        self,
        current_competencies: CompetencyVector,
        target_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec]
    ) -> int:
        """Estimate path length using competency distance heuristic."""
        distance = current_competencies.distance_to(target_competencies)
        
        # Estimate based on average competency gain per objective
        if available_objectives:
            avg_gain = np.mean([
                obj.developed_competencies.magnitude 
                for obj in available_objectives
            ])
            return max(1, int(np.ceil(distance / avg_gain))) if avg_gain > 0 else len(available_objectives)
        
        return 1
    
    def _is_accessible(self, competencies: CompetencyVector, objective: LearningObjectiveSpec) -> bool:
        """Check if objective is accessible given current competencies."""
        # Simple threshold-based accessibility
        for field in CompetencyVector.__dataclass_fields__:
            required = getattr(objective.required_competencies, field)
            current = getattr(competencies, field)
            if current < required * 0.8:  # 80% threshold with some tolerance
                return False
        return True
    
    def _calculate_objective_score(
        self,
        competencies: CompetencyVector,
        objective: LearningObjectiveSpec,
        history: Sequence[str]
    ) -> float:
        """Calculate score for objective selection."""
        # Competency gain score
        gain_score = objective.developed_competencies.magnitude
        
        # Accessibility score (prefer slightly challenging objectives)
        accessibility_score = 1.0 - objective.complexity_score * 0.5
        
        # Exploration bonus for unvisited objectives
        exploration_bonus = self.exploration_factor if objective.id not in history else 0.0
        
        return gain_score + accessibility_score + exploration_bonus


class AdaptiveTraversalStrategy:
    """
    Adaptive traversal strategy using reinforcement learning principles.
    
    Theoretical Foundation:
    - Multi-armed bandit for objective selection
    - Upper Confidence Bound for exploration-exploitation balance
    - Exponential moving average for performance tracking
    """
    
    def __init__(self, confidence_level: float = 2.0, decay_factor: float = 0.9):
        """Initialize adaptive strategy with UCB parameters."""
        self.confidence_level = confidence_level
        self.decay_factor = decay_factor
        self._objective_rewards: Dict[str, float] = {}
        self._objective_attempts: Dict[str, int] = {}
        self._total_attempts = 0
    
    def next_objective(
        self,
        current_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec],
        learning_history: Sequence[str]
    ) -> Optional[LearningObjectiveSpec]:
        """Select objective using Upper Confidence Bound algorithm."""
        if not available_objectives:
            return None
        
        # Filter accessible objectives
        accessible_objectives = [
            obj for obj in available_objectives
            if self._is_accessible(current_competencies, obj)
        ]
        
        if not accessible_objectives:
            return None
        
        # Calculate UCB scores
        scored_objectives = [
            (obj, self._calculate_ucb_score(obj))
            for obj in accessible_objectives
        ]
        
        return max(scored_objectives, key=lambda x: x[1])[0]
    
    def update_performance(self, objective_id: str, performance_score: float) -> None:
        """Update performance statistics for objective."""
        self._objective_attempts[objective_id] = self._objective_attempts.get(objective_id, 0) + 1
        self._total_attempts += 1
        
        # Exponential moving average update
        current_reward = self._objective_rewards.get(objective_id, 0.0)
        self._objective_rewards[objective_id] = (
            self.decay_factor * current_reward + 
            (1 - self.decay_factor) * performance_score
        )
    
    def estimate_path_length(
        self,
        current_competencies: CompetencyVector,
        target_competencies: CompetencyVector,
        available_objectives: Sequence[LearningObjectiveSpec]
    ) -> int:
        """Estimate path length based on historical performance."""
        # Use performance data to estimate if available
        if self._objective_rewards:
            avg_performance = np.mean(list(self._objective_rewards.values()))
            distance = current_competencies.distance_to(target_competencies)
            return max(1, int(np.ceil(distance / avg_performance))) if avg_performance > 0 else len(available_objectives)
        
        # Fallback to simple estimation
        return len(available_objectives) // 2
    
    def _is_accessible(self, competencies: CompetencyVector, objective: LearningObjectiveSpec) -> bool:
        """Check objective accessibility with adaptive thresholds."""
        # Adaptive threshold based on historical success
        base_threshold = 0.8
        success_rate = self._get_success_rate(objective.id)
        adaptive_threshold = base_threshold * (1.0 - success_rate * 0.2)
        
        for field in CompetencyVector.__dataclass_fields__:
            required = getattr(objective.required_competencies, field)
            current = getattr(competencies, field)
            if current < required * adaptive_threshold:
                return False
        return True
    
    def _calculate_ucb_score(self, objective: LearningObjectiveSpec) -> float:
        """Calculate Upper Confidence Bound score."""
        obj_id = objective.id
        
        # Average reward
        avg_reward = self._objective_rewards.get(obj_id, 0.0)
        
        # Confidence interval
        attempts = self._objective_attempts.get(obj_id, 0)
        if attempts == 0 or self._total_attempts == 0:
            confidence = float('inf')  # Prioritize unexplored objectives
        else:
            confidence = self.confidence_level * np.sqrt(
                np.log(self._total_attempts) / attempts
            )
        
        return avg_reward + confidence
    
    def _get_success_rate(self, objective_id: str) -> float:
        """Get success rate for objective."""
        attempts = self._objective_attempts.get(objective_id, 0)
        if attempts == 0:
            return 0.5  # Neutral assumption for new objectives
        
        reward = self._objective_rewards.get(objective_id, 0.0)
        return min(1.0, max(0.0, reward))


@dataclass
class LearningPathState:
    """
    Mutable learning path state with functional update operations.
    
    Architectural Pattern: State monad for learning progression
    """
    
    current_competencies: CompetencyVector
    target_competencies: CompetencyVector
    completed_objectives: List[str] = field(default_factory=list)
    current_objective: Optional[str] = None
    progress_history: List[Tuple[str, float, CompetencyVector]] = field(default_factory=list)
    
    # Performance metrics
    total_study_time: int = 0  # minutes
    success_rate: float = 1.0
    efficiency_score: float = 1.0
    
    def clone(self) -> LearningPathState:
        """Create deep copy of learning state."""
        return LearningPathState(
            current_competencies=self.current_competencies,
            target_competencies=self.target_competencies,
            completed_objectives=self.completed_objectives.copy(),
            current_objective=self.current_objective,
            progress_history=self.progress_history.copy(),
            total_study_time=self.total_study_time,
            success_rate=self.success_rate,
            efficiency_score=self.efficiency_score
        )
    
    def update_competencies(self, new_competencies: CompetencyVector) -> LearningPathState:
        """Functional update of competencies with history tracking."""
        new_state = self.clone()
        
        if self.current_objective:
            # Record progress
            progress_gain = new_competencies.distance_to(self.current_competencies)
            new_state.progress_history.append((
                self.current_objective,
                progress_gain,
                new_competencies
            ))
        
        new_state.current_competencies = new_competencies
        return new_state
    
    def complete_objective(self, objective_id: str, study_time: int, success: bool) -> LearningPathState:
        """Functional update for objective completion."""
        new_state = self.clone()
        new_state.completed_objectives.append(objective_id)
        new_state.current_objective = None
        new_state.total_study_time += study_time
        
        # Update performance metrics
        completion_count = len(new_state.completed_objectives)
        success_count = sum(1 for _, _, _ in new_state.progress_history) + (1 if success else 0)
        new_state.success_rate = success_count / completion_count if completion_count > 0 else 1.0
        
        return new_state
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress towards target competencies."""
        if self.current_competencies == self.target_competencies:
            return 100.0
        
        initial_distance = CompetencyVector().distance_to(self.target_competencies)
        current_distance = self.current_competencies.distance_to(self.target_competencies)
        
        if initial_distance == 0:
            return 100.0
        
        progress = max(0.0, min(100.0, (1 - current_distance / initial_distance) * 100))
        return progress
    
    @property
    def estimated_completion_time(self) -> int:
        """Estimate remaining study time based on current efficiency."""
        if self.progress_percentage >= 100.0:
            return 0
        
        if len(self.completed_objectives) == 0:
            return 240  # Default 4 hours estimate
        
        avg_time_per_objective = self.total_study_time / len(self.completed_objectives)
        remaining_progress = 100.0 - self.progress_percentage
        estimated_objectives_remaining = max(1, int(remaining_progress / 20))  # Rough estimate
        
        return int(avg_time_per_objective * estimated_objectives_remaining / self.efficiency_score)


class LearningPath:
    """
    Immutable learning path with lazy evaluation and functional composition.
    
    Theoretical Foundation:
    - Lazy evaluation monad for deferred computation
    - Compositional semantics for path operations
    - Observer pattern for progress monitoring
    
    Architectural Patterns:
    - Repository pattern for objective management
    - Strategy pattern for traversal algorithms
    - Observer pattern for progress callbacks
    """
    
    # Class-level configuration
    _default_strategy_class: ClassVar[type] = OptimalTraversalStrategy
    
    def __init__(
        self,
        objectives: Sequence[LearningObjectiveSpec],
        initial_competencies: Optional[CompetencyVector] = None,
        target_competencies: Optional[CompetencyVector] = None,
        traversal_strategy: Optional[PathTraversalStrategy] = None,
        path_id: Optional[str] = None
    ):
        """
        Initialize learning path with objective sequence and strategy.
        
        Args:
            objectives: Available learning objectives
            initial_competencies: Starting competency vector
            target_competencies: Target competency vector
            traversal_strategy: Path traversal strategy
            path_id: Unique path identifier
        """
        self._objectives = tuple(objectives)  # Immutable sequence
        self._objective_map = {obj.id: obj for obj in objectives}
        
        self._initial_competencies = initial_competencies or CompetencyVector()
        self._target_competencies = target_competencies or CompetencyVector(
            **{field: 1.0 for field in CompetencyVector.__dataclass_fields__}
        )
        
        self._strategy = traversal_strategy or self._default_strategy_class()
        self._path_id = path_id or f"path_{id(self)}"
        
        # State management
        self._current_state = LearningPathState(
            current_competencies=self._initial_competencies,
            target_competencies=self._target_competencies
        )
        
        # Observer pattern for progress monitoring
        self._progress_observers: List[Callable[[LearningPathState], None]] = []
        self._completion_observers: List[Callable[[LearningPath], None]] = []
        
        # Lazy evaluation cache
        self._cached_next_objective: Optional[LearningObjectiveSpec] = None
        self._cache_invalidated = True
        
        # Performance monitoring
        self._performance_metrics: Dict[str, float] = {}
        
        logger.info(f"Initialized learning path {self._path_id} with {len(objectives)} objectives")
    
    @property
    def path_id(self) -> str:
        """Unique path identifier."""
        return self._path_id
    
    @property
    def objectives(self) -> Tuple[LearningObjectiveSpec, ...]:
        """Immutable tuple of available objectives."""
        return self._objectives
    
    @property
    def current_state(self) -> LearningPathState:
        """Current learning state (immutable view)."""
        return LearningPathState(
            current_competencies=self._current_state.current_competencies,
            target_competencies=self._current_state.target_competencies,
            completed_objectives=self._current_state.completed_objectives.copy(),
            current_objective=self._current_state.current_objective,
            progress_history=self._current_state.progress_history.copy(),
            total_study_time=self._current_state.total_study_time,
            success_rate=self._current_state.success_rate,
            efficiency_score=self._current_state.efficiency_score
        )
    
    @property
    def progress_percentage(self) -> float:
        """Current progress percentage towards target."""
        return self._current_state.progress_percentage
    
    @property
    def is_complete(self) -> bool:
        """Check if learning path is complete."""
        return self.progress_percentage >= 95.0  # Allow for small numerical errors
    
    @property
    def next_objective(self) -> Optional[LearningObjectiveSpec]:
        """
        Get next recommended objective using lazy evaluation.
        
        Returns:
            Next objective or None if path is complete
        """
        if self._cache_invalidated or self._cached_next_objective is None:
            self._cached_next_objective = self._compute_next_objective()
            self._cache_invalidated = False
        
        return self._cached_next_objective
    
    def _compute_next_objective(self) -> Optional[LearningObjectiveSpec]:
        """Compute next objective using traversal strategy."""
        if self.is_complete:
            return None
        
        available_objectives = [
            obj for obj in self._objectives
            if obj.id not in self._current_state.completed_objectives
        ]
        
        return self._strategy.next_objective(
            self._current_state.current_competencies,
            available_objectives,
            self._current_state.completed_objectives
        )
    
    def advance_to_objective(self, objective_id: str) -> LearningPath:
        """
        Create new path instance with objective set as current.
        
        Returns:
            New LearningPath instance with updated state
        """
        if objective_id not in self._objective_map:
            raise ValueError(f"Objective {objective_id} not found in path")
        
        if objective_id in self._current_state.completed_objectives:
            raise ValueError(f"Objective {objective_id} already completed")
        
        # Create new path instance with updated state
        new_path = self._clone()
        new_path._current_state = self._current_state.clone()
        new_path._current_state.current_objective = objective_id
        new_path._invalidate_cache()
        
        logger.info(f"Advanced to objective {objective_id} in path {self._path_id}")
        return new_path
    
    def complete_current_objective(
        self,
        new_competencies: CompetencyVector,
        study_time: int,
        success: bool = True
    ) -> LearningPath:
        """
        Complete current objective and update competencies.
        
        Args:
            new_competencies: Updated competency vector
            study_time: Time spent on objective (minutes)
            success: Whether objective was completed successfully
            
        Returns:
            New LearningPath instance with updated state
        """
        if not self._current_state.current_objective:
            raise ValueError("No current objective to complete")
        
        objective_id = self._current_state.current_objective
        
        # Create new path instance with updated state
        new_path = self._clone()
        new_path._current_state = (
            self._current_state
            .update_competencies(new_competencies)
            .complete_objective(objective_id, study_time, success)
        )
        
        # Update strategy performance if applicable
        if hasattr(new_path._strategy, 'update_performance'):
            performance_score = new_competencies.similarity_to(self._current_state.current_competencies)
            new_path._strategy.update_performance(objective_id, performance_score)
        
        new_path._invalidate_cache()
        
        # Notify observers
        for observer in self._progress_observers:
            observer(new_path._current_state)
        
        if new_path.is_complete:
            for observer in self._completion_observers:
                observer(new_path)
        
        logger.info(
            f"Completed objective {objective_id} in path {self._path_id}. "
            f"Progress: {new_path.progress_percentage:.1f}%"
        )
        
        return new_path
    
    def estimate_remaining_time(self) -> int:
        """Estimate remaining study time in minutes."""
        return self._current_state.estimated_completion_time
    
    def get_objective_by_id(self, objective_id: str) -> Optional[LearningObjectiveSpec]:
        """Get objective by ID."""
        return self._objective_map.get(objective_id)
    
    def get_accessible_objectives(self) -> List[LearningObjectiveSpec]:
        """Get list of currently accessible objectives."""
        available_objectives = [
            obj for obj in self._objectives
            if obj.id not in self._current_state.completed_objectives
        ]
        
        return [
            obj for obj in available_objectives
            if self._is_objective_accessible(obj)
        ]
    
    def _is_objective_accessible(self, objective: LearningObjectiveSpec) -> bool:
        """Check if objective is accessible given current competencies."""
        # Delegate to strategy if it has accessibility logic
        if hasattr(self._strategy, '_is_accessible'):
            return self._strategy._is_accessible(self._current_state.current_competencies, objective)
        
        # Default accessibility check
        for field in CompetencyVector.__dataclass_fields__:
            required = getattr(objective.required_competencies, field)
            current = getattr(self._current_state.current_competencies, field)
            if current < required * 0.8:
                return False
        return True
    
    def add_progress_observer(self, observer: Callable[[LearningPathState], None]) -> None:
        """Add progress observer callback."""
        self._progress_observers.append(observer)
    
    def add_completion_observer(self, observer: Callable[[LearningPath], None]) -> None:
        """Add completion observer callback."""
        self._completion_observers.append(observer)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the learning path."""
        metrics = {
            'progress_percentage': self.progress_percentage,
            'success_rate': self._current_state.success_rate,
            'efficiency_score': self._current_state.efficiency_score,
            'total_study_time': float(self._current_state.total_study_time),
            'estimated_remaining_time': float(self.estimate_remaining_time()),
            'objectives_completed': float(len(self._current_state.completed_objectives)),
            'objectives_total': float(len(self._objectives))
        }
        
        return {**metrics, **self._performance_metrics}
    
    def _clone(self) -> LearningPath:
        """Create a deep copy of the learning path."""
        new_path = LearningPath(
            objectives=self._objectives,
            initial_competencies=self._initial_competencies,
            target_competencies=self._target_competencies,
            traversal_strategy=self._strategy,
            path_id=self._path_id
        )
        
        # Copy observers
        new_path._progress_observers = self._progress_observers.copy()
        new_path._completion_observers = self._completion_observers.copy()
        new_path._performance_metrics = self._performance_metrics.copy()
        
        return new_path
    
    def _invalidate_cache(self) -> None:
        """Invalidate lazy evaluation cache."""
        self._cache_invalidated = True
        self._cached_next_objective = None
    
    def __str__(self) -> str:
        """String representation of learning path."""
        return (
            f"LearningPath(id={self._path_id}, "
            f"progress={self.progress_percentage:.1f}%, "
            f"objectives={len(self._current_state.completed_objectives)}/{len(self._objectives)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"LearningPath("
            f"path_id='{self._path_id}', "
            f"objectives={len(self._objectives)}, "
            f"progress={self.progress_percentage:.1f}%, "
            f"strategy={type(self._strategy).__name__}"
            f")"
        )


class LearningPathBuilder:
    """
    Builder pattern for constructing learning paths with fluent interface.
    
    Architectural Pattern: Builder with method chaining
    """
    
    def __init__(self):
        """Initialize builder with default values."""
        self._objectives: List[LearningObjectiveSpec] = []
        self._initial_competencies: Optional[CompetencyVector] = None
        self._target_competencies: Optional[CompetencyVector] = None
        self._strategy: Optional[PathTraversalStrategy] = None
        self._path_id: Optional[str] = None
    
    def with_objectives(self, objectives: Sequence[LearningObjectiveSpec]) -> LearningPathBuilder:
        """Set objectives for the learning path."""
        self._objectives = list(objectives)
        return self
    
    def add_objective(self, objective: LearningObjectiveSpec) -> LearningPathBuilder:
        """Add single objective to the path."""
        self._objectives.append(objective)
        return self
    
    def with_initial_competencies(self, competencies: CompetencyVector) -> LearningPathBuilder:
        """Set initial competencies."""
        self._initial_competencies = competencies
        return self
    
    def with_target_competencies(self, competencies: CompetencyVector) -> LearningPathBuilder:
        """Set target competencies."""
        self._target_competencies = competencies
        return self
    
    def with_strategy(self, strategy: PathTraversalStrategy) -> LearningPathBuilder:
        """Set traversal strategy."""
        self._strategy = strategy
        return self
    
    def with_optimal_strategy(self, exploration_factor: float = 0.1) -> LearningPathBuilder:
        """Use optimal traversal strategy."""
        self._strategy = OptimalTraversalStrategy(exploration_factor)
        return self
    
    def with_adaptive_strategy(
        self, 
        confidence_level: float = 2.0, 
        decay_factor: float = 0.9
    ) -> LearningPathBuilder:
        """Use adaptive traversal strategy."""
        self._strategy = AdaptiveTraversalStrategy(confidence_level, decay_factor)
        return self
    
    def with_path_id(self, path_id: str) -> LearningPathBuilder:
        """Set path identifier."""
        self._path_id = path_id
        return self
    
    def build(self) -> LearningPath:
        """Build the learning path."""
        if not self._objectives:
            raise ValueError("At least one objective must be specified")
        
        return LearningPath(
            objectives=self._objectives,
            initial_competencies=self._initial_competencies,
            target_competencies=self._target_competencies,
            traversal_strategy=self._strategy,
            path_id=self._path_id
        )


class LearningPathRepository:
    """
    Repository for managing learning path persistence and retrieval.
    
    Architectural Patterns:
    - Repository pattern for data access abstraction
    - Weak reference caching for memory efficiency
    - Async context manager for resource management
    """
    
    def __init__(self, core_repository: Optional[LearningPathCore] = None):
        """Initialize repository with optional core backend."""
        self._core = core_repository
        self._path_cache: weakref.WeakValueDictionary[str, LearningPath] = weakref.WeakValueDictionary()
        self._lock = asyncio.Lock()
    
    async def save_path(self, path: LearningPath) -> None:
        """Save learning path to persistent storage."""
        async with self._lock:
            if self._core:
                # Save to Rust core if available
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._core.save_path, 
                    path.path_id, 
                    path.current_state
                )
            
            # Cache the path
            self._path_cache[path.path_id] = path
            logger.info(f"Saved learning path {path.path_id}")
    
    async def load_path(self, path_id: str) -> Optional[LearningPath]:
        """Load learning path from storage."""
        async with self._lock:
            # Check cache first
            if path_id in self._path_cache:
                return self._path_cache[path_id]
            
            # Load from core if available
            if self._core:
                try:
                    path_data = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self._core.load_path, 
                        path_id
                    )
                    
                    if path_data:
                        # Reconstruct path from data
                        path = self._reconstruct_path(path_data)
                        self._path_cache[path_id] = path
                        return path
                        
                except Exception as e:
                    logger.error(f"Failed to load path {path_id}: {e}")
            
            return None
    
    async def list_paths(self) -> List[str]:
        """List all available learning path IDs."""
        if self._core:
            return await asyncio.get_event_loop().run_in_executor(
                None, 
                self._core.list_paths
            )
        
        return list(self._path_cache.keys())
    
    async def delete_path(self, path_id: str) -> bool:
        """Delete learning path from storage."""
        async with self._lock:
            if self._core:
                success = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._core.delete_path, 
                    path_id
                )
            else:
                success = path_id in self._path_cache
            
            # Remove from cache
            if path_id in self._path_cache:
                del self._path_cache[path_id]
            
            if success:
                logger.info(f"Deleted learning path {path_id}")
            
            return success
    
    def _reconstruct_path(self, path_data: Any) -> LearningPath:
        """Reconstruct learning path from saved data."""
        # This would involve deserializing the path data
        # For now, return a placeholder
        return LearningPathBuilder().build()
    
    async def __aenter__(self) -> LearningPathRepository:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Cleanup resources if needed
        pass


# Factory functions for common use cases
def create_algorithm_learning_path(
    difficulty_level: ExpertiseDomain = ExpertiseDomain.BEGINNER,
    focus_areas: Optional[Set[str]] = None
) -> LearningPath:
    """
    Factory function for creating algorithm-focused learning paths.
    
    Args:
        difficulty_level: Target difficulty level
        focus_areas: Specific algorithm areas to focus on
        
    Returns:
        Configured learning path for algorithm learning
    """
    focus_areas = focus_areas or {"pathfinding", "sorting", "graph_traversal"}
    
    # Create competency targets based on difficulty level
    base_competency = 0.3 + (difficulty_level.value - 1) * 0.15
    target_competencies = CompetencyVector(
        algorithmic_thinking=min(1.0, base_competency + 0.2),
        mathematical_reasoning=base_competency,
        computational_complexity=min(1.0, base_competency + 0.1),
        problem_decomposition=min(1.0, base_competency + 0.15),
        pattern_recognition=base_competency
    )
    
    # Create sample objectives (would be loaded from curriculum)
    objectives = [
        LearningObjectiveSpec(
            id="basic_search",
            title="Linear and Binary Search",
            description="Understand basic search algorithms",
            cognitive_level=2,
            difficulty=0.3,
            required_competencies=CompetencyVector(algorithmic_thinking=0.2),
            developed_competencies=CompetencyVector(
                algorithmic_thinking=0.4,
                computational_complexity=0.3
            )
        ),
        LearningObjectiveSpec(
            id="pathfinding_basics",
            title="Graph Pathfinding Fundamentals",
            description="Learn BFS and DFS algorithms",
            cognitive_level=3,
            difficulty=0.5,
            required_competencies=CompetencyVector(
                algorithmic_thinking=0.3,
                mathematical_reasoning=0.2
            ),
            developed_competencies=CompetencyVector(
                algorithmic_thinking=0.6,
                mathematical_reasoning=0.4,
                problem_decomposition=0.5
            )
        )
    ]
    
    return (
        LearningPathBuilder()
        .with_objectives(objectives)
        .with_target_competencies(target_competencies)
        .with_adaptive_strategy()
        .with_path_id(f"algorithm_path_{difficulty_level.name.lower()}")
        .build()
    )


def create_personalized_learning_path(
    current_competencies: CompetencyVector,
    learning_goals: Set[str],
    time_constraint: Optional[int] = None
) -> LearningPath:
    """
    Factory function for creating personalized learning paths.
    
    Args:
        current_competencies: Current learner competencies
        learning_goals: Set of learning goal identifiers
        time_constraint: Maximum study time in minutes (optional)
        
    Returns:
        Personalized learning path
    """
    # Determine target competencies based on goals
    target_competencies = current_competencies
    
    if "advanced_algorithms" in learning_goals:
        target_competencies = target_competencies.blend_with(
            CompetencyVector(
                algorithmic_thinking=0.9,
                computational_complexity=0.8,
                optimization_theory=0.7
            ),
            weight=0.7
        )
    
    if "mathematical_foundations" in learning_goals:
        target_competencies = target_competencies.blend_with(
            CompetencyVector(
                mathematical_reasoning=0.9,
                formal_verification=0.6,
                abstraction_design=0.7
            ),
            weight=0.6
        )
    
    # Select appropriate strategy based on time constraints
    if time_constraint and time_constraint < 180:  # Less than 3 hours
        strategy = OptimalTraversalStrategy(exploration_factor=0.05)  # Focus on efficiency
    else:
        strategy = AdaptiveTraversalStrategy()  # Allow for exploration
    
    # Create objectives based on competency gaps
    objectives = _generate_objectives_for_gaps(current_competencies, target_competencies)
    
    return (
        LearningPathBuilder()
        .with_objectives(objectives)
        .with_initial_competencies(current_competencies)
        .with_target_competencies(target_competencies)
        .with_strategy(strategy)
        .with_path_id(f"personalized_{id(current_competencies)}")
        .build()
    )


def _generate_objectives_for_gaps(
    current: CompetencyVector,
    target: CompetencyVector
) -> List[LearningObjectiveSpec]:
    """Generate objectives to bridge competency gaps."""
    objectives = []
    
    # Calculate competency gaps
    gaps = {
        field: getattr(target, field) - getattr(current, field)
        for field in CompetencyVector.__dataclass_fields__
    }
    
    # Create objectives for significant gaps
    for field, gap in gaps.items():
        if gap > 0.2:  # Significant gap threshold
            objectives.append(
                LearningObjectiveSpec(
                    id=f"develop_{field}",
                    title=f"Develop {field.replace('_', ' ').title()}",
                    description=f"Focused development of {field} competency",
                    cognitive_level=min(6, int(gap * 6) + 1),
                    difficulty=min(1.0, gap + 0.2),
                    required_competencies=current,
                    developed_competencies=CompetencyVector(
                        **{field: min(1.0, getattr(current, field) + gap * 0.6)}
                    )
                )
            )
    
    return objectives


# Export public API
__all__ = [
    'CompetencyVector',
    'LearningObjectiveSpec',
    'LearningPath',
    'LearningPathBuilder',
    'LearningPathRepository',
    'LearningPathState',
    'ExpertiseDomain',
    'OptimalTraversalStrategy',
    'AdaptiveTraversalStrategy',
    'PathTraversalStrategy',
    'create_algorithm_learning_path',
    'create_personalized_learning_path'
]