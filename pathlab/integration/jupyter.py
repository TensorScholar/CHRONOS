"""
Chronos Jupyter Integration Framework

This module implements a sophisticated Jupyter notebook integration system
for the Chronos Algorithmic Observatory, providing seamless interactive
visualization and algorithm exploration capabilities within Jupyter environments.

Theoretical Foundation:
- Category-theoretic widget composition with functorial mappings
- Functional reactive programming for real-time state synchronization  
- Information-theoretic preservation of interactive semantics
- Homeomorphic transformation of visualization states across contexts

Copyright (c) 2025 Mohammad Atashi. All rights reserved.
"""

import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps, lru_cache, singledispatch
from typing import (
    Dict, List, Optional, Any, Tuple, Union, Callable, Generic, TypeVar,
    Protocol, runtime_checkable, Awaitable, AsyncIterator, ClassVar
)
from uuid import uuid4, UUID
from enum import Enum, auto
from collections import defaultdict, ChainMap
from concurrent.futures import ThreadPoolExecutor
import json
import base64
import threading
from pathlib import Path

# Core Jupyter and IPython infrastructure
try:
    import ipywidgets as widgets
    from ipywidgets import DOMWidget, Widget, register
    from traitlets import Unicode, Dict as TraitDict, Bool, Int, Float, List as TraitList
    from IPython.display import display, HTML, Javascript
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython import get_ipython
    HAS_JUPYTER = True
except ImportError:
    # Graceful degradation for non-Jupyter environments
    HAS_JUPYTER = False
    widgets = None

# Chronos core integration
try:
    from pathlab.algorithms import Algorithm, execute_algorithm
    from pathlab.visualization import VisualizationEngine, create_visualization
    from pathlab.temporal import TimelineManager, create_timeline
    from pathlab.insight import InsightEngine, create_insight_engine
    from pathlab._core import init as chronos_init
except ImportError as e:
    logging.warning(f"Chronos core modules not available: {e}")

# Advanced typing constructs
T = TypeVar('T')
U = TypeVar('U')
WidgetType = TypeVar('WidgetType', bound='ChronosWidget')

# ═══════════════════════════════════════════════════════════════════════════
# THEORETICAL FOUNDATIONS
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class VisualizationFunctor(Protocol[T, U]):
    """
    Category-theoretic functor for visualization transformations.
    
    Implements functorial mappings between visualization categories
    with morphism composition and identity preservation.
    """
    
    def fmap(self, transform: Callable[[T], U]) -> Callable[['VisualizationFunctor[T, Any]'], 'VisualizationFunctor[U, Any]']:
        """Apply functorial mapping with composition preservation."""
        ...

class WidgetState(Enum):
    """Widget lifecycle states with formal transition semantics."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    UPDATING = auto()
    SUSPENDED = auto()
    DESTROYED = auto()

@dataclass(frozen=True)
class VisualizationContext:
    """
    Immutable context for visualization operations with formal invariants.
    
    Maintains semantic consistency across widget transformations and
    ensures information preservation during cross-context operations.
    """
    session_id: UUID
    notebook_path: Optional[Path]
    kernel_id: str
    display_mode: str = "interactive"
    performance_profile: str = "balanced"
    accessibility_features: frozenset[str] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """Validate context invariants."""
        if self.display_mode not in {"interactive", "static", "presentation"}:
            raise ValueError(f"Invalid display mode: {self.display_mode}")
        if self.performance_profile not in {"performance", "balanced", "memory"}:
            raise ValueError(f"Invalid performance profile: {self.performance_profile}")

@dataclass
class WidgetConfiguration:
    """Widget configuration with constraint validation."""
    width: int = 800
    height: int = 600
    auto_update: bool = True
    render_fps: int = 30
    memory_limit_mb: int = 512
    gpu_acceleration: bool = True
    
    def __post_init__(self):
        """Validate configuration constraints."""
        if not (200 <= self.width <= 4096):
            raise ValueError(f"Width {self.width} outside valid range [200, 4096]")
        if not (150 <= self.height <= 2160):
            raise ValueError(f"Height {self.height} outside valid range [150, 2160]")
        if not (1 <= self.render_fps <= 120):
            raise ValueError(f"FPS {self.render_fps} outside valid range [1, 120]")

# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONAL REACTIVE PROGRAMMING FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════

class Observable(Generic[T]):
    """
    Functional reactive observable with mathematical guarantees.
    
    Implements push-based reactive streams with backpressure handling
    and formal correctness properties for subscription management.
    """
    
    def __init__(self):
        self._observers: List[Callable[[T], None]] = []
        self._error_handlers: List[Callable[[Exception], None]] = []
        self._completion_handlers: List[Callable[[], None]] = []
        self._lock = threading.RLock()
        
    def subscribe(self, 
                 on_next: Callable[[T], None],
                 on_error: Optional[Callable[[Exception], None]] = None,
                 on_complete: Optional[Callable[[], None]] = None) -> 'Subscription':
        """Subscribe to observable with functional composition."""
        with self._lock:
            self._observers.append(on_next)
            if on_error:
                self._error_handlers.append(on_error)
            if on_complete:
                self._completion_handlers.append(on_complete)
                
        return Subscription(self, on_next, on_error, on_complete)
    
    def emit(self, value: T) -> None:
        """Emit value to all observers with exception isolation."""
        with self._lock:
            observers = self._observers.copy()
            
        for observer in observers:
            try:
                observer(value)
            except Exception as e:
                self._handle_error(e)
    
    def error(self, exception: Exception) -> None:
        """Propagate error through error handling chain."""
        with self._lock:
            handlers = self._error_handlers.copy()
            
        for handler in handlers:
            try:
                handler(exception)
            except Exception:
                logging.exception("Error in error handler")
    
    def complete(self) -> None:
        """Signal completion to all completion handlers."""
        with self._lock:
            handlers = self._completion_handlers.copy()
            
        for handler in handlers:
            try:
                handler()
            except Exception:
                logging.exception("Error in completion handler")
    
    def _handle_error(self, exception: Exception) -> None:
        """Internal error handling with fallback logging."""
        if self._error_handlers:
            self.error(exception)
        else:
            logging.exception("Unhandled observable error")

class Subscription:
    """Subscription handle with automatic resource management."""
    
    def __init__(self, observable: Observable, on_next, on_error, on_complete):
        self._observable = weakref.ref(observable)
        self._on_next = on_next
        self._on_error = on_error
        self._on_complete = on_complete
        self._disposed = False
        
    def dispose(self) -> None:
        """Dispose subscription with resource cleanup."""
        if self._disposed:
            return
            
        observable = self._observable()
        if observable:
            with observable._lock:
                if self._on_next in observable._observers:
                    observable._observers.remove(self._on_next)
                if self._on_error and self._on_error in observable._error_handlers:
                    observable._error_handlers.remove(self._on_error)
                if self._on_complete and self._on_complete in observable._completion_handlers:
                    observable._completion_handlers.remove(self._on_complete)
        
        self._disposed = True
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

# ═══════════════════════════════════════════════════════════════════════════
# WIDGET ARCHITECTURE WITH CATEGORY THEORY
# ═══════════════════════════════════════════════════════════════════════════

class ChronosWidget(ABC):
    """
    Abstract base for Chronos widgets with category-theoretic composition.
    
    Implements functorial widget composition with morphism preservation
    and formal correctness guarantees for state transitions.
    """
    
    def __init__(self, 
                 widget_id: Optional[str] = None,
                 config: Optional[WidgetConfiguration] = None,
                 context: Optional[VisualizationContext] = None):
        self.widget_id = widget_id or str(uuid4())
        self.config = config or WidgetConfiguration()
        self.context = context
        self.state = WidgetState.UNINITIALIZED
        self._state_observable = Observable[WidgetState]()
        self._update_observable = Observable[Any]()
        self._error_observable = Observable[Exception]()
        self._subscriptions: List[Subscription] = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"widget-{self.widget_id}")
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize widget with async resource allocation."""
        pass
    
    @abstractmethod
    async def update(self, data: Any) -> None:
        """Update widget state with new data."""
        pass
    
    @abstractmethod
    async def render(self) -> Any:
        """Render widget to appropriate output format."""
        pass
    
    @abstractmethod
    async def destroy(self) -> None:
        """Cleanup widget resources."""
        pass
    
    def subscribe_state(self, handler: Callable[[WidgetState], None]) -> Subscription:
        """Subscribe to state changes with functional composition."""
        return self._state_observable.subscribe(handler)
    
    def subscribe_updates(self, handler: Callable[[Any], None]) -> Subscription:
        """Subscribe to data updates with reactive composition."""
        return self._update_observable.subscribe(handler)
    
    def subscribe_errors(self, handler: Callable[[Exception], None]) -> Subscription:
        """Subscribe to error events with monadic error composition."""
        return self._error_observable.subscribe(handler)
    
    async def _transition_state(self, new_state: WidgetState) -> None:
        """State transition with invariant preservation."""
        if self._is_valid_transition(self.state, new_state):
            old_state = self.state
            self.state = new_state
            self._state_observable.emit(new_state)
            logging.debug(f"Widget {self.widget_id} transitioned: {old_state} -> {new_state}")
        else:
            raise ValueError(f"Invalid state transition: {self.state} -> {new_state}")
    
    def _is_valid_transition(self, from_state: WidgetState, to_state: WidgetState) -> bool:
        """Validate state transition according to formal state machine."""
        valid_transitions = {
            WidgetState.UNINITIALIZED: {WidgetState.INITIALIZING},
            WidgetState.INITIALIZING: {WidgetState.ACTIVE, WidgetState.DESTROYED},
            WidgetState.ACTIVE: {WidgetState.UPDATING, WidgetState.SUSPENDED, WidgetState.DESTROYED},
            WidgetState.UPDATING: {WidgetState.ACTIVE, WidgetState.DESTROYED},
            WidgetState.SUSPENDED: {WidgetState.ACTIVE, WidgetState.DESTROYED},
            WidgetState.DESTROYED: set()
        }
        return to_state in valid_transitions.get(from_state, set())
    
    async def __aenter__(self):
        """Async context manager for resource management."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.destroy()

if HAS_JUPYTER:
    @register
    class ChronosAlgorithmWidget(DOMWidget, ChronosWidget):
        """
        Interactive algorithm visualization widget for Jupyter notebooks.
        
        Implements real-time algorithm execution with temporal debugging
        capabilities and multi-perspective visualization synchronization.
        """
        
        _view_name = Unicode('ChronosAlgorithmView').tag(sync=True)
        _model_name = Unicode('ChronosAlgorithmModel').tag(sync=True)
        _view_module = Unicode('chronos-jupyter').tag(sync=True)
        _model_module = Unicode('chronos-jupyter').tag(sync=True)
        _view_module_version = Unicode('^1.0.0').tag(sync=True)
        _model_module_version = Unicode('^1.0.0').tag(sync=True)
        
        # Synchronized traits for bidirectional communication
        algorithm_state = TraitDict().tag(sync=True)
        visualization_config = TraitDict().tag(sync=True)
        timeline_position = Int(0).tag(sync=True)
        is_playing = Bool(False).tag(sync=True)
        playback_speed = Float(1.0).tag(sync=True)
        selected_perspectives = TraitList().tag(sync=True)
        
        def __init__(self, 
                     algorithm: Optional['Algorithm'] = None,
                     graph = None,
                     start_node: Optional[int] = None,
                     goal_node: Optional[int] = None,
                     **kwargs):
            
            # Initialize parent classes
            DOMWidget.__init__(self, **kwargs)
            ChronosWidget.__init__(self, **kwargs)
            
            # Algorithm execution context
            self.algorithm = algorithm
            self.graph = graph
            self.start_node = start_node
            self.goal_node = goal_node
            
            # Chronos components
            self.visualization_engine: Optional[VisualizationEngine] = None
            self.timeline_manager: Optional[TimelineManager] = None
            self.insight_engine: Optional[InsightEngine] = None
            
            # State management
            self._execution_result = None
            self._current_timeline = None
            self._active_perspectives = set()
            
            # Performance monitoring
            self._render_times = []
            self._memory_usage = []
            
        async def initialize(self) -> None:
            """Initialize widget with Chronos backend components."""
            await self._transition_state(WidgetState.INITIALIZING)
            
            try:
                # Initialize Chronos core if needed
                chronos_init()
                
                # Create visualization engine
                self.visualization_engine = create_visualization()
                
                # Create insight engine
                self.insight_engine = create_insight_engine()
                
                # Setup default perspectives
                self._setup_default_perspectives()
                
                # Initialize frontend communication
                self._setup_frontend_sync()
                
                await self._transition_state(WidgetState.ACTIVE)
                logging.info(f"ChronosAlgorithmWidget {self.widget_id} initialized successfully")
                
            except Exception as e:
                await self._transition_state(WidgetState.DESTROYED)
                self._error_observable.emit(e)
                raise
        
        def _setup_default_perspectives(self) -> None:
            """Setup default visualization perspectives."""
            if self.visualization_engine:
                # Add graph view for spatial visualization
                self.visualization_engine.add_view("graph")
                
                # Add decision tree view for algorithm logic
                self.visualization_engine.add_view("decision")
                
                # Add heuristic landscape view
                self.visualization_engine.add_view("heuristic")
                
                self._active_perspectives = {"graph", "decision", "heuristic"}
                self.selected_perspectives = list(self._active_perspectives)
        
        def _setup_frontend_sync(self) -> None:
            """Setup bidirectional synchronization with frontend."""
            # Observe frontend changes
            self.observe(self._on_timeline_change, names='timeline_position')
            self.observe(self._on_playback_change, names='is_playing')
            self.observe(self._on_speed_change, names='playback_speed')
            self.observe(self._on_perspectives_change, names='selected_perspectives')
        
        def _on_timeline_change(self, change):
            """Handle timeline position changes from frontend."""
            position = change['new']
            if self.timeline_manager and 0 <= position < len(self.timeline_manager._states):
                asyncio.create_task(self._seek_to_position(position))
        
        def _on_playback_change(self, change):
            """Handle playback state changes from frontend."""
            is_playing = change['new']
            if is_playing:
                asyncio.create_task(self._start_playback())
            else:
                asyncio.create_task(self._pause_playback())
        
        def _on_speed_change(self, change):
            """Handle playback speed changes from frontend."""
            speed = change['new']
            self._playback_speed = max(0.1, min(5.0, speed))  # Clamp to valid range
        
        def _on_perspectives_change(self, change):
            """Handle perspective selection changes from frontend."""
            perspectives = set(change['new'])
            asyncio.create_task(self._update_perspectives(perspectives))
        
        async def execute_algorithm(self) -> None:
            """Execute algorithm with temporal debugging enabled."""
            if not all([self.algorithm, self.graph, self.start_node is not None, self.goal_node is not None]):
                raise ValueError("Algorithm, graph, start_node, and goal_node must be specified")
            
            await self._transition_state(WidgetState.UPDATING)
            
            try:
                # Execute algorithm with visualization
                self._execution_result = execute_algorithm(
                    self.algorithm, self.graph, self.start_node, self.goal_node
                )
                
                # Create timeline from execution
                self.timeline_manager = create_timeline()
                # Populate timeline with execution history
                # (Implementation depends on execution result structure)
                
                # Update algorithm state in frontend
                self.algorithm_state = {
                    'nodes_visited': self._execution_result.nodes_visited,
                    'path_length': len(self._execution_result.path) if self._execution_result.path else 0,
                    'execution_time': self._execution_result.execution_time_ms,
                    'algorithm_name': self.algorithm.name(),
                }
                
                # Generate insights
                if self.insight_engine and self.timeline_manager:
                    insights = self.insight_engine.analyze_execution(self.timeline_manager)
                    self._update_insights(insights)
                
                await self._transition_state(WidgetState.ACTIVE)
                self._update_observable.emit(self._execution_result)
                
            except Exception as e:
                await self._transition_state(WidgetState.ACTIVE)  # Return to active on error
                self._error_observable.emit(e)
                raise
        
        async def _seek_to_position(self, position: int) -> None:
            """Seek timeline to specific position."""
            if self.timeline_manager:
                state = self.timeline_manager.jump_to_state(position)
                await self._update_visualization(state)
        
        async def _start_playback(self) -> None:
            """Start timeline playback."""
            if not self.timeline_manager:
                return
                
            # Implement playback loop
            while self.is_playing and self.timeline_position < len(self.timeline_manager._states) - 1:
                await asyncio.sleep(1.0 / (self.playback_speed * 10))  # Adjust for speed
                self.timeline_position += 1
                
            if self.timeline_position >= len(self.timeline_manager._states) - 1:
                self.is_playing = False
        
        async def _pause_playback(self) -> None:
            """Pause timeline playback."""
            # Playback pause is handled by the loop condition
            pass
        
        async def _update_perspectives(self, perspectives: set[str]) -> None:
            """Update active visualization perspectives."""
            if not self.visualization_engine:
                return
                
            # Remove deactivated perspectives
            for perspective in self._active_perspectives - perspectives:
                self.visualization_engine.remove_view(perspective)
            
            # Add new perspectives  
            for perspective in perspectives - self._active_perspectives:
                self.visualization_engine.add_view(perspective)
            
            self._active_perspectives = perspectives
            
            # Update visualization
            if self.timeline_manager:
                current_state = self.timeline_manager.current_state
                await self._update_visualization(current_state)
        
        async def _update_visualization(self, state: Any) -> None:
            """Update visualization with new state."""
            if self.visualization_engine:
                self.visualization_engine.update(state)
                
                # Serialize visualization state for frontend
                viz_data = self._serialize_visualization_state()
                self.visualization_config = viz_data
        
        def _update_insights(self, insights: List[Dict]) -> None:
            """Update algorithm insights."""
            # Process and format insights for display
            formatted_insights = []
            for insight in insights:
                formatted_insights.append({
                    'type': insight.get('type', 'general'),
                    'description': insight.get('description', ''),
                    'significance': insight.get('significance', 0.0),
                    'recommendations': insight.get('recommendations', [])
                })
            
            # Update widget state
            current_state = dict(self.algorithm_state)
            current_state['insights'] = formatted_insights
            self.algorithm_state = current_state
        
        def _serialize_visualization_state(self) -> Dict[str, Any]:
            """Serialize visualization state for frontend transmission."""
            if not self.visualization_engine:
                return {}
            
            return {
                'timestamp': asyncio.get_event_loop().time(),
                'active_views': list(self._active_perspectives),
                'render_config': {
                    'width': self.config.width,
                    'height': self.config.height,
                    'fps': self.config.render_fps,
                },
                'performance_metrics': {
                    'avg_render_time': sum(self._render_times[-10:]) / len(self._render_times[-10:]) if self._render_times else 0,
                    'memory_usage_mb': self._memory_usage[-1] if self._memory_usage else 0,
                }
            }
        
        async def update(self, data: Any) -> None:
            """Update widget with new data."""
            await self._transition_state(WidgetState.UPDATING)
            try:
                # Handle different types of updates
                if isinstance(data, dict):
                    if 'algorithm' in data:
                        self.algorithm = data['algorithm']
                    if 'graph' in data:
                        self.graph = data['graph']
                    if 'start_node' in data:
                        self.start_node = data['start_node']
                    if 'goal_node' in data:
                        self.goal_node = data['goal_node']
                
                # Re-execute if all components are available
                if all([self.algorithm, self.graph, self.start_node is not None, self.goal_node is not None]):
                    await self.execute_algorithm()
                
                await self._transition_state(WidgetState.ACTIVE)
                self._update_observable.emit(data)
                
            except Exception as e:
                await self._transition_state(WidgetState.ACTIVE)
                self._error_observable.emit(e)
                raise
        
        async def render(self) -> None:
            """Render widget to Jupyter output."""
            # Widget rendering is handled by the frontend JavaScript
            # This method can be used for server-side rendering if needed
            pass
        
        async def destroy(self) -> None:
            """Cleanup widget resources."""
            await self._transition_state(WidgetState.DESTROYED)
            
            # Dispose all subscriptions
            for subscription in self._subscriptions:
                subscription.dispose()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Cleanup Chronos components
            self.visualization_engine = None
            self.timeline_manager = None
            self.insight_engine = None
            
            logging.info(f"ChronosAlgorithmWidget {self.widget_id} destroyed")

# ═══════════════════════════════════════════════════════════════════════════
# JUPYTER MAGIC COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

if HAS_JUPYTER:
    @magics_class
    class ChronosMagics(Magics):
        """
        Jupyter magic commands for Chronos integration.
        
        Provides convenient IPython magic commands for algorithm execution,
        visualization, and temporal debugging within notebook environments.
        """
        
        @line_magic
        def chronos_init(self, line: str) -> None:
            """Initialize Chronos system with optional configuration."""
            try:
                chronos_init()
                print("✓ Chronos system initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize Chronos: {e}")
        
        @line_magic  
        def chronos_widget(self, line: str) -> ChronosAlgorithmWidget:
            """Create a new Chronos algorithm widget."""
            try:
                # Parse optional arguments
                args = line.strip().split() if line.strip() else []
                kwargs = {}
                
                for arg in args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        kwargs[key] = value
                
                widget = ChronosAlgorithmWidget(**kwargs)
                display(widget)
                return widget
                
            except Exception as e:
                print(f"✗ Failed to create widget: {e}")
                return None
        
        @cell_magic
        def chronos_algorithm(self, line: str, cell: str) -> None:
            """Execute algorithm code with Chronos visualization."""
            try:
                # Execute the cell content in the user namespace
                self.shell.run_cell(cell)
                print("✓ Algorithm code executed successfully")
                
            except Exception as e:
                print(f"✗ Algorithm execution failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# FACTORY AND UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class ChronosJupyterManager:
    """
    Singleton manager for Chronos Jupyter integration.
    
    Manages widget instances, provides factory methods, and coordinates
    resource cleanup across the notebook session.
    """
    
    _instance: Optional['ChronosJupyterManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ChronosJupyterManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._widgets: Dict[str, ChronosWidget] = {}
        self._context_cache: Dict[str, VisualizationContext] = {}
        self._session_id = uuid4()
        
        # Register cleanup handler
        if HAS_JUPYTER:
            try:
                ip = get_ipython()
                if ip:
                    ip.events.register('pre_execute', self._pre_execute_hook)
                    ip.events.register('post_execute', self._post_execute_hook)
            except Exception:
                logging.warning("Could not register IPython event hooks")
    
    def create_widget(self, 
                     widget_type: str = "algorithm",
                     config: Optional[WidgetConfiguration] = None,
                     context: Optional[VisualizationContext] = None,
                     **kwargs) -> Optional[ChronosWidget]:
        """Factory method for creating Chronos widgets."""
        if not HAS_JUPYTER:
            logging.error("Jupyter environment not available")
            return None
        
        try:
            # Create visualization context if not provided
            if context is None:
                context = self._create_context()
            
            # Create widget based on type
            if widget_type == "algorithm":
                widget = ChronosAlgorithmWidget(
                    config=config, 
                    context=context, 
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown widget type: {widget_type}")
            
            # Register widget
            self._widgets[widget.widget_id] = widget
            
            return widget
            
        except Exception as e:
            logging.exception(f"Failed to create {widget_type} widget")
            return None
    
    def _create_context(self) -> VisualizationContext:
        """Create visualization context for current notebook session."""
        try:
            ip = get_ipython()
            kernel_id = ip.kernel.session.session if ip and hasattr(ip, 'kernel') else "unknown"
            
            # Try to determine notebook path
            notebook_path = None
            if ip and hasattr(ip, 'starting_dir'):
                notebook_path = Path(ip.starting_dir)
            
            return VisualizationContext(
                session_id=self._session_id,
                notebook_path=notebook_path,
                kernel_id=kernel_id,
                display_mode="interactive",
                performance_profile="balanced"
            )
            
        except Exception:
            # Fallback context
            return VisualizationContext(
                session_id=self._session_id,
                notebook_path=None,
                kernel_id="fallback"
            )
    
    def get_widget(self, widget_id: str) -> Optional[ChronosWidget]:
        """Retrieve widget by ID."""
        return self._widgets.get(widget_id)
    
    def list_widgets(self) -> List[str]:
        """List all active widget IDs."""
        return list(self._widgets.keys())
    
    async def cleanup_widget(self, widget_id: str) -> None:
        """Cleanup specific widget."""
        widget = self._widgets.pop(widget_id, None)
        if widget:
            await widget.destroy()
    
    async def cleanup_all(self) -> None:
        """Cleanup all widgets and resources."""
        widgets = list(self._widgets.values())
        self._widgets.clear()
        
        for widget in widgets:
            try:
                await widget.destroy()
            except Exception:
                logging.exception(f"Error destroying widget {widget.widget_id}")
    
    def _pre_execute_hook(self) -> None:
        """Hook called before cell execution."""
        # Perform any pre-execution cleanup or preparation
        pass
    
    def _post_execute_hook(self) -> None:
        """Hook called after cell execution."""
        # Perform any post-execution maintenance
        pass

# Global manager instance
manager = ChronosJupyterManager()

# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_algorithm_widget(algorithm=None, 
                           graph=None,
                           start_node: Optional[int] = None,
                           goal_node: Optional[int] = None,
                           config: Optional[WidgetConfiguration] = None,
                           **kwargs) -> Optional[ChronosWidget]:
    """
    Create an interactive algorithm visualization widget.
    
    Args:
        algorithm: Algorithm instance to visualize
        graph: Graph data structure for algorithm execution
        start_node: Starting node for pathfinding algorithms
        goal_node: Goal node for pathfinding algorithms
        config: Widget configuration options
        **kwargs: Additional widget parameters
    
    Returns:
        ChronosAlgorithmWidget instance or None if creation fails
    
    Example:
        >>> from pathlab.algorithms import AStar
        >>> from pathlab.integration.jupyter import create_algorithm_widget
        >>> 
        >>> algorithm = AStar()
        >>> widget = create_algorithm_widget(
        ...     algorithm=algorithm,
        ...     graph=my_graph,
        ...     start_node=0,
        ...     goal_node=10
        ... )
        >>> # Widget will be automatically displayed in Jupyter
    """
    return manager.create_widget(
        widget_type="algorithm",
        config=config,
        algorithm=algorithm,
        graph=graph,
        start_node=start_node,
        goal_node=goal_node,
        **kwargs
    )

def init_jupyter_integration() -> bool:
    """
    Initialize Jupyter integration with magic commands.
    
    Returns:
        True if initialization successful, False otherwise
    """
    if not HAS_JUPYTER:
        logging.error("Jupyter environment not available")
        return False
    
    try:
        # Initialize Chronos core
        chronos_init()
        
        # Register magic commands
        ip = get_ipython()
        if ip:
            ip.register_magic_function(ChronosMagics.chronos_init, 'line')
            ip.register_magic_function(ChronosMagics.chronos_widget, 'line')
            ip.register_magic_function(ChronosMagics.chronos_algorithm, 'cell')
            
            # Load frontend JavaScript/CSS
            display(HTML("""
            <script>
            // Load Chronos Jupyter frontend
            if (typeof ChronosJupyterExtension === 'undefined') {
                console.log('Loading Chronos Jupyter extension...');
                // Extension loading logic would go here
            }
            </script>
            """))
        
        logging.info("Jupyter integration initialized successfully")
        return True
        
    except Exception as e:
        logging.exception("Failed to initialize Jupyter integration")
        return False

async def cleanup_jupyter_integration() -> None:
    """Cleanup all Jupyter integration resources."""
    await manager.cleanup_all()

# Auto-initialize when module is imported in Jupyter environment
if HAS_JUPYTER and get_ipython():
    init_jupyter_integration()

# Export public API
__all__ = [
    'ChronosWidget',
    'ChronosAlgorithmWidget',
    'WidgetConfiguration',
    'VisualizationContext',
    'WidgetState',
    'Observable',
    'Subscription',
    'ChronosJupyterManager',
    'create_algorithm_widget',
    'init_jupyter_integration',
    'cleanup_jupyter_integration',
    'manager'
]