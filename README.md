# CHRONOS: Temporal Algorithmic Observatory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/chronos.svg)]()
[![PyPI Version](https://img.shields.io/pypi/v/chronos.svg)]()
[![CI Status](https://github.com/TensorScholar/chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorScholar/chronos/actions/workflows/ci.yml)

**CHRONOS** is a hybrid Rust/Python framework designed for algorithm visualization, education, and advanced temporal debugging.

## Installation

```bash
pip install -e ".[dev,ui,docs]"
```

## Quick Start

```python
import chronos as ch

# Create and populate a graph
graph = ch.data_structures.Graph()
n0 = graph.add_node(position=(0.0, 0.0))
n1 = graph.add_node(position=(1.0, 1.0))
graph.add_edge(n0, n1, weight=1.0)

# Run Dijkstra's algorithm
algo = ch.algorithms.Dijkstra()
executor = ch.execution.TracedExecutor(algo)
timeline = executor.run(graph, start_node=str(n0), goal_node=str(n1))

# Explore the execution
print(f"Algorithm completed in {timeline.total_steps} steps")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
