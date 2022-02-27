# ```flop```
Facility location optimization

## Installation
x

## Quick start

```python
from flop import ProblemFactory, Location

factory = ProblemFactory(
    bounds=[
        Location(lat=5.95, lon=45.81),
        Location(lat=10.49, lon=45.81),
        Location(lat=10.49, lon=47.80),
        Location(lat=5.95, lon=47.80),
    ]
)
problem = factory.sample_problem(seed=42)

```

```python
from flop import FacilityLocationOptimizer

optimizer = FacilityLocationOptimizer(
    max_seconds=60,
    max_seconds_same_incumbent=20
)
result = optimizer.solve(problem=problem)
```

```python
from flop import plot_problem, plot_solution

problem_figure = plot_problem(problem)
solution_figure = plot_solution(problem, solution, show_unused_facilities=True)
```



## Running tests
x

