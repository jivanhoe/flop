# ```flop```
Facility location optimization

## Installation
x

## Quick start

```python
from flop import FacilityLocationOptimizer, ProblemFactory, Location

factory = ProblemFactory(
    bounds=[
        Location(lat=5.95, lon=45.81),
        Location(lat=10.49, lon=45.81),
        Location(lat=10.49, lon=47.80),
        Location(lat=5.95, lon=47.80),
    ]
)
problem = factory.sample_problem()
optimizer = FacilityLocationOptimizer()
result = optimizer.solve(problem=problem)
```

## Running tests
x

