import pytest

from flop.model.base import Location, ProblemData
from flop.model.optimizer import FacilityLocationOptimizer
from flop.utils.factory import ProblemFactory


@pytest.fixture
def factory(request) -> ProblemFactory:
    return ProblemFactory(
        bounds=[
            Location(lat=5.955, lon=45.817),
            Location(lat=10.492, lon=45.817),
            Location(lat=10.492, lon=47.808),
            Location(lat=5.955, lon=47.808),
        ],
        **request.param
    )


@pytest.fixture(params=range(3))
def data(factory: ProblemFactory, request) -> ProblemData:
    return factory.sample_problem(ensure_feasibility=True, seed=request.param)


@pytest.fixture
def optimizer() -> FacilityLocationOptimizer:
    return FacilityLocationOptimizer(
        solver_name="CBC",
        max_seconds=60,
        max_seconds_same_incumbent=20,
        max_mip_gap=1e-3
    )
