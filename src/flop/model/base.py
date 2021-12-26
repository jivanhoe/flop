from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from haversine import haversine
from mip import OptimizationStatus, ProgressLog


@dataclass
class Location:

    lat: float
    lon:  float

    def __post_init__(self):
        assert -90 <= self.lat <= 90
        assert 0 <= self.lon <= 180

    def to_tuple(self) -> Tuple[float, float]:
        return self.lat, self.lon

    def distance(self, other: Location) -> float:
        return haversine(point1=self.to_tuple(), point2=other.to_tuple())


@dataclass
class Facility:

    name: str
    location: Location
    capacity: float
    service_radius: Optional[float] = None

    def __post_init__(self):
        assert self.capacity > 0
        if self.service_radius is not None:
            assert self.service_radius > 0

    def __hash__(self):
        return self.name


@dataclass
class FacilityCandidate:

    name: str
    location: Location
    cost_variable: float
    cost_fixed: float
    capacity_max: float
    capacity_min: float = 0.

    def __post_init__(self):
        assert self.cost_variable >= 0
        assert self.cost_fixed >= 0
        assert self.capacity_max >= self.capacity_min >= 0

    def __hash__(self):
        return self.name


@dataclass
class DemandCentre:

    name: str
    location: Location
    demand_variable: Optional[np.ndarray] = None
    demand_fixed: float = 0.

    def __post_init__(self):
        if self.demand_variable is not None:
            assert self.demand_variable.min() >= 0
            assert self.n_periods > 1
            if self.demand_fixed is not None:
                assert self.demand_fixed >= 0
        else:
            assert self.demand_fixed > 0

    @property
    def demand(self) -> np.ndarray:
        return (
            (np.array([self.demand_fixed]) if self.demand_fixed is not None else 0)
            + (self.demand_variable if self.demand_variable is not None else 0)
        )

    @property
    def n_periods(self) -> int:
        if self.demand_variable is not None:
            return len(self.demand_variable)
        return 1

    def __hash__(self):
        return self.name


@dataclass
class SupplySchedule:

    demand_center_name: str
    facility_name: str
    supply: Union[np.ndarray, float]

    def __post_init__(self):
        if isinstance(self.supply, np.ndarray):
            assert self.supply.min() >= 0
        else:
            assert self.supply > 0


@dataclass
class ProblemData:

    facility_candidates: List[FacilityCandidate]
    demand_centers: List[DemandCentre]
    cost_transport: float
    cost_unmet_demand: Optional[float] = None
    discount_factor: float = 1.

    def __post_init__(self):
        assert (
            len(set(facility_candidate.name for facility_candidate in self.facility_candidates))
            == len(self.facility_candidates)
        )
        assert (
            len(set(demand_center.name for demand_center in self.demand_centers))
            == len(self.demand_centers)
        )
        assert len(
            set(
                demand_center.n_periods for demand_center in self.demand_centers
                if demand_center.demand_variable is not None
            )
        ) == 1
        assert self.cost_transport >= 0
        if self.cost_unmet_demand is not None:
            assert self.cost_unmet_demand > 0
        assert 1 >= self.discount_factor >= 0

    def distances(self) -> np.ndarray:
        distances = np.full(shape=(len(self.facility_candidates), len(self.demand_centers)), fill_value=np.nan)
        for i, facility_candidate in enumerate(self.facility_candidates):
            for j, demand_center in enumerate(self.demand_centers):
                distances[i, j] = facility_candidate.location.distance(other=demand_center.location)
        return distances

    @property
    def n_periods(self) -> int:
        for demand_center in self.demand_centers:
            if demand_center.demand_variable is not None:
                return demand_center.n_periods
        return 1


@dataclass
class SolveInfo:

    status: OptimizationStatus
    progress_log: ProgressLog
    gap: float

    def __post_init__(self):
        assert self.gap >= 0


@dataclass
class ProblemSolution:

    facilities: pd.DataFrame
    schedule: pd.DataFrame
    unmet_demand: pd.DataFrame
    solve_info: SolveInfo

    def __post_init__(self):
        pass


