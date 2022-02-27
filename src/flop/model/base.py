from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from haversine import haversine
from mip import OptimizationStatus, ProgressLog


@dataclass
class Location:

    lat: float
    lon:  float

    def __post_init__(self):
        if abs(self.lat) > 90:
            raise ValueError
        if not (180 >= self.lon >= 0):
            raise ValueError

    def to_tuple(self) -> Tuple[float, float]:
        return self.lat, self.lon

    def distance(self, other: Location) -> float:
        return haversine(point1=self.to_tuple(), point2=other.to_tuple())


@dataclass
class FacilityCandidate:

    name: str
    location: Location
    cost_variable: float
    cost_fixed: float
    capacity_max: float
    capacity_min: float = 0.

    def __post_init__(self):
        if self.cost_variable < 0:
            raise ValueError
        if self.cost_fixed < 0:
            raise ValueError
        if not (self.capacity_max >= self.capacity_min >= 0):
            raise ValueError

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
            if not (1 <= len(self.demand_variable.shape) <= 2):
                raise ValueError
            if self.demand_variable.min() < 0:
                raise ValueError
            if self.n_periods <= 1:
                raise ValueError
            if self.demand_fixed is not None:
                if self.demand_fixed < 0:
                    raise ValueError
        else:
            if self.demand_fixed is None:
                raise ValueError
            if self.demand_fixed < 0:
                raise ValueError

    @property
    def demand(self) -> np.ndarray:
        return (
            (np.array([self.demand_fixed]) if self.demand_fixed is not None else 0)
            + (self.demand_variable if self.demand_variable is not None else 0)
        )

    @property
    def n_periods(self) -> int:
        if self.demand_variable is not None:
            return self.demand_variable.shape[0]
        return 1

    @property
    def n_samples(self) -> int:
        if self.demand_variable is not None:
            return self.demand_variable.shape[1]
        return 0

    def __hash__(self):
        return self.name


@dataclass
class Problem:

    facility_candidates: List[FacilityCandidate]
    demand_centers: List[DemandCentre]
    cost_transport: float
    cost_unmet_demand: Optional[float] = None
    discount_factor: float = 1.

    def __post_init__(self):
        if (
            len(set(facility_candidate.name for facility_candidate in self.facility_candidates))
            != len(self.facility_candidates)
        ):
            raise ValueError
        if (
            len(set(demand_center.name for demand_center in self.demand_centers))
            != len(self.demand_centers)
        ):
            raise ValueError
        if len(set(
            demand_center.n_periods for demand_center in self.demand_centers
            if demand_center.demand_variable is not None
        )) != 1:
            raise ValueError
        if self.cost_transport < 0:
            raise ValueError
        if self.cost_unmet_demand is not None:
            if self.cost_unmet_demand <= 0:
                raise ValueError
        if not(1 >= self.discount_factor >= 0):
            raise ValueError

    def compute_distances(self) -> np.ndarray:
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
        if self.gap < 0:
            raise ValueError


@dataclass
class Result:

    facilities: pd.DataFrame
    schedule: pd.DataFrame
    unmet_demand: pd.DataFrame
    solve_info: SolveInfo

    def __post_init__(self):
        pass


