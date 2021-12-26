from typing import Optional, Union

import numpy as np
from shapely.geometry.polygon import Polygon as Polygon, Point

from flop.base import DemandCentre, FacilityCandidate, Location, ProblemData


class ProblemFactory:

    def __init__(
            self,
            polygon: Polygon,
            n_periods: int = 12,
            n_facility_candidates: int = 20,
            n_demand_centers: int = 10,
            mean_capacity_max: float = 1e2,
            mean_cost_fixed: float = 1e2,
            mean_cost_variable: float = 5.,
            mean_demand: float = 1e1,
            cost_transport: float = 1.,
            cost_unmet_demand: Optional[float] = None,
            discount_factor: float = 1.
    ):
        self.polygon = polygon
        self.n_periods = n_periods
        self.n_facility_candidates = n_facility_candidates
        self.n_demand_centers = n_demand_centers
        self.mean_capacity_max = mean_capacity_max
        self.mean_demand = mean_demand
        self.mean_cost_fixed = mean_cost_fixed
        self.mean_cost_variable = mean_cost_variable
        self.cost_transport = cost_transport
        self.cost_unmet_demand = cost_unmet_demand
        self.discount_factor = discount_factor

    def sample_location(self) -> Location:
        lon_min, lat_min, lon_max, lat_max = self.polygon.bounds
        while True:
            lat = self._sample_uniform(low=lat_min, high=lat_max)
            lon = np.random.uniform(low=lon_min, high=lon_max)
            if self.polygon.contains(other=Point(lon, lat)):
                return Location(lat=lat, lon=lon)

    def sample_demand_center(self, name: str) -> DemandCentre:
        return DemandCentre(
            name=name,
            location=self.sample_location(),
            demand_variable=self._sample_truncnorm(
                mean=self.mean_demand,
                size=self.n_periods
            ) if self.n_periods > 1 else None,
            demand_fixed=self._sample_truncnorm(mean=self.mean_demand) if self.n_periods == 1 else None
        )

    def sample_facility_candidate(self, name: str) -> FacilityCandidate:
        capacity_max = self._sample_truncnorm(mean=self.mean_capacity_max)
        return FacilityCandidate(
            name=name,
            location=self.sample_location(),
            capacity_max=capacity_max,
            capacity_min=capacity_max * self._sample_uniform(low=0, high=0.2),
            cost_fixed=self._sample_truncnorm(mean=self.mean_cost_fixed),
            cost_variable=self._sample_truncnorm(mean=self.mean_cost_variable),
        )

    def sample_problem(self, seed: Optional[int] = None, ensure_feasibility: bool = True) -> ProblemData:

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        # Randomly sample demand centers  and facility candidates
        demand_centers = [
            self.sample_demand_center(name=f"demand_center_{i}")
            for i in range(self.n_demand_centers)
        ]
        facility_candidates = [
            self.sample_facility_candidate(name=f"facility_{i}")
            for i in range(self.n_facility_candidates)
        ]

        # If necessary, sample additional facility candidates to ensure that the problem is feasible
        if ensure_feasibility and self.cost_unmet_demand is None:
            total_unmet_demand = max(
                sum(demand_center.demand for demand_center in demand_centers).max()
                - sum(facility_candidate.capacity_max for facility_candidate in facility_candidates),
                0
            )
            i = self.n_facility_candidates + 1
            while total_unmet_demand > 0:
                facility_candidates.append(self.sample_facility_candidate(name=f"facility_candidate_{i}"))
                total_unmet_demand -= facility_candidates[-1].capacity_max
                i += 1

        return ProblemData(
            demand_centers=demand_centers,
            facility_candidates=facility_candidates,
            cost_transport=self.cost_transport,
            cost_unmet_demand=self.cost_unmet_demand
        )

    @staticmethod
    def _sample_truncnorm(mean: float, size: Optional[int] = None) -> Union[np.ndarray, float]:
        return np.clip(
            np.random.normal(loc=mean, scale=np.sqrt(mean), size=size),
            a_min=0,
            a_max=2 * mean
        )

    @staticmethod
    def _sample_uniform(low: float, high: float, size: Optional[int] = None) -> Union[np.ndarray, float]:
        return np.random.uniform(low=low, high=high, size=size)