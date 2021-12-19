from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mip
import numpy as np

from loco.base import ProblemData, ProblemSolution, Facility, SupplySchedule, SolveInfo


@dataclass
class FacilityLocationVariables:

    used: mip.LinExprTensor
    capacity: mip.LinExprTensor
    capex: mip.LinExprTensor
    opex: mip.LinExprTensor
    supply: mip.LinExprTensor
    unmet_demand: mip.LinExprTensor

    def __post_init__(self):
        for variable, ndim_expected, shape_expected in [
            (self.capex.ndim, 1, (None,)),
            (self.capacity.ndim, 1, self.capex.shape),
            (self.capex.ndim, 2, (self.capex.shape[0], None)),
            (self.opex.ndim, 3, (self.capex.shape[0], None, self.capex.shape[1])),
            (self.supply.ndim, 3, self.opex.shape),
            (self.unmet_demand.ndim, 2, self.opex.shape[1:])
        ]:
            assert variable.ndim == ndim_expected
            for i, expected in enumerate(shape_expected):
                if expected:
                    assert variable.shape[i] == expected

    @classmethod
    def from_data(cls, data: ProblemData, model: mip.Model) -> FacilityLocationVariables:
        n_periods = data.n_periods
        n_facility_candidates = len(data.facility_candidates)
        n_demand_centers = len(data.demand_centers)
        return cls(
            used=model.add_var_tensor(shape=(n_facility_candidates,), name="used"),
            capacity=model.add_var_tensor(shape=(n_facility_candidates,), name="size"),
            capex=model.add_var_tensor(shape=(n_facility_candidates, n_periods), name="capex"),
            opex=model.add_var_tensor(shape=(n_facility_candidates, n_demand_centers, n_periods), name="opex"),
            supply=model.add_var_tensor(shape=(n_facility_candidates, n_demand_centers, n_periods), name="supply"),
            unmet_demand=model.add_var_tensor(shape=(n_demand_centers, n_periods), name="unmet_demand")
        )


class FacilityLocationOptimizer:

    def __init__(
            self,
            solver_name: Optional[str] = None,
            max_mip_gap: Optional[float] = None,
            max_seconds: Optional[int] = None,
            max_seconds_same_incumbent: Optional[int] = None,
            tol: float = 1e-5,
            verbose: bool = False
    ):
        self.solver_name = solver_name
        self.max_mip_gap = max_mip_gap
        self.max_seconds = max_seconds
        self.max_seconds_same_incumbent = max_seconds_same_incumbent
        self.tol = tol
        self.verbose = verbose

    def optimize(self, data: ProblemData):
        model = self._setup_model()
        variables = self._define_variables(data=data, model=model)
        self._define_objective(data=data, model=model, variables=variables)
        self._add_constraints(data=data, model=model, variables=variables)
        model.optimize(max_seconds=self.max_seconds, max_seconds_same_incumbent=self.max_seconds_same_incumbent)
        return self._unpack_solution(data=data, model=model, variables=variables)

    def _setup_model(self) -> mip.Model:
        model = mip.Model(sense=mip.MINIMIZE, solver_name=self.solver_name)
        if self.max_mip_gap is not None:
            model.max_mip_gap = self.max_mip_gap
        model.store_search_progress_log = True
        return model

    @staticmethod
    def _define_variables(
            data: ProblemData,
            model: mip.Model
    ) -> FacilityLocationVariables:
        return FacilityLocationVariables.from_data(data=data, model=model)

    @staticmethod
    def _define_objective(
            data: ProblemData,
            model: mip.Model,
            variables: FacilityLocationVariables
    ) -> None:
        model.objective = mip.minimize(
            mip.xsum(
                (data.discount_factor ** t) * mip.xsum(
                    variables.capex[:, t]
                    + variables.opex[:, :, t].flatten()
                    + (
                        data.cost_unmet_demand * variables.unmet_demand[:, :, t].flatten()
                        if data.relax_demand_constraints else 0
                    )
                )
                for t in range(data.n_periods)
            )
        )

    @staticmethod
    def _add_constraints(
            data: ProblemData,
            model: mip.Model,
            variables: FacilityLocationVariables
    ) -> None:

        # Compute distance matrix
        distances = data.calculate_distances()

        for i, facility_candidate in enumerate(data.facility_candidates):

            # Add facility size constraints
            model.add_constr(variables.capacity[i] <= facility_candidate.capacity_max * variables.used[i])
            model.add_constr(variables.capacity[i] >= facility_candidate.capacity_min * variables.used[i])

            for t in range(data.n_periods):

                # Add capex constraint
                model.add_constr(
                    variables.capex[i, t]
                    == facility_candidate.cost_variable * variables.capacity[i]
                    + facility_candidate.cost_fixed * variables.used[i]
                )

        for j, demand_center in enumerate(data.demand_centers):
            for t in range(data.n_periods):

                # Add unmet constraints
                model.add_constr(
                    variables.unmet_demand[j, t]
                    == mip.xsum(variables.supply[:, j, t])
                    - (demand_center.demand_variable[t] if demand_center.demand_variable is not None else 0)
                )
                if not data.relax_demand_constraints:
                    model.add_constr(variables.unmet_demand == 0)

                for i, facility_candidate in enumerate(data.facility_candidates):

                    # Add opex constraint
                    model.add_constr(
                        variables.opex[i, j, t] == data.cost_transport * distances[i, j] * variables.supply[i, j, t]
                    )

                    # Add max radius constraint
                    if facility_candidate.service_radius:
                        if distances[i, j] > facility_candidate.service_radius:
                            model.add_constr(variables.supply[i, j, t] == 0)

    def _unpack_solution(
            self,
            data: ProblemData,
            model: mip.Model,
            variables: FacilityLocationVariables
    ) -> ProblemSolution:
        return ProblemSolution(
            facilities=[
                Facility(
                    name=facility_candidate.name,
                    location=facility_candidate.location,
                    capacity=variables.capacity[i],
                    service_radius=facility_candidate.service_radius
                )
                for i, facility_candidate in enumerate(data.facility_candidates)
                if variables.used[i].x > self.tol
            ],
            demand_centres=data.demand_centers,
            supply_schedules=[
                SupplySchedule(
                    facility_name=facility_candidate.name,
                    demand_center_name=demand_center.name,
                    supply=np.array([s.x for s in variables.supply[i, j]])
                    if data.n_periods > 1 else variables.supply[i, j, 0],
                )
                for i, facility_candidate in data.facility_candidates
                for j, demand_center in data.demand_centers
                if variables.supply[i, j].max() > self.tol
            ],
            unmet_demand={

            },
            unused_facility_candidates=[
                facility_candidate for i, facility_candidate in data.facility_candidates
                if not variables.used[i].x > self.tol
            ],
            solve_info=SolveInfo(
                status=model.status,
                progress_log=model.search_progress_log,
                gap=model.gap,
            )
        )