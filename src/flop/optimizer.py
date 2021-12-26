from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mip
import pandas as pd

from flop.base import ProblemData, ProblemSolution, SolveInfo


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
            (self.used, 1, (None,)),
            (self.capacity, 1, self.used.shape),
            (self.capex, 2, (self.capex.shape[0], None)),
            (self.opex, 3, (self.capex.shape[0], None, self.capex.shape[1])),
            (self.supply, 3, self.opex.shape),
            (self.unmet_demand, 2, self.opex.shape[1:])
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
            used=model.add_var_tensor(shape=(n_facility_candidates,), var_type=mip.BINARY, name="used"),
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

    def solve(self, data: ProblemData) -> Optional[ProblemSolution]:
        model = self._setup_model()
        variables = self._define_variables(data=data, model=model)
        self._define_objective(data=data, model=model, variables=variables)
        self._add_constraints(data=data, model=model, variables=variables)
        self._optimize(model=model)
        return self._unpack_solution(data=data, model=model, variables=variables)

    def _setup_model(self) -> mip.Model:
        model = mip.Model(sense=mip.MINIMIZE, **(dict(solver_name=self.solver_name) if self.solver_name else dict()))
        if self.max_mip_gap is not None:
            model.max_mip_gap = self.max_mip_gap
        model.store_search_progress_log = True
        if not self.verbose:
            model.verbose = 0
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
                (data.discount_factor ** t) * (
                    mip.xsum(variables.capex[:, t])
                    + mip.xsum(variables.opex[:, :, t].flatten())
                    + (
                        data.cost_unmet_demand * mip.xsum(variables.unmet_demand[:, :, t].flatten())
                        if data.cost_unmet_demand is not None else 0
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
        distances = data.distances()

        for i, facility_candidate in enumerate(data.facility_candidates):

            # Add facility size constraints
            model.add_constr(variables.capacity[i] <= facility_candidate.capacity_max * variables.used[i])
            model.add_constr(variables.capacity[i] >= facility_candidate.capacity_min * variables.used[i])

            for t in range(data.n_periods):

                # Add supply constraint
                model.add_constr(mip.xsum(variables.supply[i, :, t]) <= variables.capacity[i])

                # Add capex constraint
                model.add_constr(
                    variables.capex[i, t]
                    == facility_candidate.cost_variable * variables.capacity[i]
                    + facility_candidate.cost_fixed * variables.used[i]
                )

                for j, demand_center in enumerate(data.demand_centers):

                    # Add opex constraint
                    model.add_constr(
                        variables.opex[i, j, t] == data.cost_transport * distances[i, j] * variables.supply[i, j, t]
                    )

                    # Add unmet constraints
                    if not i:
                        model.add_constr(
                            variables.unmet_demand[j, t]
                            == mip.xsum(variables.supply[:, j, t]) - demand_center.demand[t]
                        )
                        if data.cost_unmet_demand is None:
                            model.add_constr(variables.unmet_demand[j, t] == 0)

    def _optimize(self, model: mip.Model) -> None:
        solve_params = dict()
        if self.max_seconds is not None:
            solve_params["max_seconds"] = self.max_seconds
        if self.max_seconds_same_incumbent is not None:
            solve_params["max_seconds_same_incumbent"] = self.max_seconds_same_incumbent
        model.optimize(**solve_params)

    @staticmethod
    def _unpack_solution(
            data: ProblemData,
            model: mip.Model,
            variables: FacilityLocationVariables
    ) -> Optional[ProblemSolution]:
        if model.status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            return ProblemSolution(
                facilities=pd.DataFrame(
                    data=[
                        {
                            "facility": facility_candidate.name,
                            "used": bool(round(variables.used[i].x)),
                            "capacity": variables.capacity[i].x,
                            "capex_per_period": variables.capex[i, 0].x
                        }
                        for i, facility_candidate in enumerate(data.facility_candidates)
                    ]
                ).set_index("facility"),
                schedule=pd.DataFrame(
                    data=[
                        {
                            "period": t,
                            "facility": facility_candidate.name,
                            "demand_center": demand_center.name,
                            "supply": variables.supply[i, j, t].x,
                            "opex": variables.opex[i, j, t].x
                        }
                        for i, facility_candidate in enumerate(data.facility_candidates)
                        for j, demand_center in enumerate(data.demand_centers)
                        for t in range(data.n_periods)
                    ]
                ).set_index(["facility", "demand_center", "period"]),
                unmet_demand=pd.DataFrame(
                    data=[
                        {
                            "period": t,
                            "demand_center": demand_center.name,
                            "unmet_demand": variables.unmet_demand[j, t].x
                        }
                        for j, demand_center in enumerate(data.demand_centers)
                        for t in range(data.n_periods)
                    ]
                ).set_index(["demand_center", "period"]),
                solve_info=SolveInfo(
                    status=model.status,
                    progress_log=model.search_progress_log,
                    gap=model.gap
                )
            )
