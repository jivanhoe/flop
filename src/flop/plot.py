import folium
import numpy as np

from flop.base import ProblemData, ProblemSolution, Location
from typing import Optional


def get_map_centre(data: ProblemData) -> Location:
    return Location(
        lat=np.mean(
            [facility_candidate.location.lat for facility_candidate in data.facility_candidates]
            + [demand_center.location.lat for demand_center in data.demand_centers]
        ),
        lon=np.mean(
            [facility_candidate.location.lon for facility_candidate in data.facility_candidates]
            + [demand_center.location.lon for demand_center in data.demand_centers]
        )
    )


def make_map(data: ProblemData) -> folium.Map:
    return folium.Map(location=get_map_centre(data=data).to_tuple(), zoom_start=7)


def add_demand_centres_to_map(m: folium.Map, data: ProblemData, solution: Optional[ProblemSolution] = None) -> None:
    for demand_center in data.demand_centers:
        popup_text = (
            f"Name: {demand_center.name}<br>"
            f"Average demand: {round(demand_center.demand.mean())}"
        )
        if solution is not None and data.cost_unmet_demand is not None:
            info = solution.unmet_demand.loc[demand_center.name]
            popup_text += f"<br>Average unmet demand: {round(info['unmet_demand'].mean())}"
        folium.Marker(
            location=demand_center.location.to_tuple(),
            popup=folium.Popup(popup_text, max_width=1000),
            icon=folium.Icon(color="orange", icon="building", prefix="fa"),
        ).add_to(m)


def add_facilities_to_map(
        m: folium.Map,
        data: ProblemData,
        solution: Optional[ProblemSolution] = None,
        show_unused_facilities: bool = True
) -> None:
    for facility in data.facility_candidates:
        popup_text = f"Name: {facility.name}<br>"
        if solution is not None:
            info = solution.facilities.loc[facility.name]
            if info["used"]:
                color = "green"
                popup_text += (
                    f"Capacity: {round(info['capacity'])}<br>"
                    f"Status: USED<br>"
                    f"Cost: {round(info['capex_per_period'])}"
                )
            else:
                if not show_unused_facilities:
                    continue
                color = "gray"
                popup_text += f"Status: UNUSED"
        else:
            color = "blue"
            popup_text += (
                f"Capacity range: {round(facility.capacity_min)}-{round(facility.capacity_max)}<br>"
                f"Fixed cost: {round(facility.cost_fixed)}<br>"
                f"Variable cost: {round(facility.cost_variable)}"
            )

        folium.Marker(
            location=facility.location.to_tuple(),
            popup=folium.Popup(popup_text, max_width=1000),
            icon=folium.Icon(color=color, icon="industry", prefix="fa"),
        ).add_to(m)


def add_supply_routes_to_map(m: folium.Map, data: ProblemData, solution: ProblemSolution) -> None:
    for facility in data.facility_candidates:
        for demand_center in data.demand_centers:
            supply = solution.schedule.loc[facility.name, demand_center.name]["supply"]
            if (supply > 0).any():
                folium.PolyLine(
                    locations=[
                        facility.location.to_tuple(),
                        demand_center.location.to_tuple()
                    ],
                    popup=folium.Popup(f"Average supply: {round(supply.mean())}", max_width=1000),
                    opacity=0.75
                ).add_to(m)


def plot_problem(data: ProblemData) -> folium.Map:
    m = make_map(data=data)
    add_demand_centres_to_map(m=m, data=data)
    add_facilities_to_map(m=m, data=data)
    return m


def plot_solution(data: ProblemData, solution: ProblemSolution, show_unused_facilities: bool = True) -> folium.Map:
    m = make_map(data=data)
    add_demand_centres_to_map(m=m, data=data, solution=solution)
    add_facilities_to_map(m=m, data=data, solution=solution, show_unused_facilities=show_unused_facilities)
    add_supply_routes_to_map(m=m, data=data, solution=solution)
    return m

