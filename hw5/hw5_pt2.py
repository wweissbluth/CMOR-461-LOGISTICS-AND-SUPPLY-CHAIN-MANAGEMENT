"""
projected_model.py

This module imports the BeerOptimization class from hw5_pt1 and defines a new class,
MultiPeriodModel, that builds a model with explicit decision periods t = 1,2,3.
It then projects the period 3 results to periods 3–20 in the objective function.
When printing the solution, only nonzero (or non‐negligible) decision variable values are printed.
Additionally, the module prints tables of:
  - Optimal total discounted transportation, facility opening, and capacity expansion costs
  - The optimal distribution plan.
A sensitivity analysis function is provided to perform scenario analysis on beer demand.
Finally, an experiment is run with the discount rate decreased to 1%, and its cost breakdown is printed.
"""

import numpy as np
from gurobipy import Model, GRB, quicksum
from hw5_pt1 import BeerOptimization


class MultiPeriodModel(BeerOptimization):
    def __init__(self):
        super().__init__()
        # Instead of t=1,...,20, we model only t=1,2,3 explicitly.
        self.T = [1, 2, 3]
        # For potential breweries (indices 2,3,4): opening/expansion costs and base capacity.
        self.opening_costs = {2: 75, 3: 70, 4: 68}
        self.expansion_costs = {2: 30, 3: 27, 4: 25}
        # Existing breweries: fixed capacities.
        self.fixed_capacities = {0: 220, 1: 200}
        # For potential breweries: if opened, base capacity is 70; if expanded, add another 70.
        self.base_capacity = 70
        self.expansion_increment = 50
        # Discount rate for NPV calculation (default = 10%).
        self.discount_rate = 0.1
        # Precompute the "steady-state" factor for periods 3–20.
        self.steady_state_factor = sum([1 / ((1 + self.discount_rate) ** t) for t in range(3, 21)])

    def build_model(self):
        """
        Builds the multi‐period model for t = 1,2,3.
        The cost in period 3 is projected to periods 3–20 via the steady_state_factor.
        """
        model = Model("Projected Beer Optimization")
        model.setParam("OutputFlag", 0)

        T = self.T
        D = range(len(self.distribution_centers))  # 6 distribution centers
        B = range(len(self.breweries))              # 5 breweries
        I = range(len(self.malt_plants))            # 3 malt plants

        # Decision variables:
        X = model.addVars(D, B, T, vtype=GRB.CONTINUOUS, name="X")
        M = model.addVars(B, I, T, vtype=GRB.CONTINUOUS, name="M")
        y = model.addVars([b for b in B if b >= 2], T, vtype=GRB.BINARY, name="y")
        z = model.addVars([b for b in B if b >= 2], T, vtype=GRB.BINARY, name="z")

        # Demand: for t=1,2 use original year 1 and 2 demand; for t=3 use year 3 demand.
        demand = {}
        for t in T:
            if t <= 2:
                demand[t] = {d: self.annual_demand[d, t - 1] for d in range(len(self.distribution_centers))}
            else:
                demand[t] = {d: self.annual_demand[d, 2] for d in range(len(self.distribution_centers))}

        # Malt capacity constraints:
        malt_caps = [30, 68, 20]
        for t in T:
            for i in range(len(self.malt_plants)):
                model.addConstr(quicksum(M[b, i, t] for b in range(len(self.breweries))) <= malt_caps[i],
                                name=f"MaltCap_{i}_t{t}")

        # Yield values:
        yield_val = {b: self.domestic_yield if b < 2 else self.import_yield for b in range(len(self.breweries))}

        # Production linking constraints:
        for t in T:
            for b in range(len(self.breweries)):
                model.addConstr(quicksum(X[d, b, t] for d in range(len(self.distribution_centers))) <=
                                yield_val[b] * quicksum(M[b, i, t] for i in range(len(self.malt_plants))),
                                name=f"ProdLink_b{b}_t{t}")

        # Brewery capacity constraints:
        for t in T:
            for b in range(len(self.breweries)):
                if b < 2:
                    model.addConstr(quicksum(X[d, b, t] for d in range(len(self.distribution_centers))) <=
                                    self.fixed_capacities[b],
                                    name=f"Cap_b{b}_t{t}")
                else:
                    model.addConstr(quicksum(X[d, b, t] for d in range(len(self.distribution_centers))) <=
                                    self.base_capacity * quicksum(y[b, tau] for tau in T if tau <= t) +
                                    self.expansion_increment * quicksum(z[b, tau] for tau in T if tau <= t),
                                    name=f"Cap_b{b}_t{t}")

        # Linking expansion to opening:
        for b in [b for b in range(len(self.breweries)) if b >= 2]:
            for t in T:
                model.addConstr(quicksum(z[b, tau] for tau in T if tau <= t) <=
                                quicksum(y[b, tau] for tau in T if tau <= t),
                                name=f"ExpandOpen_b{b}_t{t}")

        # Demand satisfaction:
        for t in T:
            for d in range(len(self.distribution_centers)):
                model.addConstr(quicksum(X[d, b, t] for b in range(len(self.breweries))) >= demand[t][d],
                                name=f"Demand_d{d}_t{t}")

        # Objective function:
        def period_cost(t):
            beer_cost = quicksum(self.beer_shipping_costs[b, d] * X[d, b, t] for d in range(len(self.distribution_centers)) for b in range(len(self.breweries)))
            malt_cost = quicksum(self.malt_shipping_costs[i, b] * M[b, i, t] for b in range(len(self.breweries)) for i in range(len(self.malt_plants)))
            open_cost = quicksum(self.opening_costs[b] * y[b, t] for b in [b for b in range(len(self.breweries)) if b >= 2])
            exp_cost = quicksum(self.expansion_costs[b] * z[b, t] for b in [b for b in range(len(self.breweries)) if b >= 2])
            return beer_cost + malt_cost + open_cost + exp_cost

        discount = lambda t: 1 / ((1 + self.discount_rate) ** t)
        obj = discount(1) * period_cost(1) + discount(2) * period_cost(2) + self.steady_state_factor * period_cost(3)
        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        self.model = model
        self.X = X
        self.M = M
        self.y = y
        self.z = z
        return model

    def solve_model(self):
        self.model.optimize()
        return self.model

    def print_solution(self, tol=1e-6):
        if self.model.status == GRB.OPTIMAL:
            print("Optimal Solution (nonzero values):")
            for var in self.model.getVars():
                if abs(var.X) > tol:
                    print(f"{var.VarName} = {var.X}")
            print(f"\nObjective Value (NPV): {self.model.objVal:.2f} Million $")
        else:
            print("No optimal solution found.")

    def print_cost_breakdown(self, tol=1e-6):
        """
        Calculates and prints a table showing the optimal total discounted costs broken down by:
          - Transportation cost (beer + malt)
          - Facility opening cost
          - Capacity expansion cost
        Also prints the optimal distribution plan.
        """
        if self.model.status != GRB.OPTIMAL:
            print("Model not solved optimally.")
            return

        discount = lambda t: 1 / ((1 + self.discount_rate) ** t)
        beer_cost_total = 0.0
        malt_cost_total = 0.0
        open_cost_total = 0.0
        exp_cost_total = 0.0

        for t in [1, 2]:
            beer_cost = sum(self.beer_shipping_costs[b, d] * self.X[d, b, t].X
                            for d in range(len(self.distribution_centers))
                            for b in range(len(self.breweries)))
            malt_cost = sum(self.malt_shipping_costs[i, b] * self.M[b, i, t].X
                            for b in range(len(self.breweries))
                            for i in range(len(self.malt_plants)))
            open_cost = sum(self.opening_costs[b] * self.y[b, t].X
                            for b in [b for b in range(len(self.breweries)) if b >= 2])
            exp_cost = sum(self.expansion_costs[b] * self.z[b, t].X
                           for b in [b for b in range(len(self.breweries)) if b >= 2])
            beer_cost_total += discount(t) * beer_cost
            malt_cost_total += discount(t) * malt_cost
            open_cost_total += discount(t) * open_cost
            exp_cost_total += discount(t) * exp_cost

        t = 3
        beer_cost = sum(self.beer_shipping_costs[b, d] * self.X[d, b, t].X
                        for d in range(len(self.distribution_centers))
                        for b in range(len(self.breweries)))
        malt_cost = sum(self.malt_shipping_costs[i, b] * self.M[b, i, t].X
                        for b in range(len(self.breweries))
                        for i in range(len(self.malt_plants)))
        open_cost = sum(self.opening_costs[b] * self.y[b, t].X
                        for b in [b for b in range(len(self.breweries)) if b >= 2])
        exp_cost = sum(self.expansion_costs[b] * self.z[b, t].X
                       for b in [b for b in range(len(self.breweries)) if b >= 2])
        beer_cost_total += self.steady_state_factor * beer_cost
        malt_cost_total += self.steady_state_factor * malt_cost
        open_cost_total += self.steady_state_factor * open_cost
        exp_cost_total += self.steady_state_factor * exp_cost

        transportation_total = beer_cost_total + malt_cost_total
        total_cost = transportation_total + open_cost_total + exp_cost_total

        print("\nOptimal Cost Breakdown (Discounted, in Million $):")
        print("{:<35} {:>15}".format("Cost Component", "Discounted Cost"))
        print("-" * 50)
        print("{:<35} {:>15.2f}".format("Transportation (Beer + Malt)", transportation_total))
        print("{:<35} {:>15.2f}".format("Facility Opening", open_cost_total))
        print("{:<35} {:>15.2f}".format("Capacity Expansion", exp_cost_total))
        print("{:<35} {:>15.2f}".format("Total Discounted Cost", total_cost))

        print("\nOptimal Distribution Plan (Beer Shipments X[d,b,t]):")
        header = f"{'Distribution Center':<20} {'Brewery':<25} {'Period':<10} {'Shipment':>10}"
        print(header)
        print("-" * len(header))
        for t in self.T:
            for d in range(len(self.distribution_centers)):
                for b in range(len(self.breweries)):
                    val = self.X[d, b, t].X
                    if abs(val) > tol:
                        dc = self.distribution_centers[d]
                        br = self.breweries[b]
                        print(f"{dc:<20} {br:<25} {t:<10} {val:>10.2f}")

    def sensitivity_analysis_demand(self, perturbation=0.05, tol=1e-6):
        """
        Performs a sensitivity analysis on beer demand by perturbing each distributor's annual demand
        by ±perturbation (default 5%). For each distributor, the model is rebuilt and resolved,
        and the change in the objective value (NPV) is reported relative to the base case.
        """
        original_demand = self.annual_demand.copy()
        base_obj = self.model.objVal if self.model.status == GRB.OPTIMAL else None
        if base_obj is None:
            print("Base model not solved optimally; cannot perform sensitivity analysis.")
            return

        results = []
        num_distr = self.annual_demand.shape[0]
        for d in range(num_distr):
            self.annual_demand[d, :] = original_demand[d, :] * (1 + perturbation)
            self.build_model()
            self.solve_model()
            increased_obj = self.model.objVal

            self.annual_demand[d, :] = original_demand[d, :] * (1 - perturbation)
            self.build_model()
            self.solve_model()
            decreased_obj = self.model.objVal

            self.annual_demand[d, :] = original_demand[d, :]
            results.append((self.distribution_centers[d],
                            increased_obj - base_obj,
                            base_obj - decreased_obj))
        print("\nSensitivity Analysis on Beer Demand (±5% perturbation):")
        print("{:<20} {:>20} {:>20}".format("Distributor", "Increase in Cost", "Decrease in Cost"))
        print("-" * 60)
        for dist, inc, dec in results:
            print("{:<20} {:>20.2f} {:>20.2f}".format(dist, inc, dec))
        print(f"\nBase Objective Value: {base_obj:.2f} Million $")


if __name__ == '__main__':
    # ------------------ Base Model (Model 1) ------------------
    print("=== Base Model (BeerOptimization) ===")
    base_optimizer = BeerOptimization()
    model1, X, malt_for_brew = base_optimizer.build_model_part1()
    base_optimizer.solve_model(model1)
    if model1.status == GRB.OPTIMAL:
        print("\nOptimal Solution for Base Model:")
        for var in model1.getVars():
            print(f"{var.VarName} = {var.X}")
        print(f"Objective Value: {model1.objVal:.2f} Million $")
    else:
        print("No optimal solution found for Base Model.")

    # ------------------ Improved Model (Model 2) ------------------
    print("\n=== Improved Model (MultiPeriodModel) ===")
    improved_optimizer = MultiPeriodModel()
    improved_optimizer.build_model()
    improved_optimizer.solve_model()
    improved_optimizer.print_solution()
    improved_optimizer.print_cost_breakdown()

    base_obj = model1.objVal if model1.status == GRB.OPTIMAL else None
    improved_obj = improved_optimizer.model.objVal if improved_optimizer.model.status == GRB.OPTIMAL else None
    if base_obj is not None and improved_obj is not None:
        print(f"\nDifference in Objective (Improved - Base): {(improved_obj - base_obj):.2f} Million $")
    else:
        print("\nUnable to compare objectives due to non-optimal solution in one of the models.")

    # ------------------ Sensitivity Analysis ------------------
    improved_optimizer.sensitivity_analysis_demand(perturbation=0.05)

    # ------------------ Experiment with Discount Rate = 1% ------------------
    print("\n=== Experiment with Discount Rate = 1% ===")
    low_discount_optimizer = MultiPeriodModel()
    low_discount_optimizer.discount_rate = 0.01
    low_discount_optimizer.steady_state_factor = sum([1 / ((1 + low_discount_optimizer.discount_rate) ** t) for t in range(3, 21)])
    low_discount_optimizer.build_model()
    low_discount_optimizer.solve_model()
    low_discount_optimizer.print_solution()
    low_discount_optimizer.print_cost_breakdown()
