# %% [markdown]
# We're operating in terms of 1,000 ton and 1,000,000 liter units

# %%
import numpy as np

# Cost of shipping malt from plant to brewery (Million $ / 1000 tons)
malt_shipping_costs = np.array([
    [0.026, 0.017, 0.020, 0.019, 0.032],  # Afyon
    [0.037, 0.017, 0.031, 0.030, 0.022],  # Konya
    [0.032, 0.033, 0.004, 0.028, 0.048]   # Import (Izmir)
])

# Malt shipping index mappings
malt_plants = ["Afyon", "Konya", "Import (Izmir)"]
brewery_sites = ["Istanbul", "Ankara", "Izmir", "Sakarya", "Adana"]

# Cost of shipping beer from brewery to distribution centers (Million $ / Million liters)
beer_shipping_costs = np.array([
    [0.000, 0.040, 0.052, 0.017, 0.055, 0.042],  # Istanbul (Existing)
    [0.032, 0.041, 0.039, 0.027, 0.023, 0.043],  # Ankara (Existing)
    [0.040, 0.000, 0.032, 0.023, 0.062, 0.002],  # Izmir (Potential)
    [0.011, 0.034, 0.041, 0.011, 0.045, 0.036],  # Sakarya (Potential)
    [0.067, 0.064, 0.040, 0.060, 0.024, 0.066]   # Adana (Potential)
])

# Beer shipping index mappings
breweries = ["Istanbul (Existing)", "Ankara (Existing)", "Izmir (Potential)", "Sakarya (Potential)", "Adana (Potential)"]
distribution_centers = ["Istanbul", "Izmir", "Antalya", "Bursa", "Kayseri", "Export (Izmir)"]

# Print the arrays
print("Malt Shipping Costs (Million $ / 1000 tons):")
print(malt_shipping_costs.tolist())  # Convert to list for readability

print("\nBeer Shipping Costs (Million $ / Million liters):")
print(beer_shipping_costs.tolist())  # Convert to list for readability


# %%
annual_demand = np.array([
    [103, 110, 125], # Istanbul
    [74, 80, 90], # Izmir
    [50, 53, 60], # Antalya
    [60, 75, 85], # Bursa
    [102, 110, 125], # Kayseri
    [13, 13, 15] # Izmir Exporting
])

years = [str(y) for y in (1, 2 ,3)]

domestic_yield = 8.333
import_yield = 9.091

# Cost of potential brewery sites (Million $)
fixed_costs = np.array([
    # [0.000, 0.00],  # Istanbul (Existing)
    # [0.000, 0.00],  # Ankara (Existing)
    [75, 30],  # Izmir (Potential)
    [70, 27],  # Sakarya (Potential)
    [68, 25]   # Adana (Potential)
])

building = ["open", "expand"]

# %%
from gurobipy import Model, GRB, quicksum

# Define model
part1_1_dist = Model("Dist Malt Beer")

# Define dimensions
rows, cols = 6, 2

X = part1_1_dist.addVars(rows, cols, vtype=GRB.CONTINUOUS, name="X")

# %%
malt_for_brew = part1_1_dist.addVars(2, 3, vtype=GRB.CONTINUOUS, name="Mb")
malt_caps = np.array((30, 68, 20))

part1_1_dist.addConstrs((malt_for_brew[0, i] + malt_for_brew[1, i] <= malt_caps[i] \
                         for i in range(3)), "MaBe")


# %%
existing_beer = beer_shipping_costs[:2]
existing_malt = np.array([maps[:2] \
                          for maps in malt_shipping_costs])

lin_expr_incremental = 0
lin_expr_incremental += quicksum(X[i, j] * existing_beer[j, i] for i in range(6) for j in range(2))
lin_expr_incremental += quicksum(malt_for_brew[i, j] * existing_malt[j, i] for i in range(2) for j in range(3))
part1_1_dist.setObjective(lin_expr_incremental, GRB.MINIMIZE)

part1_1_dist.update()

# %%
column_sums = {j: quicksum(X[i, j] for i in range(rows)) for j in range(cols)}
part1_1_dist.addConstr(column_sums[0] <= 220 \
                         , "IstanCap")
part1_1_dist.addConstr(column_sums[1] <= 200 \
                         , "AnkCap")

# %%
y1_demands = annual_demand.T[0]
row_sums = {i: quicksum(X[i, j] for j in range(cols)) for i in range(rows)}

part1_1_dist.addConstrs((X[i, 0] + X[i, 1] >= y1_demands[i] for i in range(rows)), "Satisfy")

# %%

yields = (domestic_yield, domestic_yield, import_yield)
part1_1_dist.addConstrs((column_sums[j] <= quicksum(malt_for_brew[j, k] * yields[j] for k in range(3)) for j in range(2)), "Conserve")

# %%
part1_1_dist.update()

#  Solve the model
part1_1_dist.optimize()

# Report results
if part1_1_dist.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    for var in part1_1_dist.getVars():
        print(f"{var.varName} = {var.x}")  # Print variable values
    print(f"Objective Value: {part1_1_dist.objVal}")  # Print optimal objective value
else:
    print("No optimal solution found.")


# %%
# Extract values from solved model
X_values = np.array([[X[i, j].X for j in range(2)] for i in range(6)]).T
Mb_values = np.array([[malt_for_brew[i, j].X for j in range(3)] for i in range(2)]).T

# ---- Print Beer Shipment Table ----
print("\nAmount of beer shipped from brewery to distribution center in year 1 (Million liters)")
print(f"{'':<12} " + " ".join(f"{dc:<8}" for dc in distribution_centers) + "Total   Capacity")
for i, brewery in enumerate(breweries[:2]):
    total = np.sum(X_values[i, :])
    capacity = 220 if i == 0 else 200  # Example capacities
    print(f"{brewery:<12} " + " ".join(f"{X_values[i, j]:<8.0f}" for j in range(len(distribution_centers))) + f"{total:<7.0f} {capacity}")

# ---- Print Malt Shipment Table ----
print("\nAmount of malt shipped from plant to brewery in year 1 (1000 tons)")
print(f"{'':<10} " + " ".join(f"{brewery:<8}" for brewery in breweries[:2]) + "Total   Capacity")
for i, malt in enumerate(malt_plants):
    total = np.sum(Mb_values[i, :])
    capacity = [30, 68, 20][i]  # Example capacities from exhibit
    print(f"{malt:<10} " + " ".join(f"{Mb_values[i, j]:<8.2f}" for j in range(len(breweries[:2]))) + f"{total:<7.2f} {capacity}")

# %%
# ---- Print Beer Shipment Table ----
print("\nAmount of beer shipped from brewery to distribution center in year 1 (Million liters)")
header = f"{'':<12}" + "".join(f"{dc[:8]:<10}" for dc in distribution_centers) + "Total    Capacity"
print(header)
print("-" * len(header))

for i, brewery in enumerate(breweries[:2]):
    total = np.sum(X_values[i, :])
    capacity = 220 if i == 0 else 200  # Example capacities
    row = f"{brewery[:8]:<12}" + "".join(f"{X_values[i, j]:<10.0f}" for j in range(len(distribution_centers)))
    row += f"{total:<8.0f} {capacity}"
    print(row)

# ---- Print Malt Shipment Table ----
print("\nAmount of malt shipped from plant to brewery in year 1 (1000 tons)")
header = f"{'':<12}" + "".join(f"{brewery[:8]:<10}" for brewery in breweries[:2]) + "Total    Capacity"
print(header)
print("-" * len(header))

for i, malt in enumerate(malt_plants):
    total = np.sum(Mb_values[i, :])
    capacity = [30, 68, 20][i]  # Example capacities from exhibit
    row = f"{malt[:8]:<12}" + "".join(f"{Mb_values[i, j]:<10.2f}" for j in range(len(breweries[:2])))
    row += f"{total:<8.2f} {capacity}"
    print(row)

# %%
# Extract and print shadow prices (dual values)
if part1_1_dist.status == GRB.OPTIMAL:
    print("\nShadow Prices (Dual Values):")
    for constr in part1_1_dist.getConstrs():
        print(f"{constr.constrName}: {constr.Pi}")
else:
    print("No optimal solution found.")

# %% [markdown]
# ### To get the comparsion with original production planning, I will fix variables in our model to their values

# %%
part1_3 = part1_1_dist.copy()

# Copy variables
var_map =  part1_3.getVars()
print(list(enumerate(var_map)))

part1_3.addConstr(var_map[4] >= 50, "Antalya") # "X[2,0]"
part1_3.addConstr(var_map[7] >= 60, "Bursa") # "X[3,1]"
part1_3.addConstr(var_map[3] <= 25, "Izmir") # "X[1,1]"

part1_3.addConstr(var_map[13] >= 2.42, "Konya") # "Mb[0,1]"
part1_3.addConstr(var_map[14] >= 20, "Import") # "Mb[0,2]"
part1_3.addConstr(var_map[15] >= 24, "Afyon") # "Mb[1,0]"

part1_3.addConstr(var_map[12] == 0, "Istahp")
part1_3.addConstr(var_map[16] == 0, "Ankant")

# %%
part1_3.update()
part1_3.optimize()


print("\nSolution Found:")
for var in part1_3.getVars():
    print(f"{var.varName} = {var.x}")  # Print variable values
print(f"Objective Value: {part1_3.objVal}")

# %%
part1_3.objVal - part1_1_dist.objVal

# %% [markdown]
# From here on there's only part2 features.

# %%
# Since capacities are dynamic, not yet defined as matrix.
# Baking them into constraints to start.

p2_build = Model("Exp Beer Empire")

new = p2_build.addVars(3, vtype=GRB.BINARY, name="Ne")
expan = p2_build.addVars(3, vtype=GRB.BINARY, name="E")

real_expand = p2_build.addConstrs((new[i] >= expan[i] \
                                      for i in range(3)), "ReEx")

# %%
# Looking to minimize costs while satisfying demand in
# Part II: Capacity Expansion Decisions

p2_build.setObjective(quicksum(new[b] * fixed_costs.T[0][b] \
                           + expan[b] * fixed_costs.T[1][b] for b in range(3)))


