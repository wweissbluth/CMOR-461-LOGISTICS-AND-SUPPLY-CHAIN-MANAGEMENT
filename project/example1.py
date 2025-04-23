from scipy.stats import poisson
import numpy as np

# Parameters (Christy Sports, Big Sky)
lam = 75               # daily arrivals
r = 60                 # rental cost per day ($)
B = 800                # purchase cost ($)
p = 1/6                # per-day stop probability
# Typical customers rent for about 6 days, so p = 1/6
h = 6     # holding cost per period ($)
p_r = r  # penalty cost for unmet rentals ($)

# -------------------------------
## Deterministic Policies
# POL1
k = B // r    
mu_pol1 = lam * (1 - (1 - p) ** k) / p    
gamma_pol1 = lam * (1 - p) ** k           

# POL4
k_prime = int(B**2 / (r**2 * p))
mu_pol4 = lam * (1 - (1-p)**k_prime) / p
gamma_pol4 = lam * (1-p)**k_prime
# --------------------------------

# -------------------------------
# POL3 - Rent indefinitely
mu_pol3 = lam / p
gamma_pol3 = 0 
alpha = r / (h + r)  # critical fractile
S_star = 0
while poisson.cdf(S_star, mu_pol3) < alpha:
    S_star += 1
# --------------------------------

# Display results
print(f"POL1: mu = {mu_pol1:.2f}, gamma = {gamma_pol1:.2f}")
print(f"POL3: mu = {mu_pol3:.2f}, gamma = {gamma_pol3:.2f}")
print(f"POL4: mu = {mu_pol4:.2f}, gamma = {gamma_pol4:.2f}")
print(f"POL1: k = {k}, S* = {S_star}")
print(f"POL4: k' = {k_prime}")
print(f"POL3: S* = {S_star}")