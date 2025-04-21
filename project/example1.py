from scipy.stats import poisson
import numpy as np

# Parameters (Christy Sports, Big Sky)
lam = 75               # daily arrivals
r = 60                 # rental cost per day ($)
B = 800                # purchase cost ($)
p = 1/6                # per-day stop probability
# Typical customers rent for about 6 days, so p = 1/6

# Section 3.1: POL1 (deterministic k)
k = B // r    # credited days before purchase
mu_pol1 = lam * (1 - (1 - p) ** k) / p    # ≈ 407.94
gamma_pol1 = lam * (1 - p) ** k           # ≈ 7.01

# Section 3.1: POL4 (average-case deterministic k')
k_prime = int(B**2 / (r**2 * p))
mu_pol4 = lam * (1 - (1-p)**k_prime) / p
gamma_pol4 = lam * (1-p)**k_prime

# Section 3.4: Optimal base-stock S* (POL1 scenario)
h = 6     # holding cost per period ($)
p_r = r  # penalty cost for unmet rentals ($)

alpha = p_r / (h + p_r)  # critical fractile
mu_for_S = mu_pol1       # use POL1 rental mean

# Determine S* using SciPy's Poisson CDF
S_star = 0
while poisson.cdf(S_star, mu_for_S) < alpha:
    S_star += 1

# Section 3.4: Optimal base-stock S* (POL4 scenario)
mu_for_S4 = mu_pol4
S_star4 = 0
while poisson.cdf(S_star4, mu_for_S4) < alpha:
    S_star4 += 1

# -------------------------------
# Section 4: Variable Usage Durations
# -------------------------------
# Section 4: POL-A (time-varying stop probabilities)
# Example stop probabilities p_j per day for j=1...6
p_js = np.array([0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
J = len(p_js)
q_js = 1 - p_js
S_js = np.cumprod(q_js)  # survival probabilities S(1)...S(J)
s = B / r

def delta(k):
    # Δ(k) = s/k * S(k) - sum_{j=k}^{s-1} S(j)/j
    term1 = (s / k) * S_js[k-1]
    term2 = sum(S_js[j] / (j+1) for j in range(k, int(s)))
    return term1 - term2

# Find k_star: first k where delta(k) >= 0
delta_vals = [delta(k) for k in range(1, min(J, int(s)) + 1)]
k_star = next((k for k, d in enumerate(delta_vals, start=1) if d >= 0), J)
mu_var = lam * np.sum(S_js[:k_star])       # expected rental mean
gamma_var = lam * S_js[k_star - 1]         # expected purchase mean

# Compute the base-stock S_star_var using the same critical fractile alpha
S_star_var = 0
while poisson.cdf(S_star_var, mu_var) < alpha:
    S_star_var += 1

# Display results
print("\n--- POL1 (Deterministic Policy) ---")
print(f"Credited rental days before purchase (k): {k}")
print(f"Expected rental demand (mu): {mu_pol1:.2f}  # Should be ≈ 407.94")
print(f"Expected purchase demand (gamma): {gamma_pol1:.2f}  # Should be ≈ 7.01")

print("\n--- Optimal Base-Stock Level (POL1) ---")
print(f"Critical fractile (alpha): {alpha:.4f}")
print(f"Using POL1 mean demand (mu): {mu_for_S:.2f}")
print(f"Computed optimal base-stock level S*: {S_star}")

print("\n--- POL4 (Average-Case Deterministic Policy) ---")
print(f"Average-case rental duration (k'): {k_prime}")
print(f"Expected rental demand (mu): {mu_pol4:.2f}")
print(f"Expected purchase demand (gamma): {gamma_pol4:.2f}")

print("\n--- Optimal Base-Stock Level (POL4) ---")
print(f"Using POL4 mean demand (mu): {mu_for_S4:.2f}")
print(f"Computed optimal base-stock level S* (POL4): {S_star4}")

print("\n--- POL-A (Time-Varying Stop Probability) ---")
print(f"Stop probabilities p_j: {p_js.tolist()}")
print(f"Optimal customer horizon (k*): {k_star}")
print(f"Expected rental demand (mu): {mu_var:.2f}")
print(f"Expected purchase demand (gamma): {gamma_var:.2f}")

print("\n--- Optimal Base-Stock Level (POL-A, Time-Varying) ---")
print(f"Using POL-A mean demand (mu): {mu_var:.2f}")
print(f"Computed optimal base-stock level S* (POL-A): {S_star_var}")
