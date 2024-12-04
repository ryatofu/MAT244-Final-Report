import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
t_start = 0  # Start time in decades (2024)
t_end = 5  # End time in decades (arbitrary choice, adjust as needed)
dt = 0.001  # Time step (in decades)
steps = int((t_end - t_start) / dt)

# Initialize arrays
t = np.linspace(t_start, t_end, steps)
P = np.zeros(steps)
R = np.zeros(steps)

# Initial conditions
P[0] = 5  # Initial population in millions
R[0] = 2.5  # Initial rental price in $1000s

# Improved Euler Method
for i in range(steps - 1):
    # Compute slopes at the current point
    P_slope1 = P[i] * (10 - P[i]) - R[i]
    R_slope1 = 0.02 * R[i]

    # Predictor step (Euler step)
    P_predict = P[i] + dt * P_slope1
    R_predict = R[i] + dt * R_slope1

    # Compute slopes at the predicted point
    P_slope2 = P_predict * (10 - P_predict) - R_predict
    R_slope2 = 0.02 * R_predict

    # Corrector step (average the slopes)
    P[i + 1] = P[i] + dt * 0.5 * (P_slope1 + P_slope2)
    R[i + 1] = R[i] + dt * 0.5 * (R_slope1 + R_slope2)

# Compute eigenvalues of the linearized system
# Linearized system around P = 10, R = 0 (equilibrium point)
A = np.array([
    [10 - 2 * 10, -1],  # Partial derivatives w.r.t. P and R
    [0, 0.02]  # Partial derivatives w.r.t. P and R
])
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of the linearized system:", eigenvalues)

# Phase portrait
P_vals = np.linspace(0, 12, 20)  # Population range (P)
R_vals = np.linspace(2, 3, 20)   # Rental price range (R)
P_grid, R_grid = np.meshgrid(P_vals, R_vals)

# Calculate vector field
P_prime = P_grid * (10 - P_grid) - R_grid
R_prime = 0.02 * R_grid

# Normalize the vector field for better visualization
magnitude = np.sqrt(P_prime**2 + R_prime**2)
magnitude[magnitude == 0] = 1e-10
P_prime /= magnitude
R_prime /= magnitude



# Plot phase portrait
plt.figure(figsize=(8, 6))
plt.quiver(P_grid, R_grid, P_prime, R_prime, color="blue", alpha=0.7)
plt.plot(P, R, label="Trajectory (P vs R)", color="red")
plt.xlabel("Population, P(t) (millions)")
plt.ylabel("Rental Price, R(t) ($1000s)")
plt.title("Proposed: Phase Portrait with Trajectory")
plt.grid()
plt.legend()
plt.savefig("Proposed: Phase Portrait.png")
plt.show()

# Plot phase plot P(t) vs R(t)
plt.figure(figsize=(8, 6))
plt.plot(P, R, label="Phase Plot (P vs R)")
plt.xlabel("Population, P(t) (millions)")
plt.ylabel("Rental Price, R(t) ($1000s)")
plt.title("Proposed: Phase Plot of Population vs Rental Price")
plt.grid()
plt.legend()
plt.savefig("Proposed: Phase Plot.png")
plt.show()

# Plot P(t) vs t
plt.figure(figsize=(8, 6))
plt.plot(t, P, label="Population, P(t)")
plt.xlabel("Time (decades)")
plt.ylabel("Population, P(t) (millions)")
plt.title("Proposed: Population vs Time")
plt.grid()
plt.legend()
plt.savefig("Proposed: Population vs Time.png")
plt.show()

# Plot R(t) vs t
plt.figure(figsize=(8, 6))
plt.plot(t, R, label="Rental Price, R(t)")
plt.xlabel("Time (decades)")
plt.ylabel("Rental Price, R(t) ($1000s)")
plt.title("Proposed: Rental Price vs Time")
plt.grid()
plt.legend()
plt.savefig("Proposed: Rental Price vs Time.png")
plt.show()
