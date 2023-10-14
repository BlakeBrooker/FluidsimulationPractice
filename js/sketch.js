import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 0.1  # Smoothing length
rho0 = 1.0  # Initial density
dt = 0.01  # Time step
mass = 1.0
viscosity = 0.1

# Particle data
num_particles = 100
positions = np.random.rand(num_particles, 2)
velocities = np.zeros((num_particles, 2))
densities = np.ones(num_particles) * rho0
forces = np.zeros((num_particles, 2))

def compute_density_and_forces():
    for i in range(num_particles):
        densities[i] = 0.0
        forces[i] = np.array([0.0, 0.0])
        for j in range(num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            q = r / h
            if q < 1:
                W = 315.0 / (64 * np.pi * h**9) * (h**2 - r**2)**3
                densities[i] += mass * W
                gradient = 945.0 / (32 * np.pi * h**9) * (h**2 - r**2)**2 * (r / h) * (positions[i] - positions[j])
                forces[i] += mass * gradient

def time_step():
    compute_density_and_forces()
    for i in range(num_particles):
        velocities[i] += dt * (forces[i] / densities[i] + viscosity * laplacian(i))
        positions[i] += dt * velocities[i]

def laplacian(i):
    lap = np.array([0.0, 0.0])
    for j in range(num_particles):
        r = np.linalg.norm(positions[i] - positions[j])
        q = r / h
        if q < 1:
            lap += mass / densities[j] * (velocities[j] - velocities[i]) * (45 / (np.pi * h**6)) * (h - r)**2
    return lap

# Main loop
num_steps = 100
for step in range(num_steps):
    time_step()

# Plot the final state
plt.scatter(positions[:, 0], positions[:, 1])
plt.title('2D Fluid Simulation')
plt.show()
