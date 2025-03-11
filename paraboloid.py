import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters
n_points = 100  # Number of points
radius = 5       # Radius of the base
height = 10      # Height of the paraboloid

# Generate the paraboloid
u = np.linspace(0, 2 * np.pi, n_points)  # Angle for circular base
v = np.linspace(0, radius, n_points)    # Radius of each cross-section
U, V = np.meshgrid(u, v)

# Paraboloid equations
X = V * np.cos(U)
Y = V * np.sin(U)
Z = (V**2) * (height / radius**2)  # Parabolic relationship: z = (r^2) * (h/radius^2)

# Plot the paraboloid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Customize the plot
ax.set_title("3D Paraboloid", fontsize=14)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.view_init(elev=30, azim=120)  # Adjust viewing angle

plt.show()
