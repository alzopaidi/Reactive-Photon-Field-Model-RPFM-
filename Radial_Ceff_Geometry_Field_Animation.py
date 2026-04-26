import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epsilon0 = 8.85e-12
lam = 2.4e-12
k = 2*np.pi/lam
omega = 1e15

x = np.linspace(-5e-12, 5e-12, 300)
y = np.linspace(-5e-12, 5e-12, 300)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)

# CHOOSE ONE:
use_sphere = False   # True = 4πR², False = πR²

def area(R):
    if use_sphere:
        return 4*np.pi*R**2
    else:
        return np.pi*R**2

fig, ax = plt.subplots()

def update(t):
    ax.clear()

    A_geom = area(R)

    # wave modulation on geometry (your model idea)
    field = A_geom * np.cos(k*R - omega*t)**2

    C_eff = (epsilon0 / lam) * field

    im = ax.contourf(X, Y, C_eff, levels=50, cmap='plasma')
    ax.set_title("Radial Distributed C_eff Field")

ani = FuncAnimation(fig, update, frames=120, interval=50)
plt.show()
