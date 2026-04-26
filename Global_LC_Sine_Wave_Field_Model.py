import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# ---------------- GRID ----------------
x = np.linspace(0.5, 8, 140)
y = np.linspace(0.5, 8, 140)
X, Y = np.meshgrid(x, y)

Rgrid = np.sqrt((X - 4)**2 + (Y - 4)**2) + 1e-9

# ---------------- CONSTANTS ----------------
mu0 = 1.0
eps0 = 1.0

# ---------------- FIXED VIEW ----------------
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(-8, 8)

ax.set_title("Global Sine-Wave Space Motion (LC Driven)", color='white')
ax.tick_params(colors='white')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ---------------- MODEL ----------------
def field(lam, Rgrid, t):

    # LC effective terms
    Leff = mu0 * lam / 4
    Ceff = eps0 * lam / (4 * np.pi)

    A = Leff + Ceff  # amplitude

    # wave numbers
    k = 2 * np.pi / lam
    omega = 1.5

    # global sine wave (key change)
    phase = k * Rgrid - omega * t

    F = A * np.sin(phase)

    return F


# ---------------- UPDATE ----------------
def update(frame):

    ax.cla()

    ax.view_init(elev=30, azim=-60)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(-8, 8)

    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    t = frame * 0.05

    # wavelength breathing
    lam = 1.2 + 0.6 * np.sin(t * 0.4)

    Z = field(lam, Rgrid, t)

    ax.plot_surface(
        X, Y, Z,
        cmap='RdBu_r',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=True
    )

    ax.set_title(f"Global Sine Motion | λ={lam:.3f}", color='white')

    return []


ani = animation.FuncAnimation(fig, update, interval=40, blit=False)

plt.tight_layout()
plt.show()
