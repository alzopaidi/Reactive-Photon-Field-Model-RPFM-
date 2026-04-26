import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ---------------- STYLE ----------------
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9), facecolor='black')
ax = fig.add_subplot(111, projection='3d')

# ---------------- GRID ----------------
x = np.linspace(0.5, 8, 140)
y = np.linspace(0.5, 8, 140)
X, Y = np.meshgrid(x, y)

Rgrid = np.sqrt((X - 4)**2 + (Y - 4)**2) + 1e-9

# ---------------- CONSTANTS ----------------
h = 1.0       # scaled Planck (numerical model)
Z0 = 377.0    # free space impedance

# ---------------- FIXED VIEW ----------------
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(-6, 18)

ax.set_title("A₀ Driven Curvature–Flow Field", color='white', fontsize=15)
ax.tick_params(colors='white')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ---------------- MODEL ----------------
def model(A0, Rgrid, t):

    # ---------------- YOUR EQUATION ----------------
    lam = np.sqrt((h * Z0) / (4 * np.pi * (A0**2 + 1e-9)))

    R = lam / (2 * np.pi)

    Aeff = (lam**2) / np.pi

    K0 = np.pi / (lam**2)

    # ---------------- SPATIAL STRUCTURE ----------------
    spatial = (
        1
        + 0.6 * np.sin(2.0 * Rgrid - t)
        + 0.3 * np.cos(1.5 * (X + Y))
    )

    envelope = np.exp(-(Rgrid**2) / (R**2 + 1e-12))

    # curvature field
    K = K0 * spatial * envelope

    return K, lam


# ---------------- MASS FLOW ----------------
def mass_flow(K):

    dKx = np.gradient(K, axis=0)
    dKy = np.gradient(K, axis=1)

    return np.sqrt(dKx**2 + dKy**2)


# ---------------- ANIMATION ----------------
def update(frame):
    ax.cla()

    ax.view_init(elev=30, azim=-60)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(-6, 18)

    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    t = frame * 0.05

    # ---------------- A0 evolves slowly ----------------
    A0 = 0.5 + 0.4 * np.sin(t * 0.3)

    # compute model
    K, lam = model(A0, Rgrid, t)

    # mass flow
    M = mass_flow(K)

    # normalize
    K = K / (np.max(np.abs(K)) + 1e-9)
    M = M / (np.max(M) + 1e-9)

    # combined field
    Z = K * 8 + M * 4

    ax.plot_surface(
        X, Y, Z,
        cmap='RdBu_r',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=True
    )

    ax.set_title(
        f"A₀={A0:.3f} | λ={lam:.3f}",
        color='white'
    )

    return []


ani = animation.FuncAnimation(
    fig,
    update,
    interval=40,
    blit=False
)

plt.tight_layout()
plt.show()
