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

# center-based radial structure
Rgrid = np.sqrt((X - 4)**2 + (Y - 4)**2) + 1e-9

# ---------------- FIXED CAMERA ----------------
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(-6, 18)

ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Curvature Field', color='cyan')

ax.set_title(
    'Spatial Curvature Field (λ-driven, non-uniform K)',
    color='white', fontsize=15, pad=25
)

ax.tick_params(colors='white')
ax.grid(False)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ---------------- MODEL ----------------
def curvature_field(lam, Rgrid, t):

    # base spherical mapping
    R = lam / (2 * np.pi)

    # global curvature scale (your theory)
    K0 = np.pi / (lam**2)

    # ---------------- SPATIAL MODULATION ----------------
    # this is the KEY upgrade:
    spatial_variation = (
        1
        + 0.6 * np.sin(2.5 * Rgrid - t)
        + 0.4 * np.cos(1.8 * (X + Y) + t)
    )

    # radial decay (spherical geometry)
    envelope = np.exp(-(Rgrid**2) / (R**2 + 1e-12))

    # FINAL CURVATURE FIELD (no longer constant)
    K_field = K0 * spatial_variation * envelope

    # wave-like temporal evolution
    K_field *= (1 + 0.3 * np.sin(t - Rgrid * 2.5))

    return K_field


# ---------------- ANIMATION ----------------
def update(frame):
    ax.cla()

    # fixed camera again
    ax.view_init(elev=30, azim=-60)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(-6, 18)

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Curvature Field', color='cyan')

    ax.set_title(
        'Spatial Curvature Field (λ-driven, non-uniform K)',
        color='white', fontsize=15, pad=25
    )

    ax.tick_params(colors='white')
    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # time evolution of λ
    t = frame * 0.05
    lam = 1.5 + 1.0 * np.sin(t * 0.4)

    Z = curvature_field(lam, Rgrid, t)

    # normalize for visibility (NOT physics change)
    Z = Z / (np.max(np.abs(Z)) + 1e-9)

    ax.plot_surface(
        X, Y, Z * 10,   # visual amplification only
        cmap='RdBu_r',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=True
    )

    return []


# ---------------- RUN ----------------
ani = animation.FuncAnimation(
    fig,
    update,
    interval=40,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
