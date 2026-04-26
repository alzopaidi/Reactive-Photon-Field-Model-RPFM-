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
ax.set_zlim(-5, 20)

ax.set_title("Spherical LC Photon Propagation Model", color='white')
ax.tick_params(colors='white')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ---------------- MODEL ----------------
def lc_shell(lam, Rgrid, t):

    # your definitions
    Leff = mu0 * lam / 4
    Ceff = eps0 * lam / (4 * np.pi)

    # photon-like wave scale
    R = lam / (2 * np.pi)

    # energy shell structure
    envelope = np.exp(-(Rgrid**2) / (R**2 + 1e-12))

    # oscillation (propagation)
    wave = np.sin(t - Rgrid * 2.5)

    # combined LC energy density
    field = (Leff + Ceff) * envelope * wave

    return field, Leff, Ceff


# ---------------- MASS FLOW ----------------
def flow(F):

    dFx = np.gradient(F, axis=0)
    dFy = np.gradient(F, axis=1)

    return np.sqrt(dFx**2 + dFy**2)


# ---------------- ANIMATION ----------------
def update(frame):
    ax.cla()

    ax.view_init(elev=30, azim=-60)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(-5, 20)

    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    t = frame * 0.05

    # wavelength evolves (photon breathing)
    lam = 1.0 + 0.8 * np.sin(t * 0.4)

    F, L, C = lc_shell(lam, Rgrid, t)

    M = flow(F)

    # normalize
    F = F / (np.max(np.abs(F)) + 1e-9)
    M = M / (np.max(M) + 1e-9)

    Z = F * 8 + M * 4

    ax.plot_surface(
        X, Y, Z,
        cmap='RdBu_r',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=True
    )

    ax.set_title(
        f"λ={lam:.3f} | L={L:.3f} | C={C:.3f}",
        color='white'
    )

    return []


ani = animation.FuncAnimation(fig, update, interval=40, blit=False)

plt.tight_layout()
plt.show()
