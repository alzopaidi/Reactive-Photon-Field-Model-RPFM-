import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# ================= STYLE =================
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

plt.subplots_adjust(bottom=0.12)

# ================= SLIDER =================
slider_ax = plt.axes([0.2, 0.04, 0.6, 0.03], facecolor='gray')
slider = Slider(slider_ax, 'λ', 0.5, 3.0, valinit=1.5)

# ================= GRID =================
x = np.linspace(0.5, 8, 220)
y = np.linspace(0.5, 8, 220)
X, Y = np.meshgrid(x, y)

Rgrid = np.sqrt((X - 4)**2 + (Y - 4)**2) + 1e-9

# ================= FIXED VIEW =================
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(0, 20)   # fixed bounded space

ax.set_title("λ-Driven Curvature Field (Bounded Amplitude)", color='white')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.tick_params(colors='white')
ax.set_box_aspect([1, 1, 1.5])

# ================= MODEL =================
def curvature_model(lam, t):

    curvature_gain = 6.0
    K0 = curvature_gain * np.pi / (lam**2)

    width_factor = 2.5
    R0 = width_factor * lam / (2 * np.pi)

    spatial = (
        1
        + 0.6 * np.sin(2.0 * Rgrid - t)
        + 0.3 * np.cos(1.5 * (X + Y))
    )

    envelope = np.exp(-0.7 * (Rgrid**2) / (R0**2 + 1e-12))

    K = K0 * spatial * envelope

    return K

# ================= MASS FLOW =================
def mass_flow(K):
    dKx = np.gradient(K, axis=0)
    dKy = np.gradient(K, axis=1)
    return np.sqrt(dKx**2 + dKy**2)

# ================= UPDATE =================
def update(frame):

    ax.cla()

    ax.view_init(elev=30, azim=-60)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 20)

    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_box_aspect([1, 1, 1.5])

    t = frame * 0.05
    lam = slider.val

    # curvature field
    K = curvature_model(lam, t)

    # gradient energy
    M = mass_flow(K)

    # normalize
    K = K / (np.max(np.abs(K)) + 1e-9)
    M = M / (np.max(M) + 1e-9)

    # ================= BOUNDED SURFACE =================
    Z = K * 20 + M * 10

    # shift to zero baseline
    Z = Z - np.min(Z)

    # normalize into fixed range
    Z = Z / (np.max(Z) + 1e-9) * 20

    # safety clamp
    Z = np.clip(Z, 0, 20)

    ax.plot_surface(
        X, Y, Z,
        cmap='RdBu_r',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=True
    )

    ax.set_title(
        f"λ={lam:.2f} | Bounded Curvature Field (0–20)",
        color='white'
    )

    return []

def on_change(val):
    fig.canvas.draw_idle()

slider.on_changed(on_change)

# ================= ANIMATION =================
ani = animation.FuncAnimation(
    fig,
    update,
    interval=40,
    blit=False
)

plt.show()
