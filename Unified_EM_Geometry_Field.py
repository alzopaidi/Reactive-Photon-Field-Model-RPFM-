import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Unified EM Geometry Field (Stable Propagation)")

        # ---------------- GRID ----------------
        x = np.linspace(-5e-12, 5e-12, 250)
        y = np.linspace(-5e-12, 5e-12, 250)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)

        # ---------------- FIELD STATE ----------------
        self.F = np.zeros_like(self.R)
        self.F_prev = np.zeros_like(self.R)

        # ---------------- BASE PARAMETERS ----------------
        self.gamma = 0.03  # damping (stability)

        # ---------------- UI ----------------
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("A₀ (Nonlinearity Strength)"))
        self.s_a0 = QSlider(Qt.Horizontal)
        self.s_a0.setRange(0, 100)
        layout.addWidget(self.s_a0)

        layout.addWidget(QLabel("λ (Wave Control)"))
        self.s_lam = QSlider(Qt.Horizontal)
        self.s_lam.setRange(1, 100)
        layout.addWidget(self.s_lam)

        layout.addWidget(QLabel("R (Confinement)"))
        self.s_r = QSlider(Qt.Horizontal)
        self.s_r.setRange(1, 100)
        layout.addWidget(self.s_r)

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.t = 0

        # ---------------- TIMER ----------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

    # ===================== MAIN UPDATE =====================
    def update_plot(self):
        self.ax.clear()

        # -------- sliders --------
        A0 = self.s_a0.value() / 100.0
        lam = self.s_lam.value() / 10.0
        R0 = self.s_r.value() / 10 * 1e-12

        # -------- stable wave speed (IMPORTANT FIX) --------
        self.v = 0.2 + 0.6 / (lam + 1.0)
        self.v = min(self.v, 0.7)  # CFL safety cap

        # -------- geometry factor --------
        geom_factor = (1 + 5 * A0)

        # -------- source (center emitter) --------
        source = np.zeros_like(self.R)
        cx, cy = self.R.shape[0] // 2, self.R.shape[1] // 2
        source[cx, cy] = np.sin(self.t) * (1 + 8 * A0)

        # -------- Laplacian --------
        lap = (
            np.roll(self.F, 1, axis=0) +
            np.roll(self.F, -1, axis=0) +
            np.roll(self.F, 1, axis=1) +
            np.roll(self.F, -1, axis=1) -
            4 * self.F
        )

        # -------- NONLINEARITY (soft + stable) --------
        nonlinear = A0 * 0.1 * self.F**3

        # -------- WAVE EQUATION --------
        F_next = (
            2 * self.F - self.F_prev
            + self.v**2 * lap
            - self.gamma * self.F
            - nonlinear
            + source
        )

        # -------- update states --------
        self.F_prev = self.F
        self.F = F_next

        # -------- HARD STABILITY CLAMP --------
        self.F = np.clip(self.F, -8, 8)

        # -------- envelope (geometry confinement) --------
        envelope = np.exp(-(self.R / (R0 + 1e-18))**2)

        # -------- final field --------
        F_vis = self.F * envelope * geom_factor

        # -------- normalize (no log distortion) --------
        F_vis = F_vis / (np.max(np.abs(F_vis)) + 1e-9)

        # -------- plot --------
        self.ax.contourf(self.X, self.Y, F_vis, levels=60, cmap='plasma')
        self.ax.set_title(f"A₀={A0:.2f} | λ={lam:.2f} | v={self.v:.2f}")

        self.canvas.draw()

        self.t += 0.1


# ===================== RUN =====================
app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())
