import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("λ-Driven Curvature Field Model")

        # ---------------- GRID ----------------
        x = np.linspace(-5e-12, 5e-12, 250)
        y = np.linspace(-5e-12, 5e-12, 250)
        self.X, self.Y = np.meshgrid(x, y)
        self.Rgrid = np.sqrt(self.X**2 + self.Y**2)

        # ---------------- FIELD ----------------
        self.F = np.zeros_like(self.Rgrid)
        self.F_prev = np.zeros_like(self.Rgrid)

        self.gamma = 0.03
        self.t = 0

        # ---------------- UI ----------------
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("λ (master parameter)"))
        self.s_lam = QSlider(Qt.Horizontal)
        self.s_lam.setRange(10, 200)
        layout.addWidget(self.s_lam)

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # ---------------- TIMER ----------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

    def update_plot(self):
        self.ax.clear()

        # ---------------- λ ----------------
        lam = self.s_lam.value() / 10.0

        # ---------------- DERIVED MODEL ----------------
        R = lam / (2 * np.pi)
        Aeff = (lam**2) / (4 * np.pi)
        K = (4 * np.pi) / (lam**2)

        # curvature scaling (your physics)
        curvature_strength = K

        # ---------------- SOURCE ----------------
        cx, cy = self.Rgrid.shape[0] // 2, self.Rgrid.shape[1] // 2

        source = np.zeros_like(self.Rgrid)
        source[cx, cy] = np.sin(self.t) * curvature_strength

        # ---------------- LAPLACIAN ----------------
        lap = (
            np.roll(self.F, 1, axis=0) +
            np.roll(self.F, -1, axis=0) +
            np.roll(self.F, 1, axis=1) +
            np.roll(self.F, -1, axis=1) -
            4 * self.F
        )

        # ---------------- WAVE UPDATE ----------------
        v = 0.4  # fixed propagation speed (stability)

        F_next = (
            2 * self.F - self.F_prev
            + v**2 * lap
            - self.gamma * self.F
            + source
        )

        self.F_prev = self.F
        self.F = F_next

        # stability clamp
        self.F = np.clip(self.F, -5, 5)

        # ---------------- VISUAL ----------------
        F_vis = self.F * curvature_strength

        F_vis = F_vis / (np.max(np.abs(F_vis)) + 1e-9)

        self.ax.contourf(
            self.X,
            self.Y,
            F_vis,
            levels=70,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1
        )

        self.ax.set_title(f"λ={lam:.2f} | R={R:.2e} | K={K:.2e}")

        self.canvas.draw()

        self.t += 0.1


app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())
