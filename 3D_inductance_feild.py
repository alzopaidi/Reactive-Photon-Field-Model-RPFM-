import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D Spherical Inductance Field")

        self.mu0 = 4*np.pi*1e-7

        # spherical grid
        self.theta = np.linspace(0, np.pi, 60)
        self.phi = np.linspace(0, 2*np.pi, 60)
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)

        # UI
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("A₀"))
        self.s_a0 = QSlider(Qt.Horizontal)
        self.s_a0.setRange(0, 100)
        layout.addWidget(self.s_a0)

        layout.addWidget(QLabel("λ"))
        self.s_lam = QSlider(Qt.Horizontal)
        self.s_lam.setRange(10, 100)
        layout.addWidget(self.s_lam)

        layout.addWidget(QLabel("R scale"))
        self.s_r = QSlider(Qt.Horizontal)
        self.s_r.setRange(1, 100)
        layout.addWidget(self.s_r)

        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111, projection='3d')

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.t = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(80)

    def update_plot(self):
        self.ax.clear()

        A0 = min(self.s_a0.value() / 100, 0.7)
        lam = (self.s_lam.value() / 10) * 1e-9
        R0 = (self.s_r.value() / 10) * 1e-9

        k = 2*np.pi / (lam + 1e-12)

        R = R0 * (1 + 0.6 * A0 * np.cos(3 * self.THETA))

        wave = np.abs(np.cos(k * R - self.t))
        wave = wave ** (1 + 2 * A0)

        X = R * np.sin(self.THETA) * np.cos(self.PHI)
        Y = R * np.sin(self.THETA) * np.sin(self.PHI)
        Z = R * np.cos(self.THETA)

        L = (self.mu0 / (lam + 1e-12)) * (4*np.pi*R**2) * wave

        # ================= KEEP ORIGINAL COLORS EXACTLY =================
        self.ax.plot_surface(
            X, Y, Z,
            facecolors=plt.cm.viridis(L / np.max(L)),
            linewidth=0,
            shade=False
        )

        self.ax.set_title(f"3D Inductance Field | A₀={A0:.2f}")

        self.canvas.draw()
        self.t += 0.05


app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())
