import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Strong A₀ Visible Transition Model")

        self.epsilon0 = 8.85e-12

        # grid
        x = np.linspace(-5e-12, 5e-12, 250)
        y = np.linspace(-5e-12, 5e-12, 250)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)

        widget = QWidget()
        layout = QVBoxLayout()

        self.label = QLabel()
        layout.addWidget(self.label)

        # A0 slider (0 → 0.5)
        self.s_a0 = QSlider(Qt.Horizontal)
        self.s_a0.setRange(0, 50)
        layout.addWidget(QLabel("A₀ (0 → 0.5)"))
        layout.addWidget(self.s_a0)

        # lambda
        self.s_lam = QSlider(Qt.Horizontal)
        self.s_lam.setRange(10, 100)
        layout.addWidget(QLabel("λ"))
        layout.addWidget(self.s_lam)

        # R
        self.s_r = QSlider(Qt.Horizontal)
        self.s_r.setRange(1, 100)
        layout.addWidget(QLabel("R"))
        layout.addWidget(self.s_r)

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.t = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def update_plot(self):
        self.ax.clear()

        # parameters
        A0 = self.s_a0.value() / 100
        lam = self.s_lam.value() / 10 * 1e-12
        R0 = self.s_r.value() / 10 * 1e-12

        k = 2 * np.pi / lam

        # ---------------------------
        # 🔥 A0 NOW STRONGLY ACTIVE
        # ---------------------------

        # 1. radial compression (VERY IMPORTANT)
        R_eff = self.R / (1 + 5*A0)

        # 2. sharp wave transition
        sharpness = 1 + 8*A0
        wave = np.abs(np.cos(k * R_eff - self.t)) ** sharpness

        # 3. geometry scaling (still present)
        A_eff = np.pi * R_eff**2 * (1 + 3*A0)

        # 4. confinement
        envelope = np.exp(-(self.R / (R0 + 1e-18))**2)

        # full field
        C = (self.epsilon0 / lam) * A_eff * wave * envelope

        # IMPORTANT: boost contrast so A0 is visible
        C = np.log1p(np.abs(C)) * np.sign(C)

        self.ax.contourf(self.X, self.Y, C, levels=50, cmap='plasma')

        self.ax.set_title(f"A₀={A0:.2f} | λ={lam:.2e} | R={R0:.2e}")

        self.canvas.draw()

        self.t += 0.05


app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())
