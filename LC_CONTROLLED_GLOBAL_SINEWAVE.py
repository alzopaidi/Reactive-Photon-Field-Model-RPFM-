import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------- MAIN WINDOW ----------------
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LC Controlled Global Sine Wave")

        # ---------------- GRID ----------------
        x = np.linspace(0.5, 8, 140)
        y = np.linspace(0.5, 8, 140)
        self.X, self.Y = np.meshgrid(x, y)
        self.Rgrid = np.sqrt((self.X - 4)**2 + (self.Y - 4)**2) + 1e-9

        # ---------------- CONSTANTS ----------------
        self.mu0 = 1.0
        self.eps0 = 1.0

        # ---------------- UI ----------------
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("L_eff control"))
        self.sL = QSlider(Qt.Horizontal)
        self.sL.setRange(1, 200)
        self.sL.setValue(50)
        layout.addWidget(self.sL)

        layout.addWidget(QLabel("C_eff control"))
        self.sC = QSlider(Qt.Horizontal)
        self.sC.setRange(1, 200)
        self.sC.setValue(50)
        layout.addWidget(self.sC)

        # ---------------- PLOT ----------------
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111, projection='3d')

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # ---------------- TIMER ----------------
        self.t = 0

        self.timer = self.startTimer(40)

    # ---------------- FIELD MODEL ----------------
    def field(self, Leff, Ceff, lam, t):

        A = Leff + Ceff

        k = 2 * np.pi / lam
        omega = 1.5

        phase = k * self.Rgrid - omega * t

        return A * np.sin(phase)

    # ---------------- UPDATE ----------------
    def timerEvent(self, event):

        self.ax.clear()

        # fixed view
        self.ax.view_init(elev=30, azim=-60)
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 8)
        self.ax.set_zlim(-15, 15)

        self.ax.tick_params(colors='white')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # ---------------- SLIDERS ----------------
        L = self.sL.value() / 50.0
        C = self.sC.value() / 50.0

        # wavelength tied to LC
        lam = 1.2 + 0.3 * np.sin(self.t * 0.3)

        # LC effective values (your model)
        Leff = self.mu0 * L
        Ceff = self.eps0 * C

        Z = self.field(Leff, Ceff, lam, self.t)

        self.ax.plot_surface(
            self.X,
            self.Y,
            Z,
            cmap='RdBu_r',
            linewidth=0,
            antialiased=True,
            alpha=0.95
        )

        self.ax.set_title(
            f"L={L:.2f} | C={C:.2f} | λ={lam:.2f}",
            color='white'
        )

        self.canvas.draw()

        self.t += 0.05


# ---------------- RUN ----------------
app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())
