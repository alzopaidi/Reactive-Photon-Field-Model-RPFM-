import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

class TravelingPhotonVectors(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traveling Photon Vectors - E/B 90° Out of Phase")
        self.setGeometry(100, 100, 800, 700)

        # Layout
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        self.slider_label = QLabel("Animation Speed: 1.0")
        self.layout.addWidget(self.slider_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(20)
        self.layout.addWidget(self.speed_slider)
        self.speed_slider.valueChanged.connect(self.update_speed)

        self.speed = 1.0

        # Figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_box_aspect([1,1,1])
        self.ax.axis('off')

        # Sphere parameters
        self.R0 = 1.5
        self.A = 0.3
        self.n_ripples = 6
        self.n_lat = 12
        self.n_lon = 24

        # Meshgrid
        u = np.linspace(0, 2*np.pi, self.n_lon)
        v = np.linspace(0, np.pi, self.n_lat)
        self.u, self.v = np.meshgrid(u, v)

        # Sphere coordinates
        self.X = self.R0 * np.cos(self.v)
        self.Y = self.R0 * np.sin(self.u) * np.sin(self.v)
        self.Z = self.R0 * np.cos(self.u) * np.sin(self.v)

        # Tangent vectors (theta & phi directions)
        self.theta_hat = np.array([
            np.cos(self.v) * np.cos(self.u),
            np.cos(self.v) * np.sin(self.u),
            -np.sin(self.v)
        ])

        self.phi_hat = np.array([
            -np.sin(self.u),
            np.cos(self.u),
            np.zeros_like(self.u)
        ])

        # Initial quivers
        self.e_quiver = self.ax.quiver(
            self.X, self.Y, self.Z,
            self.theta_hat[0], self.theta_hat[1], self.theta_hat[2],
            length=self.A, color='r', alpha=0.8
        )

        self.b_quiver = self.ax.quiver(
            self.X, self.Y, self.Z,
            self.phi_hat[0], self.phi_hat[1], self.phi_hat[2],
            length=self.A, color='b', alpha=0.8
        )

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=np.linspace(0, 4*np.pi, 400),
            interval=30,
            blit=False
        )

    def update_speed(self, value):
        self.speed = value / 10.0
        self.slider_label.setText(f"Animation Speed: {self.speed:.2f}")

    def update(self, t):

        ripple_mod = np.sin(self.n_ripples * self.v - t * self.speed)

        # normalize tangent vectors (FIXED)
        thx, thy, thz = self.theta_hat
        phx, phy, phz = self.phi_hat

        th_norm = np.sqrt(thx**2 + thy**2 + thz**2)
        ph_norm = np.sqrt(phx**2 + phy**2 + phz**2)

        thx, thy, thz = thx/th_norm, thy/th_norm, thz/th_norm
        phx, phy, phz = phx/ph_norm, phy/ph_norm, phz/ph_norm

        # Electric field (theta direction)
        U_e = thx * ripple_mod
        V_e = thy * ripple_mod
        W_e = thz * ripple_mod

        # Magnetic field (π/2 phase shift)
        B_phase = np.sin(self.n_ripples * self.v - t * self.speed + np.pi/2)

        U_b = phx * B_phase
        V_b = phy * B_phase
        W_b = phz * B_phase

        # update visuals
        self.e_quiver.remove()
        self.b_quiver.remove()

        self.e_quiver = self.ax.quiver(
            self.X, self.Y, self.Z,
            U_e, V_e, W_e,
            length=self.A, color='r', alpha=0.8
        )

        self.b_quiver = self.ax.quiver(
            self.X, self.Y, self.Z,
            U_b, V_b, W_b,
            length=self.A, color='b', alpha=0.8
        )

        return []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TravelingPhotonVectors()
    window.show()
    sys.exit(app.exec_())
