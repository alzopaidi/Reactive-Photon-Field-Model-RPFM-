import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

class PhotonSphereRipplesHorizontalPoles(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reactive Photon Ripples with Speed Control - Horizontal Poles")
        self.setGeometry(100, 100, 800, 700)

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

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_box_aspect([1,1,1])
        self.ax.axis('off')

        self.R0 = 1.5
        self.n_lat = 20
        self.n_lon = 40
        self.A = 0.15
        self.n_ripples = 6

        u = np.linspace(0, 2*np.pi, self.n_lon)
        v = np.linspace(0, np.pi, self.n_lat)
        self.u, self.v = np.meshgrid(u, v)

        self.e_wire = None
        self.b_wire = None

        self.anim = FuncAnimation(self.fig, self.update,
                                  frames=np.linspace(0, 2*np.pi, 200),
                                  interval=50, blit=False)

    def update_speed(self, value):
        self.speed = value / 20.0
        self.slider_label.setText(f"Animation Speed: {self.speed:.2f}")

    def update(self, t):

        if self.e_wire:
            self.e_wire.remove()
        if self.b_wire:
            self.b_wire.remove()

        phase = self.n_ripples * self.v + t * self.speed

        # Electric field ripple (red)
        R_e = self.R0 * (1 + self.A * np.sin(phase))
        X_e = R_e * np.cos(self.v)
        Y_e = R_e * np.sin(self.u) * np.sin(self.v)
        Z_e = R_e * np.cos(self.u) * np.sin(self.v)

        self.e_wire = self.ax.plot_wireframe(
            X_e, Y_e, Z_e,
            color='r', linewidth=0.8, alpha=0.7
        )

        # Magnetic field ripple (blue, π phase shift)
        R_b = self.R0 * (1 - self.A * np.sin(phase))
        X_b = R_b * np.cos(self.v)
        Y_b = R_b * np.sin(self.u) * np.sin(self.v)
        Z_b = R_b * np.cos(self.u) * np.sin(self.v)

        self.b_wire = self.ax.plot_wireframe(
            X_b, Y_b, Z_b,
            color='b', linewidth=0.8, alpha=0.7
        )

        return []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotonSphereRipplesHorizontalPoles()
    window.show()
    sys.exit(app.exec_())
