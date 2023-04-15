import sys
import numpy as np
from deepn import NeuralNetwork
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deep Learning Visualization")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()

    def init_ui(self):
        # Create input widgets
        self.input_layers = QtWidgets.QLineEdit("10,5,1", self)
        self.input_alpha = QtWidgets.QLineEdit("0.1", self)
        self.input_epochs = QtWidgets.QLineEdit("10000", self)
        self.input_display_update = QtWidgets.QLineEdit("1000", self)
        self.input_X = QtWidgets.QLineEdit("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0;1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0;", self)
        self.input_y = QtWidgets.QLineEdit("0.1,0.2,0.3,0.4,0.5;1.1,1.2,1.3,1.4,1.5;", self)

        # Create labels for the input widgets
        layers_label = QtWidgets.QLabel("Layers:", self)
        alpha_label = QtWidgets.QLabel("Alpha:", self)
        epochs_label = QtWidgets.QLabel("Epochs:", self)
        display_update_label = QtWidgets.QLabel("Display Update:", self)
        X_label = QtWidgets.QLabel("X:", self)
        y_label = QtWidgets.QLabel("y:", self)

        # Create a layout to hold the input widgets and labels
        inputs_layout = QtWidgets.QGridLayout()
        inputs_layout.addWidget(layers_label, 0, 0)
        inputs_layout.addWidget(self.input_layers, 0, 1)
        inputs_layout.addWidget(alpha_label, 1, 0)
        inputs_layout.addWidget(self.input_alpha, 1, 1)
        inputs_layout.addWidget(epochs_label, 2, 0)
        inputs_layout.addWidget(self.input_epochs, 2, 1)
        inputs_layout.addWidget(display_update_label, 3, 0)
        inputs_layout.addWidget(self.input_display_update, 3, 1)
        inputs_layout.addWidget(X_label, 4, 0)
        inputs_layout.addWidget(self.input_X, 4, 1)
        inputs_layout.addWidget(y_label, 5, 0)
        inputs_layout.addWidget(self.input_y, 5, 1)

        # Create the "Fit" button
        fit_button = QtWidgets.QPushButton("Fit", self)
        fit_button.clicked.connect(self.fit)

        # Create the "Predict" button
        predict_button = QtWidgets.QPushButton("Predict", self)
        predict_button.clicked.connect(self.predict)

        # Create a layout to hold the buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(fit_button)
        buttons_layout.addWidget(predict_button)

        # # Create a layout to hold the inputs and buttons
        # main_layout = QtWidgets.QVBoxLayout()
        # main_layout.addLayout(inputs_layout)
        # main_layout.addLayout(buttons_layout)

        # # Create a central widget to hold the main layout
        # central_widget = QtWidgets.QWidget(self)
        # central_widget.setLayout(main_layout)
        # self.setCentralWidget(central_widget)
        
        # Left panel for inputs
        left_panel = QtWidgets.QWidget()
        left_panel_layout = QtWidgets.QVBoxLayout()
        left_panel.setLayout(left_panel_layout)
        
        # Add inputs and buttons to the left panel layout
        left_panel_layout.addLayout(inputs_layout)
        left_panel_layout.addLayout(buttons_layout)
        
        # Right panel for prediction visualization
        right_panel = QtWidgets.QWidget()
        right_panel_layout = QtWidgets.QVBoxLayout()
        right_panel.setLayout(right_panel_layout)
        
        # Initialize an empty plot in the right panel
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        right_panel_layout.addWidget(self.canvas)

        # Create a container layout to hold the left and right panels
        container_layout = QtWidgets.QHBoxLayout()
        container_layout.addWidget(left_panel)
        container_layout.addWidget(right_panel)

        # Create a central widget to hold the container layout
        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(container_layout)
        self.setCentralWidget(central_widget)

    def fit(self):
        layers = list(map(int, self.input_layers.text().strip().split(',')))
        alpha = float(self.input_alpha.text().strip())
        epochs = int(self.input_epochs.text().strip())
        display_update = int(self.input_display_update.text().strip())
        X = np.array([list(map(float, x.strip().split(','))) for x in self.input_X.text().strip().rstrip(';').split(';')])
        y = np.array([list(map(float, x.strip().split(','))) for x in self.input_y.text().strip().rstrip(';').split(';')])
        y = y.reshape((-1, 1)) # reshape to have shape (n_samples, 1)

        self.model = NeuralNetwork(layers, alpha=alpha)
        self.model.fit(X, y, epochs=epochs, display_update=display_update)




    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer].T)
            out = self.sigmoid(net)
            A.append(out)
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer]).T


    def plot_results(self, y_true, y_pred):
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(y_true, '-', label='Ground Truth', linewidth=2)
        ax.plot(y_pred, '-', label='Predicted', linewidth=2)
        ax.legend()

        canvas = FigureCanvas(fig)
        canvas.setParent(self)
        canvas.move(300, 50)
        canvas.show()


    def predict(self):
        X = np.array([[float(x) for x in row.split(',')] for row in self.input_X.text().split(';')])
        y_true = np.array([list(map(float, x.strip().split(','))) for x in self.input_y.text().strip().split(';')])
        y_pred = self.model.predict(X)
        print("Prediction: {}".format(y_pred))

        self.plot_results(y_true, y_pred)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
