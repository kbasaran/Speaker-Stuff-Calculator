import sys
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets as qtw
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html

import logging
logging.basicConfig(level=logging.INFO)

class MatplotlibWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        layout = qtw.QVBoxLayout(self)
        desired_style = 'bmh'
        if desired_style in plt.style.available:
            plt.style.use(desired_style)
        else:
            print(f"Desired style '{desired_style}' not available.")

        fig = Figure()
        self.canvas = FigureCanvas(fig)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.navigation_toolbar)
        # print(self.navigation_toolbar.layout().itemAt(3).tooltip())
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.subplots()
        self.ax.grid(which='minor')
        
        # self._lines = {}  # dictionary of _lines
        # https://matplotlib.org/stable/api/_as_gen/matplotlib._lines.Line2D.html

    def update_canvas(self):

        def ceil_to_multiple(number, multiple=5):
            return multiple * np.ceil((number + 2) / multiple)

        def floor_to_multiple(number, multiple=5):
            return multiple * np.floor((number - 2) / multiple)

        if len(self.ax.get_lines()):
            y_min = floor_to_multiple(np.min(np.concatenate([line.get_ydata() for line in self.ax.get_lines()])))
            y_max = ceil_to_multiple(np.max(np.concatenate([line.get_ydata() for line in self.ax.get_lines()])))
            self.ax.set_ylim((y_min, y_max))

        self.ax.legend()
        self.canvas.draw()

    def update_line2D(self, i: int, name_with_number:str, new_data:np.ndarray):
        line = self.ax.get_lines()[i]
        line.set_data(new_data)
        line.set_label(name_with_number)
        self.update_canvas()

    def add_line2D(self, i, label, data:tuple, *args, **kwargs):
        self.ax.semilogx(data, label=label, *args, **kwargs)
        self.update_canvas()

    def remove_line2D(self, i):
        self.ax.get_lines()[i].remove()
        self.update_canvas()

    def update_labels(self, labels: str):
        for i, line in enumerate(self.ax.get_lines()):
            line.set_label(labels[i])
        self.update_canvas()


if __name__ == "__main__":

    if not (app := qtw.QApplication.instance()):
        app = qtw.QApplication(sys.argv)
        # there is a new recommendation with qApp but how to dod the sys.argv with that?

    mw = MatplotlibWidget()

    # do a test plot
    x = 100 * 2**np.arange(stop=7, step=7 / 16)
    for i in range(1, 5):
        y = 45 + 10 * np.random.random(size=len(x))
        mw.add_line2D(i, f"Random line {i}", (x, y))

    mw.show()
    app.exec()
