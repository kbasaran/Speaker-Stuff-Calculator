import sys
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets as qtw
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html

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
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.subplots()
        self.ax.grid(which='minor')
        
        self.lines = {}  # dictionary of lines
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html

    def update_canvas(self):

        def ceil_to_multiple(number, multiple=5):
            return multiple * np.ceil((number + 2) / multiple)

        def floor_to_multiple(number, multiple=5):
            return multiple * np.floor((number - 2) / multiple)

        if len(self.lines):
            y_min = floor_to_multiple(np.min(np.concatenate([line.get_ydata() for line in self.lines.values()])))
            y_max = ceil_to_multiple(np.max(np.concatenate([line.get_ydata() for line in self.lines.values()])))
            self.ax.set_ylim((y_min, y_max))

        self.ax.legend()
        self.canvas.draw()

    def update_line2D(self, i: int, new_data: tuple, description=None):
        line = self.lines[i]
        line.set_data(*new_data)
        if description:
            line.set_label(description)
        self.update_canvas()

    def add_line2D(self, label, data, *args, **kwargs):
        line, = self.ax.semilogx(*data, label=label, *args, **kwargs)
        i = 0 if len(self.lines) == 0 else max(self.lines) + 1
        self.lines[i] = line
        self.update_canvas()
        return i


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