import logging
import sys
import numpy as np
from operator import methodcaller
from functools import partial

from PySide6 import QtCore as qtc
from matplotlib.backends.qt_compat import QtWidgets as qtw
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patheffects as mpe
plt.rcParams["figure.constrained_layout.h_pad"] = 0.3
plt.rcParams["figure.constrained_layout.w_pad"] = 0.4

# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html

logging.basicConfig(level=logging.INFO)


class MatplotlibWidget(qtw.QWidget):
    def __init__(self, settings):
        self.app_settings = settings
        super().__init__()
        layout = qtw.QVBoxLayout(self)
        desired_style = 'bmh'
        if desired_style in plt.style.available:
            plt.style.use(desired_style)
        else:
            print(f"Desired style '{desired_style}' not available.")

        fig = Figure()
        fig.set_layout_engine("constrained")
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
        self.lines_in_order = []
        # self.visibility_states = {}

        # https://matplotlib.org/stable/api/_as_gen/matplotlib._lines.Line2D.html

    @qtc.Slot()
    def update_figure(self, recalculate_limits=True, update_legend=True):

        if recalculate_limits:
            def ceil_to_multiple(number, multiple=5, clearance=2):
                return multiple * np.ceil((number + clearance) / multiple)
    
            def floor_to_multiple(number, multiple=5, clearance=2):
                return multiple * np.floor((number - clearance) / multiple)
    
            if self.ax.get_lines():
                y_min = floor_to_multiple(np.min(np.concatenate(
                    [line.get_ydata() for line in self.ax.get_lines()])), clearance=1)
                y_max = ceil_to_multiple(np.max(np.concatenate(
                    [line.get_ydata() for line in self.ax.get_lines()])))
                self.ax.set_ylim((y_min, y_max))

        if update_legend:
            if self.ax.has_data() and self.app_settings.show_legend:
                self.show_legend_ordered()
            else:
                self.ax.legend().remove()

        self.canvas.draw()

    @qtc.Slot()
    def add_line2D(self, i, label, data: tuple, update_figure=True, **kwargs):
        line, = self.ax.semilogx(*data, label=label, **kwargs)
        self.lines_in_order.insert(i, line)

        self.update_zorders_from_lines_in_order()
        if update_figure:
            self.update_figure()

    @qtc.Slot()
    def remove_line2D(self, ix: list, update_figure=True):
        for i in reversed(ix):
            line = self.lines_in_order.pop(i)
            line.remove()

        self.update_zorders_from_lines_in_order()
        if update_figure:
            self.update_figure()

    def show_legend_ordered(self):
        visible_lines = [line for line in self.lines_in_order if line.get_alpha() in (None, 1)]
        handles = visible_lines[:self.app_settings.max_legend_size]
        labels = [line.get_label() for line in handles]

        self.ax.legend(handles, labels)

    def change_lines_order(self, new_positions: list):
        lines_reordered = []
        for i_line in new_positions:
            lines_reordered.append(self.lines_in_order[i_line])
        self.lines_in_order = lines_reordered
        self.update_zorders_from_lines_in_order()
        self.update_figure(recalculate_limits=False)

    def update_zorders_from_lines_in_order(self):
        for i, line in enumerate(self.lines_in_order):
            line.set_zorder(i)

    @qtc.Slot()
    def mark_selected_curve(self, i: int):
        if not hasattr(self, "default_lw"):
            self.default_lw = self.ax.get_lines()[0].get_lw()
        line = self.lines_in_order[i]
        line.set_lw(self.default_lw*2.5)
        old_alpha = line.get_alpha()
        if old_alpha:
            line.set_alpha(1)

        self.update_figure(recalculate_limits=False, update_legend=False)

        timer = qtc.QTimer()
        timer.singleShot(2000, partial(self.remove_marking, line, (old_alpha, self.default_lw)))

    def remove_marking(self, line, old_states):
        line.set_alpha(old_states[0])
        line.set_lw(old_states[1])
        self.update_figure(recalculate_limits=False, update_legend=False)

    @qtc.Slot()
    def hide_show_line2D(self, visibility_states: dict, update_figure=True):
        # self.visibility_states = visibility_states
        for i, visible in visibility_states.items():
            line = self.lines_in_order[i]
            alpha = (1 if visible else 0.2)
            line.set_alpha(alpha)

            label = self.lines_in_order[i].get_label()
            if visible and label[0] == "_":
                line.set_label(label.removeprefix("_"))
            if not visible and label[0] != "_":
                line.set_label("_" + label)

        if update_figure:
            self.update_figure(recalculate_limits=False)

    @qtc.Slot()
    def update_labels_and_colors(self, labels: dict, update_figure=True):
        colors = plt.rcParams["axes.prop_cycle"]()

        for line in self.ax.get_lines():
            zorder = line.get_zorder()
            new_label = labels[zorder] if line.get_alpha() in (None, 1) else ("_" + labels[zorder])
            line.set_label(new_label)
            line.set_color(next(colors)["color"])

        if update_figure:
            self.update_figure()


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
