import os
import sys
import numpy as np
import copy

# from matplotlib.backends.qt_compat import QtWidgets as qtw
from PySide6 import QtWidgets as qtw
from PySide6 import QtCore as qtc

from graphing import MatplotlibWidget
from signal_tools import Curve
import personalized_widgets as pwi
import pyperclip  # requires also xclip on Linux
from functools import partial

import logging
logging.basicConfig(level=logging.INFO)

# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html


class CurveAnalyze(qtw.QWidget):

    signal_good_beep = qtc.Signal()

    def __init__(self, settings):
        super().__init__()
        self._global_settings = settings
        self._create_core_objects()
        self._create_widgets()
        self._place_widgets()
        self._make_connections()

    def _create_core_objects(self):
        self._user_input_widgets = dict()

    def _create_widgets(self):
        self._graph = MatplotlibWidget()
        self._graph_buttons = pwi.PushButtonGroup(
            {"import": "Import",
             "auto_import": "Auto Import",
             "reset_indices": "Reset Indexes",
             "remove": "Remove",
             "process": "Process",
             "export": "Export",
             "settings": "Settings",
             },
            {"import": "Import 2D graph data from clipboard"},
        )
        self._graph_buttons.user_values_storage(self._user_input_widgets)
        self._user_input_widgets["auto_import_pushbutton"].setCheckable(True)
        self._user_input_widgets["auto_import_pushbutton"].setEnabled(False)

        self._curve_list = qtw.QListWidget()
        self._curve_list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)

    def _place_widgets(self):
        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(self._graph, 2)
        self.layout().addWidget(self._graph_buttons)
        self.layout().addWidget(self._curve_list)

    def _make_connections(self):
        self._user_input_widgets["remove_pushbutton"].clicked.connect(self.remove_curves)
        self._user_input_widgets["reset_indices_pushbutton"].clicked.connect(self._reset_indices)
        self._user_input_widgets["process_pushbutton"].clicked.connect(self._process_curve)
        self._user_input_widgets["import_pushbutton"].clicked.connect(
            partial(self._import_curve, self._read_clipboard)
        )

    def _read_clipboard(self):
        data = pyperclip.paste()
        new_curve = Curve(data)
        if new_curve.is_curve():
            return new_curve
        else:
            logging.debug("Unrecognized curve object")

    def _reset_indices(self):
        labels = []
        for i in range(self._curve_list.count()):
            list_item = self._curve_list.item(i)
            name = list_item.data(qtc.Qt.ItemDataRole.UserRole)["name"]
            name_with_number = f"#{i:02d} - {name}"
            labels.append(name_with_number)
            list_item.setText(name_with_number)
        self._graph.update_labels(labels)

    def _add_curve(self, curve):
        i = self._curve_list.count()
        name_with_number = f"#{i:02d} - {curve.get_name()}"
        list_item = qtw.QListWidgetItem(name_with_number)
        list_item.setData(qtc.Qt.ItemDataRole.UserRole, {"name": curve.get_name(),
                                                         "curve": curve,
                                                         "processed": False,
                                                         }
                          )
        self._curve_list.addItem(list_item)
        self._graph.add_line2D(i, name_with_number, curve.get_xy())


    def _process_curve(self, process_fun):
        for list_item in self._curve_list.selectedItems():
            i = self._curve_list.row(list_item)
            name_with_number_processed = list_item.text() + " - processed"
            user_data = list_item.data(qtc.Qt.ItemDataRole.UserRole)
            
            user_data["curve_processed"] = True
            curve = user_data["curve"]
            
            curve.set_xy(curve.get_xy(ndarray=True) + 5)
            self._graph.update_line2D(i, name_with_number_processed, curve.get_xy(ndarray=True))


    def _import_curve(self, import_fun):
        new_curve = import_fun()
        if new_curve:
            self._add_curve(new_curve)
        else:
            # beep bad
            pass

    def remove_curves(self):
        for list_item in self._curve_list.selectedItems():
            i = self._curve_list.row(list_item)
            self._curve_list.takeItem(i)
            self._graph.remove_line2D(i)


class AutoImporter(qtc.QThread):
    def __init__(self):
        super().__init__()
    
    def run(self):
        while True:
            self.sleep(1)
            print("hop")

if __name__ == "__main__":

    if not (app := qtw.QApplication.instance()):
        app = qtw.QApplication(sys.argv)
        # there is a new recommendation with qApp but how to dod the sys.argv with that?

    mw = CurveAnalyze(settings=None)

    mw.show()
    app.exec()
