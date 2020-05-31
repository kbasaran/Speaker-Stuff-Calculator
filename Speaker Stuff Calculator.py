# -*- coding: utf-8 -*-
"""Speaker Stuff Calculator main module."""

do_print = 1

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from dataclasses import dataclass
import pickle
from PySide2.QtCore import SIGNAL, SLOT, QObject, Qt  # Qt isnecessary for alignment of titles
from PySide2.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                               QDoubleSpinBox, QAbstractSpinBox, QGroupBox,
                               QFormLayout, QPlainTextEdit, QStackedWidget,
                               QVBoxLayout, QHBoxLayout, QFrame, QSpinBox,
                               QRadioButton, QComboBox, QLineEdit,
                               QButtonGroup, QFileDialog)
from PySide2.QtGui import QFont
from functools import partial
import winsound

def beep(frequency=1175, requested_duration=100):
    period = 1000 / frequency  # ms
    requested_wave_number = requested_duration / period
    for wave_number in range(int(requested_wave_number), 10,-1):
        duration = wave_number * period
        if np.abs(duration - np.round(duration)) < 0.01:
            duration_no_click = int(np.round(duration))
            winsound.Beep(frequency, duration_no_click)
            return
    winsound.Beep(frequency, requested_duration)

class Record(object):
    def setattrs(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

constants = Record()
constants.setattrs(GAMMA = 1.401,  # adiabatic index of air
                    P0 = 101325,
                    RHO = 1.1839,  # 25 degrees celcius
                    Kair = 101325 * 1.401,  # could not find a way to refer to RHO here
                    c_air = (101325 * 1.401 / 1.1839)**0.5)
setattr(constants, "VC_TABLE", pd.read_csv(".\SSC_data\VC_TABLE.csv", index_col="Name"))

def generate_freqs(freq_start, freq_end, ppo):
    """
    Create a numpy array for frequencies to use in calculation.

    ppo means points per octave
    """
    import numpy as np
    numStart = np.floor(np.log2(freq_start/1000)*ppo)
    numEnd = np.ceil(np.log2(freq_end/1000)*ppo + 1)
    freq_array = 1000*np.array(2**(np.arange(numStart, numEnd)/ppo))
    return(freq_array)

def find_nearest_freq(array, value):
    """
    Parameters
    ----------
    array : np.array
    value : int, float

    Returns
    -------
    closest_val
        Closest value found.
    idx : int
        Index of closest value.
    """
    err = [abs(i-value) for i in array]
    idx = err.index(min(err))
    return array[idx], idx

# Frequency points
f = generate_freqs(10, 5000, 48*4)

def calculate_air_mass(Sd):
    """Air mass on diaphragm; the difference between Mms and Mmd."""
    return(1.13*(Sd)**(3/2))  # m2 in, kg out

def calculate_Lm(Bl, Re, Mms, Sd):
    w_ref = 10**-12
    half_space_Q = 1/2/np.pi
    P_1W = half_space_Q * constants.Kair * Bl**2 * Sd**2 / constants.c_air**3 / Re / Mms**2 / 2 / np.pi
    return 10 * np.log10(P_1W/w_ref)

def calculate_Xmech(Xmax):
    """Proposed Xmech value for given Xmax value."""
    Xclearance = 1e-3 + (Xmax - 3e-3) / 5
    return(Xmax + Xclearance)  # mm in in same unit with Xmax


def calculate_windings(wire_type, N_layers, former_OD, h_winding):
    """Calculate coil mass, Rdc and l for a given coil."""
    d_wire = constants.VC_TABLE.loc[wire_type, "avg"] / 1000

    def calc_N_winding_per_layer(i_layer):
        """Calculate the number of windings that fit on one layer of coil."""
        val = h_winding / d_wire - i_layer * 2
        return (0 if val < 3 else val)
        # less windings on each stacked layer

    def calc_length_of_one_turn_per_layer(i_layer):
        """Calculate the length of one turn of wire on a given coil layer."""
        turn_mean_radius = former_OD/2 + d_wire/2 + 0.8 * i_layer * d_wire
        # 1.0 is stacking coefficient
        return 2*np.pi*turn_mean_radius

    # Windings amount for each layer
    N_windings = [calc_N_winding_per_layer(i)
                          for i in range(N_layers)]

    # Wire length for one full turn around for a given layer
    l_one_turn = [calc_length_of_one_turn_per_layer(i)
                  for i in range(N_layers)]

    total_length_wire_per_layer = [N_windings[i] * l_one_turn[i]
                                   for i in range(N_layers)]

    l_wire = sum(total_length_wire_per_layer)
    Rdc = l_wire * constants.VC_TABLE.loc[wire_type, "ohm/m"]
    
    N_windings_rounded = [int(np.round(i)) for i in N_windings]
    return(Rdc, N_windings_rounded, l_wire)  # round is new since 8_s6

def calculate_input_voltage(excitation, Rdc, nominal_impedance):
    """Simplify electrical input definition to input voltage."""
    val, type = excitation
    if type == "Wn":
        input_voltage = (val * nominal_impedance) ** 0.5
    elif type == "W":
        input_voltage = (val * Rdc) ** 0.5
    elif type == "V":
        input_voltage = val
    else:
        print("Input options are [float, ""V""], \
              [float, ""W""], [float, ""Wn"", float]")
        return(None)
    return(input_voltage)

@dataclass
class UserForm():

    def __post_init__(self):
        """Post-init the form."""
        return

    def load_pickle(self, pickle_to_load=""):
        try:
            filename = open(pickle_to_load, "rb")
        except:
            filename = QFileDialog.getOpenFileName(None,
                                                   caption='Open file',
                                                   dir=os.getcwd(),
                                                   filter='Pickle Files(*.pickle)')[0]

        with open(filename, "rb") as handle:
            form_dict = pickle.load(handle)
            print("Loaded file .....%s" % filename[0][-20:])

        for item_name, value in form_dict.items():
            self.set_value(item_name, value)

        update_model()

    def save_to_pickle(self):  # add save dialog
        """Save design to a file."""
        form_dict = {}
        for key, value in self.__dict__.items():
            obj_value = self.get_value(key)
            form_dict[key] = obj_value
        filename = QFileDialog.getSaveFileName(None, caption='Save to file',
                                               dir=os.getcwd(),
                                               filter='Pickle Files(*.pickle)')[0]
        with open(filename, 'wb') as handle:
            pickle.dump(form_dict, handle)
            print("Saved to file .....%s" % filename[0][-20:])

    # Convenience functions to add rows to input_form_layout layout
    def add_line(self, to_layout):
        """Add a separator line in the form layout."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setContentsMargins(0, 10, 0, 10)
        to_layout.addRow(line)

    def update_dof(self):
        return

    def add_title(self, to_layout, string):
        """Add a title to different user input form groups."""
        title = QLabel()
        title.setText(string)
        title.setStyleSheet("font-weight: bold")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        to_layout.addRow(title)

    def add_double_float_var(self, to_layout, var_name, description, min_val=0,
                             max_val=1e5, default=0, unit_modifier=1):
        """Add a row for double float user variable input."""
        item = {"obj": QDoubleSpinBox(), "unit_modifier": unit_modifier}
        item["obj"].setRange(min_val, max_val)
        item["obj"].setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        setattr(self, var_name, item)
        self.set_value(var_name, default*unit_modifier)
        to_layout.addRow(description,  getattr(self, var_name)["obj"])

    def add_combo_box(self, to_layout, box_name, combo_list, box_screen_name=False):
        """Make a combo box.

        box_name is the attribute name under form object
        item_list contains tuples as list items. first is visible name second is user_data.
        """
        item = {"obj": QComboBox()}
        item["obj"].setMaxVisibleItems(19)
        for choice in combo_list:
            item["obj"].addItem(*choice)
        setattr(self, box_name, item)
        if isinstance(box_screen_name, str):
            to_layout.addRow(box_screen_name, getattr(self, box_name)["obj"])
        else:
            to_layout.addRow(getattr(self, box_name)["obj"])

    def add_integer_var(self, to_layout, var_name, description, min_val=1,
                        max_val=1e6, default=0, unit_modifier=1):
        """Add a row for integer value user variable input."""
        item = {"obj": QSpinBox(), "unit_modifier": unit_modifier}
        item["obj"].setRange(min_val, max_val)
        setattr(self, var_name, item)
        self.set_value(var_name, default*unit_modifier)
        to_layout.addRow(description,  getattr(self, var_name)["obj"])

    def add_string_var(self, to_layout, var_name, description, default=""):
        """Add string var."""
        item = {"obj": QLineEdit()}
        setattr(self, var_name, item)
        self.set_value(var_name, default)
        to_layout.addRow(description, getattr(self, var_name)["obj"])

    def set_value(self, item_name, value):
        """Set a value of a form item."""
        item = getattr(self, item_name)
        qwidget_obj = item["obj"]
        if isinstance(qwidget_obj, QLineEdit):
            if isinstance(value, str):
                qwidget_obj.setText(value)
            else:
                raise Exception("Incorrect data type %s for %s" % (type(value), qwidget_obj))

        elif isinstance(qwidget_obj, QPlainTextEdit):
            if isinstance(value, str):
                qwidget_obj.setPlainText(value)
            else:
                raise Exception("Incorrect data type %s for %s" % (type(value), qwidget_obj))

        elif isinstance(qwidget_obj, (QDoubleSpinBox, QSpinBox)):
            if isinstance(value, (float, int)):
                qwidget_obj.setValue(value / item["unit_modifier"])
            else:
                raise Exception("Incorrect data type %s for %s" % (type(value), qwidget_obj))

        elif isinstance(qwidget_obj, QComboBox):  # recevies tuple with entry_name
            if isinstance(value, str):
                qwidget_obj.setCurrentText(value)
            else:
                raise Exception("Incorrect data type %s for %s" % (type(value), qwidget_obj))

        elif isinstance(qwidget_obj, QButtonGroup):
            for button in qwidget_obj.buttons():
                if button.isChecked():
                    return button.text()
            else:
                raise Exception("No clicked button for %s" % item_name)

        else:
            raise Exception("Don't know how to set %s with value %g" % (item_name,  value))

    def get_value(self, item_name):
        """Get a value of a form variable."""
        item = getattr(self, item_name)
        try:
            qwidget_object = item["obj"]
            if isinstance(qwidget_object, QLineEdit):
                return qwidget_object.text()
            elif isinstance(qwidget_object, QPlainTextEdit):
                return qwidget_object.toPlainText()
            elif isinstance(qwidget_object, (QDoubleSpinBox, QSpinBox)):
                return qwidget_object.value() * item["unit_modifier"]
            elif isinstance(qwidget_object, QComboBox):  # returns tuple (name, userData)
                return (qwidget_object.currentText(), qwidget_object.currentData())
            elif isinstance(qwidget_object, QButtonGroup):
                return qwidget_object.checkedButton().text()
        except:
            if not isinstance(item, QWidget):
                return item
            else:
                raise Exception("Don't know how to read %s with type %s" % (item_name, type(item)))

    def update_coil_choice_box(self):
        """Scan best matching speaker coil options."""
        self.coil_choice_box["obj"].clear()
        try:  # try to read the N_layer_options string
            layer_options = [int(str) for str in self.N_layer_options["obj"].text().split(", ")]
        except:
            self.error = "Invalid input in number of layer options"
            self.coil_choice_box["obj"].addItem("--" + self.error + "--")
            return
        table_columns = ["N_layers", "wire_type", "Bl", "Rdc", "Lm", "Qts", "former_ID", "t_former", "h_winding", "N_windings", "l_wire"]
        self.coil_options_table = pd.DataFrame(columns=table_columns)

        # Scan through winding options
        winding = Record()

        for k in ["target_Rdc", "former_ID", "t_former", "h_winding"]:
            setattr(winding, k, self.get_value(k))

        for N_layers in layer_options:
            for wire_type, row in constants.VC_TABLE.iterrows():
                Rdc, N_windings, l_wire = calculate_windings(wire_type,
                                                             N_layers,
                                                             winding.former_ID + winding.t_former * 2,
                                                             winding.h_winding)
        # if Rdc is usable, add to DataFrame
                if winding.target_Rdc / 1.1 < Rdc < winding.target_Rdc * 1.15 and 0 not in N_windings:
                    winding_name = str(N_layers) + "x " + wire_type
                    winding_data = {}
                    for k in ["wire_type", "N_layers", "Rdc", "N_windings", "l_wire"]:
                        winding_data[k] = locals()[k]
                    coil_choice = (winding_name, winding_data)
                    speaker = SpeakerDriver(coil_choice, self)
                    self.coil_options_table.loc[winding_name] = [getattr(speaker, i) for i in table_columns]  # add all the parameters of this speaker to a new dataframe row
        self.coil_options_table.sort_values("Lm", ascending=False)

        # Add the coils in dataframe to the combobox (with their userData)
        for winding_name in self.coil_options_table.index:
            # Make a string for the text to show on the combo box
            Rdc_string = ("Rdc=%.2f" % self.coil_options_table.Rdc[winding_name]).ljust(10)
            Lm_string = ("Lm=%.2f" % self.coil_options_table.Lm[winding_name]).ljust(9)
            Qes_string = ("Qts=%.2f" % self.coil_options_table.Qts[winding_name]).ljust(10)
            name_in_combo_box = winding_name.ljust(14) + Rdc_string + Lm_string + Qes_string
            user_data = self.coil_options_table.to_dict("index")[winding_name]
            self.coil_choice_box["obj"].addItem(name_in_combo_box, user_data)
        # if nothing to add to combobox
        if self.coil_choice_box["obj"].count() == 0:
            self.coil_choice_box["obj"].addItem("--no solution found--")

@dataclass
class SpeakerDriver():
    """Speaker driver class."""

    coil_choice: tuple  # (winding_name (e.g.4x SV160), user data dictionary for Rdc, l_wire etc.)
    form: UserForm  # self kullanmadığın için bunu görmüyor bile

    def __post_init__(self):
        """Post-init speaker."""
        self.error = ""
        motor_spec_choice = self.form.get_value("motor_spec_type")[1]
        read_from_form = ["fs", "Qms", "Xmax", "dead_mass", "Sd",
                          "nominal_impedance"]
        for k in read_from_form:
            exec("self." + k + " = %s" % form.get_value(k))

        if motor_spec_choice == "define_coil":
            # try:
            self.B_average = form.get_value("B_average")
            self.former_ID = form.get_value("former_ID")
            self.t_former = form.get_value("t_former")
            self.h_winding = form.get_value("h_winding")

            winding_name, winding_data = self.coil_choice
            self.wire_type = winding_data["wire_type"]
            self.N_layers = winding_data["N_layers"]
            Rdc = winding_data["Rdc"]
            self.N_windings = winding_data["N_windings"]
            self.l_wire = winding_data["l_wire"]

            self.d_max_wire = constants.VC_TABLE.loc[self.wire_type, "max"] / 1000
            self.w_coil = self.N_layers * self.d_max_wire  # worst case overlap
            self.coil_mass = self.l_wire * constants.VC_TABLE.loc[self.wire_type, "g/m"] / 1000
            Bl = self.B_average * self.l_wire
            Mmd = self.dead_mass + self.coil_mass
            # except:
            #     self.error += "Invalid coil parameters for the speaker object.\n"

        elif motor_spec_choice == "define_Bl_Re":
            try:
                Rdc = form.get_value("Rdc")
                Bl = form.get_value("Bl")
                Mmd = form.get_value("Mmd")
            except:
                self.error += "Unable to get  Bl, Re and Mmd values from user form"
        else:
            raise Exception("Unknown motor specification type type")

        # Calculate more acoustical parameters using Bl, Rdc and Mmd
        Mms = Mmd + calculate_air_mass(self.Sd)
        Kms = Mms * (self.fs * 2 * np.pi)**2
        Rms = (Mms * Kms)**0.5 / self.Qms
        Ces = Bl**2 / Rdc
        Qts = (Mms*Kms)**0.5/(Rms+Ces)
        Qes = (Mms*Kms)**0.5/(Ces)
        self.Xmax = form.get_value("Xmax")

        Lm = calculate_Lm(Bl, Rdc, Mms, self.Sd)

        # Add all acoustical variables as instance variables
        attributes = ["Rdc", "Bl", "Mmd", "Mms", "Kms", "Rms", "Ces",
                      "Qts", "Qes", "Lm"]
        for v in attributes:
            exec("self." + v + " = " + v)

        # Make a string for acoustical summary
        self.summary_ace = "Rdc=%.2f ohm    Lm=%.2f dBSPL    Qts=%.2f"\
            % (Rdc, Lm, Qts)
        self.summary_ace += "\r\nBl=%.2f Tm    Qes=%.2f"\
            % (Bl, Qes)
        if motor_spec_choice == "define_coil":
            self.summary_ace += "    Windings=%.2f g" % (self.coil_mass*1000)
        self.summary_ace += "\r\nKms=%.2f N/mm    Rms=%.2f kg/s    Mms=%.2f g"\
            % (Kms/1000, Rms, Mms*1000)

        # Make a string for mechanical summary
        self.summary_mec = \
            "Target Xmax = %.2f mm" % (self.Xmax*1000)

        if motor_spec_choice == "define_coil":
            # Add mechanical variables from user form as instance variables
            read_from_form = ["airgap_clearance_inner",
                              "airgap_clearance_outer",
                              "t_former",
                              "former_ID",
                              "h_winding",
                              "former_extension_under_coil",
                              "h_washer"]
            for k in read_from_form:
                exec("self." + k + " = %s" % form.get_value(k))

            self.air_gap_width = (self.airgap_clearance_inner + self.t_former
                                  + self.w_coil + self.airgap_clearance_outer)

            self.air_gap_dims = [self.former_ID/2 - self.airgap_clearance_inner,
                                 self.air_gap_width,
                                 self.former_ID/2 - self.airgap_clearance_inner
                                 + self.air_gap_width]

            self.overhang = (self.h_winding - self.h_washer) / 2

            self.washer_to_bottom_plate = (self.h_winding / 2
                                           + self.former_extension_under_coil
                                           + calculate_Xmech(self.Xmax)
                                           - self.h_washer / 2)

            self.summary_mec += \
                "\r\nOverhang + 15%% = %.2f mm" % float(self.overhang*1.15*1000)
            self.summary_mec += \
                "\r\nAir gap dims. = %s mm" \
                % (str(np.round([i*1000 for i in self.air_gap_dims], 2)))
            self.summary_mec += \
                "\r\nWindings per layer = %s" % (str(self.N_windings))
            self.summary_mec += \
                "\r\nWasher to bottom plate = %.2f mm (recommended minimum)" \
                % (self.washer_to_bottom_plate*1000)

        self.Xmech = calculate_Xmech(self.Xmax)
        self.summary_mec += \
            "\r\nXmech = %.2f mm (recommended minimum)" % (self.Xmech*1000)

        if self.error != "":
            raise Exception(self.error)

@dataclass
class AcousticalSystem():
    """
    One or two degree of freedom acoustical system class.

    Can be two types: ["Closed box", "Free-air"]
    """

    spk: SpeakerDriver
    constants: Record
    f: np.ndarray

    def __post_init__(self):
        self.error = self.spk.error
        """Calculate more attributes."""
        Bl = self.spk.Bl
        Rdc = self.spk.Rdc
        Mms = self.spk.Mms
        Rms = self.spk.Rms
        Kms = self.spk.Kms
        Sd = self.spk.Sd
        excitation = [form.get_value("excitation_value"), form.get_value("excitation_unit")[1]]
        nominal_impedance = form.get_value("nominal_impedance")
        self.V_in = calculate_input_voltage(excitation, Rdc, nominal_impedance)
        self.box_type = form.get_value("box_type")
        self.dof = int(form.get_value("dof")[0])

        # Read box parameters
        if self.box_type == "Closed box":
            self.Vb, self.Qal = form.get_value("Vb"), form.get_value("Qal")
        else:
            self.Vb, self.Qal = [np.inf] * 2

        # Read dof
        if self.dof > 1:
            m2, k2, c2 = form.get_value("m2"), form.get_value("k2"), form.get_value("c2")

        Kbox = Sd**2*constants.Kair/self.Vb
        Rbox = ((Kms+Kbox)*(Mms/1000))**0.5/self.Qal
        self.fb = 1/2/np.pi * ((Kms+Kbox)/Mms)**0.5
        self.Qtc = ((Kms+Kbox)*Mms)**0.5 / (Rbox + Rms + Bl**2/Rdc)
        self.Vas = constants.Kair / Kms * Sd**2

        # State space model    
        if not hasattr(self, "sysx1"):
            #State, input, output and feed-through matrices and state space system definitions
            if self.dof == 1:
                ass = np.array(
                    [[0, 1],
                    [-Kbox/Mms-Kms/Mms, -Bl**2/Rdc/Mms-Rms/Mms-Rbox/Mms]]
                    )
                bss = np.array([[0], [Bl/Rdc/Mms]])
                cssx1 = np.array([1,0])
                cssx1t = np.array([0,1])
                dss = np.array([0])
            if self.dof == 2:
                ass = np.array(
                    [[0,1,0,0], 
                    [-(Kms+Kbox)/Mms,-(Rms+Rbox+Bl**2/Rdc)/Mms,(Kms+Kbox)/Mms,(Rms+Rbox+Bl**2/Rdc)/Mms],
                    [0,0,0,1],
                    [(Kms+Kbox)/m2,(Rms+Rbox+Bl**2/Rdc)/m2,-(Kms+Kbox+k2)/m2,-(Rms+Rbox+Bl**2/Rdc+c2)/m2]]
                    )
                bss = np.array([[0], [Bl/Rdc/Mms], [0], [-Bl/Rdc/m2]])
                cssx1 = np.array([1,0,0,0])
                cssx1t = np.array([0,1,0,0])
                cssx2 = np.array([0,0,1,0])
                cssx2t = np.array([0,0,0,1])  
                dss = np.array([0])

            self.sysx1 = signal.StateSpace(ass, bss, cssx1, dss)
            self.sysx1t = signal.StateSpace(ass, bss, cssx1t, dss)
            if self.dof > 1:
                self.sysx2 = signal.StateSpace(ass, bss, cssx2, dss)
                self.sysx2t = signal.StateSpace(ass, bss, cssx2t, dss)

        # Output arrays
        w = 2*np.pi*f
        _, self.x1_1V = np.abs(signal.freqresp(self.sysx1, w=w))  # hata veriyo
        _, self.x1t_1V = signal.freqresp(self.sysx1t, w=w)
        self.x1 = self.x1_1V * self.V_in
        if self.dof > 1:
            _, self.x2_1V = signal.freqresp(self.sysx2, w=w)
            _, self.x2t_1V = signal.freqresp(self.sysx2t, w=w)
            self.x2 = self.x2_1V * self.V_in

        # SPL calculation with simplified radiation impedance * acceleration
        a = np.sqrt(Sd/np.pi)  # piston radius
        p0_1V = 0.5 * 1j * w * constants.RHO * a**2 * self.x1t_1V
        pref = 2e-5
        SPL_1V = 20*np.log10(np.abs(p0_1V)/pref)

        # Xmax limited SPL calculation
        x1_max_rms_array = [np.array(self.spk.Xmax/2**0.5)] * len(f)
        x1t_max_rms_array = x1_max_rms_array * w * 1j
        p0_xmax_limited = 0.5 * 1j * w * constants.RHO * a**2 * x1t_max_rms_array
        self.SPL_Xmax_limited = 20*np.log10(np.abs(p0_xmax_limited)/pref)

        self.SPL = SPL_1V + 20*np.log10(self.V_in)
        self.P_real = self.V_in ** 2 / Rdc
        self.Z = Rdc / (1-Bl*(self.x1t_1V))

        # Calculate some extra parameters
        self.x1tt_1V = self.x1t_1V * w * 1j
        self.x1tt = self.x1tt_1V * self.V_in
        if self.dof > 1:
            self.x2tt_1V = self.x2t_1V * w * 1j
            self.x2tt = self.x2tt_1V * self.V_in
        
        if self.box_type == "closed box":
            interested_frequency = self.fb * 4
        else:
            interested_frequency = self.spk.fs * 4
        
        f_interest, f_inter_idx = find_nearest_freq(f, interested_frequency)
        
        self.summary = "SPL at %iHz is %.1f dB" %\
            (f_interest, self.SPL[f_inter_idx])

        if self.dof == 1:
            self.summary += "\r\nPeak displacement at %iHz is %.3g mm" %\
                (f_interest, self.x1[f_inter_idx]*1e3*2**0.5)

            self.summary += "\r\nPeak displacement overall is %.3g mm" %\
                np.max(self.x1*1e3*2**0.5)

        elif self.dof == 2:
            self.summary += "\r\nPeak relative displacement at %iHz is %.3g mm" %\
                (f_interest, (self.x1[f_inter_idx]-self.x2[f_inter_idx])*1e3*2**0.5)

            self.summary += "\r\nPeak relative displacement overall is %.3g mm" %\
                np.max((self.x1-self.x2)*1e3*2**0.5)
        else:
            self.summary += "Unable to identify the total degrees of freedom"

        if self.box_type == "Closed box":
            self.summary += "\r\nQtc: %.2f    fb: %.3g Hz    Vas: %.3g l" \
                            % (self.Qtc, self.fb, self.Vas * 1e3)

        self.summary += "\r\nF_motor(V_in) / F_suspension(Xmax/2) = {:.0%}".format(
            Bl * self.V_in / Rdc / Kms / self.spk.Xmax * 2)

def update_model():
    """Update the model used in calculations."""
    global speaker, result_sys, form
    coil_choice = form.get_value("coil_choice_box")
    winding_name, winding_data = coil_choice
    if winding_data != "":
        speaker = SpeakerDriver(coil_choice, form)
        result_sys = AcousticalSystem(speaker, constants, f)
        update_view()
    else:
        print("Cannot update model with coil_choice: %s" % winding_name)

if __name__ == "__main__":

    # %% Initiate PyQT Application

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    form = UserForm()

    # %% Make CRUD widget - create, read, update, delete
    crud = QHBoxLayout()
    crud_load_button = QPushButton("Load")
    crud_load_button.clicked.connect(partial(form.load_pickle))
    crud_save_button = QPushButton("Save")
    crud_save_button.clicked.connect(partial(form.save_to_pickle))
    crud.addWidget(crud_load_button)
    crud.addWidget(crud_save_button)

    # %% Make a form layout for user input rows
    input_form_layout = QFormLayout()

    # Add basic speaker parameter
    form.add_title(input_form_layout, "General speaker specifications")

    form.add_double_float_var(input_form_layout, "fs", "fs (Hz)", default=111)
    form.add_double_float_var(input_form_layout, "Qms", "Qms", default=6.51)
    form.add_double_float_var(input_form_layout, "Xmax", "Xmax (mm)", default=4, unit_modifier=1e-3)
    form.add_double_float_var(input_form_layout, "dead_mass", "Dead mass (g)", default=3.54, unit_modifier=1e-3)
    form.add_double_float_var(input_form_layout, "Sd", "Sd (cm²)", default=53.5, unit_modifier=1e-4)

    form.add_line(input_form_layout)

    # Add motor input choice combobox
    combo_box_data = [("Define Coil Dimensions and Average B", "define_coil"),
                      ("Define Bl and Rdc", "define_Bl_Re")
                      ]
    form.add_combo_box(input_form_layout, "motor_spec_type", combo_box_data)
    form.motor_spec_type["obj"].setStyleSheet("font-weight: bold")

    # Make Widget for define coil and average B field
    motor_form_1 = QWidget()
    motor_form_1_layout = QFormLayout()
    motor_form_1.setLayout(motor_form_1_layout)
    form.add_double_float_var(motor_form_1_layout, "target_Rdc", "Target Rdc (ohm)", default=3.9)
    form.add_double_float_var(motor_form_1_layout, "former_ID", "Coil Former ID (mm)", default=25.46, unit_modifier=1e-3)
    form.add_integer_var(motor_form_1_layout, "t_former", "Former thickness (\u03BCm)", default=100, unit_modifier=1e-6)
    form.add_double_float_var(motor_form_1_layout, "h_winding", "Coil winding height (mm)", default=6.2, unit_modifier=1e-3)
    form.add_double_float_var(motor_form_1_layout, "B_average", "Average B field on coil (T)", default=0.69)
    form.add_string_var(motor_form_1_layout, "N_layer_options", "Number of layer options", default="2")
    form.N_layer_options["obj"].setToolTip("Enter the winding layer options"
                                           " as integers with \", \" (a comma"
                                           " and a space) in between")
    button_coil_choices_update = QPushButton("Update coil choices")
    button_coil_choices_update.setMaximumWidth(160)
    motor_form_1_layout.addRow(button_coil_choices_update)

    form.add_combo_box(motor_form_1_layout, "coil_choice_box", [("--empty--", "")])

    # Make Widget for define Bl and Rdc
    motor_form_2 = QWidget()
    motor_form_2_layout = QFormLayout()
    motor_form_2.setLayout(motor_form_2_layout)
    form.add_double_float_var(motor_form_2_layout, "Bl", "Bl (Tm)", default=3.43)
    form.add_double_float_var(motor_form_2_layout, "Rdc", "Rdc (ohm)", default=3.77)
    form.add_double_float_var(motor_form_2_layout, "Mmd", "Mmd (g)", default=3.98, unit_modifier=1e-3)

    # Make a stacked widget to show the motor input values based
    # on combobox cohice
    motor_data_input = QStackedWidget()
    motor_data_input.addWidget(motor_form_1)
    motor_data_input.addWidget(motor_form_2)
    # motor_data_input.setFixedHeight(200)
    QObject.connect(form.motor_spec_type["obj"], SIGNAL(
        "activated(int)"), motor_data_input, SLOT("setCurrentIndex(int)"))
    input_form_layout.addRow(motor_data_input)

    # Add mechanical stuff, clearances
    form.add_line(input_form_layout)
    form.add_title(input_form_layout, "Motor mechanical specifications")

    form.add_double_float_var(input_form_layout, "h_washer", "Washer thickness (mm)",
                         default=3, unit_modifier=1e-3)
    form.add_integer_var(input_form_layout, "airgap_clearance_inner", "Airgap inner clearance (\u03BCm)", default=260, unit_modifier=1e-6)
    form.add_integer_var(input_form_layout, "airgap_clearance_outer", "Airgap outer clearance (\u03BCm)", default=260, unit_modifier=1e-6)
    form.add_double_float_var(input_form_layout, "former_extension_under_coil", "Former extension under coil (mm)", default=0.3, unit_modifier=1e-3)

    # Closed box specifications
    form.add_line(input_form_layout)
    form.add_title(input_form_layout, "Closed box specifications")

    form.add_double_float_var(input_form_layout, "Vb", "Box internal volume (l)", default=1, unit_modifier=1e-3)
    form.add_double_float_var(input_form_layout, "Qal", "Ql - box losses", default=200)

    # Second degree of freedom specifications
    form.add_line(input_form_layout)
    form.add_title(input_form_layout, "Second degree of freedom")

    form.add_double_float_var(input_form_layout, "k2", "Stiffness (N/mm)", default = 10, unit_modifier=1e3)
    form.add_double_float_var(input_form_layout, "m2", "Mass (g)", default=1000, unit_modifier=1e-3)
    form.add_double_float_var(input_form_layout, "c2", "Damping coefficient (kg/s)", default=10)

    # Excitation information
    form.add_line(input_form_layout)
    form.add_title(input_form_layout, "Excitation")

    form.add_double_float_var(input_form_layout, "excitation_value", "Excitation value", default=2.83)

    excitation_combo_box_choices = ([("Volt","V"),
                                     ("Watt@Rdc","W"),
                                     ("Watt@Rnom","Wn")
                                     ])
    form.add_combo_box(input_form_layout, "excitation_unit", excitation_combo_box_choices, box_screen_name="Unit")
    form.set_value("excitation_unit", "V")

    form.add_double_float_var(input_form_layout, "nominal_impedance", "Nominal impedance", default=4)

    # form.excitation_unit["obj"].currentIndexChanged.connect(
    #     form.nominal_impedance["obj"].setEnabled(
    #         form.get_value("excitation_unit")[1]=="Wn"))

    # System type selection radio buttons
    form.add_line(input_form_layout)
    form.add_title(input_form_layout, "System type")
    sys_type_selection = QVBoxLayout()
    box_buttons_layout = QHBoxLayout()
    dof_buttons_layout = QHBoxLayout()
    sys_type_selection.addLayout(box_buttons_layout)
    sys_type_selection.addLayout(dof_buttons_layout)

    # Box type
    setattr(form, "box_type", {"obj": QButtonGroup()})
    rb_box = [QWidget] * 5
    rb_box[1] = QRadioButton("Free-air")
    rb_box[1].setChecked(True)
    box_buttons_layout.addWidget(rb_box[1])

    rb_box[2] = QRadioButton("Closed box")
    box_buttons_layout.addWidget(rb_box[2])

    form.box_type["obj"].addButton(rb_box[1])
    form.box_type["obj"].addButton(rb_box[2])

    # DOF choice
    setattr(form, "dof", {"obj": QButtonGroup()})
    rb_dof = [QWidget] * 5
    rb_dof[1] = QRadioButton("1 dof")
    rb_dof[1].setChecked(True)
    dof_buttons_layout.addWidget(rb_dof[1])

    rb_dof[2] = QRadioButton("2 dof")
    dof_buttons_layout.addWidget(rb_dof[2])

    form.dof["obj"].addButton(rb_dof[1])
    form.dof["obj"].addButton(rb_dof[2])

    # update form with system type toggle
    def adjust_form_for_system_type():
        """Update which buttons are enabled based on box type and dof."""
        form.Vb["obj"].setEnabled(rb_box[2].isChecked())
        form.Qal["obj"].setEnabled(rb_box[2].isChecked())
        form.k2["obj"].setEnabled(rb_dof[2].isChecked())
        form.m2["obj"].setEnabled(rb_dof[2].isChecked())
        form.c2["obj"].setEnabled(rb_dof[2].isChecked())
        rb_graph["x2"].setEnabled(rb_dof[2].isChecked())
        rb_graph["x1-x2"].setEnabled(rb_dof[2].isChecked())

    form.box_type["obj"].buttonClicked.connect(adjust_form_for_system_type)
    form.dof["obj"].buttonClicked.connect(adjust_form_for_system_type)

    # %% For right side

    # PLot data selection buttons -baffled, 1dof, 2dof etc.-----------------
    # need to grey out redundant form boxes,
    # e.g. if 1dof system, no need for K2
    plot_data_selection = QWidget()
    layout = QHBoxLayout()
    plot_data_selection.setLayout(layout)
    plot_data_selection.setFixedHeight(60)

    rb_graph = {}
    rb_graph["SPL"] =  QRadioButton("SPL")
    rb_graph["SPL"].setChecked(True)
    layout.addWidget(rb_graph["SPL"])

    rb_graph["Impedance"] = QRadioButton("Impedance")
    layout.addWidget(rb_graph["Impedance"])

    rb_graph["x1"] = QRadioButton("Excursion (x1)")
    layout.addWidget(rb_graph["x1"])

    rb_graph["x2"] = QRadioButton("Excursion (x2)")
    layout.addWidget(rb_graph["x2"])

    rb_graph["x1-x2"] = QRadioButton("Excursion (x1-x2)")
    layout.addWidget(rb_graph["x1-x2"])

    # Message_box for general data----------------------------------------
    message_box = QPlainTextEdit()
    message_box.setMinimumHeight(245)
    message_box.setMinimumWidth(350)
    message_box.setReadOnly(True)

    setattr(form, "user_notes", {"obj": QPlainTextEdit()})

    # %% Graph
    def graph_ceil(x, step=5):
        """Define max scale value for the graph based on highest value."""
        return step * np.ceil(x/step)

    def update_view():
        """Update the output information such as graph, summary strings."""
        global speaker, result_sys
        ax = figure.add_subplot(111)
        ax.clear()
        if result_sys.error == "":
            message_box.setPlainText(speaker.summary_ace
                                     + "\n\r" + speaker.summary_mec
                                     + "\n\r" + result_sys.summary)
            beep()
            if rb_graph["SPL"].isChecked():
                curve = result_sys.SPL
                curve_2 = result_sys.SPL_Xmax_limited
                upper_limit = graph_ceil(np.max(curve) + 3, 10)
                lower_limit = upper_limit - 50
                ax.semilogx(f, curve)
                ax.semilogx(f, curve_2, "m", label="Xmax limited")
                ax.legend()
                ax.set_title("SPL@1m, Half-space, %.2f Volt, %.3g Watt"
                             % (result_sys.V_in, result_sys.P_real))
                ax.set_xbound(lower=10, upper=3000)
                ax.set_ybound(lower=lower_limit, upper=upper_limit)
    
            if rb_graph["Impedance"].isChecked():
                curve = np.real(result_sys.Z)
                ax.semilogx(f, curve)
                ax.set_title("Electrical Impedance (inductance effects not calculated)")
                ax.set_xbound(lower=10, upper=3000)
                ax.set_ybound(lower=0, upper=(graph_ceil(np.max(curve) + 2, 10)))
    
            if rb_graph["x1"].isChecked():
                curve = np.abs(result_sys.x1) * 1000
                ax.semilogx(f, curve)
                ax.semilogx(f, curve * 2**0.5, "m")
                ax.set_title("Absolute RMS and Peak Displacement, One-way")
                ax.set_xbound(lower=10, upper=3000)
                ax.set_ybound(lower=0, upper=(graph_ceil(np.max(curve) * 1.5, 1)))
            
            if rb_graph["x2"].isChecked():
                curve = np.abs(result_sys.x2) * 1000
                ax.semilogx(f, curve)
                ax.semilogx(f, curve * 2**0.5, "m")
                ax.set_title("Absolute RMS and Peak Displacement, One-way")
                ax.set_xbound(lower=10, upper=3000)
                ax.set_ybound(lower=0, upper=(graph_ceil(np.max(curve) * 1.5, 1)))
            
            
            if rb_graph["x1-x2"].isChecked():
                curve = np.abs(result_sys.x1 - result_sys.x2) * 1000
                ax.semilogx(f, curve)
                ax.semilogx(f, curve * 2**0.5, "m")
                ax.set_title("Relative RMS and Peak Displacement, One-way")
                ax.set_xbound(lower=10, upper=3000)
                ax.set_ybound(lower=0, upper=(graph_ceil(np.max(curve) * 1.5, 1)))
    
            ax.grid(True, which="both")
        else:
            message_box.setPlainText("Error: %s" % result_sys.error)
        if do_print:
            canvas.draw()  # refresh canvas

    button21 = QPushButton('Update results')
    button21.clicked.connect(update_model)
    button21.setFixedHeight(40)

    button22 = QPushButton("Export values")
    button22.setToolTip("Export this graph to clipboard as a table")
    button22.clicked.connect(partial(beep, frequency=int(1175/2)))
    button22.setFixedHeight(40)

    button23 = QPushButton("Import overlay")
    button23.setToolTip("Import a table from clipboard and add it to the graph as an overlay")
    button23.clicked.connect(partial(beep, frequency=int(1175/2)))
    button23.setFixedHeight(40)

    button24 = QPushButton("Clear overlays")
    button24.clicked.connect(partial(beep, frequency=int(1175/2)))
    button24.setFixedHeight(40)

# %% Graph and navigation toolbar

    from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas)
    from matplotlib.backends.backend_qt5agg import (
            NavigationToolbar2QT as NavigationToolbar)
    from matplotlib.figure import Figure

    graphs = QWidget()
    # a figure instance to plot on
    figure = Figure(figsize=(5, 7), dpi=72, tight_layout=True)
    if do_print:
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        canvas = FigureCanvas(figure)
    
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        toolbar = NavigationToolbar(canvas, parent=graphs)  # this crashes in Spyder........
        toolbar.setMinimumWidth(400)


# %% Do the overall layout

    # Make a QGroupbox frame around all the around the user input information
    # such as the crud, the form rows and the system type selection
    input_group_box = QGroupBox("Inputs")
    input_group_box.setFixedWidth(300)
    input_group_box_layout = QVBoxLayout()
    input_group_box.setLayout(input_group_box_layout)

    input_group_box_layout.addLayout(crud)
    input_group_box_layout.addLayout(input_form_layout)
    input_group_box_layout.addLayout(sys_type_selection)

    # Place things into the main widget
    main_layout = QHBoxLayout()
    left_layout = QVBoxLayout()
    right_layout = QVBoxLayout()
    main_layout.addLayout(left_layout)
    main_layout.addLayout(right_layout)

    main_win = QWidget()
    main_win.setWindowTitle("Speaker stuff calculator")
    main_win.setLayout(main_layout)
    main_win.setMinimumHeight(800)

    # set the layout
    left_layout.addWidget(input_group_box)
    # left_layout.addWidget(QWidget()) #  empty widget to scale up down - not helping

    if do_print:
        right_layout.addWidget(toolbar)
        right_layout.addWidget(canvas)
    right_layout.addWidget(plot_data_selection)

    graph_buttons = QHBoxLayout()
    graph_buttons.addWidget(button21)
    graph_buttons.addWidget(button22)
    graph_buttons.addWidget(button23)
    graph_buttons.addWidget(button24)

    right_layout.addLayout(graph_buttons)
    text_boxes = QHBoxLayout()
    text_boxes.addWidget(message_box)
    text_boxes.addWidget(form.user_notes["obj"])
    right_layout.addLayout(text_boxes)
    
    def adjust_form_for_calc_type():
        form.h_washer["obj"].setEnabled(form.get_value("motor_spec_type")[1]=="define_coil")
        form.airgap_clearance_inner["obj"].setEnabled(form.get_value("motor_spec_type")[1]=="define_coil")
        form.airgap_clearance_outer["obj"].setEnabled(form.get_value("motor_spec_type")[1]=="define_coil")
        form.former_extension_under_coil["obj"].setEnabled(form.get_value("motor_spec_type")[1]=="define_coil")
        form.dead_mass["obj"].setEnabled(form.get_value("motor_spec_type")[1]=="define_coil")
    button_coil_choices_update.clicked.connect(partial(form.update_coil_choice_box))

    rb_graph_group = QButtonGroup()
    for key, val in rb_graph.items():
        rb_graph_group.addButton(val)
    rb_graph_group.buttonClicked.connect(update_view)

    def update_nominal_impedance_disability():
        form.nominal_impedance["obj"].setEnabled(form.get_value("excitation_unit")[1]=="Wn")
    form.excitation_unit["obj"].currentIndexChanged.connect(update_nominal_impedance_disability)

    # Initiation actions
    adjust_form_for_calc_type()
    adjust_form_for_system_type()
    update_nominal_impedance_disability()
    update_model()
    main_win.show()
    sys.exit(app.exec_())