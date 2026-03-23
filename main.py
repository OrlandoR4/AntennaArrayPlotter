import sys

# from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, \
#                              QVBoxLayout, QHBoxLayout, QPushButt on, QLabel, \
#                              QTableWidget, QTableWidgetItem

from PyQt6.QtWidgets import * # Visual Elements, windows, buttons, etc
from PyQt6.QtCore import *    # Non-visual elements, timers, signals
from PyQt6.QtGui import *     # Fonts, colors, keyboard/mouse events

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
from scipy.special import comb
from scipy.signal import find_peaks

# Dark mode
# plt.style.use('dark_background')

# Force light mode on Windows (darkmode=0, 1, or 2 may work depending on Qt version)
# 'windows:darkmode=1' or 'windows:darkmode=0' usually forces light mode.
# darkmode=2 forces dark mode.
sys.argv += ['-platform', 'windows:darkmode=0'] 

# Functions

def polar(r: float, theta: float, mode: str ='rad') -> complex:
    """
    Returns the complex number from its polar form r*e^(j*theta).
    Parameters
    ----------
    r : float
        The magnitude of the complex number.
    theta : float
        The angle of the complex number.
    mode : str, optional
        The unit of theta. Either 'rad' (radians) or 'deg' (degrees).
        Default is 'rad'.
    Returns
    -------
    complex
        The complex number r*e^(j*theta).
    Raises
    ------
    ValueError
        If mode is not 'rad' or 'deg'.
    Examples
    --------
    >>> polar(1, np.pi)
    (-1+1.2246467991473532e-16j)
    >>> polar(1, 180, mode='deg')
    (-1+1.2246467991473532e-16j)
    """
    if mode == 'rad':
        theta = np.exp(1j * theta)
    elif mode == 'deg':
        theta = np.exp(1j * np.deg2rad(theta))
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'rad' or 'deg'.")

    return r * theta

def uniform_array_coefficients(N: int, beta_rad: float) -> np.ndarray:
    """
    Retuns an array of coefficients for a uniform linear array with constant phase shifts
    """
    return np.array([1 * np.exp(1j * n * beta_rad) for n in range(N)])

def binomial_array_coefficients(N: int) -> np.ndarray:
    return np.array([comb(N-1, n, exact=True) * np.exp(0j) for n in range(N)])

# Classes

class ArrayFactor:
    """
    Computes the array factor for a uniformly spaced linear antenna array.

    The array factor is defined as:
        AF(theta) = sum_{n=0}^{N-1} a_n * e^(j * n * kd * cos(theta))

    where a_n are the excitation coefficients, kd is the normalized
    element spacing (wavenumber * distance), and theta is the observation
    angle from the array axis.

    Attributes
    ----------
    array_kd : float
        Normalized element spacing (k * d), where k = 2*pi/lambda
        and d is the physical spacing between elements.
    array_coeffs : np.ndarray
        Complex excitation coefficients for each array element.
    array_length : int
        Number of elements in the array, derived from coefficients.

    Examples
    --------
    >>> coeffs = np.array([1, 1, 1, 1], dtype=complex)
    >>> af = ArrayFactor(kd=np.pi/2, coefficients=coeffs)
    >>> af.factor(np.pi/2)
    (4+0j)
    >>> angles = np.linspace(0, np.pi, 360)
    >>> pattern = af.factor(angles)
    """
    def __init__(self, kd: float, coefficients: np.ndarray):
        """
        Parameters
        ----------
        kd : float
            Normalized element spacing (k * d).
        coefficients : np.ndarray
            Complex excitation coefficients for each element.
        """
        self.array_kd = kd
        self.array_coeffs = coefficients
        self.array_length = len(coefficients)

    def __repr__(self) -> str:
        """
        Returns a string representation of the ArrayFactor object.

        Returns
        -------
        str
            A string displaying the kd value, number of elements,
            and excitation coefficients.
        """
        return (f"ArrayFactor(kd={self.array_kd}, "
                f"elements={self.array_length}, "
                f"coeffs={self.array_coeffs})")

    def __len__(self) -> int:
        return self.array_length

    def update(self, kd: float = None, coefficients: np.ndarray = None) -> None:
        """
        Updates the array parameters. Only provided arguments are changed.

        Parameters
        ----------
        kd : float, optional
            New normalized element spacing (k * d). If None, unchanged.
        coefficients : np.ndarray, optional
            New complex excitation coefficients. If None, unchanged.
            Also updates array_length if provided.
        """
        if kd is not None:
            self.array_kd = kd

        if coefficients is not None:
            self.array_coeffs = coefficients
            self.array_length = len(coefficients)

    def factor(self, angle_rad: float | np.ndarray, phase_shift: float = 0, mode: str = 'theta') -> complex | np.ndarray:
        """
        Computes the array factor at one or more angles.

        Parameters
        ----------
        angle_rad : float or np.ndarray
            Observation angle(s) in radians. Can be a single value or an
            array of M angles for sweeping the full radiation pattern.
            Interpreted as theta or psi depending on mode.
        phase_shift : float, optional
            phase shift 'beta' to add to kd*cos(theta) or psi
        mode : str, optional
            The interpretation of angle_rad:
            - 'theta' : physical angle from array axis, kd*cos(theta)
                        is computed internally.
            - 'psi'   : phase variable psi = kd*cos(theta), used directly.
            Default is 'theta'.

        Returns
        -------
        complex or np.ndarray
            The complex array factor value(s). Returns a scalar for a
            single input angle, or a 1D array of length M for an array
            of M input angles.

        Raises
        ------
        ValueError
            If mode is not 'theta' or 'psi'.

        Examples
        --------
        >>> af.factor(np.pi/2, mode='theta')        # Broadside, single angle
        >>> af.factor(np.linspace(0, np.pi, 360))   # Full pattern sweep
        """
        if mode == 'theta':
            angle_rad = self.array_kd * np.cos(angle_rad)
        elif mode == 'psi':
            pass
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'theta' or 'psi'.")
        
        n = np.arange(self.array_length)
        value = np.dot(self.array_coeffs, np.exp(1j * np.outer(n, angle_rad + phase_shift)))
        return value

class PatternPlotCanvas(FigureCanvasQTAgg):
    def __init__(self):
        figure = Figure(figsize=(8, 8), layout="constrained")
        self.ax1 = figure.add_subplot(2, 1, 1)
        self.ax2 = figure.add_subplot(2, 1, 2)
        super().__init__(figure)

    def updatePlot(self, array: ArrayFactor, kd: float, beta: float) -> tuple[float, float]:
        # Update array with kd
        array.update(kd=kd)

        # Clear Axes
        self.ax1.cla()
        self.ax2.cla()

        # Horizontal range in multiples of pi
        plot_min_norm = -2.1
        plot_max_norm =  2.1

        # Find true maximum over full psi range
        psi_full = np.linspace(-np.pi, np.pi, 5000)
        af_full = np.abs(array.factor(psi_full, mode='psi'))
        af_max = np.max(af_full)

        # Check if the maximum is zero
        if af_max == 0:
            return
        
        # Generate range of theta and psi values
        resolution = 1000
        theta = np.linspace(0, np.pi, resolution)
        psi = np.linspace(plot_min_norm * np.pi, plot_max_norm * np.pi, resolution)

        # Generate array factor from theta and psi values
        af_magnitude_psi =   np.abs(array.factor(psi,   mode='psi')) / af_max
        af_magnitude_theta = np.abs(array.factor(theta, mode='theta', phase_shift=beta)) / af_max

        # Generate polar values from theta
        x_pattern = af_magnitude_theta * np.cos(theta)
        y_pattern = af_magnitude_theta * np.sin(theta)

        # Find maximum of the pattern
        pattern_max = np.max(af_magnitude_theta)

        # Generate max semi-circle for plotting
        x_circle = np.cos(theta) * 1.0
        y_circle = np.sin(theta) * 1.0

        ## HPBW Calculations
        af_magnitude_theta_hp = af_magnitude_theta / np.max(af_magnitude_theta) - (1.0/np.sqrt(2.0))
        af_hp_crossings_indx = np.where( np.diff(np.sign(af_magnitude_theta_hp)) )[0]
        af_maxes_indx = np.where(af_magnitude_theta - np.max(af_magnitude_theta) > -1e-12)[0]

        af_hp_crossings_indx = self.clusterIndices(af_hp_crossings_indx)
        af_maxes_indx = self.clusterIndices(af_maxes_indx)

        theta_maxes = theta[af_maxes_indx]
        theta_hps = theta[af_hp_crossings_indx]

        theta_hp_beamwidth = self.findHPBW(theta_maxes, theta_hps)

        # SLL Caculation
        theta_hp_beamwidth = self.findHPBW(theta_maxes, theta_hps)
        sll = self.findSLL(af_magnitude_theta)

        # print(f'Maxes at: {np.rad2deg(theta[af_maxes_indx])}')
        # print(f'HPs at: {np.rad2deg(theta[af_hp_crossings_indx])}\n')

        # |AF|n vs psi in multiples of pi
        self.ax1.axvline( (beta + kd) / np.pi , color='black', linestyle='--', linewidth=1.2)
        self.ax1.axvline( (beta - kd) / np.pi , color='black', linestyle='--', linewidth=1.2)

        self.ax1.plot(psi / np.pi, af_magnitude_psi)

        self.ax1.set_xlim(plot_min_norm, plot_max_norm)
        self.ax1.grid(True)
        self.ax1.set_xlabel(r'$\psi$ [rad] in multiples of $\pi$')
        self.ax1.set_title(f'|AF| normalized (maximum: {af_max:.3g})')

        # Radiation pattern from graphical solution
        self.ax2.axvline( 1, color='black', linestyle='--', linewidth=0.8)
        self.ax2.axvline(-1, color='black', linestyle='--', linewidth=0.8)

        self.ax1.axvline( (beta / np.pi) , color='gray', linestyle='--', linewidth=1.2)
        self.ax2.axvline(               0, color='gray', linestyle='--', linewidth=0.8)

        self.ax2.plot(x_pattern, y_pattern)
        
        self.ax2.plot(x_circle / np.sqrt(2) * pattern_max, 
                      y_circle / np.sqrt(2) * pattern_max, 
                      color= 'grey', linestyle=':', linewidth=1.0)
        
        self.ax2.plot(x_circle * 1.000, y_circle * 1.000,           color='black', linestyle=':', linewidth=0.9)
        self.ax2.plot(x_circle * 0.750, y_circle * 0.750,           color='black', linestyle=':', linewidth=0.9)
        self.ax2.plot(x_circle * 0.500, y_circle * 0.500,           color='black', linestyle=':', linewidth=0.9)
        self.ax2.plot(x_circle * 0.250, y_circle * 0.250,           color='black', linestyle=':', linewidth=0.9)
        self.ax2.plot(x_circle * 0.125, y_circle * 0.125,           color='black', linestyle=':', linewidth=0.9)

        self.ax2.set_yticks([])
        self.ax2.set_ylim(0, 1.1)
        self.ax2.set_xlim(-1.1 , 1.1)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel(r'|AF| normalized vs. $\theta$')

        self.ax2.text( 1.2, 0.5, r'$\theta = 0$',   va='center', ha='left')
        self.ax2.text(-1.2, 0.5, r'$\theta = \pi$', va='center', ha='right')

        # Update figure, just like plt.show()
        self.draw()

        return theta_hp_beamwidth, sll

    # Utility Functions
    def clusterIndices(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return indices
        keep = np.concatenate(([True], np.diff(indices) > 1))
        return indices[keep]
    
    def findSLL(self, af_magnitude_theta: np.ndarray) -> float | None:
        # Mirror to full -pi to pi pattern so edge peaks are fully visible
        af_full = np.concatenate([
            af_magnitude_theta[::-1],   # Mirror: -pi to 0
            af_magnitude_theta,          # Original: 0 to pi
            af_magnitude_theta[::-1]    # Mirror: pi to 2pi
        ])

        all_peaks_indx, _ = find_peaks(af_full)

        if len(all_peaks_indx) == 0:
            return None

        peak_values = af_full[all_peaks_indx]
        global_max  = np.max(peak_values)

        # Classify peaks — sidelobes are those not within tolerance of global max
        tolerance = 1e-2
        sidelobe_values = peak_values[peak_values < global_max - tolerance]

        if len(sidelobe_values) == 0:
            return None

        return 20 * np.log10(np.max(sidelobe_values) / global_max)

    def findHPBW(self, theta_maxes: np.ndarray, theta_hps: np.ndarray) -> float:
        # Mirror everything to the full -pi to pi pattern
        full_hps   = np.concatenate([-theta_hps[::-1],   theta_hps])
        full_maxes = np.concatenate([-theta_maxes[::-1],  theta_maxes])

        # Remove duplicates introduced by mirroring (points at or near theta=0)
        full_hps   = np.unique(np.round(full_hps,   decimals=6))
        full_maxes = np.unique(np.round(full_maxes, decimals=6))

        # For each beam, find the two HP crossings that bracket it
        hpbws = []

        # print(f'MAX: {np.round(np.rad2deg(full_maxes), 2)}\nHPS: {np.round(np.rad2deg(full_hps), 2)}\n')

        for max_angle in full_maxes:
            left_crossings  = full_hps[full_hps < max_angle]
            right_crossings = full_hps[full_hps > max_angle]

            has_left  = len(left_crossings)  > 0
            has_right = len(right_crossings) > 0

            if has_left and has_right:
                # Normal case — crossing on both sides
                left_hp  = left_crossings[-1]
                right_hp = right_crossings[0]
                hpbws.append(right_hp - left_hp)

            elif has_left and not has_right:
                # Beam at right edge (endfire at +180°)
                left_hp = left_crossings[-1]
                hpbws.append(2 * (max_angle - left_hp))

            elif has_right and not has_left:
                # Beam at left edge (endfire at -180°)
                right_hp = right_crossings[0]
                hpbws.append(2 * (right_hp - max_angle))

            # else: no crossings at all, skip

        return max(hpbws) if hpbws else None

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antenna Array Plotter")
        self.setMinimumSize(1000, 600)
        self.setWindowIcon(QIcon("icon.ico"))

        ## Menu bar - QMainWindow has this menu bar built in
        menubar = self.menuBar()
        menu_file = menubar.addMenu("File")
        menu_edit = menubar.addMenu("Edit")
        menu_options = menubar.addMenu("Options")

        # File Actions
        action_quit = QAction("Quit", self)
        action_quit.triggered.connect(self.close)
        menu_file.addAction(action_quit)
        
        # Edit Actions
        action_clear = QAction("Clear Element Table", self)
        action_clear.triggered.connect(self.eventClearElementTable)
        menu_edit.addAction(action_clear)

        menu_generate = menu_edit.addMenu("Generate")
        action_generate_uniform  = QAction("Uniform Array",  self)
        action_generate_binomial = QAction("Binomial Array", self)
        action_generate_uniform.triggered.connect(lambda: self.eventGenerateArray('uniform'))
        action_generate_binomial.triggered.connect(lambda: self.eventGenerateArray('binomial'))

        menu_generate.addAction(action_generate_uniform)
        menu_generate.addAction(action_generate_binomial)

        # Option Actions
        self.unit_action_group = QActionGroup(self)
        self.unit_action_group.setExclusive(True)

        action_degrees = QAction("Degrees", self, checkable=True, checked=True)
        action_radians = QAction("Radians", self, checkable=True)

        action_degrees.toggled.connect(lambda checked: self.eventPhaseUnitChanged() if checked else None)
        action_radians.toggled.connect(lambda checked: self.eventPhaseUnitChanged() if checked else None)

        self.unit_action_group.addAction(action_degrees)
        self.unit_action_group.addAction(action_radians)
        menu_options.addAction(action_degrees)
        menu_options.addAction(action_radians)

        # Central wdiget - QMainWindow requires one
        central = QWidget()
        self.setCentralWidget(central)
        
        ### Horizontal Layout - Two Panels
        # controls | plots
        layout_main = QHBoxLayout(central)

        ## Left Panel - controls
        layout_main_left = QVBoxLayout()

        # Table of antenna elements
        self.table_elements = QTableWidget()
        self.table_elements.setColumnCount(3)
        self.table_elements.setHorizontalHeaderLabels(["Amplitude", "Phase", ""])
        self.table_elements.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_elements.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Element input
        self.input_element = QLineEdit()
        self.input_element.setPlaceholderText("e.g.  1.5 20  or  1.5, 20    (Amplitude: 1.5 Phase: 20)")

        button_element_add = QPushButton("Add Element")
        button_element_add.clicked.connect(self.eventAddElement)
        self.input_element.returnPressed.connect(self.eventAddElement)

        group_input_element = QGroupBox("Add an Isotropic Antenna Element")
        layout_input_element = QHBoxLayout()
        layout_input_element.addWidget(self.input_element)
        layout_input_element.addWidget(button_element_add)
        group_input_element.setLayout(layout_input_element)

        # Phase slider
        self.RESOLUTION_SLIDER_PHASE = 10000
        self.slider_phase = QSlider(Qt.Orientation.Horizontal)
        self.slider_phase.setRange(-self.RESOLUTION_SLIDER_PHASE, self.RESOLUTION_SLIDER_PHASE)
        self.slider_phase.setValue(0)

        self.spinbox_phase = QDoubleSpinBox()
        self.spinbox_phase.setRange(-1.0, 1.0)
        self.spinbox_phase.setSingleStep(0.001)
        self.spinbox_phase.setDecimals(3)
        self.spinbox_phase.setValue(0)

        self.slider_phase.valueChanged.connect(self.eventSliderPhaseMoved)
        self.spinbox_phase.valueChanged.connect(self.eventSpinboxPhaseChanged)

        group_beta = QGroupBox("Phase Shift β (multiples of π radians)")
        layout_betabox = QHBoxLayout()
        layout_betabox.addWidget(self.slider_phase)
        layout_betabox.addWidget(self.spinbox_phase)
        group_beta.setLayout(layout_betabox)

        # Element distance slider
        initial_d = 0.25
        self.RESOLUTION_SLIDER_DISTANCE = 10000
        self.slider_distance = QSlider(Qt.Orientation.Horizontal)
        self.slider_distance.setRange(1, self.RESOLUTION_SLIDER_DISTANCE)
        self.slider_distance.setValue(int(self.RESOLUTION_SLIDER_DISTANCE * initial_d))

        self.spinbox_distance = QDoubleSpinBox()
        self.spinbox_distance.setRange(0, 1.0)
        self.spinbox_distance.setSingleStep(0.001)
        self.spinbox_distance.setDecimals(3)
        self.spinbox_distance.setValue(initial_d)

        self.slider_distance.valueChanged.connect(self.eventSliderDistanceMoved)
        self.spinbox_distance.valueChanged.connect(self.eventSpinboxDistanceChanged)

        group_distance = QGroupBox("Element Distance d (multiples of λ)")
        layout_distancebox = QHBoxLayout()
        layout_distancebox.addWidget(self.slider_distance)
        layout_distancebox.addWidget(self.spinbox_distance)
        group_distance.setLayout(layout_distancebox)

        # Readouts
        group_readouts = QGroupBox("Polar Pattern Characteristics")
        layout_readouts = QFormLayout()
        self.label_hpbw = QLabel("-")
        self.label_sll = QLabel("-")
        layout_readouts.addRow("Half-Power Beamwidth (degrees): ", self.label_hpbw)
        layout_readouts.addRow("Sidelobe-level (dB): ", self.label_sll)
        group_readouts.setLayout(layout_readouts)

        # Add all elements to left layout
        layout_main_left.addWidget(self.table_elements)
        layout_main_left.addWidget(group_input_element)
        layout_main_left.addWidget(group_beta)
        layout_main_left.addWidget(group_distance)
        layout_main_left.addWidget(group_readouts)
        layout_main_left.addStretch()

        ## Right panel - plot
        layout_main_right = QVBoxLayout()

        # Array Plot
        self.pattern_plot = PatternPlotCanvas()
        layout_main_right.addWidget(self.pattern_plot)

        ### Add to main layout
        layout_main.addLayout(layout_main_left)
        layout_main.addLayout(layout_main_right)
        layout_main.setStretch(0, 2)
        layout_main.setStretch(1, 5)

        ### Array Factor
        self.array = ArrayFactor(1e-6, [])
        
        # Initialize with an example array
        self.eventPhaseUnitChanged()
        initial_coeffs = uniform_array_coefficients(5, 0)
        for coefficient in initial_coeffs:
            self.input_element.setText(
                str(np.abs(coefficient)) +
                " " + 
                str(np.angle(coefficient))
                )
            self.eventAddElement()

        # Initialize plot
        self.updatePatternPlot()

    ## Antenna Table Functions
    def parseElementInput(self, text: str) -> tuple[float, float]:
        parts = text.replace(',', ' ').split()

        if len(parts) == 1:
            return float(parts[0]), float(0)
        elif len(parts) != 2:
            return None, None
        
        return float(parts[0]), float(parts[1])

    def getCoefficientsFromTable(self) -> np.ndarray:
        coeffs = []
        unit = self.getPhaseUnit()
        for r in range(self.table_elements.rowCount()):
            amplitude = float(self.table_elements.item(r, 0).text())
            phase     = float(self.table_elements.item(r, 1).text())
            if unit == 'Degrees':
                phase = np.deg2rad(phase)
            coeffs.append(polar(amplitude, phase))
        return np.array(coeffs)
    
    def getPhaseUnit(self) -> str:
        return self.unit_action_group.checkedAction().text() # 'Degrees' or 'Radians'

    def eventAddElement(self):
        amplitude, phase = self.parseElementInput(self.input_element.text())
        if amplitude is None or phase is None:
            return  # Invalid input, do nothing

        idx_row = self.table_elements.rowCount()
        self.table_elements.insertRow(idx_row)

        self.table_elements.setItem(idx_row, 0, QTableWidgetItem(f"{amplitude:.3g}"))
        self.table_elements.setItem(idx_row, 1, QTableWidgetItem(f"{phase:.3g}"))

        btn_remove = QPushButton("Delete")
        btn_remove.clicked.connect(lambda _, r=idx_row: self.eventRemoveElement(r))
        self.table_elements.setCellWidget(idx_row, 2, btn_remove)

        self.input_element.clear()
        self.updatePatternPlot()

    def eventRemoveElement(self, row: int):
        self.table_elements.removeRow(row)
        # Reconnect buttons
        for r in range(self.table_elements.rowCount()):
            btn = self.table_elements.cellWidget(r, 2)
            btn.clicked.disconnect()
            btn.clicked.connect(lambda _, row=r: self.eventRemoveElement(row))
        self.updatePatternPlot()        

    def eventPhaseUnitChanged(self):
        new_unit = self.getPhaseUnit()  # The unit being switched TO

        # Convert all existing phase values in the table
        for r in range(self.table_elements.rowCount()):
            phase = float(self.table_elements.item(r, 1).text())

            if new_unit == 'Degrees':
                phase = np.rad2deg(phase)   # Was radians, convert to degrees
            else:
                phase = np.deg2rad(phase)   # Was degrees, convert to radians

            self.table_elements.setItem(r, 1, QTableWidgetItem(f"{phase:.5g}"))

        self.table_elements.setHorizontalHeaderItem(1, QTableWidgetItem(f"Phase [{new_unit}]"))
        self.updatePatternPlot()
        return

    def eventClearElementTable(self):
        self.table_elements.setRowCount(0)
        self.updatePatternPlot()
        
    ## Phase Functions
    def getSliderPhaseValue(self):
        return self.slider_phase.value() / self.RESOLUTION_SLIDER_PHASE
    
    def eventSliderPhaseMoved(self):
        self.updatePatternPlot()
        self.spinbox_phase.blockSignals(True) # Prevents pattern plot from running twice
        self.spinbox_phase.setValue( self.getSliderPhaseValue() )
        self.spinbox_phase.blockSignals(False) # Prevents pattern plot from running twice

    def eventSpinboxPhaseChanged(self):
        self.updatePatternPlot()
        self.slider_phase.blockSignals(True) # Prevents pattern plot from running twice
        self.slider_phase.setValue(int( self.spinbox_phase.value() * self.RESOLUTION_SLIDER_PHASE ))
        self.slider_phase.blockSignals(False) # Prevents pattern plot from running twice

    ## Distance Functions
    def getSliderDistanceValue(self):
        return self.slider_distance.value() / self.RESOLUTION_SLIDER_DISTANCE
    
    def eventSliderDistanceMoved(self):
        self.updatePatternPlot()
        self.spinbox_distance.blockSignals(True) # Prevents pattern plot from running twice
        self.spinbox_distance.setValue( self.getSliderDistanceValue() )
        self.spinbox_distance.blockSignals(False) # Prevents pattern plot from running twice

    def eventSpinboxDistanceChanged(self):
        self.updatePatternPlot()
        self.slider_distance.blockSignals(True) # Prevents pattern plot from running twice
        self.slider_distance.setValue(int( self.spinbox_distance.value() * self.RESOLUTION_SLIDER_DISTANCE ))
        self.slider_distance.blockSignals(False) # Prevents pattern plot from running twice
    
    ## Editing Functions  
    def promptNumberOfElements(self) -> int | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Array")

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Number of elements:"))

        spinbox = QSpinBox()
        spinbox.setRange(1, 1000)
        spinbox.setValue(1)
        spinbox.selectAll()
        layout.addWidget(spinbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return spinbox.value()
        
        return None  # User cancelled

    def eventGenerateArray(self, array_type: str):
        N = self.promptNumberOfElements()
        if N is None:
            return  # User cancelled

        unit = self.getPhaseUnit()

        if array_type == 'uniform':
            coeffs = uniform_array_coefficients(N, 0)
        elif array_type == 'binomial':
            coeffs = binomial_array_coefficients(N)  # You'll implement this in arrayfactor.py

        # Clear table and repopulate
        self.table_elements.setRowCount(0)
        for coeff in coeffs:
            amplitude = np.abs(coeff)
            phase     = np.angle(coeff)
            if unit == 'Degrees':
                phase = np.rad2deg(phase)
            self.input_element.setText(f"{amplitude:.3g} {phase:.3g}")
            self.eventAddElement()

    ## Miscellaneous Functions
    def getNormalizedDistanceKD(self):
        return 2 * np.pi * self.getSliderDistanceValue()

    ## Plotter Functions
    def updatePatternPlot(self):
        coeffs = self.getCoefficientsFromTable()
        if len(coeffs) == 0:
            self.label_hpbw.setText("-")
            self.label_sll.setText("-")
            self.pattern_plot.ax1.cla()
            self.pattern_plot.ax2.cla()
            self.pattern_plot.draw()
            return
        self.array.update(coefficients=coeffs)
        [hpbw, sll] = self.pattern_plot.updatePlot(self.array, 
                                                   self.getNormalizedDistanceKD(), 
                                                   self.getSliderPhaseValue() * np.pi)
        
        if hpbw is not None:
            self.label_hpbw.setText(f'{np.rad2deg(float(hpbw)): .3g}')
        else:
            self.label_hpbw.setText('-')

        if sll is not None:
            self.label_sll.setText(f'{sll:.3g} dB')
        else:
            self.label_sll.setText('-')

# Main program

def main():
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()