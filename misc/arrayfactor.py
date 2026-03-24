"""
    Antenna Array Plotter (Prototyping code)
    Copyright (C) 2026  Orlando Rangel Morales

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License,
    or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
import numpy as np

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

# Array Factor Class
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

# Program for testing
def main():
    # Uniform array generator
    ckd = 2*np.pi * 1 / 4
    cfN = 6
    cfB = ckd

    uniform_coeffs = uniform_array_coefficients(cfN, 0)

    # Array Definition
    array = ArrayFactor(ckd, uniform_coeffs)

    ###########################################
    ## Graphical Solution Values

    # Values of beta and kd to obtain the graphical solution
    beta = cfB
    kd   = ckd

    # Horizontal range in multiples of pi
    plot_min_norm = -1.1
    plot_max_norm =  1.1

    # Generate range of theta and psi values
    theta = np.linspace(0, np.pi, 1000)
    psi = np.linspace(plot_min_norm * np.pi, plot_max_norm * np.pi, 1000)

    # Generate array factor from theta and psi values
    af_magnitude_psi =    np.abs(array.factor(psi,   mode='psi')) / len(array)
    af_magnitude_theta = np.abs(array.factor(theta, mode='theta', phase_shift=beta)) / len(array)

    # Generate polar values from theta
    x_pattern = af_magnitude_theta * np.cos(theta)
    y_pattern = af_magnitude_theta * np.sin(theta)

    # Generate unit semi-circle for plotting
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    # Figure for plots
    figure = plt.figure(figsize=(10, 8), layout="constrained")

    # |AF|n vs psi in multiples of pi
    ax1 = figure.add_subplot(2, 1, 1) # 1 row, 2 columns, 1st plot
    ax1.axvline( (beta + kd) / np.pi , color='black', linestyle='--', linewidth=1.2)
    ax1.axvline( (beta - kd) / np.pi , color='black', linestyle='--', linewidth=1.2)

    ax1.plot(psi / np.pi, af_magnitude_psi)

    ax1.set_xlim(plot_min_norm, plot_max_norm)
    ax1.grid(True)
    ax1.set_xlabel(r'$\psi$ [rad] in multiples of $\pi$')
    ax1.set_title('|AF| normalized')

    # Radiation pattern from graphical solution
    ax2 = figure.add_subplot(2, 1, 2) # 1 row, 2 columns, 1st plot
    ax2.axvline( 1, color='black', linestyle='--', linewidth=1.2)
    ax2.axvline(-1, color='black', linestyle='--', linewidth=1.2)

    ax1.axvline( (beta / np.pi) , color='gray', linestyle='--', linewidth=1.2)
    ax2.axvline(               0, color='gray', linestyle='--', linewidth=1.2)

    ax2.plot(x_pattern, y_pattern)
    ax2.plot(x_circle, y_circle, color='black', linestyle=':')
    ax2.plot(x_circle / np.sqrt(2), y_circle / np.sqrt(2), color='grey', linestyle=':')

    ax2.set_yticks([])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim( (plot_min_norm * np.pi - beta) / kd , (plot_max_norm * np.pi - beta) / kd)
    ax2.set_aspect('equal')

    # Show plot
    plt.show()

# Only run test code if this is the main script
if __name__ == "__main__":
    # This block is executed only when the script is run directly
    main()