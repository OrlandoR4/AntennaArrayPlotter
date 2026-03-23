# Antenna Array Plotter

A desktop application for visualizing and analyzing the radiation patterns of linear, uniformly spaced, antenna arrays. Built with Python, PyQt6, and Matplotlib.

## Features

- Interactive polar pattern plots
- Graphical construction of the array factor from the visible region
- Arbitrary element input with amplitude and phase control
- Uniform and binomial array generators
- Adjustable element spacing and progressive phase shift
- Pattern readouts: half-power beamwidth (HPBW) and sidelobe level (SLL)

## Requirements

- Python 3.10+
- PyQt6
- Matplotlib
- NumPy
- SciPy

Install dependencies with:

```bash
python -m pip install PyQt6 matplotlib numpy scipy
```

## Usage

Run the application directly:

```bash
python main.py
```

## Download (Windows)
A pre-compiled Windows executable is available on the [Releases](https://github.com/OrlandoR4/AntennaArrayPlotter/releases) page — no Python installation required. Download AntennaArrayPlotter.exe and run it directly.

### Adding Elements

Enter elements in the **Add an Isotropic Antenna Element** box using the format:

```
amplitude phase
```

For example, `1.5 45` adds an element with amplitude 1.5 and phase 45°. Commas are also accepted: `1.5, 45`. You may also skip the phase and just input an amplitude alone for a phase of zero.

### Generating Arrays

Use **Edit → Generate** to populate the table with a standard array type:

- **Uniform Array** — equal amplitudes and zero phase progression
- **Binomial Array** — binomial amplitude weighting for sidelobe suppression

### Phase Units

Toggle between degrees and radians under **Options**. All values in the table are converted automatically when switching modes.

## Building a Standalone Executable

To package the app into a single `.exe` (Windows) run:

```bash
python build.py
```

The executable will be output to the `dist/` folder. Requires PyInstaller:

```bash
python -m pip install pyinstaller
```

## File Structure

```
├── main.py       # Main application — UI and plotting logic
├── build.py      # PyInstaller build script
├── icon.ico      # Application icon
└── README.md
```
