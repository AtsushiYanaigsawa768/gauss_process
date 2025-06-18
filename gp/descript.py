# /root/gauss_process/gp/descript.py
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")

# ------------------------- IO utils ------------------------- #
def load_bode_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def load_all_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all .dat files and return concatenated data with filenames."""
    all_omega, all_mag, all_phase = [], [], []
    filenames = []
    
    for fp in sorted(data_dir.glob("SKE2024_data*.dat")):
        omega, mag, phase = load_bode_data(fp)
        all_omega.append(omega)
        all_mag.append(mag)
        all_phase.append(phase)
        filenames.append(fp.name)
    
    if not all_omega:
        return np.empty(0), np.empty(0), np.empty(0), []
    
    omega_concat = np.concatenate(all_omega)
    mag_concat = np.concatenate(all_mag)
    phase_concat = np.concatenate(all_phase)
    
    return omega_concat, mag_concat, phase_concat, filenames


def main():
    DATA_DIR = Path("./gp/data")
    
    # Load all data files
    omega, mag, phase, filenames = load_all_data(DATA_DIR)
    
    if omega.size == 0:
        raise RuntimeError("No data files found. Check data directory.")
    
    # Convert to complex transfer function
    G = mag * np.exp(1j * phase)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color mapping based on log frequency (log omega)
    log_omega = np.log10(omega)
    norm = Normalize(vmin=log_omega.min(), vmax=log_omega.max())
    
    # Plot Nyquist diagram with log frequency-based coloring (dark blue→green→yellow)
    scatter = ax.scatter(G.real, G.imag, c=log_omega, cmap='viridis', s=20, alpha=0.9)
    
    # Set axis limits
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.1])
    
    # Set labels and title with larger font sizes
    ax.set_xlabel("Re{G}", fontsize=18)
    ax.set_ylabel("Im{G}", fontsize=18)
    ax.set_title("Nyquist Plot - All Data Files (Color: Log Frequency log₁₀(ω))", fontsize=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log Frequency log₁₀(ω) [rad/s]', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # Add grid
    ax.grid(True, ls="--", alpha=0.3)
    
    # Make layout tight
    plt.tight_layout()
    
    # Save figure
    out_png = Path("./gp/output/nyquist_all_data_raw.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"Loaded {len(filenames)} data files:")
    for fname in filenames:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
