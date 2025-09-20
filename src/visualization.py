"""
Visualization module for system identification pipeline.
Provides plotting functions for each stage of the process.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import matplotlib.gridspec as gridspec


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_gp_interpolation(
    freq_train: np.ndarray,
    response_train: np.ndarray,
    freq_dense: np.ndarray,
    response_dense: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "GP Interpolation Results",
    show: bool = False
) -> None:
    """
    Plot GP interpolation results showing magnitude and phase.

    Parameters:
    -----------
    freq_train : np.ndarray
        Training frequency points
    response_train : np.ndarray
        Training frequency response (complex)
    freq_dense : np.ndarray
        Dense interpolation frequency points
    response_dense : np.ndarray
        Interpolated frequency response (complex)
    save_path : Optional[Path]
        Path to save the figure
    title : str
        Main title for the plot
    show : bool
        Whether to display the plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Magnitude plot (log-log)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(freq_train, np.abs(response_train), 'ko', markersize=8,
               label='Training data', alpha=0.7)
    ax1.loglog(freq_dense, np.abs(response_dense), 'b-', linewidth=2,
               label='GP interpolation')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Magnitude Response')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Phase plot
    ax2 = fig.add_subplot(gs[0, 1])
    phase_train = np.unwrap(np.angle(response_train)) * 180/np.pi
    phase_dense = np.unwrap(np.angle(response_dense)) * 180/np.pi
    ax2.semilogx(freq_train, phase_train, 'ko', markersize=8,
                 label='Training data', alpha=0.7)
    ax2.semilogx(freq_dense, phase_dense, 'b-', linewidth=2,
                 label='GP interpolation')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.set_title('Phase Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Nyquist plot
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(np.real(response_train), np.imag(response_train), 'ko',
             markersize=8, label='Training data', alpha=0.7)
    ax3.plot(np.real(response_dense), np.imag(response_dense), 'b-',
             linewidth=2, label='GP interpolation')

    # Add frequency annotations
    n_annotations = 5
    indices = np.linspace(0, len(freq_dense)-1, n_annotations, dtype=int)
    for idx in indices:
        ax3.annotate(f'{freq_dense[idx]:.1f} Hz',
                    (np.real(response_dense[idx]), np.imag(response_dense[idx])),
                    textcoords="offset points", xytext=(10, 10), ha='center',
                    fontsize=8, alpha=0.7)

    ax3.set_xlabel('Real')
    ax3.set_ylabel('Imaginary')
    ax3.set_title('Nyquist Plot with GP Interpolation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GP interpolation plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fir_results(
    time: np.ndarray,
    true_output: np.ndarray,
    predicted_output: np.ndarray,
    fir_coeffs: np.ndarray,
    metrics: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "FIR Model Results",
    show: bool = False
) -> None:
    """
    Plot FIR model validation results.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    true_output : np.ndarray
        True output signal
    predicted_output : np.ndarray
        Predicted output from FIR model
    fir_coeffs : np.ndarray
        FIR filter coefficients
    metrics : Dict[str, float]
        Performance metrics
    save_path : Optional[Path]
        Path to save the figure
    title : str
        Main title
    show : bool
        Whether to display the plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # Output comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, true_output, 'k-', linewidth=1.5, label='True output', alpha=0.8)
    ax1.plot(time, predicted_output, 'r--', linewidth=1.5, label='FIR prediction')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Output')
    ax1.set_title('Output Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add zoom inset
    if len(time) > 1000:
        # Zoom on interesting region
        t_start = len(time) // 4
        t_end = t_start + min(1000, len(time) // 10)
        axins = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
        axins.plot(time[t_start:t_end], true_output[t_start:t_end], 'k-', linewidth=1)
        axins.plot(time[t_start:t_end], predicted_output[t_start:t_end], 'r--', linewidth=1)
        axins.set_xlim(time[t_start], time[t_end])
        ax1.indicate_inset_zoom(axins, edgecolor="blue", alpha=0.5)

    # Error plot
    ax2 = fig.add_subplot(gs[1, :])
    error = true_output - predicted_output
    ax2.plot(time, error, 'b-', linewidth=1)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Prediction Error (RMSE = {metrics["rmse"]:.4f})')
    ax2.grid(True, alpha=0.3)

    # FIR coefficients
    ax3 = fig.add_subplot(gs[2, 0])
    L = min(len(fir_coeffs), 200)  # Show first 200 coefficients
    ax3.stem(range(L), fir_coeffs[:L], basefmt=' ', linefmt='g-', markerfmt='go')
    ax3.set_xlabel('Tap')
    ax3.set_ylabel('Coefficient')
    ax3.set_title(f'FIR Coefficients (first {L} of {len(fir_coeffs)})')
    ax3.grid(True, alpha=0.3)

    # Metrics display
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    metrics_text = f"""Performance Metrics:

RMSE:  {metrics['rmse']:.6f}
NRMSE: {metrics['nrmse']:.6f}
R²:    {metrics['r2']:.4f}
FIT%:  {metrics['fit_percent']:.2f}%

FIR Length: {len(fir_coeffs)}"""

    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"FIR results plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_frequency_comparison(
    frequencies: np.ndarray,
    original_response: np.ndarray,
    gp_response: np.ndarray,
    fir_response: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Frequency Response Comparison",
    show: bool = False
) -> None:
    """
    Compare frequency responses from different stages.

    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency points
    original_response : np.ndarray
        Original measured response
    gp_response : np.ndarray
        GP interpolated response
    fir_response : Optional[np.ndarray]
        FIR model response
    save_path : Optional[Path]
        Path to save figure
    title : str
        Plot title
    show : bool
        Whether to display plot
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Magnitude
    ax1.loglog(frequencies, np.abs(original_response), 'ko', markersize=6,
               label='Original data', alpha=0.5)
    ax1.loglog(frequencies, np.abs(gp_response), 'b-', linewidth=2,
               label='GP interpolation')
    if fir_response is not None:
        ax1.loglog(frequencies, np.abs(fir_response), 'r--', linewidth=2,
                   label='FIR model')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Magnitude Response Comparison')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Phase
    phase_orig = np.unwrap(np.angle(original_response)) * 180/np.pi
    phase_gp = np.unwrap(np.angle(gp_response)) * 180/np.pi
    ax2.semilogx(frequencies, phase_orig, 'ko', markersize=6,
                 label='Original data', alpha=0.5)
    ax2.semilogx(frequencies, phase_gp, 'b-', linewidth=2,
                 label='GP interpolation')
    if fir_response is not None:
        phase_fir = np.unwrap(np.angle(fir_response)) * 180/np.pi
        ax2.semilogx(frequencies, phase_fir, 'r--', linewidth=2,
                     label='FIR model')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.set_title('Phase Response Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Frequency comparison plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_data_overview(
    time: np.ndarray,
    input_data: np.ndarray,
    output_data: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Input/Output Data Overview",
    show: bool = False
) -> None:
    """
    Plot overview of raw input/output data.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    input_data : np.ndarray
        Input signal
    output_data : np.ndarray
        Output signal
    save_path : Optional[Path]
        Path to save figure
    title : str
        Plot title
    show : bool
        Whether to display plot
    """
    setup_plot_style()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Input signal
    ax1.plot(time, input_data, 'b-', linewidth=1)
    ax1.set_ylabel('Input')
    ax1.set_title('Input Signal')
    ax1.grid(True, alpha=0.3)

    # Output signal
    ax2.plot(time, output_data, 'r-', linewidth=1)
    ax2.set_ylabel('Output')
    ax2.set_title('Output Signal')
    ax2.grid(True, alpha=0.3)

    # Input/Output correlation (zoomed view)
    if len(time) > 1000:
        # Show a smaller window for detail
        start_idx = len(time) // 4
        end_idx = start_idx + min(1000, len(time) // 10)
        ax3.plot(time[start_idx:end_idx], input_data[start_idx:end_idx], 'b-',
                 linewidth=1.5, label='Input', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(time[start_idx:end_idx], output_data[start_idx:end_idx], 'r-',
                      linewidth=1.5, label='Output', alpha=0.7)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Input', color='b')
        ax3_twin.set_ylabel('Output', color='r')
        ax3.set_title('Zoomed View')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.plot(time, input_data, 'b-', linewidth=1.5, label='Input', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(time, output_data, 'r-', linewidth=1.5, label='Output', alpha=0.7)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Input', color='b')
        ax3_twin.set_ylabel('Output', color='r')
        ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data overview plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_summary_figure(
    results: Dict,
    save_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create a comprehensive summary figure with all results.

    Parameters:
    -----------
    results : Dict
        Dictionary containing all results from the pipeline
    save_path : Optional[Path]
        Path to save figure
    show : bool
        Whether to display plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Add individual plots based on available results
    # This is a template - customize based on actual results structure

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_gp_interpolation()")
    print("  - plot_fir_results()")
    print("  - plot_frequency_comparison()")
    print("  - plot_data_overview()")
    print("  - create_summary_figure()")