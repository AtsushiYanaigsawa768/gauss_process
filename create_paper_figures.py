#!/usr/bin/env python3
"""
create_paper_figures.py

論文用の高品質な図を生成するスクリプト
- 入力・出力の時系列データを2列の図として表示
- PDF（ベクター形式）とPNG（プレビュー用）を両方出力
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import glob

# 論文用の設定
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 1.5


def load_mat_data(mat_file):
    """
    Load [t, y, u] from MAT file.

    Returns:
        t: time vector [s]
        u: input signal
        y: output signal
    """
    data = loadmat(mat_file)

    # Try to find data in different formats
    t = None
    y = None
    u = None

    # Format 1: Named variables t, u, y
    if 't' in data or 'time' in data:
        t = np.ravel(data.get('t', data.get('time'))).astype(float)
        u = np.ravel(data.get('u')).astype(float)
        y = np.ravel(data.get('y')).astype(float)

    # Format 2: 3xN or Nx3 array
    if t is None:
        for key, val in data.items():
            if key.startswith('__') or not isinstance(val, np.ndarray):
                continue
            if val.ndim == 2 and (val.shape[0] == 3 or val.shape[1] == 3):
                if val.shape[0] == 3:
                    t = np.ravel(val[0, :]).astype(float)
                    y = np.ravel(val[1, :]).astype(float)
                    u = np.ravel(val[2, :]).astype(float)
                else:
                    t = np.ravel(val[:, 0]).astype(float)
                    y = np.ravel(val[:, 1]).astype(float)
                    u = np.ravel(val[:, 2]).astype(float)
                break

    if t is None or u is None or y is None:
        raise ValueError(f"Could not load [t, u, y] from {mat_file}")

    return t, u, y


def create_paper_figure(t, u, y, output_prefix, title_suffix=""):
    """
    Create publication-quality figure with input and output time series.

    Args:
        t: time vector [s]
        u: input signal
        y: output signal
        output_prefix: output file prefix (without extension)
        title_suffix: optional suffix for title
    """
    # Filter to first 5 seconds only
    time_limit = 5.0
    mask = t <= time_limit
    t = t[mask]
    u = u[mask]
    y = y[mask]

    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Input signal
    ax1.plot(t, u, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time [s]', fontsize=16)
    ax1.set_ylabel('Input Signal', fontsize=16)
    ax1.set_title('Input', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    # Right subplot: Output signal
    ax2.plot(t, y, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time [s]', fontsize=16)
    ax2.set_ylabel('Output Signal', fontsize=16)
    ax2.set_title('Output', fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    plt.tight_layout()

    # Save as EPS (vector format, high quality for papers)
    eps_path = output_prefix + '.eps'
    plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved EPS: {eps_path}")

    # Save as PNG (raster format, for preview)
    png_path = output_prefix + '.png'
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved PNG: {png_path}")

    plt.close(fig)


def main():
    """Main function to generate paper figures."""

    print("="*70)
    print("Creating Publication-Quality Figures")
    print("="*70)

    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    # Figure 1: From input folder
    print("\n[1/2] Processing input folder MAT file...")
    input_folder = Path('input')
    mat_files = sorted(input_folder.glob('*.mat'))

    if not mat_files:
        print("  [ERROR] No MAT files found in 'input' folder")
    else:
        # Use the first MAT file
        mat_file = mat_files[0]
        print(f"  Using: {mat_file.name}")

        try:
            t, u, y = load_mat_data(mat_file)
            print(f"  Data loaded: {len(t)} samples, duration: {t[-1]-t[0]:.2f} s")

            output_prefix = output_dir / 'figure_input_data'
            create_paper_figure(t, u, y, str(output_prefix),
                              title_suffix=f"Input Folder ({mat_file.name})")

        except Exception as e:
            print(f"  [ERROR] Error processing {mat_file.name}: {str(e)}")

    # Figure 2: From Wave.mat
    print("\n[2/2] Processing Wave.mat...")
    wave_file = Path('Wave.mat')

    if not wave_file.exists():
        print("  [ERROR] Wave.mat not found")
    else:
        print(f"  Using: {wave_file.name}")

        try:
            t, u, y = load_mat_data(wave_file)
            print(f"  Data loaded: {len(t)} samples, duration: {t[-1]-t[0]:.2f} s")

            output_prefix = output_dir / 'figure_wave_data'
            create_paper_figure(t, u, y, str(output_prefix),
                              title_suffix="Wave.mat")

        except Exception as e:
            print(f"  [ERROR] Error processing Wave.mat: {str(e)}")

    print("\n" + "="*70)
    print(f"All figures saved to: {output_dir.absolute()}")
    print("="*70)
    print("\nOutput files:")
    for eps_file in sorted(output_dir.glob('*.eps')):
        print(f"  - {eps_file.name} (EPS - for papers)")
    for png_file in sorted(output_dir.glob('*.png')):
        print(f"  - {png_file.name} (PNG - for preview)")
    print()


if __name__ == '__main__':
    main()
