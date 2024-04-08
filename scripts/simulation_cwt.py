"""
Continuous wavelet transform of simulated signal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet

from simulation_consumption import consumption

WAVELET_WIDTH = np.arange(1, 31)
WAVELET = morlet


def main() -> None:
    """Run script"""
    i_values = np.linspace(1, 512, 512)
    consumption_values = consumption(i_values)
    cwt_result = cwt(consumption_values, WAVELET, WAVELET_WIDTH)

    # Plot the CWT result
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(cwt_result),
        extent=[-1, 1, 1, 31],
        cmap="jet",
        aspect="auto",
        vmax=np.abs(cwt_result).max(),
        vmin=-np.abs(cwt_result).max(),
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Continuous Wavelet Transform")
    plt.show()


if __name__ == "__main__":
    main()
