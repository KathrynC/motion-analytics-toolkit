"""Signal processing utilities for motion analysis."""

import numpy as np
from scipy import signal
from typing import Tuple, Optional

def compute_phase_difference(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute average phase difference between two signals using Hilbert transform."""
    analytic1 = signal.hilbert(sig1 - np.mean(sig1))
    analytic2 = signal.hilbert(sig2 - np.mean(sig2))
    
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # Mean phase difference as complex number to avoid wrap-around issues
    phase_diff = np.mean(np.exp(1j * (phase1 - phase2)))
    return float(np.angle(phase_diff))

def compute_phase_locking_value(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute phase locking value (0 to 1) between two signals."""
    analytic1 = signal.hilbert(sig1 - np.mean(sig1))
    analytic2 = signal.hilbert(sig2 - np.mean(sig2))
    
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # PLV = |mean(e^(i*(φ1-φ2)))|
    plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
    return float(plv)

def compute_spectral_arc_length(velocity: np.ndarray, sampling_rate: float) -> float:
    """
    Compute smoothness via spectral arc length metric.
    
    Higher values = less smooth (more high-frequency content).
    """
    freqs = np.fft.rfftfreq(len(velocity), 1/sampling_rate)
    mag = np.abs(np.fft.rfft(velocity))
    
    # Normalize by DC component
    if mag[0] > 0:
        mag = mag / mag[0]
    
    # Arc length of normalized magnitude spectrum
    arc_length = np.sum(np.sqrt(np.diff(freqs)**2 + np.diff(mag)**2))
    return float(arc_length)

def compute_dimensionless_jerk(
    position: np.ndarray, 
    sampling_rate: float
) -> float:
    """
    Compute normalized jerk (smoothness metric).
    Lower values = smoother movement.
    """
    velocity = np.gradient(position, 1/sampling_rate)
    acceleration = np.gradient(velocity, 1/sampling_rate)
    jerk = np.gradient(acceleration, 1/sampling_rate)
    
    # Normalize by movement duration and amplitude
    duration = len(position) / sampling_rate
    amplitude = np.ptp(position)
    
    if amplitude < 1e-6:
        return 0.0
    
    jerk_magnitude = np.sqrt(np.mean(jerk**2))
    norm_jerk = jerk_magnitude * duration**3 / amplitude
    
    return float(norm_jerk)

def detect_peaks_with_prominence(
    signal: np.ndarray,
    sampling_rate: float,
    min_distance: float = 0.1,  # seconds
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Detect peaks in signal with minimum time between peaks.
    
    Returns:
        peak_indices: indices of detected peaks
        properties: dictionary with peak properties
    """
    min_distance_samples = int(min_distance * sampling_rate)
    
    if prominence is None:
        prominence = 0.5 * np.std(signal)
    
    peaks, properties = signal.find_peaks(
        signal,
        distance=min_distance_samples,
        prominence=prominence
    )
    
    return peaks, properties
