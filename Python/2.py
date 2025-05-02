"""
Моделирование оценки канала для систем OFDM
Реализует оценку канала LS/DFT с линейной/сплайновой интерполяцией
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
from typing import List, Tuple

# Constants
NFFT = 32          # FFT size
NG = NFFT // 8     # Cyclic prefix length
NOFDM = NFFT + NG  # Total OFDM symbol length
NSYM = 100         # Number of OFDM symbols
NPS = 4            # Pilot spacing
NP = NFFT // NPS   # Number of pilots per OFDM symbol
NBPS = 4           # Bits per symbol (16-QAM)
M = 2**NBPS        # Modulation order
SNR_DB = 30        # Signal-to-noise ratio in dB
ES = 1             # Signal energy
A = np.sqrt(3/2/(M-1)*ES)  # QAM normalization factor


def ls_channel_estimate(
    Y: np.ndarray,
    Xp: np.ndarray,
    pilot_loc: List[int],
    method: str = 'linear'
) -> np.ndarray:
    """
    Least Squares channel estimation with interpolation
    
    Args:
        Y: Received frequency domain signal
        Xp: Pilot symbols
        pilot_loc: Pilot locations
        method: Interpolation method ('linear' or 'spline')
    
    Returns:
        Estimated channel frequency response
    """
    pilot_est = Y[pilot_loc] / Xp  # LS estimate at pilot locations
    all_loc = np.arange(NFFT)
    
    if method == 'linear':
        H_est = np.interp(all_loc, pilot_loc, pilot_est)
    elif method == 'spline':
        tck = interpolate.splrep(pilot_loc, pilot_est, s=0)
        H_est = interpolate.splev(all_loc, tck, der=0)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return H_est


def mmse_channel_estimate(
    Y: np.ndarray,
    Xp: np.ndarray,
    pilot_loc: List[int],
    h_true: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    MMSE channel estimation (simplified implementation)
    
    Args:
        Y: Received frequency domain signal
        Xp: Pilot symbols
        pilot_loc: Pilot locations
        h_true: True channel impulse response
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Estimated channel frequency response
    """
    # Initial LS estimate
    H_LS = ls_channel_estimate(Y, Xp, pilot_loc, 'linear')
    
    # Convert to time domain and truncate to known channel length
    h_ls = np.fft.ifft(H_LS)
    h_mmse = h_ls[:len(h_true)]  # Keep only the known channel length
    
    # Convert back to frequency domain
    H_MMSE = np.fft.fft(h_mmse, NFFT)
    return H_MMSE


def generate_ofdm_symbol() -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate OFDM symbol with pilots
    
    Returns:
        Tuple containing:
        - Frequency domain symbols (including pilots)
        - Pilot locations
        - Data symbols (without pilots)
    """
    # Generate pilot sequence (BPSK)
    Xp = 2*(np.random.rand(NP) > 0.5) - 1
    
    # Generate data symbols (16-QAM)
    msg_int = np.random.randint(0, M, NFFT-NP)
    data = A * (2*msg_int - (M-1)) / np.sqrt(2*(M-1)/3)  # Normalized 16-QAM
    
    # Insert pilots into the OFDM symbol
    X = np.zeros(NFFT, dtype=complex)
    pilot_loc = []
    data_idx = 0
    
    for k in range(NFFT):
        if k % NPS == 0:
            X[k] = Xp[k//NPS]
            pilot_loc.append(k)
        else:
            X[k] = data[data_idx]
            data_idx += 1
    
    return X, pilot_loc, data


def apply_channel(
    xt: np.ndarray,
    snr_db: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply multipath channel and AWGN noise
    
    Args:
        xt: Time domain OFDM symbol with CP
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Tuple containing:
        - Received time domain signal
        - Channel frequency response
        - Channel impulse response
    """
    # Create 2-tap multipath channel
    h = np.array([
        (np.random.randn() + 1j*np.random.randn()),
        (np.random.randn() + 1j*np.random.randn())/2
    ])
    H = np.fft.fft(h, NFFT)
    
    # Apply channel convolution
    y_channel = np.convolve(xt, h)
    
    # Add AWGN noise
    signal_power = np.mean(np.abs(y_channel)**2)
    noise_power = signal_power * 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(y_channel)) + 
        1j*np.random.randn(len(y_channel))
    )
    yt = y_channel + noise
    
    return yt, H, h


def simulate_channel_estimation() -> None:
    """Main simulation function"""
    # Initialize performance metrics
    mse = np.zeros(6)  # For LS-linear, LS-spline, MMSE (with and without DFT)
    bit_errors = 0
    total_bits = 0
    
    # Create figure for results
    plt.figure(figsize=(12, 16))
    plt.suptitle('Channel Estimation Performance Comparison', fontsize=14)
    
    for nsym in range(NSYM):
        # Generate OFDM symbol
        X, pilot_loc, data = generate_ofdm_symbol()
        
        # Convert to time domain and add cyclic prefix
        x = np.fft.ifft(X)
        xt = np.concatenate([x[-NG:], x])  # Add CP
        
        # Apply channel and noise
        yt, H_true, h_true = apply_channel(xt, SNR_DB)
        
        # Remove CP and convert to frequency domain
        y = yt[NG:NG+NFFT]
        Y = np.fft.fft(y)
        
        # Channel estimation methods
        methods = [
            ('LS-linear', 'linear'),
            ('LS-spline', 'spline'),
            ('MMSE', None)
        ]
        
        for m, (method_name, interp_method) in enumerate(methods):
            if method_name == 'MMSE':
                H_est = mmse_channel_estimate(Y, X[pilot_loc], pilot_loc, h_true, SNR_DB)
            else:
                H_est = ls_channel_estimate(Y, X[pilot_loc], pilot_loc, interp_method)
            
            # Calculate channel power in dB
            H_true_power = 10*np.log10(np.abs(H_true)**2)
            H_est_power = 10*np.log10(np.abs(H_est)**2)
            
            # DFT-based channel estimation
            h_est = np.fft.ifft(H_est)
            h_dft = h_est[:len(h_true)]
            H_dft = np.fft.fft(h_dft, NFFT)
            H_dft_power = 10*np.log10(np.abs(H_dft)**2)
            
            # Plot first symbol results
            if nsym == 0:
                plt.subplot(3, 2, 2*m+1)
                plt.plot(H_true_power, 'b', label='True Channel')
                plt.plot(H_est_power, 'r:+', label=method_name)
                plt.legend()
                plt.title(f'{method_name} Channel Estimation')
                plt.ylabel('Power (dB)')
                
                plt.subplot(3, 2, 2*m+2)
                plt.plot(H_true_power, 'b', label='True Channel')
                plt.plot(H_dft_power, 'r:+', label=f'{method_name} with DFT')
                plt.legend()
                plt.title(f'{method_name} with DFT')
                plt.ylabel('Power (dB)')
            
            # Update MSE metrics
            mse[m] += np.sum(np.abs(H_true - H_est)**2)
            mse[m+3] += np.sum(np.abs(H_true - H_dft)**2)
        
        # Equalization and data detection (using best estimate - MMSE)
        H_est = mmse_channel_estimate(Y, X[pilot_loc], pilot_loc, h_true, SNR_DB)
        Y_eq = Y / H_est
        
        # Extract data symbols (skip pilots)
        data_est = np.array([Y_eq[k] for k in range(NFFT) if k % NPS != 0])
        
        # Simple 16-QAM demodulation (hard decision)
        constellation = A * (2*np.arange(M) - (M-1)) / np.sqrt(2*(M-1)/3)
        distances = np.abs(data_est[:, np.newaxis] - constellation)
        msg_detected = np.argmin(distances, axis=1)
        
        # Calculate original data symbols for comparison
        msg_original = ((data.real > 0).astype(int) * 2 + (data.imag > 0).astype(int))
        
        # Update error statistics
        bit_errors += np.sum(msg_detected != msg_original)
        total_bits += len(msg_original) * NBPS
    
    # Calculate final metrics
    mse /= (NFFT * NSYM)
    ber = bit_errors / total_bits
    
    print("\nSimulation Results:")
    print(f"LS-linear MSE: {mse[0]:.4f} (with DFT: {mse[3]:.4f})")
    print(f"LS-spline MSE: {mse[1]:.4f} (with DFT: {mse[4]:.4f})")
    print(f"MMSE MSE: {mse[2]:.4f} (with DFT: {mse[5]:.4f})")
    print(f"Bit Error Rate: {ber:.6f}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    simulate_channel_estimation()