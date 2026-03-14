#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Domain Acoustic Descriptor Profiling for Expressive Vocal Signal Analysis
================================================================================
Corresponding author: Kishore Dutta
Paper: IEEE Signal Processing Letters (under review)
Repository: https://github.com/username/vocal-descriptor-profiling

This implementation accompanies the manuscript:
"Multi-Domain Acoustic Descriptor Profiling for Expressive Vocal Signal Analysis"

Requirements:
    - Python 3.7+
    - librosa
    - numpy
    - scipy
    - matplotlib
    - parselmouth (Praat integration)
    - tqdm (optional, for progress bars)

Usage:
    python vocal_descriptor_profiling.py
    
Expected audio files in working directory:
    - Ya-Ali-vocal.wav
    - Zubeen-speech.wav  
    - Gregorian-chant.wav
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, filtfilt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================
# Constants and Configuration
# ==============================================================

# Audio parameters
SAMPLE_RATE = 44100
HOP_LENGTH = 512
FFT_SIZE = 2048
PRE_EMPHASIS_ALPHA = 0.97

# Pitch estimation parameters
F0_MIN = 75  # Hz
F0_MAX = 400  # Hz

# Vibrato analysis parameters
VIBRATO_MIN = 4.0  # Hz
VIBRATO_MAX = 8.0  # Hz
VIBRATO_MIN_FRAMES = 50

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_CONFIDENCE = 95
MIN_FRAMES_FOR_STATS = 5

# Roughness parameters
ROUGHNESS_PEAK_HEIGHT = 0.2  # 20% of max
ROUGHNESS_PEAK_DISTANCE = 3  # bins

# Output settings
FIGURE_DPI = 300
FIGURE_SIZE = (16, 18)

# ==============================================================
# Spectral Descriptors
# ==============================================================

def spectral_roughness(S: np.ndarray, freqs: np.ndarray) -> float:
    """
    Compute spectral roughness using Vassilakis' psychoacoustic model.
    
    Parameters
    ----------
    S : np.ndarray
        Magnitude spectrogram (frequencies × frames)
    freqs : np.ndarray
        Frequency values for each bin (Hz)
        
    Returns
    -------
    float
        Mean roughness across all voiced frames
        
    References
    ----------
    Vassilakis, P. N., & Kendall, R. A. (2010). Psychoacoustic and cognitive
    aspects of auditory roughness: Definitions, models, and applications.
    """
    rough_frames = []
    
    for frame_idx, frame in enumerate(S.T):
        # Find spectral peaks
        peaks, properties = find_peaks(
            frame, 
            height=np.max(frame) * ROUGHNESS_PEAK_HEIGHT,
            distance=ROUGHNESS_PEAK_DISTANCE
        )
        
        if len(peaks) < 2:
            continue
            
        amps = frame[peaks]
        peak_freqs = freqs[peaks]
        
        # Compute roughness contribution from all peak pairs
        frame_roughness = 0.0
        for i in range(len(peak_freqs)):
            for j in range(i + 1, len(peak_freqs)):
                fd = abs(peak_freqs[j] - peak_freqs[i])
                s = 0.24 / (0.021 * min(peak_freqs[i], peak_freqs[j]) + 19)
                z = np.exp(-3.5 * s * fd) - np.exp(-5.75 * s * fd)
                frame_roughness += (amps[i] * amps[j]) ** 0.1 * z
                
        rough_frames.append(frame_roughness)
    
    return float(np.mean(rough_frames)) if rough_frames else 0.0


# ==============================================================
# Phonatory Perturbation
# ==============================================================

def jitter_praat(audio_path: str) -> float:
    """
    Compute local jitter using Praat's point process method.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
        
    Returns
    -------
    float
        Local jitter percentage
        
    Notes
    -----
    Uses Praat's default parameters:
    - Floor: 75 Hz
    - Ceiling: 500 Hz
    - Period floor: 0.0001 s
    - Period ceiling: 0.02 s
    - Max period factor: 1.3
    """
    try:
        snd = parselmouth.Sound(audio_path)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(
            point_process, 
            "Get jitter (local)", 
            0, 0, 0.0001, 0.02, 1.3
        )
        return float(jitter)
    except Exception as e:
        print(f"Warning: Praat jitter computation failed: {e}")
        return np.nan


def shimmer_local(y: np.ndarray, sr: int, f0: np.ndarray) -> float:
    """
    Compute local shimmer from glottal cycle peak amplitudes.
    
    Parameters
    ----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate (Hz)
    f0 : np.ndarray
        Framewise fundamental frequency contour
        
    Returns
    -------
    float
        Local shimmer percentage
    """
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 3:
        return np.nan
    
    # Estimate cycle periods in samples
    periods = (sr / f0_clean).astype(int)
    
    # Extract peak amplitude per cycle
    amplitudes = []
    sample_idx = 0
    for period in periods:
        if sample_idx + period < len(y):
            cycle_samples = y[sample_idx:sample_idx + period]
            amplitudes.append(np.max(np.abs(cycle_samples)))
        sample_idx += period
    
    amplitudes = np.array(amplitudes)
    if len(amplitudes) < 3:
        return np.nan
    
    # Compute shimmer as mean absolute difference of consecutive amplitudes
    # normalized by mean amplitude
    shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
    return float(shimmer)


# ==============================================================
# Vibrato Analysis
# ==============================================================

def analyze_vibrato(f0, sr, hop):
    """
    Robust vibrato estimation from F0 contour.
    Returns (rate_Hz, extent_semitones) or (nan, nan)
    """

    f0 = f0[~np.isnan(f0)]

    if len(f0) < VIBRATO_MIN_FRAMES:
        return np.nan, np.nan

    # --- convert to cents relative to median pitch ---
    cents = 1200 * np.log2(f0 / np.median(f0))

    # remove slow pitch drift
    cents = signal.detrend(cents)

    # modulation sampling rate
    fs_mod = sr / hop

    # --- power spectrum of pitch modulation ---
    f, P = signal.welch(
        cents,
        fs=fs_mod,
        nperseg=min(256, len(cents))
    )

    # vibrato band
    band = (f >= VIBRATO_MIN) & (f <= VIBRATO_MAX)

    if not np.any(band):
        return np.nan, np.nan

    # peak in vibrato band
    peak_idx = np.argmax(P[band])
    vibrato_freq = f[band][peak_idx]
    peak_power = P[band][peak_idx]

    # noise floor outside vibrato band
    noise_floor = np.median(P[~band])

    # --- robustness checks ---
    if peak_power < 5 * noise_floor:
        return np.nan, np.nan

    # vibrato extent from band power
    extent_cents = 2 * np.sqrt(peak_power)
    extent_semitones = extent_cents / 100

    if extent_semitones < 0.2:
        return np.nan, np.nan

    if vibrato_freq < 3 or vibrato_freq > 8:
        return np.nan, np.nan

    return float(vibrato_freq), float(extent_semitones)

# ==============================================================
# Statistical Utilities
# ==============================================================

def cohens_d(a,b):
    """
    Compute Cohen's d effect size for two independent samples.
    
    Parameters
    ----------
    a, b : np.ndarray
        Sample arrays
        
    Returns
    -------
    float
        Cohen's d (positive if mean(a) > mean(b))
        
    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    a,b = np.array(a), np.array(b)
    a,b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a)>1 and len(b)>1:
        s = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
        return (np.mean(a)-np.mean(b))/s if s!=0 else 0
    am,bm = np.nanmean(a),np.nanmean(b)
    denom=np.nanstd([am,bm])
    return (am-bm)/denom if denom>0 else 0


def bootstrap_ci(
    data: np.ndarray, 
    n_boot: int = BOOTSTRAP_ITERATIONS, 
    ci: int = BOOTSTRAP_CONFIDENCE
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_boot : int
        Number of bootstrap iterations
    ci : int
        Confidence level (e.g., 95)
        
    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
        
    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) < MIN_FRAMES_FOR_STATS:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha)
    upper = np.percentile(bootstrap_means, 100 - alpha)
    
    return float(lower), float(upper)


# ==============================================================
# Main Feature Extraction
# ==============================================================

def extract_features(audio_path: str) -> Tuple[Dict, Dict]:
    """
    Extract all acoustic descriptors from audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
        
    Returns
    -------
    Tuple[Dict, Dict]
        (summary_features, framewise_trajectories)
        
    Notes
    -----
    Implements the full descriptor set from the manuscript:
    - Spectral: centroid, bandwidth, rolloff, flux, roughness
    - Cepstral: MFCCs (1-13, omitting 0th)
    - Perturbation: jitter, shimmer
    - Modulation: vibrato rate, vibrato extent
    - Dynamic: RMS, dynamic range, HNR
    """
    # Load and preprocess audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = y / np.max(np.abs(y))
    

    
    # STFT
    S = np.abs(librosa.stft(
        y, 
        n_fft=FFT_SIZE, 
        hop_length=HOP_LENGTH,
        window='hann'
    ))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    
    # Spectral descriptors
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    
    # Spectral flux (L2 norm of spectral differences)
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)) / S.shape[0]
    
    # Spectral roughness
    roughness = spectral_roughness(S, freqs)
    
    # MFCCs (omit 0th coefficient)
    mfcc_full = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=13,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )
    mfcc = mfcc_full[1:, :]  # Remove 0th coefficient
    
    # Pitch and perturbation
    f0, voiced_flag, _ = librosa.pyin(
        y, 
        fmin=F0_MIN, 
        fmax=F0_MAX,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    
    # Jitter and shimmer
    jitter = jitter_praat(audio_path)
    shimmer = shimmer_local(y, sr, f0)
    
    # Vibrato
    vibrato_rate, vibrato_extent = analyze_vibrato(f0, sr, HOP_LENGTH)
    
    # RMS and dynamic range
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    dynamic_range = 20 * np.log10(
        np.max(rms) / (max(np.min(rms), 1e-6))
    )
    energy_variability = np.std(rms)
    
    # Harmonics-to-noise ratio
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hnr = 10 * np.log10(
        np.mean(harmonic ** 2) / (np.mean(percussive ** 2) + 1e-6)
    )
    
    # Tempo and timing variability (optional)
    tempo, beats = librosa.beat.beat_track(
        y=y, 
        sr=sr, 
        hop_length=HOP_LENGTH
    )
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    beat_intervals = np.diff(beat_times)
    timing_variability = (
        np.std(beat_intervals) / np.mean(beat_intervals) 
        if len(beat_intervals) > 1 else 0
    )
    
    # Summary statistics
    features = {
        'centroid': float(np.mean(centroid)),
        'bandwidth': float(np.mean(bandwidth)),
        'rolloff': float(np.mean(rolloff)),
        'roughness': roughness,
        'vibrato_rate': vibrato_rate,
        'vibrato_extent': vibrato_extent,
        'hnr': hnr,
        'dynamic_range': dynamic_range,
        'energy_variability': energy_variability,
        'timing_variability': timing_variability,
        'jitter': jitter,
        'shimmer': shimmer,
        'mfcc': np.mean(mfcc, axis=1).tolist(),
        'tempo': tempo
    }
    
    # Framewise trajectories for statistical analysis
    framewise = {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff': rolloff,
        'flux': flux,
        'rms': rms,
        'f0': f0,
        'mfcc': mfcc
    }
    
    return features, framewise


# ==============================================================
# Visualization
# ==============================================================

def plot_single_analysis(audio_path: str, features: Dict, save: bool = True):
    """
    Generate multi-panel analysis plot for a single audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    features : Dict
        Feature dictionary from extract_features()
    save : bool
        Whether to save the figure
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = y / np.max(np.abs(y))
    
    S = np.abs(librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Compute trajectories for plotting
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    
    # Create figure
    plt.figure(figsize=FIGURE_SIZE)
    
    # 1. Spectrogram
    plt.subplot(5, 1, 1)
    librosa.display.specshow(
        S_db, 
        sr=sr, 
        hop_length=HOP_LENGTH,
        y_axis='log', 
        x_axis='time',
        cmap='magma'
    )
    plt.title(f'Spectrogram – {Path(audio_path).name}', fontsize=16, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')
    
    # Annotate key metrics
    vibrato_rate = features["vibrato_rate"]
    vibrato_text = 'none'
    if not np.isnan(vibrato_rate):
        vibrato_text = f'{vibrato_rate:.1f} Hz'

    annotation = (
        f'Vibrato: {vibrato_text}\n'
        f'HNR: {features["hnr"]:.1f} dB\n'
        f'Dynamic Range: {features["dynamic_range"]:.1f} dB'
    )

    plt.annotate(
        annotation, 
        xy=(0.02, 0.85), 
        xycoords='axes fraction',
        bbox=dict(
            boxstyle='round,pad=0.3', 
            facecolor='white', 
            alpha=0.9, 
            edgecolor='black'
        ),
        fontsize=10, 
        fontfamily='serif'
    )
    
    # 2. MFCC profile
    plt.subplot(5, 1, 2)
    mfcc_mean = features['mfcc']
    bars = plt.bar(
        range(1, len(mfcc_mean) + 1), 
        mfcc_mean, 
        color='teal', 
        alpha=0.7, 
        edgecolor='navy', 
        linewidth=0.5
    )
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('MFCC Profile (Coefficients 1-12)', fontsize=14, fontweight='bold')
    plt.xlabel('MFCC Coefficient Index', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 3. Expressivity metrics
    plt.subplot(5, 1, 3)
    metric_names = ['Roughness', 'Vibrato', 'HNR', 'Dyn Range', 'Timing Var', 'Jitter', 'Shimmer']
    metric_values = [
        features['roughness'],
        features['vibrato_rate'],
        features['hnr'],
        features['dynamic_range'],
        features['timing_variability'] * 100,
        features['jitter'],
        features['shimmer']
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#A29BFE']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title('Expressivity Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=12)
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        if not np.isnan(val):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + max(metric_values)*0.01,
                f'{val:.2f}', 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                fontweight='bold'
            )
    
    # 4. Spectral evolution
    plt.subplot(5, 1, 4)
    time_axis = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=HOP_LENGTH)
    plt.plot(time_axis, centroid, label='Spectral Centroid', color='orange', linewidth=1.5)
    plt.plot(time_axis, bandwidth, label='Spectral Bandwidth', color='blue', linewidth=1.5, alpha=0.7)
    plt.axhline(np.mean(centroid), color='orange', linestyle='--', alpha=0.6, 
                label=f'Centroid Avg: {np.mean(centroid):.0f} Hz')
    plt.axhline(np.mean(bandwidth), color='blue', linestyle='--', alpha=0.6, 
                label=f'Bandwidth Avg: {np.mean(bandwidth):.0f} Hz')
    plt.legend(fontsize=10, framealpha=0.9)
    plt.title('Spectral Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    
    # 5. Dynamic expression
    plt.subplot(5, 1, 5)
    rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
    rms_db = 20 * np.log10(rms + 1e-6)
    plt.plot(rms_time, rms_db, color='purple', linewidth=2, label='Energy Envelope')
    plt.axhline(np.mean(rms_db), color='purple', linestyle='--', 
                label=f'Avg Level: {np.mean(rms_db):.1f} dB')
    plt.title('Dynamic Expression Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude (dB)', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = f"{Path(audio_path).stem}-analysis.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_comparison_bar(
    features_ya: Dict, 
    features_sp: Dict, 
    features_gr: Dict,
    save: bool = True
):
    """
    Generate comparative bar plot for all three signals.
    
    Parameters
    ----------
    features_ya, features_sp, features_gr : Dict
        Feature dictionaries for each signal
    save : bool
        Whether to save the figure
    """
    # Define feature groups
    large_scale_keys = ['centroid', 'bandwidth']
    small_scale_keys = ['roughness', 'vibrato_rate', 'hnr', 'dynamic_range', 'jitter', 'shimmer']
    
    label_map = {
        'centroid': 'CT (Hz)',
        'bandwidth': 'BW (Hz)',
        'roughness': 'RGH',
        'vibrato_rate': 'VR (Hz)',
        'hnr': 'HNR (dB)',
        'jitter': 'JIT (%)',
        'shimmer': 'SHM (%)',
        'dynamic_range': 'DR (dB)'
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top subplot: large-scale features
    x_labels = []
    ya_vals = []
    sp_vals = []
    gr_vals = []
    
    for key in large_scale_keys:
        base_label = label_map[key]
        x_labels.extend([f'{base_label}-YA', f'{base_label}-SP', f'{base_label}-GR'])
        ya_vals.append(features_ya[key])
        sp_vals.append(features_sp[key])
        gr_vals.append(features_gr[key])
    
    x = np.arange(len(large_scale_keys))
    width = 0.25
    
    ax1.bar(x - width, ya_vals, width, label='YA', color='#FF6B6B')
    ax1.bar(x, sp_vals, width, label='SP', color='#4ECDC4')
    ax1.bar(x + width, gr_vals, width, label='GR', color='#45B7D1')
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Large-Scale Spectral Features', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map[k] for k in large_scale_keys], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom subplot: small-scale features
    x_labels = []
    ya_vals = []
    sp_vals = []
    gr_vals = []
    
    for key in small_scale_keys:
        base_label = label_map[key]
        x_labels.extend([f'{base_label}-YA', f'{base_label}-SP', f'{base_label}-GR'])
        ya_vals.append(features_ya[key])
        sp_vals.append(features_sp[key])
        gr_vals.append(features_gr[key])
    
    x = np.arange(len(small_scale_keys))
    
    ax2.bar(x - width, ya_vals, width, label='YA', color='#FF6B6B')
    ax2.bar(x, sp_vals, width, label='SP', color='#4ECDC4')
    ax2.bar(x + width, gr_vals, width, label='GR', color='#45B7D1')
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Perturbation and Modulation Features', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([label_map[k] for k in small_scale_keys], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('comparison-plot.png', dpi=FIGURE_DPI, bbox_inches='tight')
        print("Saved: comparison-plot.png")
    
    plt.show()


# ==============================================================
# Main Analysis Pipeline
# ==============================================================

def validate_audio_files(file_paths: List[str]) -> bool:
    """Check if all required audio files exist."""
    missing = [f for f in file_paths if not Path(f).exists()]
    if missing:
        print("Error: Missing audio files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease ensure the following files are in the working directory:")
        for f in file_paths:
            print(f"  - {f}")
        return False
    return True


def print_statistical_summary(features_ya, features_sp, features_gr, framewise_ya, framewise_sp, framewise_gr):
    """Print formatted statistical summary."""
    
    print("\n" + "="*60)
    print("MULTI-DOMAIN ACOUSTIC DESCRIPTOR PROFILING")
    print("="*60)
    
    # Table I: Acoustic comparison
    print("\nTABLE I: Acoustic Descriptors Across Three Vocal Signals")
    print("-"*80)
    print(f"{'Descriptor':<25} {'YA':>15} {'SP':>15} {'GR':>15}")
    print("-"*80)
    
    descriptors = [
        ('Spectral Centroid (Hz)', 'centroid', '{:.0f}'),
        ('Spectral Bandwidth (Hz)', 'bandwidth', '{:.0f}'),
        ('Spectral Roughness', 'roughness', '{:.2f}'),
        ('Vibrato Rate (Hz)', 'vibrato_rate', '{:.2f}'),
        ('Vibrato Extent (Hz)', 'vibrato_extent', '{:.1f}'),
        ('Jitter (%)', 'jitter', '{:.2f}'),
        ('Shimmer (%)', 'shimmer', '{:.2f}'),
        ('HNR (dB)', 'hnr', '{:.2f}'),
        ('Dynamic Range (dB)', 'dynamic_range', '{:.2f}')
    ]
    
    for label, key, fmt in descriptors:
        ya_val = features_ya[key] if not np.isnan(features_ya[key]) else 0
        sp_val = features_sp[key] if not np.isnan(features_sp[key]) else 0
        gr_val = features_gr[key] if not np.isnan(features_gr[key]) else 0
        
        ya_str = fmt.format(ya_val) if not np.isnan(features_ya[key]) else 'n/a'
        sp_str = fmt.format(sp_val) if not np.isnan(features_sp[key]) else 'n/a'
        gr_str = fmt.format(gr_val) if not np.isnan(features_gr[key]) else 'n/a'
        
        print(f"{label:<25} {ya_str:>15} {sp_str:>15} {gr_str:>15}")
    
    # Table II: Framewise variability
    print("\n" + "="*60)
    print("TABLE II: Framewise Statistical Variability (Mean ± SD; 95% CI)")
    print("="*60)
    
    signals = {'YA': framewise_ya, 'SP': framewise_sp, 'GR': framewise_gr}
    metrics = ['centroid', 'bandwidth', 'rolloff', 'flux', 'rms']
    
    for sig_name, fw in signals.items():
        print(f"\n-- {sig_name} --")
        for metric in metrics:
            arr = fw[metric]
            arr = arr[~np.isnan(arr)]
            if len(arr) < MIN_FRAMES_FOR_STATS:
                print(f"  {metric:12}: insufficient data")
                continue
            
            mean_val = np.mean(arr)
            std_val = np.std(arr, ddof=1)
            ci_low, ci_high = bootstrap_ci(arr)
            
            unit = 'Hz' if metric in ['centroid', 'bandwidth', 'rolloff'] else ''
            unit = '' if metric == 'flux' else unit
            unit = '' if metric == 'rms' else unit
            
            print(f"  {metric:12}: {mean_val:.1f} ± {std_val:.1f} {unit}")
            print(f"             95% CI: [{ci_low:.1f}, {ci_high:.1f}] {unit}")
    
    # Effect sizes
    print("\n" + "="*60)
    print("Cohen's d Effect Sizes (YA vs. SP, YA vs. GR)")
    print("="*60)
    print(f"{'Descriptor':<20} {'YA vs SP':>15} {'YA vs GR':>15} {'Interpretation':>20}")
    print("-"*75)
    
    # Compute effect sizes
    frame_keys = ['centroid', 'bandwidth', 'rolloff', 'flux', 'rms']
    for key in frame_keys:
        d_sp = cohens_d(framewise_ya[key], framewise_sp[key])
        d_gr = cohens_d(framewise_ya[key], framewise_gr[key])
        
        def interpret(d):
            if abs(d) >= 0.8:
                return "large"
            elif abs(d) >= 0.5:
                return "medium"
            elif abs(d) >= 0.2:
                return "small"
            else:
                return "negligible"
        
        print(f"{key:<20} {d_sp:>15.2f} {d_gr:>15.2f} {interpret(d_gr):>20}")
    
    print("\n" + "="*60)


def main():
    """Main execution function."""
    
    # Define audio files
    audio_files = [
        "Ya-Ali-vocal.wav",
        "Zubeen-speech.wav", 
        "Gregorian-chant.wav"
    ]
    
    # Validate files
    if not validate_audio_files(audio_files):
        sys.exit(1)
    
    print("\nProcessing audio files...")
    
    # Extract features
    features_ya, framewise_ya = extract_features(audio_files[0])
    print(f"✓ Processed: {audio_files[0]}")
    
    features_sp, framewise_sp = extract_features(audio_files[1])
    print(f"✓ Processed: {audio_files[1]}")
    
    features_gr, framewise_gr = extract_features(audio_files[2])
    print(f"✓ Processed: {audio_files[2]}")
    
    # Print statistical summary
    print_statistical_summary(
        features_ya, features_sp, features_gr,
        framewise_ya, framewise_sp, framewise_gr
    )
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_single_analysis(audio_files[0], features_ya, save=True)
    plot_single_analysis(audio_files[1], features_sp, save=True)
    plot_single_analysis(audio_files[2], features_gr, save=True)
    
    plot_comparison_bar(features_ya, features_sp, features_gr, save=True)
    
    print("\n Analysis complete!")
    print("Output files generated:")
    print("  - Ya-Ali-vocal_analysis.png")
    print("  - Zubeen-speech_analysis.png")
    print("  - Gregorian-chant_analysis.png")
    print("  - comparison_bar_plot.png")


# ==============================================================
# Entry Point
# ==============================================================

if __name__ == "__main__":
    main()
