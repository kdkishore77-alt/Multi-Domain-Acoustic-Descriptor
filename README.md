# Multi-Domain Acoustic Descriptor Profiling Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1109/LSP.2026.XXXXX-blue)](https://doi.org/10.1109/LSP.2026.XXXXX)

This repository contains the official Python implementation of the multi-domain acoustic descriptor profiling framework introduced in:

> K. Dutta, "Multi-Domain Acoustic Descriptor Profiling for Expressive Vocal Signal Analysis," *IEEE Signal Processing Letters*, vol. XX, no. X, pp. XX-XX, 2026. doi: 10.1109/LSP.2026.XXXXX

## 📋 Overview

The framework enables reproducible, statistically grounded analysis of vocal signals by integrating spectral, cepstral, perturbation, modulation, and dynamic descriptors within a unified processing pipeline. Unlike traditional feature extraction toolkits that output raw framewise values, this implementation generates comprehensive **descriptor profiles** with bootstrap confidence intervals and effect-size comparisons, facilitating quantitative comparison across vocal conditions.

![Framework Overview](docs/figures/framework.png)
*Figure 1: Multi-domain acoustic descriptor profiling pipeline*

## ✨ Key Features

| Domain | Descriptors | Applications |
|--------|-------------|--------------|
| **Spectral** | Centroid, bandwidth, rolloff, flux, roughness | Timbre characterization, spectral shape analysis |
| **Cepstral** | MFCCs (1-12) | Spectral envelope, formant structure |
| **Perturbation** | Jitter (%), shimmer (%) | Phonatory stability, voice quality |
| **Modulation** | Vibrato rate (Hz), vibrato extent (Hz) | Pitch modulation, expressive singing |
| **Dynamic** | HNR (dB), dynamic range (dB) | Harmonic content, amplitude variation |

### Core Capabilities

- **Multi-domain descriptor extraction**: 20+ acoustic descriptors computed framewise
- **Statistical profiling**: Moment statistics (mean, std) with 95% bootstrap confidence intervals (500 iterations)
- **Comparative analysis**: Cohen's *d* effect size with pooled standard deviation for pairwise signal comparisons
- **Vocal isolation**: Integration with harmonic-percussive source separation and Spleeter for vocal extraction
- **Visualization utilities**: Trajectory plots, MFCC profiles, comparative bar charts, spectrograms

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-domain-acoustic-profiling.git
cd multi-domain-acoustic-profiling

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
