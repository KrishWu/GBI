"""Configuration parameters for R-Anode analysis.

This module defines the time region boundaries and other analysis
parameters used throughout the R-Anode workflow for gravitational wave
anomaly detection. The signal region is defined based on the expected
time windows where gravitational wave signals might occur.
"""

# Signal region boundaries in milliseconds  
# Based on gravitational wave detection time windows
SR_MIN = 75.0  # ms - Lower bound of signal region
SR_MAX = 225.0  # ms - Upper bound of signal region