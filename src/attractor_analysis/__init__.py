from pathlib import Path
import os

# Core exports
from .utils import *
from .profiling_compile import *

# Main class import (after package is installed)
try:
    from .analysis import SLDSAnalyzer
    __all__ = [
        'SLDSAnalyzer',
        'get_project_root', 
        'Plot2D', 
        'Plot3D',
        'loader', 
        'coordinate_choice', 
        'get_data_path', 
        'read_config',
        'plot_normalized_attractor', 
        'read_attractor_data',
        'profile_resources',
        'profile_to_logs'
    ]
except ImportError:
    # Graceful fallback for direct file execution
    __all__ = [
        'get_project_root', 
        'Plot2D', 
        'Plot3D',
        'loader', 
        'coordinate_choice', 
        'get_data_path', 
        'read_config',
        'plot_normalized_attractor', 
        'read_attractor_data',
        'profile_resources',
        'profile_to_logs'
    ]
