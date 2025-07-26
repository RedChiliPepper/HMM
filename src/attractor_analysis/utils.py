import sys
import yaml
import numpy as np
import time
import os
import reservoirpy.datasets as rsvp_d
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from .profiling_compile import *


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def coordinate_choice(attractor, xxn, yyn, zzn):
    """Select coordinate based on attractor type.
    
    Args:
        attractor: One of "Lorenz", "Rossler", "DoubleScroll", "MultiScroll", or "RabinovichFabrikant"
        xxn: x coordinate value
        yyn: y coordinate value
        zzn: z coordinate value
        
    Returns:
        Selected coordinate value
        
    Raises:
        ValueError: If attractor type is invalid
    """
    match attractor:
        case "Lorenz":
            return xxn
        case "Rossler":
            return zzn
        case "DoubleScroll" | "MultiScroll" | "RabinovichFabrikant":
            return xxn  # or choose appropriate coordinate for each
        case _:
            raise ValueError(f"Unknown attractor type: {attractor}. Must be one of: 'Lorenz', 'Rossler', 'DoubleScroll', 'MultiScroll', 'RabinovichFabrikant'")


"""
def read_config(file):
    # Read YAML configuration file
    #filepath = #'../../experiments/config.yaml'
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs
"""

@profile_to_logs(log_dir="logs/line_profiles")
def read_config(file):
    """Read YAML configuration file"""
    filepath = Path(file)
    
    # Check if file exists as-is
    if filepath.exists():
        with open(filepath, 'r') as stream:
            return yaml.safe_load(stream)
    
    # Check if file exists with .yaml extension
    filepath_yaml = filepath.with_suffix('.yaml')
    if filepath_yaml.exists():
        with open(filepath_yaml, 'r') as stream:
            return yaml.safe_load(stream)
    
    # If neither exists
    raise FileNotFoundError(
        f"Config file not found at either:\n"
        f"1. {filepath}\n"
        f"2. {filepath_yaml}"
    )

def read_attractor_data(filepath):
    # Read attractor data from file
    if filepath.endswith('.npy'):
        data = np.load(filepath)
    elif filepath.endswith('.csv'):
        data = np.loadtxt(filepath)
    else:
        raise ValueError("Unsupported file format")
    
    # Assuming data is in shape (3, n)
    return {
        'x': data[0],
        'y': data[1],
        'z': data[2]
    }

def read_attractor_data(filepath):
    """Read attractor data from file"""
    filepath = str(filepath)  # Convert Path to string if needed
    if filepath.endswith('.npy'):
        data = np.load(filepath)
    elif filepath.endswith('.csv'):
        data = np.loadtxt(filepath)
    else:
        raise ValueError("Unsupported file format")
    
    return {
        'x': data[0],
        'y': data[1],
        'z': data[2]
    }


def Plot2D(xs, ys, zs, titl1, titl2, titl3, start, pt_number, lw = 1, figsize = (15, 5)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = figsize)

    ax1.plot(xs[start:start+pt_number], ys[start:start+pt_number], lw=lw)
    ax1.set_title(titl1)
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")

    ax2.plot(xs[start:start+pt_number], zs[start:start+pt_number], lw=lw)
    ax2.set_title(titl2)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Z Axis")

    ax3.plot(ys[start:start+pt_number], zs[start:start+pt_number], lw=lw)
    ax3.set_title(titl3)
    ax3.set_xlabel("Y Axis")
    ax3.set_ylabel("Z Axis")

    plt.tight_layout()
    plt.show()

def Plot3D(xs, ys, zs, title, start, pt_number, lw = 1, figsize = (15, 5)):
    ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
    ax.plot(xs[start:start+pt_number], ys[start:start+pt_number], zs[start:start+pt_number], lw=lw)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

"""
def get_data_path(filename=""):
    # Get absolute path to data directory (sibling of src folder)
    root = get_project_root()
    os.makedirs(os.path.join(root, 'data'), exist_ok=True)
    return os.path.join(os.path.join(root, 'data'), filename)
"""

def get_data_path(filename=""):
    """Get path compliant with Python packaging standards"""
    # Try standard package data location first
    installed_path = Path(sys.prefix) / 'share' / 'attractor_analysis' / 'data' / filename
    if installed_path.exists():
        return str(installed_path)
    
    # Fallback to development location
    dev_path = get_project_root() / 'data' / filename
    dev_path.parent.mkdir(exist_ok=True)
    return str(dev_path)

@profile_to_logs(log_dir="logs/line_profiles")
def loader(attractor, formatt, save, **kwargs):
    """Load or generate attractor data with default parameters for all attractor types.
    
    Args:
        attractor: Type of attractor ("Lorenz", "Rossler", "DoubleScroll", etc.)
        formatt: File format (".npy" or ".csv")
        save: Whether to save the generated data
        **kwargs: Optional parameters to override defaults
        
    Returns:
        numpy.ndarray: Array containing the attractor data
    """
    # Default parameters for all attractors
    defaults = {
        "n": 1e4,          # Number of points
        "x0": np.array([0.1, 0.1, 0.1]),  # Initial conditions
        "h": 0.01,          # Time step
        "method": "RK45",    # Integration method
        "rtol": 1e-6,       # Relative tolerance
        "atol": 1e-8,       # Absolute tolerance
        # Lorenz defaults
        "params": [28.0, 10.0, 8/3],  # rho, sigma, beta
        # Rossler defaults
        "a": 0.2, "b": 0.2, "c": 5.7,
        # DoubleScroll defaults
        "r1": 1.2, "r2": 3.44, "r4": 0.193, "ir": 4.5e-05, "beta": 11.6,
        # MultiScroll defaults
        "a_ms": 40.0, "b_ms": 3.0, "c_ms": 28.0,  # Different names to avoid conflict
        # Rabinovich-Fabrikant defaults
        "alpha": 1.1, "gamma": 0.89
    }
    
    # Update defaults with any provided kwargs
    params = {**defaults, **kwargs}

    def convert_number(value):
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            value = value.strip()
            if '/' in value:  # Handle fractions
                num, den = value.split('/')
                return float(num) / float(den)
            try:
                if 'e' in value.lower():  # Scientific notation
                    return int(float(value))  # Convert to float first, then int
                if '.' in value:  # Float
                    return float(value)
                return int(value)  # Plain integer
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to number")
        raise TypeError(f"Unsupported type {type(value)} for conversion")
    
    # Convert all parameters
    try:
        params["n"] = int(convert_number(params["n"]))
        params["x0"] = np.array([convert_number(x) for x in params["x0"]], dtype=np.float64)
        params["h"] = convert_number(params["h"])
        params["rtol"] = convert_number(params["rtol"])
        params["atol"] = convert_number(params["atol"])
        params["params"] = [convert_number(p) for p in params["params"]]
        
        # Convert atol to array
        params["atol"] = np.full(3, params["atol"], dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Parameter conversion error: {str(e)}") from e
    
    # Debug print to verify types
    print("\nVerified parameter types:")
    print(f"n: {type(params['n'])}, {params['n']}")
    print(f"x0: {type(params['x0'])}, {params['x0'].dtype}")
    print(f"h: {type(params['h'])}, {params['h']}")
    print(f"rtol: {type(params['rtol'])}, {params['rtol']}")
    print(f"atol: {type(params['atol'])}, {params['atol'].dtype}")
    print(f"params: {[type(p) for p in params['params']]}")

    
    # Generate filename based on parameters
    match attractor:
        case "Lorenz":
            name = "Lorenz_rho_%s_sigma_%s_beta_%s_x_%s_y_%s_z_%s_h_%s_method_%s_rtol_%s_atol_%s_n_%s" % (
                params["params"][0], params["params"][1], params["params"][2],
                params["x0"][0], params["x0"][1], params["x0"][2],
                params["h"], params["method"], params["rtol"], params["atol"], params["n"])
        case "Rossler":
            name = "Rossler_a_%s_b_%s_c_%s_x_%s_y_%s_z_%s_h_%s_method_%s_rtol_%s_atol_%s_n_%s" % (
                params["a"], params["b"], params["c"],
                params["x0"][0], params["x0"][1], params["x0"][2],
                params["h"], params["method"], params["rtol"], params["atol"], params["n"])
        case "DoubleScroll":
            name = "DoubleScroll_r1_%s_r2_%s_r4_%s_ir_%s_beta_%s_x_%s_y_%s_z_%s_h_%s_n_%s" % (
                params["r1"], params["r2"], params["r4"],
                params["ir"], params["beta"],
                params["x0"][0], params["x0"][1], params["x0"][2],
                params["h"], params["n"])
        case "MultiScroll":
            name = "MultiScroll_a_%s_b_%s_c_%s_x_%s_y_%s_z_%s_h_%s_n_%s" % (
                params["a_ms"], params["b_ms"], params["c_ms"],
                params["x0"][0], params["x0"][1], params["x0"][2],
                params["h"], params["n"])
        case "RabinovichFabrikant":
            name = "RabinovichFabrikant_alpha_%s_gamma_%s_x_%s_y_%s_z_%s_h_%s_n_%s" % (
                params["alpha"], params["gamma"],
                params["x0"][0], params["x0"][1], params["x0"][2],
                params["h"], params["n"])
    
    # Get full path to data file
    filepath = get_data_path(name + formatt)
    print("filename:\t", filepath)
    
    if not os.path.exists(filepath):
        print('file not found, proceeding with computation')
        t1 = time.time()
        
        match attractor:
            case "Lorenz":
                print(f"\nDEBUG - Parameters before lorenz call:")
                print(f"rho type: {type(params['params'][0])}, value: {params['params'][0]}")
                print(f"sigma type: {type(params['params'][1])}, value: {params['params'][1]}")
                print(f"beta type: {type(params['params'][2])}, value: {params['params'][2]}")
                print(f"x0 type: {type(params['x0'])}, value: {params['x0']}")
                print(f"x0 dtype: {params['x0'].dtype if hasattr(params['x0'], 'dtype') else 'N/A'}")

                dataset = rsvp_d.lorenz(
                    params["n"], 
                    rho=params["params"][0], 
                    sigma=params["params"][1], 
                    beta=params["params"][2], 
                    x0=params["x0"], 
                    h=params["h"], 
                    method=params["method"], 
                    rtol=params["rtol"], 
                    atol=np.ones(3)*params["atol"]
                )
            case "Rossler":
                dataset = rsvp_d.rossler(
                    params["n"], 
                    a=params["a"], 
                    b=params["b"], 
                    c=params["c"], 
                    x0=params["x0"], 
                    h=params["h"], 
                    method=params["method"], 
                    rtol=params["rtol"], 
                    atol=params["atol"]
                )
            case "DoubleScroll":
                dataset = rsvp_d.doublescroll(
                    params["n"],
                    r1=params["r1"],
                    r2=params["r2"],
                    r4=params["r4"],
                    ir=params["ir"],
                    beta=params["beta"],
                    x0=params["x0"],
                    h=params["h"]
                )
            case "MultiScroll":
                dataset = rsvp_d.multiscroll(
                    params["n"],
                    a=params["a_ms"],
                    b=params["b_ms"],
                    c=params["c_ms"],
                    x0=params["x0"],
                    h=params["h"]
                )
            case "RabinovichFabrikant":
                dataset = rsvp_d.rabinovich_fabrikant(
                    params["n"],
                    alpha=params["alpha"],
                    gamma=params["gamma"],
                    x0=params["x0"],
                    h=params["h"]
                )
        
        dt = time.time() - t1
        print('computation took ', dt, ' seconds.')
        dataset = np.array(dataset).T
        
        if save:
            match formatt:
                case ".npy":
                    np.save(filepath, dataset)
                case ".csv":
                    np.savetxt(filepath, dataset, delimiter=",")
            print("file saved")
    else:
        print("file correctly loaded")
        match formatt:
            case ".npy":
                dataset = np.load(filepath)
            case ".csv":
                dataset = np.loadtxt(filepath)
    
    return dataset

def plot_normalized_attractor(data, colors=("y", "b", "r"), linewidth=0.1, 
                            arrow_scale=0.1, head_width=0.01, figsize=(15,10)):
    """
    Normalize, center, and plot attractor data with directional arrows.
    
    Args:
        data: Numpy array of shape (n_points, 3) containing x,y,z coordinates
        colors: Tuple of 3 colors for (xy, yz, zx) projections
        linewidth: Width of trajectory lines
        arrow_scale: Scaling factor for arrow lengths
        head_width: Width of arrow heads
        figsize: Figure size
    """
    # Normalize and center each dimension
    y = data.copy()
    for i in range(3):
        y[:, i] = (y[:, i] - np.mean(y[:, i])) / (np.std(y[:, i]))
    
    # Set up plot
    plt.figure(figsize=figsize)
    
    # Plot the three 2D projections
    projections = [(0,1), (1,2), (2,0)]  # (x,y), (y,z), (z,x)
    for idx, (i,j) in enumerate(projections):
        plt.plot(y[:, i], y[:, j], lw=linewidth, color=colors[idx])
        
        # Add directional arrows
        for t in range(y.shape[0]-1):
            # Calculate midpoint for arrow base
            mid_x = np.mean([y[t, i], y[t+1, i]])
            mid_y = np.mean([y[t, j], y[t+1, j]])
            
            # Calculate direction vector
            dx = y[t+1, i] - y[t, i]
            dy = y[t+1, j] - y[t, j]
            
            # Plot arrow
            plt.arrow(mid_x, mid_y, 
                     arrow_scale*dx, arrow_scale*dy,
                     shape='full', lw=0, length_includes_head=True,
                     head_width=head_width, color=colors[idx])
    
    # Format plot
    plt.xlabel("Dimension 1", fontsize=15)
    plt.ylabel("Dimension 2", fontsize=15)
    plt.title("Normalized Attractor Projections", fontsize=20)
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.show()
