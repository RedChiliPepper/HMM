import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import argparse

project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from attractor_analysis.utils import *
from attractor_analysis.profiling_compile import *
from attractor_analysis.analysis import SLDSAnalyzer


def main(config_path: Path, data_path: Optional[Path] = None):
    """Run attractor analysis with given config and data paths.
    
    Args:
        config_path: Path to the YAML config file (e.g., experiments/config.yaml)
        data_path: Optional path to data directory. If None, defaults to ../data/
                  (sister directory of the experiments folder)
    """
    # Verify config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    # Read config
    try:
        config = read_config(config_path)
    except Exception as e:
        raise RuntimeError(f"Error reading config: {str(e)}") from e

    # If data_path not provided, set it to ../data/ relative to experiments folder
    if data_path is None:
        experiments_dir = config_path.parent
        project_root = experiments_dir.parent
        data_path = project_root / "data"

    print(f"Using config: {config_path}")
    print(f"Using data directory: {data_path}")
    
    # Get base config and attractor-specific parameters
    base_config = config["base"]
    attractor = base_config["attractor"]
    attractor_params = config.get(attractor, {})
    
    # Merge parameters (attractor-specific overrides base)
    params = {**base_config, **attractor_params}
    
    # Generate or load attractor data
    print(f"\nGenerating/Loading {attractor} attractor data...")
    data = loader(
        attractor=attractor,
        formatt=params["file_format"],
        save=params["save_data"],
        data_path = data_path,
        **attractor_params
    ).T

    print(data.shape)

    # Normalize and center each dimension
    y = data.copy()
    for i in range(3):
        y[:, i] = (y[:, i] - np.mean(y[:, i])) / (np.std(y[:, i]))

    # Visualization using dictionary format
    print("\nPlotting raw attractor data...")
    Plot3D(
        y[:, 0], y[:, 1], y[:, 2],
        title=f"{attractor} Attractor",
        start=0,
        pt_number=len(data[:, 0]),
        lw=0.1
    )
    
    # Initialize and run analyzer
    print("\nRunning SLDS analysis...")
    analyzer = SLDSAnalyzer(K=config["analysis"]["K"], 
                           N=config["analysis"].get("N", 3), 
                           backend=config["analysis"].get("backend", "plain"))

    # Fit the model
    print("\nFitting model...")
    analyzer.fit(
        y[:, :analyzer.N],  # Using all dimensions
        L=params["gibbs_iterations"],
        random_seed=params["random_seed"]
    )

    # Get parameters after burn-in
    burn_in = params["burn_in"]
    M, A_hat, Q, z_mode = analyzer.get_parameters(burn_in=burn_in)
    N = analyzer.N
    K = analyzer.K

    # Prepare colors
    colors = config["analysis"]["colors"]
    if isinstance(colors, str):
        if colors.startswith(('tab', 'viridis', 'plasma', 'inferno', 'magma', 'cividis')):
            cmap = plt.get_cmap(colors)
            colors = [cmap(i) for i in np.linspace(0, 1, K)]
        else:
            colors = [colors] * K
    elif not isinstance(colors, list):
        raise ValueError("colors must be either a string or list of color specifications")

    # Plot results
    print("\nPlotting analysis results...")
    y_plot = y[:, :N]  # Use only the first N dimensions

    # 1. Plot the inferred trajectory with state coloring
    analyzer.plot_attractor_dynamics(
        y=y_plot,
        burn_in=burn_in,
        plot_type="trajectory",
        normalize=config["analysis"]["normalize"],
        colors=colors,
        linewidth=config["analysis"]["linewidth"],
        arrow_scale=0.1,
        head_width=0.01,
        figsize=(24, 8),
        projection_3d=config["analysis"]["projection_3d"]
    )

    # 2. Plot the vector field with state coloring
    analyzer.plot_attractor_dynamics(
        y=y_plot,
        burn_in=burn_in,
        plot_type="vector_field",
        normalize=config["analysis"]["normalize"],
        colors=colors,
        linewidth=config["analysis"]["linewidth"],
        arrow_scale=0.2,
        head_width=0.05,
        figsize=(10, 8),
        projection_3d=config["analysis"]["projection_3d"]
    )

    # 3. Plot latent dynamics for each state
    print("\nPlotting latent dynamics for each state...")
    x_range = np.linspace(min(y_plot[:, 0])*1.5, max(y_plot[:, 0])*1.5, 20)
    y_range = np.linspace(min(y_plot[:, 1])*1.5, max(y_plot[:, 1])*1.5, 20)
    X1, X2 = np.meshgrid(x_range, y_range)

    for k in range(K):
        analyzer.plot_latent_dynamics(
            X1, X2, 
            k=k,
            burn_in=burn_in,
            show_fixed_point=config["analysis"]["show_fixed_point"],
            show_states=True,
            y=y_plot,
            z=z_mode  # Using the mode of z across samples
        )

    # 4. Additional projections for N >= 3
    if N >= 3:
        print("\nPlotting additional 2D projections...")
        fig, axes = plt.subplots(N, N, figsize=(15, 15))
        
        for i in range(N):
            for j in range(N):
                ax = axes[i,j]
                if i != j:
                    # Color by state
                    for t in range(len(y_plot)-1):
                        ax.plot(y_plot[t:t+2, i], y_plot[t:t+2, j],
                               color=colors[z_mode[t]], 
                               lw=config["analysis"]["linewidth"]/2)
                    ax.set_title(f"Dim {i+1} vs {j+1}")
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run attractor analysis with a given config file and data path.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to YAML config file (e.g., experiments/config.yaml)"
    )
    parser.add_argument(
        '--data-path', 
        type=str,
        help='Override data directory path if provided'
    )
    
    args = parser.parse_args()
    
    # Get absolute path to config file
    config_path = Path(args.config).absolute()
    print(f"Using config file: {config_path}")
    
    # Load config file first
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file - {e}")
        sys.exit(1)
    
    # Handle data path - command line argument overrides config file
    if args.data_path:
        data_path = Path(args.data_path)
    elif 'data_path' in config:
        data_path = Path(config['data_path'])
    else:
        experiments_dir = config_path.parent
        project_root = experiments_dir.parent
        data_path = project_root / 'data'
    
    # Resolve to absolute path
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    data_path = data_path.resolve()

    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Using data directory: {data_path}")
    
    try:
        main(config_path, data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required config parameter - {e}")
        sys.exit(1)
