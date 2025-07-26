import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from attractor_analysis.utils import *
from attractor_analysis.profiling_compile import *
from attractor_analysis.analysis import SLDSAnalyzer


def main(config_file):
    # Load configuration
    config_path = Path(get_project_root()) / "experiments" / str(config_file)
    print(f"Looking for config at: {config_path}")  # Debug line
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    # Pass the Path object directly
    config = read_config(config_path)

    
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
    parser = argparse.ArgumentParser(description="Run attractor analysis with a given config file.")
    parser.add_argument(
        '--config_file', 
        type=str, 
        required=True,
        help="YAML config file located in experiments folder (e.g., config.yaml)"
    )
    
    args = parser.parse_args()
    
    
    try:
        main(args.config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required config file parameter - {e}")
        sys.exit(1)
