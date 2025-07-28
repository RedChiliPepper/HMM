import pytest
import numpy as np
import sys
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# First try regular package import (works when installed)
try:
    from attractor_analysis.analysis import SLDSAnalyzer
except ImportError:
    # Fallback to development import (when running tests from source)
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from attractor_analysis.analysis import SLDSAnalyzer

from attractor_analysis.utils import *

warnings.filterwarnings("ignore", 
                       category=DeprecationWarning,
                       module="PIL.Image")  # The warning actually comes from Pillow

# Test data setup
@pytest.fixture
def sample_attractor_data():
    t = np.linspace(0, 10, 100)
    return {
        'x': np.sin(t),
        'y': np.cos(t),
        'z': t/10
    }

@pytest.fixture
def sample_3d_data():
    return np.random.rand(100, 3)

@pytest.fixture
def temp_config_file(tmp_path):
    config = {
        'test_param': 42,
        'attractor': 'Lorenz',
        'params': [28.0, 10.0, 8/3]
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        import yaml
        yaml.dump(config, f)
    return config_file

def test_coordinate_choice():
    assert coordinate_choice("Lorenz", 1, 2, 3) == 1
    assert coordinate_choice("Rossler", 1, 2, 3) == 3
    assert coordinate_choice("DoubleScroll", 1, 2, 3) == 1
    
    with pytest.raises(ValueError):
        coordinate_choice("InvalidAttractor", 1, 2, 3)

def test_read_config(temp_config_file):
    config = read_config(str(temp_config_file).replace('.yaml', ''))
    assert config['test_param'] == 42
    assert config['attractor'] == 'Lorenz'
    assert config['params'] == [28.0, 10.0, 8/3]

def test_read_attractor_data(tmp_path, sample_3d_data):
    # Test .npy file
    npy_file = tmp_path / "test.npy"
    np.save(npy_file, sample_3d_data)
    data = read_attractor_data(npy_file)
    assert 'x' in data
    assert 'y' in data
    assert 'z' in data
    
    # Test .csv file
    csv_file = tmp_path / "test.csv"
    np.savetxt(csv_file, sample_3d_data)
    data = read_attractor_data(csv_file)
    assert 'x' in data
    assert 'y' in data
    assert 'z' in data
    
    # Test invalid format
    invalid_file = tmp_path / "test.txt"
    invalid_file.touch()
    with pytest.raises(ValueError):
        read_attractor_data(invalid_file)

def test_plot_functions(sample_attractor_data, monkeypatch):
    
    # Mock plt.show to avoid displaying plots during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    xs, ys, zs = sample_attractor_data['x'], sample_attractor_data['y'], sample_attractor_data['z']
    
    # Test 2D plot
    Plot2D(xs, ys, zs, "Test XY", "Test XZ", "Test YZ", 0, 100)
    
    # Test 3D plot
    Plot3D(xs, ys, zs, "Test 3D", 0, 100)
    
    # Test normalized plot
    data = np.column_stack([xs, ys, zs])
    plot_normalized_attractor(data)


def test_loader(tmp_path, monkeypatch):
    """Test the loader function with different attractor types."""
    # Setup test parameters
    test_params = {
        "n": 100,  # Reduced number of points for faster testing
        "save": True,
        "formatt": ".npy"
    }

    # Test with different attractor types
    attractors = ["Lorenz", "Rossler", "DoubleScroll"]
    
    for attractor in attractors:
        # Test saving and loading
        data = loader(
            attractor=attractor,
            formatt=test_params["formatt"],
            save=test_params["save"],
            data_path=tmp_path,  # Use pytest's tmp_path fixture
            n=test_params["n"]
        )
        
        # Verify output shape
        assert data.shape == (3, test_params["n"]), \
            f"{attractor} attractor has wrong shape"
        
        # Verify file was saved
        saved_files = list(tmp_path.glob(f"*{test_params['formatt']}"))
        assert len(saved_files) > 0, \
            f"No {test_params['formatt']} file saved for {attractor}"
        
        # Test reloading
        reloaded_data = loader(
            attractor=attractor,
            formatt=test_params["formatt"],
            save=False,  # Don't save this time
            data_path=tmp_path,
            n=test_params["n"]
        )
        
        # Verify data consistency
        np.testing.assert_allclose(
            data,
            reloaded_data,
            err_msg=f"Reloaded {attractor} data doesn't match original"
        )
        
        # Clean up for next test
        for f in saved_files:
            f.unlink()

if __name__ == "__main__":
    pytest.main(["-v"])
