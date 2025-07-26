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

def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "src").exists()
    assert (root / "test").exists()

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

def test_get_data_path():
    path = get_data_path("test_file.npy")
    assert isinstance(path, str)
    assert path.endswith("data/test_file.npy")
    assert "data" in path
    
    # Test directory creation
    path = get_data_path()
    assert os.path.exists(path)

def test_loader(tmp_path, monkeypatch):
    # Mock the data path to use temp directory
    monkeypatch.setattr('attractor_analysis.utils.get_data_path', lambda x="": str(tmp_path / x))
    
    # Test loading Lorenz attractor
    data = loader("Lorenz", ".npy", save=True, n=100)
    assert data.shape == (3, 100)
    
    # Test file is saved and can be reloaded
    assert len(list(tmp_path.glob("*.npy"))) == 1
    data2 = loader("Lorenz", ".npy", save=False, n=100)
    np.testing.assert_array_equal(data, data2)
    
    # Test Rossler attractor
    data = loader("Rossler", ".npy", save=True, n=100)
    assert data.shape == (3, 100)
    
    # Test DoubleScroll attractor
    data = loader("DoubleScroll", ".npy", save=True, n=100)
    assert data.shape == (3, 100)

if __name__ == "__main__":
    pytest.main(["-v"])
