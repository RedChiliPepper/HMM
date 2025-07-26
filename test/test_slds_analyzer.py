import pytest
import numpy as np
import sys
import os
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


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    T = 100  # time points
    N = 2    # observation dimension
    y = np.random.randn(T, N)
    return y

@pytest.fixture
def analyzer():
    """Initialize analyzer for testing."""
    return SLDSAnalyzer(K=3, N=2)

def test_initialization(analyzer):
    """Test class initialization."""
    assert analyzer.K == 3
    assert analyzer.N == 2
    #assert analyzer.S.shape == (2, 2)
    #assert analyzer.V.shape == (3, 3)
    #assert analyzer.C.shape == (2, 3)

def test_initialize_priors(analyzer):
    """Test prior initialization."""
    analyzer.initialize_priors()
    assert analyzer.S.shape == (2, 2)
    assert analyzer.nu == 2
    assert analyzer.C.shape == (2, 3)
    assert analyzer.V.shape == (3, 3)

def test_fit(sample_data, analyzer):
    """Test fitting procedure."""
    analyzer.fit(sample_data, L=5, random_seed=42)
    
    # Check shapes of inferred variables
    assert analyzer.z_inf.shape == (5, 100)
    assert analyzer.M_inf.shape == (5, 3, 3)
    assert analyzer.A_hat_inf.shape == (5, 3, 2, 3)
    assert analyzer.Q_inf.shape == (5, 3, 2, 2)
    assert analyzer.alpha_inf.shape == (5, 3, 3)
    
    # Check transition matrix rows sum to 1
    for l in range(5):
        for k in range(3):
            assert np.allclose(analyzer.M_inf[l, k].sum(), 1.0)

"""
def test_message_passing(sample_data, analyzer):
    # Test message passing implementation including message computation validation.
    # First run the normal fitting procedure
    analyzer.fit(sample_data, L=5, random_seed=42)
    
    # Original tests - check sampled states
    assert np.all(analyzer.z_inf >= 0)
    assert np.all(analyzer.z_inf < analyzer.K)
    
    # New validation - check message computations
    # Get one iteration's parameters (last sample)
    M = analyzer.M_inf[-1]
    A_hat = analyzer.A_hat_inf[-1]
    Q = analyzer.Q_inf[-1]
    
    # Extract dimensions
    K, N = analyzer.K, analyzer.N
    T = sample_data.shape[0]
    
    # Run reference implementation (without bias term for y_hat)
    log_m_ref = analyzer._message_passing(M, A_hat, Q, sample_data)
    
    # Run optimized implementation through analyzer's method
    log_m_opt = analyzer._message_passing_optimized(M, A_hat, Q, sample_data)
    
    # Compare message computations
    for t in range(T):
        assert np.allclose(
            log_m_ref[t], 
            log_m_opt[t], 
            atol=1e-6, 
            rtol=1e-6
        ), f"Message computation differs at t={t}"
    
    # Additional check - verify message dimensions
    assert len(log_m_opt) == T
    for msg in log_m_opt:
        assert msg.shape == (K,)
    
    # Check that messages are used correctly in sampling
    # (This verifies the connection between messages and sampling)
    last_log_m = log_m_opt[-1]
    assert last_log_m.shape == (K,)
    assert np.all(np.isfinite(last_log_m))
"""

def test_message_passing_equivalence():
    # Setup test parameters
    K, N, T = 3, 2, 100
    np.random.seed(42)
    
    # Generate test data
    M = np.random.dirichlet(np.ones(K), size=K)
    A_hat = np.random.randn(K, N, N+1)  # +1 for bias term
    Q = np.array([np.eye(N) for _ in range(K)])
    y = np.random.randn(T, N)
    
    # Initialize analyzer
    analyzer = SLDSAnalyzer(K=3, N=2)
    
    # Run both implementations
    z_orig = analyzer._message_passing(M, A_hat, Q, y)
    log_m_orig = analyzer._get_log_messages()  # Need to implement this similarly
    
    # Run optimized
    z_opt = analyzer._message_passing_optimized(M, A_hat, Q, y)
    log_m_opt = analyzer._get_log_messages_optimized()
    
    # Compare outputs
    assert z_orig.shape == z_opt.shape
    print(f"Original and optimized z agreement: {np.mean(z_orig == z_opt)*100:.2f}%")
    
    for t in range(T):
        assert np.allclose(log_m_orig[t], log_m_opt[t], atol=1e-6, rtol=1e-6), \
            f"Messages differ at t={t}"

def test_message_passing_shapes(analyzer):
    K, N, T = 3, 2, 100
    M = np.random.dirichlet(np.ones(K), size=K)
    A_hat = np.random.randn(K, N, N+1)
    Q = np.array([np.eye(N) for _ in range(K)])
    y = np.random.randn(T, N)
    
    z_orig = analyzer._message_passing(M, A_hat, Q, y)
    z_opt = analyzer._message_passing_optimized(M, A_hat, Q, y)
    
    assert z_orig.shape == (T,)
    assert z_opt.shape == (T,)
    assert np.all(z_orig < K) and np.all(z_orig >= 0)
    assert np.all(z_opt < K) and np.all(z_opt >= 0)

"""
def test_statistical_message_passing_equivalence(analyzer):
    K, N, T = 3, 2, 1000
    n_trials = 30
    state_freq_diff = []
    
    for _ in range(n_trials):
        # Generate random inputs
        M = np.random.dirichlet(np.ones(K), size=K)
        A_hat = np.random.randn(K, N, N+1)
        Q = np.array([np.eye(N) for _ in range(K)])
        y = np.random.randn(T, N)
        
        # Run both implementations
        z_orig = analyzer._message_passing(M, A_hat, Q, y)
        z_opt = analyzer._message_passing_optimized(M, A_hat, Q, y)
        
        # Compare state frequencies instead of exact matches
        orig_counts = np.bincount(z_orig, minlength=K)/T
        opt_counts = np.bincount(z_opt, minlength=K)/T
        state_freq_diff.append(np.max(np.abs(orig_counts - opt_counts)))
    
    avg_diff = np.mean(state_freq_diff)
    print(f"Average maximum state frequency difference: {avg_diff:.4f}")
    assert avg_diff < 0.05, "State frequencies should differ by <5%"
"""

def test_update_transition_matrix(sample_data, analyzer):
    """Test transition matrix update."""
    analyzer.fit(sample_data, L=5, random_seed=42)
    
    # Check that alpha parameters are updated
    for l in range(4):
        assert np.any(analyzer.alpha_inf[l+1] != analyzer.alpha_inf[l])
    
    # Check transition matrices remain valid
    for l in range(5):
        for k in range(3):
            assert np.all(analyzer.M_inf[l, k] >= 0)
            assert np.allclose(analyzer.M_inf[l, k].sum(), 1.0)

def test_update_parameters(sample_data, analyzer):
    """Test parameter update for A_hat and Q."""
    analyzer.fit(sample_data, L=5, random_seed=42)
    
    # Check Q matrices are positive definite
    for l in range(5):
        for k in range(3):
            eigenvalues = np.linalg.eigvals(analyzer.Q_inf[l, k])
            assert np.all(eigenvalues > 0)

def test_get_parameters(sample_data, analyzer):
    """Test parameter estimation after burn-in."""
    analyzer.fit(sample_data, L=10, random_seed=42)
    M, A_hat, Q, z = analyzer.get_parameters(burn_in=5)
    
    K = analyzer.K  # Get actual number of states from analyzer
    N = analyzer.N  # Get system dimension from analyzer
    T = sample_data.shape[0]  # Number of time points
    
    # Test parameter shapes
    assert M.shape == (K, K)  # Transition matrix should be K×K
    assert A_hat.shape == (K, N, N+1)  # Dynamics matrices (including bias)
    assert Q.shape == (K, N, N)  # Noise covariance matrices
    
    # Test z shape and values
    assert z.shape == (T,)  # Should match number of time points
    assert np.all(z >= 0) and np.all(z < K)  # Valid state indices (0 to K-1)
    
    # Check transition matrix rows sum to 1
    for k in range(K):
        assert np.allclose(M[k].sum(), 1.0, atol=1e-6), \
            f"Transition matrix row {k} doesn't sum to 1"
    
    # Check Q matrices are positive definite
    for k in range(K):
        eigenvalues = np.linalg.eigvals(Q[k])
        assert np.all(eigenvalues > -1e-6), \
            f"Q matrix for state {k} is not positive definite"
    
    # Additional test for z properties
    assert len(np.unique(z)) <= K, "More unique states than K"
    assert isinstance(z, np.ndarray), "z should be a numpy array"
    assert np.issubdtype(z.dtype, np.integer), "States should be integers"

def test_plot_latent_dynamics(sample_data, analyzer, monkeypatch):
    """Test plotting functionality (mocked)."""
    # Mock plt.show to avoid displaying plots during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    analyzer.fit(sample_data, L=10, random_seed=42)
    
    # Create grid for plotting
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X1, X2 = np.meshgrid(x, y)
    
    # Test plotting for each state
    for k in range(analyzer.K):
        analyzer.plot_latent_dynamics(X1, X2, k, burn_in=5)
        analyzer.plot_latent_dynamics(X1, X2, k, burn_in=5, show_fixed_point=True)
        
        # Test with state coloring
        analyzer.plot_latent_dynamics(X1, X2, k, burn_in=5, show_states=True, z=analyzer.z_inf[-1])

def test_plot_attractor_dynamics(analyzer, sample_data, monkeypatch):
    """Test the attractor dynamics plotting functionality."""
    # Mock plt.show to avoid displaying plots during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    # Fit the analyzer first with the sample data
    analyzer.fit(sample_data, L=10, random_seed=42)
    
    # Determine expected dimension from analyzer
    expected_dim = analyzer.N
    
    # Create test variants
    test_cases = []
    if expected_dim == 2:
        test_cases.append((sample_data, False))  # 2D data, no projection
        # Create malformed cases
        malformed_data = np.column_stack([sample_data, np.random.randn(sample_data.shape[0])])
        test_cases.append((malformed_data, "dimension"))  # 3D data should fail
        test_cases.append((sample_data, "3D projection"))  # 3D projection should fail
    elif expected_dim == 3:
        test_cases.append((sample_data, True))   # 3D data with projection
        test_cases.append((sample_data, False))  # 3D data without projection
        # Create malformed 2D data
        malformed_data = sample_data[:,:2]
        test_cases.append((malformed_data, "dimension"))  # 2D data should fail
    
    # Run tests
    for test_data, projection in test_cases:
        if isinstance(projection, bool):
            # Valid case - should run without errors
            analyzer.plot_attractor_dynamics(
                y=test_data,
                burn_in=5,
                plot_type='vector_field',
                projection_3d=projection,
                colors=("r", "g", "b"),
                figsize=(10, 8)
            )
        else:
            # Invalid case - should raise ValueError
            with pytest.raises(ValueError, match=projection):
                analyzer.plot_attractor_dynamics(
                    y=test_data,
                    plot_type='vector_field',
                    projection_3d=(projection == "3D projection")
                )
    
    # Test trajectory plots
    if expected_dim >= 3:
        analyzer.plot_attractor_dynamics(
            y=sample_data,
            plot_type='trajectory'
        )
    else:
        with pytest.raises(ValueError, match="Trajectory plotting requires at least 3 dimensions"):
            analyzer.plot_attractor_dynamics(
                y=sample_data,
                plot_type='trajectory'
            )
    
    plt.close('all')

def test_plot_attractor_dynamics_edge_cases(analyzer, sample_data, monkeypatch):
    """Test edge cases and error handling for plotting functionality."""
    # Mock plt.show to avoid displaying plots during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    # Fit the analyzer first
    analyzer.fit(sample_data, L=10, random_seed=42)
    
    # Create 3D version of the data for trajectory tests
    sample_data_3d = np.column_stack([sample_data, np.random.randn(sample_data.shape[0])])
    
    # Test invalid plot type
    with pytest.raises(ValueError, match="plot_type must be 'vector_field' or 'trajectory'"):
        analyzer.plot_attractor_dynamics(
            y=sample_data_3d,
            plot_type='invalid_type'
        )
    
    # Test trajectory plot with 2D data (should fail)
    with pytest.raises(ValueError, match="Trajectory plotting requires at least 3 dimensions"):
        analyzer.plot_attractor_dynamics(
            y=sample_data,
            plot_type='trajectory'
        )
    
    # Test with empty data
    with pytest.raises(ValueError):
        analyzer.plot_attractor_dynamics(
            y=np.empty((0, 2)),  # Empty 2D data
            plot_type='vector_field'
        )
    
    # Test with invalid burn-in (larger than number of samples)
    with pytest.raises(ValueError):
        analyzer.plot_attractor_dynamics(
            y=sample_data,
            burn_in=20  # Only 10 samples available
        )
    
    # Test with invalid data dimensions
    with pytest.raises(ValueError):
        analyzer.plot_attractor_dynamics(
            y=np.random.randn(10, 4),  # N=4 but analyzer initialized with N=2
            plot_type='vector_field'
        )
    
    # Clean up any figures
    plt.close('all')


def test_edge_cases():
    """Test edge cases like single state or single dimension."""
    # Test with K=1 (single state)
    analyzer1 = SLDSAnalyzer(K=1, N=2)
    y = np.random.randn(50, 2)
    analyzer1.fit(y, L=3)
    assert analyzer1.z_inf.shape == (3, 50)
    assert np.all(analyzer1.z_inf == 0)  # Only one state
    
    # Test with N=1 (single dimension)
    analyzer2 = SLDSAnalyzer(K=2, N=1)
    y = np.random.randn(50, 1)
    analyzer2.fit(y, L=3)
    assert analyzer2.Q_inf.shape == (3, 2, 1, 1)
