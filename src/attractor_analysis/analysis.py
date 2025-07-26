import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invwishart, matrix_normal
from scipy.special import logsumexp
from typing import Tuple, Optional, Union
from tqdm import tqdm
import time
from numba import njit, prange
import math
from .profiling_compile import *

@profile_resources(log_dir="logs/analyzer")
@profile_to_logs(log_dir="logs/line_profiles")   
@njit(parallel=True, fastmath=True)
def message_passing_optimized(M: np.ndarray, 
                   A_hat: np.ndarray, 
                   Q: np.ndarray, 
                   y: np.ndarray) -> np.ndarray:

    K = M.shape[0]        # Number of states from transition matrix
    N = Q.shape[-1]       # Dimension from covariance matrix
    T = y.shape[0]        # Time steps from observation data
    log_m = [np.zeros(K)]
    
    # Constants
    log_2pi = math.log(2 * math.pi)
    half_log_2pi = 0.5 * log_2pi * N
    
    # Pre-allocate arrays
    y_hat = np.empty(N + 1)
    y_hat[0] = 1.0
    
    for t in range(T-1, 0, -1):
        # Prepare y_hat
        y_hat[1:] = y[t-1]
        somm = np.zeros((K, K))
        m_t = np.zeros(K)
        
        for k in prange(K):
            for j in range(K):
                # Manual MVN logpdf computation
                Q_j = Q[j]
                x = y[t] - np.dot(A_hat[j], y_hat)
                
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(Q_j)
                except:
                    # Handle non-positive definite matrices
                    L = np.linalg.cholesky(Q_j + 1e-6 * np.eye(N))
                
                # Solve L*y = x
                sol = np.linalg.solve(L, x)
                
                # Compute quadratic form and logdet
                quad_form = np.dot(sol, sol)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                
                # Compute logpdf
                log_pdf = -0.5 * quad_form - 0.5 * logdet - half_log_2pi
                
                # Update somm
                somm[k,j] = math.log(M[k,j]) + log_pdf + log_m[T-1-t][j]
        
        # Compute logsumexp for each row
        for k in range(K):
            max_val = np.max(somm[k])
            m_t[k] = max_val + math.log(np.sum(np.exp(somm[k] - max_val)))
        
        log_m.append(m_t)

    # Sampling part
    log_p_z = np.zeros((T, K))
    z = np.zeros(T, dtype=np.int32)
    
    # Initial state
    log_p_z[0] = log_m[T-1]
    max_val = np.max(log_p_z[0])
    log_p_z[0] -= max_val + math.log(np.sum(np.exp(log_p_z[0] - max_val)))
    probs = np.exp(log_p_z[0])
    probs /= probs.sum()
    z[0] = np.searchsorted(np.cumsum(probs), np.random.random())
    
    # Subsequent states
    for t in range(1, T):
        y_hat[1:] = y[t-1]
        
        for k in prange(K):
            Q_k = Q[k]
            x = y[t] - np.dot(A_hat[k], y_hat)
            
            # Manual MVN computation as before
            try:
                L = np.linalg.cholesky(Q_k)
            except:
                L = np.linalg.cholesky(Q_k + 1e-6 * np.eye(N))
            
            sol = np.linalg.solve(L, x)
            quad_form = np.dot(sol, sol)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            
            log_pdf = -0.5 * quad_form - 0.5 * logdet - half_log_2pi
            log_p_z[t,k] = math.log(M[z[t-1],k]) + log_pdf + log_m[T-1-t][k]
        
        # Normalize
        max_val = np.max(log_p_z[t])
        log_p_z[t] -= max_val + math.log(np.sum(np.exp(log_p_z[t] - max_val)))
        probs = np.exp(log_p_z[t])
        probs /= probs.sum()
        z[t] = np.searchsorted(np.cumsum(probs), np.random.random())
    
    return z, log_m

class SLDSAnalyzer:

    _available_backends_ = {
    'plain': '_message_passing',          # Pure Python
    'optimized': '_message_passing_optimized_no_numba',  # Optimized Python
    'numba': '_message_passing_optimized' # Optimized + Numba
}

    def __init__(self, K: int = 5, N: int = 2, backend: str = 'plain'):
        """
        Initialize SLDS Analyzer.
        
        Args:
            K: Number of states
            N: Observation dimension
            backend: Implementation backend ('plain' or 'numba')
        """
        if not isinstance(K, int) or not isinstance(N, int):
            raise TypeError('K and N must be integers')
        
        if backend not in self._available_backends_:
            raise ValueError(f"Backend must be one of {list(self._available_backends_.keys())}")

        self.K = K
        self.N = N
        self.backend = backend
        self._message_passing_method = getattr(self, self._available_backends_[backend])
        self._log_m_cache = None
        self._log_m_opt_cache = None
        
    def initialize_priors(self) -> None:
        """Initialize prior distributions for the model parameters."""
        self.S = 0.0001 * np.eye(self.N)
        self.nu = self.N
        self.C = 0.1 * np.random.randn(self.N, self.N+1)
        self.V = 0.1 * np.eye(self.N+1)
    
    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")    
    def fit(self, y: np.ndarray, L: int = 100, random_seed: Optional[int] = None) -> None:
        """Run Gibbs sampling inference.
        
        Args:
            y: Observation data of shape (T, N)
            L: Number of Gibbs iterations
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        T = y.shape[0]
        self.initialize_priors()
        
        # Initialize variables
        self.z_inf = np.zeros((L, T), dtype=int)
        self.M_inf = np.zeros((L, self.K, self.K))
        self.A_hat_inf = np.zeros((L, self.K, self.N, self.N+1))
        self.Q_inf = np.zeros((L, self.K, self.N, self.N))
        self.alpha_inf = np.ones((L, self.K, self.K))
        
        # Initial values
        self.z_inf[0] = np.random.randint(self.K, size=T)
        self.M_inf[0] = np.stack([np.random.dirichlet([1]*self.K) for _ in range(self.K)])
        self.A_hat_inf[0] = np.random.randn(self.K, self.N, self.N+1)
        self.Q_inf[0] = np.stack([0.0001*np.eye(self.N) for _ in range(self.K)])
        
        # Gibbs sampling
        pbar = tqdm(total=L-1, 
                desc="Gibbs Sampling",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    
        iter_times = []
        start_time = time.time()
        for l in range(L-1):
            start_iter = time.time()
            self._gibbs_step(y, l)
            iter_time = time.time() - start_iter
            iter_times.append(iter_time)
            
            # Update progress bar
            avg_time = np.mean(iter_times[-10:])  # Moving average of last 10 iters
            pbar.set_postfix({"avg_iter": f"{avg_time:.2f}s"})
            pbar.update(1)
        
        pbar.close()
        
        total_time = time.time() - start_time
        print(f"\nCompleted {L} iterations in {total_time:.2f} seconds")
        print(f"Fastest iter: {min(iter_times):.4f}s | Slowest iter: {max(iter_times):.4f}s")
        print(f"Average iter time: {np.mean(iter_times):.4f}s")
    
    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def _gibbs_step(self, y: np.ndarray, l: int) -> None:
        """Single Gibbs sampling step.
        
        Args:
            y: Observation data
            l: Current iteration index
        """
        # Update latent states z
        self.z_inf[l+1] = self._message_passing_method(
            self.M_inf[l], 
            self.A_hat_inf[l], 
            self.Q_inf[l], 
            y
        )
        
        # Update transition matrix M
        self._update_transition_matrix(l)
        
        # Update model parameters A_hat and Q
        self._update_parameters(y, l)
    
    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def _message_passing(self, 
                       M: np.ndarray, 
                       A_hat: np.ndarray, 
                       Q: np.ndarray, 
                       y: np.ndarray) -> np.ndarray:
        """Perform message passing to sample latent states z.
        
        Args:
            M: Current transition matrix
            A_hat: Current system matrices
            Q: Current noise covariance matrices
            y: Observation data
            
        Returns:
            Sampled latent states z
        """
        T = y.shape[0]
        log_m = [np.zeros(self.K)]  # log messages
        
        for t in reversed(range(1, T)):
            y_hat = np.concatenate([[1], y[t-1]])
            somm = np.zeros((self.K, self.K))
            m_t = np.zeros(self.K) # log of message at time t
            for k in range(self.K):
                for j in range(self.K):
                    rv = multivariate_normal(np.zeros(self.N), Q[j])
                    x = y[t] - np.dot(A_hat[j], y_hat) #A_hat[j] @ y_hat
                    somm[k,j] = (np.log(M[k,j]) + 
                                rv.logpdf(x) + 
                                log_m[T-1-t][j])

            for k in range(self.K):
                m_t[k] = logsumexp(somm[k])
            
            log_m.append(m_t)

        log_p_z = np.zeros((T, self.K))
        z = np.zeros(T, dtype=int)
        
        # Initial state
        for k in range(self.K):
            log_p_z[0, k] = log_m[T-1][k]
        log_p_z[0] = log_p_z[0] - logsumexp(log_p_z[0])
        probs = np.exp(log_p_z[0])
        probs /= probs.sum()  # Ensure probabilities sum to 1
        z[0] = np.random.choice(self.K, p=probs)
        
        # Subsequent states
        for t in range(1, T):
            y_hat = np.concatenate([[1], y[t-1]])
            for k in range(self.K):
                rv = multivariate_normal(np.zeros(self.N), Q[k])
                x = y[t] - np.dot(A_hat[k], y_hat)#A_hat[k] @ y_hat
                log_p_z[t,k] = (np.log(M[z[t-1],k]) + 
                               rv.logpdf(x) + 
                               log_m[T-1-t][k])
            
            log_p_z[t] -= logsumexp(log_p_z[t])
            probs = np.exp(log_p_z[t])
            probs /= probs.sum()  # Ensure probabilities sum to 1
            z[t] = np.random.choice(self.K, p=probs)
        self._log_m_cache = log_m
        return z

    def _get_log_messages(self):
        """Get messages from original implementation"""
        if self._log_m_cache is None:
            raise ValueError("No messages available - run _message_passing first")
        return self._log_m_cache

    def _message_passing_optimized(self, 
                       M: np.ndarray, 
                       A_hat: np.ndarray, 
                       Q: np.ndarray, 
                       y: np.ndarray) -> np.ndarray:
        z, log_m = message_passing_optimized(M, A_hat, Q, y)
        self._log_m_opt_cache = log_m  # Store messages for testing
        return z

    def _get_log_messages_optimized(self):
        """For testing only - expose optimized messages"""
        if self._log_m_opt_cache is None:
            raise ValueError("No cached messages - run _message_passing_optimized first")
        return self._log_m_opt_cache

    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def _message_passing_optimized_no_numba(self,
                   M: np.ndarray,
                   A_hat: np.ndarray, 
                   Q: np.ndarray, 
                   y: np.ndarray) -> np.ndarray:

        T = y.shape[0]        # Time steps from observation data
        log_m = [np.zeros(self.K)]
        
        # Constants
        log_2pi = math.log(2 * math.pi)
        half_log_2pi = 0.5 * log_2pi * self.N
        
        # Pre-allocate arrays
        y_hat = np.empty(self.N + 1)
        y_hat[0] = 1.0
        
        for t in range(T-1, 0, -1):
            # Prepare y_hat
            y_hat[1:] = y[t-1]
            somm = np.zeros((self.K, self.K))
            m_t = np.zeros(self.K)
            
            for k in range(self.K):
                for j in range(self.K):
                    # Manual MVN logpdf computation
                    Q_j = Q[j]
                    x = y[t] - np.dot(A_hat[j], y_hat)
                    
                    # Cholesky decomposition
                    try:
                        L = np.linalg.cholesky(Q_j)
                    except:
                        # Handle non-positive definite matrices
                        L = np.linalg.cholesky(Q_j + 1e-6 * np.eye(self.N))
                    
                    # Solve L*y = x
                    sol = np.linalg.solve(L, x)
                    
                    # Compute quadratic form and logdet
                    quad_form = np.dot(sol, sol)
                    logdet = 2.0 * np.sum(np.log(np.diag(L)))
                    
                    # Compute logpdf
                    log_pdf = -0.5 * quad_form - 0.5 * logdet - half_log_2pi
                    
                    # Update somm
                    somm[k,j] = math.log(M[k,j]) + log_pdf + log_m[T-1-t][j]
            
            # Compute logsumexp for each row
            for k in range(self.K):
                max_val = np.max(somm[k])
                m_t[k] = max_val + math.log(np.sum(np.exp(somm[k] - max_val)))
            
            log_m.append(m_t)

        # Sampling part
        log_p_z = np.zeros((T, self.K))
        z = np.zeros(T, dtype=np.int32)
        
        # Initial state
        log_p_z[0] = log_m[T-1]
        max_val = np.max(log_p_z[0])
        log_p_z[0] -= max_val + math.log(np.sum(np.exp(log_p_z[0] - max_val)))
        probs = np.exp(log_p_z[0])
        probs /= probs.sum()
        z[0] = np.searchsorted(np.cumsum(probs), np.random.random())
        
        # Subsequent states
        for t in range(1, T):
            y_hat[1:] = y[t-1]
            
            for k in range(self.K):
                Q_k = Q[k]
                x = y[t] - np.dot(A_hat[k], y_hat)
                
                # Manual MVN computation as before
                try:
                    L = np.linalg.cholesky(Q_k)
                except:
                    L = np.linalg.cholesky(Q_k + 1e-6 * np.eye(self.N))
                
                sol = np.linalg.solve(L, x)
                quad_form = np.dot(sol, sol)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                
                log_pdf = -0.5 * quad_form - 0.5 * logdet - half_log_2pi
                log_p_z[t,k] = math.log(M[z[t-1],k]) + log_pdf + log_m[T-1-t][k]
            
            # Normalize
            max_val = np.max(log_p_z[t])
            log_p_z[t] -= max_val + math.log(np.sum(np.exp(log_p_z[t] - max_val)))
            probs = np.exp(log_p_z[t])
            probs /= probs.sum()
            z[t] = np.searchsorted(np.cumsum(probs), np.random.random())
        
        return z

    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def _update_transition_matrix(self, l: int) -> None:
        """Update transition matrix M using Dirichlet priors.
        
        Args:
            l: Current iteration index
        """
        # Count transitions
        chi = np.zeros((self.K, self.K))
        for t in range(len(self.z_inf[l+1])-1):
            k = self.z_inf[l+1,t]
            j = self.z_inf[l+1,t+1]
            chi[k,j] += 1
        
        # Update Dirichlet parameters
        self.alpha_inf[l+1] += chi
        
        # Sample new transition matrix
        for k in range(self.K):
            self.M_inf[l+1,k] = np.random.dirichlet(self.alpha_inf[l+1,k])
    
    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def _update_parameters(self, y: np.ndarray, l: int) -> None:
        """Update system parameters A_hat and Q.
        
        Args:
            y: Observation data
            l: Current iteration index
        """
        T = y.shape[0]
        Y = [[] for _ in range(self.K)]
        Y_bar = [[] for _ in range(self.K)]
        N_k = np.zeros(self.K)
        
        # Group observations by state
        for k in range(self.K):
            for t in range(1, T):
                if self.z_inf[l+1,t] == k:
                    Y[k].append(y[t])
                    Y_bar[k].append(np.concatenate([[1], y[t-1]]))
            
            N_k[k] = len(Y[k])
            if N_k[k] > 0:
                Y[k] = np.array(Y[k]).T
                Y_bar[k] = np.array(Y_bar[k]).T
        
        # Compute sufficient statistics
        S_ybar_ybar = np.zeros((self.K, self.N+1, self.N+1))
        S_y_ybar = np.zeros((self.K, self.N, self.N+1))
        S_y_y = np.zeros((self.K, self.N, self.N))
        S_y_given_ybar = np.zeros((self.K, self.N, self.N))
        
        for k in range(self.K):
            if N_k[k] > 0:
                S_ybar_ybar[k] = Y_bar[k] @ Y_bar[k].T + self.V
                S_y_ybar[k] = Y[k] @ Y_bar[k].T + self.C @ self.V
                S_y_y[k] = Y[k] @ Y[k].T + self.C @ self.V @ self.C.T
                S_y_given_ybar[k] = (S_y_y[k] - 
                                    S_y_ybar[k] @ np.linalg.inv(S_ybar_ybar[k]) @ 
                                    S_y_ybar[k].T)
        
        # Sample new parameters
        for k in range(self.K):
            if N_k[k] > 0:
                # Add small regularization to ensure positive definiteness
                regularization = 1e-6 * np.eye(self.N)

                # Sample from Inverse Wishart (with regularization)
                rv_iw = invwishart(self.nu + N_k[k], self.S + S_y_given_ybar[k] + regularization)
                self.Q_inf[l+1,k] = rv_iw.rvs()
                
                # Sample from Matrix Normal
                mean = S_y_ybar[k] @ np.linalg.inv(S_ybar_ybar[k])
                row_cov = self.Q_inf[l+1,k]
                col_cov = np.linalg.inv(S_ybar_ybar[k])
                
                # Add regularization to column covariance
                col_cov = col_cov + 1e-6 * np.eye(col_cov.shape[0])
                # Sample from Matrix Normal
                rv_mn = matrix_normal(mean, row_cov, col_cov)
                self.A_hat_inf[l+1,k] = rv_mn.rvs()
            else:
                # Sample from priors if no data
                rv_iw = invwishart(self.nu, self.S)
                self.Q_inf[l+1,k] = rv_iw.rvs()
                #rv_mn = matrix_normal(self.C, self.Q_inf[l+1,k], self.V)
                # Ensure column covariance is positive definite
                col_cov = self.V + 1e-6 * np.eye(self.V.shape[0]) #col_cov = np.linalg.inv(self.V) + 1e-6 * np.eye(self.V.shape[0])
                rv_mn = matrix_normal(self.C, self.Q_inf[l+1,k], col_cov)
                self.A_hat_inf[l+1,k] = rv_mn.rvs()
    
    @profile_resources(log_dir="logs/analyzer")
    @profile_to_logs(log_dir="logs/line_profiles")
    def get_parameters(self, burn_in: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get posterior parameter estimates after burn-in, including averaged latent states.
        
        Args:
            burn_in: Number of initial samples to discard
            
        Returns:
            Tuple of (M, A_hat, Q, z) where:
            - M: Posterior mean of observation matrices
            - A_hat: Posterior mean of dynamics matrices
            - Q: Posterior mean of noise covariance matrices
            - z: Averaged latent state sequence after burn-in
        """
            # Calculate parameter means
        M = np.mean(self.M_inf[burn_in:], axis=0)
        A_hat = np.mean(self.A_hat_inf[burn_in:], axis=0)
        Q = np.mean(self.Q_inf[burn_in:], axis=0)
        
        # Calculate most likely latent states by taking mode across samples
        if hasattr(self, 'z_inf') and len(self.z_inf) > 0:
            # Stack all z samples after burn-in
            z_samples = np.array(self.z_inf[burn_in:])
            
            # For each time point, find the most frequent state across samples
            z_mode = np.zeros(z_samples.shape[1], dtype=int)
            for t in range(z_samples.shape[1]):
                counts = np.bincount(z_samples[:, t])
                z_mode[t] = np.argmax(counts)
        else:
            z_mode = np.array([])
        
        return M, A_hat, Q, z_mode
        
    def plot_latent_dynamics(self, 
                           X1: np.ndarray, 
                           X2: np.ndarray,
                           X3: Optional[np.ndarray] = None,
                           k: Optional[int] = None,
                           burn_in: int = 0,
                           show_fixed_point: bool = False,
                           show_states: bool = False,
                           y: Optional[np.ndarray] = None,
                           z: Optional[np.ndarray] = None,
                           projection_3d: bool = False) -> None:
        """Plot vector fields for latent states in 2D or 3D with optional trajectory.
        
        Args:
            X1: Meshgrid X coordinates
            X2: Meshgrid Y coordinates
            X3: Meshgrid Z coordinates (required for 3D)
            k: Which latent state to plot (None for all)
            burn_in: Burn-in period for parameter estimation
            show_fixed_point: Whether to mark fixed points
            show_states: Whether to show trajectory with state coloring
            y: Observation data for trajectory plotting
            z: Latent state sequence for coloring
            projection_3d: Whether to create a 3D plot
        """
        _, A_hat, _, _ = self.get_parameters(burn_in)
        A = A_hat[:,:,1:]  # State matrices (K x N x N)
        b = A_hat[:,:,0]   # Bias terms (K x N)
        
        n = X1.shape[0]
        K = A.shape[0]
        
        # If k is specified, only plot that state
        if k is not None:
            K = 1
            A = A[k:k+1]
            b = b[k:k+1]
        
        if projection_3d:
            if X3 is None:
                raise ValueError("X3 must be provided for 3D projection")
            if self.N < 3:
                raise ValueError("System dimension must be ≥3 for 3D plotting")
                
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            
            # Calculate vector field
            X = np.stack([X1, X2, X3])
            U = np.zeros((n, n, n))
            V = np.zeros((n, n, n))
            W = np.zeros((n, n, n))
            
            for i in range(n):
                for j in range(n):
                    for k_idx in range(n):
                        state = X[:,i,j,k_idx]
                        dx = A[0] @ state.reshape(-1, 1) + b[0].reshape(-1, 1) - state.reshape(-1, 1)
                        U[i,j,k_idx] = dx[0,0]
                        V[i,j,k_idx] = dx[1,0]
                        W[i,j,k_idx] = dx[2,0]
            
            # Plot every 2nd point for better visibility
            stride = max(1, n//10)
            ax.quiver(X1[::stride,::stride,::stride],
                     X2[::stride,::stride,::stride],
                     X3[::stride,::stride,::stride],
                     U[::stride,::stride,::stride],
                     V[::stride,::stride,::stride],
                     W[::stride,::stride,::stride],
                     length=0.1, normalize=True, color='k')
            
            if show_fixed_point:
                x_star = np.linalg.solve(np.eye(self.N) - A[0], b[0])
                ax.scatter(x_star[0], x_star[1], x_star[2], c='red', s=100)
                
            if show_states and y is not None and z is not None:
                # Create color map for states
                cmap = plt.get_cmap('tab10')
                state_colors = [cmap(k % cmap.N) for k in range(self.K)]
                
                # Plot trajectory with state coloring
                for t in range(y.shape[0]-1):
                    ax.plot(y[t:t+2, 0], y[t:t+2, 1], y[t:t+2, 2], 
                           color=state_colors[z[t]], lw=0.5)
                    
            ax.set_title(f"3D Latent State Dynamics")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
        else:
            # 2D Visualization - create subplot for each state if k is None
            if k is None:
                fig, axes = plt.subplots(nrows=K, ncols=1, figsize=(10, 6*K))
                if K == 1:
                    axes = [axes]
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                axes = [ax]
            
            for k_idx in range(K):
                ax = axes[k_idx]
                X = np.stack([X1, X2])
                U = np.zeros((n, n))
                V = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        state = np.zeros(self.N)
                        state[:2] = X[:,i,j]  # Only use first two dimensions
                        dx = A[k_idx] @ state.reshape(-1, 1) + b[k_idx].reshape(-1, 1) - state.reshape(-1, 1)
                        U[i,j] = dx[0,0]
                        V[i,j] = dx[1,0]
                
                ax.quiver(X1, X2, U, V, width=0.0025)
                
                if show_fixed_point:
                    A_2d = A[k_idx][:2,:2]  # Take 2x2 submatrix
                    b_2d = b[k_idx][:2]
                    x_star = np.linalg.solve(np.eye(2) - A_2d, b_2d)
                    ax.scatter(x_star[0], x_star[1], c='red', s=100)
                
                if show_states and y is not None and z is not None:
                    # Create color map for states
                    cmap = plt.get_cmap('tab10')
                    state_colors = [cmap(k % cmap.N) for k in range(self.K)]
                    
                    # Plot trajectory with state coloring
                    for t in range(y.shape[0]-1):
                        ax.plot(y[t:t+2, 0], y[t:t+2, 1], 
                               color=state_colors[z[t]], lw=0.5)
                        # Add directional arrows
                        arr_mean_y1 = np.mean([y[t, 0], y[t+1, 0]])
                        arr_mean_y2 = np.mean([y[t, 1], y[t+1, 1]])
                        dy1 = y[t+1, 0] - y[t, 0]
                        dy2 = y[t+1, 1] - y[t, 1]
                        ax.arrow(arr_mean_y1, arr_mean_y2, 0.01*dy1, 0.01*dy2, 
                                 shape='full', lw=0, length_includes_head=True, 
                                 head_width=0.02, color=state_colors[z[t]])
                
                ax.set_title(f"2D Latent State {k_idx if k is None else k} Dynamics")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()

    def plot_attractor_dynamics(self, y: np.ndarray, 
                              burn_in: int = 0,
                              plot_type: str = 'vector_field',
                              normalize: bool = True,
                              colors: Union[str, tuple] = 'tab10',
                              linewidth: float = 0.1,
                              arrow_scale: float = 0.1,
                              head_width: float = 0.01,
                              figsize: tuple = (15, 10),
                              projection_3d: bool = False):
        """
        Plot the attractor dynamics with different visualization options.
        
        Args:
            y: Observation data of shape (T, N)
            burn_in: Number of initial Gibbs samples to discard
            plot_type: Type of plot ('vector_field' or 'trajectory')
            normalize: Whether to normalize the data
            colors: Either a matplotlib colormap name or tuple of color specs
            linewidth: Width of trajectory lines
            arrow_scale: Scaling factor for arrow lengths
            head_width: Width of arrow heads
            figsize: Figure size
            projection_3d: Whether to show 3D projection (for vector_field)
        """
        if y.size == 0:
            raise ValueError("Input data y cannot be empty")
        if y.shape[1] < 2:
            raise ValueError("Data must have at least 2 dimensions")
        if plot_type not in ['vector_field', 'trajectory']:
            raise ValueError("plot_type must be 'vector_field' or 'trajectory'")
        if plot_type == 'trajectory' and y.shape[1] < 3:
            raise ValueError("Trajectory plotting requires at least 3 dimensions")
        if projection_3d and y.shape[1] < 3:
            raise ValueError("3D projection requires 3D data")
        if burn_in >= len(self.z_inf):
            raise ValueError("burn_in must be less than number of samples")

        # Get parameters after burn-in
        M, A_hat, Q, z_mode = self.get_parameters(burn_in=burn_in)
        z = z_mode
        
        # Extract A matrices and bias terms
        A = A_hat[:, :, 1:]  # State matrices
        b = A_hat[:, :, 0]   # Bias terms
        
        # Normalize data if requested
        if normalize:
            y_norm = y.copy()
            for i in range(y.shape[1]):
                y_norm[:, i] = (y[:, i] - np.mean(y[:, i])) / np.std(y[:, i])
        else:
            y_norm = y.copy()
        
        # Calculate fixed points for each state
        x_star = np.stack([np.linalg.solve(np.eye(self.N) - A[k], b[k]) 
                          for k in range(self.K)])
        
        # Handle color mapping
        if isinstance(colors, str):
            # Use colormap
            color_map = plt.get_cmap(colors)
            state_colors = [color_map(k % color_map.N) for k in range(self.K)]
        else:
            # Use provided colors, cycling if needed
            state_colors = [colors[k % len(colors)] for k in range(self.K)]
        
        if plot_type == 'vector_field':
            self._plot_vector_field(y_norm, z, A, b, x_star, 
                                  projection_3d, state_colors, figsize)
        elif plot_type == 'trajectory':
            self._plot_trajectory(y_norm, z, state_colors, linewidth, 
                                arrow_scale, head_width, figsize)
        else:
            raise ValueError("plot_type must be 'vector_field' or 'trajectory'")

    def _plot_vector_field(self, y, z, A, b, x_star, projection_3d, state_colors, figsize):
        """Plot vector field with strict dimension matching and state coloring"""
        if y.shape[1] != self.N:
            raise ValueError(
                f"Data dimension {y.shape[1]} doesn't match "
                f"system dimension {self.N}"
            )
        
        # Calculate vector field components
        U = np.zeros_like(y)
        for l in range(y.shape[0]):
            k = z[l]
            U[l] = A[k] @ y[l] + b[k] - y[l]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        if projection_3d and self.N >= 3:
            ax = fig.add_subplot(111, projection='3d')
            # Plot vector field
            ax.quiver(y[:,0], y[:,1], y[:,2],
                     U[:,0], U[:,1], U[:,2],
                     color='k',
                     length=0.5,
                     lw=0.7,
                     normalize=True)

            for k in range(self.K):
                mask = (z == k)
                ax.scatter(y[mask,0], y[mask,1], y[mask,2],
                         color=state_colors[k], s=10)
            """
            # Plot trajectory with state coloring
            for t in range(y.shape[0]-1):
                ax.plot(y[t:t+2,0], y[t:t+2,1], y[t:t+2,2],
                       color=state_colors[z[t]], lw=0.5)

            # Plot fixed points
            for k in range(self.K):
                ax.scatter(x_star[k,0], x_star[k,1], x_star[k,2],
                         color=state_colors[k], s=100, marker='*', edgecolor='k')
                
            ax.set_zlabel('Z')
            """

        elif self.N >= 2:
            ax = fig.add_subplot(111)
            # Plot vector field
            ax.quiver(y[:,0], y[:,1],
                     U[:,0], U[:,1], 
                     color=[state_colors[k] for k in z],
                     lw=0.7,
                     width=0.01)

            for k in range(self.K):
                mask = (z == k)
                ax.scatter(y[mask,0], y[mask,1],
                         color='k', s=1)
            
            """
            # Plot trajectory with state coloring
            for t in range(y.shape[0]-1):
                ax.plot(y[t:t+2,0], y[t:t+2,1],
                       color=state_colors[z[t]], lw=0.5)
                
                # Add directional arrows
                arr_mean_y1 = np.mean([y[t,0], y[t+1,0]])
                arr_mean_y2 = np.mean([y[t,1], y[t+1,1]])
                dy1 = y[t+1,0] - y[t,0]
                dy2 = y[t+1,1] - y[t,1]
                ax.arrow(arr_mean_y1, arr_mean_y2, 0.01*dy1, 0.01*dy2,
                        shape='full', lw=0, length_includes_head=True,
                        head_width=0.02, color=state_colors[z[t]])
            """
            # Plot fixed points
            for k in range(self.K):
                ax.scatter(x_star[k,0], x_star[k,1],
                         color=state_colors[k], s=10, marker='*', edgecolor='k')
        else:
            raise ValueError(f"Unsupported dimension N={self.N}")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        plt.show()

    def _plot_trajectory(self, y, z, state_colors, linewidth, arrow_scale, head_width, figsize):
        """Plot the normalized attractor trajectory with state coloring."""
        if y.shape[1] < 3:
            raise ValueError("Trajectory plotting requires at least 3 dimensions")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot the three 2D projections
        projections = [(0,1), (1,2), (2,0)]  # (x,y), (y,z), (z,x)
        for idx, (i,j) in enumerate(projections):

            # Add directional arrows
            for t in range(y.shape[0]-1):
                axes[idx].plot(y[t:t+2, i], y[t:t+2, j], lw=linewidth, color=state_colors[z[t]])
                # Calculate midpoint for arrow base
                mid_x = np.mean([y[t, i], y[t+1, i]])
                mid_y = np.mean([y[t, j], y[t+1, j]])
                
                # Calculate direction vector
                dx = y[t+1, i] - y[t, i]
                dy = y[t+1, j] - y[t, j]
                
                # Plot arrow
                axes[idx].arrow(mid_x, mid_y, 
                        arrow_scale*dx, arrow_scale*dy,
                        shape='full', lw=0.1, length_includes_head=True,
                        head_width=head_width, color=state_colors[z[t]])
        
                # Format plot
                axes[idx].set_xlabel("Dimension 1", fontsize=15)
                axes[idx].set_ylabel("Dimension 2", fontsize=15)
                axes[idx].set_title("Normalized Attractor Projections", fontsize=20)
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
