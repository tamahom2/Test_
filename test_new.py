import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from scipy.stats import norm



class TTMatrix:
    """
    Simplified implementation of a matrix in TT format.
    This is a helper class for the main tensor train implementation.
    """
    def __init__(self, cores):
        self.cores = cores
    
    def to_full(self):
        """Convert TT-matrix to full matrix"""
        result = self.cores[0]
        for i in range(1, len(self.cores)):
            n1, n2, r = result.shape
            r1, m1, m2, r2 = self.cores[i].shape
            
            # Reshape and transpose
            result = result.reshape(n1 * n2, r)
            core = self.cores[i].reshape(r1, m1 * m2 * r2)
            
            # Matrix multiplication
            result = result @ core
            
            # Reshape result
            result = result.reshape(n1, n2, m1, m2, r2)
            
            # Permute dimensions
            result = np.transpose(result, (0, 2, 1, 3, 4))
            
            # Combine dimensions
            result = result.reshape(n1 * m1, n2 * m2, r2)
        
        # Final shape
        n1, n2, r = result.shape
        return result.reshape(n1, n2)


class TensorTrain:
    """
    Implementation of a tensor in TT format.
    Based on the tensor train decomposition algorithm from the paper.
    """
    def __init__(self, cores=None):
        self.cores = cores if cores else []
    
    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self.cores)
    
    @property
    def shape(self):
        """Shape of the tensor"""
        if not self.cores:
            return ()
        return tuple(core.shape[1] for core in self.cores)
    
    @property
    def ranks(self):
        """TT-ranks of the tensor"""
        if not self.cores:
            return (1, 1)
        ranks = [1]
        for core in self.cores:
            ranks.append(core.shape[2])
        return tuple(ranks)
    
    def __getitem__(self, indices):
        """Get element at specified indices"""
        if not self.cores:
            return None
        
        if not hasattr(indices, '__len__'):
            indices = (indices,)
        
        # Start with the first core's slice
        result = self.cores[0][:, indices[0], :]
        
        # Multiply with remaining cores
        for i in range(1, len(indices)):
            result = result @ self.cores[i][:, indices[i], :]
        
        return result[0, 0]  # Return scalar value
    
    def to_full(self):
        """Convert from TT format to full tensor"""
        if not self.cores:
            return None
        
        # Start with the first core
        result = self.cores[0]
        
        # Contract with remaining cores
        for i in range(1, len(self.cores)):
            # Contract along the last dimension of result and first dimension of next core
            result = np.tensordot(result, self.cores[i], axes=([-1], [0]))
        
        # Remove singleton dimensions at start and end if present
        if result.shape[0] == 1:
            result = result[0]
        if result.shape[-1] == 1:
            result = result[..., 0]
        
        return result
    
    @staticmethod
    def from_tensor(tensor, epsilon=1e-10, max_rank=np.inf):
        """Convert a full tensor to TT format using TT-SVD algorithm"""
        # Based on Algorithm 1 from the paper
        d = tensor.ndim
        n = tensor.shape
        
        # Prepare for TT-SVD
        cores = []
        r = [1] * (d + 1)
        
        # Reshape tensor
        curr_tensor = tensor.reshape(1, -1)
        
        # TT-SVD main loop
        for k in range(d-1):
            # Reshape to matrix
            curr_shape = r[k] * n[k]
            curr_tensor = curr_tensor.reshape(curr_shape, -1)
            
            # SVD
            u, s, vh = np.linalg.svd(curr_tensor, full_matrices=False)
            
            # Truncate SVD
            delta = epsilon / np.sqrt(d-1) * np.linalg.norm(s)
            r[k+1] = min(max_rank, np.sum(s > delta))
            
            # Create core
            cores.append(u[:, :r[k+1]].reshape(r[k], n[k], r[k+1]))
            
            # Next tensor
            curr_tensor = np.diag(s[:r[k+1]]) @ vh[:r[k+1], :]
        
        # Last core
        cores.append(curr_tensor.reshape(r[d-1], n[d-1], r[d]))
        
        return TensorTrain(cores)
    
    def inner_product(self, vectors):
        """
        Compute inner product between the TT tensor and a set of vectors,
        one for each dimension.
        """
        # Check input
        if len(vectors) != self.ndim:
            raise ValueError(f"Expected {self.ndim} vectors, got {len(vectors)}")
        
        # Initialize result matrix
        result = np.ones((1, 1))
        
        # Compute contractions
        for i, vector in enumerate(vectors):
            # Get core
            core = self.cores[i]
            
            # Contract core with vector
            temp = np.tensordot(core, vector, axes=([1], [0]))
            
            # Contract with accumulated result
            result = result @ temp
        
        return result[0, 0]  # Return scalar value


class ChebyshevInterpolation:
    """
    Implementation of Algorithm 4: Combined methodology for Chebyshev interpolation 
    in parametric option pricing.
    
    From the paper "Low-rank tensor approximation for Chebyshev interpolation in 
    parametric option pricing" by Glau, Kressner, and Statti.
    """
    
    def __init__(self, domains, degrees):
        """
        Initialize the Chebyshev interpolation.
        
        Parameters:
        -----------
        domains : list of tuples
            List of (min, max) tuples defining the domain in each dimension.
        degrees : list of int
            List of degrees for Chebyshev polynomials in each dimension.
        """
        self.domains = domains
        self.degrees = degrees
        self.dims = len(domains)
        
        # Tensors for coefficients and prices
        self.P = None  # Tensor of prices at Chebyshev nodes
        self.C = None  # Tensor of Chebyshev coefficients in TT format
        
        # TT decomposition information
        self.tt_ranks = None
        
        # Generate Chebyshev points for each dimension
        self.points = []
        self.cheb_points = []
        
        for i, (a, b) in enumerate(domains):
            N = degrees[i]
            # Chebyshev nodes in [-1, 1]
            k = np.arange(N+1)
            z = np.cos(np.pi * k / N)
            
            # Map to domain [a, b]
            x = 0.5 * (b - a) * (z + 1) + a
            
            self.cheb_points.append(z)  # Store points in [-1, 1]
            self.points.append(x)       # Store mapped points
        
        # Cache for Chebyshev polynomials
        self._poly_cache = {}
    
    def _chebyshev_polynomial(self, n, x):
        """
        Evaluate Chebyshev polynomial T_n(x).
        
        Parameters:
        -----------
        n : int
            Degree of the polynomial.
        x : float
            Point at which to evaluate the polynomial, in [-1, 1].
            
        Returns:
        --------
        float
            Value of T_n(x).
        """
        # Create a hashable key for the cache
        key = (n, float(x))
        
        # Check if result is in cache
        if key in self._poly_cache:
            return self._poly_cache[key]
        
        # Calculate the result
        if n == 0:
            result = 1.0
        elif n == 1:
            result = float(x)
        else:
            # Use recurrence relation: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
            Tnm2 = 1.0  # T_0(x)
            Tnm1 = float(x)  # T_1(x)
            
            for i in range(2, n+1):
                result = 2.0 * x * Tnm1 - Tnm2
                Tnm2 = Tnm1
                Tnm1 = result
        
        # Store in cache (limit cache size)
        if len(self._poly_cache) < 10000:
            self._poly_cache[key] = result
            
        return result
    
    def construct_tensor_P(self, ref_method, subset_indices=None):
        """
        Construct tensor P containing prices at Chebyshev grid points.
        This implements Step 2 of Algorithm 4 (offline phase).
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_indices : list or None
            List of indices where to evaluate the reference method. If None,
            evaluate at all Chebyshev nodes.
            
        Returns:
        --------
        numpy.ndarray
            Tensor P with prices at Chebyshev nodes.
        """
        print("Offline Phase - Step 2: Computing reference prices at Chebyshev nodes")
        start_time = time.time()
        
        # Initialize tensor P
        P_shape = tuple(d + 1 for d in self.degrees)
        P = np.zeros(P_shape)
        
        # If no subset provided, use all indices
        if subset_indices is None:
            # Generate all indices
            all_indices = list(product(*[range(d + 1) for d in self.degrees]))
            subset_indices = all_indices
        
        total_points = len(subset_indices)
        print(f"Computing {total_points} prices...")
        
        # Evaluate at each point in the subset
        for i, indices in enumerate(subset_indices):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (total_points - i)
                print(f"  Progress: {i}/{total_points} ({i/total_points*100:.1f}%)")
                print(f"  Estimated time remaining: {remaining:.1f} seconds")
            
            # Construct parameter point
            param_point = tuple(self.points[dim][idx] for dim, idx in enumerate(indices))
            
            # Evaluate reference pricing method
            price = ref_method(param_point)
            
            # Store in tensor P
            P[indices] = price
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        self.P = P
        return P
    
    def tt_decomposition(self, tensor, epsilon=1e-10, max_rank=np.inf):
        """
        Perform TT decomposition on a tensor.
        This implements Step 3 of Algorithm 4 (offline phase).
        
        Parameters:
        -----------
        tensor : numpy.ndarray
            Input tensor.
        epsilon : float
            Desired accuracy.
        max_rank : int
            Maximum TT-rank.
            
        Returns:
        --------
        TensorTrain
            Tensor in TT format.
        """
        print("Offline Phase - Step 3: Performing TT decomposition")
        start_time = time.time()
        
        # Perform TT-SVD
        tt_tensor = TensorTrain.from_tensor(tensor, epsilon, max_rank)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        # Store TT-ranks
        self.tt_ranks = tt_tensor.ranks
        print(f"TT-ranks: {self.tt_ranks}")
        
        return tt_tensor
    
    def _construct_basis_matrix(self, n):
        """
        Construct the Chebyshev basis matrix F_n as described in equation (11).
        
        Parameters:
        -----------
        n : int
            Interpolation order (degree of Chebyshev polynomial)
        
        Returns:
        --------
        np.ndarray
            Chebyshev basis matrix F_n
        """
        F_n = np.zeros((n+1, n+1))
        
        for j in range(n+1):
            for k in range(n+1):
                # Factor for all terms
                factor = 2.0 / n
                
                # Additional factor for edge coefficients
                if j == 0 or j == n:
                    factor *= 0.5
                if(k==0 or k==n):
                    factor *= 0.5
                # Compute cosine term
                angle = j * np.pi * k / n
                F_n[j, k] = factor * np.cos(angle)
        
        return F_n
    
    def _mode_multiply(self, tensor, matrix, mode):
        """
        Perform mode-m multiplication of a tensor with a matrix.
        
        Parameters:
        -----------
        tensor : np.ndarray
            Input tensor
        matrix : np.ndarray
            Matrix to multiply with
        mode : int
            Mode along which to multiply
            
        Returns:
        --------
        np.ndarray
            Result of mode multiplication
        """
        # Get tensor shape
        shape = tensor.shape
        
        # Reshape tensor for matrix multiplication
        # Move the mode to the first dimension
        tensor_reshaped = np.moveaxis(tensor, mode, 0)
        # Reshape to 2D
        tensor_flat = tensor_reshaped.reshape(shape[mode], -1)
        
        # Perform matrix multiplication
        result_flat = matrix @ tensor_flat
        
        # Reshape back to tensor
        result_shape = list(shape)
        result_shape[mode] = matrix.shape[0]
        result_reshaped = result_flat.reshape([matrix.shape[0]] + list(tensor_reshaped.shape[1:]))
        result = np.moveaxis(result_reshaped, 0, mode)
        
        return result
    
    def construct_tensor_C(self):
        """
        Construct tensor C containing Chebyshev coefficients.
        This implements Step 4 of Algorithm 4 (offline phase).
        
        Returns:
        --------
        TensorTrain
            Tensor C with Chebyshev coefficients in TT format.
        """
        print("Offline Phase - Step 4: Computing Chebyshev coefficients")
        start_time = time.time()
        
        if self.P is None:
            raise ValueError("Tensor P not constructed. Call construct_tensor_P first.")
        
        # Initialize C as a copy of P
        C = self.P.copy()
        
        # Compute Chebyshev coefficients using efficient algorithm
        # as described in Section 2.4.2 of the paper
        
        # Loop through each dimension/mode
        for m in range(self.dims):
            # Construct the basis matrix F_n for this dimension
            F_n = self._construct_basis_matrix(self.degrees[m])
            
            # Perform mode-m multiplication: C = C ×_m F_n
            C = self._mode_multiply(C, F_n, m)
        
        # Convert to TT format
        C_tt = self.tt_decomposition(C)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        self.C = C_tt
        return C_tt
    
    def evaluate_chebyshev_basis(self, point):
        """
        Evaluate Chebyshev basis at a point (Step 8 of Algorithm 4).
        
        Parameters:
        -----------
        point : tuple
            Point in parameter space at which to evaluate
            
        Returns:
        --------
        list
            List of vectors containing Chebyshev polynomial values for each dimension
        """
        # Handle single point in 1D case
        if isinstance(point, (int, float)) and self.dims == 1:
            point = (point,)
        
        # Map point to [-1, 1]^d
        z = []
        for i, (a, b) in enumerate(self.domains):
            p = point[i]
            z.append(2.0 * (p - a) / (b - a) - 1.0)
        
        # Evaluate Chebyshev polynomials at mapped point
        T_vectors = []
        for dim, z_val in enumerate(z):
            T_dim = np.zeros(self.degrees[dim] + 1)
            for j in range(self.degrees[dim] + 1):
                T_dim[j] = self._chebyshev_polynomial(j, z_val)
            T_vectors.append(T_dim)
        
        return T_vectors
    
    def evaluate_interpolation(self, point):
        """
        Evaluate the Chebyshev interpolation at a point.
        This implements Steps 8-9 of Algorithm 4 (online phase).
        
        Parameters:
        -----------
        point : tuple
            Point in parameter space at which to evaluate
            
        Returns:
        --------
        float
            Interpolated value
        """
        if self.C is None:
            raise ValueError("Tensor C not constructed. Run offline phase first.")
        
        # Step 8: Evaluate Chebyshev basis at the point
        T_vectors = self.evaluate_chebyshev_basis(point)
        
        # Step 9: Compute interpolated price as inner product <C, T_p>
        interpolated_price = self.C.inner_product(T_vectors)
        
        return interpolated_price
    
    def evaluate_batch(self, points):
        """
        Evaluate the Chebyshev interpolation at multiple points.
        
        Parameters:
        -----------
        points : list of tuples
            Points in parameter space at which to evaluate
            
        Returns:
        --------
        numpy.ndarray
            Interpolated values
        """
        if self.C is None:
            raise ValueError("Tensor C not constructed. Run offline phase first.")
        
        # Special case for 1D
        if self.dims == 1 and isinstance(points[0], (int, float)):
            points = [(p,) for p in points]
        
        # Process multiple points
        results = np.zeros(len(points))
        
        print(f"Evaluating at {len(points)} points...")
        start_time = time.time()
        
        for i, point in enumerate(points):
            results[i] = self.evaluate_interpolation(point)
            
            # Progress reporting
            if (i+1) % 100 == 0 or i == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i+1)
                remaining = avg_time * (len(points) - (i+1))
                print(f"  Processed {i+1}/{len(points)} points. Estimated time remaining: {remaining:.1f} seconds")
        
        total_time = time.time() - start_time
        print(f"Batch evaluation completed in {total_time:.2f} seconds")
        
        return results
    
    def run_offline_phase(self, ref_method, subset_size=None, epsilon=1e-10, max_rank=np.inf):
        """
        Run the complete offline phase of Algorithm 4.
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_size : int or None
            Size of the subset of Chebyshev nodes to evaluate. If None,
            evaluate at all nodes.
        epsilon : float
            Desired accuracy for TT decomposition.
        max_rank : int
            Maximum TT-rank.
            
        Returns:
        --------
        TensorTrain
            Tensor C with Chebyshev coefficients in TT format.
        """
        print("=== OFFLINE PHASE ===")
        overall_start = time.time()
        
        # Generate subset of indices if specified
        if subset_size is not None:
            # Generate all possible indices
            all_indices = list(product(*[range(d + 1) for d in self.degrees]))
            total_points = len(all_indices)
            
            if subset_size >= total_points:
                subset_indices = all_indices
            else:
                # Randomly sample indices
                np.random.seed(42)
                sample_idx = np.random.choice(total_points, size=subset_size, replace=False)
                subset_indices = [all_indices[i] for i in sample_idx]
        else:
            subset_indices = None
        
        # Step 2: Construct tensor P
        self.P = self.construct_tensor_P(ref_method, subset_indices)
        
        # Step 4: Construct tensor C (with implicit TT decomposition)
        self.C = self.construct_tensor_C()
        
        overall_elapsed = time.time() - overall_start
        print(f"Offline phase completed in {overall_elapsed:.2f} seconds")
        
        return self.C
    
    def run_online_phase(self, points):
        """
        Run the online phase of Algorithm 4 for one or more points.
        
        Parameters:
        -----------
        points : tuple or list of tuples
            Point(s) in parameter space at which to evaluate
            
        Returns:
        --------
        float or numpy.ndarray
            Interpolated value(s)
        """
        print("=== ONLINE PHASE ===")
        
        # Handle both single point and multiple points
        if isinstance(points, tuple) or (isinstance(points, (int, float)) and self.dims == 1):
            # Single point
            return self.evaluate_interpolation(points)
        else:
            # Multiple points
            return self.evaluate_batch(points)
    
    def run_algorithm4(self, ref_method, evaluation_points, subset_size=None, 
                      epsilon=1e-10, max_rank=np.inf):
        """
        Run the complete Algorithm 4 (offline + online phases).
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        evaluation_points : tuple or list of tuples
            Point(s) in parameter space at which to evaluate
        subset_size : int or None
            Size of the subset of Chebyshev nodes to evaluate. If None,
            evaluate at all nodes.
        epsilon : float
            Desired accuracy for TT decomposition.
        max_rank : int
            Maximum TT-rank.
            
        Returns:
        --------
        float or numpy.ndarray
            Interpolated value(s)
        """
        # Run offline phase
        self.run_offline_phase(ref_method, subset_size, epsilon, max_rank)
        
        # Run online phase
        return self.run_online_phase(evaluation_points)


# Test with sine function
def test_sine_function():
    """
    Test the Chebyshev interpolation with a sine function.
    Plots the exact and interpolated functions for comparison.
    """
    # Define sine function
    def sine_function(x):
        if isinstance(x, tuple):
            x = x[0]  # Extract first element if tuple
        return np.sin(x)
    
    # Domain and degree
    domain = [(-np.pi, np.pi)]  # Domain for x
    degree = [10]              # Degree of Chebyshev polynomial
    
    # Create interpolation
    cheb = ChebyshevInterpolation(domain, degree)
    
    # Run offline phase (construct interpolation)
    cheb.run_offline_phase(sine_function)
    
    # Generate test points for plotting
    x_test = np.linspace(-np.pi, np.pi, 100)
    
    # Evaluate exact function
    y_exact = np.sin(x_test)
    
    # Evaluate interpolation
    y_interp = cheb.evaluate_batch(x_test)
    
    # Calculate error
    max_error = np.max(np.abs(y_exact - y_interp))
    rms_error = np.sqrt(np.mean((y_exact - y_interp)**2))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(x_test, y_exact, 'b-', label='Exact sine')
    plt.plot(x_test, y_interp, 'r--', label='Chebyshev Interpolation')
    plt.scatter(cheb.points[0], [sine_function(x) for x in cheb.points[0]], 
                color='k', marker='o', s=50, label='Chebyshev Nodes')
    plt.grid(True)
    plt.legend()
    plt.title(f'Chebyshev Interpolation of Sine Function (degree={degree[0]})')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    
    # Plot error
    plt.subplot(212)
    plt.plot(x_test, np.abs(y_exact - y_interp), 'g-')
    plt.grid(True)
    plt.title(f'Absolute Error (Max: {max_error:.2e}, RMS: {rms_error:.2e})')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Maximum absolute error: {max_error:.2e}")
    print(f"RMS error: {rms_error:.2e}")
    
    # Show convergence with increasing degree
    degrees = [2, 4, 6, 8, 10, 12, 14]
    max_errors = []
    rms_errors = []
    
    for deg in degrees:
        # Create interpolation with current degree
        cheb = ChebyshevInterpolation(domain, [deg])
        cheb.run_offline_phase(sine_function)
        
        # Evaluate and compute errors
        y_interp = cheb.evaluate_batch(x_test)
        max_errors.append(np.max(np.abs(y_exact - y_interp)))
        rms_errors.append(np.sqrt(np.mean((y_exact - y_interp)**2)))
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(degrees, max_errors, 'bo-', label='Max Error')
    plt.semilogy(degrees, rms_errors, 'ro-', label='RMS Error')
    plt.grid(True)
    plt.legend()
    plt.title('Convergence of Chebyshev Interpolation for Sine Function')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error (log scale)')
    plt.show()




def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call options.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset
        
    Returns:
    --------
    float
        Call option price
    """
    if T <= 0:
        return max(0, S - K)
        
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def test_vanilla_call_option():
    """
    Test the Chebyshev interpolation on a vanilla call option.
    We'll interpolate with respect to two parameters: 
    stock price S and volatility sigma.
    """
    # Fixed parameters
    K = 100  # Strike price
    T = 1.0  # Time to expiration: 1 year
    r = 0.05  # Risk-free rate: 5%
    
    # Define parameter domains
    domains = [(1, 120), (0.1, 0.4)]  # S in [80, 120], sigma in [0.1, 0.4]
    degrees = [25, 25]  # Degree of Chebyshev polynomials
    
    # Create reference method (Black-Scholes)
    def ref_method(params):
        S, sigma = params
        return black_scholes_call(S, K, T, r, sigma)
    
    print("Creating Chebyshev interpolation for vanilla call option...")
    
    cheb = ChebyshevInterpolation(domains, degrees)
    
    # Run offline phase
    start_time = time.time()
    cheb.run_offline_phase(ref_method)
    offline_time = time.time() - start_time
    print(f"Offline phase completed in {offline_time:.2f} seconds")
    
    # Test points
    n_test = 150
    S_test = np.linspace(domains[0][0], domains[0][1], n_test)
    sigma_test = np.linspace(domains[1][0], domains[1][1], n_test)
    
    # Create 2D grid of test points
    S_grid, sigma_grid = np.meshgrid(S_test, sigma_test)
    test_points = []
    for i in range(n_test):
        for j in range(n_test):
            test_points.append((S_grid[i, j], sigma_grid[i, j]))
    
    # Evaluate Black-Scholes (exact)
    start_time = time.time()
    exact_values = np.array([ref_method(p) for p in test_points])
    exact_time = time.time() - start_time
    print(f"Exact Black-Scholes calculation completed in {exact_time:.2f} seconds")
    
    # Evaluate interpolation
    start_time = time.time()
    interp_values = cheb.evaluate_batch(test_points)
    interp_time = time.time() - start_time
    print(f"Interpolation evaluation completed in {interp_time:.2f} seconds")
    print(f"Speed-up factor in online phase: {exact_time/interp_time:.1f}x")
    
    # Calculate errors
    errors = np.abs(exact_values - interp_values)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    rms_error = np.sqrt(np.mean(errors**2))
    rel_error = np.mean(errors / np.abs(exact_values)) * 100
    
    print(f"Maximum absolute error: {max_error:.6f}")
    print(f"Mean absolute error: {mean_error:.6f}")
    print(f"RMS error: {rms_error:.6f}")
    print(f"Mean relative error: {rel_error:.6f}%")
    
    # Reshape results for plotting
    Z_exact = np.zeros((n_test, n_test))
    Z_interp = np.zeros((n_test, n_test))
    Z_error = np.zeros((n_test, n_test))
    
    idx = 0
    for i in range(n_test):
        for j in range(n_test):
            Z_exact[i, j] = exact_values[idx]
            Z_interp[i, j] = interp_values[idx]
            Z_error[i, j] = errors[idx]
            idx += 1
    
    # Create plots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot exact values
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(S_grid, sigma_grid, Z_exact, cmap='viridis', alpha=0.8)
    ax1.set_title('Exact Black-Scholes')
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Volatility (σ)')
    ax1.set_zlabel('Option Price')
    
    # Plot interpolated values
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(S_grid, sigma_grid, Z_interp, cmap='plasma', alpha=0.8)
    ax2.set_title('Chebyshev Interpolation')
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Volatility (σ)')
    ax2.set_zlabel('Option Price')
    
    # Plot error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(S_grid, sigma_grid, Z_error, cmap='hot', alpha=0.8)
    ax3.set_title(f'Absolute Error (Max: {max_error:.6f})')
    ax3.set_xlabel('Stock Price (S)')
    ax3.set_ylabel('Volatility (σ)')
    ax3.set_zlabel('|Error|')
    
    plt.tight_layout()
    plt.show()
    
    # Also show 2D slice at fixed volatility
    fixed_sigma_idx = n_test // 2  # Middle of volatility range
    fixed_sigma = sigma_test[fixed_sigma_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_test, Z_exact[:, fixed_sigma_idx], 'b-', label='Exact')
    plt.plot(S_test, Z_interp[:, fixed_sigma_idx], 'r--', label='Interpolated')
    plt.fill_between(S_test, 
                     Z_exact[:, fixed_sigma_idx] - Z_error[:, fixed_sigma_idx],
                     Z_exact[:, fixed_sigma_idx] + Z_error[:, fixed_sigma_idx],
                     color='gray', alpha=0.3, label='Error')
    plt.grid(True)
    plt.legend()
    plt.title(f'Call Option Price vs Stock Price (σ = {fixed_sigma:.2f})')
    plt.xlabel('Stock Price (S)')
    plt.ylabel('Option Price')
    plt.show()
    
    # Show relative speedup for batch evaluation
    print("\nBatch evaluation performance:")
    print(f"Black-Scholes: {exact_time:.4f} seconds for {len(test_points)} points")
    print(f"Interpolation: {interp_time:.4f} seconds for {len(test_points)} points")
    print(f"Speedup factor: {exact_time/interp_time:.1f}x")

def test_vanilla_call_option_1d():
    """
    Test the Chebyshev interpolation on a vanilla call option in 1D.
    We'll interpolate with respect to stock price S while fixing other parameters.
    """
    # Fixed parameters
    K = 100      # Strike price
    T = 1.0      # Time to expiration: 1 year
    r = 0.05     # Risk-free rate: 5%
    sigma = 0.2  # Volatility: 20%
    
    # Define parameter domain for stock price
    domain = [(70, 130)]  # S in [70, 130]
    
    # Test different polynomial degrees
    degrees_to_test = [4, 8, 12, 16]
    
    # Create reference method (Black-Scholes)
    def ref_method(params):
        if isinstance(params, tuple):
            S = params[0]
        else:
            S = params
        return black_scholes_call(S, K, T, r, sigma)
    
    print("Testing 1D Chebyshev interpolation for vanilla call option...")
    
    # Create test points for evaluation
    S_test = np.linspace(domain[0][0], domain[0][1], 200)
    
    # Calculate exact values
    start_time = time.time()
    exact_values = np.array([ref_method(S) for S in S_test])
    exact_time = time.time() - start_time
    print(f"Exact Black-Scholes calculation completed in {exact_time:.4f} seconds")
    
    # Create figure for results
    plt.figure(figsize=(15, 10))
    
    # Plot exact values
    plt.subplot(221)
    plt.plot(S_test, exact_values, 'b-', linewidth=2, label='Exact')
    plt.grid(True)
    plt.axvline(x=K, color='k', linestyle='--', label='Strike (K)')
    plt.xlabel('Stock Price (S)')
    plt.ylabel('Option Price')
    plt.title('Black-Scholes Call Option Price')
    plt.legend()
    
    # Prepare plot for error comparison
    plt.subplot(222)
    
    # Test different degrees
    error_stats = []
    for degree in degrees_to_test:
        # Create interpolation
        cheb = ChebyshevInterpolation(domain, [degree])
        
        # Run offline phase
        start_time = time.time()
        cheb.run_offline_phase(ref_method)
        offline_time = time.time() - start_time
        
        # Evaluate interpolation
        start_time = time.time()
        interp_values = cheb.evaluate_batch(S_test)
        interp_time = time.time() - start_time
        
        # Calculate errors
        errors = np.abs(exact_values - interp_values)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        rms_error = np.sqrt(np.mean(errors**2))
        rel_error = np.mean(errors / np.abs(exact_values)) * 100
        
        error_stats.append({
            'degree': degree,
            'max_error': max_error,
            'mean_error': mean_error,
            'rms_error': rms_error,
            'rel_error': rel_error,
            'offline_time': offline_time,
            'interp_time': interp_time
        })
        
        # Plot error
        plt.subplot(222)
        plt.semilogy(S_test, errors, label=f'Degree {degree}')
    
    # Finalize error plot
    plt.subplot(222)
    plt.grid(True)
    plt.xlabel('Stock Price (S)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Interpolation Error by Polynomial Degree')
    plt.axvline(x=K, color='k', linestyle='--')
    plt.legend()
    
    # Plot convergence of errors
    plt.subplot(223)
    plt.semilogy(degrees_to_test, [stat['max_error'] for stat in error_stats], 'ro-', label='Max Error')
    plt.semilogy(degrees_to_test, [stat['rms_error'] for stat in error_stats], 'bo-', label='RMS Error')
    plt.grid(True)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error (log scale)')
    plt.title('Error Convergence with Polynomial Degree')
    plt.legend()
    
    # Plot timing comparison
    plt.subplot(224)
    offline_times = [stat['offline_time'] for stat in error_stats]
    interp_times = [stat['interp_time'] for stat in error_stats]
    
    # Bar width
    width = 0.35
    x = np.arange(len(degrees_to_test))
    
    plt.bar(x - width/2, offline_times, width, label='Offline Phase')
    plt.bar(x + width/2, interp_times, width, label='Online Phase')
    plt.axhline(y=exact_time, color='r', linestyle='--', label='Black-Scholes')
    
    plt.xticks(x, degrees_to_test)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print("\nError statistics by polynomial degree:")
    print("-" * 75)
    print(f"{'Degree':^7} | {'Max Error':^12} | {'Mean Error':^12} | {'RMS Error':^12} | {'Rel Error (%)':^12}")
    print("-" * 75)
    for stat in error_stats:
        print(f"{stat['degree']:^7} | {stat['max_error']:^12.8f} | {stat['mean_error']:^12.8f} | "
              f"{stat['rms_error']:^12.8f} | {stat['rel_error']:^12.8f}")
    
    # Print timing statistics
    print("\nTiming statistics (seconds):")
    print("-" * 70)
    print(f"{'Degree':^7} | {'Offline Phase':^14} | {'Online Phase':^14} | {'Speedup':^10}")
    print("-" * 70)
    for stat in error_stats:
        speedup = exact_time / stat['interp_time']
        print(f"{stat['degree']:^7} | {stat['offline_time']:^14.6f} | {stat['interp_time']:^14.6f} | {speedup:^10.2f}x")
    
    # Show detailed plot for best degree
    best_idx = np.argmin([stat['rms_error'] for stat in error_stats])
    best_degree = degrees_to_test[best_idx]
    
    # Recreate best interpolation
    cheb = ChebyshevInterpolation(domain, [best_degree])
    cheb.run_offline_phase(ref_method)
    best_interp = cheb.evaluate_batch(S_test)
    best_errors = np.abs(exact_values - best_interp)
    
    plt.figure(figsize=(12, 6))
    
    # Price comparison
    plt.subplot(121)
    plt.plot(S_test, exact_values, 'b-', label='Exact')
    plt.plot(S_test, best_interp, 'r--', label='Interpolation')
    plt.scatter(cheb.points[0], [ref_method(x) for x in cheb.points[0]], 
                color='k', marker='o', s=50, label='Chebyshev Nodes')
    plt.grid(True)
    plt.axvline(x=K, color='k', linestyle='--', label='Strike (K)')
    plt.xlabel('Stock Price (S)')
    plt.ylabel('Option Price')
    plt.title(f'Call Option Price (Degree {best_degree})')
    plt.legend()
    
    # Error plot
    plt.subplot(122)
    plt.semilogy(S_test, best_errors, 'g-')
    plt.grid(True)
    plt.axvline(x=K, color='k', linestyle='--', label='Strike (K)')
    plt.xlabel('Stock Price (S)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title(f'Error (Max: {np.max(best_errors):.8f})')
    
    plt.tight_layout()
    plt.show()

# Run the test
if __name__ == "__main__":
    test_sine_function()
    test_vanilla_call_option()
    test_vanilla_call_option_1d()
