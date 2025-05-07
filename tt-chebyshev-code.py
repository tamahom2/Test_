import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
import time
from itertools import product
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib.ticker import ScalarFormatter


class TTCore:
    """
    A single core in the TT-decomposition.
    """
    def __init__(self, data):
        """
        Initialize a TT-core with data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            3D array of shape (r_left, n, r_right)
        """
        self.data = np.array(data)
        self.shape = self.data.shape
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    @property
    def rank_left(self):
        return self.shape[0]
    
    @property
    def rank_right(self):
        return self.shape[2]
    
    @property
    def mode_size(self):
        return self.shape[1]


class TTTensor:
    """
    Tensor Train decomposition for multi-dimensional tensors.
    """
    def __init__(self, cores=None):
        """
        Initialize a TT-tensor with a list of cores.
        
        Parameters:
        -----------
        cores : list of TTCore or None
            List of TT-cores
        """
        self.cores = cores if cores is not None else []
        
    def __getitem__(self, idx):
        return self.cores[idx]
    
    def __setitem__(self, idx, value):
        self.cores[idx] = value if isinstance(value, TTCore) else TTCore(value)
    
    def append(self, core):
        """
        Append a core to the TT-tensor.
        
        Parameters:
        -----------
        core : TTCore or numpy.ndarray
            TT-core to append
        """
        self.cores.append(core if isinstance(core, TTCore) else TTCore(core))
    
    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self.cores)
    
    @property
    def ranks(self):
        """TT-ranks"""
        if not self.cores:
            return []
        
        r = [self.cores[0].rank_left]
        for core in self.cores:
            r.append(core.rank_right)
        return r
    
    @property
    def shape(self):
        """Shape of the full tensor"""
        if not self.cores:
            return ()
        
        return tuple(core.mode_size for core in self.cores)
    
    def copy(self):
        """Create a deep copy"""
        new_cores = [TTCore(core.data.copy()) for core in self.cores]
        return TTTensor(new_cores)
    
    def to_full(self):
        """
        Convert to full tensor representation.
        
        Returns:
        --------
        numpy.ndarray
            Full tensor
        """
        if not self.cores:
            return None
        
        # Start with the first core
        result = self.cores[0].data
        
        # Contract with each subsequent core
        for i in range(1, self.ndim):
            # Reshape for contraction
            result = np.reshape(result, (-1, self.cores[i-1].rank_right))
            core_mat = np.reshape(self.cores[i].data, 
                                 (self.cores[i].rank_left, -1))
            
            # Contract
            result = np.matmul(result, core_mat)
            
            # Reshape to include the new dimension
            result = np.reshape(result, (-1, self.cores[i].mode_size, 
                                       self.cores[i].rank_right))
        
        # Final reshape to get the full tensor shape
        result = np.reshape(result, self.shape)
        
        return result
    
    def orthogonalize(self, target_idx=None):
        """
        Orthogonalize the TT-tensor.
        
        Parameters:
        -----------
        target_idx : int or None
            Target index for orthogonalization. If None, orthogonalize to the right.
        
        Returns:
        --------
        TTTensor
            Orthogonalized TT-tensor
        """
        if not self.cores or self.ndim <= 1:
            return self.copy()
        
        if target_idx is None:
            target_idx = self.ndim - 1
        
        # Create a deep copy to avoid modifying the original tensor
        result = self.copy()
        
        # Left-to-right orthogonalization
        for i in range(target_idx):
            # Reshape core to matrix
            core_mat = np.reshape(result.cores[i].data, 
                                 (result.cores[i].rank_left * result.cores[i].mode_size, 
                                  result.cores[i].rank_right))
            
            # QR decomposition
            Q, R = np.linalg.qr(core_mat, mode='reduced')
            
            # Update current core
            result.cores[i].data = np.reshape(Q, 
                                             (result.cores[i].rank_left, 
                                              result.cores[i].mode_size, 
                                              Q.shape[1]))
            
            # Update next core
            next_core = np.reshape(result.cores[i+1].data, 
                                   (result.cores[i+1].rank_left, -1))
            next_core = np.matmul(R, next_core)
            result.cores[i+1].data = np.reshape(next_core, 
                                               (R.shape[0], 
                                                result.cores[i+1].mode_size, 
                                                result.cores[i+1].rank_right))
        
        # Right-to-left orthogonalization
        for i in range(self.ndim - 1, target_idx, -1):
            # Reshape core to matrix
            core_mat = np.reshape(result.cores[i].data, 
                                 (result.cores[i].rank_left, 
                                  result.cores[i].mode_size * result.cores[i].rank_right))
            core_mat = core_mat.T  # Transpose for RQ decomposition
            
            # QR decomposition (equivalent to RQ when transposed)
            Q, R = np.linalg.qr(core_mat, mode='reduced')
            R = R.T  # Transpose back
            Q = Q.T  # Transpose back
            
            # Update current core
            result.cores[i].data = np.reshape(Q, 
                                             (Q.shape[0], 
                                              result.cores[i].mode_size, 
                                              result.cores[i].rank_right))
            
            # Update previous core
            prev_core = np.reshape(result.cores[i-1].data, 
                                   (-1, result.cores[i-1].rank_right))
            prev_core = np.matmul(prev_core, R)
            result.cores[i-1].data = np.reshape(prev_core, 
                                               (result.cores[i-1].rank_left, 
                                                result.cores[i-1].mode_size, 
                                                R.shape[1]))
        
        return result
    
    def contract_with_vectors(self, vectors):
        """
        Contract the TT-tensor with vectors along each mode.
        
        Parameters:
        -----------
        vectors : list of numpy.ndarray
            List of vectors to contract with each mode
            
        Returns:
        --------
        float
            Result of contraction
        """
        if len(vectors) != self.ndim:
            raise ValueError(f"Expected {self.ndim} vectors, got {len(vectors)}")
        
        # Start with the first core
        result = np.einsum('ijk,j->ik', self.cores[0].data, vectors[0])
        
        # Contract with each subsequent core
        for i in range(1, self.ndim):
            result = np.einsum('ij,jkl,k->il', result, self.cores[i].data, vectors[i])
        
        return result[0, 0]


def tt_svd(tensor, max_rank=np.inf, epsilon=1e-10):
    """
    TT-SVD algorithm for decomposing a full tensor into TT format.
    
    Parameters:
    -----------
    tensor : numpy.ndarray
        Full tensor to decompose
    max_rank : int or float
        Maximum TT-rank
    epsilon : float
        Approximation accuracy
        
    Returns:
    --------
    TTTensor
        TT-decomposition of the tensor
    """
    # Get tensor shape
    shape = tensor.shape
    d = len(shape)
    
    # Create result tensor
    result = TTTensor()
    
    # Handle 1D case
    if d == 1:
        data = tensor.reshape(1, -1, 1)
        result.append(TTCore(data))
        return result
    
    # Initialize for TT-SVD
    C = tensor.copy()
    r = 1  # First rank is always 1
    
    # Iterate through dimensions
    for k in range(d - 1):
        # Reshape C
        C_mat = C.reshape(r * shape[k], -1)
        
        # SVD
        U, S, Vt = np.linalg.svd(C_mat, full_matrices=False)
        
        # Truncate
        delta = epsilon / np.sqrt(d - k - 1) * np.linalg.norm(S)
        r_new = min(max_rank, np.sum(S > delta))
        r_new = max(r_new, 1)  # Ensure at least rank 1
        
        # Create core
        core_data = U[:, :r_new].reshape(r, shape[k], r_new)
        result.append(TTCore(core_data))
        
        # Update C for next iteration
        C = np.diag(S[:r_new]) @ Vt[:r_new, :]
        r = r_new
    
    # Last core
    core_data = C.reshape(r, shape[-1], 1)
    result.append(TTCore(core_data))
    
    return result


def tt_round(tt_tensor, max_rank=np.inf, epsilon=1e-10):
    """
    Truncate a TT-tensor to lower ranks while maintaining accuracy.
    
    Parameters:
    -----------
    tt_tensor : TTTensor
        Tensor to truncate
    max_rank : int or float
        Maximum TT-rank after truncation
    epsilon : float
        Desired approximation accuracy
        
    Returns:
    --------
    TTTensor
        Truncated TT-tensor
    """
    # Orthogonalize
    tt_result = tt_tensor.orthogonalize()
    d = tt_result.ndim
    
    # Iterate from right to left (except the first core)
    for k in range(d - 1, 0, -1):
        # Get core
        core = tt_result.cores[k]
        
        # Reshape to matrix
        core_mat = np.reshape(core.data, (core.rank_left, core.mode_size * core.rank_right))
        
        # SVD
        U, S, Vt = np.linalg.svd(core_mat, full_matrices=False)
        
        # Truncate
        delta = epsilon / np.sqrt(d) * np.linalg.norm(S)
        r_new = min(max_rank, np.sum(S > delta))
        r_new = max(r_new, 1)  # Ensure at least rank 1
        
        # Update core
        tt_result.cores[k].data = np.reshape(Vt[:r_new, :], 
                                       (r_new, core.mode_size, core.rank_right))
        
        # Update previous core
        prev_core = tt_result.cores[k-1]
        prev_core_data = np.reshape(prev_core.data, 
                                    (prev_core.rank_left * prev_core.mode_size, prev_core.rank_right))
        prev_core_data = prev_core_data @ U[:, :r_new] @ np.diag(S[:r_new])
        tt_result.cores[k-1].data = np.reshape(prev_core_data,
                                         (prev_core.rank_left, prev_core.mode_size, r_new))
    
    return tt_result


def tt_elem(tt_tensor, indices):
    """
    Compute specific elements of a TT-tensor.
    
    Parameters:
    -----------
    tt_tensor : TTTensor
        TT-tensor
    indices : list or numpy.ndarray
        Multi-indices of the elements to compute
        
    Returns:
    --------
    float or numpy.ndarray
        Values of the tensor at the specified indices
    """
    if isinstance(indices[0], (list, tuple, np.ndarray)):
        # Multiple indices
        return np.array([tt_elem(tt_tensor, idx) for idx in indices])
    
    # Single index
    result = tt_tensor.cores[0].data[0, indices[0], :]
    
    for i in range(1, tt_tensor.ndim):
        result = result @ tt_tensor.cores[i].data[:, indices[i], :]
    
    return result.item()


class ChebyshevInterpolationTT:
    """
    Chebyshev interpolation in TT-format.
    """
    def __init__(self, domain, degrees, max_rank=np.inf, epsilon=1e-10):
        """
        Initialize Chebyshev interpolation with TT-format.
        
        Parameters:
        -----------
        domain : list of tuples
            List of (min, max) tuples defining the domain in each dimension
        degrees : list of int
            Degrees for Chebyshev polynomials in each dimension
        max_rank : int or float
            Maximum TT-rank
        epsilon : float
            Approximation accuracy
        """
        self.domain = domain
        self.dims = len(domain)
        
        if isinstance(degrees, int):
            self.degrees = [degrees] * self.dims
        else:
            self.degrees = degrees
        
        self.max_rank = max_rank
        self.epsilon = epsilon
        
        # Generate Chebyshev points for each dimension
        self.cheby_points = []
        self.mapped_points = []
        
        for i, (a, b) in enumerate(self.domain):
            n = self.degrees[i]
            # Chebyshev nodes in [-1, 1]
            k = np.arange(n + 1)
            z = np.cos(np.pi * k / n)
            
            # Map to domain [a, b]
            x = 0.5 * (b - a) * (z + 1) + a
            
            self.cheby_points.append(z)  # Store points in [-1, 1]
            self.mapped_points.append(x)  # Store mapped points
        
        # TT-tensors for values and coefficients
        self.tt_values = None  # TT-tensor of function values at Chebyshev nodes
        self.tt_coeffs = None  # TT-tensor of Chebyshev coefficients
        
        # Cache for polynomial evaluations
        self._poly_cache = {}
    
    def _chebyshev_polynomial(self, n, x):
        """
        Evaluate Chebyshev polynomial T_n(x).
        
        Parameters:
        -----------
        n : int
            Degree of the polynomial
        x : float
            Point at which to evaluate
            
        Returns:
        --------
        float
            Value of T_n(x)
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
            t_prev = 1.0  # T_0(x)
            t_curr = float(x)  # T_1(x)
            
            for i in range(2, n+1):
                t_next = 2.0 * x * t_curr - t_prev
                t_prev = t_curr
                t_curr = t_next
            
            result = t_curr
        
        # Store in cache (limit cache size)
        if len(self._poly_cache) < 10000:
            self._poly_cache[key] = result
            
        return result
    
    def _cheby_basis_matrix(self, degree):
        """
        Construct the Chebyshev basis matrix for computing coefficients.
        
        Parameters:
        -----------
        degree : int
            Degree of the polynomial
            
        Returns:
        --------
        numpy.ndarray
            Chebyshev basis matrix
        """
        n = degree + 1
        F = np.zeros((n, n))
        
        for j in range(n):
            for k in range(n):
                # Factor for all terms
                factor = 2.0 / degree
                
                # Additional factor for edge coefficients
                if j == 0 or j == degree:
                    factor *= 0.5
                    
                # Compute cosine term
                angle = j * np.pi * k / degree
                F[j, k] = factor * np.cos(angle)
        
        return F
    
    def _map_to_unit(self, points, dim_idx=None):
        """
        Map points from domain to [-1, 1].
        
        Parameters:
        -----------
        points : float or numpy.ndarray
            Points to map
        dim_idx : int or None
            Dimension index. If None, apply to all dimensions.
            
        Returns:
        --------
        float or numpy.ndarray
            Mapped points
        """
        if dim_idx is not None:
            a, b = self.domain[dim_idx]
            return 2.0 * (points - a) / (b - a) - 1.0
        
        # Apply to all dimensions
        result = []
        for i, (a, b) in enumerate(self.domain):
            result.append(2.0 * (points[i] - a) / (b - a) - 1.0)
        
        return np.array(result)
    
    def _map_from_unit(self, points, dim_idx=None):
        """
        Map points from [-1, 1] to domain.
        
        Parameters:
        -----------
        points : float or numpy.ndarray
            Points to map
        dim_idx : int or None
            Dimension index. If None, apply to all dimensions.
            
        Returns:
        --------
        float or numpy.ndarray
            Mapped points
        """
        if dim_idx is not None:
            a, b = self.domain[dim_idx]
            return 0.5 * (b - a) * (points + 1.0) + a
        
        # Apply to all dimensions
        result = []
        for i, (a, b) in enumerate(self.domain):
            result.append(0.5 * (b - a) * (points[i] + 1.0) + a)
        
        return np.array(result)
    
    def _eval_basis(self, point):
        """
        Evaluate Chebyshev basis functions at a point.
        
        Parameters:
        -----------
        point : list or numpy.ndarray
            Point in parameter space
            
        Returns:
        --------
        list of numpy.ndarray
            List of basis function values for each dimension
        """
        # Map point to [-1, 1] for each dimension
        mapped_point = [self._map_to_unit(point[i], i) for i in range(self.dims)]
        
        # Evaluate basis functions
        basis_values = []
        for dim, x in enumerate(mapped_point):
            dim_values = np.zeros(self.degrees[dim] + 1)
            for j in range(self.degrees[dim] + 1):
                dim_values[j] = self._chebyshev_polynomial(j, x)
            basis_values.append(dim_values)
        
        return basis_values
    
    def _generate_grid_indices(self):
        """
        Generate all grid indices for the tensorized Chebyshev grid.
        
        Returns:
        --------
        list of tuple
            List of multi-indices for the grid
        """
        ranges = [range(d + 1) for d in self.degrees]
        return list(product(*ranges))
    
    def construct_tensor_values(self, func, subset_indices=None):
        """
        Construct a tensor of function values at Chebyshev nodes.
        
        Parameters:
        -----------
        func : callable
            Function to interpolate, takes a point in parameter space and returns a scalar
        subset_indices : list of tuple or None
            Subset of grid indices to evaluate. If None, evaluate at all points.
            
        Returns:
        --------
        TTTensor
            TT-tensor of function values
        """
        print("Constructing tensor of function values at Chebyshev nodes...")
        start_time = time.time()
        
        # If no subset provided, use all indices
        if subset_indices is None:
            subset_indices = self._generate_grid_indices()
        
        # Initialize a full tensor to store values
        tensor_shape = tuple(d + 1 for d in self.degrees)
        values_tensor = np.zeros(tensor_shape)
        
        # Evaluate function at each point in the subset
        for indices in subset_indices:
            # Construct parameter point
            if self.dims == 1:
                param_point = self.mapped_points[0][indices]
            else:
                param_point = tuple(self.mapped_points[dim][idx] for dim, idx in enumerate(indices))
            
            # Evaluate function
            value = func(param_point)
            
            # Store in tensor
            values_tensor[indices] = value
        
        # Convert to TT-format
        self.tt_values = tt_svd(values_tensor, self.max_rank, self.epsilon)
        
        elapsed = time.time() - start_time
        print(f"Constructed tensor of values in {elapsed:.2f} seconds")
        
        return self.tt_values
    
    def _mode_multiply(self, tt_tensor, matrix, mode):
        """
        Perform mode-m multiplication of a TT-tensor with a matrix.
        
        Parameters:
        -----------
        tt_tensor : TTTensor
            Input TT-tensor
        matrix : numpy.ndarray
            Matrix to multiply with
        mode : int
            Mode along which to multiply
            
        Returns:
        --------
        TTTensor
            Result of mode multiplication
        """
        result = tt_tensor.copy()
        
        # Get the core for the specified mode
        core = result.cores[mode]
        
        # Reshape core for multiplication
        core_mat = np.reshape(core.data, (core.rank_left, core.mode_size * core.rank_right))
        
        # Perform matrix multiplication
        new_core_mat = matrix @ np.reshape(core_mat.T, (core.mode_size, -1))
        new_core_mat = new_core_mat.T
        
        # Reshape back to core format
        result.cores[mode].data = np.reshape(new_core_mat, 
                                           (core.rank_left, matrix.shape[0], core.rank_right))
        
        return result
    
    def compute_coefficients(self):
        """
        Compute Chebyshev coefficients from function values.
        
        Returns:
        --------
        TTTensor
            TT-tensor of Chebyshev coefficients
        """
        if self.tt_values is None:
            raise ValueError("Tensor of values not constructed. Call construct_tensor_values first.")
        
        print("Computing Chebyshev coefficients...")
        start_time = time.time()
        
        # Start with a copy of the values tensor
        self.tt_coeffs = self.tt_values.copy()
        
        # Compute Chebyshev coefficients for each dimension
        for m in range(self.dims):
            # Construct the basis matrix for this dimension
            F = self._cheby_basis_matrix(self.degrees[m])
            
            # Perform mode-m multiplication
            self.tt_coeffs = self._mode_multiply(self.tt_coeffs, F, m)
        
        elapsed = time.time() - start_time
        print(f"Computed coefficients in {elapsed:.2f} seconds")
        
        return self.tt_coeffs
    
    def train(self, func, subset_fraction=1.0):
        """
        Train the interpolation by computing values and coefficients.
        
        Parameters:
        -----------
        func : callable
            Function to interpolate
        subset_fraction : float
            Fraction of grid points to use for training
            
        Returns:
        --------
        TTTensor
            TT-tensor of Chebyshev coefficients
        """
        print(f"=== TRAINING WITH {subset_fraction*100:.1f}% OF GRID POINTS ===")
        overall_start = time.time()
        
        # Generate subset indices if specified
        subset_indices = None
        if subset_fraction < 1.0:
            all_indices = self._generate_grid_indices()
            
            total_points = len(all_indices)
            subset_size = int(total_points * subset_fraction)
            
            if subset_size >= total_points:
                subset_indices = all_indices
            else:
                # Randomly sample indices
                np.random.seed(42)
                sample_idx = np.random.choice(total_points, size=subset_size, replace=False)
                subset_indices = [all_indices[i] for i in sample_idx]
        
        # Step 1: Construct tensor of values
        self.construct_tensor_values(func, subset_indices)
        
        # Step 2: Compute Chebyshev coefficients
        self.compute_coefficients()
        
        # Optional: Round the coefficients tensor to lower ranks
        self.tt_coeffs = tt_round(self.tt_coeffs, self.max_rank, self.epsilon)
        
        overall_elapsed = time.time() - overall_start
        print(f"Training completed in {overall_elapsed:.2f} seconds")
        
        # Print TT-ranks
        print(f"TT-ranks of coefficient tensor: {self.tt_coeffs.ranks}")
        
        return self.tt_coeffs
    
    def evaluate(self, point):
        """
        Evaluate the interpolation at a single point.
        
        Parameters:
        -----------
        point : list, tuple or numpy.ndarray
            Point in parameter space
            
        Returns:
        --------
        float
            Interpolated value
        """
        if self.tt_coeffs is None:
            raise ValueError("Coefficients not computed. Call train first.")
        
        # Evaluate Chebyshev basis at the point
        basis_values = self._eval_basis(point)
        
        # Contract basis with coefficient tensor
        return self.tt_coeffs.contract_with_vectors(basis_values)
    
    def evaluate_batch(self, points):
        """
        Evaluate the interpolation at multiple points.
        
        Parameters:
        -----------
        points : list of lists/tuples or numpy.ndarray
            Points in parameter space
            
        Returns:
        --------
        numpy.ndarray
            Interpolated values
        """
        return np.array([self.evaluate(point) for point in points])
    
    def test(self, func, test_points, reference=None):
        """
        Test the interpolation on a set of points.
        
        Parameters:
        -----------
        func : callable or None
            Function to compute exact values. If None, only interpolated values are returned.
        test_points : list of lists/tuples or numpy.ndarray
            Test points
        reference : numpy.ndarray or None
            Pre-computed reference values. If None and func is provided, 
            they will be computed.
            
        Returns:
        --------
        dict
            Dictionary with test results including interpolated values, exact values,
            errors, and error statistics
        """
        print("=== TESTING INTERPOLATION ===")
        start_time = time.time()
        
        # Evaluate interpolation
        interp_values = self.evaluate_batch(test_points)
        
        # Compute reference values if needed
        if reference is None and func is not None:
            reference = np.array([func(point) for point in test_points])
        
        # Compute errors if reference is available
        if reference is not None:
            abs_errors = np.abs(interp_values - reference)
            rel_errors = abs_errors / (np.abs(reference) + 1e-10)  # Avoid division by zero
            
            # Error statistics
            results = {
                'points': test_points,
                'interpolated': interp_values,
                'reference': reference,
                'abs_errors': abs_errors,
                'rel_errors': rel_errors,
                'max_abs_error': np.max(abs_errors),
                'mean_abs_error': np.mean(abs_errors),
                'rms_abs_error': np.sqrt(np.mean(abs_errors**2)),
                'max_rel_error': np.max(rel_errors),
                'mean_rel_error': np.mean(rel_errors),
                'rms_rel_error': np.sqrt(np.mean(rel_errors**2))
            }
        else:
            results = {
                'points': test_points,
                'interpolated': interp_values
            }
        
        elapsed = time.time() - start_time
        print(f"Testing completed in {elapsed:.2f} seconds")
        
        # Print error statistics if available
        if reference is not None:
            print("\nError statistics:")
            print(f"  Max absolute error: {results['max_abs_error']:.8f}")
            print(f"  Mean absolute error: {results['mean_abs_error']:.8f}")
            print(f"  RMS absolute error: {results['rms_abs_error']:.8f}")
            print(f"  Max relative error: {results['max_rel_error']:.8f}")
            print(f"  Mean relative error: {results['mean_rel_error']:.8f}")
            print(f"  RMS relative error: {results['rms_rel_error']:.8f}")
        
        return results
    
    def plot_results(self, test_results, log_scale=True, title=None):
        """
        Plot test results for 1D and 2D functions.
        
        Parameters:
        -----------
        test_results : dict
            Dictionary with test results from the test method
        log_scale : bool
            Whether to use log scale for error plots
        title : str or None
            Title for the plots
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with plots
        """
        if self.dims == 1:
            return self._plot_1d_results(test_results, log_scale, title)
        elif self.dims == 2:
            return self._plot_2d_results(test_results, log_scale, title)
        else:
            print("Plotting is only supported for 1D and 2D functions.")
            return None
    
    def _plot_1d_results(self, test_results, log_scale=True, title=None):
        """Plot results for 1D functions"""
        # Extract data
        x = np.array([p[0] for p in test_results['points']])
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y_interp = test_results['interpolated'][sort_idx]
        
        fig = plt.figure(figsize=(12, 8))
        
        if 'reference' in test_results:
            y_exact = test_results['reference'][sort_idx]
            abs_errors = test_results['abs_errors'][sort_idx]
            
            # Top plot: Function values
            ax1 = fig.add_subplot(211)
            ax1.plot(x, y_exact, 'k-', label='Exact Function')
            ax1.plot(x, y_interp, 'r--', label='TT-Chebyshev Interpolation')
            ax1.grid(True)
            ax1.legend()
            if title:
                ax1.set_title(title)
            else:
                ax1.set_title('Function and TT-Chebyshev Interpolation')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            
            # Bottom plot: Errors
            ax2 = fig.add_subplot(212)
            if log_scale:
                ax2.semilogy(x, abs_errors, 'r-', label='Absolute Error')
                ax2.set_ylabel('Absolute Error (log scale)')
            else:
                ax2.plot(x, abs_errors, 'r-', label='Absolute Error')
                ax2.set_ylabel('Absolute Error')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Error of TT-Chebyshev Interpolation')
            ax2.set_xlabel('x')
            
        else:
            # Only interpolated values
            ax = fig.add_subplot(111)
            ax.plot(x, y_interp, 'r-', label='TT-Chebyshev Interpolation')
            ax.grid(True)
            ax.legend()
            if title:
                ax.set_title(title)
            else:
                ax.set_title('TT-Chebyshev Interpolation')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
        
        plt.tight_layout()
        return fig
    
    def _plot_2d_results(self, test_results, log_scale=True, title=None):
        """Plot results for 2D functions"""
        # Extract data
        x = np.array([p[0] for p in test_results['points']])
        y = np.array([p[1] for p in test_results['points']])
        z_interp = test_results['interpolated']
        
        # Check if points form a grid
        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))
        
        if len(x_unique) * len(y_unique) == len(x):
            # Regular grid
            X, Y = np.meshgrid(x_unique, y_unique)
            Z_interp = z_interp.reshape(len(y_unique), len(x_unique))
            
            if 'reference' in test_results:
                Z_exact = test_results['reference'].reshape(len(y_unique), len(x_unique))
                Z_error = np.abs(Z_interp - Z_exact)
            
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            
            if 'reference' in test_results:
                # 3D plots
                ax1 = fig.add_subplot(231, projection='3d')
                surf1 = ax1.plot_surface(X, Y, Z_exact, cmap='viridis', alpha=0.8)
                ax1.set_title('Exact Function')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('f(x,y)')
                
                ax2 = fig.add_subplot(232, projection='3d')
                surf2 = ax2.plot_surface(X, Y, Z_interp, cmap='plasma', alpha=0.8)
                ax2.set_title('TT-Chebyshev Interpolation')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_zlabel('f(x,y)')
                
                ax3 = fig.add_subplot(233, projection='3d')
                if log_scale:
                    surf3 = ax3.plot_surface(X, Y, np.log10(Z_error + 1e-15), cmap='hot', alpha=0.8)
                    ax3.set_title('Log10 Absolute Error')
                    ax3.set_zlabel('log10(Error)')
                else:
                    surf3 = ax3.plot_surface(X, Y, Z_error, cmap='hot', alpha=0.8)
                    ax3.set_title('Absolute Error')
                    ax3.set_zlabel('Error')
                ax3.set_xlabel('x')
                ax3.set_ylabel('y')
                
                # 2D contour plots
                ax4 = fig.add_subplot(234)
                contour1 = ax4.contourf(X, Y, Z_exact, cmap='viridis')
                fig.colorbar(contour1, ax=ax4)
                ax4.set_title('Exact Function (Contour)')
                ax4.set_xlabel('x')
                ax4.set_ylabel('y')
                
                ax5 = fig.add_subplot(235)
                contour2 = ax5.contourf(X, Y, Z_interp, cmap='plasma')
                fig.colorbar(contour2, ax=ax5)
                ax5.set_title('TT-Chebyshev Interpolation (Contour)')
                ax5.set_xlabel('x')
                ax5.set_ylabel('y')
                
                ax6 = fig.add_subplot(236)
                if log_scale:
                    contour3 = ax6.contourf(X, Y, np.log10(Z_error + 1e-15), cmap='hot')
                    fig.colorbar(contour3, ax=ax6)
                    ax6.set_title('Log10 Absolute Error (Contour)')
                else:
                    contour3 = ax6.contourf(X, Y, Z_error, cmap='hot')
                    fig.colorbar(contour3, ax=ax6)
                    ax6.set_title('Absolute Error (Contour)')
                ax6.set_xlabel('x')
                ax6.set_ylabel('y')
                
            else:
                # Only interpolated values
                ax1 = fig.add_subplot(121, projection='3d')
                surf = ax1.plot_surface(X, Y, Z_interp, cmap='viridis', alpha=0.8)
                ax1.set_title('TT-Chebyshev Interpolation (3D)')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('f(x,y)')
                
                ax2 = fig.add_subplot(122)
                contour = ax2.contourf(X, Y, Z_interp, cmap='viridis')
                fig.colorbar(contour, ax=ax2)
                ax2.set_title('TT-Chebyshev Interpolation (Contour)')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
            
        else:
            # Scattered points
            fig = plt.figure(figsize=(12, 6))
            
            if 'reference' in test_results:
                z_exact = test_results['reference']
                abs_errors = test_results['abs_errors']
                
                ax1 = fig.add_subplot(121, projection='3d')
                scatter1 = ax1.scatter(x, y, z_exact, c=z_exact, cmap='viridis', label='Exact')
                scatter2 = ax1.scatter(x, y, z_interp, c=z_interp, cmap='plasma', alpha=0.5, label='Interpolated')
                ax1.set_title('Function Values')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('f(x,y)')
                ax1.legend()
                
                ax2 = fig.add_subplot(122, projection='3d')
                if log_scale:
                    scatter3 = ax2.scatter(x, y, np.log10(abs_errors + 1e-15), c=np.log10(abs_errors + 1e-15), cmap='hot')
                    ax2.set_title('Log10 Absolute Error')
                    ax2.set_zlabel('log10(Error)')
                else:
                    scatter3 = ax2.scatter(x, y, abs_errors, c=abs_errors, cmap='hot')
                    ax2.set_title('Absolute Error')
                    ax2.set_zlabel('Error')
                fig.colorbar(scatter3, ax=ax2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                
            else:
                # Only interpolated values
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(x, y, z_interp, c=z_interp, cmap='viridis')
                fig.colorbar(scatter, ax=ax)
                ax.set_title('TT-Chebyshev Interpolation')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('f(x,y)')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)  # Adjust for suptitle
            
        return fig


def generate_test_data_1d(func, domain, n_points):
    """
    Generate test data for 1D functions.
    
    Parameters:
    -----------
    func : callable
        Function to evaluate
    domain : tuple
        Domain as (min, max)
    n_points : int
        Number of test points
        
    Returns:
    --------
    tuple
        (x_test, y_test) as numpy arrays
    """
    x_test = np.linspace(domain[0], domain[1], n_points)
    y_test = np.array([func(x) for x in x_test])
    
    return x_test, y_test


def split_train_test(x, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input data
    y : numpy.ndarray
        Output data
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (x_train, x_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test


def test_function_1d(x):
    """
    Test function with non-differentiable points.
    
    Parameters:
    -----------
    x : float or list/tuple
        Input point
        
    Returns:
    --------
    float
        Function value
    """
    # Convert tuple to float if needed
    if isinstance(x, (list, tuple)):
        x = x[0]
        
    if x < 0:
        return x**2 - 1
    elif x < 0.5:
        return 2*x
    else:
        return 1 - (x-1)**2


def test_function_2d(point):
    """
    Test function for 2D interpolation.
    
    Parameters:
    -----------
    point : tuple or list
        (x, y) coordinates
        
    Returns:
    --------
    float
        Function value
    """
    x, y = point
    return np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-0.1 * (x**2 + y**2))


def test_blackscholes_call(S, K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Black-Scholes call option price.
    
    Parameters:
    -----------
    S : float or tuple
        Stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
        
    Returns:
    --------
    float
        Option price
    """
    from scipy.stats import norm
    
    if isinstance(S, (list, tuple)):
        S = S[0]
        
    if T <= 0:
        return max(0, S - K)
        
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def demo_1d_function():
    """
    Demonstrate TT-Chebyshev interpolation on a 1D function with non-differentiable points.
    """
    print("\n=== DEMONSTRATION: 1D FUNCTION WITH NON-DIFFERENTIABLE POINTS ===")
    
    # Define domain and function
    domain = [(-1, 1)]
    
    # Create interpolation with different degrees
    degrees = 10
    max_rank = 10
    epsilon = 1e-10
    
    # Create TT-Chebyshev interpolation
    tt_cheby = ChebyshevInterpolationTT(domain, degrees, max_rank, epsilon)
    
    # Train with subset of points
    tt_cheby.train(test_function_1d, subset_fraction=0.8)
    
    # Generate test points
    x_test = np.linspace(domain[0][0], domain[0][1], 1000)
    test_points = [(x,) for x in x_test]
    
    # Test interpolation
    test_results = tt_cheby.test(test_function_1d, test_points)
    
    # Plot results
    fig = tt_cheby.plot_results(test_results, title="1D Function with Non-differentiable Points")
    plt.savefig("tt_cheby_1d_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test with different subset fractions
    print("\nTesting with different subset fractions:")
    fractions = [0.1, 0.5, 1.0]
    error_stats = {'fraction': [], 'max_error': [], 'mean_error': [], 'rms_error': []}
    
    for frac in fractions:
        print(f"\nTraining with {frac*100:.1f}% of grid points...")
        tt_cheby = ChebyshevInterpolationTT(domain, degrees, max_rank, epsilon)
        tt_cheby.train(test_function_1d, subset_fraction=frac)
        test_results = tt_cheby.test(test_function_1d, test_points)
        
        error_stats['fraction'].append(frac)
        error_stats['max_error'].append(test_results['max_abs_error'])
        error_stats['mean_error'].append(test_results['mean_abs_error'])
        error_stats['rms_error'].append(test_results['rms_abs_error'])
    
    # Plot error vs. fraction
    plt.figure(figsize=(10, 6))
    plt.semilogy(error_stats['fraction'], error_stats['max_error'], 'ro-', label='Max Error')
    plt.semilogy(error_stats['fraction'], error_stats['mean_error'], 'go-', label='Mean Error')
    plt.semilogy(error_stats['fraction'], error_stats['rms_error'], 'bo-', label='RMS Error')
    plt.grid(True)
    plt.xlabel('Fraction of Grid Points Used')
    plt.ylabel('Error (log scale)')
    plt.title('Error vs. Sampling Fraction')
    plt.legend()
    plt.savefig("tt_cheby_1d_sampling.png", dpi=150, bbox_inches='tight')
    plt.show()


def demo_2d_function():
    """
    Demonstrate TT-Chebyshev interpolation on a 2D function.
    """
    print("\n=== DEMONSTRATION: 2D FUNCTION ===")
    
    # Define domain and function
    domain = [(-1, 1), (-1, 1)]
    
    # Create interpolation
    degrees = [15, 15]
    max_rank = 10
    epsilon = 1e-10
    
    # Create TT-Chebyshev interpolation
    tt_cheby = ChebyshevInterpolationTT(domain, degrees, max_rank, epsilon)
    
    # Train
    tt_cheby.train(test_function_2d, subset_fraction=0.5)
    
    # Generate test points on a grid
    x_test = np.linspace(domain[0][0], domain[0][1], 50)
    y_test = np.linspace(domain[1][0], domain[1][1], 50)
    X, Y = np.meshgrid(x_test, y_test)
    test_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Test interpolation
    test_results = tt_cheby.test(test_function_2d, test_points)
    
    # Plot results
    fig = tt_cheby.plot_results(test_results, title="2D Function Interpolation")
    plt.savefig("tt_cheby_2d_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test with different ranks
    print("\nTesting with different maximum ranks:")
    ranks = [2, 5, 10, 20]
    error_stats = {'rank': [], 'max_error': [], 'mean_error': [], 'rms_error': []}
    
    for rank in ranks:
        print(f"\nTraining with max_rank={rank}...")
        tt_cheby = ChebyshevInterpolationTT(domain, degrees, rank, epsilon)
        tt_cheby.train(test_function_2d, subset_fraction=0.5)
        test_results = tt_cheby.test(test_function_2d, test_points)
        
        error_stats['rank'].append(rank)
        error_stats['max_error'].append(test_results['max_abs_error'])
        error_stats['mean_error'].append(test_results['mean_abs_error'])
        error_stats['rms_error'].append(test_results['rms_abs_error'])
    
    # Plot error vs. rank
    plt.figure(figsize=(10, 6))
    plt.semilogy(error_stats['rank'], error_stats['max_error'], 'ro-', label='Max Error')
    plt.semilogy(error_stats['rank'], error_stats['mean_error'], 'go-', label='Mean Error')
    plt.semilogy(error_stats['rank'], error_stats['rms_error'], 'bo-', label='RMS Error')
    plt.grid(True)
    plt.xlabel('Maximum TT-Rank')
    plt.ylabel('Error (log scale)')
    plt.title('Error vs. Maximum TT-Rank')
    plt.legend()
    plt.savefig("tt_cheby_2d_rank.png", dpi=150, bbox_inches='tight')
    plt.show()


def demo_option_pricing():
    """
    Demonstrate TT-Chebyshev interpolation on a call option pricing problem.
    """
    print("\n=== DEMONSTRATION: OPTION PRICING ===")
    
    # Define domain and parameters
    S_min, S_max = 50, 150
    K = 100  # Strike price
    domain = [(S_min, S_max)]
    
    # Create interpolation
    degrees = 15
    max_rank = 5
    epsilon = 1e-8
    
    # Create TT-Chebyshev interpolation
    tt_cheby = ChebyshevInterpolationTT(domain, degrees, max_rank, epsilon)
    
    # Black-Scholes function
    bs_func = lambda S: test_blackscholes_call(S, K=K)
    
    # Train
    tt_cheby.train(bs_func, subset_fraction=0.8)
    
    # Generate test points
    S_test = np.linspace(S_min, S_max, 1000)
    test_points = [(S,) for S in S_test]
    
    # Test interpolation
    test_results = tt_cheby.test(bs_func, test_points)
    
    # Plot results
    fig = tt_cheby.plot_results(test_results, title="Call Option Price Interpolation")
    plt.savefig("tt_cheby_option_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Focus on region around strike price
    print("\nFocusing on region around strike price:")
    near_K_min, near_K_max = 95, 105
    S_near_K = np.linspace(near_K_min, near_K_max, 500)
    test_points_near_K = [(S,) for S in S_near_K]
    
    # Test interpolation in this region
    test_results_near_K = tt_cheby.test(bs_func, test_points_near_K)
    
    # Plot results for this region
    fig = plt.figure(figsize=(12, 8))
    
    # Top plot: Function values
    ax1 = fig.add_subplot(211)
    ax1.plot(S_near_K, test_results_near_K['reference'], 'k-', label='Exact BS Price')
    ax1.plot(S_near_K, test_results_near_K['interpolated'], 'r--', label='TT-Chebyshev Interpolation')
    ax1.grid(True)
    ax1.axvline(x=K, color='k', linestyle=':', label='Strike (K)')
    ax1.legend()
    ax1.set_title('Call Option Price Near Strike')
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Option Price')
    
    # Bottom plot: Errors
    ax2 = fig.add_subplot(212)
    ax2.semilogy(S_near_K, test_results_near_K['abs_errors'], 'r-', label='Absolute Error')
    ax2.grid(True)
    ax2.axvline(x=K, color='k', linestyle=':', label='Strike (K)')
    ax2.legend()
    ax2.set_title('Error of TT-Chebyshev Interpolation Near Strike')
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Absolute Error (log scale)')
    
    plt.tight_layout()
    plt.savefig("tt_cheby_option_strike.png", dpi=150, bbox_inches='tight')
    plt.show()


def tf_tensor_train_layer(input_dim, output_shape, tt_ranks, activation=None):
    """
    Create a TensorFlow layer implementing a Tensor Train transformation.
    
    Parameters:
    -----------
    input_dim : int or list
        Input dimension(s)
    output_shape : list
        Output tensor shape
    tt_ranks : list
        TT-ranks
    activation : callable or None
        Activation function
        
    Returns:
    --------
    tf.keras.layers.Layer
        TensorFlow layer implementing the TT transformation
    """
    class TTLayer(tf.keras.layers.Layer):
        def __init__(self, input_dim, output_shape, tt_ranks, activation=None):
            super(TTLayer, self).__init__()
            
            # Convert to lists if needed
            if isinstance(input_dim, int):
                self.input_dim = [input_dim]
            else:
                self.input_dim = list(input_dim)
                
            self.output_shape = list(output_shape)
            self.tt_ranks = list(tt_ranks)
            self.activation = activation
            
            # Number of dimensions
            self.d = len(self.output_shape)
            
            # Create TT-cores as variables
            self.cores = []
            for k in range(self.d):
                if k == 0:
                    core_shape = [1, self.output_shape[k], self.tt_ranks[k]]
                elif k == self.d - 1:
                    core_shape = [self.tt_ranks[k-1], self.output_shape[k], 1]
                else:
                    core_shape = [self.tt_ranks[k-1], self.output_shape[k], self.tt_ranks[k]]
                
                # Initialize with random values
                initializer = tf.keras.initializers.GlorotUniform()
                core = self.add_weight(
                    name=f'tt_core_{k}',
                    shape=core_shape,
                    initializer=initializer,
                    trainable=True
                )
                
                self.cores.append(core)
        
        def call(self, inputs):
            # Check input shape
            if len(inputs.shape) != 2 or inputs.shape[1] != np.prod(self.input_dim):
                raise ValueError(f"Input shape must be (batch_size, {np.prod(self.input_dim)})")
            
            # Perform TT transformation
            batch_size = tf.shape(inputs)[0]
            output = tf.ones((batch_size, 1, 1))
            
            for k in range(self.d):
                # Contract with the k-th core
                output = tf.einsum('bij,jkl->bikl', output, self.cores[k])
                
                # Reshape
                output = tf.reshape(output, (batch_size, -1, self.cores[k].shape[2]))
            
            # Final reshape
            output = tf.reshape(output, (batch_size, -1))
            
            # Apply activation if provided
            if self.activation is not None:
                output = self.activation(output)
                
            return output
    
    return TTLayer(input_dim, output_shape, tt_ranks, activation)


def demo_tt_regression_tf():
    """
    Demonstrate TensorFlow-based Tensor Train regression.
    """
    print("\n=== DEMONSTRATION: TT REGRESSION WITH TENSORFLOW ===")
    
    # Define function to approximate
    func = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 0.25 * np.sin(5 * np.pi * x)
    
    # Generate data
    x_data = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_data = func(x_data).reshape(-1, 1)
    
    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
    
    # Create TT-based model
    input_dim = 1
    output_shape = [5, 5, 5]  # 3D tensor with 5x5x5 = 125 elements
    tt_ranks = [1, 3, 3, 1]  # Ranks for TT representation
    
    tt_layer = tf_tensor_train_layer(input_dim, output_shape, tt_ranks)
    
    # Create model
    model = tf.keras.Sequential([
        tt_layer,
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.semilogy(history.history['loss'], 'b-', label='Training Loss')
    plt.semilogy(history.history['val_loss'], 'r-', label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('TT Regression Training History')
    plt.legend()
    plt.savefig("tt_regression_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Evaluate model
    y_pred = model.predict(x_test)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(x_test, y_test, 'b.', label='True')
    plt.plot(x_test, y_pred, 'r.', label='Predicted')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('TT Regression Predictions')
    plt.legend()
    
    plt.subplot(122)
    plt.semilogy(np.abs(y_test - y_pred), 'r.')
    plt.grid(True)
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('TT Regression Errors')
    
    plt.tight_layout()
    plt.savefig("tt_regression_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute error metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    
    print("\nTT Regression Error Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.8f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.8f}")
    print(f"  Mean Absolute Error (MAE): {mae:.8f}")
    print(f"  Maximum Absolute Error: {max_error:.8f}")


def main():
    """
    Main function to run all demos.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run demos
    demo_1d_function()
    demo_2d_function()
    demo_option_pricing()
    demo_tt_regression_tf()
    
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()