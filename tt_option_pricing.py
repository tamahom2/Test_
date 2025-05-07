import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensornetwork as tn
from itertools import product
from sklearn.model_selection import train_test_split
import time
from scipy.fftpack import dct
from scipy.linalg import qr, svd, lstsq


class TTDecomposition:
    """
    Tensor Train decomposition implementation for option pricing.
    This follows the methodology from the paper but is implemented from scratch.
    """
    
    def __init__(self, shapes):
        """
        Initialize a Tensor Train decomposition.
        
        Parameters:
        -----------
        shapes : list
            List of mode sizes for the tensor.
        """
        self.shapes = shapes
        self.dims = len(shapes)
        self.cores = None
        self.ranks = None
    
    def initialize_random(self, ranks):
        """
        Initialize TT cores with random values.
        
        Parameters:
        -----------
        ranks : list
            TT-ranks for the decomposition (length should be dims+1).
        """
        if len(ranks) != self.dims + 1:
            raise ValueError(f"Expected ranks of length {self.dims+1}, got {len(ranks)}")
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("First and last rank must be 1")
            
        self.ranks = ranks
        
        # Initialize random cores
        self.cores = []
        for d in range(self.dims):
            shape = (self.ranks[d], self.shapes[d], self.ranks[d+1])
            self.cores.append(np.random.normal(0, 0.01, size=shape))
        
        return self
    
    def orthogonalize(self):
        """
        Left-orthogonalize all cores except the last one.
        """
        for i in range(self.dims - 1):
            core = self.cores[i]
            r1, n, r2 = core.shape
            
            # Reshape the core to a matrix
            core_mat = core.reshape(r1 * n, r2)
            
            # QR decomposition
            Q, R = qr(core_mat, mode='economic')
            
            # Update current core
            self.cores[i] = Q.reshape(r1, n, Q.shape[1])
            
            # Update next core
            next_core = self.cores[i+1]
            next_core = np.tensordot(R, next_core, axes=(1, 0))
            self.cores[i+1] = next_core
            
            # Update ranks
            self.ranks[i+1] = Q.shape[1]
        
        return self
        
    def truncate(self, eps=1e-10, max_rank=None):
        """
        Truncate TT-ranks based on specified accuracy.
        
        Parameters:
        -----------
        eps : float
            Desired accuracy of truncation.
        max_rank : int or None
            Maximum allowed rank.
        """
        # First, orthogonalize
        self.orthogonalize()
        
        # Truncate from right to left
        for i in range(self.dims - 1, 0, -1):
            # Get current core
            core = self.cores[i]
            r1, n, r2 = core.shape
            
            # Reshape to matrix
            core_mat = core.reshape(r1, n * r2)
            
            # SVD
            U, S, Vh = svd(core_mat, full_matrices=False)
            
            # Determine rank based on singular values
            if max_rank is None:
                tol = eps * S[0]
                rank = np.sum(S > tol)
            else:
                rank = min(len(S), max_rank)
            
            # Truncate
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Update cores
            self.cores[i] = (Vh.reshape(rank, n, r2))
            
            # Update previous core
            prev_core = self.cores[i-1]
            r0, n_prev, _ = prev_core.shape
            self.cores[i-1] = np.tensordot(prev_core, U * S, axes=(2, 0)).reshape(r0, n_prev, rank)
            
            # Update ranks
            self.ranks[i] = rank
        
        return self
    
    def get_full_tensor(self):
        """
        Convert the TT format to a full tensor (only for small tensors).
        """
        result = self.cores[0]
        for i in range(1, self.dims):
            result = np.tensordot(result, self.cores[i], axes=([-1], [0]))
        
        return result.squeeze()
    
    def evaluate(self, indices):
        """
        Evaluate TT tensor at given indices.
        
        Parameters:
        -----------
        indices : list or array
            Indices where to evaluate the tensor.
            
        Returns:
        --------
        numpy.ndarray
            Values of the tensor at the specified indices.
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        
        # Handle both single index and multiple indices
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
        
        values = np.zeros(indices.shape[0])
        
        for i, idx in enumerate(indices):
            # Start with 1x1 matrix
            result = np.ones((1, 1))
            
            # Multiply by each core's appropriate slice
            for d, id_value in enumerate(idx):
                core_slice = self.cores[d][:, id_value, :]
                result = result @ core_slice
            
            values[i] = result[0, 0]
            
        return values


class MaxvolHelper:
    """Helper class for maxvol algorithm used in tensor completion."""
    
    @staticmethod
    def maxvol(A, tau=1.05, max_iters=100):
        """
        Find submatrix with maximum volume.
        
        Parameters:
        -----------
        A : numpy.ndarray
            Matrix of shape (m, n) with m >= n.
        tau : float
            Stopping criterion.
        max_iters : int
            Maximum number of iterations.
            
        Returns:
        --------
        tuple
            (indices, coefficients) where indices are row indices of the 
            maximum volume submatrix and coefficients are the coefficient matrix.
        """
        m, n = A.shape
        if m < n:
            raise ValueError("Matrix should have more rows than columns")
        
        # Initial indices using QR with pivoting
        Q, R, P = qr(A, mode='economic', pivoting=True)
        indices = P[:n]
        
        # Get square submatrix
        A_sub = A[indices, :]
        
        # Initial coefficients
        B = A @ np.linalg.inv(A_sub)
        
        # Iterative improvement
        for k in range(max_iters):
            # Find maximum magnitude element
            max_idx = np.argmax(np.abs(B))
            i, j = max_idx // n, max_idx % n
            
            if np.abs(B[i, j]) <= tau:
                break
                
            # Swap rows
            old_idx = indices[j]
            indices[j] = i
            
            # Update coefficients
            B_j = B[:, j].copy()
            B_i = B[i, :].copy()
            B_i[j] -= 1
            
            # Sherman-Morrison formula
            B -= np.outer(B_j, B_i) / B[i, j]
            
        return indices, B
    
    @staticmethod
    def maxvol_rect(A, tau=1.1, dr_min=0, dr_max=None, tau0=1.05, k0=100):
        """
        Find rectangular submatrix with maximum volume.
        
        Parameters:
        -----------
        A : numpy.ndarray
            Matrix of shape (m, n) with m >= n.
        tau : float
            Accuracy parameter.
        dr_min : int
            Minimum number of added rows.
        dr_max : int or None
            Maximum number of added rows.
        tau0 : float
            Accuracy parameter for initial maxvol.
        k0 : int
            Maximum iterations for initial maxvol.
            
        Returns:
        --------
        tuple
            (indices, coefficients) where indices are row indices of the 
            maximum volume submatrix and coefficients are the coefficient matrix.
        """
        m, n = A.shape
        if m <= n:
            return np.arange(m), np.eye(m)
        
        # Set default dr_max
        if dr_max is None:
            dr_max = m - n
        else:
            dr_max = min(dr_max, m - n)
            
        dr_min = min(dr_min, dr_max)
        
        # Start with square submatrix using maxvol
        indices, B = MaxvolHelper.maxvol(A, tau0, k0)
        
        # Initialize selection vector and F
        selected = np.zeros(m, dtype=bool)
        selected[indices] = True
        F = np.sum(B**2, axis=1) * (1 - selected)
        
        # Add rows iteratively
        for k in range(n, n + dr_max):
            i = np.argmax(F)
            
            # Check if we can stop
            if k >= n + dr_min and F[i] <= tau**2:
                break
                
            # Add row to selection
            indices = np.append(indices, i)
            selected[i] = True
            
            # Update B matrix and F
            v = B @ B[i, :]
            l = 1.0 / (1.0 + v[i])
            B = np.hstack([B - l * np.outer(v, B[i, :]), l * v.reshape(-1, 1)])
            F = (1 - selected) * (F - l * v**2)
        
        # Create final coefficient matrix
        B_final = np.zeros((m, len(indices)))
        B_final[selected] = np.eye(len(indices))
        B_final[~selected] = B[~selected]
        
        return indices, B_final


class TTChebyshevApproximation:
    """
    Tensor Train based Chebyshev interpolation for parametric option pricing.
    
    This class implements the approach from the paper "Low-rank tensor approximation for 
    Chebyshev interpolation in parametric option pricing" by Glau et al.
    """
    
    def __init__(self, domains, degrees):
        """
        Initialize the TT-Chebyshev interpolation.
        
        Parameters:
        -----------
        domains : list of tuples
            List of (min, max) tuples defining the domain in each dimension.
        degrees : list of int or int
            Degrees for Chebyshev polynomials in each dimension.
        """
        self.domains = domains
        self.degrees = degrees if isinstance(degrees, list) else [degrees] * len(domains)
        self.dims = len(domains)
        
        # Shapes for the tensor (nodes per dimension)
        self.shapes = [degree + 1 for degree in self.degrees]
        
        # TT tensors
        self.P = None  # TT tensor of values at Chebyshev nodes
        self.C = None  # TT tensor of Chebyshev coefficients
        
        # Generate Chebyshev nodes for each dimension
        self.nodes = []
        for i, (a, b) in enumerate(self.domains):
            N = self.degrees[i]
            # Chebyshev nodes in canonical domain
            k = np.arange(N+1)
            z = np.cos(np.pi * k / N)
            
            # Map to parameter domain [a, b]
            x = 0.5 * (b - a) * (z + 1) + a
            self.nodes.append(x)
    
    def _idx_to_params(self, indices):
        """Convert grid indices to parameter values."""
        params = []
        for idx in indices:
            param_values = []
            for dim, i in enumerate(idx):
                param_values.append(self.nodes[dim][i])
            params.append(param_values)
        return np.array(params)
    
    def _construct_basis_matrix(self, n):
        """
        Construct the Chebyshev basis matrix for computing coefficients.
        
        Parameters:
        -----------
        n : int
            Degree of Chebyshev interpolation
            
        Returns:
        --------
        numpy.ndarray
            Basis matrix for computing Chebyshev coefficients
        """
        F_n = np.zeros((n+1, n+1))
        
        for j in range(n+1):
            for k in range(n+1):
                # Coefficient
                coef = 2.0 / n
                
                # Special case for first and last elements
                if j == 0 or j == n:
                    coef *= 0.5
                
                # Compute basis function
                F_n[j, k] = coef * np.cos(j * np.pi * k / n)
        
        return F_n
    
    def cross_approximation(self, pricing_func, max_rank=20, eps=1e-10, 
                           dr_min=1, dr_max=5, tau=1.1, max_sweeps=20):
        """
        Perform cross approximation to build TT tensor P.
        
        Parameters:
        -----------
        pricing_func : callable
            Function that takes parameter values and returns option price.
        max_rank : int
            Maximum TT-rank allowed.
        eps : float
            Desired approximation accuracy.
        dr_min : int
            Minimum rank increment in maxvol_rect.
        dr_max : int
            Maximum rank increment in maxvol_rect.
        tau : float
            Accuracy parameter for maxvol_rect.
        max_sweeps : int
            Maximum number of sweeps for cross approximation.
            
        Returns:
        --------
        TTDecomposition
            TT tensor of option prices at Chebyshev nodes.
        """
        print("Running TT-Cross approximation...")
        start_time = time.time()
        
        # Create initial TT tensor with ranks 1
        tt = TTDecomposition(self.shapes).initialize_random([1] * (self.dims + 1))
        
        # Function to evaluate pricing function at grid indices
        def evaluate_pricing_func(indices):
            if len(indices) == 0:
                return np.array([])
                
            # Convert indices to parameter values
            params = self._idx_to_params(indices)
            
            # Evaluate option pricing function
            values = np.array([pricing_func(p) for p in params])
            return values
        
        # Cache to avoid redundant evaluations
        cache = {}
        
        # Initialize row/column selection indices for each core
        row_indices = [None] * (self.dims + 1)
        col_indices = [None] * (self.dims + 1)
        
        # Initial grid points (all indices for each dimension)
        grid_indices = [np.arange(shape).reshape(-1, 1) for shape in self.shapes]
        
        # Initial R matrix is ones (scalar)
        R = np.ones((1, 1))
        
        # Initialization sweep (left-to-right)
        for i in range(self.dims):
            # Get current core
            core = tt.cores[i]
            r1, n, r2 = core.shape
            
            # Prepare indices for evaluation
            if row_indices[i] is None:
                row_idx = np.arange(1).reshape(-1, 1)  # Just index 0 for rank 1
            else:
                row_idx = row_indices[i]
                
            col_grid = grid_indices[i]
            
            # Prepare all combinations of indices
            idx_combinations = []
            for r in range(len(row_idx)):
                for c in range(len(col_grid)):
                    idx_combinations.append(np.hstack([row_idx[r], col_grid[c]]))
                    
            # Evaluate function if needed
            for idx in idx_combinations:
                idx_tuple = tuple(idx.flatten())
                if idx_tuple not in cache:
                    cache[idx_tuple] = pricing_func(self._idx_to_params([idx_tuple])[0])
            
            # Get values and reshape
            values = np.array([cache[tuple(idx.flatten())] for idx in idx_combinations])
            G = values.reshape(len(row_idx), len(col_grid))
            
            # Perform decomposition
            U, S, Vh = svd(G, full_matrices=False)
            tt.cores[i] = U @ np.diag(S) @ Vh
            
            # Set next row indices using maxvol
            _, row_indices[i+1] = MaxvolHelper.maxvol(Vh.T)
        
        # Main iteration
        for sweep in range(max_sweeps):
            tt_old = TTDecomposition(tt.shapes)
            tt_old.cores = [core.copy() for core in tt.cores]
            tt_old.ranks = tt.ranks.copy()
            
            # Left-to-right half-sweep
            for i in range(self.dims):
                # Prepare indices for current slice
                if i == 0:
                    row_idx = np.ones((1, 1))  # Just a dummy index for first core
                else:
                    row_idx = row_indices[i]
                    
                col_idx = col_indices[i+1] if i < self.dims - 1 else np.ones((1, 1))
                
                # Build evaluation indices
                eval_indices = []
                for r in row_idx:
                    for g in grid_indices[i]:
                        for c in col_idx:
                            eval_indices.append(np.hstack([r, g, c]))
                
                # Evaluate function at indices
                values = []
                for idx in eval_indices:
                    idx_tuple = tuple(idx.flatten())
                    if idx_tuple not in cache:
                        param = self._idx_to_params([idx_tuple[1:-1]])[0]
                        cache[idx_tuple] = pricing_func(param)
                    values.append(cache[idx_tuple])
                
                # Reshape values into tensor
                Z = np.array(values).reshape(len(row_idx), -1, len(col_idx))
                
                # Update core using maxvol_rect
                tt.cores[i], R, row_indices[i+1] = self._update_core(
                    Z, grid_indices[i], row_indices[i], tau, dr_min, dr_max, 
                    left_to_right=True
                )
            
            # Right-to-left half-sweep
            for i in range(self.dims-1, -1, -1):
                # Prepare indices
                row_idx = row_indices[i] if i > 0 else np.ones((1, 1))
                col_idx = col_indices[i+1]
                
                # Build evaluation indices
                eval_indices = []
                for r in row_idx:
                    for g in grid_indices[i]:
                        for c in col_idx:
                            eval_indices.append(np.hstack([r, g, c]))
                
                # Evaluate function at indices
                values = []
                for idx in eval_indices:
                    idx_tuple = tuple(idx.flatten())
                    if idx_tuple not in cache:
                        param = self._idx_to_params([idx_tuple[1:-1]])[0]
                        cache[idx_tuple] = pricing_func(param)
                    values.append(cache[idx_tuple])
                
                # Reshape values into tensor
                Z = np.array(values).reshape(len(row_idx), -1, len(col_idx))
                
                # Update core using maxvol_rect
                tt.cores[i], R, col_indices[i] = self._update_core(
                    Z, grid_indices[i], col_indices[i+1], tau, dr_min, dr_max, 
                    left_to_right=False
                )
            
            # Check convergence
            error = self._estimate_relative_change(tt, tt_old)
            
            # Update TT ranks
            tt.ranks = [1] + [core.shape[2] for core in tt.cores[:-1]] + [1]
            
            print(f"Sweep {sweep+1}, error: {error:.6e}, ranks: {tt.ranks}")
            
            if error < eps:
                print(f"Converged after {sweep+1} sweeps")
                break
        
        # Truncate to desired accuracy
        tt.truncate(eps=eps, max_rank=max_rank)
        
        elapsed = time.time() - start_time
        print(f"Cross-approximation completed in {elapsed:.2f} seconds")
        print(f"Final TT ranks: {tt.ranks}")
        print(f"Function evaluations: {len(cache)}")
        
        self.P = tt
        return tt
    
    def _update_core(self, Z, grid_indices, row_indices, tau, dr_min, dr_max, left_to_right=True):
        """
        Update TT core using maxvol algorithm.
        
        Parameters:
        -----------
        Z : numpy.ndarray
            Tensor slice of shape (r1, n, r2)
        grid_indices : numpy.ndarray
            Indices of the current dimension
        row_indices : numpy.ndarray
            Row indices from previous iteration
        tau : float
            Accuracy parameter for maxvol
        dr_min : int
            Minimum rank increment
        dr_max : int
            Maximum rank increment
        left_to_right : bool
            Direction of sweep
            
        Returns:
        --------
        tuple
            (updated_core, R_matrix, new_indices)
        """
        r1, n, r2 = Z.shape
        
        if left_to_right:
            # Reshape Z for left-to-right sweep
            Z_mat = Z.reshape(r1 * n, r2)
            
            # QR decomposition
            Q, R = qr(Z_mat, mode='economic')
            
            # Find maximum volume submatrix
            indices, B = MaxvolHelper.maxvol_rect(Q, tau, dr_min, dr_max)
            
            # Reshape core
            G = B.reshape(r1, n, -1)
            
            # Create new row indices
            I_new = np.kron(grid_indices, np.ones((r1, 1), dtype=int))
            if row_indices is not None:
                I_old = np.kron(np.ones((n, 1), dtype=int), row_indices)
                I_new = np.hstack((I_old, I_new))
            
            # Select indices
            new_indices = I_new[indices]
        else:
            # Reshape Z for right-to-left sweep
            Z_mat = Z.reshape(r1, n * r2).T
            
            # QR decomposition
            Q, R = qr(Z_mat, mode='economic')
            
            # Find maximum volume submatrix
            indices, B = MaxvolHelper.maxvol_rect(Q, tau, dr_min, dr_max)
            
            # Reshape core
            G = B.T.reshape(-1, n, r2)
            
            # Create new column indices
            I_new = np.kron(np.ones((r2, 1), dtype=int), grid_indices)
            if row_indices is not None:
                I_old = np.kron(row_indices, np.ones((n, 1), dtype=int))
                I_new = np.hstack((I_new, I_old))
            
            # Select indices
            new_indices = I_new[indices]
        
        return G, R, new_indices
    
    def _estimate_relative_change(self, tt_new, tt_old):
        """
        Estimate relative change between two TT tensors.
        
        Parameters:
        -----------
        tt_new : TTDecomposition
            New TT tensor
        tt_old : TTDecomposition
            Old TT tensor
            
        Returns:
        --------
        float
            Estimated relative change
        """
        # For small tensors, we can compute the full tensors
        if np.prod(self.shapes) < 1e6:
            full_new = tt_new.get_full_tensor()
            full_old = tt_old.get_full_tensor()
            return np.linalg.norm(full_new - full_old) / np.linalg.norm(full_old)
        
        # For large tensors, sample random points
        n_samples = 1000
        indices = np.array([
            [np.random.randint(0, shape) for shape in self.shapes]
            for _ in range(n_samples)
        ])
        
        vals_new = tt_new.evaluate(indices)
        vals_old = tt_old.evaluate(indices)
        
        return np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_old)
    
    def construct_coefficients_tensor(self):
        """
        Construct tensor C of Chebyshev coefficients from tensor P.
        
        Returns:
        --------
        TTDecomposition
            TT tensor of Chebyshev coefficients.
        """
        print("Computing Chebyshev coefficients...")
        start_time = time.time()
        
        if self.P is None:
            raise ValueError("Tensor P not constructed. Call cross_approximation first.")
        
        # Create deep copy of P
        self.C = TTDecomposition(self.shapes)
        self.C.cores = [core.copy() for core in self.P.cores]
        self.C.ranks = self.P.ranks.copy()
        
        # Apply DCT-I transformation to each core
        for mode in range(self.dims):
            # Construct the basis matrix for this dimension
            F_n = self._construct_basis_matrix(self.degrees[mode])
            
            # Get current core
            core = self.C.cores[mode]
            r1, n, r2 = core.shape
            
            # Apply transformation
            core_transformed = np.zeros_like(core)
            
            for i in range(r1):
                for j in range(r2):
                    slice_vec = core[i, :, j]
                    # Apply DCT-I transformation
                    core_transformed[i, :, j] = F_n @ slice_vec
            
            # Update core
            self.C.cores[mode] = core_transformed
        
        elapsed = time.time() - start_time
        print(f"Coefficient computation completed in {elapsed:.2f} seconds")
        
        return self.C
    
    def evaluate_chebyshev_basis(self, point):
        """
        Evaluate Chebyshev basis functions at a point.
        
        Parameters:
        -----------
        point : array-like
            Parameter values at which to evaluate
            
        Returns:
        --------
        list
            List of arrays with Chebyshev polynomial values for each dimension
        """
        # Map point to canonical domain [-1, 1]
        z = []
        for i, (a, b) in enumerate(self.domains):
            p = point[i]
            z.append(2.0 * (p - a) / (b - a) - 1.0)
        
        # Evaluate Chebyshev polynomials
        T = []
        for dim, z_val in enumerate(z):
            T_dim = np.zeros(self.degrees[dim] + 1)
            T_dim[0] = 1.0  # T_0(x) = 1
            
            if self.degrees[dim] > 0:
                T_dim[1] = z_val  # T_1(x) = x
                
                # Use recurrence relation for higher order polynomials
                for j in range(2, self.degrees[dim] + 1):
                    T_dim[j] = 2.0 * z_val * T_dim[j-1] - T_dim[j-2]
            
            T.append(T_dim)
        
        return T
    
    def evaluate(self, params):
        """
        Evaluate the TT-Chebyshev interpolation at parameter values.
        
        Parameters:
        -----------
        params : array-like
            Parameter values at which to evaluate
            
        Returns:
        --------
        float
            Interpolated option price
        """
        if self.C is None:
            raise ValueError("Coefficients tensor not constructed.")
            
        # Make sure params is a list/array
        if np.isscalar(params):
            params = [params]
            
        # Evaluate Chebyshev basis at the point
        basis_vals = self.evaluate_chebyshev_basis(params)
        
        # Set up TensorNetwork for contraction
        nodes = []
        
        # Create nodes for each core
        for i, core in enumerate(self.C.cores):
            node = tn.Node(core, name=f"core_{i}")
            nodes.append(node)
        
        # Connect the cores
        for i in range(len(nodes) - 1):
            nodes[i][2] ^ nodes[i+1][0]
        
        # Create and connect basis nodes
        for i, basis in enumerate(basis_vals):
            basis_node = tn.Node(basis, name=f"basis_{i}")
            basis_node[0] ^ nodes[i][1]
        
        # Contract the network
        result = nodes[0]
        for i in range(1, len(nodes)):
            result = result @ nodes[i]
            
        for i, basis in enumerate(basis_vals):
            basis_node = tn.Node(basis, name=f"basis_{i}")
            result = result @ basis_node
        
        # Get the final value
        return result.tensor.item()
    
    def evaluate_batch(self, params_list):
        """
        Evaluate the TT-Chebyshev interpolation at multiple parameter points.
        
        Parameters:
        -----------
        params_list : list of array-like
            List of parameter values at which to evaluate
            
        Returns:
        --------
        numpy.ndarray
            Interpolated option prices
        """
        results = np.zeros(len(params_list))
        
        for i, params in enumerate(params_list):
            results[i] = self.evaluate(params)
            
        return results

    def run_offline_phase(self, pricing_func, validation_size=0.2, max_rank=20, eps=1e-8, 
                         dr_min=1, dr_max=5, tau=1.1, max_sweeps=10):
        """
        Run complete offline phase: cross approximation and coefficient construction.
        
        Parameters:
        -----------
        pricing_func : callable
            Function that takes parameter values and returns option price.
        validation_size : float or int
            Fraction or number of points to use for validation.
        max_rank : int
            Maximum TT-rank allowed.
        eps : float
            Desired approximation accuracy.
        dr_min : int
            Minimum rank increment in maxvol_rect.
        dr_max : int
            Maximum rank increment in maxvol_rect.
        tau : float
            Accuracy parameter for maxvol_rect.
        max_sweeps : int
            Maximum number of sweeps for cross approximation.
            
        Returns:
        --------
        TTDecomposition
            TT tensor of Chebyshev coefficients.
        """
        print("=== OFFLINE PHASE ===")
        offline_start = time.time()
        
        # Create validation function that wraps pricing_func
        if validation_size > 0:
            # Generate validation data
            n_val = int(validation_size) if validation_size > 1 else int(validation_size * 1000)
            
            # Generate random points in the parameter space
            val_params = []
            for i in range(n_val):
                point = []
                for dim, (a, b) in enumerate(self.domains):
                    point.append(a + (b - a) * np.random.random())
                val_params.append(point)
            
            # Evaluate reference prices
            val_prices = np.array([pricing_func(p) for p in val_params])
            
            print(f"Generated {n_val} validation points")
        
        # Step 1: Perform cross approximation to build tensor P
        print("Step 1: Building tensor P using cross approximation...")
        self.P = self.cross_approximation(
            pricing_func, max_rank=max_rank, eps=eps, dr_min=dr_min, 
            dr_max=dr_max, tau=tau, max_sweeps=max_sweeps
        )
        
        # Step 2: Construct Chebyshev coefficients tensor C
        print("Step 2: Computing Chebyshev coefficients...")
        self.C = self.construct_coefficients_tensor()
        
        # Validate the approximation
        if validation_size > 0:
            print("Validating approximation...")
            val_approx = self.evaluate_batch(val_params)
            val_error = np.abs(val_approx - val_prices)
            rel_error = np.linalg.norm(val_error) / np.linalg.norm(val_prices)
            
            print(f"Validation results:")
            print(f"  Mean absolute error: {np.mean(val_error):.8e}")
            print(f"  Max absolute error: {np.max(val_error):.8e}")
            print(f"  Relative L2 error: {rel_error:.8e}")
        
        offline_elapsed = time.time() - offline_start
        print(f"Offline phase completed in {offline_elapsed:.2f} seconds")
        
        return self.C
    
    def run_online_phase(self, params):
        """
        Run the online phase to evaluate the interpolation at parameter values.
        
        Parameters:
        -----------
        params : array-like or list of array-like
            Parameter value(s) at which to evaluate
            
        Returns:
        --------
        float or numpy.ndarray
            Interpolated option price(s)
        """
        print("=== ONLINE PHASE ===")
        online_start = time.time()
        
        # Check if we have a single point or multiple points
        if np.array(params).ndim == 1:
            result = self.evaluate(params)
        else:
            result = self.evaluate_batch(params)
        
        online_elapsed = time.time() - online_start
        print(f"Online phase completed in {online_elapsed:.6f} seconds")
        
        return result


# Example usage with Black-Scholes option pricing
def test_tt_option_pricing():
    """Test TT-Chebyshev approximation with Black-Scholes option pricing."""
    from scipy.stats import norm
    
    # Black-Scholes call option price
    def black_scholes_call(params):
        S, K, T, r, sigma = params
        
        if T <= 0:
            return max(0, S - K)
            
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    # Define parameter domains
    domains = [(90, 110),   # S - stock price
              (95, 105),    # K - strike price
              (0.1, 2),     # T - time to maturity
              (0.01, 0.1),  # r - interest rate
              (0.1, 0.4)]   # sigma - volatility
    
    # Degrees for Chebyshev interpolation
    degrees = [5, 5, 5, 4, 4]
    
    # Create TT-Chebyshev approximation
    tt_approx = TTChebyshevApproximation(domains, degrees)
    
    # Run offline phase
    tt_approx.run_offline_phase(
        black_scholes_call, 
        validation_size=200,  # Use 200 random points for validation
        max_rank=15,         # Maximum allowed TT-rank
        eps=1e-6,            # Desired accuracy
        dr_min=1,            # Minimum rank increment
        dr_max=3,            # Maximum rank increment
        max_sweeps=10        # Maximum number of sweeps
    )
    
    # Test the approximation
    print("\nTesting TT-Chebyshev approximation...")
    
    # Generate test points
    n_test = 100
    test_params = []
    
    for i in range(n_test):
        point = []
        for dim, (a, b) in enumerate(domains):
            point.append(a + (b - a) * np.random.random())
        test_params.append(point)
    
    # Evaluate using both methods
    tt_prices = tt_approx.evaluate_batch(test_params)
    exact_prices = np.array([black_scholes_call(p) for p in test_params])
    
    # Calculate errors
    errors = np.abs(tt_prices - exact_prices)
    rel_errors = errors / exact_prices
    
    print(f"Test results on {n_test} random points:")
    print(f"  Mean absolute error: {np.mean(errors):.8e}")
    print(f"  Max absolute error: {np.max(errors):.8e}")
    print(f"  Mean relative error: {np.mean(rel_errors):.8e}")
    print(f"  Max relative error: {np.max(rel_errors):.8e}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot prices
    plt.subplot(2, 2, 1)
    plt.scatter(exact_prices, tt_prices)
    plt.plot([min(exact_prices), max(exact_prices)], 
             [min(exact_prices), max(exact_prices)], 'r--')
    plt.xlabel('Exact prices')
    plt.ylabel('TT-Chebyshev prices')
    plt.title('Price Comparison')
    plt.grid(True)
    
    # Plot absolute errors
    plt.subplot(2, 2, 2)
    plt.semilogy(range(n_test), errors, 'bo')
    plt.xlabel('Test point index')
    plt.ylabel('Absolute error')
    plt.title('Absolute Errors')
    plt.grid(True)
    
    # Plot relative errors
    plt.subplot(2, 2, 3)
    plt.semilogy(range(n_test), rel_errors, 'go')
    plt.xlabel('Test point index')
    plt.ylabel('Relative error')
    plt.title('Relative Errors')
    plt.grid(True)
    
    # Plot error histogram
    plt.subplot(2, 2, 4)
    plt.hist(errors, bins=20)
    plt.xlabel('Absolute error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compare online phase timing
    print("\nComparing execution time...")
    
    # Time for exact computation
    start_time = time.time()
    for p in test_params:
        black_scholes_call(p)
    exact_time = time.time() - start_time
    
    # Time for TT-Chebyshev
    start_time = time.time()
    tt_approx.evaluate_batch(test_params)
    tt_time = time.time() - start_time
    
    print(f"Timing results for {n_test} evaluations:")
    print(f"  Exact method: {exact_time:.6f} seconds")
    print(f"  TT-Chebyshev: {tt_time:.6f} seconds")
    print(f"  Speedup: {exact_time/tt_time:.2f}x")

if __name__ == "__main__":
    test_tt_option_pricing()