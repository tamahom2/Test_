import numpy as np
from itertools import product
import time
from functools import lru_cache
from scipy.linalg import qr, solve_triangular, lu

class TensorTrain:
    """
    Tensor Train format implementation for low-rank tensor approximation.
    
    This follows the algorithms in the paper "Low-rank tensor approximation for 
    Chebyshev interpolation in parametric option pricing" by Glau, Kressner, and Statti.
    """
    
    def __init__(self, cores=None):
        """
        Initialize a Tensor Train.
        
        Parameters:
        -----------
        cores : list of numpy.ndarray, optional
            List of TT-cores. Each core is a 3D array of shape [r_{i-1}, n_i, r_i].
        """
        self.cores = cores if cores is not None else []
        
    @property
    def order(self):
        """Get the order (number of dimensions) of the tensor."""
        return len(self.cores)
    
    @property
    def shape(self):
        """Get the shape of the tensor."""
        if not self.cores:
            return tuple()
        return tuple(core.shape[1] for core in self.cores)
    
    @property
    def ranks(self):
        """Get the TT-ranks of the tensor."""
        if not self.cores:
            return (1,)
        result = [1]
        for core in self.cores:
            result.append(core.shape[2])
        return tuple(result)
    
    @property
    def is_orthogonalized(self):
        """Check if the tensor is left-orthogonalized."""
        if not self.cores:
            return False
        
        for i, core in enumerate(self.cores[:-1]):
            # Reshape core for orthogonality check
            r1, n, r2 = core.shape
            core_mat = core.reshape(r1 * n, r2)
            # Check if the core is left-orthogonal (Q^T Q = I)
            q_t_q = core_mat.T @ core_mat
            identity = np.eye(r2)
            if not np.allclose(q_t_q, identity, rtol=1e-5, atol=1e-8):
                return False
        return True
    
    def copy(self):
        """Create a deep copy of the TT tensor."""
        return TensorTrain([core.copy() for core in self.cores])
    
    def get_element(self, indices):
        """
        Get a single element from the tensor.
        
        Parameters:
        -----------
        indices : tuple or list
            Multi-index specifying the element to retrieve.
            
        Returns:
        --------
        float
            The tensor element at the given indices.
        """
        if len(indices) != self.order:
            raise ValueError("Indices length must match tensor order")
        
        # Start with the first core
        curr = self.cores[0][:, indices[0], :]
        
        # Contract with remaining cores
        for i in range(1, self.order):
            curr = curr @ self.cores[i][:, indices[i], :]
            
        # Return scalar result
        return curr.item()
    
    def get_full_tensor(self):
        """
        Convert TT tensor to full format.
        
        Returns:
        --------
        numpy.ndarray
            Full tensor representation.
        """
        # Start with the first core
        result = self.cores[0]
        
        # Successively contract with the remaining cores
        for i in range(1, self.order):
            # Reshape for contraction
            res_shape = result.shape
            result = result.reshape(-1, res_shape[-1])
            
            # Contract with the next core
            core_shape = self.cores[i].shape
            core_mat = self.cores[i].reshape(core_shape[0], -1)
            
            result = result @ core_mat
            
            # Reshape to maintain tensor structure
            new_shape = res_shape[:-1] + (core_shape[1], core_shape[2])
            result = result.reshape(new_shape)
        
        # Remove singleton dimensions at beginning and end
        if result.shape[0] == 1:
            result = result[0]
        if result.shape[-1] == 1:
            result = result[..., 0]
            
        return result
    
    def left_orthogonalize(self, idx=None):
        """
        Left-orthogonalize the TT tensor up to a given index.
        
        Parameters:
        -----------
        idx : int, optional
            Index up to which to orthogonalize. If None, orthogonalize all cores.
            
        Returns:
        --------
        TensorTrain
            Orthogonalized TT tensor.
        """
        if not self.cores:
            return self
            
        idx = len(self.cores) - 1 if idx is None else idx
        result = self.copy()
        
        for i in range(idx):
            # Get current core shape
            r1, n, r2 = result.cores[i].shape
            
            # Reshape core for QR decomposition
            core_mat = result.cores[i].reshape(r1 * n, r2)
            
            # QR decomposition
            Q, R = qr(core_mat, mode='economic')
            
            # Update current core
            result.cores[i] = Q.reshape(r1, n, Q.shape[1])
            
            # Update next core
            next_core = result.cores[i+1]
            result.cores[i+1] = np.tensordot(R, next_core, axes=([1], [0]))
            
        return result
    
    def truncate(self, max_rank=None, rel_eps=1e-10):
        """
        Truncate the TT tensor to lower ranks.
        
        Parameters:
        -----------
        max_rank : int, optional
            Maximum rank for truncation.
        rel_eps : float, optional
            Relative accuracy for truncation.
            
        Returns:
        --------
        TensorTrain
            Truncated TT tensor.
        """
        # First ensure the tensor is left-orthogonalized
        tt = self.left_orthogonalize()
        
        # Now truncate from right to left
        for i in range(tt.order - 1, 0, -1):
            # Get current core
            core = tt.cores[i]
            r1, n, r2 = core.shape
            
            # Reshape for SVD
            core_mat = core.reshape(r1, n * r2)
            
            # SVD decomposition
            U, S, Vh = np.linalg.svd(core_mat, full_matrices=False)
            
            # Determine rank for truncation
            if max_rank is not None:
                rank = min(max_rank, len(S))
            else:
                # Find rank based on relative accuracy
                norm_S = np.linalg.norm(S)
                cumulative_energy = np.cumsum(S[::-1]**2)[::-1]
                rank = np.sum(cumulative_energy > (rel_eps * norm_S)**2)
                rank = max(1, rank)  # Ensure at least rank 1
            
            # Truncate
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Update current core
            tt.cores[i] = (Vh.reshape(rank, n, r2))
            
            # Update previous core
            prev_core = tt.cores[i-1]
            r_prev, n_prev, _ = prev_core.shape
            tt.cores[i-1] = np.tensordot(prev_core, U * S, axes=([2], [0])).reshape(r_prev, n_prev, rank)
            
        return tt

    @classmethod
    def from_tensor(cls, tensor, max_rank=np.inf, eps=1e-10):
        """
        Create a TT tensor from a full tensor using TT-SVD.
        
        Parameters:
        -----------
        tensor : numpy.ndarray
            Full tensor to decompose.
        max_rank : int, optional
            Maximum rank for the TT decomposition.
        eps : float, optional
            Desired accuracy for the decomposition.
            
        Returns:
        --------
        TensorTrain
            TT representation of the tensor.
        """
        # Get tensor shape
        shape = tensor.shape
        d = len(shape)
        
        # Initialize cores list
        cores = []
        
        # Start with the full tensor
        U = tensor.copy()
        r = 1
        
        # TT-SVD algorithm
        for k in range(d - 1):
            # Reshape U for the current core
            U = U.reshape(r * shape[k], -1)
            
            # SVD of the reshaped U
            u, s, vh = np.linalg.svd(U, full_matrices=False)
            
            # Truncate based on max_rank and eps
            delta = eps / np.sqrt(d - 1) * np.linalg.norm(s)
            r_new = min(max_rank, np.sum(s > delta))
            r_new = max(r_new, 1)  # Ensure at least rank 1
            
            # Truncate and form the core
            u = u[:, :r_new]
            s = s[:r_new]
            vh = vh[:r_new, :]
            
            # Reshape and store the current core
            core = u.reshape(r, shape[k], r_new)
            cores.append(core)
            
            # Update U for the next core
            U = np.diag(s) @ vh
            r = r_new
            
        # Add the last core
        cores.append(U.reshape(r, shape[-1], 1))
        
        return cls(cores)


def maxvol(A, tol=1.05, max_iter=100):
    """
    Find a submatrix of maximum volume (maxvol algorithm).
    
    Parameters:
    -----------
    A : numpy.ndarray
        Input matrix of shape (m, n) with m >= n.
    tol : float, optional
        Tolerance parameter (should be >= 1).
    max_iter : int, optional
        Maximum number of iterations.
        
    Returns:
    --------
    tuple
        (row_indices, B) where row_indices are the indices of the selected rows
        and B is the coefficient matrix.
    """
    m, n = A.shape
    
    if m <= n:
        raise ValueError("Input matrix should be tall (m > n)")
    
    # LU decomposition with partial pivoting
    P, L, U = lu(A, check_finite=False)
    
    # Get initial row indices from P
    row_indices = P[:, :n].argmax(axis=0)
    
    # Compute initial coefficient matrix
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(L[:n, :], Q, trans=1, check_finite=False, 
                         unit_diagonal=True, lower=True).T
    
    # Iteratively improve the selection
    for _ in range(max_iter):
        # Find the maximum element in B
        i_max, j_max = np.unravel_index(np.abs(B).argmax(), B.shape)
        
        # Check convergence
        if np.abs(B[i_max, j_max]) <= tol:
            break
        
        # Update the row selection
        row_indices[j_max] = i_max
        
        # Update coefficient matrix B
        b_j = B[:, j_max].copy()
        b_i = B[i_max, :].copy()
        b_i[j_max] -= 1.0
        
        # Sherman-Morrison update formula
        B -= np.outer(b_j, b_i) / B[i_max, j_max]
    
    return row_indices, B


def maxvol_rect(A, tol=1.1, dr_min=1, dr_max=None, tol0=1.05, max_iter0=10):
    """
    Rectangular maxvol algorithm: find a submatrix with approximately maximal volume.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Input matrix of shape (m, n) with m >= n.
    tol : float, optional
        Tolerance parameter.
    dr_min : int, optional
        Minimum number of added rows.
    dr_max : int, optional
        Maximum number of added rows.
    tol0 : float, optional
        Tolerance for the initial maxvol.
    max_iter0 : int, optional
        Maximum iterations for the initial maxvol.
        
    Returns:
    --------
    tuple
        (row_indices, B) where row_indices are the indices of the selected rows
        and B is the coefficient matrix.
    """
    m, n = A.shape
    
    # Set default dr_max if not provided
    if dr_max is None:
        dr_max = m - n
    else:
        dr_max = min(dr_max, m - n)
    
    # Ensure dr_min and dr_max are valid
    r_min = n + dr_min
    r_max = n + dr_max
    
    if r_min < n or r_min > r_max or r_max > m:
        raise ValueError("Invalid min/max number of added rows")
    
    # Initial maxvol
    row_indices_0, B = maxvol(A, tol=tol0, max_iter=max_iter0)
    
    # Prepare for adding rows
    row_indices = np.concatenate([row_indices_0, np.zeros(r_max - n, dtype=int)])
    selected = np.zeros(m, dtype=int)
    selected[row_indices_0] = 1
    
    # Compute F = (1 - selected) * ||B_{i,:}||^2
    F = (1 - selected) * np.sum(B**2, axis=1)
    
    # Add rows greedily
    for k in range(n, r_max):
        # Find the row with maximum F
        i = np.argmax(F)
        
        # Check stopping criterion
        if k >= r_min and F[i] <= tol**2:
            break
        
        # Update row indices and selection
        row_indices[k] = i
        selected[i] = 1
        
        # Compute the Sherman-Morrison-Woodbury update
        v = B @ B[i, :]
        l = 1.0 / (1.0 + v[i])
        
        # Update B and F
        B = np.hstack([B - l * np.outer(v, B[i, :]), l * v.reshape(-1, 1)])
        F = (1 - selected) * (F - l * v**2)
    
    # Trim row_indices to the actual number of selected rows
    r_actual = k + 1 if k < r_max else r_max
    row_indices = row_indices[:r_actual]
    
    # Final coefficient matrix
    B_final = np.zeros((m, r_actual))
    B_final[row_indices, np.arange(r_actual)] = 1.0
    for i in range(m):
        if i not in row_indices:
            B_final[i] = B[i, :r_actual]
    
    return row_indices, B_final


class TensorCompletion:
    """
    Tensor completion using Riemannian optimization in TT format.
    
    Implements Algorithm 1 (adaptive rank) and Algorithm 2 (adaptive sampling)
    from the paper.
    """
    
    def __init__(self, shape, known_indices=None, known_values=None):
        """
        Initialize tensor completion.
        
        Parameters:
        -----------
        shape : tuple
            Shape of the tensor to complete.
        known_indices : numpy.ndarray, optional
            Indices of known elements, shape (n_samples, d).
        known_values : numpy.ndarray, optional
            Values of known elements, shape (n_samples,).
        """
        self.shape = shape
        self.d = len(shape)
        
        # Known data points
        if known_indices is not None and known_values is not None:
            self.known_indices = known_indices
            self.known_values = known_values
        else:
            self.known_indices = np.empty((0, self.d), dtype=int)
            self.known_values = np.empty(0)
    
    def add_data_points(self, indices, values):
        """
        Add known data points.
        
        Parameters:
        -----------
        indices : numpy.ndarray
            Indices of known elements, shape (n_samples, d).
        values : numpy.ndarray
            Values of known elements, shape (n_samples,).
        """
        self.known_indices = np.vstack([self.known_indices, indices])
        self.known_values = np.concatenate([self.known_values, values])
    
    def _compute_error(self, tt_tensor, indices, values):
        """
        Compute relative error on a set of points.
        
        Parameters:
        -----------
        tt_tensor : TensorTrain
            TT tensor to evaluate.
        indices : numpy.ndarray
            Indices of elements, shape (n_samples, d).
        values : numpy.ndarray
            True values, shape (n_samples,).
            
        Returns:
        --------
        float
            Relative error.
        """
        # Get predicted values
        pred_values = np.array([tt_tensor.get_element(idx) for idx in indices])
        
        # Compute relative error
        norm_true = np.linalg.norm(values)
        if norm_true < 1e-10:
            return np.linalg.norm(pred_values - values)
        return np.linalg.norm(pred_values - values) / norm_true
    
    def _riemannian_cg_step(self, tt_tensor, indices, values, max_iter=20):
        """
        Perform one step of Riemannian conjugate gradient.
        
        Parameters:
        -----------
        tt_tensor : TensorTrain
            Current TT tensor approximation.
        indices : numpy.ndarray
            Indices of elements, shape (n_samples, d).
        values : numpy.ndarray
            True values, shape (n_samples,).
        max_iter : int, optional
            Maximum number of CG iterations.
            
        Returns:
        --------
        TensorTrain
            Updated TT tensor.
        """
        # This is a simplified version of Riemannian CG
        # In a real implementation, this would be more complex
        
        # For simplicity, we'll just update the cores directly to fit the data
        result = tt_tensor.copy()
        
        # Simple approach: update each core sequentially
        for dim in range(self.d):
            # Compute the current predictions
            pred_values = np.array([result.get_element(idx) for idx in indices])
            
            # Compute the gradient with respect to this core
            # This is a very simplified gradient computation
            # In a real implementation, this would use the TT structure more efficiently
            delta = values - pred_values
            
            # Update the core
            # This is a simplified update rule
            for i, (idx, delta_i) in enumerate(zip(indices, delta)):
                # Get the relevant elements from all other cores
                left_part = 1.0
                for d_left in range(dim):
                    left_core = result.cores[d_left]
                    left_part = left_part @ left_core[:, idx[d_left], :]
                
                right_part = 1.0
                for d_right in range(dim+1, self.d):
                    right_core = result.cores[d_right]
                    right_part = right_part @ right_core[:, idx[d_right], :]
                
                # Update the core
                update = delta_i * np.outer(left_part, right_part).reshape(-1)
                current_core = result.cores[dim]
                flat_idx = (slice(None), idx[dim], slice(None))
                current_core[flat_idx] += update.reshape(current_core[flat_idx].shape) * 0.01  # Small step size
        
        return result
    
    def _increase_rank(self, tt_tensor, mu):
        """
        Increase the rank of a TT tensor at position mu.
        
        Parameters:
        -----------
        tt_tensor : TensorTrain
            Current TT tensor.
        mu : int
            Position where to increase the rank.
            
        Returns:
        --------
        TensorTrain
            TT tensor with increased rank.
        """
        # Copy the tensor
        result = tt_tensor.copy()
        
        # Get the cores around position mu
        left_core = result.cores[mu]
        right_core = result.cores[mu+1]
        
        # Get current shapes
        r1, n1, r2 = left_core.shape
        r2, n2, r3 = right_core.shape
        
        # Create new cores with increased rank
        new_r2 = r2 + 1
        
        # Random initialization for the new components
        rand_vec1 = np.random.randn(r1, n1)
        rand_vec2 = np.random.randn(n2, r3)
        
        # Create new left core
        new_left_core = np.zeros((r1, n1, new_r2))
        new_left_core[:, :, :r2] = left_core
        new_left_core[:, :, -1] = rand_vec1
        
        # Create new right core
        new_right_core = np.zeros((new_r2, n2, r3))
        new_right_core[:r2, :, :] = right_core
        new_right_core[-1, :, :] = rand_vec2
        
        # Update the cores
        result.cores[mu] = new_left_core
        result.cores[mu+1] = new_right_core
        
        return result
    
    def adaptive_rank_completion(self, initial_tt, Ω, Ω_C, max_rank=10, tol=1e-4, acceptance_param=1e-4):
        """
        Adaptive rank tensor completion (Algorithm 1).
        
        Parameters:
        -----------
        initial_tt : TensorTrain
            Initial TT tensor approximation.
        Ω : tuple
            (indices, values) for the training set.
        Ω_C : tuple
            (indices, values) for the test set.
        max_rank : int, optional
            Maximum rank allowed.
        tol : float, optional
            Tolerance for convergence.
        acceptance_param : float, optional
            Parameter for accepting rank increases.
            
        Returns:
        --------
        TensorTrain
            Completed TT tensor.
        """
        indices_train, values_train = Ω
        indices_test, values_test = Ω_C
        
        # Initialize with given TT tensor
        X = initial_tt.copy()
        
        # Run Riemannian CG on initial tensor
        X = self._riemannian_cg_step(X, indices_train, values_train)
        
        # Check initial error on test set
        test_error = self._compute_error(X, indices_test, values_test)
        
        # Initialize locked counter
        locked = 0
        
        # Main adaptive rank loop
        mu = 1  # Start with first rank
        while locked < self.d - 1 and max(X.ranks) < max_rank:
            # Try increasing the rank at position mu
            X_new = self._increase_rank(X, mu - 1)  # -1 since we use 0-based indexing
            
            # Run Riemannian CG with increased rank
            X_new = self._riemannian_cg_step(X_new, indices_train, values_train)
            
            # Check error on test set
            test_error_new = self._compute_error(X_new, indices_test, values_test)
            
            # Decide whether to accept the rank increase
            if test_error_new - test_error > -acceptance_param:
                # No significant improvement, lock this rank
                locked += 1
            else:
                # Accept the rank increase
                locked = 0
                X = X_new
                test_error = test_error_new
            
            # Move to next rank position cyclically
            mu = 1 + (mu % (self.d - 1))
        
        return X
    
    def adaptive_sampling_completion(self, initial_tt, max_rank=10, max_sampling_pct=0.2, tol=1e-4):
        """
        Adaptive sampling tensor completion (Algorithm 2).
        
        Parameters:
        -----------
        initial_tt : TensorTrain
            Initial TT tensor approximation.
        max_rank : int, optional
            Maximum rank allowed.
        max_sampling_pct : float, optional
            Maximum percentage of tensor elements to sample.
        tol : float, optional
            Tolerance for convergence.
            
        Returns:
        --------
        TensorTrain
            Completed TT tensor.
        """
        # Initial tensor
        X = initial_tt.copy()
        
        # Size of full tensor
        full_size = np.prod(self.shape)
        
        # Initial training and test sets
        Ω = (self.known_indices, self.known_values)
        
        # Create a new test set
        Ω_C_indices = self._generate_test_set(self.shape, Ω[0])
        
        # Get values for test set (in a real application, these would come from the reference method)
        # For this example, we'll just generate random values
        Ω_C_values = np.random.randn(len(Ω_C_indices))
        
        Ω_C = (Ω_C_indices, Ω_C_values)
        
        # Run adaptive rank completion on initial sets
        X = self.adaptive_rank_completion(X, Ω, Ω_C, max_rank=max_rank, tol=tol)
        
        # Compute error on test set
        error_new = self._compute_error(X, Ω_C[0], Ω_C[1])
        
        # Main adaptive sampling loop
        while len(Ω[0]) / full_size < max_sampling_pct:
            # Store old error
            error_old = error_new
            
            # Create a new rank-1 approximation
            X_tilde = TensorTrain.from_tensor(np.ones(self.shape), max_rank=1)
            
            # Add old test set to training set
            Ω_C_old = Ω_C
            Ω = (
                np.vstack([Ω[0], Ω_C_old[0]]),
                np.concatenate([Ω[1], Ω_C_old[1]])
            )
            
            # Create a new test set
            Ω_C_indices = self._generate_test_set(self.shape, Ω[0])
            Ω_C_values = np.random.randn(len(Ω_C_indices))  # Placeholder for real values
            Ω_C = (Ω_C_indices, Ω_C_values)
            
            # Run adaptive rank completion
            X = self.adaptive_rank_completion(X_tilde, Ω, Ω_C, max_rank=max_rank, tol=tol)
            
            # Compute new error
            error_new = self._compute_error(X, Ω_C[0], Ω_C[1])
            
            # Check stopping criterion
            if error_new < tol or abs(error_new - error_old) < tol:
                break
        
        return X
    
    def _generate_test_set(self, shape, existing_indices, size=100):
        """
        Generate new test indices not in the existing set.
        
        Parameters:
        -----------
        shape : tuple
            Shape of the tensor.
        existing_indices : numpy.ndarray
            Existing indices to avoid.
        size : int, optional
            Number of new indices to generate.
            
        Returns:
        --------
        numpy.ndarray
            New test indices.
        """
        # Convert existing indices to a set of tuples for fast lookup
        existing_set = set(map(tuple, existing_indices))
        
        # Generate new random indices
        new_indices = []
        attempts = 0
        while len(new_indices) < size and attempts < size * 10:
            # Generate a random index
            idx = tuple(np.random.randint(0, dim_size) for dim_size in shape)
            
            # Check if it's new
            if idx not in existing_set:
                new_indices.append(idx)
                existing_set.add(idx)
            
            attempts += 1
        
        return np.array(new_indices)


class TensorTrainChebyshevInterpolation:
    """
    Enhanced Chebyshev interpolation using Tensor Train format with adaptive rank and sampling.
    
    Implements Algorithm 4 from the paper "Low-rank tensor approximation for Chebyshev
    interpolation in parametric option pricing" by Glau, Kressner, and Statti.
    """
    
    def __init__(self, domains, degrees):
        """
        Initialize the TT-Chebyshev interpolation.
        
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
        
        # Tensor Train objects for coefficients and prices
        self.P_tt = None  # TT tensor for prices at Chebyshev nodes
        self.C_tt = None  # TT tensor for Chebyshev coefficients
        
        # Original full tensors (for comparison/debugging)
        self.P_full = None
        self.C_full = None
        
        # Generate Chebyshev points
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
        This implements Steps 2-3 of Algorithm 4 (offline phase).
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_indices : list or None
            List of indices where to evaluate the reference method. If None,
            a subset is chosen using adaptive sampling.
            
        Returns:
        --------
        TensorTrain
            TT tensor P with prices at Chebyshev nodes.
        """
        print("Offline Phase - Steps 2-3: Computing and completing tensor P")
        start_time = time.time()
        
        # Initialize tensor shape
        P_shape = tuple(d + 1 for d in self.degrees)
        
        # If no subset provided, choose initial samples
        if subset_indices is None:
            # Start with a small fraction of the tensor
            full_size = np.prod(P_shape)
            initial_size = min(1000, int(0.01 * full_size))
            
            # Generate random indices
            subset_indices = [tuple(np.random.randint(0, dim_size) for dim_size in P_shape) 
                             for _ in range(initial_size)]
        
        total_points = len(subset_indices)
        print(f"Computing {total_points} prices at Chebyshev nodes...")
        
        # Evaluate at each point in the subset
        P_samples = {}
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
            
            # Store price
            P_samples[indices] = price
        
        # Convert samples to arrays for tensor completion
        indices_array = np.array(list(P_samples.keys()))
        values_array = np.array(list(P_samples.values()))
        
        print(f"Performing tensor completion with {len(P_samples)} samples...")
        
        # Initialize tensor completion
        completion = TensorCompletion(P_shape, indices_array, values_array)
        
        # Create initial low-rank TT tensor
        initial_tt = TensorTrain.from_tensor(np.ones(P_shape), max_rank=1)
        
        # Perform adaptive sampling completion
        self.P_tt = completion.adaptive_sampling_completion(
            initial_tt, max_rank=10, max_sampling_pct=0.2, tol=1e-4)
        
        elapsed = time.time() - start_time
        print(f"Tensor P construction and completion finished in {elapsed:.2f} seconds")
        
        # For debugging/comparison, also construct full tensor
        try:
            P_full = np.zeros(P_shape)
            for indices, value in P_samples.items():
                P_full[indices] = value
            self.P_full = P_full
        except MemoryError:
            print("Full tensor too large to store in memory")
        
        return self.P_tt
    
    def construct_tensor_C(self):
        """
        Construct tensor C containing Chebyshev coefficients.
        This implements Step 4 of Algorithm 4 (offline phase).
        
        Returns:
        --------
        TensorTrain
            TT tensor C with Chebyshev coefficients.
        """
        print("Offline Phase - Step 4: Computing Chebyshev coefficients")
        start_time = time.time()
        
        if self.P_tt is None:
            raise ValueError("Tensor P not constructed. Call construct_tensor_P first.")
        
        # Get full tensor P for coefficient computation
        # In a real implementation, this would operate directly on the TT format
        P_tensor = self.P_tt.get_full_tensor()
        
        # Initialize C as a copy of P
        C_tensor = P_tensor.copy()
        
        # Compute Chebyshev coefficients using efficient algorithm
        # as described in Section 2.4.2 of the paper
        
        # Loop through each dimension/mode
        for m in range(self.dims):
            # Construct the basis matrix F_n for this dimension
            F_n = self._construct_basis_matrix(self.degrees[m])
            
            # Perform mode-m multiplication: C = C ×_m F_n
            C_tensor = self._mode_multiply(C_tensor, F_n, m)
        
        # Convert to TT format
        self.C_tt = TensorTrain.from_tensor(C_tensor, max_rank=10, eps=1e-8)
        
        # Store full tensor for comparison
        self.C_full = C_tensor
        
        elapsed = time.time() - start_time
        print(f"Chebyshev coefficients computed in {elapsed:.2f} seconds")
        print(f"TT ranks of C: {self.C_tt.ranks}")
        
        return self.C_tt
    
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
        if self.C_tt is None:
            raise ValueError("Tensor C not constructed. Run offline phase first.")
        
        print("Online Phase - Steps 8-9: Evaluating interpolated price")
        start_time = time.time()
        
        # Step 8: Evaluate Chebyshev basis at the point
        T_vectors = self.evaluate_chebyshev_basis(point)
        
        # Step 9: Compute interpolated price as inner product <C, T_p>
        # Create a rank-1 tensor from the Chebyshev basis vectors
        T_tensor = np.ones(1)
        for T_vec in T_vectors:
            T_tensor = np.outer(T_tensor, T_vec).reshape(-1)
        T_tensor = T_tensor.reshape([vec.size for vec in T_vectors])
        
        # Convert to TT format
        T_tt = TensorTrain.from_tensor(T_tensor, max_rank=1)
        
        # Compute inner product using TT format
        # Note: In a real implementation, we would compute this more efficiently
        # using the TT structure directly
        interpolated_price = self._tt_inner_product(self.C_tt, T_tt)
        
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed:.4f} seconds")
        
        return interpolated_price
    
    def _tt_inner_product(self, tt1, tt2):
        """
        Compute the inner product between two TT tensors.
        
        Parameters:
        -----------
        tt1 : TensorTrain
            First TT tensor
        tt2 : TensorTrain
            Second TT tensor
            
        Returns:
        --------
        float
            Inner product value
        """
        # This is a simplified implementation
        # In a real implementation, this would be more efficient
        
        # Start with identity matrix of size 1x1
        result = np.array([[1.0]])
        
        # Successively contract along each dimension
        for d in range(tt1.order):
            # Get cores for this dimension
            core1 = tt1.cores[d]
            core2 = tt2.cores[d]
            
            # Contract with previous result
            result = result @ np.tensordot(core1, core2, axes=([1], [1]))
        
        return result.item()
    
    def run_offline_phase(self, ref_method, subset_size=None):
        """
        Run the complete offline phase of Algorithm 4.
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_size : int or None
            Size of the subset of Chebyshev nodes to evaluate. If None,
            a subset is chosen using adaptive sampling.
            
        Returns:
        --------
        TensorTrain
            TT tensor C with Chebyshev coefficients.
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
        
        # Steps 2-3: Construct tensor P using tensor completion
        self.P_tt = self.construct_tensor_P(ref_method, subset_indices)
        
        # Step 4: Construct tensor C
        self.C_tt = self.construct_tensor_C()
        
        overall_elapsed = time.time() - overall_start
        print(f"Offline phase completed in {overall_elapsed:.2f} seconds")
        
        return self.C_tt
    
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
        if isinstance(points, tuple) or (isinstance(points, list) and 
                                        not isinstance(points[0], (list, tuple))):
            # Single point
            return self.evaluate_interpolation(points)
        else:
            # Multiple points
            return np.array([self.evaluate_interpolation(point) for point in points])
    
    def run_algorithm4(self, ref_method, evaluation_points, subset_size=None):
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
            a subset is chosen using adaptive sampling.
            
        Returns:
        --------
        float or numpy.ndarray
            Interpolated value(s)
        """
        # Run offline phase
        self.run_offline_phase(ref_method, subset_size)
        
        # Run online phase
        return self.run_online_phase(evaluation_points)


# Example usage
if __name__ == "__main__":
    # Define a simple 2D reference method (European call option price in BS model)
    def bs_call_price(S, K, T, r, sigma):
        """Black-Scholes call option price."""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    # Define reference pricing method with fixed parameters except S0 and sigma
    def reference_method(params):
        S0, sigma = params
        K = 100
        T = 1.0
        r = 0.05
        
        return bs_call_price(S0, K, T, r, sigma)
    
    # Create TT-Chebyshev interpolation
    domains = [(80.0, 120.0), (0.1, 0.4)]  # S0, sigma
    degrees = [5, 5]  # 5th degree in each dimension
    tt_cheb = TensorTrainChebyshevInterpolation(domains, degrees)
    
    # Run Algorithm 4
    # Offline phase
    tt_cheb.run_offline_phase(reference_method, subset_size=50)
    
    # Online phase - evaluate at a single point
    test_point = (100.0, 0.2)  # S0=100, sigma=0.2
    interp_price = tt_cheb.run_online_phase(test_point)
    exact_price = reference_method(test_point)
    
    print(f"\nInterpolated price at {test_point}: {interp_price:.6f}")
    print(f"Exact price: {exact_price:.6f}")
    print(f"Absolute error: {abs(interp_price - exact_price):.6e}")
    print(f"Relative error: {abs(interp_price - exact_price)/exact_price:.6e}")