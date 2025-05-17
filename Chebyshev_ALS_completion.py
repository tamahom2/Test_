import numpy as np
from itertools import product
import time
from copy import deepcopy

# Import the Tensor Train Completion implementation
class TTCompletion:
    """Tensor completion using Tensor Train decomposition with adaptive rank selection."""
    
    def __init__(self, shape, initial_tt_ranks=None, random_seed=None, max_rank=30):
        """
        Initialize TT decomposition for tensor completion with adaptive ranks.
        
        Args:
            shape: Shape of the tensor to complete
            initial_tt_ranks: Initial TT ranks (if None, will start with all ranks = 2)
            random_seed: Optional random seed for reproducibility
            max_rank: Maximum allowed rank for any bond dimension
        """
        self.shape = shape
        self.ndim = len(shape)
        self.max_rank = max_rank
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # If no initial ranks provided, start with low ranks
        if initial_tt_ranks is None:
            initial_tt_ranks = [2] * (self.ndim - 1)
        elif len(initial_tt_ranks) != self.ndim - 1:
            raise ValueError(f"Expected {self.ndim-1} TT ranks, got {len(initial_tt_ranks)}")
            
        # Include the boundary ranks (r₀=r_N=1 by definition)
        self.tt_ranks = [1] + list(initial_tt_ranks) + [1]
        
        # Initialize TT cores with appropriate scaling
        self.tt_cores = self._initialize_cores()
        
    def _initialize_cores(self):
        """Initialize TT-cores with appropriate dimensions and scaling."""
        cores = []
        for k in range(self.ndim):
            # Each core has shape (r_{k-1}, n_k, r_k)
            core_shape = (self.tt_ranks[k], self.shape[k], self.tt_ranks[k+1])
            
            # Scale initialization based on core size
            scale = 1.0 / np.sqrt(max(core_shape[0], core_shape[2]))
            core = np.random.randn(*core_shape) * scale
            
            cores.append(core)
        return cores
    
    def fit(self, coords, values, max_iter=20, tol=1e-4, max_time=600, verbose=True,
            validation_split=0.2, rank_increase_tol=0.01):
        """
        Fit the TT model with batch rank increases.
        
        Args:
            coords: Coordinates of observed entries, shape (n_samples, ndim)
            values: Values of observed entries, shape (n_samples,)
            max_iter: Maximum number of iterations for each rank configuration
            tol: Convergence tolerance for relative improvement
            max_time: Maximum training time in seconds
            verbose: Whether to print progress information
            validation_split: Fraction of data to use for validation
            rank_increase_tol: Tolerance for rank increase (progress threshold)
            
        Returns:
            Dictionary with training history
        """
        n_samples = len(values)
        if coords.shape[0] != n_samples:
            raise ValueError("Number of coordinates must match number of values")
        
        # Split data into training and validation sets
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_coords = coords[train_indices]
        train_values = values[train_indices]
        val_coords = coords[val_indices]
        val_values = values[val_indices]
        
        # Convert to proper types
        train_coords = train_coords.astype(int)
        train_values = train_values.astype(np.float64)
        val_coords = val_coords.astype(int)
        val_values = val_values.astype(np.float64)
        
        # Normalize values for better training stability
        values_mean = np.mean(train_values)
        values_std = np.std(train_values)
        if values_std > 0:
            normalized_train_values = (train_values - values_mean) / values_std
            normalized_val_values = (val_values - values_mean) / values_std
        else:
            normalized_train_values = train_values - values_mean
            normalized_val_values = val_values - values_mean
            
        # Store normalization parameters
        self.values_mean = values_mean
        self.values_std = values_std if values_std > 0 else 1.0
        
        # Training history
        history = {
            'train_rmse': [], 
            'val_rmse': [], 
            'time': [],
            'ranks': []
        }
        
        # First optimize with initial ranks
        if verbose:
            initial_ranks_str = "-".join(str(r) for r in self.tt_ranks[1:-1])
            print(f"Initial optimization with ranks [{initial_ranks_str}]...")
        
        start_time = time.time()
        
        # Train the initial model with more iterations
        self._optimize_model(train_coords, normalized_train_values, max_iter*2, tol/10, verbose)
        
        # Evaluate initial model on normalized values
        train_pred = self.predict(train_coords, normalize=True)
        train_rmse = np.sqrt(np.mean((train_pred - normalized_train_values) ** 2))
        
        val_pred = self.predict(val_coords, normalize=True)
        val_rmse = np.sqrt(np.mean((val_pred - normalized_val_values) ** 2))
        
        # Record unnormalized RMSE for display
        train_pred_unnorm = self.predict(train_coords, normalize=False)
        train_rmse_unnorm = np.sqrt(np.mean((train_pred_unnorm - train_values) ** 2))
        
        val_pred_unnorm = self.predict(val_coords, normalize=False)
        val_rmse_unnorm = np.sqrt(np.mean((val_pred_unnorm - val_values) ** 2))
        
        curr_time = time.time() - start_time
        
        # Record history
        history['train_rmse'].append(train_rmse_unnorm)
        history['val_rmse'].append(val_rmse_unnorm)
        history['time'].append(curr_time)
        history['ranks'].append(self.tt_ranks.copy())
        
        if verbose:
            print(f"Initial Train RMSE: {train_rmse_unnorm:.6f}, Val RMSE: {val_rmse_unnorm:.6f}, Time: {curr_time:.2f}s")
        
        # Previous validation RMSE for comparing improvement (use normalized values internally)
        prev_val_rmse = val_rmse
        best_val_rmse = val_rmse
        
        # Track consecutive failures
        consecutive_failures = 0
        max_consecutive_failures = 10  # Stop after this many failures
        
        # Batch rank increase strategy
        try:
            rank_iteration = 0
            
            while rank_iteration < 10:  # Maximum 10 rank increases
                rank_iteration += 1
                
                if verbose:
                    print(f"\nTrying rank increase iteration {rank_iteration}...")
                
                # Check if we've exceeded time limit
                if time.time() - start_time > max_time:
                    print(f"\nStopping early: reached {max_time} seconds time limit")
                    break
                
                # Save current model state
                old_cores = [core.copy() for core in self.tt_cores]
                old_ranks = self.tt_ranks.copy()
                
                # Determine which ranks to increase
                increased = False
                for bond_idx in range(1, self.ndim):
                    if self.tt_ranks[bond_idx] < self.max_rank:
                        if verbose:
                            print(f"  Increasing rank at bond {bond_idx} from {self.tt_ranks[bond_idx]} to {self.tt_ranks[bond_idx]+1}")
                        self._increase_bond_rank(bond_idx, self.tt_ranks[bond_idx] + 1, noise_scale=0.01)
                        increased = True
                
                if not increased:
                    if verbose:
                        print("  All ranks at maximum. Stopping rank increases.")
                    break
                
                # Optimize with new ranks - more iterations for stability
                self._optimize_model(train_coords, normalized_train_values, max_iter*2, tol/10, verbose)
                
                # Evaluate new model
                train_pred = self.predict(train_coords, normalize=True)
                train_rmse = np.sqrt(np.mean((train_pred - normalized_train_values) ** 2))
                
                val_pred = self.predict(val_coords, normalize=True)
                val_rmse = np.sqrt(np.mean((val_pred - normalized_val_values) ** 2))
                
                # Calculate unnormalized RMSE for display
                train_pred_unnorm = self.predict(train_coords, normalize=False)
                train_rmse_unnorm = np.sqrt(np.mean((train_pred_unnorm - train_values) ** 2))
                
                val_pred_unnorm = self.predict(val_coords, normalize=False)
                val_rmse_unnorm = np.sqrt(np.mean((val_pred_unnorm - val_values) ** 2))
                
                # Calculate progress on normalized validation set
                progress = (prev_val_rmse - val_rmse) / prev_val_rmse
                
                if verbose:
                    print(f"  New Train RMSE: {train_rmse_unnorm:.6f}, Val RMSE: {val_rmse_unnorm:.6f}")
                    print(f"  Relative progress: {progress:.2%}")
                
                # Accept increase only if sufficient progress made
                if progress > rank_increase_tol:
                    if verbose:
                        print(f"  Rank increase accepted!")
                    
                    # Record new best validation RMSE
                    prev_val_rmse = val_rmse
                    best_val_rmse = val_rmse
                    consecutive_failures = 0
                    
                    # Record history
                    curr_time = time.time() - start_time
                    history['train_rmse'].append(train_rmse_unnorm)
                    history['val_rmse'].append(val_rmse_unnorm)
                    history['time'].append(curr_time)
                    history['ranks'].append(self.tt_ranks.copy())
                else:
                    # Revert to previous model if not enough improvement
                    if verbose:
                        print(f"  Rank increase rejected. Reverting.")
                    
                    self.tt_cores = old_cores
                    self.tt_ranks = old_ranks
                    
                    # Track consecutive failures
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        if verbose:
                            print(f"  {consecutive_failures} consecutive failures. Stopping rank increases.")
                        break
        
        except (KeyboardInterrupt, TimeoutError):
            print("\nTraining interrupted! Returning current model.")
        
        # Final optimization with more iterations 
        if verbose:
            print("\nPerforming final optimization...")
        
        self._optimize_model(train_coords, normalized_train_values, max_iter*3, tol/100, verbose)
        
        # Final evaluation
        train_pred = self.predict(train_coords)
        train_rmse = np.sqrt(np.mean((train_pred - train_values) ** 2))
        
        val_pred = self.predict(val_coords)
        val_rmse = np.sqrt(np.mean((val_pred - val_values) ** 2))
        
        curr_time = time.time() - start_time
        
        # Record final history
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['time'].append(curr_time)
        history['ranks'].append(self.tt_ranks.copy())
        
        if verbose:
            final_ranks_str = "-".join(str(r) for r in self.tt_ranks[1:-1])
            print(f"Final ranks: [{final_ranks_str}]")
            print(f"Final Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, Time: {curr_time:.2f}s")
        
        return history
    
    def _optimize_model(self, coords, values, max_iter=20, tol=1e-4, verbose=False):
        """Optimize the model with fixed ranks using ALS."""
        prev_rmse = float('inf')
        
        for iteration in range(max_iter):
            # Optimize each core separately
            for core_idx in range(self.ndim):
                if verbose and iteration % 5 == 0:
                    print(f"Iteration {iteration+1}/{max_iter}, Core {core_idx+1}/{self.ndim}", end="\r")
                
                # Optimize this core
                self._optimize_core_als(core_idx, coords, values)
            
            # Check convergence every few iterations
            if iteration % 5 == 0 or iteration == max_iter - 1:
                pred = self.predict(coords, normalize=True)
                rmse = np.sqrt(np.mean((pred - values) ** 2))
                
                rel_improvement = (prev_rmse - rmse) / prev_rmse if prev_rmse > 0 else 1.0
                if rel_improvement < tol and iteration > 5:
                    if verbose:
                        print(f"\nConverged at iteration {iteration+1}/{max_iter}, RMSE: {rmse:.6f}")
                    break
                    
                prev_rmse = rmse
    
    def _increase_bond_rank(self, bond_idx, new_rank, noise_scale=0.01):
        """
        Increase the rank at a specific bond between cores with better initialization.
        
        Args:
            bond_idx: Index of the bond (between cores)
            new_rank: New rank for this bond
            noise_scale: Scale of random noise for new components
        """
        # Get current rank
        current_rank = self.tt_ranks[bond_idx]
        if new_rank <= current_rank:
            return
        
        # Get the cores on either side of the bond
        left_core = self.tt_cores[bond_idx - 1].copy()  # Shape: (r_{bond_idx-1}, n_{bond_idx-1}, r_{bond_idx})
        right_core = self.tt_cores[bond_idx].copy()     # Shape: (r_{bond_idx}, n_{bond_idx}, r_{bond_idx+1})
        
        # Create new cores with increased rank
        left_shape = left_core.shape
        right_shape = right_core.shape
        
        new_left_core = np.zeros((left_shape[0], left_shape[1], new_rank))
        new_right_core = np.zeros((new_rank, right_shape[1], right_shape[2]))
        
        # Copy existing data
        new_left_core[:, :, :current_rank] = left_core
        new_right_core[:current_rank, :, :] = right_core
        
        # For new rank components, use SVD-based initialization
        if new_rank > current_rank:
            # Calculate scale based on existing values
            left_scale = np.std(left_core) * noise_scale if np.std(left_core) > 0 else noise_scale
            right_scale = np.std(right_core) * noise_scale if np.std(right_core) > 0 else noise_scale
            
            # Initialize with small random values scaled by existing data
            new_left_core[:, :, current_rank:] = np.random.randn(
                left_shape[0], left_shape[1], new_rank - current_rank) * left_scale
            
            new_right_core[current_rank:, :, :] = np.random.randn(
                new_rank - current_rank, right_shape[1], right_shape[2]) * right_scale
            
            # Orthogonalize new components for numerical stability
            for i in range(current_rank, new_rank):
                # Left core slice
                slice_matrix = new_left_core[:, :, i].reshape(-1, 1)
                if i > current_rank:
                    # Orthogonalize against previous new components
                    for j in range(current_rank, i):
                        prev_slice = new_left_core[:, :, j].reshape(-1, 1)
                        projection = (slice_matrix.T @ prev_slice) * prev_slice
                        slice_matrix = slice_matrix - projection
                
                # Normalize
                norm = np.linalg.norm(slice_matrix)
                if norm > 1e-10:
                    new_left_core[:, :, i] = (slice_matrix / norm).reshape(left_shape[0], left_shape[1])
            
            # Repeat for right core
            for i in range(current_rank, new_rank):
                # Right core slice
                slice_matrix = new_right_core[i, :, :].reshape(-1, 1)
                if i > current_rank:
                    # Orthogonalize against previous new components
                    for j in range(current_rank, i):
                        prev_slice = new_right_core[j, :, :].reshape(-1, 1)
                        projection = (slice_matrix.T @ prev_slice) * prev_slice
                        slice_matrix = slice_matrix - projection
                
                # Normalize
                norm = np.linalg.norm(slice_matrix)
                if norm > 1e-10:
                    new_right_core[i, :, :] = (slice_matrix / norm).reshape(right_shape[1], right_shape[2])
        
        # Update the cores and the ranks
        self.tt_cores[bond_idx - 1] = new_left_core
        self.tt_cores[bond_idx] = new_right_core
        self.tt_ranks[bond_idx] = new_rank
    
    def _optimize_core_als(self, core_idx, coords, normalized_values):
        """
        Optimize a single TT-core using alternating least squares.
        
        Args:
            core_idx: Index of the core to optimize
            coords: Coordinates of samples
            normalized_values: Normalized tensor values
        """
        n_samples = len(normalized_values)
        
        # Current core dimensions
        r_left = self.tt_ranks[core_idx]
        n_k = self.shape[core_idx]
        r_right = self.tt_ranks[core_idx + 1]
        
        # Compute factor matrices (left and right interfaces)
        left_interfaces = self._compute_left_interfaces(coords, core_idx)
        right_interfaces = self._compute_right_interfaces(coords, core_idx)
        
        # Initialize the new core
        new_core = np.zeros((r_left, n_k, r_right))
        
        # Process each slice of the core separately using mode-specific samples
        for mode_idx in range(n_k):
            # Find samples with this mode index
            mode_mask = (coords[:, core_idx] == mode_idx)
            
            if np.any(mode_mask):
                # Get interfaces for this mode
                mode_left = left_interfaces[mode_mask]
                mode_right = right_interfaces[mode_mask]
                mode_values = normalized_values[mode_mask]
                
                # Build design matrix for this mode
                mode_A = np.zeros((np.sum(mode_mask), r_left * r_right))
                for s in range(len(mode_values)):
                    mode_A[s, :] = np.kron(mode_left[s], mode_right[s])
                
                # Solve the least squares problem for this mode
                try:
                    # Try direct solve first
                    if mode_A.shape[0] >= mode_A.shape[1]:  # Overdetermined
                        mode_solution, _, _, _ = np.linalg.lstsq(mode_A, mode_values, rcond=None)
                    else:  # Underdetermined - use regularization
                        # A^T A + λI for stability
                        reg_factor = 1e-8 * np.eye(mode_A.shape[1])
                        ATA = mode_A.T @ mode_A + reg_factor
                        ATb = mode_A.T @ mode_values
                        mode_solution = np.linalg.solve(ATA, ATb)
                        
                    # Reshape the solution to the core slice format
                    new_core[:, mode_idx, :] = mode_solution.reshape(r_left, r_right)
                except np.linalg.LinAlgError:
                    # Fallback to pseudoinverse if direct solve fails
                    mode_solution = np.linalg.pinv(mode_A) @ mode_values
                    new_core[:, mode_idx, :] = mode_solution.reshape(r_left, r_right)
        
        # Update the core
        self.tt_cores[core_idx] = new_core
        
        # Orthogonalize the core for numerical stability
        self._orthogonalize_core(core_idx)
    
    def _compute_left_interfaces(self, coords, core_idx):
        """Compute left interfaces for all samples up to core_idx."""
        n_samples = len(coords)
        r_left = self.tt_ranks[core_idx]
        
        if core_idx == 0:
            # Trivial case: left interface is just ones
            return np.ones((n_samples, 1))
        
        # Initialize with ones - each row will hold an interface
        left_interfaces = np.zeros((n_samples, r_left))
        
        # Calculate for each sample
        for s in range(n_samples):
            # Start with 1x1 identity
            curr_interface = np.ones((1, 1))
            
            # Multiply through the cores from left to right
            for i in range(core_idx):
                idx = coords[s, i]
                curr_matrix = self.tt_cores[i][:, idx, :]
                curr_interface = curr_interface @ curr_matrix
            
            # Store the result
            left_interfaces[s] = curr_interface.flatten()
        
        return left_interfaces
    
    def _compute_right_interfaces(self, coords, core_idx):
        """Compute right interfaces for all samples from right to current core."""
        n_samples = len(coords)
        r_right = self.tt_ranks[core_idx + 1]
        
        if core_idx == self.ndim - 1:
            # Trivial case: right interface is just ones
            return np.ones((n_samples, 1))
        
        # Initialize interfaces
        right_interfaces = np.zeros((n_samples, r_right))
        
        # Calculate for each sample
        for s in range(n_samples):
            # Start with 1x1 identity
            curr_interface = np.ones((1, 1))
            
            # Multiply through the cores from right to left
            for i in range(self.ndim - 1, core_idx, -1):
                idx = coords[s, i]
                curr_matrix = self.tt_cores[i][:, idx, :]
                curr_interface = curr_matrix @ curr_interface
            
            # Store the result
            right_interfaces[s] = curr_interface.flatten()
        
        return right_interfaces
    
    def _orthogonalize_core(self, core_idx):
        """Orthogonalize a core to improve numerical stability."""
        if core_idx < self.ndim - 1:  # Can't right-orthogonalize the last core
            core = self.tt_cores[core_idx]
            r_left, n_k, r_right = core.shape
            
            # Reshape and compute QR decomposition
            core_mat = core.reshape(r_left * n_k, r_right)
            Q, R = np.linalg.qr(core_mat)
            
            # Update the current core
            self.tt_cores[core_idx] = Q.reshape(r_left, n_k, Q.shape[1])
            
            # Pass R to the next core
            next_core = self.tt_cores[core_idx + 1]
            self.tt_cores[core_idx + 1] = np.tensordot(R, next_core, axes=(1, 0))
    
    def predict(self, coords, normalize=False):
        """
        Predict values at given coordinates using the TT model.
        
        Args:
            coords: Coordinates to predict at, shape (n_samples, ndim)
            normalize: Whether to return normalized predictions (training mode)
                      or denormalized predictions (inference mode)
            
        Returns:
            Predicted values, shape (n_samples,)
        """
        n_samples = coords.shape[0]
        predictions = np.zeros(n_samples)
        
        # For each sample
        for s in range(n_samples):
            # Start with a 1x1 matrix
            result = np.ones((1, 1))
            
            # Multiply through the TT train
            for d in range(self.ndim):
                idx = coords[s, d]
                # Extract the matrix corresponding to this index
                curr_matrix = self.tt_cores[d][:, idx, :]
                # Multiply with the running result
                result = result @ curr_matrix
            
            # Result should be a 1x1 matrix
            predictions[s] = result[0, 0]
        
        # Denormalize if requested
        if not normalize and hasattr(self, 'values_mean'):
            predictions = predictions * self.values_std + self.values_mean
            
        return predictions

    def full_tensor(self):
        """
        Convert the TT representation to a full tensor.
        WARNING: Only use for small tensors, as this will be memory intensive.
        
        Returns:
            Full tensor representation
        """
        # Start with the first core
        result = self.tt_cores[0].copy()
        
        # Iteratively contract with remaining cores
        for i in range(1, self.ndim):
            # Reshape result for contraction
            curr_shape = result.shape
            result = result.reshape(-1, curr_shape[-1])
            
            # Reshape core for contraction
            core = self.tt_cores[i]
            core_shape = core.shape
            core = core.reshape(core_shape[0], -1)
            
            # Contract
            result = result @ core
            
            # Reshape result
            new_shape = curr_shape[:-1] + (core_shape[1], core_shape[2])
            result = result.reshape(new_shape)
        
        # Reshape to final tensor
        final_shape = tuple(self.shape)
        return result.reshape(final_shape)

    def tt_mode_multiply(self, matrix, mode):
        """
        Perform mode-m multiplication of the TT representation with a matrix.
        This modifies the TT-cores in place.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Matrix to multiply with
        mode : int
            Mode along which to multiply
        """

        core = self.tt_cores[mode]
        r_left, n_k, r_right = core.shape
        
        if matrix.shape[1] != n_k:
            raise ValueError(f"Matrix second dimension ({matrix.shape[1]}) must match core dimension ({n_k})")
        
        # Reshape the core for multiplication - the key is to move dimension 1 to the front
        core_reshaped = np.transpose(core, (1, 0, 2))  # Now shape (n_k, r_left, r_right)
        core_reshaped = core_reshaped.reshape(n_k, -1)  # Now shape (n_k, r_left * r_right)
        
        # Multiply with matrix: (m, n_k) @ (n_k, r_left * r_right) -> (m, r_left * r_right)
        result = matrix @ core_reshaped
        
        # Reshape back to core format
        m = matrix.shape[0]
        result = result.reshape(m, r_left, r_right)  # Shape (m, r_left, r_right)
        
        # Move the first dimension back to the middle
        new_core = np.transpose(result, (1, 0, 2))  # Back to shape (r_left, m, r_right)
        
        # Update the core
        self.tt_cores[mode] = new_core
        
        return self
    
    def copy(self,deep=True):
        return deepcopy(self)

class TTChebyshevInterpolation:
    """
    Implementation of Chebyshev interpolation using TT-format for efficient storage
    and computation, based on Algorithm 4 from the paper "Low-rank tensor approximation 
    for Chebyshev interpolation in parametric option pricing" by Glau, Kressner, and Statti.
    """
    
    def __init__(self, domains, degrees, initial_tt_ranks=None, max_rank=10):
        """
        Initialize the TT-Chebyshev interpolation.
        
        Parameters:
        -----------
        domains : list of tuples
            List of (min, max) tuples defining the domain in each dimension.
        degrees : list of int
            List of degrees for Chebyshev polynomials in each dimension.
        initial_tt_ranks : list of int or None
            Initial TT-ranks for tensor completion. If None, use rank-2.
        max_rank : int
            Maximum TT-rank for tensor completion.
        """
        self.domains = domains
        self.degrees = degrees
        self.dims = len(domains)
        
        # TT representation for P and C tensors
        self.P_tt = None
        self.C_tt = None
        
        # Initial TT-ranks and max rank
        self.initial_tt_ranks = initial_tt_ranks if initial_tt_ranks else [2] * (self.dims - 1)
        self.max_rank = max_rank
        
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
        
        # Shape of the full tensor (for TT representation)
        self.tensor_shape = tuple(d + 1 for d in self.degrees)
        
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
        if len(self._poly_cache) < 10000:  # Limit cache size
            self._poly_cache[key] = result
            
        return result
    
    def construct_tensor_P(self, ref_method, subset_size=None):
        """
        Construct tensor P containing prices at Chebyshev grid points.
        This implements Step 2 of Algorithm 4 (offline phase).
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_size : int or None
            Number of points to evaluate. If None, all points are evaluated.
            
        Returns:
        --------
        TTCompletion
            TT representation of tensor P.
        """
        print("Offline Phase - Step 2: Computing reference prices at Chebyshev nodes")
        start_time = time.time()
        
        # Generate all possible indices for the tensor
        all_indices = list(product(*[range(d + 1) for d in self.degrees]))
        total_points = len(all_indices)
        
        # If subset_size is specified, randomly sample points
        if subset_size is not None and subset_size < total_points:
            # Randomly sample indices
            np.random.seed(42)  # For reproducibility
            sample_idx = np.random.choice(total_points, size=subset_size, replace=False)
            subset_indices = [all_indices[i] for i in sample_idx]
        else:
            subset_indices = all_indices
            
        # Evaluate at selected points
        sample_count = len(subset_indices)
        print(f"Computing {sample_count} prices ({sample_count/total_points*100:.1f}% of full tensor)...")
        
        # Create arrays for TT completion
        coords = np.array(subset_indices)
        values = np.zeros(sample_count)
        
        # Evaluate reference method at each point
        for i, indices in enumerate(subset_indices):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (sample_count - i)
                print(f"  Progress: {i}/{sample_count} ({i/sample_count*100:.1f}%)")
                print(f"  Estimated time remaining: {remaining:.1f} seconds")
            
            # Construct parameter point
            param_point = tuple(self.points[dim][idx] for dim, idx in enumerate(indices))
            
            # Evaluate reference pricing method
            values[i] = ref_method(param_point)
        
        print("Performing tensor completion...")
        
        # Create TT model
        self.P_tt = TTCompletion(
            shape=self.tensor_shape,
            initial_tt_ranks=self.initial_tt_ranks,
            max_rank=self.max_rank
        )
        
        # Fit TT model to the data
        self.P_tt.fit(
            coords=coords,
            values=values,
            max_iter=20,
            tol=1e-4,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        return self.P_tt
    
    def construct_tensor_C(self):
        """
        Construct tensor C containing Chebyshev coefficients using TT format.
        This implements Step 4 of Algorithm 4 (offline phase) with TT optimization.
        
        Returns:
        --------
        TTCompletion
            TT representation of tensor C.
        """
        print("Offline Phase - Step 4: Computing Chebyshev coefficients")
        start_time = time.time()
        
        if self.P_tt is None:
            raise ValueError("Tensor P not constructed. Call construct_tensor_P first.")
        
        # Initialize C_tt as a copy of P_tt
        self.C_tt = self.P_tt.copy()
        
        
        # Apply Chebyshev transformation to each core
        for mode in range(self.dims):
            # Construct the basis matrix for this dimension
            F_n = self._construct_basis_matrix(self.degrees[mode])
            
            # Perform mode-m multiplication with the TT representation
            self.C_tt.tt_mode_multiply(F_n, mode)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
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
                if k==0 or k==n:
                    factor *= 0.5
                # Compute cosine term
                angle = j * np.pi * k / n
                F_n[j, k] = factor * np.cos(angle)
        
        return F_n
    
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
        Evaluate the Chebyshev interpolation at a point using TT representation.
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
        
        # Step 9: Compute interpolated price using TT inner product
        interpolated_price = self._tt_inner_product(T_vectors)
        if hasattr(self.C_tt, "values_mean") and hasattr(self.C_tt, "values_std"):
            interpolated_price = interpolated_price * self.C_tt.values_std + self.C_tt.values_mean
        else:            # just in case the tensor was built without normalising
            interpolated_price = interpolated_price
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed:.4f} seconds")
        print(interpolated_price)
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
        if self.C_tt is None:
            raise ValueError("Tensor C not constructed. Run offline phase first.")
        
        # Handle single point case
        if isinstance(points, tuple):
            return self.evaluate_interpolation(points)
        
        # Process multiple points
        results = np.zeros(len(points))
        
        print(f"Evaluating at {len(points)} points...")
        start_time = time.time()
        
        for i, point in enumerate(points):
            results[i] = self.evaluate_interpolation(point)
            
            # Progress reporting
            if (i+1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i+1)
                remaining = avg_time * (len(points) - (i+1))
                print(f"  Processed {i+1}/{len(points)} points. Estimated time remaining: {remaining:.1f} seconds")
        
        total_time = time.time() - start_time
        print(f"Batch evaluation completed in {total_time:.2f} seconds")
        
        return results
    
    def _tt_inner_product(self, vectors):
        """
        Compute the inner product between TT tensor and a set of vectors.
        More efficient than reconstructing the full tensor.
        
        Parameters:
        -----------
        vectors : list of numpy.ndarray
            List of vectors (Chebyshev basis vectors T_j)
            
        Returns:
        --------
        float
            Inner product value
        """
        # Start with the leftmost core
        result = np.ones((1,1))
        # Process remaining cores
        for i,vector in enumerate(vectors):
            # Get the next core
            core = self.C_tt.tt_cores[i]  # Shape: (r_{k-1}, n_k, r_k)
            
            # Contract previous result with the left index of the core
            tmp = np.tensordot(core, vector, axes=([1], [0]))  # Shape: (1, n_k, r_k)
            
            # Contract with the k-th vector
            result = result @ tmp
        
        # Final result should be a scalar
        return result[0, 0]
    
    def run_offline_phase(self, ref_method, subset_size=None):
        """
        Run the complete offline phase of Algorithm 4 with TT optimization.
        
        Parameters:
        -----------
        ref_method : callable
            Reference pricing method that takes a point in parameter space 
            and returns a price.
        subset_size : int or None
            Size of the subset of Chebyshev nodes to evaluate. If None,
            evaluate at all nodes.
            
        Returns:
        --------
        TTCompletion
            TT representation of tensor C with Chebyshev coefficients.
        """
        print("=== OFFLINE PHASE ===")
        overall_start = time.time()
        
        # Step 2-3: Construct tensor P in TT format (with tensor completion if subset_size < total nodes)
        self.P_tt = self.construct_tensor_P(ref_method, subset_size)
        
        # Step 4: Construct tensor C in TT format
        self.C_tt = self.construct_tensor_C()
        
        overall_elapsed = time.time() - overall_start
        print(f"Offline phase completed in {overall_elapsed:.2f} seconds")
        
        return self.C_tt
    
    def run_online_phase(self, points):
        """
        Run the online phase of Algorithm 4 for one or more points using TT representation.
        
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
            return self.evaluate_batch(points)
    
    def run_algorithm4(self, ref_method, evaluation_points, subset_size=None):
        """
        Run the complete Algorithm 4 (offline + online phases) with TT optimization.
        
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
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # Define a simple 2D reference method (European call option price in BS model)
    def bs_call_price(S, K, T, r, sigma):
        """Black-Scholes call option price."""
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
    
    # Create interpolation with TT format
    domains = [(80.0, 120.0), (0.1, 0.4)]  # S0, sigma
    degrees = [200, 200]  # 10th degree in each dimension
    
    # Initialize TT-Chebyshev interpolation
    tt_cheb = TTChebyshevInterpolation(
        domains, 
        degrees,
        initial_tt_ranks=[1],  # Start with rank-3
        max_rank=17               # Allow up to rank-10
    )
    
    # Run offline phase with 20% of the tensor points
    total_points = np.prod([d+1 for d in degrees])
    subset_size = int(0.2 * total_points)  # Use 20% of total points
    
    tt_cheb.run_offline_phase(reference_method, subset_size=subset_size)
    
    # Online phase - evaluate at a grid of test points
    S_grid = np.linspace(80, 120, 21)
    sigma_grid = np.linspace(0.1, 0.4, 21)
    
    # Create a 2D grid of test points
    test_points = []
    for S in S_grid:
        for sigma in sigma_grid:
            test_points.append((S, sigma))
    
    # Evaluate on test points
    test_values = tt_cheb.run_online_phase(test_points)
    
    # Reshape results to a grid for plotting
    interp_grid = test_values.reshape(len(S_grid), len(sigma_grid))
    
    # Compute exact values for comparison
    exact_grid = np.zeros((len(S_grid), len(sigma_grid)))
    rel_error_grid = np.zeros((len(S_grid), len(sigma_grid)))
    
    for i, S in enumerate(S_grid):
        for j, sigma in enumerate(sigma_grid):
            exact_grid[i, j] = reference_method((S, sigma))
            rel_error_grid[i, j] = abs(interp_grid[i, j] - exact_grid[i, j]) / exact_grid[i, j]
    
    # Calculate max and average errors
    max_rel_error = np.max(rel_error_grid)
    avg_rel_error = np.mean(rel_error_grid)
    
    print(f"\nInterpolation Errors:")
    print(f"Max relative error: {max_rel_error:.6e}")
    print(f"Avg relative error: {avg_rel_error:.6e}")
    print(f"TT ranks of coefficient tensor: {tt_cheb.C_tt.tt_ranks}")
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Interpolated values
    ax1 = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(S_grid, sigma_grid)
    surf1 = ax1.plot_surface(X, Y, interp_grid.T, cmap='viridis')
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Volatility (σ)')
    ax1.set_zlabel('Option Price')
    ax1.set_title('TT-Chebyshev Interpolation')
    
    # Plot 2: Exact values
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, exact_grid.T, cmap='viridis')
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Volatility (σ)')
    ax2.set_zlabel('Option Price')
    ax2.set_title('Exact Black-Scholes Price')
    
    # Plot 3: Relative error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, rel_error_grid.T, cmap='hot')
    ax3.set_xlabel('Stock Price (S)')
    ax3.set_ylabel('Volatility (σ)')
    ax3.set_zlabel('Relative Error')
    ax3.set_title(f'Relative Error (Max: {max_rel_error:.2e})')
    
    fig.tight_layout()
    plt.show()
