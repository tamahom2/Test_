import numpy as np
import time
from typing import List, Tuple, Dict, Optional


def create_norm_function_tensor(grid_size: int = 20):
    """
    Create a tensor representing the Euclidean norm function f(x) = ||x|| 
    where x ∈ [0,1]^4.
    
    Args:
        grid_size: Number of grid points in each dimension
        
    Returns:
        Coordinates and values of the full tensor
    """
    # Create grid points in [0,1] for each dimension
    grid = np.linspace(0, 1, grid_size)
    
    # Create all coordinates
    coords = np.array(list(np.ndindex(grid_size, grid_size, grid_size, grid_size)))
    
    # Map grid indices to actual coordinate values in [0,1]^4
    points = np.zeros_like(coords, dtype=float)
    for i in range(4):  # 4 dimensions
        points[:, i] = grid[coords[:, i]]
    
    # Compute Euclidean norm ||x|| for each point
    values = np.sqrt(np.sum(points**2, axis=1))
    
    return coords, values, points


def test_tensor_completion_norm_function(tt_model, test_coords, test_values, test_points):
    """
    Test tensor completion on the norm function and compute relative errors.
    
    Args:
        tt_model: Trained tensor completion model
        test_coords: Grid coordinates of test points
        test_values: True function values at test points
        test_points: Actual [0,1]^4 coordinates of test points
        
    Returns:
        Dictionary with various error metrics
    """
    # Get predictions from the model
    test_pred = tt_model.predict(test_coords)
    
    # Compute absolute errors
    abs_error = np.abs(test_pred - test_values)
    
    # Compute relative errors, avoiding division by zero
    non_zero = test_values > 1e-10
    rel_error_safe = np.zeros_like(test_values)
    rel_error_safe[non_zero] = abs_error[non_zero] / test_values[non_zero]
    
    # Compute overall metrics
    results = {
        'rmse': np.sqrt(np.mean(abs_error**2)),
        'mae': np.mean(abs_error),
        'max_abs_error': np.max(abs_error),
        'mean_rel_error': np.mean(rel_error_safe),
        'median_rel_error': np.median(rel_error_safe),
        'max_rel_error': np.max(rel_error_safe)
    }
    
    # For visualization, create mapping from function input to error
    results['points'] = test_points
    results['values'] = test_values
    results['predictions'] = test_pred
    results['abs_errors'] = abs_error
    results['rel_errors'] = rel_error_safe
    
    return results


class TTCompletion:
    """Tensor completion using Tensor Train decomposition with sequential rank increases."""
    
    def __init__(self, shape: Tuple[int, ...], initial_tt_ranks: List[int] = None, 
                 random_seed: Optional[int] = None, max_rank: int = 30):
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
        
    def _initialize_cores(self) -> List[np.ndarray]:
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
    
    def fit(self, coords: np.ndarray, values: np.ndarray, 
            max_iter: int = 20, tol: float = 1e-4,
            max_time: int = 600, verbose: bool = True,
            validation_split: float = 0.2,
            rank_increase_tol: float = 0.01,
            max_rank_iterations: int = 10) -> Dict:
        """
        Fit the TT model with sequential rank increases.
        
        Args:
            coords: Coordinates of observed entries, shape (n_samples, ndim)
            values: Values of observed entries, shape (n_samples,)
            max_iter: Maximum number of iterations for each rank configuration
            tol: Convergence tolerance for relative improvement
            max_time: Maximum training time in seconds
            verbose: Whether to print progress information
            validation_split: Fraction of data to use for validation
            rank_increase_tol: Tolerance for rank increase (progress threshold)
            max_rank_iterations: Maximum number of rank increase cycles
            
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
        
        # Get unnormalized RMSE
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
        
        # Best validation RMSE so far
        best_val_rmse = val_rmse
        
        # Track locked bonds (where increasing rank doesn't help)
        locked_bonds = [True] + [False] * (self.ndim - 1) + [True]  # Boundary ranks always locked
        
        # Sequential rank increase strategy
        try:
            rank_iteration = 0
            
            while rank_iteration < max_rank_iterations:
                rank_iteration += 1
                
                if verbose:
                    print(f"\nRank increase cycle {rank_iteration}...")
                
                # Check if we've exceeded time limit
                if time.time() - start_time > max_time:
                    print(f"\nStopping early: reached {max_time} seconds time limit")
                    break
                
                # Track if any bond was improved in this cycle
                any_improvement = False
                
                # Try each bond sequentially
                for bond_idx in range(1, self.ndim):
                    # Skip locked bonds
                    if locked_bonds[bond_idx]:
                        if verbose:
                            print(f"  Bond {bond_idx} is locked. Skipping.")
                        continue
                    
                    # Skip bonds at max rank
                    if self.tt_ranks[bond_idx] >= self.max_rank:
                        if verbose:
                            print(f"  Bond {bond_idx} already at max rank {self.max_rank}. Locking.")
                        locked_bonds[bond_idx] = True
                        continue
                    
                    # Check time limit again
                    if time.time() - start_time > max_time:
                        print(f"\nStopping early: reached {max_time} seconds time limit")
                        break
                    
                    # Try increasing this bond's rank
                    old_rank = self.tt_ranks[bond_idx]
                    new_rank = old_rank + 1
                    
                    if verbose:
                        print(f"  Trying to increase rank at bond {bond_idx} from {old_rank} to {new_rank}...")
                    
                    # Save current cores and ranks
                    old_cores = [core.copy() for core in self.tt_cores]
                    old_ranks = self.tt_ranks.copy()
                    
                    # Increase the rank
                    self._increase_bond_rank(bond_idx, new_rank, noise_scale=0.01)
                    
                    # Optimize with new rank
                    self._optimize_model(train_coords, normalized_train_values, max_iter, tol/10, False)
                    
                    # Evaluate new model
                    val_pred = self.predict(val_coords, normalize=True)
                    new_val_rmse = np.sqrt(np.mean((val_pred - normalized_val_values) ** 2))
                    
                    # Get unnormalized RMSE for display
                    val_pred_unnorm = self.predict(val_coords, normalize=False)
                    val_rmse_unnorm = np.sqrt(np.mean((val_pred_unnorm - val_values) ** 2))
                    
                    # Calculate improvement
                    rel_improvement = (best_val_rmse - new_val_rmse) / best_val_rmse
                    
                    if verbose:
                        print(f"    New Val RMSE: {val_rmse_unnorm:.6f}, Rel. improvement: {rel_improvement:.2%}")
                    
                    # Accept if improved enough
                    if rel_improvement > rank_increase_tol:
                        if verbose:
                            print(f"    Rank increase accepted!")
                        
                        # Update best validation RMSE
                        best_val_rmse = new_val_rmse
                        
                        # Get train RMSE
                        train_pred = self.predict(train_coords, normalize=False)
                        train_rmse = np.sqrt(np.mean((train_pred - train_values) ** 2))
                        
                        # Record in history
                        curr_time = time.time() - start_time
                        history['train_rmse'].append(train_rmse)
                        history['val_rmse'].append(val_rmse_unnorm)
                        history['time'].append(curr_time)
                        history['ranks'].append(self.tt_ranks.copy())
                        
                        any_improvement = True
                    else:
                        # Revert to previous model
                        if verbose:
                            print(f"    Rank increase rejected. Reverting.")
                        
                        self.tt_cores = old_cores
                        self.tt_ranks = old_ranks
                        
                        # Lock this bond if the improvement is minimal
                        if rel_improvement < tol / 10:
                            if verbose:
                                print(f"    Negligible improvement. Locking bond {bond_idx}.")
                            locked_bonds[bond_idx] = True
                
                # If no improvement in full cycle, stop
                if not any_improvement:
                    if verbose:
                        print(f"No further improvements in rank increase cycle {rank_iteration}. Stopping.")
                    break
                
                # If all bonds are locked, stop
                if all(locked_bonds[1:-1]):
                    if verbose:
                        print("All bonds are locked. Stopping rank increases.")
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
    
    def _optimize_model(self, coords: np.ndarray, values: np.ndarray, 
                        max_iter: int = 20, tol: float = 1e-4, verbose: bool = False):
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
            if iteration % 2 == 0 or iteration == max_iter - 1:
                pred = self.predict(coords, normalize=True)
                rmse = np.sqrt(np.mean((pred - values) ** 2))
                
                rel_improvement = (prev_rmse - rmse) / prev_rmse if prev_rmse > 0 else 1.0
                if rel_improvement < tol and iteration > 5:
                    if verbose:
                        print(f"\nConverged at iteration {iteration+1}/{max_iter}, RMSE: {rmse:.6f}")
                    break
                    
                prev_rmse = rmse
    
    def _increase_bond_rank(self, bond_idx: int, new_rank: int, noise_scale: float = 0.01):
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
    
    def _optimize_core_als(self, core_idx: int, coords: np.ndarray, normalized_values: np.ndarray):
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
    
    def _compute_left_interfaces(self, coords: np.ndarray, core_idx: int) -> np.ndarray:
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
    
    def _compute_right_interfaces(self, coords: np.ndarray, core_idx: int) -> np.ndarray:
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
    
    def _orthogonalize_core(self, core_idx: int):
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
    
    def predict(self, coords: np.ndarray, normalize: bool = False) -> np.ndarray:
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


# Main function to test on the Euclidean norm function
if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    # ===== Step 1: Create the norm function tensor =====
    grid_size = 20  # 20 points per dimension -> (20,20,20,20) tensor
    print(f"Creating norm function tensor with grid size {grid_size}...")
    
    coords, values, points = create_norm_function_tensor(grid_size)
    print(f"Tensor shape: {(grid_size, grid_size, grid_size, grid_size)}")
    print(f"Total entries: {len(coords)}")
    
    # ===== Step 2: Split data into train, validation, and test sets =====
    # Use 5% for training, 0.5% for validation, and hold out 100 points for testing
    n_samples = len(coords)
    train_ratio = 0.05
    val_ratio = 0.01
    
    # Calculate how many samples to use
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = 100
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    
    # Select separate indices for train, val, and test
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[-n_test:]  # Use last 100 points for test
    
    # Create the datasets
    train_coords = coords[train_indices]
    train_values = values[train_indices]
    train_points = points[train_indices]
    
    val_coords = coords[val_indices]
    val_values = values[val_indices]
    val_points = points[val_indices]
    
    test_coords = coords[test_indices]
    test_values = values[test_indices]
    test_points = points[test_indices]
    
    print(f"Training with {len(train_coords)} entries ({train_ratio*100:.1f}% of tensor)")
    print(f"Validation with {len(val_coords)} entries ({val_ratio*100:.1f}% of tensor)")
    print(f"Testing on {len(test_coords)} separate points")
    
    # ===== Step 3: Train tensor completion model =====
    # For the Euclidean norm function, we expect low TT-ranks
    tt_model = TTCompletion(
        shape=(grid_size, grid_size, grid_size, grid_size), 
        initial_tt_ranks=[2, 2, 2],  # Start with low ranks
        random_seed=random_seed,
        max_rank=8  # Maximum rank allowed
    )
    
    # Train the model
    print("\nTraining tensor completion model...")
    history = tt_model.fit(
        np.concatenate([train_coords, val_coords]), 
        np.concatenate([train_values, val_values]), 
        max_iter=15,      # Iterations per rank configuration 
        tol=1e-4,         # Convergence tolerance
        max_time=300,     # 5 minutes maximum
        verbose=True,     # Show progress
        validation_split=len(val_coords)/(len(train_coords)+len(val_coords)),  # Proper validation split
        rank_increase_tol=0.01  # Require 1% improvement for rank increase
    )
    
    # ===== Step 4: Evaluate on test set =====
    print("\nEvaluating on test set...")
    test_results = test_tensor_completion_norm_function(tt_model, test_coords, test_values, test_points)
    
    # Print key metrics
    print("\nTest Results:")
    print(f"RMSE: {test_results['rmse']:.6f}")
    print(f"Mean Absolute Error: {test_results['mae']:.6f}")
    print(f"Max Absolute Error: {test_results['max_abs_error']:.6f}")
    print(f"Mean Relative Error: {test_results['mean_rel_error']:.2%}")
    print(f"Median Relative Error: {test_results['median_rel_error']:.2%}")
    print(f"Max Relative Error: {test_results['max_rel_error']:.2%}")
    
    # ===== Step 5: Show detailed results for a few test points =====
    print("\nDetailed results for 5 random test points:")
    sample_indices = np.random.choice(len(test_coords), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        point = test_points[idx]
        true_value = test_values[idx]
        pred_value = test_results['predictions'][idx]
        abs_error = test_results['abs_errors'][idx]
        rel_error = test_results['rel_errors'][idx]
        
        print(f"Point {i+1}: {point} → True: {true_value:.6f}, Pred: {pred_value:.6f}, "
              f"Abs Error: {abs_error:.6f}, Rel Error: {rel_error:.2%}")
    
    # Print final ranks
    print(f"\nFinal TT ranks: {tt_model.tt_ranks[1:-1]}")
