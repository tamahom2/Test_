import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import time
import matplotlib.pyplot as plt

class TTCompletion:
    """
    Implementation of tensor completion in TT format using Riemannian optimization.
    Based on the papers:
    - "Riemannian optimization for high-dimensional tensor completion" by Steinlechner
    - "Low-rank tensor approximation for Chebyshev interpolation in parametric option pricing" by Glau, Kressner, and Statti
    """
    def __init__(self, tensor_train_class):
        """
        Initialize the TTCompletion class.
        
        Parameters:
        -----------
        tensor_train_class : class
            The TensorTrain class to use for tensor operations.
        """
        self.TensorTrain = tensor_train_class
    
    def complete_tensor(self, sample_indices, sample_values, tensor_shape, 
                        initial_rank=(1,), max_rank=10, tol=1e-4, tol_stagnation=1e-4, 
                        max_iter=100, rho=0, verbose=True):
        """
        Complete a tensor from a subset of its entries using Riemannian optimization.
        
        Parameters:
        -----------
        sample_indices : list of tuples
            List of indices where the tensor values are known.
        sample_values : numpy.ndarray
            Values of the tensor at the sample indices.
        tensor_shape : tuple
            Shape of the full tensor to be completed.
        initial_rank : tuple, optional
            Initial TT ranks to use for completion.
        max_rank : int, optional
            Maximum allowed TT rank.
        tol : float, optional
            Tolerance for relative error on the training set.
        tol_stagnation : float, optional
            Tolerance for stagnation detection.
        max_iter : int, optional
            Maximum number of iterations for each rank.
        rho : float, optional
            Acceptance parameter for rank increase.
        verbose : bool, optional
            Whether to print progress information.
            
        Returns:
        --------
        TensorTrain
            Completed tensor in TT format.
        """
        # Setup training and test sets
        train_indices = sample_indices[:len(sample_indices)//2]
        train_values = sample_values[:len(sample_indices)//2]
        test_indices = sample_indices[len(sample_indices)//2:]
        test_values = sample_values[len(sample_indices)//2:]
        
        # Initialize tensor with random cores and initial rank
        X = self._initialize_random_tt(tensor_shape, initial_rank)
        
        # Adaptive rank strategy
        locked = 0
        mu = 0  # Mode index for rank increase
        d = len(tensor_shape)
        current_tt_ranks = list(initial_rank)
        
        while locked < d - 1 and max(current_tt_ranks) < max_rank:
            # Run Riemannian CG to get current completion
            X = self._riemannian_cg(X, train_indices, train_values, 
                                   test_indices, test_values, 
                                   tol=tol, tol_stagnation=tol_stagnation, 
                                   max_iter=max_iter, verbose=verbose)
            
            # Compute error on test set for current completion
            err_old = self._relative_error(X, test_indices, test_values)
            
            if verbose:
                print(f"Current rank: {current_tt_ranks}, Test error: {err_old:.6e}")
            
            # Try to increase rank at mode mu
            if mu < d - 1:  # Don't increase last rank as it's fixed to 1
                # Create a copy of X with increased rank at mode mu
                X_new = self._increase_rank(X, mu, current_tt_ranks[mu] + 1)
                current_tt_ranks[mu] += 1
                
                # Run Riemannian CG again with increased rank
                X_new = self._riemannian_cg(X_new, train_indices, train_values, 
                                           test_indices, test_values, 
                                           tol=tol, tol_stagnation=tol_stagnation, 
                                           max_iter=max_iter, verbose=verbose)
                
                # Compute error on test set for new completion
                err_new = self._relative_error(X_new, test_indices, test_values)
                
                if verbose:
                    print(f"Increased rank at mode {mu} to {current_tt_ranks[mu]}, Test error: {err_new:.6e}")
                
                # Accept or reject rank increase
                if err_new > err_old - rho * err_old:  # No significant improvement
                    current_tt_ranks[mu] -= 1  # Revert rank increase
                    locked += 1
                    if verbose:
                        print(f"Rank increase rejected. Locked: {locked}/{d-1}")
                else:
                    X = X_new  # Accept rank increase
                    locked = 0  # Reset locked counter
                    if verbose:
                        print(f"Rank increase accepted.")
            
            # Move to next mode
            mu = (mu + 1) % (d - 1)
        
        # Final run to ensure convergence
        X = self._riemannian_cg(X, train_indices, train_values, 
                               test_indices, test_values, 
                               tol=tol, tol_stagnation=tol_stagnation, 
                               max_iter=max_iter, verbose=verbose)
        
        return X
    
    def adaptive_sampling_strategy1(self, reference_method, tensor_shape, max_sample_percentage=0.1, 
                                    initial_sample_size=None, test_size=None, 
                                    max_rank=10, tol=1e-4, tol_stagnation=1e-4, 
                                    max_iter=100, rho=0, verbose=True):
        """
        Adaptive sampling strategy for tensor completion as described in Algorithm 2.
        
        Parameters:
        -----------
        reference_method : callable
            Function that computes tensor values at given indices.
        tensor_shape : tuple
            Shape of the full tensor to be completed.
        max_sample_percentage : float, optional
            Maximum percentage of tensor entries to sample.
        initial_sample_size : int, optional
            Initial number of samples to use.
        test_size : int, optional
            Number of test samples to use.
        max_rank : int, optional
            Maximum allowed TT rank.
        tol : float, optional
            Tolerance for relative error on the training set.
        tol_stagnation : float, optional
            Tolerance for stagnation detection.
        max_iter : int, optional
            Maximum number of iterations for each rank.
        rho : float, optional
            Acceptance parameter for rank increase.
        verbose : bool, optional
            Whether to print progress information.
            
        Returns:
        --------
        TensorTrain
            Completed tensor in TT format.
        """
        # Setup initial parameters
        if initial_sample_size is None:
            initial_sample_size = 100
        if test_size is None:
            test_size = 100
            
        # Maximum number of samples based on percentage
        tensor_size = np.prod(tensor_shape)
        max_samples = int(max_sample_percentage * tensor_size)
        
        # Generate initial random sampling indices
        train_indices = self._generate_random_indices(tensor_shape, initial_sample_size)
        train_values = np.array([reference_method(idx) for idx in train_indices])
        
        # Generate test set
        test_indices = self._generate_random_indices(tensor_shape, test_size, exclude=train_indices)
        test_values = np.array([reference_method(idx) for idx in test_indices])
        
        # Initial completion
        if verbose:
            print(f"Starting completion with {len(train_indices)} samples ({len(train_indices)/tensor_size*100:.4f}% of tensor)")
        
        X = self._initialize_random_tt(tensor_shape, (1,)*len(tensor_shape))
        X = self.complete_tensor(
            train_indices, train_values, tensor_shape,
            initial_rank=(1,)*len(tensor_shape), max_rank=max_rank,
            tol=tol, tol_stagnation=tol_stagnation,
            max_iter=max_iter, rho=rho, verbose=verbose
        )
        
        # Compute error on test set
        err_new = self._relative_error(X, test_indices, test_values)
        if verbose:
            print(f"Initial completion error on test set: {err_new:.6e}")
            
        # Adaptive sampling
        while len(train_indices) < max_samples:
            err_old = err_new
            
            # Create rank (1,...,1) approximation of X as starting point
            X_approx = self._rank1_approximation(X)
            
            # Add test set to training set
            old_test_indices = test_indices
            train_indices = np.append(train_indices, test_indices, axis=0)
            train_values = np.append(train_values, test_values)
            
            # Generate new test set
            test_indices = self._generate_random_indices(tensor_shape, test_size, exclude=train_indices)
            test_values = np.array([reference_method(idx) for idx in test_indices])
            
            if verbose:
                print(f"Expanded training set to {len(train_indices)} samples ({len(train_indices)/tensor_size*100:.4f}% of tensor)")
            
            # Run completion again
            X = self.complete_tensor(
                train_indices, train_values, tensor_shape,
                initial_rank=tuple([1] * len(tensor_shape)), max_rank=max_rank,
                tol=tol, tol_stagnation=tol_stagnation,
                max_iter=max_iter, rho=rho, verbose=verbose
            )
            
            # Compute error on new test set
            err_new = self._relative_error(X, test_indices, test_values)
            
            if verbose:
                print(f"New completion error on test set: {err_new:.6e}")
            
            # Check stopping criteria
            if err_new < tol:
                if verbose:
                    print(f"Target accuracy reached. Stopping.")
                break
                
            if abs(err_new - err_old) < tol_stagnation * err_old:
                if verbose:
                    print(f"Error stagnation detected. Stopping.")
                break
                
            # Check for max rank reached
            if any(r >= max_rank for r in X.ranks):
                if verbose:
                    print(f"Maximum rank reached. Stopping.")
                break
        
        return X
    
    def _riemannian_cg(self, X, train_indices, train_values, test_indices, test_values, 
                     tol=1e-4, tol_stagnation=1e-4, max_iter=100, verbose=True):
        """
        Riemannian conjugate gradient method for tensor completion.
        
        Parameters:
        -----------
        X : TensorTrain
            Initial guess for completed tensor in TT format.
        train_indices : list of tuples
            Indices of known tensor entries for training.
        train_values : numpy.ndarray
            Values of known tensor entries for training.
        test_indices : list of tuples
            Indices of known tensor entries for testing.
        test_values : numpy.ndarray
            Values of known tensor entries for testing.
        tol : float, optional
            Tolerance for relative error on the training set.
        tol_stagnation : float, optional
            Tolerance for stagnation detection.
        max_iter : int, optional
            Maximum number of iterations.
        verbose : bool, optional
            Whether to print progress information.
            
        Returns:
        --------
        TensorTrain
            Completed tensor in TT format.
        """
        # Initialize variables
        err_train_prev = float('inf')
        err_test_prev = float('inf')
        
        # Get Riemannian gradient of initial point
        grad = self._riemannian_gradient(X, train_indices, train_values)
        
        # Initial search direction is negative gradient
        eta = self._scale_tangent_vector(grad, -1.0)
        
        # Main CG loop
        for k in range(max_iter):
            # Compute step size using line search
            alpha = self._line_search(X, eta, train_indices, train_values)
            
            # Take step and retract to manifold
            X_next = self._retraction(X, eta, alpha)
            
            # Compute errors
            err_train = self._relative_error(X_next, train_indices, train_values)
            err_test = self._relative_error(X_next, test_indices, test_values)
            
            if verbose and (k % 5 == 0 or k == max_iter-1):
                print(f"  Iteration {k}: Train error = {err_train:.6e}, Test error = {err_test:.6e}")
            
            # Check convergence
            if err_train < tol:
                if verbose:
                    print(f"  Converged to target accuracy.")
                break
                
            # Check for stagnation in both training and test error
            if (abs(err_train - err_train_prev) < tol_stagnation * err_train_prev and
                abs(err_test - err_test_prev) < tol_stagnation * err_test_prev):
                if verbose:
                    print(f"  Error stagnation detected.")
                break
            
            # Compute new gradient
            grad_next = self._riemannian_gradient(X_next, train_indices, train_values)
            
            # Compute beta using Fletcher-Reeves formula
            beta = self._compute_fr_beta(grad_next, grad)
            
            # Vector transport of previous search direction
            transported_eta = self._vector_transport(X, X_next, eta)
            
            # Update search direction
            eta = self._combine_directions(grad_next, transported_eta, beta)
            
            # Update variables for next iteration
            X = X_next
            grad = grad_next
            err_train_prev = err_train
            err_test_prev = err_test
        
        return X
    
    def _riemannian_gradient(self, X, indices, values):
        """
        Compute the Riemannian gradient of the cost function at point X.
        
        The cost function is f(X) = 0.5 * ||P_Omega(X) - P_Omega(A)||^2
        where P_Omega is the projection onto the sampling set Omega.
        
        Parameters:
        -----------
        X : TensorTrain
            Current point on the manifold.
        indices : list of tuples
            Indices of known tensor entries.
        values : numpy.ndarray
            Values of known tensor entries.
            
        Returns:
        --------
        TensorTrain
            Riemannian gradient in TT format.
        """
        # Compute Euclidean gradient first
        euclidean_grad = self._euclidean_gradient(X, indices, values)
        
        # Project onto tangent space at X
        riemannian_grad = self._project_onto_tangent_space(X, euclidean_grad)
        
        return riemannian_grad
    
    def _euclidean_gradient(self, X, indices, values):
        """
        Compute the Euclidean gradient of the cost function at point X.
        
        Parameters:
        -----------
        X : TensorTrain
            Current point on the manifold.
        indices : list of tuples
            Indices of known tensor entries.
        values : numpy.ndarray
            Values of known tensor entries.
            
        Returns:
        --------
        numpy.ndarray
            Euclidean gradient as a sparse tensor.
        """
        # Evaluate X at the given indices
        X_values = np.array([X[idx] for idx in indices])
        
        # Compute residuals
        residuals = X_values - values
        
        # Create sparse gradient tensor
        grad = np.zeros(X.shape)
        for i, idx in enumerate(indices):
            grad[idx] = residuals[i]
        
        return grad
    
    def _project_onto_tangent_space(self, X, Z):
        """
        Project a tensor Z onto the tangent space at X.
        
        Parameters:
        -----------
        X : TensorTrain
            Point on the manifold.
        Z : numpy.ndarray
            Tensor to project.
            
        Returns:
        --------
        TensorTrain
            Projected tensor in TT format.
        """
        # For simplicity, we'll approximate the projection by converting Z to TT format
        # and truncating to the same rank as X
        Z_tt = self.TensorTrain.from_tensor(Z, epsilon=1e-10, max_rank=max(X.ranks))
        
        # In a complete implementation, we would compute the exact projection as per
        # the formulas in the Steinlechner paper, but that's complex to implement here
        
        return Z_tt
    
    def _line_search(self, X, eta, indices, values):
        """
        Perform a line search to find an appropriate step size.
        
        Parameters:
        -----------
        X : TensorTrain
            Current point on the manifold.
        eta : TensorTrain
            Search direction in TT format.
        indices : list of tuples
            Indices of known tensor entries.
        values : numpy.ndarray
            Values of known tensor entries.
            
        Returns:
        --------
        float
            Step size.
        """
        # Evaluate direction at indices
        eta_values = np.array([eta[idx] for idx in indices])
        
        # Evaluate current point at indices
        X_values = np.array([X[idx] for idx in indices])
        
        # Compute residuals
        residuals = X_values - values
        
        # Compute optimal step size for quadratic approximation
        numerator = np.sum(eta_values * residuals)
        denominator = np.sum(eta_values**2)
        
        if denominator < 1e-10:
            return 0.0
        
        alpha = -numerator / denominator
        
        # Ensure alpha is positive and not too large
        alpha = max(0.0, min(1.0, alpha))
        
        return alpha
    
    def _retraction(self, X, eta, alpha):
        """
        Retract a point X + alpha*eta back to the manifold.
        
        Parameters:
        -----------
        X : TensorTrain
            Current point on the manifold.
        eta : TensorTrain
            Search direction in TT format.
        alpha : float
            Step size.
            
        Returns:
        --------
        TensorTrain
            Retracted point on the manifold in TT format.
        """
        # Create a full tensor approximation (this is inefficient but simpler to implement)
        X_full = X.to_full()
        eta_full = eta.to_full()
        
        # Take the step
        Y_full = X_full + alpha * eta_full
        
        # Retract back to manifold using TT-SVD
        Y = self.TensorTrain.from_tensor(Y_full, epsilon=1e-10, max_rank=max(X.ranks))
        
        return Y
    
    def _vector_transport(self, X, Y, eta):
        """
        Transport a tangent vector eta from X to Y.
        
        Parameters:
        -----------
        X : TensorTrain
            Source point on the manifold.
        Y : TensorTrain
            Destination point on the manifold.
        eta : TensorTrain
            Tangent vector at X in TT format.
            
        Returns:
        --------
        TensorTrain
            Transported tangent vector at Y in TT format.
        """
        # For simplicity, we'll approximate the vector transport by projecting
        # eta onto the tangent space at Y
        return self._project_onto_tangent_space(Y, eta.to_full())
    
    def _compute_fr_beta(self, grad_next, grad):
        """
        Compute beta using the Fletcher-Reeves formula.
        
        Parameters:
        -----------
        grad_next : TensorTrain
            Gradient at next point.
        grad : TensorTrain
            Gradient at current point.
            
        Returns:
        --------
        float
            Beta coefficient.
        """
        # Compute squared norm of gradients
        norm_next = self._squared_norm(grad_next)
        norm_curr = self._squared_norm(grad)
        
        if norm_curr < 1e-10:
            return 0.0
        
        return norm_next / norm_curr
    
    def _squared_norm(self, X):
        """
        Compute the squared norm of a TT tensor.
        
        Parameters:
        -----------
        X : TensorTrain
            Tensor in TT format.
            
        Returns:
        --------
        float
            Squared norm.
        """
        # In a TT tensor, the squared norm can be computed efficiently
        # but for simplicity, we'll use the full tensor
        X_full = X.to_full()
        return np.sum(X_full**2)
    
    def _combine_directions(self, grad, eta, beta):
        """
        Combine gradient and previous direction to get new direction.
        
        Parameters:
        -----------
        grad : TensorTrain
            Gradient at current point.
        eta : TensorTrain
            Previous search direction.
        beta : float
            Combination coefficient.
            
        Returns:
        --------
        TensorTrain
            New search direction in TT format.
        """
        # For simplicity, we'll compute in full format
        grad_full = grad.to_full()
        eta_full = eta.to_full()
        
        # Combine directions
        new_dir_full = -grad_full + beta * eta_full
        
        # Convert back to TT format
        new_dir = self.TensorTrain.from_tensor(new_dir_full, epsilon=1e-10, max_rank=max(grad.ranks))
        
        return new_dir
    
    def _scale_tangent_vector(self, X, scale):
        """
        Scale a tangent vector by a scalar.
        
        Parameters:
        -----------
        X : TensorTrain
            Tangent vector in TT format.
        scale : float
            Scaling factor.
            
        Returns:
        --------
        TensorTrain
            Scaled tangent vector in TT format.
        """
        # For simplicity, scale the full tensor
        X_full = X.to_full()
        scaled_X_full = scale * X_full
        
        # Convert back to TT format
        scaled_X = self.TensorTrain.from_tensor(scaled_X_full, epsilon=1e-10, max_rank=max(X.ranks))
        
        return scaled_X
    
    def _relative_error(self, X, indices, values):
        """
        Compute the relative error on a set of indices.
        
        Parameters:
        -----------
        X : TensorTrain
            Tensor in TT format.
        indices : list of tuples
            Indices to evaluate.
        values : numpy.ndarray
            True values at the indices.
            
        Returns:
        --------
        float
            Relative error.
        """
        # Evaluate X at the given indices
        X_values = np.array([X[idx] for idx in indices])
        
        # Compute relative error
        error = np.linalg.norm(X_values - values)
        norm_values = np.linalg.norm(values)
        
        if norm_values < 1e-10:
            return error
        
        return error / norm_values
    
    def _initialize_random_tt(self, tensor_shape, ranks):
        """
        Initialize a random TT tensor with given shape and ranks.
        
        Parameters:
        -----------
        tensor_shape : tuple
            Shape of the tensor.
        ranks : tuple
            TT ranks.
            
        Returns:
        --------
        TensorTrain
            Random TT tensor.
        """
        d = len(tensor_shape)
        
        # Ensure ranks has the correct length
        if len(ranks) != d - 1:
            ranks = (1,) + tuple(ranks) + (1,)
        
        # Create cores with random values
        cores = []
        
        for i in range(d):
            if i == 0:
                core_shape = (1, tensor_shape[i], ranks[i])
            elif i == d - 1:
                core_shape = (ranks[i-1], tensor_shape[i], 1)
            else:
                core_shape = (ranks[i-1], tensor_shape[i], ranks[i])
            
            core = np.random.randn(*core_shape) * 0.1
            cores.append(core)
        
        # Create TT tensor
        return self.TensorTrain(cores)
    
    def _increase_rank(self, X, mu, new_rank):
        """
        Increase the rank of a TT tensor at mode mu.
        
        Parameters:
        -----------
        X : TensorTrain
            Tensor in TT format.
        mu : int
            Mode at which to increase rank.
        new_rank : int
            New rank value.
            
        Returns:
        --------
        TensorTrain
            TT tensor with increased rank.
        """
        # Get cores and current ranks
        cores = X.cores
        current_ranks = list(X.ranks)
        
        # Check if rank increase is needed
        if current_ranks[mu] >= new_rank:
            return X
        
        # Increase rank by adding random components
        rank_diff = new_rank - current_ranks[mu]
        
        # Modify cores adjacent to the bond to be increased
        if mu > 0:  # Not the first core
            # Update left core
            old_core = cores[mu-1]  # Shape: (r_{mu-2}, n_{mu-1}, r_{mu-1})
            r_left, n, r_right = old_core.shape
            
            # Create random component to increase rank
            random_component = np.random.randn(r_left, n, rank_diff) * 1e-2
            
            # Concatenate along the rank dimension
            new_core = np.zeros((r_left, n, new_rank))
            new_core[:, :, :r_right] = old_core
            new_core[:, :, r_right:] = random_component
            
            cores[mu-1] = new_core
        
        if mu < len(cores) - 1:  # Not the last core
            # Update right core
            old_core = cores[mu]  # Shape: (r_{mu-1}, n_mu, r_mu)
            r_left, n, r_right = old_core.shape
            
            # Create random component to increase rank
            random_component = np.random.randn(rank_diff, n, r_right) * 1e-2
            
            # Concatenate along the rank dimension
            new_core = np.zeros((new_rank, n, r_right))
            new_core[:r_left, :, :] = old_core
            new_core[r_left:, :, :] = random_component
            
            cores[mu] = new_core
        
        # Create new TT tensor with updated cores
        return self.TensorTrain(cores)
    
    def _rank1_approximation(self, X):
        """
        Create a rank-1 approximation of a TT tensor.
        
        Parameters:
        -----------
        X : TensorTrain
            Tensor in TT format.
            
        Returns:
        --------
        TensorTrain
            Rank-1 approximation in TT format.
        """
        # Get cores and shape
        cores = X.cores
        d = len(cores)
        
        # Create new rank-1 cores
        new_cores = []
        
        for i in range(d):
            core = cores[i]
            r_left, n, r_right = core.shape
            
            # For rank-1, we take the first slice of each core
            if i == 0:
                new_core = np.zeros((1, n, 1))
                new_core[0, :, 0] = core[0, :, 0]
            elif i == d - 1:
                new_core = np.zeros((1, n, 1))
                new_core[0, :, 0] = core[0, :, 0]
            else:
                new_core = np.zeros((1, n, 1))
                new_core[0, :, 0] = core[0, :, 0]
            
            new_cores.append(new_core)
        
        # Create new TT tensor with rank-1 cores
        return self.TensorTrain(new_cores)
    
    def _generate_random_indices(self, tensor_shape, num_samples, exclude=None):
        """
        Generate random indices for sampling.
        
        Parameters:
        -----------
        tensor_shape : tuple
            Shape of the tensor.
        num_samples : int
            Number of samples to generate.
        exclude : list of tuples, optional
            Indices to exclude from sampling.
            
        Returns:
        --------
        numpy.ndarray
            Array of random indices.
        """
        d = len(tensor_shape)
        
        # Generate all possible indices
        total_size = np.prod(tensor_shape)
        
        if exclude is not None:
            # Convert multi-indices to linear indices
            linear_exclude = self._multiindices_to_linear(exclude, tensor_shape)
            
            # Generate random linear indices excluding the excluded ones
            available_indices = np.setdiff1d(np.arange(total_size), linear_exclude)
            
            if len(available_indices) < num_samples:
                num_samples = len(available_indices)
                
            linear_indices = np.random.choice(available_indices, size=num_samples, replace=False)
        else:
            # Generate random linear indices
            linear_indices = np.random.choice(total_size, size=num_samples, replace=False)
        
        # Convert linear indices to multi-indices
        return self._linear_to_multiindices(linear_indices, tensor_shape)
    
    def _multiindices_to_linear(self, indices, tensor_shape):
        """
        Convert multi-indices to linear indices.
        
        Parameters:
        -----------
        indices : list of tuples
            Multi-indices to convert.
        tensor_shape : tuple
            Shape of the tensor.
            
        Returns:
        --------
        numpy.ndarray
            Array of linear indices.
        """
        # Convert each multi-index to a linear index
        d = len(tensor_shape)
        linear_indices = np.zeros(len(indices), dtype=int)
        
        # Compute strides
        strides = np.ones(d, dtype=int)
        for i in range(d-2, -1, -1):
            strides[i] = strides[i+1] * tensor_shape[i+1]
        
        # Convert indices
        for i, idx in enumerate(indices):
            linear_idx = 0
            for j in range(d):
                linear_idx += idx[j] * strides[j]
            linear_indices[i] = linear_idx
        
        return linear_indices
    
    def _linear_to_multiindices(self, linear_indices, tensor_shape):
        """
        Convert linear indices to multi-indices.
        
        Parameters:
        -----------
        linear_indices : numpy.ndarray
            Linear indices to convert.
        tensor_shape : tuple
            Shape of the tensor.
            
        Returns:
        --------
        numpy.ndarray
            Array of multi-indices.
        """
        # Convert each linear index to a multi-index
        d = len(tensor_shape)
        multi_indices = np.zeros((len(linear_indices), d), dtype=int)
        
        # Compute strides
        strides = np.ones(d, dtype=int)
        for i in range(d-2, -1, -1):
            strides[i] = strides[i+1] * tensor_shape[i+1]
        
        # Convert indices
        for i, idx in enumerate(linear_indices):
            remaining = idx
            for j in range(d):
                multi_indices[i, j] = remaining // strides[j]
                remaining %= strides[j]
        
        return multi_indices


def test_tt_completion():
    """
    Simple test function for TTCompletion.
    """
    from test import TensorTrain
    
    # Create a random tensor
    tensor_shape = (5, 5, 5)
    X = np.random.rand(*tensor_shape)
    
    # Sample some entries
    indices = np.array([(i, j, k) for i in range(5) for j in range(5) for k in range(5)])
    np.random.shuffle(indices)
    sample_indices = indices[:50]  # Use 50 samples
    sample_values = np.array([X[tuple(idx)] for idx in sample_indices])
    
    # Create TTCompletion instance
    tt_completion = TTCompletion(TensorTrain)
    
    # Complete tensor
    X_completed = tt_completion.complete_tensor(
        sample_indices, sample_values, tensor_shape,
        initial_rank=(1, 1), max_rank=3, 
        tol=1e-4, max_iter=50, verbose=True
    )
    
    # Compute full tensor
    X_completed_full = X_completed.to_full()
    
    # Compute relative error
    relative_error = np.linalg.norm(X - X_completed_full) / np.linalg.norm(X)
    print(f"Relative error of completed tensor: {relative_error:.6e}")
    
    # Test adaptive sampling strategy
    def reference_method(idx):
        return X[tuple(idx)]
    
    X_adaptive = tt_completion.adaptive_sampling_strategy1(
        reference_method, tensor_shape,
        max_sample_percentage=0.3, initial_sample_size=20, test_size=20,
        max_rank=3, tol=1e-4, max_iter=50, verbose=True
    )
    
    # Compute full tensor
    X_adaptive_full = X_adaptive.to_full()
    
    # Compute relative error
    relative_error_adaptive = np.linalg.norm(X - X_adaptive_full) / np.linalg.norm(X)
    print(f"Relative error with adaptive sampling: {relative_error_adaptive:.6e}")


# Test the implementation if this file is run directly
if __name__ == "__main__":
    test_tt_completion()