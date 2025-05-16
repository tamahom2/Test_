import numpy as np
from copy import deepcopy
import time


def euclidgrad(A_Omega, X, Omega):
    """
    Compute Euclidean gradient from residuals.
    
    Parameters
    ----------
    A_Omega : ndarray
        Known values of the tensor
    X : TensorTrain
        Current tensor train approximation
    Omega : ndarray
        Indices of known values
    
    Returns
    -------
    ndarray
        Euclidean gradient values
    """
    return X.gather(Omega) - A_Omega


def func(A_Omega, X, Omega):
    """
    Cost function for tensor completion.
    
    Parameters
    ----------
    A_Omega : ndarray
        Known values of the tensor
    X : TensorTrain
        Current tensor train approximation
    Omega : ndarray
        Indices of known values
    
    Returns
    -------
    float
        Value of the cost function (1/2 * squared Frobenius norm of error)
    """
    diff = X.gather(Omega) - A_Omega
    return 0.5 * np.linalg.norm(diff)**2


def completion(A_Omega, Omega, A_Gamma=None, Gamma=None, X0=None, opts=None):
    """
    Core optimization function for TT completion using Riemannian optimization.
    
    This implements a Riemannian optimization approach for tensor train completion
    based on the GeomCG algorithm from the paper:
    "Riemannian Optimization for High-Dimensional Tensor Completion"
    by M. Steinlechner.
    
    Parameters
    ----------
    A_Omega : ndarray
        Known values of the tensor
    Omega : ndarray
        Indices of known values
    A_Gamma : ndarray, optional
        Test set values (for validation)
    Gamma : ndarray, optional
        Indices of test set
    X0 : TensorTrain
        Initial guess tensor train
    opts : dict
        Options for the algorithm including:
        - maxiter : Maximum number of iterations (default: 100)
        - cg : Whether to use conjugate gradient (default: True)
        - tol : Tolerance for convergence (default: 1e-6)
        - reltol : Relative tolerance for convergence (default: 1e-8)
        - gradtol : Gradient norm tolerance (default: 10*eps)
        - verbose : Whether to print progress (default: False)
    
    Returns
    -------
    X : TensorTrain
        Completed tensor train
    cost : ndarray
        Cost function values during optimization
    test : ndarray or None
        Test error values during optimization
    stats : dict
        Additional statistics from the optimization
    """
    # Set default options
    if opts is None:
        opts = {}
    maxiter = opts.get('maxiter', 100)
    use_cg = opts.get('cg', True)
    tol = opts.get('tol', 1e-6)
    reltol = opts.get('reltol', 1e-8)
    gradtol = opts.get('gradtol', 10 * np.finfo(float).eps)
    verbose = opts.get('verbose', False)
    
    # Initialize variables
    xL = deepcopy(X0)
    xR = deepcopy(X0)
    xR.orthogonalize(mode='r')
    
    norm_A_Omega = np.linalg.norm(A_Omega)
    if A_Gamma is not None and Gamma is not None:
        norm_A_Gamma = np.linalg.norm(A_Gamma)
    
    cost = np.zeros(maxiter)
    test = np.zeros(maxiter) if A_Gamma is not None and Gamma is not None else None
    stats = {'gradnorm': np.zeros(maxiter), 'time': [0], 'conv': False}
    reltol_val = -np.inf
    
    # Main optimization loop
    start_time = time.time()
    
    for i in range(maxiter):
        # Compute Euclidean gradient
        grad = euclidgrad(A_Omega, xL, Omega)
        
        # Project to tangent space
        xi = xL.rgrad_sparse(grad, Omega)
        ip_xi_xi = xi.inner(xi)
        stats['gradnorm'][i] = np.sqrt(abs(ip_xi_xi))
        
        # Check gradient norm for convergence
        if np.sqrt(abs(ip_xi_xi)) < gradtol:
            if cost[i] < tol:
                if verbose:
                    print(f"CONVERGED AFTER {i} STEPS. Gradient is smaller than {gradtol:.3g}")
                stats['conv'] = True
            else:
                if verbose:
                    print("No more progress in gradient change, but not converged. Aborting!")
                stats['conv'] = False
            
            # Trim arrays to current iteration
            cost = cost[:i]
            if test is not None:
                test = test[:i]
            stats['gradnorm'] = stats['gradnorm'][:i]
            
            stats['time'].append(stats['time'][-1] + time.time() - start_time)
            stats['time'] = stats['time'][1:]
            
            return xL, cost, test, stats
        
        # Determine search direction
        if (i == 0) or (not use_cg):
            eta = -1 * xi  # Steepest descent
        else:
            ip_xitrans_xi = xi_trans.inner(xi)
            theta = ip_xitrans_xi / ip_xi_xi
            
            if theta >= 0.1:
                if verbose:
                    print('steepest descent step')
                eta = -1 * xi  # Steepest descent
            else:
                if verbose:
                    print('CG step')
                beta = ip_xi_xi / ip_xi_xi_old
                print(beta.dtype)
                eta = -1 * xi + float(beta) * xL.grad_proj(eta)  # Conjugate gradient
        
        # Line search to determine step size
        eta_Omega = eta.to_eucl(Omega)
        alpha = -(np.dot(eta_Omega, grad)) / np.linalg.norm(eta_Omega)**2
        
        # Apply update via retraction
        X = xL.apply_grad(eta, alpha=alpha, round=True, inplace=False)
        xL = X.orthogonalize(mode=X.order-1, inplace=False)
        xR = X.orthogonalize(mode=0, inplace=False)
        
        # Compute cost
        cost[i] = np.sqrt(2*func(A_Omega, xL, Omega)) / norm_A_Omega
        
        # Check for convergence based on cost
        if cost[i] < tol:
            if verbose:
                print(f"CONVERGED AFTER {i} STEPS. Rel. residual smaller than {tol:.3g}")
            stats['conv'] = True
            cost = cost[:i+1]
            stats['gradnorm'] = stats['gradnorm'][:i+1]
            
            stats['time'].append(stats['time'][-1] + time.time() - start_time)
            
            if test is not None and A_Gamma is not None and Gamma is not None:
                test[i] = np.sqrt(2*func(A_Gamma, xL, Gamma)) / norm_A_Gamma
                test = test[:i+1]
            
            stats['time'] = stats['time'][1:]
            return xL, cost, test, stats
        
        # Check for relative change in cost
        if i > 0:
            reltol_val = abs(cost[i] - cost[i-1]) / cost[i]
            if reltol_val < reltol:
                if cost[i] < tol:
                    if verbose:
                        print(f"CONVERGED AFTER {i} STEPS. Relative change is smaller than {reltol:.3g}")
                    stats['conv'] = True
                else:
                    if verbose:
                        print("No more progress in relative change, but not converged. Aborting!")
                    stats['conv'] = False
                
                cost = cost[:i+1]
                stats['gradnorm'] = stats['gradnorm'][:i+1]
                
                stats['time'].append(stats['time'][-1] + time.time() - start_time)
                
                if test is not None and A_Gamma is not None and Gamma is not None:
                    test[i] = np.sqrt(2*func(A_Gamma, xL, Gamma)) / norm_A_Gamma
                    test = test[:i+1]
                
                stats['time'] = stats['time'][1:]
                return xL, cost, test, stats
        
        # Save for next iteration
        ip_xi_xi_old = ip_xi_xi
        xi_trans = xi
        
        # Update timing and test error
        stats['time'].append(stats['time'][-1] + time.time() - start_time)
        
        if test is not None and A_Gamma is not None and Gamma is not None:
            test[i] = np.sqrt(2*func(A_Gamma, xL, Gamma)) / norm_A_Gamma
        
        start_time = time.time()
        
        if verbose:
            print(f'k: {i}, cost: {cost[i]}, test: {test[i] if test is not None else "N/A"}, ' 
                  f'Riem grad: {stats["gradnorm"][i]}, rel_cost {reltol_val}')
    
    stats['time'] = stats['time'][1:]
    return xL, cost, test, stats


def evaluate_on_control(X, A_Omega_C, Omega_C):
    """
    Evaluate the cost function on a control set.
    
    Parameters
    ----------
    X : TensorTrain
        Current tensor train approximation
    A_Omega_C : ndarray
        Control set values
    Omega_C : ndarray
        Indices of control set
    
    Returns
    -------
    float
        Value of the cost function on control set
    """
    diff = X.gather(Omega_C) - A_Omega_C
    return 0.5 * np.linalg.norm(diff)**2


def completion_with_rank_adaptation(method, A_Omega, Omega, A_Omega_C, Omega_C, 
                                   A_Gamma, Gamma, X0, opts=None):
    """
    Tensor completion with adaptive rank.
    
    This function implements tensor completion with adaptive rank adaptation,
    iteratively trying to increase the rank of each core and accepting
    increases that improve performance.
    
    Parameters
    ----------
    method : str
        Completion method: 'GeomCG' or 'ALS'
    A_Omega : ndarray
        Known values of the tensor
    Omega : ndarray
        Indices of known values
    A_Omega_C : ndarray
        Control set values
    Omega_C : ndarray
        Indices of control set
    A_Gamma : ndarray
        Test set values
    Gamma : ndarray
        Indices of test set
    X0 : TensorTrain
        Initial guess tensor train
    opts : dict
        Options for the algorithm including:
        - maxrank : Maximum rank to try (default: 4)
        - cg : Whether to use conjugate gradient (default: True)
        - tol : Tolerance for convergence (default: 1e-6)
        - reltol : Relative tolerance for convergence (default: 1e-8)
        - reltol_final : Final relative tolerance (default: eps)
        - maxiter : Maximum iterations per optimization (default: 10)
        - maxiter_final : Maximum iterations for final optimization (default: 20)
        - locked_tol : Tolerance for locking cores (default: 1)
        - epsilon : Small value for rank increase (default: 1e-8)
        - verbose : Whether to print progress (default: False)
    
    Returns
    -------
    X : TensorTrain
        Completed tensor train
    cost : ndarray
        Cost function values during optimization
    test : ndarray
        Test error values during optimization
    stats : dict
        Additional statistics
    ranks : list
        Rank information tracking during optimization
    """
    # Set default options
    if opts is None:
        opts = {}
    
    maxrank = opts.get('maxrank', 4)
    use_cg = opts.get('cg', True)
    tol = opts.get('tol', 1e-6)
    reltol = opts.get('reltol', 1e-8)
    reltol_final = opts.get('reltol_final', np.finfo(float).eps)
    maxiter = opts.get('maxiter', 10)
    maxiter_final = opts.get('maxiter_final', 20)
    locked_tol = opts.get('locked_tol', 1)
    epsilon = opts.get('epsilon', 1e-8)
    verbose = opts.get('verbose', False)
    
    # Select completion method
    if method.lower() == 'geomcg':
        completion_func = completion
    elif method.lower() == 'als':
        # Not implemented in this code, would need to be added
        raise NotImplementedError("ALS method not implemented")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    d = X0.order
    control_old = np.inf
    
    # Start timing
    start_time = time.time()
    
    # Initial completion with starting rank
    if verbose:
        print('____________________________________________________________________')
        print(f'Completion with starting rank r = {X0.tt_rank} ...')
    
    X, cost, control, stats = completion_func(A_Omega, Omega, A_Gamma, Gamma, X0, opts)
    
    # Initialize test array and timing stats
    if A_Gamma is not None:
        test = [np.linalg.norm(X0.gather(Gamma) - A_Gamma) / np.linalg.norm(A_Gamma)]
        test.append(np.linalg.norm(X.gather(Gamma) - A_Gamma) / np.linalg.norm(A_Gamma))
    else:
        test = []
    
    stats['time'] = [0, time.time() - start_time]
    ranks = [X0.tt_rank, X.tt_rank]  # Track rank history
    
    stats['rankidx'] = len(cost)
    
    if verbose:
        print('____________________________________________________________________')
        print('Increasing rank ...')
    
    # Initialize locked cores array (which cores can no longer increase in rank)
    locked = np.zeros(d+1, dtype=bool)
    
    # Loop over ranks to try increasing
    for k in range(2, maxrank+1):
        for i in range(d-1):  # Adjust indices to match Python's 0-indexing for cores
            if verbose:
                print(f'Locked cores: {locked}')
            
            if locked[i]:
                if verbose:
                    print(f'Rank r({i}) is locked. Skipping.')
            else:
                curr_rank = X.tt_rank[i-1] if i-1 < len(X.tt_rank) else X.tt_rank[-1]
                new_rank = curr_rank + 1
                
                if verbose:
                    print(f'Trying to increase rank r({i}) from {curr_rank} to {new_rank}:')
                
                # Create new TT with increased rank
                Xnew = X.copy()
                Xnew.increase_rank(1, i)  # 0-indexed in Python
                Xnew.orthogonalize()
                
                # Adjust max iterations for final rank
                if i == d-1 and k == maxrank:
                    opts_copy = opts.copy()
                    opts_copy['maxiter'] = maxiter_final
                else:
                    opts_copy = opts
                
                # Run completion with new rank
                Xnew, cost_tmp, control_tmp, stats_tmp = completion_func(
                    A_Omega, Omega, A_Omega_C, Omega_C, Xnew, opts_copy)
                
                stats['rankidx'] = np.append(stats['rankidx'], len(cost_tmp))
                
                if verbose:
                    print(f'Current cost function: {cost_tmp[-1]}')
                
                # Calculate progress on control set
                new_control = evaluate_on_control(Xnew, A_Omega_C, Omega_C)
                progress = (new_control - control_old) / control_old
                
                if verbose:
                    print(f'Current rel. progress on control: {progress}')
                
                # Accept or reject rank increase
                if progress > locked_tol:
                    if verbose:
                        print('     ... failed. Reverting.')
                    locked[i] = True
                else:
                    if verbose:
                        print('     ... accepted.')
                    X = Xnew
                    control_old = new_control
                    
                    if A_Gamma is not None:
                        test_current = np.linalg.norm(X.gather(Gamma) - A_Gamma) / np.linalg.norm(A_Gamma)
                        if verbose:
                            print(f'Current error on test set Gamma: {test_current}')
                        test.append(test_current)
                    
                    # Record time
                    stats['time'].append(time.time() - start_time)
                    ranks.append(X.tt_rank)
                
                # Combine results
                cost = np.append(cost, cost_tmp)
                control = np.append(control, control_tmp) if control is not None else control_tmp
    
    return X, cost, test, stats, ranks


def create_random_completion_problem(dims, rank, sampling_factor=5, test_factor=0.1):
    """
    Create a random tensor completion problem.
    
    Parameters
    ----------
    dims : tuple
        Dimensions of the tensor
    rank : int
        Rank of the tensor train to generate
    sampling_factor : float
        Factor determining how many samples to take (sampling_factor * rank * sum(dims))
    test_factor : float
        Fraction of samples to use for testing
    
    Returns
    -------
    X_full : TensorTrain
        Full tensor train
    A_Omega : ndarray
        Known values
    Omega : ndarray
        Indices of known values
    A_Gamma : ndarray
        Test values
    Gamma : ndarray
        Indices of test values
    """
    from tt_core import TensorTrain
    
    # Create a random TT tensor
    X_full = TensorTrain.random(dims, rank)
    
    # Generate total number of samples
    n_samples = int(sampling_factor * rank * sum(dims))
    
    # Generate random indices
    total_elements = np.prod(dims)
    linear_indices = np.random.choice(total_elements, n_samples, replace=False)
    
    # Convert linear indices to multi-indices
    multi_indices = np.zeros((n_samples, len(dims)), dtype=int)
    for i, idx in enumerate(linear_indices):
        for j in range(len(dims)-1, -1, -1):
            multi_indices[i, j] = idx % dims[j]
            idx //= dims[j]
    
    # Split into training and test sets
    n_test = int(test_factor * n_samples)
    n_train = n_samples - n_test
    
    # Training set
    Omega = multi_indices[:n_train]
    A_Omega = X_full.gather(Omega)
    
    # Test set
    Gamma = multi_indices[n_train:]
    A_Gamma = X_full.gather(Gamma)
    
    return X_full, A_Omega, Omega, A_Gamma, Gamma


def example_usage():
    """Example of how to use the tensor completion functions."""
    # Create a random tensor completion problem
    dims = (10, 10, 10, 10)  # 4D tensor with dimension 10 in each mode
    rank = 3                  # TT-rank
    X_full, A_Omega, Omega, A_Gamma, Gamma = create_random_completion_problem(dims, rank)
    print(A_Omega)
    print(Omega)
    # Create a random initial guess with lower rank
    from tt_core import TensorTrain
    X0 = TensorTrain.random(dims, 1)
    
    # Set options for completion
    opts = {
        'maxrank': 5,     # Maximum rank to try
        'maxiter': 20,    # Maximum iterations per optimization
        'tol': 1e-6,      # Tolerance for convergence
        'verbose': True   # Print progress
    }
    
    # Run completion with adaptive rank
    X, cost, test, stats, ranks = completion_with_rank_adaptation(
        'GeomCG', A_Omega, Omega, A_Omega, Omega, A_Gamma, Gamma, X0, opts
    )
    
    # Calculate relative error
    rel_error = np.linalg.norm(X.gather(Gamma) - A_Gamma) / np.linalg.norm(A_Gamma)
    print(f"Final relative error on test set: {rel_error}")
    
    return X, cost, test, stats, ranks

example_usage()