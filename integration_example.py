import numpy as np
import matplotlib.pyplot as plt
from test import TensorTrain, ChebyshevInterpolation
from tt_completion import TTCompletion

def example_chebyshev_interpolation_with_completion():
    """
    Example of how to integrate the tensor completion algorithm with the ChebyshevInterpolation class.
    This demonstrates using tensor completion to construct the tensor P efficiently.
    """
    # Define a reference function for testing - we'll use a sine function
    def sine_function(x):
        if isinstance(x, tuple):
            x = x[0]  # Extract first element if tuple
        return np.sin(x)
    
    # Define domain and interpolation order
    domain = [(-np.pi, np.pi)]  # Domain for x
    degree = [10]               # Degree of Chebyshev polynomial
    
    # Create the Chebyshev interpolation object
    cheb = ChebyshevInterpolation(domain, degree)
    
    # Define a sampling strategy that only computes a subset of points in the Chebyshev grid
    def compute_subset_of_chebyshev_grid(subset_percentage=0.3):
        """
        Compute prices at a subset of Chebyshev grid points.
        
        Parameters:
        -----------
        subset_percentage : float
            Percentage of grid points to compute explicitly.
            
        Returns:
        --------
        tuple
            (indices, values) where indices are the sampled indices and values are the corresponding values.
        """
        # Get the full Chebyshev grid shape
        grid_shape = tuple(d + 1 for d in cheb.degrees)
        
        # Compute total number of grid points
        total_points = np.prod(grid_shape)
        
        # Determine number of points to sample
        num_samples = int(subset_percentage * total_points)
        
        # Generate random indices for the grid
        all_indices = np.array(np.unravel_index(np.arange(total_points), grid_shape)).T
        np.random.shuffle(all_indices)
        
        # Take subset of indices
        subset_indices = all_indices[:num_samples]
        
        # Compute values at these indices
        subset_values = []
        for idx in subset_indices:
            # Convert grid indices to Chebyshev nodes
            params = tuple(cheb.points[dim][idx[dim]] for dim in range(len(domain)))
            # Compute function value
            value = sine_function(params)
            subset_values.append(value)
        
        return subset_indices, np.array(subset_values)
    
    # Compute values at a subset of grid points
    sample_indices, sample_values = compute_subset_of_chebyshev_grid(0.3)
    print(f"Computed {len(sample_indices)} out of {np.prod(tuple(d + 1 for d in cheb.degrees))} grid points ({len(sample_indices)/np.prod(tuple(d + 1 for d in cheb.degrees))*100:.1f}%)")
    
    # Initialize the TT completion object
    tt_completion = TTCompletion(TensorTrain)
    
    # Complete the tensor P using tensor completion
    tensor_shape = tuple(d + 1 for d in cheb.degrees)
    P_completed = tt_completion.complete_tensor(
        sample_indices, sample_values, tensor_shape,
        initial_rank=(1,), max_rank=5,
        tol=1e-6, max_iter=100, verbose=True
    )
    
    # Store the completed tensor in the Chebyshev interpolation object
    cheb.P = P_completed
    
    # Construct tensor C as in the normal flow
    cheb.construct_tensor_C()
    
    # Generate test points for plotting
    x_test = np.linspace(-np.pi, np.pi, 100)
    
    # Evaluate exact function
    y_exact = np.sin(x_test)
    
    # Evaluate interpolation
    y_interp = []
    for x in x_test:
        y_interp.append(cheb.evaluate_interpolation(x))
    y_interp = np.array(y_interp)
    
    # Calculate error
    max_error = np.max(np.abs(y_exact - y_interp))
    rms_error = np.sqrt(np.mean((y_exact - y_interp)**2))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(x_test, y_exact, 'b-', label='Exact sine')
    plt.plot(x_test, y_interp, 'r--', label='Chebyshev Interpolation with TT Completion')
    
    # Plot the original sampled points
    cheb_nodes = cheb.points[0]
    sampled_indices = np.unique([idx[0] for idx in sample_indices])
    plt.scatter(cheb_nodes[sampled_indices], np.sin(cheb_nodes[sampled_indices]), 
                color='k', marker='o', s=50, label='Sampled Chebyshev Nodes')
    
    plt.grid(True)
    plt.legend()
    plt.title(f'Chebyshev Interpolation with TT Completion (degree={degree[0]})')
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


def example_american_option_pricing():
    """
    Example of using tensor completion for American option pricing in the Heston model.
    """
    try:
        from test import TensorTrain, ChebyshevInterpolation
        from tt_completion import TTCompletion
        
        # We'll simulate a simple placeholder reference method for pricing
        # In a real application, this would be your PDC solver from the Heston model
        def mock_american_option_pricing(params):
            # This is just a placeholder function
            # params = (K, ρ, σ, κ, θ)
            K, rho, sigma, kappa, theta = params
            
            # Some arbitrary function that simulates option price behavior
            return 0.5 + 0.1*K - 0.2*rho + 0.3*sigma + 0.15*kappa + 0.25*theta + 0.1*K*sigma*theta
        
        # Define parameter domains
        domains = [
            (2, 4),      # K (strike)
            (-1, 1),     # ρ (correlation)
            (0.2, 0.5),  # σ (volatility of volatility)
            (1, 2),      # κ (mean reversion rate)
            (0.05, 0.2)  # θ (long-term mean)
        ]
        
        # Define interpolation degrees
        degrees = [4, 4, 4, 4, 4]  # Using small degrees for demonstration
        
        # Create the Chebyshev interpolation object
        cheb = ChebyshevInterpolation(domains, degrees)
        
        # Compute a small subset of the tensor P explicitly
        subset_percentage = 0.01  # Just 1% of the total grid points
        grid_shape = tuple(d + 1 for d in degrees)
        total_points = np.prod(grid_shape)
        num_samples = int(subset_percentage * total_points)
        
        print(f"Total grid points: {total_points}")
        print(f"Computing {num_samples} points ({subset_percentage*100:.1f}% of total)")
        
        # Generate random indices for sampling
        all_indices = np.array(np.unravel_index(np.arange(total_points), grid_shape)).T
        np.random.shuffle(all_indices)
        subset_indices = all_indices[:num_samples]
        
        # Compute values at these indices
        subset_values = []
        for idx in subset_indices:
            # Convert grid indices to Chebyshev nodes
            params = tuple(cheb.points[dim][idx[dim]] for dim in range(len(domains)))
            # Compute option price
            value = mock_american_option_pricing(params)
            subset_values.append(value)
        
        subset_values = np.array(subset_values)
        
        # Initialize the TT completion object
        tt_completion = TTCompletion(TensorTrain)
        
        # Complete the tensor P using adaptive sampling strategy
        print("\nRunning TT completion with adaptive sampling...")
        
        def reference_method(idx):
            # Convert grid indices to Chebyshev nodes
            params = tuple(cheb.points[dim][idx[dim]] for dim in range(len(domains)))
            # Compute option price
            return mock_american_option_pricing(params)
        
        P_completed = tt_completion.adaptive_sampling_strategy1(
            reference_method, grid_shape,
            max_sample_percentage=0.05,  # Use at most 5% of total grid points
            initial_sample_size=num_samples, 
            test_size=100,
            max_rank=5, tol=1e-4, max_iter=50, verbose=True
        )
        
        # Store the completed tensor in the Chebyshev interpolation object
        cheb.P = P_completed
        
        # Construct tensor C as in the normal flow
        cheb.construct_tensor_C()
        
        # Generate test points
        num_test_points = 100
        test_params = []
        for _ in range(num_test_points):
            params = []
            for domain in domains:
                params.append(np.random.uniform(domain[0], domain[1]))
            test_params.append(tuple(params))
        
        # Evaluate exact function and interpolation
        exact_values = []
        interp_values = []
        
        for params in test_params:
            exact = mock_american_option_pricing(params)
            interp = cheb.evaluate_interpolation(params)
            
            exact_values.append(exact)
            interp_values.append(interp)
        
        exact_values = np.array(exact_values)
        interp_values = np.array(interp_values)
        
        # Calculate error
        abs_errors = np.abs(exact_values - interp_values)
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        rel_errors = abs_errors / np.abs(exact_values)
        mean_rel_error = np.mean(rel_errors) * 100  # as percentage
        
        print("\nInterpolation results:")
        print(f"Maximum absolute error: {max_error:.6f}")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Mean relative error: {mean_rel_error:.4f}%")
        
        # Plot histogram of absolute errors
        plt.figure(figsize=(10, 6))
        plt.hist(abs_errors, bins=20, alpha=0.7)
        plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.6f}')
        plt.grid(True)
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.title('Distribution of Interpolation Errors')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in example_american_option_pricing: {e}")


def example_basket_option_pricing():
    """
    Example of using tensor completion for basket option pricing in the Black-Scholes model.
    """
    try:
        from test import TensorTrain, ChebyshevInterpolation
        from tt_completion import TTCompletion
        import time
        
        # Define a simplified basket option pricing function
        def mock_basket_option_pricing(initial_prices, strike=1.0, r=0, T=0.25, sigma=0.2, weights=None):
            """
            Mock function for basket option pricing in Black-Scholes model.
            
            Parameters:
            -----------
            initial_prices : tuple
                Initial stock prices.
            strike : float
                Strike price.
            r : float
                Risk-free rate.
            T : float
                Time to maturity.
            sigma : float
                Volatility.
            weights : list or None
                Weights for stocks in the basket. Equal weights if None.
                
            Returns:
            --------
            float
                Option price.
            """
            d = len(initial_prices)
            
            if weights is None:
                weights = [1.0/d] * d
                
            # Simplified Black-Scholes formula for basket option
            # This is just a placeholder approximation
            weighted_sum = sum(w * S0 for w, S0 in zip(weights, initial_prices))
            vol_term = sigma * np.sqrt(T) * np.sqrt(sum(w**2 for w in weights))
            
            # Approximate price (not accurate, just for demonstration)
            intrinsic = max(0, weighted_sum - strike)
            time_value = 0.4 * vol_term * weighted_sum
            
            return intrinsic + time_value
        
        # Define the dimension (number of assets)
        d = 10  # Using a smaller value for demonstration
        
        # Define domains for each asset (initial price)
        domains = [(1.0, 1.5)] * d
        
        # Define interpolation degrees
        degrees = [3] * d  # Using smaller degrees for demonstration
        
        # Create the Chebyshev interpolation object
        cheb = ChebyshevInterpolation(domains, degrees)
        
        # Compute a tiny subset of the tensor P explicitly
        grid_shape = tuple(deg + 1 for deg in degrees)
        total_points = np.prod(grid_shape)
        
        # For high dimensions, we can only afford a very small percentage
        subset_percentage = 1.0 / total_points * 1000  # Aim for ~1000 points
        subset_percentage = min(subset_percentage, 0.01)  # Cap at 1%
        
        num_samples = max(500, int(subset_percentage * total_points))
        
        print(f"Total grid points: {total_points}")
        print(f"Computing {num_samples} points ({num_samples/total_points*100:.8f}% of total)")
        
        # Generate random indices for sampling
        all_indices = np.array(np.unravel_index(np.arange(total_points), grid_shape)).T
        np.random.shuffle(all_indices)
        subset_indices = all_indices[:num_samples]
        
        # Compute values at these indices
        subset_values = []
        for idx in subset_indices:
            # Convert grid indices to Chebyshev nodes
            params = tuple(cheb.points[dim][idx[dim]] for dim in range(d))
            # Compute option price
            value = mock_basket_option_pricing(params)
            subset_values.append(value)
        
        subset_values = np.array(subset_values)
        
        # Initialize the TT completion object
        tt_completion = TTCompletion(TensorTrain)
        
        # Define reference method for adaptive sampling
        def reference_method(idx):
            # Convert grid indices to Chebyshev nodes
            params = tuple(cheb.points[dim][idx[dim]] for dim in range(d))
            # Compute option price
            return mock_basket_option_pricing(params)
        
        # Complete the tensor P using adaptive sampling strategy
        print("\nRunning TT completion with adaptive sampling...")
        start_time = time.time()
        
        P_completed = tt_completion.adaptive_sampling_strategy1(
            reference_method, grid_shape,
            max_sample_percentage=0.001,  # Very small percentage due to high dimensionality
            initial_sample_size=num_samples, 
            test_size=100,
            max_rank=4, tol=1e-3, max_iter=30, verbose=True
        )
        
        completion_time = time.time() - start_time
        print(f"Completion time: {completion_time:.2f} seconds")
        
        # Store the completed tensor in the Chebyshev interpolation object
        cheb.P = P_completed
        
        # Construct tensor C as in the normal flow
        cheb.construct_tensor_C()
        
        # Generate test points
        num_test_points = 100
        test_params = []
        for _ in range(num_test_points):
            params = []
            for domain in domains:
                params.append(np.random.uniform(domain[0], domain[1]))
            test_params.append(tuple(params))
        
        # Evaluate using both methods
        exact_values = []
        interp_values = []
        exact_time = 0
        interp_time = 0
        
        for params in test_params:
            # Exact computation
            start_time = time.time()
            exact = mock_basket_option_pricing(params)
            exact_time += time.time() - start_time
            
            # Interpolation
            start_time = time.time()
            interp = cheb.evaluate_interpolation(params)
            interp_time += time.time() - start_time
            
            exact_values.append(exact)
            interp_values.append(interp)
        
        exact_values = np.array(exact_values)
        interp_values = np.array(interp_values)
        
        # Calculate error
        abs_errors = np.abs(exact_values - interp_values)
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        rel_errors = abs_errors / np.abs(exact_values)
        mean_rel_error = np.mean(rel_errors) * 100  # as percentage
        
        print("\nInterpolation results:")
        print(f"Maximum absolute error: {max_error:.6f}")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Mean relative error: {mean_rel_error:.4f}%")
        
        print("\nTiming comparison:")
        print(f"Average time for exact computation: {exact_time/num_test_points*1000:.3f} ms")
        print(f"Average time for interpolation: {interp_time/num_test_points*1000:.3f} ms")
        print(f"Speed-up factor: {exact_time/interp_time:.1f}x")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of exact vs interpolated values
        plt.subplot(221)
        plt.scatter(exact_values, interp_values, alpha=0.6)
        min_val = min(np.min(exact_values), np.min(interp_values))
        max_val = max(np.max(exact_values), np.max(interp_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.grid(True)
        plt.xlabel('Exact Option Price')
        plt.ylabel('Interpolated Option Price')
        plt.title('Exact vs Interpolated Prices')
        
        # Histogram of absolute errors
        plt.subplot(222)
        plt.hist(abs_errors, bins=20, alpha=0.7)
        plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.6f}')
        plt.grid(True)
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.title('Distribution of Absolute Errors')
        plt.legend()
        
        # Histogram of relative errors
        plt.subplot(223)
        plt.hist(rel_errors * 100, bins=20, alpha=0.7)
        plt.axvline(mean_rel_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_rel_error:.4f}%')
        plt.grid(True)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Count')
        plt.title('Distribution of Relative Errors')
        plt.legend()
        
        # Timing comparison
        plt.subplot(224)
        plt.bar(['Exact', 'Interpolation'], [exact_time/num_test_points*1000, interp_time/num_test_points*1000])
        plt.ylabel('Time per Evaluation (ms)')
        plt.title(f'Performance Comparison (Speed-up: {exact_time/interp_time:.1f}x)')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in example_basket_option_pricing: {e}")


if __name__ == "__main__":
    # Run the examples
    example_chebyshev_interpolation_with_completion()
    example_american_option_pricing()
    example_basket_option_pricing()