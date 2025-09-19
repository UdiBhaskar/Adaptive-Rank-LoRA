import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math


def spectral_analysis_for_adaptive_lora(
    model=None,
    eigenvalue_threshold=0.00001,
    histogram_bins=100,
    power_law_fitting_method=None,
    xmin_position=2,
    conv_normalization=0.5,
    filter_near_zero_eigenvalues=False,
    apply_tracy_widom_correction=True,
):
    """
    Performs spectral analysis on neural network layers to compute metrics for adaptive LoRA rank assignment.
    
    This function implements Empirical Spectral Density (ESD) analysis using power law fitting (Hill estimator)
    and Marchenko-Pastur (MP) bulk detection to identify the spectral properties of weight matrices.
    
    Args:
        model (torch.nn.Module, optional): The neural network model to analyze. Defaults to None.
        eigenvalue_threshold (float, optional): Threshold to filter near-zero eigenvalues. Defaults to 0.00001.
        histogram_bins (int, optional): Number of bins for histogram-based power law fitting. Defaults to 100.
        power_law_fitting_method (str, optional): Method for power law fitting. 
            Options: ['median', 'goodness-of-fit', 'fix-finger']. Defaults to None.
        xmin_position (int, optional): Position in spectrum for xmin selection. 
            2 = middle of spectrum, larger values select smaller eigenvalues. Defaults to 2.
        conv_normalization (float, optional): Normalization factor for convolutional layers. Defaults to 0.5.
        filter_near_zero_eigenvalues (bool, optional): Whether to filter eigenvalues near zero. Defaults to False.
        apply_tracy_widom_correction (bool, optional): Whether to apply Tracy-Widom correction to MP bulk edge. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing spectral analysis results for each layer:
            - layer_index: Sequential index of the layer
            - layer_name: Name of the layer
            - alpha_hill: Power law exponent (Hill estimator)
            - spectral_norm: Largest singular value of the weight matrix
            - D: Kolmogorov-Smirnov statistic for power law fit quality
            - eigs: Array of squared singular values (eigenvalues)
            - norm: Frobenius norm of the weight matrix
            - alphahat: Scaled alpha value (alpha * log10(spectral_norm))
            - num_spikes: Number of eigenvalues above Marchenko-Pastur bulk edge
    
    Notes:
        - The function analyzes Conv2d and Linear layers only
        - Convolutional layers are reshaped and normalized appropriately
        - Power law fitting uses the Hill estimator method
        - Marchenko-Pastur analysis detects eigenvalues outside the bulk distribution
        - Tracy-Widom correction improves bulk edge estimation for finite matrices
    """
    # Initialize results dictionary to store spectral analysis metrics for each layer
    results = {
        "layer_index": [],      # Sequential layer index
        "alpha_hill": [],       # Power law exponent from Hill estimator
        "spectral_norm": [],    # Largest singular value (operator norm)
        "D": [],                # Kolmogorov-Smirnov statistic for goodness of fit
        "layer_name": [],       # Layer name from model
        "eigs": [],             # Eigenvalues (squared singular values)
        "norm": [],             # Frobenius norm of weight matrix
        "alphahat": [],         # Scaled alpha: alpha * log10(spectral_norm)
        "num_spikes": [],       # Number of eigenvalues above MP bulk edge
    }
    
    # Small epsilon to prevent division by zero and log of zero
    eps = 1e-8
    
    # Iterate through all named modules in the neural network
    layer_index = 0
    for layer_name, module in model.named_modules():
        # Clear GPU cache to prevent memory issues during analysis
        torch.cuda.empty_cache()
        layer_index += 1
        
        # Only analyze Conv2d and Linear layers as they have weight matrices suitable for spectral analysis
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Clone weight matrix to avoid modifying original weights
            weight_matrix = module.weight.data.clone()
            
            # Reshape convolutional layers for spectral analysis
            if isinstance(module, nn.Conv2d):
                # Flatten spatial dimensions and apply normalization
                # Shape: (out_channels, in_channels, H, W) -> (out_channels, in_channels * H * W)
                weight_matrix = torch.flatten(weight_matrix, start_dim=2) * math.sqrt(conv_normalization)
                # Transpose to get proper matrix shape for SVD
                weight_matrix = weight_matrix.transpose(1, 2).transpose(0, 1)

            # Calculate matrix dimensions for Marchenko-Pastur (MP) analysis
            # M = smaller dimension, N = larger dimension
            M, N = min(weight_matrix.shape), max(weight_matrix.shape)
            Q = N / M  # Aspect ratio for MP law

            # Compute eigenvalues as squared singular values
            # Using SVD to get singular values, then squaring them to get eigenvalues of W^T W
            singular_values = torch.linalg.svdvals(weight_matrix.to(torch.float32)).flatten()
            eigenvalues = torch.square(singular_values)
            
            # Sort eigenvalues in ascending order (standard for spectral analysis)
            eigenvalues, _ = torch.sort(eigenvalues, descending=False)
            
            # Calculate key matrix norms
            spectral_norm = eigenvalues[-1].item()  # Largest eigenvalue (operator norm)
            frobenius_norm = torch.sum(eigenvalues).item()  # Sum of eigenvalues (Frobenius norm squared)

            # Filter near-zero eigenvalues if requested
            if filter_near_zero_eigenvalues:
                # Remove eigenvalues below threshold to avoid numerical issues
                non_zero_eigenvalues = eigenvalues[eigenvalues > eigenvalue_threshold]
                num_eigenvalues = len(non_zero_eigenvalues)
                
                # Handle edge case where all eigenvalues are filtered out
                if num_eigenvalues == 0:
                    non_zero_eigenvalues = eigenvalues
                    num_eigenvalues = len(non_zero_eigenvalues)
            else:
                non_zero_eigenvalues = eigenvalues
                num_eigenvalues = len(non_zero_eigenvalues)

            # Marchenko-Pastur (MP) bulk edge calculation
            # The MP law describes the eigenvalue distribution of random matrices
            # sigma_mp estimates the scale parameter from the empirical eigenvalue distribution
            sigma_mp = torch.sqrt(torch.mean(non_zero_eigenvalues)).item()
            
            # Calculate the theoretical bulk edge using MP law
            # bulk_max = sigma^2 * (1 + sqrt(1/Q))^2 where Q = N/M is the aspect ratio
            bulk_max = (sigma_mp * (1 + 1 / torch.sqrt(torch.tensor(Q)))) ** 2

            # Apply Tracy-Widom correction for finite matrix effects
            if apply_tracy_widom_correction:
                # Tracy-Widom correction accounts for finite size effects in random matrix theory
                # The correction scales with matrix size and bulk edge value
                tracy_widom_correction = (
                    1
                    / torch.sqrt(torch.tensor(Q))
                    * torch.pow(torch.tensor(bulk_max), 2 / 3)
                    * torch.pow(torch.tensor(M), -2 / 3)
                )
                bulk_edge_corrected = bulk_max + torch.sqrt(tracy_widom_correction)
            else:
                bulk_edge_corrected = bulk_max

            # Count eigenvalues above the bulk edge (spikes)
            # These represent structured, non-random components in the weight matrix
            num_spikes = torch.sum(non_zero_eigenvalues > bulk_edge_corrected).item()

            # Compute log eigenvalues for power law fitting (add eps to prevent log(0))
            log_eigenvalues = torch.log(non_zero_eigenvalues + eps)

            # Power Law Fitting using Hill Estimator
            # The Hill estimator fits a power law to the tail of the eigenvalue distribution
            # This helps identify the degree of heavy-tailedness in the spectral density
            
            if power_law_fitting_method == "median":
                # Simple median-based approach for xmin selection
                xmin_index = int(len(non_zero_eigenvalues) / xmin_position)
                xmin_value = non_zero_eigenvalues[xmin_index]
                n_tail = float(num_eigenvalues - xmin_index)
                sequence = torch.arange(n_tail).cuda()
                
                # Hill estimator formula: alpha = 1 + n / sum(log(xi/xmin))
                hill_alpha = 1 + n_tail / (
                    torch.sum(log_eigenvalues[xmin_index:]) - n_tail * log_eigenvalues[xmin_index] + eps
                )
                
                # Kolmogorov-Smirnov test for goodness of fit
                ks_statistic = torch.max(
                    torch.abs(
                        1
                        - (non_zero_eigenvalues[xmin_index:] / (xmin_value + eps)) ** (-hill_alpha + 1)
                        - sequence / (n_tail + eps)
                    )
                )
            else:
                # Search over multiple xmin values to find the best power law fit
                alphas = torch.zeros(num_eigenvalues - 1)
                ks_statistics = torch.ones(num_eigenvalues - 1)
                
                # Fix-finger method uses histogram to constrain xmin search range
                if power_law_fitting_method == "fix-finger":
                    # Create histogram of log eigenvalues to find modal region
                    hist_log_eigenvalues = torch.log10(non_zero_eigenvalues + eps)
                    min_log, max_log = hist_log_eigenvalues.min(), hist_log_eigenvalues.max()
                    histogram_counts = torch.histc(hist_log_eigenvalues, histogram_bins, min=min_log, max=max_log)
                    histogram_boundaries = torch.linspace(min_log, max_log, histogram_bins + 1)
                    
                    # Find peak of histogram to set xmin constraints
                    peak_index = torch.argmax(histogram_counts)
                    xmin_peak = 10 ** histogram_boundaries[peak_index]
                    xmin_min_threshold = torch.log10(0.95 * xmin_peak + eps)
                    xmin_max_threshold = 1.5 * xmin_peak

                # Test each potential xmin value
                for i, xmin_candidate in enumerate(non_zero_eigenvalues[:-1]):
                    # Apply fix-finger constraints if specified
                    if power_law_fitting_method == "fix-finger":
                        if xmin_candidate < xmin_min_threshold:
                            continue
                        if xmin_candidate > xmin_max_threshold:
                            break

                    # Calculate Hill estimator for this xmin
                    n_tail = float(num_eigenvalues - i)
                    sequence = torch.arange(n_tail).cuda()
                    alpha = 1 + n_tail / (
                        torch.sum(log_eigenvalues[i:]) - n_tail * log_eigenvalues[i] + eps
                    )
                    alphas[i] = alpha
                    
                    # Calculate goodness of fit (KS statistic) only for valid alpha values
                    if alpha > 1:
                        ks_statistics[i] = torch.max(
                            torch.abs(
                                1
                                - (non_zero_eigenvalues[i:] / (xmin_candidate + eps)) ** (-alpha + 1)
                                - sequence / (n_tail + eps)
                            )
                        )

                # Select the alpha and D values corresponding to the best fit (minimum KS statistic)
                best_fit_index = torch.argmin(ks_statistics)
                hill_alpha = alphas[best_fit_index]
                ks_statistic = ks_statistics[best_fit_index]

            # Convert to scalar values and compute scaled alpha
            hill_alpha = hill_alpha.item()
            ks_statistic = ks_statistic.item()
            # Alpha-hat combines the power law exponent with the spectral norm scale
            alpha_hat = hill_alpha * np.log10(spectral_norm + eps)

            # Store analysis results for this layer
            results["layer_index"].append(layer_index)
            results["layer_name"].append(layer_name)
            results["alpha_hill"].append(hill_alpha)
            results["spectral_norm"].append(spectral_norm)
            results["D"].append(ks_statistic)
            results["eigs"].append(eigenvalues.detach().cpu().numpy())
            results["norm"].append(frobenius_norm)
            results["alphahat"].append(alpha_hat)
            results["num_spikes"].append(num_spikes)
    
    # Convert results to DataFrame for easy manipulation and analysis
    results_dataframe = pd.DataFrame(results)
    return results_dataframe


def compute_adaptive_lora_ranks(
    model,
    layer_selection_percentile=0.5,
    minimum_rank=4,
    maximum_rank=64,
    rank_scaling_method="linear",
    alpha_scaling_factor=2,
    hill_weight=0.7,
    mp_spikes_weight=0.3,
):
    """
    Computes adaptive LoRA ranks for neural network layers based on spectral analysis.
    
    This function performs a two-stage process:
    1. Analyzes the spectral properties of all layers using Hill estimator and MP theory
    2. Filters layers and assigns adaptive ranks based on a weighted scoring function
    
    The scoring function combines:
    - Hill alpha (power law exponent): measures heavy-tailedness of eigenvalue distribution
    - MP spikes count: measures number of eigenvalues above random matrix bulk edge
    
    Args:
        model (torch.nn.Module): PyTorch neural network model to analyze.
        layer_selection_percentile (float): Fraction of layers to select for LoRA adaptation 
            (e.g., 0.5 for top 50% of layers). Defaults to 0.5.
        minimum_rank (int): Minimum LoRA rank to assign. Defaults to 4.
        maximum_rank (int): Maximum LoRA rank to assign. Defaults to 64.
        rank_scaling_method (str): Method for scaling ranks between min and max. 
            Options: "linear", "log", "sqrt". Defaults to "linear".
        alpha_scaling_factor (float): Multiplicative factor for computing LoRA alpha 
            from rank (alpha = rank * factor). Defaults to 2.
        hill_weight (float): Weight for Hill alpha in the composite score. Defaults to 0.7.
        mp_spikes_weight (float): Weight for MP spikes count in the composite score. Defaults to 0.3.

    Returns:
        pandas.DataFrame: DataFrame containing filtered layers with assigned ranks and alphas:
            - layer_index: Sequential index of the layer
            - layer_name: Name of the layer
            - alpha_hill: Power law exponent from Hill estimator
            - alpha_hill_normalized: Normalized Hill alpha (0-1 scale)
            - num_spikes: Number of eigenvalues above MP bulk edge
            - num_spikes_normalized: Normalized spike count (0-1 scale, inverted)
            - weighted_score: Composite score combining Hill alpha and MP spikes
            - rank_pattern: Assigned LoRA rank for this layer
            - alpha_pattern: Assigned LoRA alpha for this layer
            - spectral_norm, D, eigs, norm, alphahat: Additional spectral metrics
    
    Notes:
        - Lower weighted scores indicate layers more suitable for low-rank adaptation
        - MP spikes are inverted in scoring (fewer spikes = higher adaptability)
        - Hill alpha is used directly (higher alpha = more heavy-tailed = more adaptable)
        - The function automatically handles normalization and scaling of metrics
    """
    # Step 1: Perform spectral analysis on all layers
    spectral_analysis_results = spectral_analysis_for_adaptive_lora(model)
    
    # Step 2: Normalize spectral metrics for fair comparison across layers
    eps = 1e-8  # Small epsilon to handle edge cases where min == max
    
    # Normalize Hill alpha: higher alpha = more heavy-tailed = better for low-rank adaptation
    alpha_min = spectral_analysis_results["alpha_hill"].min()
    alpha_max = spectral_analysis_results["alpha_hill"].max()
    spectral_analysis_results["alpha_hill_normalized"] = (
        spectral_analysis_results["alpha_hill"] - alpha_min
    ) / (alpha_max - alpha_min + eps)
    
    # Normalize and invert MP spikes: fewer spikes = more compressible = better for LoRA
    # Inversion: high spike count -> low score (less suitable for LoRA)
    spikes_min = spectral_analysis_results["num_spikes"].min()
    spikes_max = spectral_analysis_results["num_spikes"].max()
    spectral_analysis_results["num_spikes_normalized"] = 1 - (
        (spectral_analysis_results["num_spikes"] - spikes_min)
        / (spikes_max - spikes_min + eps)
    )

    # Step 3: Compute composite weighted score for layer ranking
    # Lower scores indicate layers more suitable for low-rank adaptation
    spectral_analysis_results["weighted_score"] = (
        spectral_analysis_results["alpha_hill_normalized"] * hill_weight
    ) + (spectral_analysis_results["num_spikes_normalized"] * mp_spikes_weight)

    # Step 4: Select top percentile of layers based on weighted score
    num_layers_to_select = int(len(spectral_analysis_results) * layer_selection_percentile)
    # Get indices of layers with smallest weighted scores (most suitable for LoRA)
    selected_layer_indices = spectral_analysis_results["weighted_score"].nsmallest(num_layers_to_select).index
    filtered_results = spectral_analysis_results.loc[selected_layer_indices]

    # Step 5: Assign adaptive ranks based on weighted scores
    min_score = filtered_results["weighted_score"].min()
    max_score = filtered_results["weighted_score"].max()
    
    # Scale ranks according to the specified scaling method
    if rank_scaling_method == "linear":
        # Linear interpolation between min and max rank
        filtered_results["rank_pattern"] = minimum_rank + (maximum_rank - minimum_rank) * (
            filtered_results["weighted_score"] - min_score
        ) / (max_score - min_score + eps)
    elif rank_scaling_method == "log":
        # Logarithmic scaling for more gradual rank assignment
        filtered_results["rank_pattern"] = minimum_rank + (maximum_rank - minimum_rank) * (
            np.log1p(filtered_results["weighted_score"]) - np.log1p(min_score)
        ) / (np.log1p(max_score) - np.log1p(min_score) + eps)
    elif rank_scaling_method == "sqrt":
        # Square root scaling for intermediate between linear and log
        filtered_results["rank_pattern"] = minimum_rank + (maximum_rank - minimum_rank) * (
            np.sqrt(filtered_results["weighted_score"]) - np.sqrt(min_score)
        ) / (np.sqrt(max_score) - np.sqrt(min_score) + eps)

    # Round ranks to integers and ensure they're within bounds
    filtered_results["rank_pattern"] = filtered_results["rank_pattern"].round().astype(int)
    filtered_results["rank_pattern"] = filtered_results["rank_pattern"].clip(minimum_rank, maximum_rank)

    # Step 6: Compute corresponding LoRA alpha values
    # LoRA alpha controls the scaling of the low-rank adaptation
    filtered_results["alpha_pattern"] = filtered_results["rank_pattern"] * alpha_scaling_factor
    
    # Step 7: Sort results by layer index for consistent ordering
    filtered_results = filtered_results.sort_values(by="layer_index")
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
    return filtered_results


def get_adaptive_lora_config(
    model,
    layer_selection_percentile=0.5,
    minimum_rank=4,
    maximum_rank=64,
    rank_scaling_method="linear",
    alpha_scaling_factor=2,
    hill_weight=0.7,
    mp_spikes_weight=0.3,
):
    """
    Generates adaptive LoRA configuration including rank patterns, alpha patterns, and target modules.
    
    This is the main interface function for the Adaptive Rank LoRA system. It performs spectral
    analysis on the model and returns the configuration needed for PEFT (Parameter Efficient Fine-Tuning).
    
    Args:
        model (torch.nn.Module): PyTorch neural network model to analyze.
        layer_selection_percentile (float): Fraction of layers to select for LoRA adaptation.
        minimum_rank (int): Minimum LoRA rank to assign.
        maximum_rank (int): Maximum LoRA rank to assign.
        rank_scaling_method (str): Method for scaling ranks ("linear", "log", "sqrt").
        alpha_scaling_factor (float): Factor for computing LoRA alpha from rank.
        hill_weight (float): Weight for Hill alpha in the composite score.
        mp_spikes_weight (float): Weight for MP spikes count in the composite score.
    
    Returns:
        tuple: A tuple containing:
            - rank_pattern (dict): Dictionary mapping layer names to LoRA ranks
            - alpha_pattern (dict): Dictionary mapping layer names to LoRA alphas  
            - target_modules_regex (str): Regex pattern for matching target modules in PEFT
    
    Example:
        >>> rank_pattern, alpha_pattern, target_regex = get_adaptive_lora_config(model)
        >>> peft_config = LoraConfig(
        ...     target_modules=target_regex,
        ...     rank_pattern=rank_pattern,
        ...     alpha_pattern=alpha_pattern
        ... )
    """
    # Compute adaptive ranks and alphas for selected layers
    layer_analysis_results = compute_adaptive_lora_ranks(
        model,
        layer_selection_percentile=layer_selection_percentile,
        minimum_rank=minimum_rank,
        maximum_rank=maximum_rank,
        rank_scaling_method=rank_scaling_method,
        alpha_scaling_factor=alpha_scaling_factor,
        hill_weight=hill_weight,
        mp_spikes_weight=mp_spikes_weight,
    )
    
    # Extract rank pattern as a dictionary: {layer_name: rank}
    rank_pattern = (
        layer_analysis_results[["layer_name", "rank_pattern"]]
        .set_index("layer_name")
        .to_dict()["rank_pattern"]
    )
    
    # Extract alpha pattern as a dictionary: {layer_name: alpha}
    alpha_pattern = (
        layer_analysis_results[["layer_name", "alpha_pattern"]]
        .set_index("layer_name")
        .to_dict()["alpha_pattern"]
    )
    
    # Create regex pattern for PEFT target_modules by joining all layer names
    # This allows PEFT to automatically match the selected layers
    target_modules_regex = "|".join(rank_pattern.keys())
    
    return rank_pattern, alpha_pattern, target_modules_regex
