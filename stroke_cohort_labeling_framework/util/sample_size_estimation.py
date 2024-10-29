from scipy.stats import norm


def required_sample_size_1(N, Z, p, E):
    """
    Calculate required sample size for a population.

    Parameters:
    N (int): Total population size
    Z (float): Z-value (e.g., 1.96 for 95% confidence)
    p (float): Estimated proportion of population (use 0.5 if unknown)
    E (float): Margin of error (e.g., 0.05 for 5%)

    Returns:
    int: Required sample size
    """
    X = Z**2 * p * (1 - p) / E**2
    n = N * X / ((N - 1) * E**2 + X) 
    # Return the required sample size rounded up to the nearest whole number
    return int(round(n))


# this only works if we know an estimated AUC
def required_sample_size_roc(auc, power, alpha, ratio, auc_0=0.5):
    """
    Calculate sample size per group for given AUC, power, significance level, and ratio.

    Parameters:
    auc (float): Expected AUC for the study
    power (float): Desired power
    alpha (float): Significance level
    ratio (float): Ratio of negative to positive cases
    auc_0 (float): AUC under the null hypothesis

    Returns:
    int: Required sample size per group
    """
    # Validate the inputs
    if auc <= auc_0 or auc >= 1.0:
        raise ValueError("Expected AUC should be greater than Null AUC and less than 1.0")

    # Get the z-scores for the desired power and significance level
    z_beta = norm.ppf(power)
    z_alpha = norm.ppf(1 - alpha / 2)
    
    # Adjust the formula to calculate sample size
    q1 = auc / (2 - auc)
    q0 = (2 * auc**2) / (1 + auc)
    
    q1_0 = auc_0 / (2 - auc_0)
    q0_0 = (2 * auc_0**2) / (1 + auc_0)

    numerator = (z_alpha * (q0 + q1) + z_beta)**2 * (q1 * (1 - q1) / q1_0 + q0 * (1 - q0) / q0_0)
    denominator = (auc - auc_0)**2
    
    # Calculate the sample size per group
    n_per_group = numerator / denominator
    n_positive = n_per_group / (1 + ratio)
    n_negative = n_positive * ratio
    
    return int(n_positive), int(n_negative)

alpha = 0.05
power = 0.99
auc = 0.975
ratio = 1
auc_0 = 0.9
print(required_sample_size_roc(auc, power, alpha, ratio, auc_0=auc_0))