import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Any
import logging
from collections import deque
from entropy_buffer import EntropyBuffer
import time
import random

logger = logging.getLogger(__name__)

def compute_surprisal(cdf_value: float) -> float:
    """
    Calculate surprisal value based on CDF value.
    Surprisal measures how unexpected an event is, defined as log2(1/p).
    
    Args:
        cdf_value: Cumulative distribution function value (probability)
        
    Returns:
        Surprisal value in bits
    """
    if cdf_value <= 0 or cdf_value >= 1.0:
        return 0.0
    return math.log2(1.0 / cdf_value)

def find_optimal_bound_size(predictability: float, regime_adaptive: bool = True) -> Tuple[int, Dict[str, float]]:
    """
    Calculate the optimal bound size based on input predictability.
    Implements Wilber's regime-specific scaling factors for different predictability ranges.
    
    Args:
        predictability: Predictability of the input bits (0.5 <= p <= 1.0)
        regime_adaptive: Whether to use predictability regime-specific scaling
        
    Returns:
        Tuple of (optimal_bound_size, metrics)
    """
    # Convert to effect size (ES = 2p - 1)
    effect_size = 2 * predictability - 1
    
    # Ensure predictability is in valid range
    if predictability < 0.5 or predictability > 1.0:
        raise ValueError(f"Predictability must be in range [0.5, 1.0], got {predictability}")
    
    # If no bias or negligible bias, use large bound
    if abs(effect_size) < 1e-10:
        return 101, {"regime": "zero_bias", "effect_size": 0, "scaling_factor": "n/a"}
    
    # Define bound size based on predictability regime if adaptive mode is enabled
    if regime_adaptive:
        # Identify the predictability regime
        if predictability <= 0.51:  # Near-random inputs (p ≈ 0.5)
            # Use scaling factor: n ≈ 1/(2p-1)²
            bound_size = int(1.0 / (effect_size ** 2))
            regime = "near_random"
            # Cap at reasonable maximum
            bound_size = min(bound_size, 300)
            # Ensure minimum bound size of 101 for near-random regime
            bound_size = max(bound_size, 101)
            
        elif predictability <= 0.55:  # Moderate bias (0.52 < p < 0.55)
            # Use scaling factor: n ≈ 1/(2p-1)
            bound_size = int(1.0 / abs(effect_size))
            regime = "moderate_bias"
            # Cap between 31-50 for moderate bias
            bound_size = max(31, min(bound_size, 50))
            
        else:  # Strong bias (p > 0.55)
            # Use scaling factor: n ≈ 1/√(2p-1)
            bound_size = int(1.0 / math.sqrt(abs(effect_size)))
            regime = "strong_bias"
            # Cap between 15-30 for strong bias
            bound_size = max(15, min(bound_size, 30))
            
    else:
        # Original method: Find bound size that optimizes statistical efficiency
        bound_size = 31  # Default medium bound size
        max_efficiency = 0
        
        # Test bound sizes from 11 to 101 in steps of 2
        for n in range(11, 102, 2):
            # Calculate output probability
            if predictability > 0.5:
                q_over_p = (1 - predictability) / predictability
                p_out = 1 / (1 + q_over_p ** n)
            else:
                p_over_q = predictability / (1 - predictability)
                p_out = p_over_q ** n / (1 + p_over_q ** n)
                
            # Calculate expected steps to bound
            if abs(effect_size) > 1e-10:
                denom = abs(effect_size) * (1 + (1-predictability)/predictability ** n)
                N = n * (1 - ((1-predictability)/predictability) ** n) / denom
            else:
                N = n ** 2  # For unbiased case
                
            # Calculate amplification
            if abs(effect_size) > 1e-10:
                amplification = (2 * p_out - 1) / effect_size
            else:
                amplification = 1.0
                
            # Calculate efficiency
            efficiency = amplification ** 2 / N
            
            if efficiency > max_efficiency:
                max_efficiency = efficiency
                bound_size = n
                
        regime = "efficiency_optimized"
    
    # Ensure bound size is odd (conventional for RWBAs)
    if bound_size % 2 == 0:
        bound_size += 1
        
    # Calculate metrics for the selected bound size
    if predictability > 0.5:
        q_over_p = (1 - predictability) / predictability
        p_out = 1 / (1 + q_over_p ** bound_size)
    else:
        p_over_q = predictability / (1 - predictability)
        p_out = p_over_q ** bound_size / (1 + p_over_q ** bound_size)
        
    # Calculate expected steps to bound
    if abs(effect_size) > 1e-10:
        denom = abs(effect_size) * (1 + (1-predictability)/predictability ** bound_size)
        N = bound_size * (1 - ((1-predictability)/predictability) ** bound_size) / denom
    else:
        N = bound_size ** 2  # For unbiased case
        
    # Calculate amplification
    if abs(effect_size) > 1e-10:
        amplification = (2 * p_out - 1) / effect_size
    else:
        amplification = 1.0
        
    # Calculate efficiency
    efficiency = amplification ** 2 / N
    
    # Return bound size and metrics
    metrics = {
        "regime": regime,
        "effect_size": effect_size,
        "expected_output_probability": p_out,
        "expected_steps_to_bound": N,
        "amplification": amplification,
        "efficiency": efficiency
    }
    
    logger.info(f"Selected bound size {bound_size} for predictability {predictability:.6f} "
               f"(regime: {regime}, efficiency: {efficiency:.6f})")
    
    return bound_size, metrics

def calculate_output_probability(predictability: float, bound_size: int) -> float:
    """
    Calculate the expected output probability for a given input predictability and bound size.
    
    Args:
        predictability: Input probability (0.5 <= p <= 1.0)
        bound_size: Size of the bound
        
    Returns:
        Expected output probability
    """
    # Handle edge cases
    if predictability <= 0.5:
        return 0.5
    if predictability >= 1.0:
        return 1.0
        
    # Calculate output probability using Wilber's formula
    q_over_p = (1 - predictability) / predictability
    p_out = 1 / (1 + q_over_p ** bound_size)
    
    return p_out

class WilberRWBA:
    """
    Pure implementation of Scott Wilber's Random Walk Bias Amplifier (RWBA) algorithm.
    This implementation strictly follows the mathematical relationships described in
    "Advances in Mind-Matter Interaction Technology: Is 100 Percent Effect Size Possible?"
    and related papers.
    
    Enhanced with:
    - Time-to-bound weighting: Faster walks receive higher weights
    - Partial-boundary logging: Tracks intermediate positions for early pattern detection
    - Early lead detection: Identifies strong initial biases
    - Dynamic bound adjustment: Support for real-time bound size recalibration
    - Rolling hit rate tracking: For adaptive Bayesian updating
    - Regime-specific bound optimization: Adapts bound size to predictability regime
    - Surprisal-based weighting: Dynamically adjusts bound size based on statistical rarity
    """
    
    def __init__(self, bound_size: int = 31, entropy_source: Optional[EntropyBuffer] = None,
                hit_rate_window: int = 100, weight_window: int = 50, 
                regime_adaptive: bool = True, surprisal_adaptive: bool = True):
        """
        Initialize RWBA with specified bound size.
        
        Args:
            bound_size: Number of positions from center to bound (default: 31)
            entropy_source: Optional entropy source (EntropyBuffer)
            hit_rate_window: Number of recent trials to calculate rolling hit rate
            weight_window: Window size for tracking weighted hit rates
            regime_adaptive: Whether to use predictability regime-specific scaling for bound size
            surprisal_adaptive: Whether to enable surprisal-based dynamic bound adjustment
        """
        self._bound_size = bound_size
        self.entropy_source = entropy_source
        self.position = 0
        self.steps_taken = 0
        self.total_steps = 0
        self.hit_upper_count = 0
        self.hit_lower_count = 0
        self.trial_count = 0
        self.regime_adaptive = regime_adaptive
        self.surprisal_adaptive = surprisal_adaptive
        
        # Enhanced logging for visualization and advanced analysis
        self.history = []  # Optional position history for visualization
        self.save_history = False  # Set to True to enable position history logging
        
        # Path tracking for partial boundary analysis
        self.path_points = []  # Periodic snapshots of position
        self.path_interval = 10  # Record position every N steps
        self.early_lead = False  # Flag for detecting strong initial movements
        self.early_lead_threshold = 0.4  # Fraction of bound size considered a "lead"
        self.early_lead_bonus = 1.1  # Multiplier for trials with early leads (10% bonus)
        
        # Parameters for time-to-bound weighting
        self.predicted_avg_steps = bound_size ** 2  # Initial prediction based on unbiased walk
        self.alpha = 0.5  # Slope factor for time weighting (0.3-0.5 recommended)
        self.use_rolling_avg = True  # Whether to update the predicted average over time
        self.rolling_avg_factor = 0.9  # How much to weight existing avg vs. new data (0.9 = 90% old, 10% new)
        
        # Parameters for advanced weighting
        self.use_time_weighting = True  # Whether to weight trials by time-to-bound
        self.use_early_lead_detection = True  # Whether to apply bonus for early leads
        self.use_partial_boundary = True  # Whether to track intermediate positions
        
        # Dynamic recalibration tracking
        self.recalibration_history = []  # Track bound size changes over time
        
        # Rolling hit rate tracking for adaptive Bayesian updating
        self.hit_rate_window = hit_rate_window
        self.recent_trials = deque(maxlen=hit_rate_window)  # Store recent trial results
        self.recent_weights = deque(maxlen=weight_window)   # Store recent trial weights
        self.recent_weighted_results = deque(maxlen=weight_window)  # Store weighted results
        
        # Surprisal-based weighting
        self.surprisal_values = deque(maxlen=weight_window)  # Track surprisal values for adaptive bounds
        self.step_counts = deque(maxlen=hit_rate_window)  # Track step counts for CDF estimation
        self.surprisal_smoothing_factor = 0.8  # Smoothing factor for bound size updates (0.7-0.9 recommended)
        self.previous_bound_size = bound_size  # Store previous bound size for smoothing
        
        # CDF approximation parameters
        self.cdf_samples = {}  # Dict to track empirical CDF {steps: cdf_value}
        self.use_empirical_cdf = True  # Whether to use empirical CDF or theoretical one
        
        logger.info(f"Initializing enhanced WilberRWBA with bound_size={bound_size}, "
                   f"time_weighting={'enabled' if self.use_time_weighting else 'disabled'}, "
                   f"partial_boundary={'enabled' if self.use_partial_boundary else 'disabled'}, "
                   f"hit_rate_window={hit_rate_window}, "
                   f"regime_adaptive={'enabled' if regime_adaptive else 'disabled'}, "
                   f"surprisal_adaptive={'enabled' if surprisal_adaptive else 'disabled'}")
    
    @property
    def bound_size(self) -> int:
        """Get current bound size."""
        return self._bound_size
        
    @property
    def configuration(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            "bound_size": self._bound_size,
            "hit_rate_window": self.hit_rate_window,
            "path_interval": self.path_interval,
            "time_weighting_enabled": self.use_time_weighting,
            "early_lead_detection": self.use_early_lead_detection,
            "partial_boundary_tracking": self.use_partial_boundary,
            "regime_adaptive": self.regime_adaptive,
            "surprisal_adaptive": self.surprisal_adaptive,
            "surprisal_smoothing_factor": self.surprisal_smoothing_factor
        }
        
    def update_bound_size(self, new_bound_size: int) -> None:
        """
        Update bound size. This is called by the recalibration system or
        surprisal-based adaptation.
        
        Args:
            new_bound_size: New bound size value
        """
        if new_bound_size < 7:
            logger.warning(f"Requested bound size {new_bound_size} is too small, using minimum of 7")
            new_bound_size = 7
            
        # Record the change
        self.recalibration_history.append({
            "trial": self.trial_count,
            "old_bound": self._bound_size,
            "new_bound": new_bound_size,
            "timestamp": time.time()
        })
        
        # Update the bound size
        self.previous_bound_size = self._bound_size
        self._bound_size = new_bound_size
        
        # Update predicted average steps based on new bound size
        self.predicted_avg_steps = new_bound_size ** 2
        
    def get_recalibration_history(self) -> List[Dict[str, Any]]:
        """
        Get history of bound size recalibrations.
        
        Returns:
            List of recalibration events
        """
        return self.recalibration_history
    
    def recalibrate_bound_size(self, new_predictability: float) -> bool:
        """
        Recalibrate bound size based on new predictability value.
        Uses regime-specific scaling if enabled.
        
        Args:
            new_predictability: New predictability value
            
        Returns:
            True if bound size was changed, False otherwise
        """
        old_bound = self.bound_size
        new_bound, metrics = find_optimal_bound_size(new_predictability, self.regime_adaptive)
        
        if new_bound != old_bound:
            logger.info(f"Recalibrating bound size from {old_bound} to {new_bound} based on "
                       f"predictability {new_predictability:.6f} (regime: {metrics['regime']})")
            self.update_bound_size(new_bound)
            return True
            
        return False
    
    def get_empirical_cdf(self, steps: int) -> float:
        """
        Get empirical CDF value for a given number of steps.
        
        Args:
            steps: Number of steps to reach bound
            
        Returns:
            Empirical CDF value (probability that X ≤ steps)
        """
        if not self.step_counts:
            # If no data, use theoretical CDF for unbiased walk
            return self._get_theoretical_cdf(steps)
            
        # Calculate empirical CDF
        count = sum(1 for s in self.step_counts if s <= steps)
        return count / len(self.step_counts)
    
    def _get_theoretical_cdf(self, steps: int) -> float:
        """
        Get theoretical CDF for unbiased random walk.
        This is an approximation based on the normal distribution.
        
        Args:
            steps: Number of steps to reach bound
            
        Returns:
            Theoretical CDF value
        """
        # For an unbiased walk, expected steps is bound_size^2
        expected_steps = self._bound_size ** 2
        
        # Use normal approximation for large number of steps
        if steps <= 0:
            return 0.0
            
        z = (steps - expected_steps) / (expected_steps ** 0.5)
        # Simple approximation of normal CDF
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return cdf
    
    def apply_surprisal_adjustment(self, steps: int) -> None:
        """
        Adjust bound size based on surprisal value of the observed number of steps.
        
        Args:
            steps: Number of steps taken to reach the bound
        """
        if not self.surprisal_adaptive:
            return
            
        # Store step count for future CDF estimation
        self.step_counts.append(steps)
        
        # Calculate CDF and surprisal
        cdf_value = self.get_empirical_cdf(steps) if self.use_empirical_cdf else self._get_theoretical_cdf(steps)
        surprisal = compute_surprisal(cdf_value)
        
        # Store surprisal for tracking
        self.surprisal_values.append(surprisal)
        
        # Calculate adjustment based on surprisal
        if surprisal > 4.0:
            # Rare event (highly unexpected) - increase bound size for greater sensitivity
            adjustment_factor = 1.1  # 10% increase
        elif surprisal < 2.0:
            # Common event (highly expected) - decrease bound size for quicker decisions
            adjustment_factor = 0.9  # 10% decrease
        else:
            # Moderate surprisal - maintain current bound size
            adjustment_factor = 1.0
            
        # Calculate new bound size
        new_bound_size = int(self._bound_size * adjustment_factor)
        
        # Apply smoothing to prevent oscillations
        smoothed_bound_size = int(
            self.surprisal_smoothing_factor * self._bound_size + 
            (1 - self.surprisal_smoothing_factor) * new_bound_size
        )
        
        # Ensure bound is odd
        if smoothed_bound_size % 2 == 0:
            smoothed_bound_size += 1
            
        # Only update if the change is significant enough (at least 2)
        if abs(smoothed_bound_size - self._bound_size) >= 2:
            logger.info(f"Surprisal-based adjustment: Steps={steps}, CDF={cdf_value:.4f}, "
                       f"Surprisal={surprisal:.2f}, Old bound={self._bound_size}, "
                       f"New bound={smoothed_bound_size}")
            self.update_bound_size(smoothed_bound_size)
    
    def reset(self) -> None:
        """Reset the RWBA for a new trial."""
        self.position = 0
        self.steps_taken = 0
        self.early_lead = False
        self.path_points = []
        if self.save_history:
            self.history = []
            
    def step(self, direction: int) -> Tuple[bool, int]:
        """
        Step the random walker in the specified direction.
        
        Args:
            direction: Direction to step (1 for up, -1 for down, or 0 to stay)
            
        Returns:
            Tuple of (reached_bound, hit_boundary_value)
        """
        if abs(direction) > 1:
            direction = 1 if direction > 0 else -1
        
        self.position += direction
        self.steps_taken += 1
        self.total_steps += 1
        
        # Record history if enabled
        if self.save_history:
            self.history.append(self.position)
            
        # Record path points at regular intervals for partial boundary analysis
        if self.use_partial_boundary and self.steps_taken % self.path_interval == 0:
            self.path_points.append(self.position)
            
        # Check for early lead
        if (not self.early_lead and self.use_early_lead_detection and 
            self.steps_taken >= self.path_interval and
            abs(self.position) >= self._bound_size * self.early_lead_threshold):
            self.early_lead = True
            
        # Check if we've reached a bound
        if abs(self.position) >= self._bound_size:
            boundary_value = 1 if self.position >= self._bound_size else -1
            return True, boundary_value
            
        return False, 0
    
    def run_trial(self, outcome_hint: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete RWBA trial until a bound is reached.
        
        Args:
            outcome_hint: Optional hint for expected outcome (1 for upper bound, -1 for lower)
            
        Returns:
            Dictionary with trial results
        """
        self.reset()
        self.trial_count += 1
        start_time = time.time()
        
        reached_bound = False
        result = 0
        
        while not reached_bound:
            # Get next bit from entropy source if available, otherwise random
            if self.entropy_source:
                bit = self.entropy_source.get_bit()
            else:
                bit = random.randint(0, 1)
                
            # Convert bit to direction (0 -> -1, 1 -> 1)
            direction = 2 * bit - 1
            
            # Step the random walker
            reached_bound, bound_value = self.step(direction)
            
            if reached_bound:
                result = bound_value
                if result == 1:
                    self.hit_upper_count += 1
                else:
                    self.hit_lower_count += 1
                    
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate time-based weight
        time_weight = 1.0
        if self.use_time_weighting:
            # Faster walks get higher weights
            if self.steps_taken < self.predicted_avg_steps:
                # Linear scaling between 1.0 and max_weight
                max_weight = 1.0 + self.alpha
                time_weight = 1.0 + self.alpha * (1.0 - self.steps_taken / self.predicted_avg_steps)
            
            # Update rolling average if enabled
            if self.use_rolling_avg:
                self.predicted_avg_steps = (
                    self.rolling_avg_factor * self.predicted_avg_steps + 
                    (1 - self.rolling_avg_factor) * self.steps_taken
                )
                
        # Apply early lead bonus if detected
        if self.early_lead and self.use_early_lead_detection:
            time_weight *= self.early_lead_bonus
            
        # Track recent trials
        self.recent_trials.append(result == 1)  # Store True for upper, False for lower
        self.recent_weights.append(time_weight)
        self.recent_weighted_results.append(time_weight if result == 1 else 0)
        
        # Apply surprisal-based bound adjustment
        self.apply_surprisal_adjustment(self.steps_taken)
        
        return {
            "result": result,
            "steps": self.steps_taken,
            "duration": duration,
            "path_points": self.path_points,
            "early_lead": self.early_lead,
            "weight": time_weight,
            "trial_number": self.trial_count
        }
    
    def get_rolling_hit_rate(self, weighted: bool = True) -> float:
        """
        Get rolling hit rate (upper bound hits) for adaptive belief updating.
        
        Args:
            weighted: Whether to use time-weighted hit rate
            
        Returns:
            Rolling hit rate as value between 0.0 and 1.0
        """
        if not self.recent_trials:
            return 0.5  # Default to unbiased if no data
            
        if weighted and self.use_time_weighting:
            # Use time-weighted hit rate
            if not self.recent_weighted_results or sum(self.recent_weights) == 0:
                return 0.5
                
            return sum(self.recent_weighted_results) / sum(self.recent_weights)
        else:
            # Use unweighted hit rate
            return sum(self.recent_trials) / len(self.recent_trials)
    
    def get_average_surprisal(self) -> float:
        """
        Get average surprisal value from recent trials.
        
        Returns:
            Average surprisal value
        """
        if not self.surprisal_values:
            return 0.0
            
        return sum(self.surprisal_values) / len(self.surprisal_values)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about RWBA performance.
        
        Returns:
            Dictionary of statistics
        """
        hit_rate = self.hit_upper_count / self.trial_count if self.trial_count > 0 else 0.5
        
        return {
            "bound_size": self._bound_size,
            "trials": self.trial_count,
            "hit_upper": self.hit_upper_count,
            "hit_lower": self.hit_lower_count,
            "hit_rate": hit_rate,
            "total_steps": self.total_steps,
            "avg_steps_per_trial": self.total_steps / self.trial_count if self.trial_count > 0 else 0,
            "rolling_hit_rate": self.get_rolling_hit_rate(),
            "rolling_hit_rate_unweighted": self.get_rolling_hit_rate(weighted=False),
            "avg_surprisal": self.get_average_surprisal()
        }


# Mathematical functions for RWBA parameter calculation
def calculate_expected_steps(p: float, bound_size: int) -> float:
    """
    Calculate expected number of steps to reach bound using Wilber's Equation 1.
    
    Args:
        p: Input probability (≥.5)
        bound_size: Number of positions to bound
        
    Returns:
        Expected number of steps
    """
    # Ensure p is valid
    p = max(0.5, min(p, 1.0))
    
    # Exact implementation of Wilber's Equation 1
    if abs(p - 0.5) < 1e-10:
        return bound_size * bound_size  # Eq 1b: n^2
    
    # Eq 1a: n * (1 - ((1-p)/p)^n) / ((2p-1) * (1 + ((1-p)/p)^n))
    ratio = (1-p)/p
    ratio_pow = ratio ** bound_size
    
    numerator = bound_size * (1 - ratio_pow)
    denominator = (2*p - 1) * (1 + ratio_pow)
    
    return numerator / denominator

def calculate_output_probability(p: float, bound_size: int) -> float:
    """
    Calculate expected output probability using Wilber's Equation 2.
    
    Args:
        p: Input probability (≥ 0.5)
        bound_size: Number of positions to bound
        
    Returns:
        Expected output probability
    """
    # Ensure p is valid
    p = max(0.5, min(p, 1.0))
    
    # Edge case
    if p == 0.5:
        return 0.5
    
    # Eq 2: 1 / (1 + ((1-p)/p)^n)
    term = ((1.0 - p) / p) ** bound_size
    return 1.0 / (1.0 + term)

def calculate_amplification_factor(p_in: float, p_out: float) -> float:
    """
    Calculate amplification factor using Wilber's Equation 3.
    
    Args:
        p_in: Input probability
        p_out: Output probability
        
    Returns:
        Amplification factor
    """
    # Ensure valid inputs
    p_in = max(0.5, min(p_in, 1.0))
    p_out = max(0.5, min(p_out, 1.0))
    
    # Handle special case
    if p_in == 0.5:
        return float('inf') if p_out > 0.5 else 0.0
    
    # Eq 3: (2*Pout - 1)/(2*p - 1)
    return (2.0 * p_out - 1.0) / (2.0 * p_in - 1.0)

def calculate_bound_size(p: float, target_pout: float) -> int:
    """
    Calculate bound size needed for desired output probability.
    
    Args:
        p: Input probability (> 0.5)
        target_pout: Desired output probability
        
    Returns:
        Required bound size (integer)
    """
    # Ensure valid inputs
    p = max(0.5, min(p, 1.0))
    target_pout = max(0.5, min(target_pout, 1.0))
    
    # Special cases
    if abs(p - 0.5) < 1e-10 or abs(target_pout - 0.5) < 1e-10:
        return 1  # Any bound size gives Pout = 0.5 when p = 0.5
    
    # Derived formula: n = Ln((1-Pout)/Pout) / Ln((1-p)/p)
    n = math.log((1.0 - target_pout) / target_pout) / math.log((1.0 - p) / p)
    
    # Round to nearest integer as per Wilber's practice
    return max(1, round(n))

def calculate_statistical_efficiency(p: float, bound_size: int) -> float:
    """
    Calculate statistical efficiency of the RWBA.
    
    Args:
        p: Input probability (≥ 0.5)
        bound_size: Number of positions to bound
        
    Returns:
        Statistical efficiency (0 to 1)
    """
    # Edge case
    if p == 0.5:
        return 0.0  # No effect, no efficiency
    
    # Calculate output probability
    p_out = calculate_output_probability(p, bound_size)
    
    # Calculate amplification
    amp = calculate_amplification_factor(p, p_out)
    
    # Calculate average steps
    avg_steps = calculate_expected_steps(p, bound_size)
    
    # Statistical Efficiency (SE) = Amp^2 / N
    return (amp * amp) / avg_steps

def find_optimal_bound_size(p: float, target_metric: str = "efficiency") -> Tuple[int, Dict]:
    """
    Find the optimal bound size for a given input probability.
    
    Args:
        p: Input probability (> 0.5)
        target_metric: Optimization target ('efficiency', 'amplification', 'output')
        
    Returns:
        Tuple of (optimal bound size, metrics dictionary)
    """
    # Ensure valid input
    p = max(0.5, min(p, 1.0))
    
    # No optimization possible at p=0.5
    if p == 0.5:
        return 1, {
            "bound_size": 1,
            "efficiency": 0.0,
            "amplification": 0.0,
            "output_probability": 0.5,
            "average_steps": 1.0
        }
    
    # Search range - based on Wilber's typical bounds
    max_bound = 1000
    best_bound = 1
    best_metric_value = 0.0
    
    # For each possible bound size
    results = {}
    for n in range(1, max_bound + 1):
        p_out = calculate_output_probability(p, n)
        amp = calculate_amplification_factor(p, p_out)
        avg_steps = calculate_expected_steps(p, n)
        efficiency = (amp * amp) / avg_steps
        
        # Store results
        results[n] = {
            "output_probability": p_out,
            "amplification": amp,
            "average_steps": avg_steps,
            "efficiency": efficiency
        }
        
        # Update best bound based on target metric
        metric_value = 0.0
        if target_metric == "efficiency":
            metric_value = efficiency
        elif target_metric == "amplification":
            metric_value = amp
        elif target_metric == "output":
            metric_value = p_out
            
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_bound = n
            
        # Break early if we've reached diminishing returns
        if n > 50 and results[n]["efficiency"] < 0.01 * results[best_bound]["efficiency"]:
            break
    
    # Return best bound and its metrics
    return best_bound, results[best_bound] 