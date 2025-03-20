import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime
import json
import os
import logging
import math
import time
from collections import defaultdict

from qng import QNG
from entropy_buffer import EntropyBuffer
from qng_preprocessor import QNGPreprocessor
from predictability_analyzer import PredictabilityAnalyzer
from wilber_rwba import WilberRWBA, calculate_output_probability, find_optimal_bound_size
from config import RecalibrationConfig, XorPreprocessingConfig

logger = logging.getLogger(__name__)

@dataclass
class TrialData:
    """Data structure for a single trial."""
    timestamp: datetime
    question: str
    direction: int  # -1: No, 1: Yes
    step_count: int
    result: int  # 0: No, 1: Yes
    weight: float  # Trial weight based on time-to-bound
    metrics: Dict[str, float]
    confidence: float
    beliefs: Dict[str, float]


class MEQASystem:
    """
    Mind-Energized Quantum Analysis (MEQA) System
    
    This class implements a complete prediction and analysis system based on 
    the Random Walk Bias Amplifier (RWBA) algorithm by Scott Wilber, with 
    additional Bayesian analysis and enhanced processing features.
    
    Key Features:
    - XOR preprocessing for non-predictable patterns 
    - Dynamic recalibration with stability controls
    - Multi-hypothesis support
    - Adaptive Bayesian updating based on predictability
    - Confidence thresholds for reliable decision-making
    - Surprisal-based weighting for bound size adaptation
    """
    
    def __init__(self, 
                 xor_window_size: int = 0,
                 bound_size: int = 31, 
                 trial_count: int = 500,
                 batch_size: int = 100,
                 prior_probability: float = 0.5,
                 recalibration_enabled: bool = True,
                 min_recalibration_interval: int = 100,
                 predictability_window: int = 1000,
                 recalibration_threshold: float = 0.01,
                 hysteresis_factor: float = 0.005,
                 bound_change_threshold: int = 10,
                 confidence_level: float = 0.95,
                 extension_enabled: bool = True,
                 min_confidence_threshold: float = 0.01,
                 decision_margin_threshold: float = 0.01,
                 max_auto_extension_trials: int = 500,
                 extension_batch_size: int = 100,
                 regime_adaptive_bounds: bool = True,
                 surprisal_adaptive_bounds: bool = True,
                 surprisal_smoothing_factor: float = 0.8):
        """
        Initialize the MEQA system.
        
        Args:
            xor_window_size: Window size for XOR preprocessing (0 = disabled)
            bound_size: Initial bound size for RWBA algorithm
            trial_count: Default number of trials to run
            batch_size: Number of trials to run in each batch
            prior_probability: Prior probability for Bayesian analysis
            recalibration_enabled: Enable dynamic recalibration
            min_recalibration_interval: Minimum trials before recalibration
            predictability_window: Window size for predictability calculation
            recalibration_threshold: Threshold for recalibration
            hysteresis_factor: Hysteresis factor to prevent oscillation
            bound_change_threshold: Minimum change in bound size to recalibrate
            confidence_level: Target confidence level for Bayesian analysis
            extension_enabled: Whether to enable automatic extension of trials
            min_confidence_threshold: Minimum confidence (p diff) to make decision
            decision_margin_threshold: Minimum margin between yes/no probs
            max_auto_extension_trials: Maximum additional trials for auto-extension
            extension_batch_size: Number of trials in each extension batch
            regime_adaptive_bounds: Whether to use predictability regime-specific bound scaling
            surprisal_adaptive_bounds: Whether to enable surprisal-based bound adaptation
            surprisal_smoothing_factor: Smoothing factor for surprisal-based bound updates (0.7-0.9)
        """
        # Initialize entropy sources
        self.entropy_buffer = EntropyBuffer(window_size=predictability_window)
        
        # Initialize RWBA
        self.rwba = WilberRWBA(bound_size=bound_size, 
                              entropy_source=self.entropy_buffer,
                              regime_adaptive=regime_adaptive_bounds,
                              surprisal_adaptive=surprisal_adaptive_bounds)
        
        # XOR preprocessing settings
        self.xor_window_size = xor_window_size
        self.xor_buffer = []
        
        # Trial settings
        self.trial_count = trial_count
        self.batch_size = batch_size
        self.trials_completed = 0
        
        # Bayesian settings
        self.prior_probability = prior_probability
        self.posterior_probability = prior_probability
        self.confidence_level = confidence_level
        
        # Dynamic recalibration settings
        self.recalibration_enabled = recalibration_enabled
        self.min_recalibration_interval = min_recalibration_interval
        self.predictability_window = predictability_window
        self.recalibration_threshold = recalibration_threshold
        self.hysteresis_factor = hysteresis_factor
        self.bound_change_threshold = bound_change_threshold
        self.last_recalibration_trial = 0
        self.regime_adaptive_bounds = regime_adaptive_bounds
        
        # Surprisal-based adaptation settings
        self.surprisal_adaptive_bounds = surprisal_adaptive_bounds
        self.surprisal_smoothing_factor = surprisal_smoothing_factor
        
        # Confidence threshold settings
        self.extension_enabled = extension_enabled
        self.min_confidence_threshold = min_confidence_threshold
        self.decision_margin_threshold = decision_margin_threshold
        self.max_auto_extension_trials = max_auto_extension_trials
        self.extension_batch_size = extension_batch_size
        
        # Session statistics
        self.session_start_time = None
        self.session_end_time = None
        self.session_duration = None
        self.session_stats = {}
        
        # Performance measurements
        self.performance_metrics = {
            "predictability_history": [],
            "bound_size_history": [],
            "trial_durations": [],
            "recalibrations": [],
            "posterior_history": [],
            "surprisal_history": []
        }
        
        logger.info(f"Initialized MEQA System with: "
                   f"bound_size={bound_size}, "
                   f"xor_window={xor_window_size}, "
                   f"recalibration={'enabled' if recalibration_enabled else 'disabled'}, "
                   f"regime_adaptive_bounds={'enabled' if regime_adaptive_bounds else 'disabled'}, "
                   f"surprisal_adaptive_bounds={'enabled' if surprisal_adaptive_bounds else 'disabled'}")
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update system configuration with new parameters.
        
        Args:
            config: Dictionary of parameters to update
        """
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Update RWBA if regime_adaptive_bounds or surprisal_adaptive_bounds was changed
        if "regime_adaptive_bounds" in config:
            self.rwba.regime_adaptive = self.regime_adaptive_bounds
            logger.info(f"Updated RWBA regime_adaptive to {self.regime_adaptive_bounds}")
            
        if "surprisal_adaptive_bounds" in config:
            self.rwba.surprisal_adaptive = self.surprisal_adaptive_bounds
            logger.info(f"Updated RWBA surprisal_adaptive to {self.surprisal_adaptive_bounds}")
            
        if "surprisal_smoothing_factor" in config:
            self.rwba.surprisal_smoothing_factor = self.surprisal_smoothing_factor
            logger.info(f"Updated RWBA surprisal_smoothing_factor to {self.surprisal_smoothing_factor}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current system configuration.
        
        Returns:
            Dictionary of current configuration parameters
        """
        return {
            "xor_window_size": self.xor_window_size,
            "bound_size": self.rwba.bound_size,
            "trial_count": self.trial_count,
            "batch_size": self.batch_size,
            "prior_probability": self.prior_probability,
            "recalibration_enabled": self.recalibration_enabled,
            "min_recalibration_interval": self.min_recalibration_interval,
            "predictability_window": self.predictability_window,
            "recalibration_threshold": self.recalibration_threshold,
            "hysteresis_factor": self.hysteresis_factor,
            "bound_change_threshold": self.bound_change_threshold,
            "confidence_level": self.confidence_level,
            "extension_enabled": self.extension_enabled,
            "min_confidence_threshold": self.min_confidence_threshold,
            "decision_margin_threshold": self.decision_margin_threshold,
            "max_auto_extension_trials": self.max_auto_extension_trials,
            "extension_batch_size": self.extension_batch_size,
            "regime_adaptive_bounds": self.regime_adaptive_bounds,
            "surprisal_adaptive_bounds": self.surprisal_adaptive_bounds,
            "surprisal_smoothing_factor": self.surprisal_smoothing_factor
        }

    def check_recalibration(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if recalibration is needed based on current predictability.
        Uses regime-specific bound optimization when enabled.
        
        Returns:
            Tuple of (recalibration_performed, metrics)
        """
        if not self.recalibration_enabled:
            return False, {"status": "recalibration_disabled"}
            
        # Skip if not enough trials since last recalibration
        trials_since_last = self.trials_completed - self.last_recalibration_trial
        if trials_since_last < self.min_recalibration_interval:
            return False, {"status": "too_soon", "trials_since_last": trials_since_last}
            
        # Get current predictability
        curr_predictability = self.entropy_buffer.get_predictability()
        
        # Skip if not enough data for reliable estimate
        if self.entropy_buffer.bits_processed < self.predictability_window / 2:
            return False, {"status": "insufficient_data", 
                          "bits_processed": self.entropy_buffer.bits_processed}
        
        # Calculate optimal bound size based on current predictability
        optimal_bound, bound_metrics = find_optimal_bound_size(
            curr_predictability, 
            regime_adaptive=self.regime_adaptive_bounds
        )
        
        # Check if bound size change is significant
        current_bound = self.rwba.bound_size
        bound_diff = abs(optimal_bound - current_bound)
        
        # Apply hysteresis to prevent oscillation
        if bound_diff <= self.bound_change_threshold:
            return False, {"status": "change_too_small", 
                          "current_bound": current_bound, 
                          "optimal_bound": optimal_bound,
                          "bound_diff": bound_diff,
                          "metrics": bound_metrics}
        
        # Update bound size
        self.rwba.update_bound_size(optimal_bound)
        self.last_recalibration_trial = self.trials_completed
        
        # Record recalibration
        recalibration_info = {
            "trial": self.trials_completed,
            "old_bound": current_bound,
            "new_bound": optimal_bound,
            "predictability": curr_predictability,
            "regime": bound_metrics["regime"],
            "amplification": bound_metrics["amplification"],
            "efficiency": bound_metrics["efficiency"],
            "method": "predictability_based"
        }
        
        self.performance_metrics["recalibrations"].append(recalibration_info)
        
        logger.info(f"Recalibrated bound size from {current_bound} to {optimal_bound} "
                   f"(predictability: {curr_predictability:.6f}, "
                   f"regime: {bound_metrics['regime']})")
                   
        return True, {
            "status": "recalibrated",
            "old_bound": current_bound,
            "new_bound": optimal_bound,
            "predictability": curr_predictability,
            "metrics": bound_metrics
        }

    def reset_beliefs(self, hypotheses: Optional[List[str]] = None) -> None:
        """
        Reset beliefs to initial state.
        
        Args:
            hypotheses: Optional list of hypothesis names (defaults to ["yes", "no"])
        """
        if hypotheses:
            self.supported_hypotheses = set(hypotheses)
        elif not self.supported_hypotheses or not self.multi_hypothesis:
            self.supported_hypotheses = {"yes", "no"}
            
        # Initialize with uniform prior
        prior = 1.0 / len(self.supported_hypotheses)
        self.current_beliefs = {h: prior for h in self.supported_hypotheses}
        
        logger.info(f"Reset beliefs to uniform distribution over {len(self.supported_hypotheses)} hypotheses")
    
    def recalibrate_system(self, force: bool = False) -> bool:
        """
        Perform dynamic recalibration of RWBA parameters if needed.
        Uses hysteresis control to prevent oscillations.
        
        Args:
            force: Force recalibration even if threshold not exceeded
            
        Returns:
            bool: True if system was recalibrated, False otherwise
        """
        if not self.recalibration_config.enabled and not force:
            return False
            
        # Check if it's time to recalibrate
        current_time = time.time()
        minutes_since_last = (current_time - self.last_recalibration_time) / 60
        
        if not force and minutes_since_last < self.recalibration_config.interval_minutes:
            logger.debug(f"Recalibration locked for {self.recalibration_config.interval_minutes - minutes_since_last:.1f} more minutes")
            return False
            
        # Check for significant predictability drift
        should_recalibrate, new_predictability = self.entropy_buffer.should_recalibrate()
        
        if force or should_recalibrate:
            if force:
                logger.info(f"Forced recalibration after {minutes_since_last:.1f} minutes")
                new_predictability = self.entropy_buffer.calculate_current_predictability()
            else:
                logger.info(f"Automatic recalibration triggered after {minutes_since_last:.1f} minutes")
                
            # Calculate new optimal bound size
            old_bound = self.rwba.bound_size
            new_bound, metrics = find_optimal_bound_size(new_predictability)
            
            # Update RWBA parameters
            if self.rwba.update_bound_size(new_bound):
                # Record recalibration event
                self.recalibration_events.append({
                    "timestamp": datetime.now(),
                    "old_predictability": self.predictability,
                    "new_predictability": new_predictability,
                    "old_bound": old_bound,
                    "new_bound": new_bound,
                    "forced": force,
                })
                
                # Update system parameters
                self.predictability = new_predictability
                p_out = calculate_output_probability(new_predictability, new_bound)
                
                # Update expected metrics
                self.expected_metrics = {
                    "input_probability": new_predictability,
                    "bound_size": new_bound,
                    "expected_output_probability": p_out,
                    "expected_effect_size": 2 * p_out - 1,
                }
                
                logger.info(f"Recalibrated: bound={old_bound}->{new_bound}, "
                           f"predictability={self.predictability:.6f}->{new_predictability:.6f}")
                           
                # Reset timer
                self.last_recalibration_time = current_time
                
                return True
                
        return False
        
    def ask_question(self, question: str, hypotheses: Optional[List[str]] = None, 
                    num_trials: Optional[int] = None, auto_extend: bool = True) -> Dict:
        """Ask a question and get answer.
        
        Args:
            question: Question to ask
            hypotheses: Optional list of hypothesis names (defaults to ["yes", "no"])
            num_trials: Optional number of trials to run
            auto_extend: Whether to automatically extend session for ambiguous results
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Question: {question}, hypotheses: {hypotheses}, trials: {num_trials}")
        
        # Reset beliefs for new question
        self.reset_beliefs(hypotheses)
        self.current_question = question
        
        # Determine number of trials
        if num_trials is None:
            # Default to a value based on predictability
            effect_size = 2 * self.predictability - 1
            if abs(effect_size) > 1e-6:
                num_trials = int(10 / (effect_size * effect_size))
            else:
                num_trials = 100
            
            # Cap at reasonable values
            num_trials = max(10, min(num_trials, 1000))
            
        logger.info(f"Running {num_trials} trials for question: {question}")
        
        # Recalibrate before starting a new question if configured
        if self.recalibration_config.force_at_new_question:
            self.recalibrate_system(force=True)
        
        # Run initial trials with periodic recalibration for long sessions
        trials_run = 0
        for i in range(num_trials):
            # Periodically check if recalibration is needed for long sessions
            if i > 0 and i % self.recalibration_config.check_frequency == 0:
                self.recalibrate_system()
                
            self.run_trial()
            trials_run += 1
            
        # Check if we've reached sufficient confidence
        answer, confidence, decision_state = self.get_answer_with_state()
        total_trials = trials_run
        
        # Auto-extend session if confidence is too low and auto_extend is enabled
        if auto_extend and decision_state == "insufficient_evidence":
            logger.info(f"Insufficient evidence after {trials_run} trials. Auto-extending session.")
            
            # Continue running trials in batches until reaching confidence threshold
            # or hitting the maximum extension limit
            extension_batches = 0
            max_extension_trials = self.max_auto_extension_trials
            
            while (decision_state == "insufficient_evidence" and 
                   trials_run < num_trials + max_extension_trials):
                
                extension_batches += 1
                logger.info(f"Extension batch {extension_batches}: Running {self.extension_batch_size} more trials")
                
                # Run a batch of additional trials
                for i in range(self.extension_batch_size):
                    if trials_run % self.recalibration_config.check_frequency == 0:
                        self.recalibrate_system()
                        
                    self.run_trial()
                    trials_run += 1
                
                # Recheck confidence
                answer, confidence, decision_state = self.get_answer_with_state()
                
                logger.info(f"After {trials_run} total trials: decision_state={decision_state}, "
                          f"confidence={confidence:.4f}")
                
                # Break if we've reached sufficient confidence
                if decision_state != "insufficient_evidence":
                    break
                    
            logger.info(f"Session extension complete. Total trials: {trials_run}, "
                      f"Extended by: {trials_run - total_trials}")
            total_trials = trials_run
            
        # Get hit rate statistics for analysis
        hit_rate_stats = self.rwba.get_hit_rate_statistics()
        
        result = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "decision_state": decision_state,
            "beliefs": self.current_beliefs,
            "initial_trials": num_trials,
            "total_trials": total_trials,
            "extended_trials": total_trials - num_trials,
            "input_predictability": self.predictability,
            "bound_size": self.rwba.bound_size,
            "hit_rate": hit_rate_stats["overall_hit_rate"],
            "rolling_hit_rate": hit_rate_stats["rolling_hit_rate"],
            "weighted_hit_rate": hit_rate_stats["weighted_hit_rate"],
            "hit_rate_confidence": self.rwba.get_hit_rate_confidence(),
            "average_steps": self.rwba.get_average_steps(),
            "recalibrations": self.get_recalibration_info()
        }
        
        logger.info(f"Final answer: {result['answer']}, Confidence: {confidence:.2%}, "
                  f"State: {decision_state}, Total trials: {total_trials}")
        
        return result
        
    def run_trial(self) -> TrialData:
        """Run a single trial.
        
        Returns:
            TrialData object for the trial
        """
        # Assign direction randomly for now (in real implementation, would be based on intention)
        direction = 1 if np.random.random() < 0.5 else -1
        
        # Run RWBA trial with weighted result
        result, steps, weight = self.rwba.run_trial()
        
        # Update beliefs based on result and weight
        self.update_beliefs_adaptive(result, weight)
        
        # Create trial data
        trial_data = TrialData(
            timestamp=datetime.now(),
            question=self.current_question,
            direction=direction,
            step_count=steps,
            result=result,
            weight=weight,
            metrics=self.expected_metrics.copy(),
            confidence=self.get_confidence(),
            beliefs=self.current_beliefs.copy()
        )
        
        # Store trial
        self.trials.append(trial_data)
        
        return trial_data
    
    def update_beliefs_adaptive(self, result: int, weight: float) -> None:
        """
        Update beliefs using adaptive Bayesian updating with real-time hit rates.
        
        Args:
            result: Trial result (0 or 1)
            weight: Trial weight based on time-to-bound and other factors
        """
        # Get current hit rate from rolling window
        if self.use_weighted_updating:
            hit_rate = self.rwba.get_weighted_hit_rate()
        else:
            hit_rate = self.rwba.get_rolling_hit_rate()
            
        # Get hit rate confidence (how reliable our estimate is)
        hit_rate_confidence = self.rwba.get_hit_rate_confidence()
        
        # Apply stability threshold to prevent updating on noise
        if abs(hit_rate - 0.5) < self.stability_threshold:
            logger.debug(f"Hit rate {hit_rate:.4f} within stability threshold {self.stability_threshold}, "
                        f"skipping belief update")
            return
            
        # Skip updates if hit rate confidence is too low
        if hit_rate_confidence < 0.6:
            logger.debug(f"Hit rate confidence {hit_rate_confidence:.4f} too low, skipping belief update")
            return
        
        # For binary hypothesis case (yes/no)
        if len(self.supported_hypotheses) == 2 and "yes" in self.supported_hypotheses and "no" in self.supported_hypotheses:
            self._update_binary_beliefs(result, hit_rate, weight)
        else:
            # For multi-hypothesis case
            self._update_multi_beliefs(result, hit_rate, weight)
    
    def _update_binary_beliefs(self, result: int, hit_rate: float, weight: float) -> None:
        """
        Update beliefs for binary yes/no case.
        
        Args:
            result: Trial result (0 or 1)
            hit_rate: Current RWBA hit rate
            weight: Trial weight
        """
        # Current beliefs
        p_yes = self.current_beliefs["yes"]
        p_no = self.current_beliefs["no"]
        
        # Calculate likelihoods based on actual hit rate
        # For 'yes' hypothesis, hit rate represents p(1|yes)
        # For 'no' hypothesis, (1-hit_rate) represents p(1|no)
        if result == 1:  # Hit upper bound
            # The higher the hit rate, the more likely "yes" is and the less likely "no" is
            p_e_yes = hit_rate
            p_e_no = 1 - hit_rate
        else:  # Hit lower bound
            # The lower the hit rate, the more likely "no" is and the less likely "yes" is
            p_e_yes = 1 - hit_rate
            p_e_no = hit_rate
        
        # Apply weight to modulate evidence strength
        # Weight > 1 strengthens evidence, weight < 1 weakens it
        p_e_yes = 0.5 + (p_e_yes - 0.5) * weight
        p_e_no = 0.5 + (p_e_no - 0.5) * weight
        
        # Apply Bayes rule with normalization
        unnorm_p_yes = p_yes * p_e_yes
        unnorm_p_no = p_no * p_e_no
        
        # Normalize
        total = unnorm_p_yes + unnorm_p_no
        if total > 0:
            self.current_beliefs = {
                "yes": unnorm_p_yes / total,
                "no": unnorm_p_no / total
            }
            
        logger.debug(f"Updated beliefs: yes={self.current_beliefs['yes']:.4f}, "
                    f"no={self.current_beliefs['no']:.4f}, hit_rate={hit_rate:.4f}, "
                    f"result={result}, weight={weight:.2f}")
    
    def _update_multi_beliefs(self, result: int, hit_rate: float, weight: float) -> None:
        """
        Update beliefs for multi-hypothesis case.
        
        Args:
            result: Trial result (0 or 1)
            hit_rate: Current RWBA hit rate
            weight: Trial weight
        """
        # Store intermediate results
        unnorm_beliefs = {}
        
        # For each hypothesis, calculate unnormalized posterior
        # We assume each hypothesis has a different "expected" hit rate
        # that would be observed if that hypothesis were true
        hypothesis_count = len(self.supported_hypotheses)
        
        for i, hypothesis in enumerate(self.supported_hypotheses):
            # Simple model: linear mapping of hypotheses to expected hit rates
            # This should be customized for actual applications
            expected_hit_rate = 0.5 + (i / (hypothesis_count - 1) - 0.5) * 0.4
            
            # Calculate likelihood based on difference between expected and actual hit rate
            if result == 1:  # Hit upper bound
                likelihood = 1.0 - abs(hit_rate - expected_hit_rate)
            else:  # Hit lower bound
                likelihood = 1.0 - abs((1 - hit_rate) - expected_hit_rate)
                
            # Apply weight
            likelihood = 0.5 + (likelihood - 0.5) * weight
            
            # Apply Bayes rule (unnormalized)
            unnorm_beliefs[hypothesis] = self.current_beliefs[hypothesis] * likelihood
        
        # Normalize
        total = sum(unnorm_beliefs.values())
        if total > 0:
            for hypothesis in self.supported_hypotheses:
                self.current_beliefs[hypothesis] = unnorm_beliefs[hypothesis] / total
                
        logger.debug(f"Updated multi-hypothesis beliefs: {', '.join([f'{h}={v:.3f}' for h, v in self.current_beliefs.items()])}")
    
    def get_answer(self) -> Tuple[str, float]:
        """
        Get the most likely answer and confidence.
        
        Returns:
            Tuple of (answer, confidence)
        """
        if not self.current_beliefs:
            return "Unknown", 0.5
            
        # Find hypothesis with highest probability
        max_prob = 0.0
        best_hypothesis = None
        
        for hypothesis, prob in self.current_beliefs.items():
            if prob > max_prob:
                max_prob = prob
                best_hypothesis = hypothesis
                
        # Check if we should return "No Decision" based on confidence thresholds
        if self.allow_no_decision:
            # For binary case (yes/no)
            if len(self.current_beliefs) == 2 and "yes" in self.current_beliefs and "no" in self.current_beliefs:
                margin = abs(self.current_beliefs["yes"] - self.current_beliefs["no"])
                if margin < self.decision_margin_threshold or max_prob < self.min_confidence_threshold:
                    return "No Decision", max(max_prob, 0.5)
        
        return best_hypothesis, max_prob
    
    def get_answer_with_state(self) -> Tuple[str, float, str]:
        """
        Get the most likely answer with decision state information.
        
        Returns:
            Tuple of (answer, confidence, decision_state) where decision_state is one of:
            - "confident_decision": Clear decision with high confidence
            - "tentative_decision": Decision with moderate confidence
            - "insufficient_evidence": Not enough evidence to make a reliable decision
        """
        if not self.current_beliefs:
            return "Unknown", 0.5, "insufficient_evidence"
        
        # Find hypothesis with highest probability
        max_prob = 0.0
        best_hypothesis = None
        
        for hypothesis, prob in self.current_beliefs.items():
            if prob > max_prob:
                max_prob = prob
                best_hypothesis = hypothesis
        
        # Determine decision state based on confidence thresholds
        decision_state = "tentative_decision"  # Default
        
        # For binary case (yes/no)
        if len(self.current_beliefs) == 2 and "yes" in self.current_beliefs and "no" in self.current_beliefs:
            margin = abs(self.current_beliefs["yes"] - self.current_beliefs["no"])
            
            # Check if we have insufficient evidence
            if margin < self.decision_margin_threshold or max_prob < self.min_confidence_threshold:
                decision_state = "insufficient_evidence"
                if self.allow_no_decision:
                    best_hypothesis = "No Decision"
            # Check if we have a confident decision
            elif max_prob > 0.7 and margin > 0.2:  # Strong confidence thresholds
                decision_state = "confident_decision"
        else:
            # For multi-hypothesis case, use different logic
            probs = sorted(self.current_beliefs.values(), reverse=True)
            if len(probs) > 1:
                margin = probs[0] - probs[1]  # Difference between top two hypotheses
                
                if margin < self.decision_margin_threshold or max_prob < self.min_confidence_threshold:
                    decision_state = "insufficient_evidence"
                    if self.allow_no_decision:
                        best_hypothesis = "No Decision"
                elif max_prob > 0.7 and margin > 0.2:
                    decision_state = "confident_decision"
        
        return best_hypothesis, max_prob, decision_state
    
    def get_confidence(self) -> float:
        """
        Calculate confidence in current beliefs.
        
        Returns:
            Confidence value (0.5 to 1.0)
        """
        if not self.current_beliefs:
            return 0.5
            
        # Find highest probability
        max_prob = max(self.current_beliefs.values())
        
        # For binary beliefs, a confidence of 1.0 means 100% certainty
        if len(self.current_beliefs) == 2:
            return max_prob
        
        # For multi-hypothesis, we need to calculate differently
        # Using the ratio between the highest and second highest
        probs = sorted(self.current_beliefs.values(), reverse=True)
        if len(probs) > 1 and probs[1] > 0:
            ratio = probs[0] / probs[1]
            # Scale to a reasonable range
            confidence = 0.5 + 0.5 * min(ratio / 3.0, 1.0)
            return confidence
        
        return max_prob
    
    def save_session(self, filename: str) -> None:
        """Save session data to file.
        
        Args:
            filename: Path to save file
        """
        session_data = {
            "session_start": self.session_start.isoformat(),
            "trials": [
                {
                    "timestamp": trial.timestamp.isoformat(),
                    "question": trial.question,
                    "direction": trial.direction,
                    "step_count": trial.step_count,
                    "result": trial.result,
                    "weight": trial.weight,
                    "metrics": trial.metrics,
                    "confidence": trial.confidence,
                    "beliefs": trial.beliefs
                }
                for trial in self.trials
            ],
            "configuration": {
                "predictability": self.predictability,
                "bound_size": self.rwba.bound_size,
                "nx": self.entropy_buffer.preprocessor.nx,
                "buffer_size": self.entropy_buffer.buffer_size
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Session saved to {filename}")
    
    def load_session(self, filename: str) -> None:
        """Load session data from file.
        
        Args:
            filename: Path to session file
        """
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        self.session_start = datetime.fromisoformat(session_data["session_start"])
        
        self.trials = []
        for trial_data in session_data["trials"]:
            trial = TrialData(
                timestamp=datetime.fromisoformat(trial_data["timestamp"]),
                question=trial_data["question"],
                direction=trial_data["direction"],
                step_count=trial_data["step_count"],
                result=trial_data["result"],
                weight=trial_data["weight"],
                metrics=trial_data["metrics"],
                confidence=trial_data["confidence"],
                beliefs=trial_data["beliefs"]
            )
            self.trials.append(trial)
            
        # Update current beliefs to latest trial if available
        if self.trials:
            self.current_beliefs = self.trials[-1].beliefs.copy()
            self.current_question = self.trials[-1].question
            
        logger.info(f"Session loaded from {filename}: {len(self.trials)} trials")
    
    def get_session_summary(self) -> Dict:
        """Get summary of the current session.
        
        Returns:
            Dictionary with session summary
        """
        if not self.trials:
            return {
                "session_start": self.session_start.isoformat(),
                "trials_count": 0,
                "questions": [],
                "system_stats": {
                    "predictability": self.predictability,
                    "bound_size": self.rwba.bound_size,
                }
            }
        
        # Group trials by question
        questions = {}
        for trial in self.trials:
            if trial.question not in questions:
                questions[trial.question] = []
            questions[trial.question].append(trial)
        
        # Calculate stats for each question
        question_stats = []
        for question, trials in questions.items():
            yes_count = sum(1 for trial in trials if trial.result == 1)
            no_count = len(trials) - yes_count
            
            # Get final beliefs from last trial
            final_beliefs = trials[-1].beliefs
            
            # Determine answer
            if final_beliefs["yes"] > final_beliefs["no"]:
                answer = "Yes"
                confidence = final_beliefs["yes"]
            else:
                answer = "No"
                confidence = final_beliefs["no"]
            
            question_stats.append({
                "question": question,
                "trials_count": len(trials),
                "yes_count": yes_count,
                "no_count": no_count,
                "answer": answer,
                "confidence": confidence,
                "beliefs": final_beliefs,
                "start_time": trials[0].timestamp.isoformat(),
                "end_time": trials[-1].timestamp.isoformat()
            })
        
        # System-wide statistics
        yes_count = sum(1 for trial in self.trials if trial.result == 1)
        no_count = len(self.trials) - yes_count
        
        return {
            "session_start": self.session_start.isoformat(),
            "trials_count": len(self.trials),
            "yes_count": yes_count,
            "no_count": no_count,
            "questions": question_stats,
            "system_stats": {
                "predictability": self.predictability,
                "bound_size": self.rwba.bound_size,
                "rwba_statistics": self.rwba.get_statistics(),
                "entropy_buffer": {
                    "nx": self.entropy_buffer.preprocessor.nx,
                    "buffer_size": self.entropy_buffer.buffer_size,
                    "raw_entropy": self.entropy_buffer.preprocessor.raw_entropy_estimate,
                    "processed_entropy": self.entropy_buffer.preprocessor.processed_entropy_estimate
                }
            }
        }
    
    def reset_session(self) -> None:
        """Reset the session, clearing all trials."""
        self.trials = []
        self.session_start = datetime.now()
        self.current_beliefs = {}
        self.current_question = None
        self.rwba.hit_upper_count = 0
        self.rwba.hit_lower_count = 0
        self.rwba.trial_count = 0
        self.rwba.total_steps = 0
        
        logger.info("Session reset")
    
    def verify_system(self, num_trials: int = 1000) -> Dict:
        """
        Run verification tests to ensure the system is functioning correctly.
        
        Args:
            num_trials: Number of verification trials to run
            
        Returns:
            Dictionary with verification results
        """
        logger.info(f"Running system verification with {num_trials} trials")
        
        # 1. Verify XOR preprocessor
        xor_verification = self.entropy_buffer.preprocessor.verify_preprocessing(50000)
        
        # 2. Verify RWBA behavior
        # Save current state
        original_position = self.rwba.position
        original_steps = self.rwba.steps_taken
        original_hit_upper = self.rwba.hit_upper_count
        original_hit_lower = self.rwba.hit_lower_count
        original_trial_count = self.rwba.trial_count
        
        # Run verification trials with random bits
        self.rwba.reset()
        results = []
        
        # Generate random bits with slight bias for verification
        random_bits = [1 if np.random.random() < 0.53 else 0 for _ in range(num_trials * 1000)]
        
        # Process through RWBA
        output_bits = self.rwba.process_bits(random_bits)
        
        # Verify output statistics
        rwba_stats = self.rwba.get_statistics()
        hit_rate = rwba_stats["hit_rate"]
        
        # Calculate expected output probability
        input_p = 0.53  # Our test bias
        expected_p_out = calculate_output_probability(input_p, self.rwba.bound_size)
        
        # Calculate error
        error = abs(hit_rate - expected_p_out) / expected_p_out if expected_p_out > 0 else 0
        
        # Restore original state
        self.rwba.position = original_position
        self.rwba.steps_taken = original_steps
        self.rwba.hit_upper_count = original_hit_upper
        self.rwba.hit_lower_count = original_hit_lower
        self.rwba.trial_count = original_trial_count
        
        verification_results = {
            "xor_preprocessing": xor_verification,
            "rwba_verification": {
                "input_probability": input_p,
                "expected_output_probability": expected_p_out,
                "actual_hit_rate": hit_rate,
                "error_percentage": error * 100,
                "output_bits_generated": len(output_bits),
                "average_steps_per_trial": rwba_stats["average_steps"],
                "is_valid": error < 0.05  # Less than 5% error is acceptable
            },
            "system_state": {
                "predictability": self.predictability,
                "bound_size": self.rwba.bound_size,
                "nx": self.entropy_buffer.preprocessor.nx,
                "entropy": self.entropy_buffer.get_entropy_estimate()
            }
        }
        
        logger.info(f"System verification complete: "
                  f"RWBA hit_rate={hit_rate:.6f}, expected={expected_p_out:.6f}, "
                  f"error={error*100:.2f}%")
        
        return verification_results
    
    def get_recalibration_info(self) -> Dict:
        """
        Get information about recalibration events.
        
        Returns:
            Dictionary with recalibration statistics
        """
        if not self.recalibration_events:
            return {"count": 0}
            
        return {
            "count": len(self.recalibration_events),
            "last_event": self.recalibration_events[-1],
            "average_bound_change": np.mean([abs(e["new_bound"] - e["old_bound"]) for e in self.recalibration_events]),
            "average_predictability_change": np.mean([abs(e["new_predictability"] - e["old_predictability"]) for e in self.recalibration_events]),
        }
    
    def toggle_xor_preprocessing(self, enabled: bool) -> None:
        """
        Enable or disable XOR preprocessing.
        
        Args:
            enabled: Whether to enable XOR preprocessing
        """
        # Toggle in entropy buffer
        self.entropy_buffer.toggle_xor_preprocessing(enabled)
        
        # Update system predictability
        self.predictability = self.entropy_buffer.get_predictability()
        
        # Recalculate optimal bound size
        old_bound = self.rwba.bound_size
        optimal_bound, metrics = find_optimal_bound_size(self.predictability)
        
        # Update bound size if automatic bound sizing is enabled
        rwba_config = self.config.get('rwba', {})
        if rwba_config.get('use_optimal_bound', True):
            self.rwba.bound_size = optimal_bound
            logger.info(f"Updated bound size after XOR toggle: {old_bound} -> {optimal_bound}")
        
        # Update expected metrics
        p_out = calculate_output_probability(self.predictability, self.rwba.bound_size)
        self.expected_metrics = {
            "input_probability": self.predictability,
            "bound_size": self.rwba.bound_size,
            "expected_output_probability": p_out,
            "expected_effect_size": 2 * p_out - 1,
        }
        
        logger.info(f"XOR preprocessing {'enabled' if enabled else 'disabled'}, "
                   f"new predictability: {self.predictability:.6f}, "
                   f"expected_output_probability: {p_out:.6f}")
                   
    def test_xor_vs_raw(self, num_trials: int = 1000) -> Dict:
        """
        Run empirical tests comparing XOR-processed bits vs raw bits.
        
        Args:
            num_trials: Number of trials to run for each mode
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Starting empirical test: XOR vs Raw bits with {num_trials} trials each")
        results = {}
        
        # Store original state
        original_xor_enabled = self.entropy_buffer.xor_config.enabled
        original_bound_size = self.rwba.bound_size
        
        # Test with XOR enabled
        if not self.entropy_buffer.xor_config.enabled:
            self.toggle_xor_preprocessing(True)
        else:
            # Still recalibrate to ensure predictability is current
            self.recalibrate_system(force=True)
            
        # Reset RWBA statistics
        self.rwba.hit_upper_count = 0
        self.rwba.hit_lower_count = 0
        self.rwba.trial_count = 0
        self.rwba.total_steps = 0
        
        # Run trials with XOR
        logger.info(f"Running {num_trials} trials with XOR preprocessing enabled")
        xor_results = []
        for _ in range(num_trials):
            result, steps = self.rwba.run_trial()
            xor_results.append((result, steps))
            
        xor_stats = self.rwba.get_statistics()
        logger.info(f"XOR enabled results: hit_rate={xor_stats['hit_rate']:.6f}, "
                  f"effect_size={xor_stats['effect_size']:.6f}, "
                  f"avg_steps={xor_stats['average_steps']:.2f}")
                  
        results["xor_enabled"] = {
            "hit_rate": xor_stats["hit_rate"],
            "effect_size": xor_stats["effect_size"],
            "average_steps": xor_stats["average_steps"],
            "predictability": self.entropy_buffer.get_predictability(),
            "bound_size": self.rwba.bound_size,
            "trial_results": xor_results
        }
        
        # Test with raw bits
        self.toggle_xor_preprocessing(False)
        
        # Reset RWBA statistics
        self.rwba.hit_upper_count = 0
        self.rwba.hit_lower_count = 0
        self.rwba.trial_count = 0
        self.rwba.total_steps = 0
        
        # Run trials with raw bits
        logger.info(f"Running {num_trials} trials with raw bits (XOR disabled)")
        raw_results = []
        for _ in range(num_trials):
            result, steps = self.rwba.run_trial()
            raw_results.append((result, steps))
            
        raw_stats = self.rwba.get_statistics()
        logger.info(f"Raw bits results: hit_rate={raw_stats['hit_rate']:.6f}, "
                  f"effect_size={raw_stats['effect_size']:.6f}, "
                  f"avg_steps={raw_stats['average_steps']:.2f}")
                  
        results["raw_bits"] = {
            "hit_rate": raw_stats["hit_rate"],
            "effect_size": raw_stats["effect_size"],
            "average_steps": raw_stats["average_steps"],
            "predictability": self.entropy_buffer.get_raw_predictability(),
            "bound_size": self.rwba.bound_size,
            "trial_results": raw_results
        }
        
        # Compare results
        effect_size_diff = abs(raw_stats["effect_size"]) - abs(xor_stats["effect_size"])
        effect_size_percent = (effect_size_diff / abs(xor_stats["effect_size"])) * 100 if xor_stats["effect_size"] != 0 else float('inf')
        
        results["comparison"] = {
            "effect_size_difference": effect_size_diff,
            "effect_size_percent_change": effect_size_percent,
            "steps_difference": raw_stats["average_steps"] - xor_stats["average_steps"],
            "predictability_difference": self.entropy_buffer.get_raw_predictability() - self.entropy_buffer.get_predictability()
        }
        
        recommendation = "use_xor" if abs(xor_stats["effect_size"]) >= abs(raw_stats["effect_size"]) else "use_raw"
        recommendation_reason = (
            "XOR preprocessing shows stronger effect size" if recommendation == "use_xor" 
            else "Raw bits show stronger effect size"
        )
        
        results["recommendation"] = {
            "mode": recommendation,
            "reason": recommendation_reason
        }
        
        logger.info(f"Test results comparison: effect_size_diff={effect_size_diff:.6f} ({effect_size_percent:.2f}%), "
                  f"recommendation: {recommendation} - {recommendation_reason}")
        
        # Restore original state
        if original_xor_enabled != self.entropy_buffer.xor_config.enabled:
            self.toggle_xor_preprocessing(original_xor_enabled)
        
        if original_bound_size != self.rwba.bound_size:
            self.rwba.bound_size = original_bound_size
        
        return results 