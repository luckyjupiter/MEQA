from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
import time
from qng import QNG
from qng_preprocessor import QNGPreprocessor
from predictability_analyzer import PredictabilityAnalyzer
from config import XorPreprocessingConfig

logger = logging.getLogger(__name__)

class EntropyBuffer:
    """Buffer for quantum random bits with predictability analysis."""
    
    def __init__(self, qng: QNG, buffer_size: int = 8192, 
                analysis_sample_size: int = 100000, 
                nx: Optional[int] = None,
                xor_config: Optional[XorPreprocessingConfig] = None,
                recalibration_threshold: float = 0.005,
                lock_duration: int = 300):
        """
        Initialize entropy buffer with QNG source.
        
        Args:
            qng: Quantum Number Generator instance
            buffer_size: Size of the buffer in bits
            analysis_sample_size: Number of bits to sample for predictability analysis
            nx: Number of bits to XOR (None for auto-calculation)
            xor_config: XOR preprocessing configuration
            recalibration_threshold: Minimum delta change in p_in to trigger recalibration
            lock_duration: Minimum time (seconds) between recalibrations
        """
        self.qng = qng
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
        self.analysis_sample_size = analysis_sample_size
        
        # Set XOR preprocessing configuration
        self.xor_config = xor_config or XorPreprocessingConfig()
        if nx is not None:
            self.xor_config.nx = nx
            self.xor_config.auto_calculate_nx = False
        
        logger.info(f"Initializing EntropyBuffer with buffer_size={buffer_size}, "
                   f"XOR preprocessing={'enabled' if self.xor_config.enabled else 'disabled'}")
        
        # Initialize preprocessor if XOR is enabled
        if self.xor_config.enabled:
            self.preprocessor = QNGPreprocessor(self.qng, nx=self.xor_config.nx)
        else:
            logger.info("XOR preprocessing disabled, using raw quantum bits")
            self.preprocessor = None
        
        # Initialize predictability analyzer
        self.analyzer = PredictabilityAnalyzer()
        
        # Statistical characteristics
        self.statistics = None
        self.raw_statistics = None
        self.optimal_parameters = None
        
        # Dynamic recalibration parameters
        self.recalibration_threshold = recalibration_threshold
        self.lock_duration = lock_duration
        self.last_recalibration_time = time.time()
        self.monitoring_buffer = []
        self.monitoring_buffer_size = 1000  # Size of buffer for continuous monitoring

        # Fill buffer and analyze initial predictability
        self.fill()
        self.analyze_entropy_source()
    
    def fill(self):
        """Fill buffer with preprocessed or raw random bits."""
        logger.debug("Filling entropy buffer")
        
        if self.xor_config.enabled and self.preprocessor:
            # Get XOR-processed bits
            self.buffer = self.preprocessor.get_processed_bits(self.buffer_size)
        else:
            # Get raw bits directly
            self.buffer = []
            while len(self.buffer) < self.buffer_size:
                int32_val = self.qng.rand_int32()
                for j in range(32):
                    self.buffer.append((int32_val >> j) & 1)
                    if len(self.buffer) >= self.buffer_size:
                        break
        
        # Update monitoring buffer with new bits for continuous analysis
        self.update_monitoring_buffer(self.buffer[:min(len(self.buffer), 100)])  # Sample of new bits
        
        self.position = 0
    
    def analyze_entropy_source(self):
        """
        Analyze entropy source to determine statistical characteristics 
        and optimal parameters.
        """
        logger.info("Analyzing entropy source characteristics")
        
        # Get raw bits for analysis
        raw_sample = []
        while len(raw_sample) < self.analysis_sample_size:
            int32_val = self.qng.rand_int32()
            for j in range(32):
                raw_sample.append((int32_val >> j) & 1)
                if len(raw_sample) >= self.analysis_sample_size:
                    break
        
        # Calculate raw bit statistics
        self.raw_statistics = self.analyzer.calculate_statistical_characteristics(raw_sample)
        
        # If XOR is enabled, analyze preprocessed bits
        if self.xor_config.enabled and self.preprocessor:
            # Sample preprocessed bits for analysis
            sample = self.preprocessor.get_processed_bits(self.analysis_sample_size)
            
            # Calculate statistical characteristics
            self.statistics = self.analyzer.calculate_statistical_characteristics(sample)
        else:
            # Use raw statistics
            self.statistics = self.raw_statistics
        
        # Calculate optimal parameters
        self.optimal_parameters = self.analyzer.calculate_optimal_parameters(
            self.statistics["predictability"])
        
        logger.info(f"Entropy source analysis: predictability={self.statistics['predictability']:.6f}, "
                   f"entropy={self.statistics['entropy']:.6f}")
        
        # Log comparison if XOR is enabled
        if self.xor_config.enabled and self.preprocessor:
            logger.info(f"Raw vs Processed: raw_pred={self.raw_statistics['predictability']:.6f}, "
                       f"proc_pred={self.statistics['predictability']:.6f}, "
                       f"entropy_gain={self.statistics['entropy'] - self.raw_statistics['entropy']:.6f}")
    
    def get_bit(self) -> int:
        """
        Get a single bit from the buffer.
        
        Returns:
            Single random bit (0 or 1)
        """
        # Check if buffer needs refilling
        if self.position >= len(self.buffer):
            self.fill()
            
        # Get bit from current position
        bit = self.buffer[self.position]
        self.position += 1
        
        return bit
    
    def get_raw_bit(self) -> int:
        """
        Get a single raw (unprocessed) bit directly from QNG.
        
        Returns:
            Single raw random bit (0 or 1)
        """
        int32_val = self.qng.rand_int32()
        return int32_val & 1
    
    def get_predictability(self) -> float:
        """
        Get the measured predictability of the entropy source.
        
        Returns:
            Predictability value (0.5 ≤ p ≤ 1.0)
        """
        if self.statistics is None:
            self.analyze_entropy_source()
        return self.statistics["predictability"]
    
    def get_raw_predictability(self) -> float:
        """
        Get the measured predictability of the raw entropy source.
        
        Returns:
            Raw predictability value (0.5 ≤ p ≤ 1.0)
        """
        if self.raw_statistics is None:
            self.analyze_entropy_source()
        return self.raw_statistics["predictability"]
    
    def get_entropy_estimate(self) -> float:
        """
        Get the estimated Shannon entropy of the entropy source.
        
        Returns:
            Estimated entropy in bits/bit
        """
        if self.statistics is None:
            self.analyze_entropy_source()
        return self.statistics["entropy"]
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """
        Get optimal RWBA parameters based on entropy source characteristics.
        
        Returns:
            Dictionary of optimal parameters
        """
        if self.optimal_parameters is None:
            self.analyze_entropy_source()
        return self.optimal_parameters
    
    def toggle_xor_preprocessing(self, enabled: bool) -> None:
        """
        Enable or disable XOR preprocessing.
        
        Args:
            enabled: Whether to enable XOR preprocessing
        """
        if self.xor_config.enabled != enabled:
            logger.info(f"{'Enabling' if enabled else 'Disabling'} XOR preprocessing")
            self.xor_config.enabled = enabled
            
            # Initialize preprocessor if enabling
            if enabled and not self.preprocessor:
                self.preprocessor = QNGPreprocessor(self.qng, nx=self.xor_config.nx)
            
            # Clear buffer to force refill with new mode
            self.buffer = []
            self.position = 0
            
            # Re-analyze entropy source with new mode
            self.analyze_entropy_source()
    
    def consume(self, n: int) -> List[int]:
        """
        Consume n bits from buffer, refilling if necessary.
        
        Args:
            n: Number of bits to consume
            
        Returns:
            List of n consumed bits
        """
        result = []
        
        while len(result) < n:
            # Refill if needed
            if self.position >= len(self.buffer):
                self.fill()
            
            # Calculate how many bits to take from current buffer
            take = min(n - len(result), len(self.buffer) - self.position)
            result.extend(self.buffer[self.position:self.position + take])
            self.position += take
        
        return result
    
    def peek(self, n: int) -> List[int]:
        """
        Peek at next n bits without consuming.
        
        Args:
            n: Number of bits to peek
            
        Returns:
            List of n peeked bits
        """
        # Ensure buffer has enough bits
        if self.position + n > len(self.buffer):
            # Save current buffer and position
            current_buffer = self.buffer
            current_position = self.position
            
            # Fill buffer
            self.fill()
            
            # Combine buffers to have enough bits
            combined_buffer = current_buffer[current_position:] + self.buffer
            result = combined_buffer[:n]
            
            # Restore state
            self.buffer = current_buffer
            self.position = current_position
            
            return result
        else:
            return self.buffer[self.position:self.position + n]
    
    def get_bit_sequence(self, n: int) -> np.ndarray:
        """
        Get a sequence of n bits as a numpy array, consuming them from the buffer.
        
        Args:
            n: Number of bits to get
            
        Returns:
            Numpy array of bits
        """
        bits = self.consume(n)
        return np.array(bits, dtype=np.int32)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the entropy source.
        
        Returns:
            Dictionary with statistics
        """
        if self.statistics is None:
            self.analyze_entropy_source()
            
        # Add preprocessor stats to the output
        combined_stats = {
            **self.statistics,
            "buffer_size": self.buffer_size,
            "buffer_position": self.position,
            "nx": self.preprocessor.nx,
            "raw_entropy_estimate": self.preprocessor.raw_entropy_estimate,
            "processed_entropy_estimate": self.preprocessor.processed_entropy_estimate
        }
        
        return combined_stats 

    def update_monitoring_buffer(self, bits: List[int]) -> None:
        """
        Update the monitoring buffer used for continuous predictability tracking.
        
        Args:
            bits: New bits to add to the monitoring buffer
        """
        self.monitoring_buffer.extend(bits)
        if len(self.monitoring_buffer) > self.monitoring_buffer_size:
            self.monitoring_buffer = self.monitoring_buffer[-self.monitoring_buffer_size:]
        
        logger.debug(f"Updated monitoring buffer, size={len(self.monitoring_buffer)}")
    
    def calculate_current_predictability(self) -> float:
        """
        Calculate predictability based on the current monitoring buffer.
        
        Returns:
            Current predictability estimate
        """
        if len(self.monitoring_buffer) < self.monitoring_buffer_size / 2:
            # Not enough data, use the last full analysis
            return self.get_predictability()
            
        ones_count = sum(self.monitoring_buffer)
        total = len(self.monitoring_buffer)
        p = ones_count / total
        
        # Convert to predictability (0.5 <= p <= 1.0)
        predictability = max(p, 1 - p)
        
        logger.debug(f"Calculated current predictability: {predictability:.6f} from {total} bits")
        return predictability
    
    def should_recalibrate(self) -> Tuple[bool, Optional[float]]:
        """
        Determine if recalibration is necessary based on predictability drift and time lock.
        
        Returns:
            Tuple of (should_recalibrate, new_predictability)
        """
        current_time = time.time()
        time_since_last = current_time - self.last_recalibration_time
        
        # Check time lock first
        if time_since_last < self.lock_duration:
            logger.debug(f"Recalibration locked for {self.lock_duration - time_since_last:.1f} more seconds")
            return False, None
            
        # Calculate current predictability
        current_pred = self.calculate_current_predictability()
        last_pred = self.get_predictability()
        
        # Check predictability drift
        pred_drift = abs(current_pred - last_pred)
        if pred_drift > self.recalibration_threshold:
            logger.info(f"Predictability drift {pred_drift:.6f} exceeds threshold {self.recalibration_threshold:.6f}")
            self.last_recalibration_time = current_time  # Update lock timer
            return True, current_pred
            
        return False, None 