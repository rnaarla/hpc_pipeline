import random
import time
import logging
import threading
import signal
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ChaosMonkey:
    """HPC-focused chaos monkey for distributed training experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dry_run = config.get('dry_run', False)
        self.running = False
        self.thread = None
        
        # Chaos probabilities
        self.kill_probability = config.get('kill_probability', 0.1)
        self.slow_probability = config.get('slow_probability', 0.2)
        self.oom_probability = config.get('oom_probability', 0.1)
        
        # Timing configuration
        self.check_interval = config.get('check_interval', 30)  # seconds
        self.slow_min_duration = config.get('slow_min_duration', 5)
        self.slow_max_duration = config.get('slow_max_duration', 30)
        
        # MPI/rank configuration
        self.world_size = config.get('world_size', 1)
        self.current_rank = config.get('current_rank', 0)
        
        # OOM simulation config
        self.oom_size_mb = config.get('oom_size_mb', 1024)
        
        # Setup logging with required format
        self.setup_logging()
        
        # Set random seed if specified for deterministic testing
        if 'random_seed' in config:
            random.seed(config['random_seed'])
            
        self.logger.info(f"ChaosMonkey initialized - dry_run={self.dry_run}, world_size={self.world_size}")

    def setup_logging(self):
        """Configure logging with the required format: 2025-01-01T12:00:00 | CHAOS | Rank X | <event>."""
        self.logger = logging.getLogger('chaos_monkey')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Custom formatter for required format
        class ChaosFormatter(logging.Formatter):
            def format(self, record):
                timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                rank = getattr(record, 'rank', 'N/A')
                message = record.getMessage()
                return f"{timestamp} | CHAOS | Rank {rank} | {message}"
        
        handler.setFormatter(ChaosFormatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _log_event(self, message: str, rank: Optional[int] = None):
        """Log chaos event with proper format."""
        if rank is None:
            rank = self.current_rank
        
        # Create log record with rank attribute
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.rank = rank
        self.logger.handle(record)

    def kill_random_rank(self) -> bool:
        """Kill a random rank (simulated in dry_run mode)."""
        if self.world_size <= 1:
            target_rank = 0
        else:
            target_rank = random.randint(0, self.world_size - 1)
        
        if self.dry_run:
            self._log_event(f"SIMULATED: Would kill rank {target_rank}", target_rank)
            return True
        
        try:
            # In real mode, attempt to kill the process
            if target_rank == self.current_rank:
                self._log_event(f"Killing current rank {target_rank}", target_rank)
                # Kill self with SIGTERM
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                self._log_event(f"Would kill remote rank {target_rank} (not implemented)", target_rank)
                # In a real implementation, this would use MPI or cluster management
                # to kill remote ranks
            return True
            
        except Exception as e:
            self._log_event(f"Failed to kill rank {target_rank}: {e}", target_rank)
            return False

    def slow_random_rank(self) -> bool:
        """Slow down a random rank by sleeping."""
        if self.world_size <= 1:
            target_rank = 0
        else:
            target_rank = random.randint(0, self.world_size - 1)
        
        duration = random.randint(self.slow_min_duration, self.slow_max_duration)
        
        if target_rank == self.current_rank:
            self._log_event(f"Slowing down rank {target_rank} for {duration}s", target_rank)
            if not self.dry_run:
                time.sleep(duration)
            else:
                self._log_event(f"SIMULATED: Would sleep for {duration}s", target_rank)
        else:
            self._log_event(f"Would slow down remote rank {target_rank} for {duration}s", target_rank)
            
        return True

    def oom_simulation(self) -> bool:
        """Simulate out-of-memory condition."""
        target_rank = self.current_rank
        
        if self.dry_run:
            self._log_event(f"SIMULATED: Would trigger OOM on rank {target_rank}", target_rank)
            return True
        
        try:
            # Try CUDA OOM first if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._log_event(f"Triggering GPU OOM on rank {target_rank}", target_rank)
                device = torch.cuda.current_device()
                
                # Allocate large tensor to trigger OOM
                try:
                    # Get available memory and try to allocate most of it
                    free_mem = torch.cuda.get_device_properties(device).total_memory
                    alloc_size = int(free_mem * 0.9 // 4)  # 90% of memory, 4 bytes per float
                    large_tensor = torch.ones(alloc_size, device=device, dtype=torch.float32)
                    
                    # Hold for a brief moment then release
                    time.sleep(1)
                    del large_tensor
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError:
                    self._log_event(f"Successfully triggered GPU OOM on rank {target_rank}", target_rank)
                    torch.cuda.empty_cache()
                except Exception as e:
                    self._log_event(f"GPU OOM simulation failed: {e}", target_rank)
                    
            else:
                # Fallback to CPU memory stress
                self._log_event(f"Triggering CPU memory stress on rank {target_rank}", target_rank)
                
                # Allocate large list to stress memory
                size = self.oom_size_mb * 1024 * 1024 // 8  # 8 bytes per int
                try:
                    large_list = [0] * size
                    time.sleep(1)
                    del large_list
                except MemoryError:
                    self._log_event(f"Successfully triggered CPU OOM on rank {target_rank}", target_rank)
                except Exception as e:
                    self._log_event(f"CPU OOM simulation failed: {e}", target_rank)
                    
            return True
            
        except Exception as e:
            self._log_event(f"OOM simulation failed on rank {target_rank}: {e}", target_rank)
            return False

    def inject_event(self) -> bool:
        """Randomly inject one chaos event based on configured probabilities."""
        rand_val = random.random()
        
        if rand_val < self.kill_probability:
            self._log_event(f"Injecting kill event (p={rand_val:.3f})")
            return self.kill_random_rank()
        elif rand_val < self.kill_probability + self.slow_probability:
            self._log_event(f"Injecting slow event (p={rand_val:.3f})")
            return self.slow_random_rank()
        elif rand_val < self.kill_probability + self.slow_probability + self.oom_probability:
            self._log_event(f"Injecting OOM event (p={rand_val:.3f})")
            return self.oom_simulation()
        else:
            # No event this time
            return False

    def _chaos_loop(self):
        """Main chaos loop running in background thread."""
        self._log_event("Chaos loop started")
        
        while self.running:
            try:
                # Check if we should inject an event
                self.inject_event()
                
                # Wait for next check
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self._log_event(f"Error in chaos loop: {e}")
                time.sleep(5)  # Brief pause before continuing
                
        self._log_event("Chaos loop stopped")

    def start(self) -> bool:
        """Start the chaos monkey in a background thread."""
        if self.running:
            self._log_event("Chaos monkey already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._chaos_loop, daemon=True)
        self.thread.start()
        
        self._log_event("Chaos monkey started")
        return True

    def stop(self) -> bool:
        """Stop the chaos monkey."""
        if not self.running:
            self._log_event("Chaos monkey not running")
            return False
            
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            
        self._log_event("Chaos monkey stopped")
        return True

    def is_running(self) -> bool:
        """Check if chaos monkey is currently running."""
        return self.running

    def get_stats(self) -> Dict[str, Any]:
        """Get current configuration and stats."""
        return {
            'running': self.running,
            'dry_run': self.dry_run,
            'world_size': self.world_size,
            'current_rank': self.current_rank,
            'probabilities': {
                'kill': self.kill_probability,
                'slow': self.slow_probability,
                'oom': self.oom_probability
            },
            'config': {
                'check_interval': self.check_interval,
                'slow_duration_range': (self.slow_min_duration, self.slow_max_duration),
                'oom_size_mb': self.oom_size_mb
            }
        }


def main():
    """Standalone mode for testing."""
    print("ChaosMonkey - Standalone Mode")
    
    # Example configuration
    config = {
        'dry_run': True,  # Safe for testing
        'world_size': 4,
        'current_rank': 0,
        'kill_probability': 0.1,
        'slow_probability': 0.3,
        'oom_probability': 0.1,
        'check_interval': 10,
        'random_seed': 42  # For deterministic testing
    }
    
    chaos_monkey = ChaosMonkey(config)
    
    print(f"Configuration: {chaos_monkey.get_stats()}")
    
    # Test individual methods
    print("\n--- Testing individual methods ---")
    chaos_monkey._log_event("Testing kill_random_rank")
    chaos_monkey.kill_random_rank()
    
    chaos_monkey._log_event("Testing slow_random_rank")
    chaos_monkey.slow_random_rank()
    
    chaos_monkey._log_event("Testing oom_simulation")
    chaos_monkey.oom_simulation()
    
    chaos_monkey._log_event("Testing inject_event")
    for i in range(5):
        result = chaos_monkey.inject_event()
        chaos_monkey._log_event(f"inject_event #{i+1} returned: {result}")
    
    # Test continuous mode for a short time
    print("\n--- Testing continuous mode ---")
    chaos_monkey.start()
    
    try:
        time.sleep(30)  # Run for 30 seconds
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        chaos_monkey.stop()
    
    print("ChaosMonkey test completed")


if __name__ == "__main__":
    main()
