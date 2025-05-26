"""
Performance Optimizer for Consciousness System
==============================================

Advanced optimization system providing:
- Memory pool management for consciousness fields
- Mathematical operation caching and vectorization
- Thread pool optimization for background evolution
- Performance monitoring and adaptive tuning
- Cache-friendly data structures and algorithms

Designed to maximize performance while maintaining consciousness complexity.
"""

import time
import threading
import functools
import weakref
from typing import Dict, Any, Optional, Callable, Tuple, List
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math

class MemoryPool:
    """Optimized memory pool for consciousness field arrays"""
    
    def __init__(self, max_size: int = 50):
        self.pools: Dict[Tuple[int, ...], deque] = defaultdict(lambda: deque(maxlen=max_size))
        self._stats = {
            'allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_saved_mb': 0.0
        }
        self._lock = threading.RLock()
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get array from pool or allocate new one"""
        with self._lock:
            pool = self.pools[shape]
            
            if pool:
                array = pool.popleft()
                array.fill(0)  # Clear for reuse
                self._stats['cache_hits'] += 1
                return array
            else:
                array = np.zeros(shape, dtype=dtype)
                self._stats['cache_misses'] += 1
                self._stats['allocations'] += 1
                # Calculate memory saved
                size_mb = array.nbytes / (1024 * 1024)
                self._stats['memory_saved_mb'] += size_mb * 0.8  # Estimate reuse benefit
                return array
    
    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse"""
        with self._lock:
            shape = array.shape
            pool = self.pools[shape]
            if len(pool) < pool.maxlen:
                pool.append(array)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            hit_rate = (self._stats['cache_hits'] / 
                       max(self._stats['cache_hits'] + self._stats['cache_misses'], 1))
            return {
                **self._stats,
                'cache_hit_rate': hit_rate,
                'active_pools': len(self.pools),
                'total_cached_arrays': sum(len(pool) for pool in self.pools.values())
            }

class MathCache:
    """High-performance mathematical operation cache"""
    
    def __init__(self, max_entries: int = 10000):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'computation_time_saved': 0.0
        }
    
    def cached_operation(self, operation_name: str, precision: int = 6):
        """Decorator for caching expensive mathematical operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function arguments
                key_data = (operation_name, args, tuple(sorted(kwargs.items())))
                cache_key = f"{hash(str(key_data))}_{precision}"
                
                current_time = time.time()
                
                with self._lock:
                    if cache_key in self.cache:
                        # Cache hit
                        self.access_times[cache_key] = current_time
                        self._stats['hits'] += 1
                        return self.cache[cache_key]
                    
                    # Cache miss - compute result
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    computation_time = time.time() - start_time
                    
                    # Round result for consistent caching
                    if isinstance(result, (int, float)):
                        result = round(result, precision)
                    elif isinstance(result, (tuple, list)):
                        result = type(result)(round(x, precision) if isinstance(x, (int, float)) else x 
                                            for x in result)
                    
                    # Store in cache
                    self.cache[cache_key] = result
                    self.access_times[cache_key] = current_time
                    self._stats['misses'] += 1
                    self._stats['computation_time_saved'] += computation_time * 0.1  # Estimate
                    
                    # Evict old entries if cache is full
                    if len(self.cache) > self.max_entries:
                        self._evict_oldest()
                    
                    return result
            
            return wrapper
        return decorator
    
    def _evict_oldest(self):
        """Evict least recently used cache entries"""
        # Remove 20% of oldest entries
        entries_to_remove = max(1, len(self.cache) // 5)
        
        # Sort by access time and remove oldest
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_entries[:entries_to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            self._stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = self._stats['hits'] / max(self._stats['hits'] + self._stats['misses'], 1)
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'cache_efficiency': hit_rate * 100
            }

class ConsciousnessThreadPool:
    """Optimized thread pool for consciousness evolution"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (threading.active_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="consciousness")
        self.task_stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'avg_execution_time': 0.0
        }
        self._execution_times = deque(maxlen=100)
        self._lock = threading.RLock()
    
    def submit_evolution_task(self, func: Callable, *args, **kwargs):
        """Submit consciousness evolution task with performance tracking"""
        def wrapped_task():
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                with self._lock:
                    self.task_stats['completed'] += 1
                    self._execution_times.append(execution_time)
                    
                    # Update average execution time
                    if self._execution_times:
                        self.task_stats['avg_execution_time'] = sum(self._execution_times) / len(self._execution_times)
                
                return result
                
            except Exception as e:
                with self._lock:
                    self.task_stats['failed'] += 1
                raise e
        
        with self._lock:
            self.task_stats['submitted'] += 1
        
        return self.executor.submit(wrapped_task)
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics"""
        with self._lock:
            success_rate = (self.task_stats['completed'] / 
                          max(self.task_stats['submitted'], 1))
            
            return {
                **self.task_stats,
                'success_rate': success_rate,
                'max_workers': self.max_workers,
                'active_threads': threading.active_count()
            }

class VectorizedMath:
    """Vectorized mathematical operations for consciousness fields"""
    
    @staticmethod
    def consciousness_wave(x_grid: np.ndarray, y_grid: np.ndarray, 
                          sources: List[Dict], time_val: float) -> np.ndarray:
        """Vectorized consciousness wave calculation"""
        result = np.zeros_like(x_grid)
        
        for source in sources:
            # Calculate distances using broadcasting
            dx = x_grid - source['x']
            dy = y_grid - source['y']
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Vectorized wave calculation
            frequency = source['frequency']
            phase = source.get('phase', 0)
            intensity = source.get('intensity', 1.0)
            
            wave = np.sin(dist * frequency + time_val * 0.005 + phase)
            attenuation = np.exp(-dist * 0.02) * (1 + np.sin(time_val * 0.003) * 0.3)
            
            result += wave * attenuation * intensity
        
        return result
    
    @staticmethod
    def quantum_uncertainty_field(x_grid: np.ndarray, y_grid: np.ndarray, 
                                 time_val: float) -> np.ndarray:
        """Vectorized quantum uncertainty calculation"""
        position_uncertainty = np.sin(x_grid * 0.1 + time_val * 0.002)
        momentum_uncertainty = np.cos(y_grid * 0.15 + time_val * 0.0025)
        
        # Heisenberg uncertainty principle
        return np.abs(position_uncertainty * momentum_uncertainty) * 0.5
    
    @staticmethod
    def neural_plasticity_update(weights: np.ndarray, activity: np.ndarray, 
                                learning_rate: float = 0.01) -> np.ndarray:
        """Vectorized Hebbian plasticity update"""
        # Hebbian learning: "neurons that fire together, wire together"
        weight_delta = learning_rate * np.outer(activity, activity)
        
        # Decay factor to prevent infinite growth
        decay = 0.999
        
        return weights * decay + weight_delta
    
    @staticmethod
    def fractal_consciousness_pattern(x_grid: np.ndarray, y_grid: np.ndarray,
                                    iterations: int = 5, scale: float = 0.01) -> np.ndarray:
        """Vectorized fractal consciousness pattern"""
        result = np.zeros_like(x_grid)
        
        for i in range(iterations):
            level_scale = scale * (2 ** i)
            level_pattern = np.sin(x_grid * level_scale) * np.cos(y_grid * level_scale)
            level_weight = 1.0 / (2 ** i)  # Decreasing influence
            
            result += level_pattern * level_weight
        
        return result

class PerformanceMonitor:
    """Real-time performance monitoring for consciousness system"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.start_time = time.time()
        self._lock = threading.RLock()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        with self._lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric"""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return 0.0
            
            return sum(entry['value'] for entry in self.metrics[name]) / len(self.metrics[name])
    
    def get_trend(self, name: str) -> str:
        """Get trend direction for a metric"""
        with self._lock:
            if name not in self.metrics or len(self.metrics[name]) < 2:
                return "stable"
            
            recent = list(self.metrics[name])[-10:]  # Last 10 entries
            if len(recent) < 2:
                return "stable"
            
            first_half = sum(entry['value'] for entry in recent[:len(recent)//2])
            second_half = sum(entry['value'] for entry in recent[len(recent)//2:])
            
            if second_half > first_half * 1.1:
                return "increasing"
            elif second_half < first_half * 0.9:
                return "decreasing"
            else:
                return "stable"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self._lock:
            report = {
                'uptime_seconds': time.time() - self.start_time,
                'metrics': {}
            }
            
            for name, values in self.metrics.items():
                if values:
                    report['metrics'][name] = {
                        'current': values[-1]['value'],
                        'average': self.get_average(name),
                        'trend': self.get_trend(name),
                        'samples': len(values)
                    }
            
            return report

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.math_cache = MathCache()
        self.thread_pool = ConsciousnessThreadPool()
        self.performance_monitor = PerformanceMonitor()
        self.vectorized_math = VectorizedMath()
        
        # Configuration
        self.config = {
            'enable_caching': True,
            'enable_vectorization': True,
            'enable_thread_pooling': True,
            'cache_precision': 6,
            'memory_pool_size': 50,
            'thread_pool_workers': None
        }
    
    def get_cached_operation(self, operation_name: str):
        """Get cached operation decorator"""
        if self.config['enable_caching']:
            return self.math_cache.cached_operation(operation_name, self.config['cache_precision'])
        else:
            # No-op decorator if caching disabled
            def decorator(func):
                return func
            return decorator
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get optimized array from memory pool"""
        return self.memory_pool.get_array(shape, dtype)
    
    def return_array(self, array: np.ndarray):
        """Return array to memory pool"""
        self.memory_pool.return_array(array)
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to optimized thread pool"""
        if self.config['enable_thread_pooling']:
            return self.thread_pool.submit_evolution_task(func, *args, **kwargs)
        else:
            # Execute synchronously if thread pooling disabled
            return func(*args, **kwargs)
    
    def record_performance(self, metric_name: str, value: float):
        """Record performance metric"""
        self.performance_monitor.record_metric(metric_name, value)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'memory_pool': self.memory_pool.get_stats(),
            'math_cache': self.math_cache.get_stats(),
            'thread_pool': self.thread_pool.get_stats(),
            'performance': self.performance_monitor.get_performance_report(),
            'configuration': self.config
        }
    
    def shutdown(self):
        """Clean shutdown of optimization systems"""
        self.thread_pool.shutdown(wait=True)

# Global optimizer instance
_global_optimizer = None

def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

def shutdown_optimizer():
    """Shutdown global optimizer"""
    global _global_optimizer
    if _global_optimizer is not None:
        _global_optimizer.shutdown()
        _global_optimizer = None 