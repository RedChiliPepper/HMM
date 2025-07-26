# profiling_compile.py
import os
import time
from datetime import datetime
from functools import wraps
from memory_profiler import memory_usage
from line_profiler import LineProfiler

class NumbaCompilationTracker:
    """Tracks which Numba functions have been compiled"""
    _instance = None
    _compiled_functions = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_compiled(self, func_id):
        return func_id in self._compiled_functions

    def mark_compiled(self, func_id):
        self._compiled_functions.add(func_id)

tracker = NumbaCompilationTracker()

def profile_to_logs(func=None, *, log_dir="logs"):
    """Line profiler that skips Numba functions"""
    if func is None:
        return lambda f: profile_to_logs(f, log_dir=log_dir)
    
    # Skip line profiling for Numba functions
    if hasattr(func, '__numba__'):
        return func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(log_dir, exist_ok=True)
        profiler = LineProfiler()
        profiler.add_function(func)
        result = profiler.runcall(func, *args, **kwargs)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{func.__name__}_lineprofile_{timestamp}.log")
        
        with open(log_file, 'w') as f:
            profiler.print_stats(stream=f)
        
        return result
    return wrapper

def profile_resources(func=None, *, log_dir="logs", interval=0.1):
    """Enhanced resource profiler with Numba compilation tracking"""
    if func is None:
        return lambda f: profile_resources(f, log_dir=log_dir, interval=interval)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Handle Numba functions
        if hasattr(func, '__numba__'):
            func_id = id(func)
            is_compiled = tracker.is_compiled(func_id)
            
            start_time = time.perf_counter()
            mem_usage, result = memory_usage(
                (func, args, kwargs),
                interval=interval,
                retval=True
            )
            duration = time.perf_counter() - start_time
            
            # Log to separate Numba-specific file
            log_file = os.path.join(log_dir, "resource_usage.log")
            with open(log_file, 'a') as f:
                f.write(
                    f"[{timestamp}] {func.__name__} | "
                    f"Compiled: {'No' if not is_compiled else 'Yes'} | "
                    f"Time: {duration:.4f}s | "
                    f"Peak Memory: {max(mem_usage):.2f} MiB\n"
                )
            
            if not is_compiled:
                tracker.mark_compiled(func_id)
            
            return result
            
        # Standard Python function profiling
        start_time = time.perf_counter()
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            interval=interval,
            retval=True
        )
        duration = time.perf_counter() - start_time
        
        with open(os.path.join(log_dir, "resource_usage.log"), 'a') as f:
            f.write(
                f"[{timestamp}] {func.__name__} | "
                f"Time: {duration:.4f}s | "
                f"Peak Memory: {max(mem_usage):.2f} MiB\n"
            )
        
        return result
    return wrapper
