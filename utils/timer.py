import time

class Timer:
    """
    Context manager for timing execution of code blocks.
    
    Example:
        with Timer("Training Epoch"):
            train_epoch()
    """
    def __init__(self, name="Task", logger=None):
        self.name = name
        self.logger = logger
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        
        msg = f"{self.name} took {self.interval:.4f} seconds"
        
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
