import json
import os
import time

class ExperimentTracker:
    """
    Logs metadata, configuration, and final metrics for reproducibility and tracking.
    Results are saved dynamically to a JSON file.
    """
    def __init__(self, log_path="experiments.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
    def log_experiment(self, config, best_metrics, total_time_seconds):
        """
        Appends the run summary to the JSON log file.
        """
        experiment_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": config.get("experiment_name", "unnamed"),
            "model_type": config.get("model", {}).get("type", "unknown"),
            "config": config,
            "best_validation_metrics": best_metrics,
            "training_duration_seconds": total_time_seconds
        }
        
        records = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except json.JSONDecodeError:
                pass
                
        records.append(experiment_record)
        
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
