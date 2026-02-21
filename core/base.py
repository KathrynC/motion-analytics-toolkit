"""Base classes for all motion analyzers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import json
from .schemas import Telemetry

class MotionAnalyzer(ABC):
    """Base class for all motion analyzers."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.results = {}
        
    @abstractmethod
    def analyze(self, telemetry: Telemetry) -> Dict[str, Any]:
        """Run analysis on telemetry data."""
        pass
    
    def to_json_compatible(self) -> Dict:
        """Convert results to JSON-serializable format."""
        return self._convert_to_json(self.results)
    
    def _convert_to_json(self, obj: Any) -> Any:
        """Recursively convert numpy types to Python native."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: self._convert_to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_json(v) for v in obj]
        if hasattr(obj, '__dict__'):  # Handle dataclasses
            return self._convert_to_json(obj.__dict__)
        return obj
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_json_compatible(), f, indent=2)

class CompositeAnalyzer(MotionAnalyzer):
    """Run multiple analyzers and combine results."""
    
    def __init__(self, analyzers: Dict[str, MotionAnalyzer]):
        super().__init__()
        self.analyzers = analyzers
        
    def analyze(self, telemetry: Telemetry) -> Dict[str, Any]:
        results = {}
        for name, analyzer in self.analyzers.items():
            results[name] = analyzer.analyze(telemetry)
        self.results = results
        return results
