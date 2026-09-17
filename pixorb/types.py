from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class ImageRecord:
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

@dataclass
class Match:
    x_src: float
    y_src: float
    x_ref: float
    y_ref: float
    confidence: float = 0.0
    reprojection_error_px: float = float("nan")

    def as_dict(self):
        return {
            "x_src": float(self.x_src), "y_src": float(self.y_src),
            "x_ref": float(self.x_ref), "y_ref": float(self.y_ref),
            "confidence": float(self.confidence),
            "reprojection_error_px": float(self.reprojection_error_px),
        }

@dataclass
class RegistrationResult:
    registered: np.ndarray
    transform: np.ndarray
    matches: List[Match]
    metrics: Dict[str, Any]
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)
    stage_info: Dict[str, Any] = field(default_factory=dict)
