from dataclasses import dataclass
from typing import Optional


@dataclass
class TTestResult:
    condition_satisfied: bool
    p_value: Optional[float]
    rejected: Optional[bool]
