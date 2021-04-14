from dataclasses import dataclass
from typing import Optional


@dataclass
class CompareVariableResult:
    condition_satisfied: bool
    p_value: Optional[float]
    rejected: Optional[bool]
