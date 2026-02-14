from abc import ABC, abstractmethod
from typing import Callable, Union
from .sa_measure import SaMeasure
import torch as th
import torch.distributions as thd

class InfraDistribution(ABC):
    """Abstract infradistribution - convex set of sa-measures."""
    
    @abstractmethod
    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """Compute expected value wrt this infradistribution."""
        
    @abstractmethod
    def entropy(self) -> th.Tensor:
        """Maximum entropy of sa-measures in the infradistribution."""

class InfraPolytope(InfraDistribution):
    """Infradistribution represented by polytope of sa-measures."""
    
    def __init__(self, batched_measure: Union[thd.Distribution, SaMeasure]):
        """Construct from batch of sa-measures."""
        if isinstance(batched_measure, thd.Distribution):
            batched_measure = SaMeasure(batched_measure)
        
        if not batched_measure.mu.batch_shape:
            raise ValueError("SaMeasure should have at least one batch dimension")
        
        self._batched_measure = batched_measure

    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """Compute infimum of expectations (min over sa-measures)."""
        return self._batched_measure(f).min(dim=0).values

    def entropy(self) -> th.Tensor:
        """Maximum entropy over sa-measures."""
        return self._batched_measure.entropy().max(dim=0).values

    def __repr__(self):
        return f"InfraPolytope({self._batched_measure})"
