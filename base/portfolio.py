"""Base portfolio class for managing trading positions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Portfolio(ABC):
    """Abstract base class for portfolio management.
    
    This class provides an interface for managing trading positions
    across different exchanges.
    """
    
    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions.
        
        Returns:
            List of position dictionaries.
        """
        pass
    
    @abstractmethod
    def get_balances(self) -> dict[str, float]:
        """Get current account balances.
        
        Returns:
            Dictionary mapping asset symbols to balances.
        """
        pass
