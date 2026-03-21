"""
Budget class with deadline tracking for token and time limits.
Manages resource budgets with TTL and automatic expiration.
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class BudgetStatus(Enum):
    """Budget status enumeration."""
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"

@dataclass
class Budget:
    """
    Budget tracker for tokens and time with deadline enforcement.
    
    Attributes:
        ttl_s: Time-to-live in seconds
        max_tokens: Maximum tokens allowed
        retries: Maximum retry attempts
        created_at: Creation timestamp
        tokens_used: Current token usage
        retries_used: Current retry count
    """
    ttl_s: float
    max_tokens: int
    retries: int
    created_at: float
    tokens_used: int = 0
    retries_used: int = 0
    
    @classmethod
    def create(
        cls,
        ttl_s: float = 90.0,
        max_tokens: int = 800,
        retries: int = 4
    ) -> "Budget":
        """Create a new budget instance."""
        return cls(
            ttl_s=ttl_s,
            max_tokens=max_tokens,
            retries=retries,
            created_at=time.time()
        )
    
    def consume_tokens(self, tokens: int) -> bool:
        """
        Consume tokens from the budget.
        
        Returns:
            True if tokens were consumed successfully, False if budget exhausted
        """
        if self.status() != BudgetStatus.ACTIVE:
            return False
        
        if self.tokens_used + tokens > self.max_tokens:
            return False
        
        self.tokens_used += tokens
        return True
    
    def consume_retry(self) -> bool:
        """
        Consume a retry attempt.
        
        Returns:
            True if retry was consumed successfully, False if retries exhausted
        """
        if self.status() != BudgetStatus.ACTIVE:
            return False
        
        if self.retries_used >= self.retries:
            return False
        
        self.retries_used += 1
        return True
    
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        if time.time() - self.created_at > self.ttl_s:
            return BudgetStatus.EXPIRED
        
        if self.tokens_used >= self.max_tokens:
            return BudgetStatus.EXHAUSTED
        
        return BudgetStatus.ACTIVE
    
    def remaining_tokens(self) -> int:
        """Get remaining tokens."""
        return max(0, self.max_tokens - self.tokens_used)
    
    def remaining_retries(self) -> int:
        """Get remaining retries."""
        return max(0, self.retries - self.retries_used)
    
    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        elapsed = time.time() - self.created_at
        return max(0.0, self.ttl_s - elapsed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert budget to dictionary."""
        return {
            "ttl_s": self.ttl_s,
            "max_tokens": self.max_tokens,
            "retries": self.retries,
            "created_at": self.created_at,
            "tokens_used": self.tokens_used,
            "retries_used": self.retries_used,
            "status": self.status().value,
            "remaining_tokens": self.remaining_tokens(),
            "remaining_retries": self.remaining_retries(),
            "remaining_time": self.remaining_time()
        }






