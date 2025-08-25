from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


# <types:autogen start>
class TradeSignal(BaseModel):
    """Auto-generated from trade_signal.schema.json"""
    symbol: str
    side: Literal["LONG", "SHORT"]
    strength: float = Field(ge=0, le=1)
    timestamp: datetime
    strategy: str
    meta: dict | None
# <types:autogen end>

__all__ = ["TradeSignal"]
