"""Feed adapters for FX liquidity providers."""

from .base import FXPriceFeed
from .dukascopy import DukascopyFeed, DukascopyHistory
from .fix_session import FIXFeed
from .ib import IBFeed
from .oanda import OANDAFeed

__all__ = ["FXPriceFeed", "OANDAFeed", "IBFeed", "FIXFeed", "DukascopyFeed", "DukascopyHistory"]
