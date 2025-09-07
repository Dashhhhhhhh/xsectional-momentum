from __future__ import annotations
from pathlib import Path
from typing import List
import os
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class Settings(BaseModel):
    # Strategy configuration
    universe: List[str]
    start_date: str
    lookback_days: int = Field(gt=0)
    skip_days: int = Field(ge=0)
    trail_cov_days: int = Field(gt=0)
    top_decile: float = Field(gt=0, le=0.5)
    bottom_decile: float = Field(gt=0, le=0.5)
    target_ann_vol: float = Field(gt=0)
    max_weight_per_name: float = Field(gt=0)
    fixed_cost_bps: int = Field(ge=0)
    impact_k: int = Field(ge=0)
    adv_window: int = Field(gt=0)
    is_end: str
    use_alpaca: bool = False

    # API keys from environment
    alphavantage_api_key: str = Field(default="")
    alpaca_api_key_id: str = Field(default="")
    alpaca_api_secret_key: str = Field(default="")

    @validator('universe')
    def _universe_non_empty(cls, v):
        if not v:
            raise ValueError('universe must be non-empty')
        return v

    @validator('top_decile', 'bottom_decile')
    def _deciles_in_range(cls, v):
        if not (0 < v <= 0.5):
            raise ValueError('deciles must be in range (0, 0.5]')
        return v


def load_settings(path: str | Path) -> Settings:
    load_dotenv()  # Load environment variables from .env file
    path = Path(path)
    data = yaml.safe_load(path.read_text())

    # Merge with environment variables for API keys
    env_data = {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY', ''),
        'alpaca_api_key_id': os.getenv('ALPACA_API_KEY_ID', ''),
        'alpaca_api_secret_key': os.getenv('ALPACA_API_SECRET_KEY', ''),
    }

    # Combine YAML data with environment data
    combined_data = {**data, **env_data}
    return Settings(**combined_data)


