"""Where the data lives. Set LYNX_PRICES to run any script in YANNICK/ on another file.

    LYNX_PRICES=prices/prices_old_example/example_prices.csv uv run python YANNICK/edge_tests.py
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICES = Path(os.environ.get("LYNX_PRICES", ROOT / "prices" / "close_prices_insample.csv"))
MACRO = ROOT / "prices" / "macro_data_insample.csv"
