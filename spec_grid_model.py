# spec_grid_model.py

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

Tier = Literal["Tier 1+", "Tier 1", "Tier 2", "Tier 3", "Generic"]


@dataclass
class PayupQuote:
    coupon_bucket: int                # 1–8
    coupon_rate: Optional[float]      # optional, from CSV
    story_code: str
    payup_32nds: int
    payup_points: float
    tier: Tier


PAYUP_GRID_32NDS: Dict[int, Dict[str, int]] = {}   # coupon_bucket -> story_code -> payup
COUPON_RATES: Dict[int, float] = {}                # coupon_bucket -> coupon_rate


def classify_tier(payup_32nds: int) -> Tier:
    if payup_32nds >= 24:
        return "Tier 1+"
    elif payup_32nds >= 16:
        return "Tier 1"
    elif payup_32nds >= 8:
        return "Tier 2"
    elif payup_32nds > 0:
        return "Tier 3"
    else:
        return "Generic"


def load_payup_grid(csv_path: Union[str, Path]) -> None:
    """
    CSV layout:
      - rows = coupon buckets (1..8)
      - columns = stories (LLB_85, GEO_TX_NY, etc.)
      - each cell = payup in 32nds
    """
    global PAYUP_GRID_32NDS, COUPON_RATES

    csv_path = Path(csv_path)
    PAYUP_GRID_32NDS = {}
    COUPON_RATES = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "coupon_bucket" not in fieldnames:
            raise ValueError("CSV must have a 'coupon_bucket' column")

        has_coupon_rate = "coupon_rate" in fieldnames

        story_columns = [
            col for col in fieldnames
            if col not in ("coupon_bucket", "coupon_rate")
        ]

        for row in reader:
            bucket = int(row["coupon_bucket"])
            if has_coupon_rate and row.get("coupon_rate"):
                COUPON_RATES[bucket] = float(row["coupon_rate"])

            PAYUP_GRID_32NDS[bucket] = {}
            for story_code in story_columns:
                cell = row.get(story_code, "")
                if cell == "" or cell is None:
                    payup = 0
                else:
                    payup = int(cell)
                PAYUP_GRID_32NDS[bucket][story_code] = payup


def get_payup(coupon_bucket: int, story_code: str) -> PayupQuote:
    if coupon_bucket not in PAYUP_GRID_32NDS:
        raise ValueError(f"No row found for coupon_bucket={coupon_bucket}")

    row = PAYUP_GRID_32NDS[coupon_bucket]
    if story_code not in row:
        raise ValueError(
            f"No payup found for story '{story_code}' in coupon_bucket={coupon_bucket}"
        )

    payup_32 = row[story_code]
    tier = classify_tier(payup_32)
    coupon_rate = COUPON_RATES.get(coupon_bucket)

    return PayupQuote(
        coupon_bucket=coupon_bucket,
        coupon_rate=coupon_rate,
        story_code=story_code,
        payup_32nds=payup_32,
        payup_points=payup_32 / 32.0,
        tier=tier,
    )


def list_stories_for_coupon(coupon_bucket: int) -> List[PayupQuote]:
    if coupon_bucket not in PAYUP_GRID_32NDS:
        raise ValueError(f"No row found for coupon_bucket={coupon_bucket}")

    row = PAYUP_GRID_32NDS[coupon_bucket]
    coupon_rate = COUPON_RATES.get(coupon_bucket)
    quotes: List[PayupQuote] = []

    for story_code, payup_32 in row.items():
        tier = classify_tier(payup_32)
        quotes.append(
            PayupQuote(
                coupon_bucket=coupon_bucket,
                coupon_rate=coupon_rate,
                story_code=story_code,
                payup_32nds=payup_32,
                payup_points=payup_32 / 32.0,
                tier=tier,
            )
        )
    return quotes