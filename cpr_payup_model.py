# cpr_payup_model.py

import math
from dataclasses import dataclass
from typing import Dict, List


# -------- Basic utilities --------

def cpr_to_smm(cpr_annual: float) -> float:
    """Convert annual CPR to Single Monthly Mortality (SMM)."""
    return 1 - (1 - cpr_annual) ** (1 / 12.0)


def level_payment(principal: float, coupon_rate_annual: float, term_months: int) -> float:
    """Level monthly payment for a fixed-rate mortgage."""
    r = coupon_rate_annual / 12.0
    if r == 0:
        return principal / term_months
    return principal * (r / (1 - (1 + r) ** -term_months))


@dataclass
class CashflowPoint:
    month: int
    balance_start: float
    interest: float
    scheduled_prin: float
    prepay_prin: float
    total_prin: float
    total_cf: float
    df: float
    pv: float


@dataclass
class PriceResult:
    price: float
    yield_annual: float
    cpr_annual: float
    term_months: int
    cashflows: List[CashflowPoint]
    macaulay_duration: float
    modified_duration: float
    convexity: float


# -------- PV / duration / convexity engine --------

def price_with_detail(
    principal: float = 100.0,
    coupon_rate_annual: float = 0.06,
    cpr_annual: float = 0.10,
    yield_annual: float = 0.05,
    term_months: int = 360,
) -> PriceResult:
    """Return price, detailed cashflows, duration, and convexity."""
    payment = level_payment(principal, coupon_rate_annual, term_months)
    y_m = yield_annual / 12.0
    smm = cpr_to_smm(cpr_annual)

    cashflows: List[CashflowPoint] = []
    bal = principal
    price = 0.0
    t_pv_sum = 0.0       # for duration
    t2_pv_sum = 0.0      # for convexity

    for m in range(1, term_months + 1):
        if bal <= 0:
            break

        interest = bal * (coupon_rate_annual / 12.0)
        sched_p = payment - interest
        bal_after_sched = bal - sched_p
        prepay = bal_after_sched * smm
        total_p = sched_p + prepay
        cf = interest + total_p

        df = (1 + y_m) ** -m
        pv = cf * df

        price += pv
        t_pv_sum += m * pv
        t2_pv_sum += (m ** 2) * pv

        cashflows.append(
            CashflowPoint(
                month=m,
                balance_start=bal,
                interest=interest,
                scheduled_prin=sched_p,
                prepay_prin=prepay,
                total_prin=total_p,
                total_cf=cf,
                df=df,
                pv=pv,
            )
        )
        bal -= total_p

    if price == 0:
        macaulay = 0.0
        modified = 0.0
        convexity = 0.0
    else:
        # time in years
        macaulay = (t_pv_sum / price) / 12.0
        modified = macaulay / (1 + yield_annual)
        convexity = (t2_pv_sum / price) / (12.0 ** 2)

    return PriceResult(
        price=price,
        yield_annual=yield_annual,
        cpr_annual=cpr_annual,
        term_months=term_months,
        cashflows=cashflows,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
    )


def implied_yield_for_price(
    target_price: float,
    coupon_rate_annual: float,
    cpr_annual: float,
    term_months: int = 360,
    initial_yield: float = 0.05,
    tolerance: float = 0.01,
) -> float:
    """Solve for yield such that model price ≈ target_price (very simple solver)."""
    y = initial_yield
    for _ in range(40):
        pr = price_with_detail(
            principal=100.0,
            coupon_rate_annual=coupon_rate_annual,
            cpr_annual=cpr_annual,
            yield_annual=y,
            term_months=term_months,
        ).price
        diff = pr - target_price
        if abs(diff) < tolerance:
            break
        # crude step: 5 bps per 1 point of price error
        y += diff * 0.0005
    return y


def spec_payup_from_cpr(
    tba_price: float,
    tba_cpr: float,
    spec_cpr: float,
    coupon_rate_annual: float,
    term_months: int = 360,
) -> Dict[str, float]:
    """Given TBA price/CPR and spec CPR, return spec price & payup using a consistent yield curve."""
    y = implied_yield_for_price(
        target_price=tba_price,
        coupon_rate_annual=coupon_rate_annual,
        cpr_annual=tba_cpr,
        term_months=term_months,
    )

    tba_result = price_with_detail(
        principal=100.0,
        coupon_rate_annual=coupon_rate_annual,
        cpr_annual=tba_cpr,
        yield_annual=y,
        term_months=term_months,
    )
    spec_result = price_with_detail(
        principal=100.0,
        coupon_rate_annual=coupon_rate_annual,
        cpr_annual=spec_cpr,
        yield_annual=y,
        term_months=term_months,
    )

    payup_points = spec_result.price - tba_result.price
    payup_32nds = round(payup_points * 32)

    return {
        "yield_annual": y,
        "tba_price_model": tba_result.price,
        "spec_price_model": spec_result.price,
        "tba_cpr": tba_cpr,
        "spec_cpr": spec_cpr,
        "payup_points": payup_points,
        "payup_32nds": payup_32nds,
        "tba_duration": tba_result.modified_duration,
        "tba_convexity": tba_result.convexity,
        "spec_duration": spec_result.modified_duration,
        "spec_convexity": spec_result.convexity,
    }

# -------- Story → CPR mapping (very toy / adjustable) --------

def story_base_cpr_from_balance(bucket: str) -> float:
    """
    Map balance bucket to a base CPR.
    Example inputs: 'LLB', 'MidBal', 'HiBal', 'JumboConf'
    """
    bucket = bucket.upper()
    if "LLB" in bucket:
        return 0.06   # 6% CPR
    if "MID" in bucket:
        return 0.09   # 9%
    if "HI" in bucket or "HLB" in bucket:
        return 0.11   # 11%
    if "JUMBO" in bucket or "CONF" in bucket:
        return 0.10   # 10%
    return 0.10       # fallback generic


def story_cpr_from_traits(
    balance_bucket: str,
    wala_months: float,
    avg_fico: float,
    avg_ltv: float,
    slow_geo: bool,
    fast_geo: bool,
    investor_share: float,
    bank_originator_share: float,
    wac_vs_market_bps: float,
) -> float:
    """
    Coarse CPR mapping from pool traits.
    All adjustments are indicative; feel free to tweak.
    """
    cpr = story_base_cpr_from_balance(balance_bucket)

    # Seasoning / burnout
    if wala_months >= 36:
        cpr -= 0.02  # burnout
    elif wala_months >= 12:
        cpr -= 0.01  # moderate seasoning
    else:
        cpr += 0.00

    # Credit / LTV
    if avg_fico < 700 or avg_ltv > 85:
        cpr -= 0.01  # credit constrained
    elif avg_fico > 760 and avg_ltv < 75:
        cpr += 0.01  # cleaner, faster

    # Geo
    if slow_geo:
        cpr -= 0.01
    if fast_geo:
        cpr += 0.02

    # Occupancy
    if investor_share >= 0.2:
        cpr -= 0.02

    # Originator type
    if bank_originator_share >= 0.5:
        cpr -= 0.005

    # WAC vs market (in bps)
    if wac_vs_market_bps > 100:
        cpr += 0.04   # deep refi incentive
    elif wac_vs_market_bps > 50:
        cpr += 0.02
    elif wac_vs_market_bps < 0:
        cpr -= 0.01

    # Clip to reasonable range
    cpr = max(0.01, min(0.30, cpr))
    return cpr
