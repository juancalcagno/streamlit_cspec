# spec_payup_app.py

import os
import glob
import streamlit as st
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Literal

from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    ColumnsAutoSizeMode,
    JsCode,
)

import spec_grid_model as grid  # your Fannie-style grid model
import cpr_payup_model as cprm

Tier = Literal["Tier 1+", "Tier 1", "Tier 2", "Tier 3", "Generic"]


# ---------- Helper for st-aggrid selection ----------

def safe_selected_rows(grid_response):
    """
    Normalize st-aggrid's selected_rows to a list of dicts.
    Handles [], list-of-dicts, and DataFrame cases.
    """
    selected = grid_response.get("selected_rows", [])

    if isinstance(selected, list):
        return selected

    try:
        if isinstance(selected, pd.DataFrame):
            return selected.to_dict("records")
    except Exception:
        pass

    return [selected] if selected else []


# ---------- Dataclasses ----------

@dataclass
class PoolInputs:
    avg_balance: float
    wala_months: float
    wac: float
    current_market_rate: float
    avg_fico: float
    avg_ltv: float
    slow_geo_share: float
    fast_geo_share: float
    investor_share: float
    bank_originator_share: float


@dataclass
class PayupResult:
    total_payup_32nds: int
    payup_components_32nds: Dict[str, int]
    spec_price: float
    tier: Tier


# ---------- Bucketing helpers (heuristic model) ----------

def bucket_balance(avg_balance: float) -> str:
    if avg_balance < 175_000:
        return "LLB"
    elif avg_balance < 225_000:
        return "MidBal"
    elif avg_balance < 300_000:
        return "HiBal"
    else:
        return "JumboConf"


def bucket_seasoning(wala_months: float) -> str:
    if wala_months < 6:
        return "New"
    elif wala_months < 24:
        return "Seasoned"
    else:
        return "Burnout"


def bucket_wac_spread(wac: float, market_rate: float) -> str:
    spread = wac - market_rate
    if spread < 0.25:
        return "LowWAC"
    elif spread < 0.75:
        return "AtMarket"
    else:
        return "HighWAC"


# ---------- Component payups (32nds) for heuristic model ----------

def balance_payup_32nds(avg_balance: float) -> int:
    bucket = bucket_balance(avg_balance)
    mapping = {
        "LLB": 18,
        "MidBal": 10,
        "HiBal": 4,
        "JumboConf": 1,
    }
    return mapping[bucket]


def seasoning_payup_32nds(wala_months: float) -> int:
    bucket = bucket_seasoning(wala_months)
    mapping = {
        "New": 0,
        "Seasoned": 4,
        "Burnout": 8,
    }
    return mapping[bucket]


def wac_payup_32nds(wac: float, market_rate: float) -> int:
    bucket = bucket_wac_spread(wac, market_rate)
    mapping = {
        "LowWAC": 6,
        "AtMarket": 0,
        "HighWAC": -4,
    }
    return mapping[bucket]


def geo_payup_32nds(slow_geo_share: float, fast_geo_share: float) -> int:
    slow_geo_share = max(0.0, min(1.0, slow_geo_share))
    fast_geo_share = max(0.0, min(1.0, fast_geo_share))
    slow_component = round(6 * slow_geo_share)
    fast_component = round(-4 * fast_geo_share)
    return slow_component + fast_component


def credit_payup_32nds(avg_fico: float, avg_ltv: float) -> int:
    avg_fico = max(500, min(850, avg_fico))
    avg_ltv = max(40, min(100, avg_ltv))

    fico_friction = (740 - avg_fico) / 40.0
    ltv_friction = (avg_ltv - 75) / 10.0

    friction_score = 0.8 * fico_friction + 0.6 * ltv_friction
    return int(round(friction_score))


def investor_payup_32nds(investor_share: float) -> int:
    investor_share = max(0.0, min(1.0, investor_share))
    return int(round(6 * investor_share))


def originator_payup_32nds(bank_originator_share: float) -> int:
    bank_originator_share = max(0.0, min(1.0, bank_originator_share))
    return int(round(4 * bank_originator_share))


def classify_tier(total_payup_32nds: int) -> Tier:
    if total_payup_32nds >= 24:
        return "Tier 1+"
    elif total_payup_32nds >= 16:
        return "Tier 1"
    elif total_payup_32nds >= 8:
        return "Tier 2"
    elif total_payup_32nds > 0:
        return "Tier 3"
    else:
        return "Generic"


def estimate_spec_payup_and_tier(pool: PoolInputs, tba_price: float) -> PayupResult:
    components: Dict[str, int] = {}
    components["balance"] = balance_payup_32nds(pool.avg_balance)
    components["seasoning"] = seasoning_payup_32nds(pool.wala_months)
    components["wac"] = wac_payup_32nds(pool.wac, pool.current_market_rate)
    components["geo"] = geo_payup_32nds(pool.slow_geo_share, pool.fast_geo_share)
    components["credit"] = credit_payup_32nds(pool.avg_fico, pool.avg_ltv)
    components["investor"] = investor_payup_32nds(pool.investor_share)
    components["originator"] = originator_payup_32nds(pool.bank_originator_share)

    total_payup_32nds = sum(components.values())
    payup_points = total_payup_32nds / 32.0
    spec_price = tba_price + payup_points
    tier = classify_tier(total_payup_32nds)

    return PayupResult(
        total_payup_32nds=total_payup_32nds,
        payup_components_32nds=components,
        spec_price=round(spec_price, 3),
        tier=tier,
    )


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(layout="wide")
    st.title("MBS Spec Payup Explorer")

    # Load grid once (if CSV present)
    try:
        grid.load_payup_grid("spec_payup_grid.csv")
        grid_loaded = True
    except Exception as e:
        grid_loaded = False
        st.sidebar.warning(f"Grid not loaded: {e}")

    mode = st.sidebar.radio(
        "Mode",
        [
            "Heuristic model (sliders)",
            "Grid lookup (Fannie-style)",
            "Full grid view (Excel-style)",
            "Snapshot comparison (Δ vs prior)",
            "CPR-based analytics & Story Builder",
        ],
    )
    def _to_tier_safe(v) -> str:
       if pd.isna(v):
          return ""
       try:
          return grid.classify_tier(int(v))
       except Exception:
          return ""

    # ---------- Mode 1: Heuristic model (sliders) ----------

    if mode == "Heuristic model (sliders)":
        st.write(
            "Toy heuristic payup model: tweak pool traits and see payup, tier, "
            "and spec price vs TBA."
        )

        st.sidebar.header("TBA and Market Inputs")
        tba_price = st.sidebar.number_input("TBA Price (points)", value=100.0, step=0.25)
        current_market_rate = st.sidebar.number_input(
            "Current Primary Mortgage Rate (%)", value=6.0, step=0.125
        )

        st.sidebar.header("Pool Characteristics")

        avg_balance = st.sidebar.number_input(
            "Average Loan Balance (USD)",
            min_value=50_000.0,
            max_value=1_000_000.0,
            value=160_000.0,
            step=5_000.0,
            format="%.0f",
        )

        wala_months = st.sidebar.slider(
            "WALA (months)", min_value=0.0, max_value=120.0, value=26.0, step=1.0
        )

        wac = st.sidebar.number_input(
            "Weighted Average Coupon (WAC, %)",
            min_value=1.0,
            max_value=12.0,
            value=6.25,
            step=0.125,
        )

        avg_fico = st.sidebar.slider(
            "Average FICO", min_value=500, max_value=850, value=705, step=5
        )

        avg_ltv = st.sidebar.slider(
            "Average LTV (%)", min_value=40, max_value=100, value=82, step=1
        )

        slow_geo_share = st.sidebar.slider(
            "Slow Geo Share (TX, NY, CA, etc.)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )

        fast_geo_share = st.sidebar.slider(
            "Fast Geo Share (FL, AZ, etc.)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )

        investor_share = st.sidebar.slider(
            "Investor / Second Home Share",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

        bank_originator_share = st.sidebar.slider(
            "Bank Originator Share",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

        pool = PoolInputs(
            avg_balance=avg_balance,
            wala_months=wala_months,
            wac=wac,
            current_market_rate=current_market_rate,
            avg_fico=avg_fico,
            avg_ltv=avg_ltv,
            slow_geo_share=slow_geo_share,
            fast_geo_share=fast_geo_share,
            investor_share=investor_share,
            bank_originator_share=bank_originator_share,
        )

        result = estimate_spec_payup_and_tier(pool, tba_price)

        st.subheader("Results")
        st.metric(
            "Total Spec Payup (32nds)",
            f"{result.total_payup_32nds}",
            help="Sum of all story components in 32nds of a point.",
        )
        st.metric("Total Spec Payup (points)", f"{result.total_payup_32nds / 32.0:.3f}")
        st.metric("Spec Pool Price", f"{result.spec_price:.3f}")
        st.metric("Tier Classification", result.tier)

        st.subheader("Component Breakdown (32nds)")
        st.json(result.payup_components_32nds)

        st.subheader("Raw Inputs")
        st.json(asdict(pool))

    # ---------- Mode 2: Fannie-style grid lookup ----------

    elif mode == "Grid lookup (Fannie-style)":
        st.write("Grid lookup mode: coupon rows 1–8, stories as columns, payups loaded from CSV.")

        if not grid_loaded:
            st.error("Grid not loaded. Make sure 'spec_payup_grid.csv' exists in the same folder.")
            return

        buckets = sorted(grid.PAYUP_GRID_32NDS.keys())
        coupon_bucket = st.sidebar.selectbox("Coupon bucket (row 1–8)", options=buckets, index=0)

        stories = sorted(grid.PAYUP_GRID_32NDS[coupon_bucket].keys())
        story_code = st.sidebar.selectbox("Story (column)", options=stories)

        quote = grid.get_payup(coupon_bucket, story_code)

        st.subheader("Grid Payup")
        if quote.coupon_rate is not None:
            st.write(f"Coupon rate: {quote.coupon_rate}%")
        st.write(f"Coupon bucket: {quote.coupon_bucket}")
        st.write(f"Story: {quote.story_code}")
        st.write(f"Payup: {quote.payup_32nds} /32nds ({quote.payup_points:.3f} pts)")
        st.write(f"Tier: {quote.tier}")

        st.subheader("All stories for this coupon bucket")
        quotes = grid.list_stories_for_coupon(coupon_bucket)
        tier_counts: Dict[Tier, int] = {}
        for q in quotes:
            tier_counts[q.tier] = tier_counts.get(q.tier, 0) + 1

        st.write("Counts by tier:")
        st.json(tier_counts)

    # ---------- Mode 3: Full grid view (Excel-style, AgGrid) ----------

    elif mode == "Full grid view (Excel-style)":
        st.write("Full spec payup grid loaded from CSV. Rows = coupon buckets (1–8), columns = stories.")

        if not grid_loaded:
            st.error("Grid not loaded. Make sure 'spec_payup_grid.csv' exists in the same folder.")
            return

        df = pd.read_csv("spec_payup_grid.csv")

        st.subheader("Payup Grid (32nds) – interactive")

        col_left, col_right = st.columns([3, 1])
        with col_left:
            search_text = st.text_input("Search in grid (any column)", value="")
        with col_right:
            big_view = st.checkbox("Enlarge", value=True)

        view_df = df.copy()
        if search_text:
            mask = pd.DataFrame(
                {
                    col: view_df[col].astype(str).str.contains(search_text, case=False, na=False)
                    for col in view_df.columns
                }
            ).any(axis=1)
            view_df = view_df[mask]

        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download current grid as CSV",
            data=csv_bytes,
            file_name="spec_payup_grid_filtered.csv",
            mime="text/csv",
        )

        gb = GridOptionsBuilder.from_dataframe(view_df)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        gb.configure_grid_options(enableRangeSelection=True, domLayout="normal")
        grid_options = gb.build()

        height = 700 if big_view else 450

        AgGrid(
            view_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            height=height,
        )

        # Tier view
        st.subheader("Tier View (by cell) – interactive")

        tier_df = df.copy()
        story_cols = [c for c in df.columns if c not in ("coupon_bucket", "coupon_rate")]

        for col in story_cols:
            tier_df[col] = tier_df[col].apply(grid.classify_tier)

        gb_tier = GridOptionsBuilder.from_dataframe(tier_df)
        gb_tier.configure_default_column(resizable=True, sortable=True, filter=True)
        gb_tier.configure_grid_options(enableRangeSelection=True, domLayout="normal")
        grid_options_tier = gb_tier.build()

        AgGrid(
            tier_df,
            gridOptions=grid_options_tier,
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            height=height,
        )

    # ---------- Mode 4: Snapshot comparison (Δ vs prior) ----------

    elif mode == "Snapshot comparison (Δ vs prior)":
        st.write(
            "Compare payups between two snapshots (as-of dates) and see changes in a wide grid "
            "layout (coupons × stories), plus a detailed long-form view with history."
        )

        # ---- 1. Discover snapshot CSVs ----
        snapshot_files = sorted(glob.glob("spec_payup_grid_*.csv"))
        if len(snapshot_files) < 2:
            st.error(
                "Need at least two snapshot CSVs named like 'spec_payup_grid_YYYYMMDD.csv' "
                "in this folder to do comparisons."
            )
            return

        def label_for(path: str) -> str:
            base = os.path.basename(path)
            date_part = base.replace("spec_payup_grid_", "").replace(".csv", "")
            if len(date_part) == 8 and date_part.isdigit():
                return f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"
            return base

        labels = [label_for(p) for p in snapshot_files]
        label_to_path = dict(zip(labels, snapshot_files))

        story_cpr_map = None  # used when comparing against CPR model

        # ---- 2. Choose comparison basis (previous snapshot vs CPR model) ----
        compare_basis = st.radio(
            "Compare current payups against:",
            ["Previous snapshot", "CPR model (prepayment valuation)"],
            index=0,
        )

        # Pick current snapshot in all cases
        col_curr, col_prev = st.columns(2)
        with col_curr:
            curr_label = st.selectbox(
                "Current snapshot (as-of)", options=labels, index=len(labels) - 1
            )

        curr_path = label_to_path[curr_label]

        # If comparing to a previous snapshot, let user choose it
        if compare_basis == "Previous snapshot":
            with col_prev:
                prev_index_default = max(0, labels.index(curr_label) - 1)
                prev_label = st.selectbox(
                    "Compare to snapshot", options=labels, index=prev_index_default
                )
            prev_path = label_to_path[prev_label]

            st.markdown(
                f"- New: **{curr_label}** (`{os.path.basename(curr_path)}`)\n"
                f"- Curr (baseline): **{prev_label}** (`{os.path.basename(prev_path)}`)"
            )

        # If comparing to a CPR-based model, configure model inputs in the sidebar
        else:
            prev_label = "CPR model"
            prev_path = None

            st.markdown(
                f"- Actual: **{curr_label}** (`{os.path.basename(curr_path)}`)\n"
                "- Model: **CPR model** (synthetic, from prepayment-based valuation)"
            )

            st.sidebar.header("CPR model inputs (snapshot comparison)")
            term_choice_snap = st.sidebar.radio(
                "Pool term (for model pricing)",
                options=["30-year", "20-year", "15-year", "10-year"],
                index=0,
            )
            term_map_snap = {
                "30-year": 360,
                "20-year": 240,
                "15-year": 180,
                "10-year": 120,
            }
            term_months_snap = term_map_snap[term_choice_snap]

            tba_price_snap = st.sidebar.number_input(
                "TBA Price (points, snapshot comparison)",
                min_value=80.0,
                max_value=110.0,
                value=100.0,
                step=0.25,
            )
            tba_cpr_snap = st.sidebar.slider(
                "Generic TBA CPR (annualized, snapshot comparison)",
                min_value=0.01,
                max_value=0.25,
                value=0.12,
                step=0.01,
            )

        # ---- 3. Load current snapshot and build comparison baseline (wide) ----
        df_curr = pd.read_csv(curr_path)

        id_cols = ["coupon_bucket", "coupon_rate"]
        story_cols = [c for c in df_curr.columns if c not in id_cols]

        # Sort current snapshot so rows align deterministically
        df_curr = df_curr.sort_values(id_cols).reset_index(drop=True)

        # Branch: either use a previous snapshot, or a CPR-based model as the baseline
        if compare_basis == "Previous snapshot":
            df_prev = pd.read_csv(prev_path)
            df_prev = df_prev.sort_values(id_cols).reset_index(drop=True)

        else:
            df_model = df_curr.copy()

            # Simple heuristic CPR per story (you can refine later)
            story_cpr_map = {}
            for s in story_cols:
                s_up = s.upper()
                if "LLB" in s_up:
                    story_cpr_map[s] = 0.06
                elif "MID" in s_up:
                    story_cpr_map[s] = 0.09
                elif "INV" in s_up or "INVEST" in s_up:
                    story_cpr_map[s] = 0.05
                elif "SEAS" in s_up or "BURN" in s_up:
                    story_cpr_map[s] = 0.07
                else:
                    story_cpr_map[s] = 0.10  # generic fallback

            # For each coupon row and story, compute a model payup in 32nds
            for idx, row in df_curr.iterrows():
                coupon_rate_annual = row["coupon_rate"] / 100.0
                for s in story_cols:
                    spec_cpr_story = story_cpr_map[s]
                    try:
                        res_story = cprm.spec_payup_from_cpr(
                            tba_price=tba_price_snap,
                            tba_cpr=tba_cpr_snap,
                            spec_cpr=spec_cpr_story,
                            coupon_rate_annual=coupon_rate_annual,
                            term_months=term_months_snap,
                        )
                        df_model.at[idx, s] = res_story["payup_32nds"]
                    except Exception:
                        df_model.at[idx, s] = pd.NA

            df_prev = df_model

        # ---- 4. Build a single wide comparison grid ----
        # Columns: id_cols + for each story: story_curr, story_prev, story_delta
        df_compare = df_curr[id_cols].copy()

        for story in story_cols:
            curr_col = f"{story}_curr"
            prev_col = f"{story}_prev"
            delta_col = f"{story}_delta"

            df_compare[curr_col] = df_curr[story]
            if story in df_prev.columns:
                df_compare[prev_col] = df_prev[story]
                df_compare[delta_col] = df_curr[story].fillna(0) - df_prev[story].fillna(0)
            else:
                df_compare[prev_col] = pd.NA
                df_compare[delta_col] = df_curr[story].fillna(0)

        # ---- 5. Tabs: wide comparison grid + long-form view ----
        if compare_basis == "Previous snapshot":
            left_label = "Curr"       # baseline (older)
            right_label = "New"       # newest snapshot
            delta_label = "Δ"
            wide_subtitle = "Wide comparison grid (coupons × stories, Curr / New / Δ)"
        else:
            left_label = "Model"
            right_label = "Actual"
            delta_label = "Δ"
            wide_subtitle = "Wide comparison grid (coupons × stories, Model / Actual / Δ)"

        tab_wide, tab_long = st.tabs(["Wide comparison grid", "Long view + history"])
    
        # ---------- TAB 1: Wide comparison grid ----------
        with tab_wide:
            st.subheader(wide_subtitle)

            # Optional search on the key columns
            search_text = st.text_input("Search by coupon bucket or rate", value="")
            view_df = df_compare.copy()
            if search_text:
                mask = pd.DataFrame(
                    {
                        col: view_df[col].astype(str).str.contains(search_text, case=False, na=False)
                        for col in ["coupon_bucket", "coupon_rate"]
                    }
                ).any(axis=1)
                view_df = view_df[mask]

            # ---- Tier controls ----
            tier_options = ["Tier 1+", "Tier 1", "Tier 2", "Tier 3", "Generic"]
            show_tiers = st.checkbox("Show tiers (instead of payups)", value=False)
            enabled_tiers = st.multiselect(
                "Tiers to show (hides whole stories that have no cells in these tiers)",
                options=tier_options,
                default=tier_options,
            )
            enabled_tiers = set(enabled_tiers)

            def _to_tier_safe(v) -> str:
                if pd.isna(v):
                    return ""
                try:
                    return grid.classify_tier(int(v))
                except Exception:
                    return ""

            # --- 1) Decide which stories are visible ---
            # Rule: show story if ANY row's (New/Actual) tier is in enabled_tiers.
            story_cols_visible = []
            for story in story_cols:
                curr_col = f"{story}_curr"
                if curr_col not in view_df.columns:
                    continue
                tiers_series = view_df[curr_col].apply(_to_tier_safe)
                if tiers_series.isin(enabled_tiers).any():
                    story_cols_visible.append(story)

            # --- 2) Mask / render for visible stories only ---
            if show_tiers:
                # Render tiers as strings; blank out tiers not selected
                for story in story_cols_visible:
                    for suffix in ("_prev", "_curr"):
                        col = f"{story}{suffix}"
                        if col in view_df.columns:
                            view_df[col] = view_df[col].apply(_to_tier_safe)
                            view_df[col] = view_df[col].apply(lambda t: t if t in enabled_tiers else "")
            else:
                # Keep numeric values only where curr tier is enabled; blank prev/curr/delta otherwise
                for story in story_cols_visible:
                    curr_col = f"{story}_curr"
                    prev_col = f"{story}_prev"
                    del_col  = f"{story}_delta"

                    tiers_series = view_df[curr_col].apply(_to_tier_safe)
                    mask_keep = tiers_series.isin(enabled_tiers)
                    view_df.loc[~mask_keep, [prev_col, curr_col, del_col]] = pd.NA

            # --- 3) Drop whole story groups not visible ---
            for story in story_cols:
                if story in story_cols_visible:
                    continue
                drop_cols = [
                    c for c in (f"{story}_prev", f"{story}_curr", f"{story}_delta")
                    if c in view_df.columns
                ]
                if drop_cols:
                    view_df = view_df.drop(columns=drop_cols)

            st.caption(f"Stories shown: {len(story_cols_visible)} / {len(story_cols)}")

            # Download current wide comparison view
            csv_bytes = view_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download wide comparison grid as CSV",
                data=csv_bytes,
                file_name=f"spec_payup_wide_compare_{curr_label}_vs_{prev_label}.csv",
                mime="text/csv",
            )

            # JS for delta cell coloring
            delta_cell_style = JsCode(
                """
                function(params) {
                    if (params.value > 0) {
                        return {color: 'green', fontWeight: 'bold'};
                    } else if (params.value < 0) {
                        return {color: 'red', fontWeight: 'bold'};
                    } else {
                        return {};
                    }
                }
                """
            )

            # Build grid options
            gb_wide = GridOptionsBuilder.from_dataframe(view_df)
            gb_wide.configure_default_column(
                resizable=True,
                sortable=True,
                filter=True,
                minWidth=90,
                wrapHeaderText=True,
                autoHeaderHeight=True,
            )
            gb_wide.configure_selection("single")
            gb_wide.configure_grid_options(enableRangeSelection=True, domLayout="normal")
            grid_options_wide = gb_wide.build()

            # Column defs
            column_defs = [
                {
                    "headerName": "Key",
                    "children": [
                        {"field": "coupon_rate", "headerName": "Coupon", "pinned": "left", "minWidth": 110},
                    ],
                }
            ]

            # IMPORTANT: iterate only visible stories
            for i, story in enumerate(story_cols_visible):
                prev_field = f"{story}_prev"
                curr_field = f"{story}_curr"
                delta_field = f"{story}_delta"

                header_class = "story-group-even" if i % 2 == 0 else "story-group-odd"
                cell_class = "story-group-even-cell" if i % 2 == 0 else "story-group-odd-cell"

                column_defs.append(
                    {
                        "headerName": story,
                        "headerClass": header_class,
                        "children": [
                            {"field": prev_field, "headerName": left_label, "width": 95, "minWidth": 90,
                            "headerClass": header_class, "cellClass": cell_class},
                            {"field": curr_field, "headerName": right_label, "width": 95, "minWidth": 90,
                            "headerClass": header_class, "cellClass": cell_class},
                            {"field": delta_field, "headerName": "Δ", "width": 95, "minWidth": 90,
                            "cellStyle": delta_cell_style, "headerClass": header_class, "cellClass": cell_class},
                        ],
                    }
                )

            grid_options_wide["columnDefs"] = column_defs

            custom_css = {
                ".ag-header-group-cell-label": {"justify-content": "center"},
                ".story-group-even": {
                    "background-color": "#111827 !important",
                    "border-right": "2px solid #4b5563 !important",
                    "color": "#f9fafb !important",
                },
                ".story-group-odd": {
                    "background-color": "#1f2937 !important",
                    "border-right": "2px solid #4b5563 !important",
                    "color": "#f9fafb !important",
                },
                ".story-group-even-cell": {
                    "background-color": "rgba(31,41,55,0.35) !important",
                    "border-right": "1px solid #374151 !important",
                },
                ".story-group-odd-cell": {
                    "background-color": "rgba(17,24,39,0.35) !important",
                    "border-right": "1px solid #374151 !important",
                },
                ".ag-cell-focus": {
                    "border": "2px solid #3b82f6 !important",
                    "background-color": "#1d4ed8 !important",
                    "color": "#f9fafb !important",
                },
                ".ag-row.ag-row-selected .ag-cell": {
                    "background-color": "#78350f !important",
                    "color": "#fefce8 !important",
                },
                ".ag-row.ag-row-selected .ag-cell.ag-cell-focus": {
                    "border": "2px solid #fbbf24 !important",
                    "background-color": "#b91c1c !important",
                    "color": "#fefce8 !important",
                },
            }

            AgGrid(
                view_df,
                gridOptions=grid_options_wide,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
                height=650,
                allow_unsafe_jscode=True,
                theme="alpine",
                custom_css=custom_css,
            )

        # ---------- TAB 2: Long-form comparison + history ----------
        with tab_long:
            st.subheader("Long-form comparison table + historical payups for selected story")

            curr_long = df_curr.melt(
                id_vars=id_cols,
                value_vars=story_cols,
                var_name="story_code",
                value_name="payup_curr",
            )
            prev_long = df_prev.melt(
                id_vars=id_cols,
                value_vars=story_cols,
                var_name="story_code",
                value_name="payup_prev",
            )

            merged = curr_long.merge(prev_long, on=id_cols + ["story_code"], how="outer")

            merged["payup_curr"] = merged["payup_curr"].fillna(0).astype(int)
            merged["payup_prev"] = merged["payup_prev"].fillna(0).astype(int)
            merged["delta_32nds"] = merged["payup_curr"] - merged["payup_prev"]

            merged["tier_curr"] = merged["payup_curr"].apply(grid.classify_tier)
            merged["tier_prev"] = merged["payup_prev"].apply(grid.classify_tier)

            merged = merged.sort_values(["coupon_bucket", "coupon_rate", "story_code"]).reset_index(drop=True)

            search_text_long = st.text_input("Search (coupon, rate, or story)", value="")
            view_long = merged.copy()
            if search_text_long:
                mask = pd.DataFrame(
                    {
                        col: view_long[col].astype(str).str.contains(search_text_long, case=False, na=False)
                        for col in ["coupon_bucket", "coupon_rate", "story_code"]
                    }
                ).any(axis=1)
                view_long = view_long[mask]

            csv_bytes_long = view_long.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download long-form comparison as CSV",
                data=csv_bytes_long,
                file_name=f"spec_payup_long_compare_{curr_label}_vs_{prev_label}.csv",
                mime="text/csv",
            )

            delta_cell_style_long = JsCode(
                """
                function(params) {
                    if (params.value > 0) {
                        return {color: 'green', fontWeight: 'bold'};
                    } else if (params.value < 0) {
                        return {color: 'red', fontWeight: 'bold'};
                    } else {
                        return {};
                    }
                }
                """
            )

            row_highlight_js_long = JsCode(
                """
                function(params) {
                    if (params.node != null && params.node.selected) {
                        return {
                            backgroundColor: '#fff3cd',
                            fontWeight: 'bold'
                        };
                    }
                    return {};
                }
                """
            )

            gb_long = GridOptionsBuilder.from_dataframe(view_long)
            gb_long.configure_default_column(resizable=True, sortable=True, filter=True)
            gb_long.configure_selection("single")
            gb_long.configure_grid_options(enableRangeSelection=True, domLayout="normal")
            grid_options_long = gb_long.build()

            grid_options_long["getRowStyle"] = row_highlight_js_long
            grid_options_long["columnDefs"] = [
                {
                    "headerName": "Key",
                    "children": [
                        {"field": "coupon_bucket", "pinned": "left"},
                        {"field": "coupon_rate", "pinned": "left"},
                        {"field": "story_code", "pinned": "left"},
                    ],
                },
                {
                    "headerName": f"Current ({curr_label})",
                    "children": [
                        {"field": "payup_curr", "headerName": "Payup (32nds)", "type": "numericColumn"},
                        {"field": "tier_curr", "headerName": "Tier"},
                    ],
                },
                {
                    "headerName": f"Previous ({prev_label})",
                    "children": [
                        {"field": "payup_prev", "headerName": "Payup (32nds)", "type": "numericColumn"},
                        {"field": "tier_prev", "headerName": "Tier"},
                    ],
                },
                {
                    "headerName": "Change",
                    "children": [
                        {
                            "field": "delta_32nds",
                            "headerName": "Δ 32nds",
                            "type": "numericColumn",
                            "cellStyle": delta_cell_style_long,
                        },
                    ],
                },
            ]

            grid_response_long = AgGrid(
                view_long,
                gridOptions=grid_options_long,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                height=600,
                allow_unsafe_jscode=True,
            )

            st.subheader("Historical payups for selected story (across all snapshots)")

            selected_rows = safe_selected_rows(grid_response_long)

            if not selected_rows:
                st.info(
                    "Select a row in the grid above to see the time-series "
                    "for that story/coupon across all snapshots."
                )
            else:
                sel = selected_rows[0]
                sel_bucket = sel["coupon_bucket"]
                sel_rate = sel["coupon_rate"]
                sel_story = sel["story_code"]

                st.markdown(
                    f"**Story:** `{sel_story}` &nbsp;&nbsp; "
                    f"**Coupon bucket:** `{sel_bucket}` &nbsp;&nbsp; "
                    f"**Coupon rate:** `{sel_rate}`"
                )

                history_rows = []
                for path in snapshot_files:
                    lab = label_for(path)
                    df_snap = pd.read_csv(path)

                    if sel_story not in df_snap.columns:
                        continue

                    matches = df_snap[
                        (df_snap["coupon_bucket"] == sel_bucket)
                        & (df_snap["coupon_rate"] == sel_rate)
                    ]
                    if matches.empty:
                        continue

                    payup_val = matches[sel_story].iloc[0]
                    history_rows.append({"as_of": lab, "payup_32nds": payup_val})

                if not history_rows:
                    st.warning("No history found for this story/coupon across the available snapshots.")
                else:
                    history_df = pd.DataFrame(history_rows).sort_values("as_of")
                    history_df["as_of"] = pd.to_datetime(history_df["as_of"])
                    history_df = history_df.set_index("as_of")

                    st.line_chart(history_df["payup_32nds"])

                    # Optional: explanation when CPR model is used as baseline
                    if story_cpr_map is not None and compare_basis == "CPR model (prepayment valuation)":
                        s_code = sel_story
                        spec_cpr_story = story_cpr_map.get(s_code)
                        if spec_cpr_story is not None:
                            delta_cpr_bps = (spec_cpr_story - tba_cpr_snap) * 10000
                            direction = "slower" if delta_cpr_bps < 0 else "faster"
                            st.markdown(
                                f"**Model rationale:** Story `{s_code}` is modeled at "
                                f"{spec_cpr_story*100:.1f}% CPR vs {tba_cpr_snap*100:.1f}% for TBA "
                                f"({abs(delta_cpr_bps):.0f} bps {direction} prepayments)."
                            )

    # ---------- Mode 5: CPR-based analytics & Story Builder ----------
    elif mode == "CPR-based analytics & Story Builder":
        st.title("CPR-based Spec Payup Analytics & Story Builder")

        st.sidebar.header("Term (Tier)")
        term_choice = st.sidebar.radio(
            "Pool term",
            options=["30-year", "20-year", "15-year", "10-year"],
            index=0,
        )
        term_map = {"30-year": 360, "20-year": 240, "15-year": 180, "10-year": 120}
        term_months = term_map[term_choice]

        st.sidebar.header("TBA Inputs")
        coupon = st.sidebar.number_input(
            "MBS Coupon (%)", min_value=1.0, max_value=10.0, value=6.0, step=0.25
        ) / 100.0

        tba_price = st.sidebar.number_input(
            "TBA Price (points)",
            min_value=80.0,
            max_value=110.0,
            value=100.0,
            step=0.25,
        )

        tba_cpr = st.sidebar.slider(
            "Generic TBA CPR (annualized)",
            min_value=0.01,
            max_value=0.25,
            value=0.12,
            step=0.01,
        )

        st.sidebar.header("Story Builder – Pool Traits")

        balance_bucket = st.sidebar.selectbox(
            "Balance bucket (story-like)",
            options=["LLB", "MidBal", "HiBal", "JumboConf"],
            index=0,
        )

        wala = st.sidebar.slider("WALA (months)", min_value=0, max_value=120, value=36, step=1)

        avg_fico = st.sidebar.slider("Average FICO", min_value=500, max_value=850, value=720, step=5)

        avg_ltv = st.sidebar.slider("Average LTV (%)", min_value=40, max_value=100, value=82, step=1)

        slow_geo = st.sidebar.checkbox("Slow GEO (NY/TX/IL etc.)", value=True)
        fast_geo = st.sidebar.checkbox("Fast GEO (FL/AZ/NV etc.)", value=False)

        investor_share = st.sidebar.slider(
            "Investor / 2nd Home share", min_value=0.0, max_value=1.0, value=0.25, step=0.05
        )

        bank_share = st.sidebar.slider(
            "Bank originator share", min_value=0.0, max_value=1.0, value=0.6, step=0.05
        )

        wac_vs_mkt_bps = st.sidebar.slider(
            "WAC vs current primary mortgage rate (bps)",
            min_value=-100,
            max_value=300,
            value=75,
            step=5,
        )

        spec_cpr = cprm.story_cpr_from_traits(
            balance_bucket=balance_bucket,
            wala_months=wala,
            avg_fico=avg_fico,
            avg_ltv=avg_ltv,
            slow_geo=slow_geo,
            fast_geo=fast_geo,
            investor_share=investor_share,
            bank_originator_share=bank_share,
            wac_vs_market_bps=wac_vs_mkt_bps,
        )

        st.subheader("CPR assumptions")
        col_c1, col_c2 = st.columns(2)
        col_c1.metric("Generic TBA CPR", f"{tba_cpr*100:.1f} %")
        col_c2.metric("Story CPR (from traits)", f"{spec_cpr*100:.1f} %")

        result = cprm.spec_payup_from_cpr(
            tba_price=tba_price,
            tba_cpr=tba_cpr,
            spec_cpr=spec_cpr,
            coupon_rate_annual=coupon,
            term_months=term_months,
        )

        st.subheader("Price & Payup vs TBA")
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Model TBA Price", f"{result['tba_price_model']:.3f}")
        col_p2.metric("Spec Price (Story)", f"{result['spec_price_model']:.3f}")
        col_p3.metric("Payup (32nds)", f"{result['payup_32nds']}")

        st.caption(
            f"Using implied yield = {result['yield_annual']*100:.3f}% "
            f"for {term_choice} term."
        )

        st.subheader("Duration & Convexity (Model)")
        col_d1, col_d2 = st.columns(2)
        col_d1.metric("TBA Modified Duration (yrs)", f"{result['tba_duration']:.2f}")
        col_d1.metric("TBA Convexity", f"{result['tba_convexity']:.2f}")
        col_d2.metric("Spec Modified Duration (yrs)", f"{result['spec_duration']:.2f}")
        col_d2.metric("Spec Convexity", f"{result['spec_convexity']:.2f}")

        st.subheader("Price vs CPR curve for this coupon & term")

        cpr_grid = [x / 100.0 for x in range(2, 26, 2)]
        prices = []
        for c in cpr_grid:
            y = cprm.implied_yield_for_price(
                target_price=tba_price,
                coupon_rate_annual=coupon,
                cpr_annual=c,
                term_months=term_months,
            )
            pr = cprm.price_with_detail(
                principal=100.0,
                coupon_rate_annual=coupon,
                cpr_annual=c,
                yield_annual=y,
                term_months=term_months,
            ).price
            prices.append(pr)

        df_curve = pd.DataFrame(
            {"CPR (%)": [c * 100 for c in cpr_grid], "Model price": prices}
        ).set_index("CPR (%)")

        st.line_chart(df_curve)

        st.markdown(
            f"- Marker CPRs: **TBA CPR {tba_cpr*100:.1f}%**, "
            f"**Story CPR {spec_cpr*100:.1f}%**"
        )

        st.subheader("Auto-price stories in current CSV using CPR model (toy)")

        if not grid_loaded:
            st.warning("Grid (spec_payup_grid.csv) not loaded; only CPR analytics above are available.")
        else:
            df_grid = pd.read_csv("spec_payup_grid.csv")
            all_story_cols = [c for c in df_grid.columns if c not in ("coupon_bucket", "coupon_rate")]

            story_cpr_map = {}
            for s in all_story_cols:
                s_up = s.upper()
                if "LLB" in s_up:
                    story_cpr_map[s] = 0.06
                elif "MID" in s_up:
                    story_cpr_map[s] = 0.09
                elif "INV" in s_up or "INVEST" in s_up:
                    story_cpr_map[s] = 0.05
                elif "SEAS" in s_up or "BURN" in s_up:
                    story_cpr_map[s] = 0.07
                else:
                    story_cpr_map[s] = 0.10

            buckets = sorted(df_grid["coupon_rate"].unique())
            sel_coupon = st.selectbox("Coupon rate to price (from CSV)", options=buckets)

            tba_price_for_grid = st.number_input(
                "TBA price for this coupon (grid pricing)",
                min_value=80.0,
                max_value=110.0,
                value=tba_price,
                step=0.25,
            )

            tba_cpr_for_grid = st.slider(
                "Generic TBA CPR for grid pricing",
                min_value=0.01,
                max_value=0.25,
                value=tba_cpr,
                step=0.01,
            )

            sub = df_grid[df_grid["coupon_rate"] == sel_coupon].copy()
            if sub.empty:
                st.warning("No row found for that coupon in the CSV.")
            else:
                payup_row = {"story": [], "story_cpr": [], "payup_32nds": [], "spec_price": []}
                for s in all_story_cols:
                    spec_cpr_story = story_cpr_map[s]
                    res_story = cprm.spec_payup_from_cpr(
                        tba_price=tba_price_for_grid,
                        tba_cpr=tba_cpr_for_grid,
                        spec_cpr=spec_cpr_story,
                        coupon_rate_annual=sel_coupon / 100.0,
                        term_months=term_months,
                    )
                    payup_row["story"].append(s)
                    payup_row["story_cpr"].append(spec_cpr_story * 100.0)
                    payup_row["payup_32nds"].append(res_story["payup_32nds"])
                    payup_row["spec_price"].append(res_story["spec_price_model"])

                df_pay = pd.DataFrame(payup_row)
                st.dataframe(df_pay, use_container_width=True)
                st.caption(
                    "Toy grid: CPR per story is driven by name heuristics; "
                    "wire your own CPRs or a full mapping later."
                )


if __name__ == "__main__":
    main()