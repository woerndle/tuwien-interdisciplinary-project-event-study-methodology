import argparse
import collections
import json
import os
import re
import sys
import time
import threading

import pandas as pd
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def _build_trading_calendar(market_df):
    return np.sort(market_df["date"].dropna().unique())


def _trading_day_offset(trading_cal, event_date, offset):
    idx = np.searchsorted(trading_cal, np.datetime64(event_date), side="right") - 1
    if idx < 0:
        idx = 0
    target = idx + offset
    target = max(0, min(target, len(trading_cal) - 1))
    return trading_cal[target]


DEFAULT_MODELS = ["mistral-small-3.2-24b"]
BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
DEFAULT_API_KEY = os.environ.get("AQUEDUCT_API_KEY", "")
MAX_REQUESTS_PER_MINUTE = 30

CHECKPOINT_FILE = "llm_ar_checkpoint.csv"
OUTPUT_FILE = "llm_abnormal_returns.csv"

SYSTEM_MESSAGE = (
    "You are a financial analyst. Perform the calculations and return ONLY "
    "a JSON object with the numeric results. No explanation."
)



_rate_lock = threading.Lock()
_rate_timestamps = collections.deque()


def _rate_limit_wait(max_per_minute):
    if max_per_minute <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        while _rate_timestamps and now - _rate_timestamps[0] >= 60:
            _rate_timestamps.popleft()
        if len(_rate_timestamps) >= max_per_minute:
            sleep_time = 60 - (now - _rate_timestamps[0])
            if sleep_time > 0:
                _rate_lock.release()
                time.sleep(sleep_time)
                _rate_lock.acquire()
        _rate_timestamps.append(time.monotonic())


PROMPT_WITH_FORMULA = """\
Perform an event study Abnormal Return calculation.

**Step 1 — Estimate Market Model parameters using OLS regression on this estimation window data:**

Day | Stock_Return | Market_Return
{estimation_window_table}

Regression model: Stock_Return = α + β × Market_Return + ε

**Step 2 — Calculate the Abnormal Return for this single trading day:**

Stock_Return: {stock_return}
Market_Return: {market_return}

Expected_Return = α + β × Market_Return
AR = Stock_Return - Expected_Return

**Return JSON only:**
{{"alpha": <number>, "beta": <number>, "expected_return": <number>, "ar": <number>}}"""

PROMPT_WITHOUT_FORMULA = """\
Perform an event study Abnormal Return calculation.

**Step 1 — Estimate α and β using OLS on this estimation window data:**

Day | Stock_Return | Market_Return
{estimation_window_table}

**Step 2 — Calculate the Abnormal Return using the Market Model for this single trading day:**

Stock_Return: {stock_return}
Market_Return: {market_return}

**Return JSON only:**
{{"alpha": <number>, "beta": <number>, "expected_return": <number>, "ar": <number>}}"""


def load_data():
    # ground truth ARs
    ar_df = pd.read_csv("event_study_abnormal_returns.csv")
    ar_df["event_date"] = pd.to_datetime(ar_df["event_date"])
    ar_df["date"] = pd.to_datetime(ar_df["date"])

    # ground truth alpha/beta per event
    car_df = pd.read_csv("event_study_metadata.csv")
    car_df["event_date"] = pd.to_datetime(car_df["event_date"])

    # portfolio returns (wide -> long)
    port_df = pd.read_csv("portfolio_returns.csv")
    port_df["date"] = pd.to_datetime(port_df["date"])
    port_long = port_df.melt(id_vars=["date"], var_name="ticker", value_name="stock_return")
    port_long["stock_return"] = pd.to_numeric(port_long["stock_return"], errors="coerce")

    # market returns
    mkt_df = pd.read_csv("market_returns.csv")
    mkt_df["date"] = pd.to_datetime(mkt_df["date"], errors="coerce")
    mkt_df["market_return"] = pd.to_numeric(mkt_df["market_return"], errors="coerce")
    mkt_df = mkt_df.dropna(subset=["date"])

    return ar_df, car_df, port_long, mkt_df


def build_tasks(ar_df, car_df, port_long, mkt_df):
    # merge stock + market returns
    returns = port_long.merge(mkt_df, on="date", how="inner")

    # unique events from ground truth
    events = car_df[["ticker", "event_date", "alpha", "beta"]].drop_duplicates()
    print(f"  Ground-truth events: {len(events)}")
    print(f"  Ground-truth AR rows: {len(ar_df)}")

    trading_cal = _build_trading_calendar(mkt_df)

    tasks = []
    skipped_est_data = 0
    skipped_no_returns = 0
    skipped_nan = 0
    events_with_tasks = set()

    for _, event in events.iterrows():
        ticker = event["ticker"]
        event_date = event["event_date"]
        trad_alpha = event["alpha"]
        trad_beta = event["beta"]

        # estimation window: trading days [-220, -21] relative to event_date
        est_start = _trading_day_offset(trading_cal, event_date, -220)
        est_end = _trading_day_offset(trading_cal, event_date, -21)
        est_data = returns[
            (returns["ticker"] == ticker)
            & (returns["date"] >= est_start)
            & (returns["date"] <= est_end)
        ].sort_values("date")

        if len(est_data) < 200:
            skipped_est_data += 1
            print(f"  SKIP {ticker} event={event_date.date()}: only {len(est_data)} estimation days (<200)")
            continue

        # event days from ground truth AR file
        event_days = ar_df[
            (ar_df["ticker"] == ticker) & (ar_df["event_date"] == event_date)
        ]

        event_task_count = 0
        for _, day_row in event_days.iterrows():
            day_date = day_row["date"]
            trad_ar = day_row["AR"]

            # get actual returns for the event day
            day_returns = returns[
                (returns["ticker"] == ticker) & (returns["date"] == day_date)
            ]
            if day_returns.empty:
                skipped_no_returns += 1
                continue

            stock_ret = day_returns.iloc[0]["stock_return"]
            mkt_ret = day_returns.iloc[0]["market_return"]

            if pd.isna(stock_ret) or pd.isna(mkt_ret) or pd.isna(trad_ar):
                skipped_nan += 1
                continue

            tasks.append({
                "ticker": ticker,
                "event_date": event_date,
                "date": day_date,
                "estimation_data": est_data[["stock_return", "market_return"]].copy(),
                "stock_return": stock_ret,
                "market_return": mkt_ret,
                "traditional_AR": trad_ar,
                "traditional_alpha": trad_alpha,
                "traditional_beta": trad_beta,
            })
            event_task_count += 1

        if event_task_count > 0:
            events_with_tasks.add((ticker, event_date))

    # summary
    print(f"\n  Task build summary:")
    print(f"    Events in ground truth:       {len(events)}")
    print(f"    Events skipped (est <30 days): {skipped_est_data}")
    print(f"    Events with AR tasks:          {len(events_with_tasks)}")
    print(f"    Total AR tasks (API calls):    {len(tasks)}")
    print(f"    Skipped days (no returns):     {skipped_no_returns}")
    print(f"    Skipped days (NaN values):     {skipped_nan}")
    expected = len(events_with_tasks) * 41
    print(f"    Expected if 41 days/event:     {expected}")
    if len(tasks) != expected:
        print(f"    MISMATCH: {expected - len(tasks)} fewer tasks than expected")

    return tasks


def format_estimation_table(est_df):
    lines = []
    for i, (_, row) in enumerate(est_df.iterrows(), 1):
        lines.append(
            f"{i} | {row['stock_return']:.6f} | {row['market_return']:.6f}"
        )
    return "\n".join(lines)


def build_prompt(estimation_data, stock_return, market_return, include_formula):
    table = format_estimation_table(estimation_data)
    template = PROMPT_WITH_FORMULA if include_formula else PROMPT_WITHOUT_FORMULA
    return template.format(
        estimation_window_table=table,
        stock_return=f"{stock_return:.6f}",
        market_return=f"{market_return:.6f}",
    )


def parse_llm_response(raw_response):
    # try JSON extraction first
    json_match = re.search(r"\{[^{}]*\}", raw_response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "alpha": float(data["alpha"]),
                "beta": float(data["beta"]),
                "expected_return": float(data.get("expected_return", float("nan"))),
                "ar": float(data["ar"]),
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # fallback withh regex for individual values
    result = {}
    for key in ["alpha", "beta", "expected_return", "ar"]:
        pattern = rf'"{key}"\s*:\s*(-?[\d.eE+-]+)'
        m = re.search(pattern, raw_response)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass

    if "alpha" in result and "beta" in result and "ar" in result:
        result.setdefault("expected_return", float("nan"))
        return result

    return None


def compute_ar_via_llm(client, model, estimation_data_str, stock_return, market_return, include_formula, max_retries=3):
    user_msg = build_prompt(
        estimation_data_str, stock_return, market_return, include_formula
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                seed=42,
            )
            raw = response.choices[0].message.content
            parsed = parse_llm_response(raw)
            return parsed, raw

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None, str(e)


RESULT_COLUMNS = [
    "ticker", "event_date", "date", "model", "prompt_variant",
    "traditional_AR", "traditional_alpha", "traditional_beta",
    "llm_AR", "llm_alpha", "llm_beta", "llm_expected_return",
    "ar_error", "alpha_error", "beta_error", "raw_response",
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        done = set(
            zip(df["ticker"], df["event_date"], df["date"], df["model"])
        )
        return df, done
    return pd.DataFrame(columns=RESULT_COLUMNS), set()


def append_result(result_row):
    write_header = not os.path.exists(CHECKPOINT_FILE)
    pd.DataFrame([result_row]).to_csv(
        CHECKPOINT_FILE, mode="a", header=write_header, index=False
    )


def print_summary(results_df):
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for model, grp in results_df.groupby("model"):
        print(f"\n── {model} ──")
        total = len(grp)
        parsed = grp["llm_AR"].notna().sum()
        failed = total - parsed

        print(f"  Total API calls: {total}")
        print(f"  Successful parses: {parsed} ({100*parsed/total:.1f}%)")
        print(f"  Parse failures: {failed}")

        if parsed > 0:
            valid = grp[grp["llm_AR"].notna()]
            ar_mae = valid["ar_error"].abs().mean()
            ar_within = (valid["ar_error"].abs() < 1e-4).mean() * 100
            alpha_mae = valid["alpha_error"].abs().mean()
            beta_mae = valid["beta_error"].abs().mean()

            print(f"  AR  — MAE: {ar_mae:.8f}, within 1e-4: {ar_within:.1f}%")
            print(f"  Alpha MAE: {alpha_mae:.8f}")
            print(f"  Beta  MAE: {beta_mae:.8f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-based Abnormal Return computation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to use")
    parser.add_argument("--num-events", type=int, default=5,
                        help="Number of events to process (0 = all)")
    parser.add_argument("--prompt-variant", choices=["with_formula", "without_formula"],
                        default="with_formula")
    parser.add_argument("--max-rpm", type=int, default=MAX_REQUESTS_PER_MINUTE,
                        help="Max requests per minute (0 = unlimited)")
    args = parser.parse_args()

    include_formula = args.prompt_variant == "with_formula"
    max_rpm = args.max_rpm

    api_key = DEFAULT_API_KEY
    if not api_key:
        print("Error: set AQUEDUCT_API_KEY env var", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    print("Loading data...")
    ar_df, car_df, port_long, mkt_df = load_data()

    print("Building tasks...")
    tasks = build_tasks(ar_df, car_df, port_long, mkt_df)
    print(f"  Total AR points: {len(tasks)}")

    # limit by number of events (not AR points)
    if args.num_events > 0:
        unique_events = list({(t["ticker"], t["event_date"]) for t in tasks})
        unique_events.sort()
        selected_events = set(unique_events[:args.num_events])
        tasks = [t for t in tasks if (t["ticker"], t["event_date"]) in selected_events]
        print(f"  Selected {args.num_events} events → {len(tasks)} AR points")

    checkpoint_df, done_keys = load_checkpoint()
    print(f"  Already completed: {len(done_keys)} results")

    # process concurrent requests with rate limiting
    checkpoint_lock = threading.Lock()

    def _process_one(task, model, idx, total):
        key = (task["ticker"], str(task["event_date"]), str(task["date"]), model)

        _rate_limit_wait(max_rpm)
        print(f"  [{idx+1}/{total}] {task['ticker']} | event={task['event_date'].date()} | day={task['date'].date()}")

        parsed, raw = compute_ar_via_llm(
            client, model,
            task["estimation_data"],
            task["stock_return"],
            task["market_return"],
            include_formula,
        )

        row = {
            "ticker": task["ticker"],
            "event_date": task["event_date"],
            "date": task["date"],
            "model": model,
            "prompt_variant": args.prompt_variant,
            "traditional_AR": task["traditional_AR"],
            "traditional_alpha": task["traditional_alpha"],
            "traditional_beta": task["traditional_beta"],
            "llm_AR": parsed["ar"] if parsed else None,
            "llm_alpha": parsed["alpha"] if parsed else None,
            "llm_beta": parsed["beta"] if parsed else None,
            "llm_expected_return": parsed.get("expected_return") if parsed else None,
            "ar_error": (parsed["ar"] - task["traditional_AR"]) if parsed else None,
            "alpha_error": (parsed["alpha"] - task["traditional_alpha"]) if parsed else None,
            "beta_error": (parsed["beta"] - task["traditional_beta"]) if parsed else None,
            "raw_response": raw,
        }

        with checkpoint_lock:
            append_result(row)
            done_keys.add(key)

    # workers are how many requests can be in-flight at once.
    # at ~3s/request and 30 RPM (1 every 2s), 2 workers keeps the pipe full.
    n_workers = max(1, min(4, max_rpm // 15)) if max_rpm > 0 else 1

    for model in args.models:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        pending = []
        for i, task in enumerate(tasks):
            key = (task["ticker"], str(task["event_date"]), str(task["date"]), model)
            if key in done_keys:
                continue
            pending.append((i, task))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one, task, model, idx, len(tasks)): idx
                for idx, task in pending
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    print(f"  ERROR in task {futures[future]}: {exc}")

    # finalize
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        print_summary(final_df)
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
