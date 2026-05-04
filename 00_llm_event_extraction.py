import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI


# config
DEFAULT_BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
FALLBACK_API_KEY = "<-api-key->"
MAX_REQUESTS_PER_MINUTE = 30

SYSTEM_MESSAGE = (
    "You are a financial data assistant with knowledge of SEC Form 4 filings. "
    "Return ONLY valid JSON. No explanation or markdown."
)


class RateLimiter:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.timestamps = collections.deque()

    def wait(self):
        if self.max_per_minute <= 0:
            return
        now = time.monotonic()
        while self.timestamps and now - self.timestamps[0] >= 60:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_per_minute:
            sleep_time = 60 - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.timestamps.append(time.monotonic())


def load_ground_truth(path="sec_filings.csv"):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # parse Trade Date from DD.MM.YYYY to datetime
    df["trade_date"] = pd.to_datetime(df["Trade Date"], format="%d.%m.%Y")

    # normalize fields
    df["ticker"] = df["Ticker"].str.strip()
    df["company_name"] = df["Company Name"].str.strip()
    df["insider_name"] = df["Insider Name"].str.strip()
    df["trade_type"] = df["Trade Type"].str.strip()

    # parse numeric fields (handle dollar signs, commas, negatives)
    df["price"] = (
        df["Price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["qty"] = (
        df["Qty"]
        .str.replace(",", "", regex=False)
        .astype(float)
        .abs()
    )
    df["value"] = (
        df["Value"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
        .abs()
    )

    return df


def load_managers(path="managers.csv"):
    df = pd.read_csv(path)
    df["ticker"] = df["Ticker"].str.strip()
    df["company_name"] = df["Company Name"].str.strip()
    df["insider_name"] = df["Insider Name"].str.strip()
    return df


def build_company_prompt(ticker, company_name):
    return (
        f"List ALL director dealings (insider trading) transactions for "
        f"{company_name} ({ticker}) between November 1, 2024 and May 31, 2025. "
        f"Include purchases, sales, and option-exercise sales.\n\n"
        f"For each transaction, return: date (YYYY-MM-DD), insider_name, role, "
        f'transaction_type (one of: "P - Purchase", "S - Sale", "S - Sale+OE"), '
        f"shares, price_per_share, total_value.\n\n"
        f"Return ONLY a JSON array of objects. No explanation or markdown."
    )


def build_manager_prompt(ticker, company_name, insider_name):
    return (
        f"List ALL director dealings (insider trading) transactions performed by "
        f"{insider_name} at {company_name} ({ticker}) between November 1, 2024 "
        f"and May 31, 2025.\n\n"
        f"For each transaction, return: date (YYYY-MM-DD), insider_name, role, "
        f'transaction_type (one of: "P - Purchase", "S - Sale", "S - Sale+OE"), '
        f"shares, price_per_share, total_value.\n\n"
        f"Return ONLY a JSON array of objects. No explanation or markdown."
    )


def query_llm(client, model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                seed=42,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return None


def parse_response(raw_text):
    if not raw_text:
        return []

    text = raw_text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # fallback: extract [...] from markdown code blocks or raw text
    # remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # find the outermost JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def normalize_event(raw_event, default_ticker=None, default_insider=None):
    e = {}

    e["ticker"] = (
        str(raw_event.get("ticker", default_ticker or "")).strip().upper()
    )

    # date — try multiple formats
    raw_date = str(raw_event.get("date", "")).strip()
    e["trade_date"] = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            e["trade_date"] = pd.to_datetime(raw_date, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    if e["trade_date"] is None:
        try:
            e["trade_date"] = pd.to_datetime(raw_date)
        except (ValueError, TypeError):
            pass

    e["insider_name"] = str(
        raw_event.get("insider_name", default_insider or "")
    ).strip()

    # trade type — normalize to canonical form
    raw_type = str(raw_event.get("transaction_type", "")).strip()
    type_lower = raw_type.lower()
    if "purchase" in type_lower or type_lower == "p":
        e["trade_type"] = "P - Purchase"
    elif "sale+oe" in type_lower or "option" in type_lower or "exercise" in type_lower:
        e["trade_type"] = "S - Sale+OE"
    elif "sale" in type_lower or type_lower == "s":
        e["trade_type"] = "S - Sale"
    else:
        e["trade_type"] = raw_type  # keep as-is if unrecognized

    for field, keys in [
        ("shares", ["shares", "qty", "quantity", "number_of_shares"]),
        ("price", ["price_per_share", "price", "share_price"]),
        ("value", ["total_value", "value", "transaction_value"]),
    ]:
        val = None
        for k in keys:
            if k in raw_event and raw_event[k] is not None:
                val = raw_event[k]
                break
        if val is not None:
            try:
                val = str(val).replace("$", "").replace(",", "")
                e[field] = abs(float(val))
            except (ValueError, TypeError):
                e[field] = np.nan
        else:
            e[field] = np.nan

    return e


def run_company_extraction(client, model, ground_truth_df, rate_limiter, tickers=None):
    pairs = ground_truth_df[["ticker", "company_name"]].drop_duplicates()
    if tickers:
        pairs = pairs[pairs["ticker"].isin(tickers)]

    all_events = []
    for _, row in pairs.iterrows():
        ticker, company = row["ticker"], row["company_name"]
        print(f"  [company] {ticker} ({company})...")

        rate_limiter.wait()
        prompt = build_company_prompt(ticker, company)
        raw = query_llm(client, model, prompt)
        parsed = parse_response(raw)

        for item in parsed:
            event = normalize_event(item, default_ticker=ticker)
            event["raw_response_preview"] = (raw or "")[:200]
            all_events.append(event)

        print(f" {len(parsed)} events extracted")

    expected_cols = ["ticker", "trade_date", "insider_name", "trade_type",
                     "shares", "price", "value", "raw_response_preview"]
    if not all_events:
        return pd.DataFrame(columns=expected_cols)
    return pd.DataFrame(all_events)


def run_manager_extraction(client, model, managers_df, rate_limiter, tickers=None):
    mgrs = managers_df.copy()
    if tickers:
        mgrs = mgrs[mgrs["ticker"].isin(tickers)]

    all_events = []
    total = len(mgrs)
    for i, (_, row) in enumerate(mgrs.iterrows()):
        ticker = row["ticker"]
        company = row["company_name"]
        insider = row["insider_name"]
        print(f"  [manager {i+1}/{total}] {ticker} / {insider}...")

        rate_limiter.wait()
        prompt = build_manager_prompt(ticker, company, insider)
        raw = query_llm(client, model, prompt)
        parsed = parse_response(raw)

        for item in parsed:
            event = normalize_event(item, default_ticker=ticker, default_insider=insider)
            event["raw_response_preview"] = (raw or "")[:200]
            all_events.append(event)

        print(f" {len(parsed)} events extracted")

    expected_cols = ["ticker", "trade_date", "insider_name", "trade_type",
                     "shares", "price", "value", "raw_response_preview"]
    if not all_events:
        return pd.DataFrame(columns=expected_cols)
    return pd.DataFrame(all_events)


def _name_match(name_a, name_b):
    if not name_a or not name_b:
        return False
    words_a = set(name_a.lower().replace(".", "").replace(",", "").split())
    words_b = set(name_b.lower().replace(".", "").replace(",", "").split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    # Match if at least half the words from the shorter name appear in the other
    min_len = min(len(words_a), len(words_b))
    return overlap >= max(1, min_len * 0.5)


def _trade_direction(trade_type):
    t = str(trade_type).lower()
    if "purchase" in t or t == "p":
        return "buy"
    return "sell"


def match_events(extracted_df, truth_df, ticker):
    ext = extracted_df[extracted_df["ticker"] == ticker].copy()
    tru = truth_df[truth_df["ticker"] == ticker].copy()

    if ext.empty:
        return [], [], list(tru.index)
    if tru.empty:
        return [], list(ext.index), []

    matched_pairs = []
    used_truth = set()
    used_extracted = set()

    for ei, erow in ext.iterrows():
        if ei in used_extracted:
            continue
        if erow["trade_date"] is None or pd.isna(erow.get("trade_date")):
            continue

        for ti, trow in tru.iterrows():
            if ti in used_truth:
                continue

            # same trade direction
            if _trade_direction(erow["trade_type"]) != _trade_direction(trow["trade_type"]):
                continue

            # name match
            if not _name_match(erow.get("insider_name", ""), trow["insider_name"]):
                continue

            # date within +-1 business day
            try:
                date_diff = abs((erow["trade_date"] - trow["trade_date"]).days)
            except (TypeError, AttributeError):
                continue
            if date_diff > 3:  # +-1 bday arount up to 3 calendar days over weekends
                continue

            matched_pairs.append((ei, ti))
            used_truth.add(ti)
            used_extracted.add(ei)
            break

    false_positives = [i for i in ext.index if i not in used_extracted]
    false_negatives = [i for i in tru.index if i not in used_truth]

    return matched_pairs, false_positives, false_negatives


def compute_metrics(matched_pairs, false_positives, false_negatives,
                    extracted_df, truth_df):
    tp = len(matched_pairs)
    fp = len(false_positives)
    fn = len(false_negatives)
    total_extracted = tp + fp
    total_truth = tp + fn

    precision = tp / total_extracted if total_extracted > 0 else 0.0
    recall = tp / total_truth if total_truth > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hallucination_rate = fp / total_extracted if total_extracted > 0 else 0.0

    # accuracy metrics on matched events
    exact_date = 0
    exact_type = 0
    value_errors = []

    for ei, ti in matched_pairs:
        erow = extracted_df.loc[ei]
        trow = truth_df.loc[ti]

        # exact date match
        try:
            if erow["trade_date"] == trow["trade_date"]:
                exact_date += 1
        except (TypeError, AttributeError):
            pass

        # exact trade type match
        if str(erow.get("trade_type", "")).strip() == str(trow["trade_type"]).strip():
            exact_type += 1

        # value accuracy (MAPE on matched)
        try:
            ext_val = float(erow.get("value", np.nan))
            tru_val = float(trow["value"])
            if tru_val > 0 and not np.isnan(ext_val):
                value_errors.append(abs(ext_val - tru_val) / tru_val)
        except (ValueError, TypeError):
            pass

    date_accuracy = exact_date / tp if tp > 0 else 0.0
    type_accuracy = exact_type / tp if tp > 0 else 0.0
    mean_value_error = np.mean(value_errors) if value_errors else np.nan

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_extracted": total_extracted,
        "total_ground_truth": total_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": hallucination_rate,
        "date_accuracy": date_accuracy,
        "type_accuracy": type_accuracy,
        "mean_value_pct_error": mean_value_error,
    }


def evaluate(extracted_df, truth_df, approach_name):
    all_tickers = sorted(
        set(truth_df["ticker"].unique()) | set(extracted_df["ticker"].unique())
        if not extracted_df.empty else set(truth_df["ticker"].unique())
    )

    rows = []
    for ticker in all_tickers:
        matched, fp, fn = match_events(extracted_df, truth_df, ticker)
        metrics = compute_metrics(matched, fp, fn, extracted_df, truth_df)
        metrics["ticker"] = ticker
        metrics["approach"] = approach_name
        rows.append(metrics)

    # Aggregate row
    if rows:
        agg = {}
        total_tp = sum(r["true_positives"] for r in rows)
        total_fp = sum(r["false_positives"] for r in rows)
        total_fn = sum(r["false_negatives"] for r in rows)
        total_ext = total_tp + total_fp
        total_tru = total_tp + total_fn

        agg["ticker"] = "AGGREGATE"
        agg["approach"] = approach_name
        agg["true_positives"] = total_tp
        agg["false_positives"] = total_fp
        agg["false_negatives"] = total_fn
        agg["total_extracted"] = total_ext
        agg["total_ground_truth"] = total_tru
        agg["precision"] = total_tp / total_ext if total_ext > 0 else 0.0
        agg["recall"] = total_tp / total_tru if total_tru > 0 else 0.0
        agg["f1"] = (
            2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
            if (agg["precision"] + agg["recall"]) > 0 else 0.0
        )
        agg["hallucination_rate"] = total_fp / total_ext if total_ext > 0 else 0.0

        # Average accuracy metrics across tickers with matches
        tickers_with_matches = [r for r in rows if r["true_positives"] > 0]
        agg["date_accuracy"] = (
            np.mean([r["date_accuracy"] for r in tickers_with_matches])
            if tickers_with_matches else 0.0
        )
        agg["type_accuracy"] = (
            np.mean([r["type_accuracy"] for r in tickers_with_matches])
            if tickers_with_matches else 0.0
        )
        val_errs = [r["mean_value_pct_error"] for r in tickers_with_matches
                    if not np.isnan(r["mean_value_pct_error"])]
        agg["mean_value_pct_error"] = np.mean(val_errs) if val_errs else np.nan

        rows.append(agg)

    return pd.DataFrame(rows)


def print_summary(eval_df):
    for approach, grp in eval_df.groupby("approach"):
        agg = grp[grp["ticker"] == "AGGREGATE"]
        if agg.empty:
            continue
        agg = agg.iloc[0]
        print(f"  Approach: {approach}")
        print(f"  Total extracted:    {int(agg['total_extracted'])}")
        print(f"  Total ground truth: {int(agg['total_ground_truth'])}")
        print(f"  True positives:     {int(agg['true_positives'])}")
        print(f"  False positives:    {int(agg['false_positives'])}")
        print(f"  False negatives:    {int(agg['false_negatives'])}")
        print(f"  Precision:          {agg['precision']:.3f}")
        print(f"  Recall:             {agg['recall']:.3f}")
        print(f"  F1 Score:           {agg['f1']:.3f}")
        print(f"  Hallucination rate: {agg['hallucination_rate']:.3f}")
        print(f"  Date accuracy:      {agg['date_accuracy']:.3f}")
        print(f"  Type accuracy:      {agg['type_accuracy']:.3f}")
        val_err = agg['mean_value_pct_error']
        print(f"  Mean value % error: {val_err:.3f}" if not np.isnan(val_err) else "  Mean value % error: N/A")

    # Per-ticker breakdown (excluding aggregate)
    per_ticker = eval_df[eval_df["ticker"] != "AGGREGATE"]
    if not per_ticker.empty:
        print("  Per-Ticker Breakdown")
        cols = ["approach", "ticker", "precision", "recall", "f1",
                "total_extracted", "total_ground_truth"]
        print(per_ticker[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based insider event extraction with evaluation"
    )
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set AQUEDUCT_API_KEY env var)")
    parser.add_argument("--approach", choices=["company", "manager", "both"],
                        default="both", help="Extraction approach to run")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Optional subset of tickers to test on")
    parser.add_argument("--max-rpm", type=int, default=MAX_REQUESTS_PER_MINUTE,
                        help="Max requests per minute (0 = unlimited)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("AQUEDUCT_API_KEY", FALLBACK_API_KEY)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    rate_limiter = RateLimiter(args.max_rpm)

    print("Loading ground truth")
    truth_df = load_ground_truth()
    print(f"  {len(truth_df)} events loaded from sec_filings.csv")

    managers_df = load_managers()
    print(f"  {len(managers_df)} (ticker, insider) pairs loaded from managers.csv")

    tickers = [t.upper() for t in args.tickers] if args.tickers else None
    if tickers:
        print(f"  Filtering to tickers: {tickers}")

    model_safe = args.model.replace("/", "-").replace(":", "-")
    all_eval_dfs = []

    # company-level extraction
    if args.approach in ("company", "both"):
        print("Running COMPANY-LEVEL extraction...")
        company_df = run_company_extraction(
            client, args.model, truth_df, rate_limiter, tickers
        )

        out_path = f"llm_extracted_events_company_{model_safe}.csv"
        company_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(company_df)} extracted events to {out_path}")

        company_eval = evaluate(company_df, truth_df, "company")
        all_eval_dfs.append(company_eval)

    # manager-level extraction
    if args.approach in ("manager", "both"):
        print("Running MANAGER-LEVEL extraction...")
        manager_df = run_manager_extraction(
            client, args.model, managers_df, rate_limiter, tickers
        )

        out_path = f"llm_extracted_events_manager_{model_safe}.csv"
        manager_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(manager_df)} extracted events to {out_path}")

        manager_eval = evaluate(manager_df, truth_df, "manager")
        all_eval_dfs.append(manager_eval)

    # combined evaluation output
    if all_eval_dfs:
        eval_df = pd.concat(all_eval_dfs, ignore_index=True)
        eval_path = f"llm_extraction_evaluation_{model_safe}.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"\nSaved evaluation to {eval_path}")
        print_summary(eval_df)


if __name__ == "__main__":
    main()
