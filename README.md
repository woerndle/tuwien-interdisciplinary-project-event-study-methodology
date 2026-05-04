# Directors' Dealings Event Study

Event study analysis of insider trading in 30 major US high-tech firms (Nasdaq) following the 2024 US presidential election (Nov 2024 -- May 2025). Conducted as part of the Interdisciplinary Project in Data Science at TU Wien.

## Structure

| Notebook / Script | Description |
|---|---|
| `00_llm_event_extraction.py` | LLM-based event extraction using mistral-small-3.2-24b |
| `01_exploratory_data_anlysis.ipynb` | Data loading, cleaning, EDA, and sample construction |
| `02_traditional_ar_calculation.ipynb` | Market model estimation and abnormal return calculation |
| `02_llm_ar_computation.py` | LLM-based abnormal return calculation for comparison |
| `03_car_calculation.ipynb` | Cumulative abnormal return aggregation and overlap filtering |
| `04_cross_sectional_analysis.ipynb` | G-Rank significance tests and paired comparison tests |
| `05_visualization.ipynb` | Publication-quality figures and earnings proximity analysis |

## Data

- **Input:** `sec_filings.csv`, `market_returns.csv`, `portfolio_returns.csv`, `earnings_dates_cache.csv`
- **Generated:** Abnormal returns, CARs, cross-sectional test results, LLM extraction evaluations

See `report.tex` for full methodology and results.

## Requirements

```bash
# python 3.12+
pip install -r requirements.txt
```

## Running the LLM scripts

Both scripts require an Aqueduct API key via the `AQUEDUCT_API_KEY` environment variable:

```bash
export AQUEDUCT_API_KEY="your-key-here"
```

**Event extraction** (extracts insider trading events via LLM and evaluates against ground truth):

```bash
python 00_llm_event_extraction.py --model mistral-small-3.2-24b --approach both
```

**Abnormal return computation** (computes ARs via LLM for comparison with OLS pipeline):

```bash
python 02_llm_ar_computation.py --model mistral-small-3.2-24b
```

Both scripts support `--max-rpm` to control rate limiting and `--help` for all options.
