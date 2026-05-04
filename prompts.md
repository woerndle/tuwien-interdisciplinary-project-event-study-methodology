# LLM Prompts Used in This Project

## 1. Event Extraction (00\_llm\_event\_extraction.py)

### 1.1 System Message

```
You are a financial data assistant with knowledge of SEC Form 4 filings.
Return ONLY valid JSON. No explanation or markdown.
```

### 1.2 Company-Level Prompt

One prompt per ticker (30 prompts total). Placeholders: `{company_name}`, `{ticker}`.

```
List ALL director dealings (insider trading) transactions for {company_name}
({ticker}) between November 1, 2024 and May 31, 2025. Include purchases,
sales, and option-exercise sales.

For each transaction, return: date (YYYY-MM-DD), insider_name, role,
transaction_type (one of: "P - Purchase", "S - Sale", "S - Sale+OE"),
shares, price_per_share, total_value.

Return ONLY a JSON array of objects. No explanation or markdown.
```

### 1.3 Manager-Level Prompt

One prompt per (manager, ticker) pair (~218 prompts). Placeholders: `{insider_name}`, `{company_name}`, `{ticker}`.

```
List ALL director dealings (insider trading) transactions performed by
{insider_name} at {company_name} ({ticker}) between November 1, 2024
and May 31, 2025.

For each transaction, return: date (YYYY-MM-DD), insider_name, role,
transaction_type (one of: "P - Purchase", "S - Sale", "S - Sale+OE"),
shares, price_per_share, total_value.

Return ONLY a JSON array of objects. No explanation or markdown.
```

---

## 2. Abnormal Return Computation (02\_llm\_ar\_computation.py)

### 2.1 System Message

```
You are a financial analyst. Perform the calculations and return ONLY
a JSON object with the numeric results. No explanation.
```

### 2.2 Prompt With Formula

One prompt per event-day (up to ~19,000 calls). Placeholders: `{estimation_window_table}`, `{stock_return}`, `{market_return}`.

```
Perform an event study Abnormal Return calculation.

**Step 1 — Estimate Market Model parameters using OLS regression on this
estimation window data:**

Day | Stock_Return | Market_Return
{estimation_window_table}

Regression model: Stock_Return = α + β × Market_Return + ε

**Step 2 — Calculate the Abnormal Return for this single trading day:**

Stock_Return: {stock_return}
Market_Return: {market_return}

Expected_Return = α + β × Market_Return
AR = Stock_Return - Expected_Return

**Return JSON only:**
{"alpha": <number>, "beta": <number>, "expected_return": <number>, "ar": <number>}
```

### 2.3 Prompt Without Formula

Same structure but omits the explicit regression equation and AR formula.

```
Perform an event study Abnormal Return calculation.

**Step 1 — Estimate α and β using OLS on this estimation window data:**

Day | Stock_Return | Market_Return
{estimation_window_table}

**Step 2 — Calculate the Abnormal Return using the Market Model for this
single trading day:**

Stock_Return: {stock_return}
Market_Return: {market_return}

**Return JSON only:**
{"alpha": <number>, "beta": <number>, "expected_return": <number>, "ar": <number>}
```
