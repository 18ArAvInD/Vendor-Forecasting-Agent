# Vendor Forecasting Agent

An MVP decision-support agent for semiconductor supply-chain vendor risk. It predicts expected supplier delivery performance using an explainable deterministic heuristic, identifies the main risk drivers, estimates potential inventory exposure, recommends actions, and uses an optional Mistral LLM for natural-language explanations and interactive decision support.

## What it does

The application follows this flow:

**Input → Metrics → Trend → Forecast → Rules → Risk Score → Inventory Impact → Recommendation → LLM Explanation → UI**

The deterministic engine is the source of truth for all calculated values. The LLM is used only to explain already-computed results and answer supported business questions. It does not calculate or change risk scores, forecasts, inventory quantities, or recommendations.

### MVP dimensions

- On-time delivery
- Lead-time trend
- Quantity fulfillment
- Capacity utilization
- Allocation / commitment
- Quality / defect rate

Inventory runway is used as an impact measure, not as a vendor-risk scoring dimension.

## Project structure

```text
.
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── .kiro/
├── src/
│   └── vendor_forecasting_agent/
│       ├── pipeline.py
│       ├── metrics.py
│       ├── trend.py
│       ├── forecast.py
│       ├── rules.py
│       ├── scoring.py
│       ├── impact.py
│       ├── recommend.py
│       ├── explanation.py
│       ├── llm_provider.py
│       ├── llm_context.py
│       ├── llm_prompts.py
│       ├── decision_support.py
│       ├── conversation.py
│       ├── whatif.py
│       ├── whatif_scenarios.py
│       ├── demo.py
│       ├── demo_scenario.py
│       ├── demo_scenarios.py
│       ├── upstream.py
│       ├── schema.py
│       ├── config.py
│       ├── synthetic.py
│       └── ui/
│           ├── app.py
│           └── view_model.py
└── tests/
```

Generated files such as `__pycache__`, `.pytest_cache`, `.egg-info`, virtual environments, and local secrets are intentionally excluded from source control by `.gitignore`.

## Requirements

- Python **3.11.x**
- A Mistral API key is optional. The deterministic analysis works without an LLM key.
- AWS/Bedrock configuration is optional and is only needed if the Bedrock provider is used.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "Vendour forecasting agent"
```

### 2. Create a Python 3.11 virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

For development, Streamlit, and direct Mistral API support:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,mistral]"
```

If Bedrock support is also required:

```bash
python -m pip install -e ".[dev,ui,mistral,llm]"
```

### 4. Configure environment variables

Copy the example file:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env` and add your own credentials.

For direct Mistral API usage:

```text
MISTRAL_API_KEY=your_real_key_here
MISTRAL_MODEL_ID=mistral-medium-latest
```

Do **not** commit `.env` or any real API key to GitHub. Only `.env.example` belongs in the repository.

The application can run without `MISTRAL_API_KEY`; in that case it uses the deterministic/offline provider and does not make an external LLM call.

## Run the Streamlit application

From the project root, with the virtual environment activated and the package installed:

```bash
streamlit run src/vendor_forecasting_agent/ui/app.py
```

The default demo scenario is the semiconductor supply-chain scenario used by the MVP. The UI also provides vendor selection, risk details, inventory impact, recommendations, Ask the Vendor Agent, What-If analysis, and an AI audit section.

## Run tests

```bash
python -m pytest -q
```

The current project test suite contains **412 tests**.

The tests are designed to remain offline. Provider tests use fakes/mocks and do not require a real Mistral API key or AWS credentials.

## LLM provider behavior

The provider selection order is:

1. `MISTRAL_API_KEY` configured → direct Mistral API provider
2. Otherwise, `BEDROCK_MODEL_ID` configured → Mistral on Amazon Bedrock
3. Otherwise → offline deterministic provider

Credentials are loaded from environment configuration and are not embedded in prompts, logs, or the repository.

## Architecture principles

- Deterministic calculations are performed in Python.
- Risk thresholds and scoring rules are deterministic and configurable.
- The LLM explains computed results; it does not invent or recalculate business numbers.
- Inventory impact is explicitly labeled as potential exposure.
- What-If calculations remain deterministic.
- No autonomous purchase-order creation.
- No ERP/WMS/TMS integration in this MVP.
- No ML forecasting, RAG, vector database, or multi-agent orchestration is required for the current MVP.

## Demo scenario

The curated semiconductor scenario includes:

- **Product:** Automotive Control Unit
- **Component:** PMIC-450
- **Vendors:** Alpha Semiconductors, Beta Electronics, Gamma Micro

The upstream handoff model represents the component/request context passed into this agent. Vendor identification itself is outside this agent's responsibility.

## Security

Never commit:

- `.env`
- Mistral API keys
- AWS access keys or secret keys
- local credentials or secret configuration

If a credential is ever accidentally committed, revoke/rotate it immediately and remove it from Git history as appropriate; simply deleting the file in a later commit does not make the exposed credential safe.
