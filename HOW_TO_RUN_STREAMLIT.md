# EHA Disease Risk Predictor — Streamlit App Guide

One of the four core docs retained for this project.

Use this guide to install, configure, and run the Streamlit app locally.

---

## What is This App?

An interactive web interface for predicting Cholera and Malaria risk in Kenya's 47 counties. The app allows you to:

- **Make predictions** with custom climate features (rainfall, temperature)
- **Choose two modeling approaches**: Option A (annual data) or Option B (monthly disaggregated data)
- **View risk drivers** — which climate factors push risk up or down
- **Generate explanations** using an AI provider (optional)
- **Explore historical training data**

---

## Supported AI Providers

The app supports the following providers for explanation generation:

- Gemini
- Claude
- Anthropic
- OpenRouter
- DeepSeek
- Kimi

For OpenRouter, you can specify any model name in the additional model field.

---

## Prerequisites

Check that you have:

1. **Python 3.9+**
   ```bash
   # macOS/Linux
   python3 --version
   ```
   
   ```powershell
   # Windows (PowerShell)
   python --version
   ```
   
   If not installed:
   - **macOS**: Install via Homebrew: `brew install python`
   - **Windows**: Download from [python.org](https://python.org) and check "Add Python to PATH" during installation. If already installed, ensure Python is in your PATH by running `python --version` in a new terminal.

2. **Internet connection** — to download dependencies and (optionally) call an AI provider API

3. **(Optional) AI provider API key** — for AI-generated explanations
   - Choose Gemini, Claude, Anthropic, OpenRouter, DeepSeek, or Kimi
   - The app works without it, but you won't get explanations

---

## Installation (First Time Only)

### Step 1: Navigate to the project folder

```bash
# macOS/Linux
cd /path/to/mvp2
```

```powershell
# Windows (PowerShell)
cd C:\path\to\mvp2
```

(Replace `/path/to/mvp2` or `C:\path\to\mvp2` with your actual path to the project)

### Step 2: Create and activate a Python virtual environment

```bash
# macOS/Linux
python3 -m venv streamlit-env
source streamlit-env/bin/activate
```

```powershell
# Windows (PowerShell)
python -m venv streamlit-env
streamlit-env\Scripts\activate
```

```cmd
# Windows (Command Prompt)
python -m venv streamlit-env
streamlit-env\Scripts\activate.bat
```

You should see `(streamlit-env)` in your terminal prompt.

### Step 3: Install dependencies

```bash
pip install -r streamlit_app/requirements.txt
```

This installs:
- **streamlit** — the web framework
- **xgboost**, **numpy**, **pandas** — model inference
- **shap** — feature importance calculation
- **plotly** — interactive charts
- **google-generativeai** — (optional) Google Gemini support
- **requests** — generic provider HTTP calls

---

## Running the App

### Before every session: Activate the environment

```bash
# macOS/Linux
source streamlit-env/bin/activate
```

```powershell
# Windows (PowerShell)
streamlit-env\Scripts\activate
```

```cmd
# Windows (Command Prompt)
streamlit-env\Scripts\activate.bat
```

### Start the app

```bash
# macOS/Linux
cd /path/to/mvp2
streamlit run streamlit_app/app.py
```

```powershell
# Windows (PowerShell)
cd C:\path\to\mvp2
streamlit run streamlit_app/app.py
```

You should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Open in your browser

Click the **Local URL** or copy-paste `http://localhost:8501` into your browser.

---

## Using the App

### Main Interface

The app is organized into four tabs, with **Actual Climate Data** as the primary workflow. The left sidebar contains global model controls and AI settings, while the main panel presents tab-specific inputs.

**Sidebar controls:**
- **Disease selection**: Cholera or Malaria
- **Model option**: A (annual risk model) or B (monthly risk model)
- **Explanation language**: English or Kiswahili
- **AI provider and API key**: Required for optional natural language explanations

**Main panel results:**
- **Risk badge**: HIGH or LOW
- **Risk probability**: Model probability (0–1)
- **Confidence level**: How far the result is from the decision threshold
- **Risk drivers**: Top climate features pushing risk up or down
- **Threshold used**: The disease-specific decision cutoff

### Tabs

**Actual Climate Data Tab:**
1. Select disease, option, and county
2. Select year and, for Option B, month
3. Click "**Fetch & Predict**" to retrieve real climate data from Open-Meteo and run the model
4. For the current year, data is fetched only up to today to avoid requesting future dates

**Custom Scenario Tab:**
- Use manual climate sliders only for what-if simulation
- This path is secondary to the actual climate workflow
- For Option B, month seasonality is encoded automatically

**AI Assistant Tab:**
- Enter a free-form scenario description
- The app extracts county, year, disease, and climate feature values
- It then runs the selected model and generates a bilingual explanation

**Historical Data Tab:**
- Browse training and feature data
- Compare model behaviour against historical county/year climate conditions

---

## Configuration

### Provider API Keys

The app currently accepts a provider API key at runtime in the sidebar. Select your provider from the dropdown and paste the key into the input field.

The selected provider is used by the Streamlit app to parse free-form scenario text into climate feature values and to generate the final bilingual explanation of the predicted risk.

**Supported providers:** Gemini, Claude, Anthropic, OpenRouter, DeepSeek, Kimi.

For OpenRouter, you can specify any model name in the additional model input field that appears when OpenRouter is selected.

**⚠️ Security note**: Do not share your API key publicly.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"

→ You forgot to activate the virtual environment. Run:
```bash
# macOS/Linux
source streamlit-env/bin/activate
```

```powershell
# Windows (PowerShell)
streamlit-env\Scripts\activate
```

Then try `streamlit run streamlit_app/app.py` again.

### "No module named 'xgboost'" (or other packages)

→ Dependencies didn't install. Run:
```bash
pip install -r streamlit_app/requirements.txt
```

### Models not found / "Model key not found"

→ The model JSON files are missing from `data/models/`. You need to:
1. Run the training pipeline first (see [HOW_TO_RUN.md](HOW_TO_RUN.md))
2. Or download pre-trained models from your backup/GCS

Models expected:
- `data/models/xgb_cholera_a.json`
- `data/models/xgb_cholera_b.json`
- `data/models/xgb_malaria_a.json`
- `data/models/xgb_malaria_b.json`

> **Note:** The Streamlit app loads the models from these fixed filenames. If you retrain the models and want to use a new run, replace the corresponding JSON file in `data/models/` with the selected version. Copy or rename the current files first if you want to keep the old serving models.

### App is very slow

→ First run loads models (30–60 seconds). Streamlit caches them, so subsequent predictions are instant.

### Provider API returns "quota exceeded" or "invalid key"

→ Check your API key with your provider. Different providers have different rate limits and error handling.

### Port 8501 already in use

→ Kill the old process or specify a different port:
```bash
# macOS/Linux
streamlit run streamlit_app/app.py --server.port 8502
```

```powershell
# Windows (PowerShell)
streamlit run streamlit_app/app.py --server.port 8502
```

---

## Sharing with Others

### Option A: Local Demo (Same Computer)

1. Activate the environment and start the app (as above)
2. Share the "Network URL" (e.g., `http://192.168.x.x:8501`) with colleagues on the same WiFi
3. They can visit that URL in their browser

⚠️ **Limitation**: Only works while the app is running and on the same network.

### Option B: Deploy to Cloud (Recommended)

For persistent, shareable access, deploy to:

#### **Streamlit Cloud** (Easiest)

1. Push code to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Log in with GitHub → Select your repo → Deploy

**Cost**: Free tier available; auto-scales with usage

**Setup**:
- Create `.streamlit/config.toml` in repo root:
  ```toml
  [theme]
  primaryColor = "#2E86AB"
  backgroundColor = "#0d253f"
  secondaryBackgroundColor = "#1a3a5c"
  ```
- Store secrets in Streamlit Cloud dashboard (no `.streamlit/secrets.toml` in repo)

#### **Google Cloud Run** (More Control)

1. Push to GitHub
2. Create `Dockerfile` (see template below)
3. Deploy:
   ```bash
   # macOS/Linux
   gcloud run deploy eha-risk-predictor \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

   ```powershell
   # Windows (PowerShell)
   gcloud run deploy eha-risk-predictor `
     --source . `
     --platform managed `
     --region us-central1 `
     --allow-unauthenticated
   ```

**Dockerfile template:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY streamlit_app/ ./streamlit_app/
COPY data/ ./data/
COPY requirements.txt .
RUN pip install -r streamlit_app/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Heroku** (Deprecated, use alternatives)

GitHub has deprecated Heroku free tier. Consider Streamlit Cloud or Cloud Run instead.

---

## Advanced: Running Multiple Instances

If you need different configs (e.g., different API keys per user), run on different ports:

```bash
# Terminal 1 (macOS/Linux)
streamlit run streamlit_app/app.py --server.port 8501
```

```powershell
# Terminal 1 (Windows PowerShell)
streamlit run streamlit_app/app.py --server.port 8501
```

```bash
# Terminal 2 (macOS/Linux, same directory, new terminal tab)
source streamlit-env/bin/activate
streamlit run streamlit_app/app.py --server.port 8502
```

```powershell
# Terminal 2 (Windows PowerShell, same directory, new terminal)
streamlit-env\Scripts\activate
streamlit run streamlit_app/app.py --server.port 8502
```

---

## File Structure Reference

```
mvp2/
├── streamlit_app/
│   ├── app.py                (Main Streamlit code)
│   ├── requirements.txt       (Python dependencies)
│   └── .streamlit/
│       ├── config.toml        (UI theme — optional)
│       └── secrets.toml       (API keys — not committed)
├── data/
│   ├── models/               (Trained XGBoost model JSONs)
│   │   ├── xgb_cholera_a.json
│   │   ├── xgb_cholera_b.json
│   │   ├── xgb_malaria_a.json
│   │   ├── xgb_malaria_b.json
│   │   ├── threshold_*.json   (Decision thresholds)
│   │   └── feature_importance_*.csv
│   └── processed/
│       ├── option_a_features.csv  (Historical training data)
│       └── option_b_features.csv
└── HOW_TO_RUN_STREAMLIT.md  (This file)
```

**Note**: On Windows, use backslashes (`\`) instead of forward slashes (`/`) in file paths.

---

## Key Model Details

### Option A (Annual Aggregation)

- **Time grain**: One row per county per year
- **Training data**: 47 counties × 5 years (2020–2024) = ~235 rows
- **Features**: Total/mean/max rainfall, temperature stats, peak month
- **Use case**: Broad, annual-scale planning

### Option B (Monthly Disaggregation)

- **Time grain**: One row per county per month
- **Training data**: ~2,800 rows (monthly-level Bayesian disaggregation)
- **Features**: Rainfall lags/rolling averages, temperature lags, seasonal sine/cosine
- **Use case**: Finer temporal resolution; more data for training

Both models use **walk-forward cross-validation** and **Optuna hyperparameter tuning** for best performance.

---

## Support & Feedback

- **Model questions**: Refer to [HOW_TO_RUN.md](HOW_TO_RUN.md) for training details
- **Streamlit issues**: See [streamlit.io/docs](https://streamlit.io/docs)
- **Bug reports**: File an issue in your project repo

---

## Disclaimer

This app provides **model-generated risk signals for early-warning purposes only**. Always validate predictions with:
- Field surveillance data
- Expert epidemiological review
- Local health authority protocols

Models are trained on climate and historical outbreak data and should not replace established public health decision-making frameworks.

---

**Last updated**: May 2, 2026
