# Real-Time Credit Card Fraud Detection Console

An end-to-end, high-throughput machine learning system and real-time operations console designed for continuous credit card transaction ingestion, automated fraud risk scoring, threat intelligence aggregation, and deep forensic transaction auditing.

The system addresses extreme class imbalance using gradient-boosted decision trees, delivering sub-millisecond inference latencies coupled with a decoupled producer-consumer architecture that streams live transaction telemetry over Server-Sent Events (SSE).

> **Disclaimer**: This project is intended for educational, research, and technical demonstration purposes. It is not an officially certified financial diagnostic system and should not be used as the sole basis for production banking or regulatory compliance without institutional verification.

---

## Key Features

- **Extreme Gradient Boosting Classification**: Production pipeline driven by an optimized XGBoost classifier trained on anonymized PCA transaction features, with secondary support for LightGBM, Random Forest, and Logistic Regression baselines.
- **Asynchronous Producer-Consumer Architecture**: Transaction generation and scoring run in an isolated background thread (`MonitoringService`), eliminating thread contention and maintaining low-latency API responsiveness.
- **Real-Time Telemetry Streaming**: Event-driven streaming architecture leveraging Server-Sent Events (SSE) at `/api/stream/sse` to push metrics, alerts, and live transaction records to connected clients without client-side polling.
- **Dynamic Incident Timeline**: Live chronological feed of flagged security threats displaying risk severity, merchant identity, transaction amount, and operational resolution.
- **Risk Stratification and Explainability Engine**: Automated categorization of transactions into Low, Medium, and High threat tiers, accompanied by feature attribution identifying the primary principal components driving the decision.
- **In-Browser Web Audio Synthesizer**: Client-side audio chime alerts synthesized via the Web Audio API on critical fraud detections, equipped with a persistent user toggle.
- **Zero-State State Machine**: Stateful engine controls allowing operators to launch, pause, throttle, tune fraud injection rates, or reset all operational metrics to zero without uncontrolled restarts.
- **Forensic Investigation Modal**: Deep inspection interface rendering the complete 28-dimensional feature vector (V1-V28), inference latency, and hardware telemetry for any individual transaction.
- **Audit Logging and Export**: Searchable security audit tables with on-the-fly CSV generation for fraud alerts and transaction history.
- **Lightweight Single Page Application**: Custom responsive dark-mode dashboard implemented in pure Vanilla JavaScript and CSS with zero frontend framework overhead.

---

## Datasets

The machine learning models are evaluated and calibrated using the benchmark European Cardholders Credit Card Fraud Detection dataset:

- **Total Records**: 284,807 transactions recorded over a 48-hour monitoring window.
- **Class Imbalance**: Highly skewed distribution containing 492 fraudulent transactions (0.172% of total transactions) and 284,315 legitimate transactions (99.828%).
- **Dimensionality**: 30 numerical input features:
  - **V1 through V28**: Anonymized principal components obtained via Principal Component Analysis (PCA) due to privacy constraints.
  - **Time**: Elapsed seconds between each transaction and the first transaction in the dataset.
  - **Amount**: Transaction transaction value in Euros.
- **Storage and Compression**: The repository utilizes a gzip-compressed dataset (`data/creditcard.csv.gz`, ~65.6 MB) to comply with repository file size constraints while preserving full raw sample fidelity.

---

## Architecture and Methodology

The system is organized into modular processing layers:

### 1. Data Ingestion and Perturbation
The `TransactionGenerator` samples genuine and fraudulent vectors from empirical distributions, applying bounded Gaussian perturbation to PCA features and temporal adjustments to simulate an authentic, non-repetitive banking stream.

### 2. Feature Standardization and Alignment
Raw `Time` and `Amount` fields are standardized using a fitted `StandardScaler`. Features are strictly validated against a serialized schema (`feature_columns.pkl`) before matrix construction.

### 3. Inference and Probability Calibration
Input vectors are processed through the `PredictionEngine` using `xgboost.pkl`. The model generates calibrated continuous fraud probabilities alongside binary classifications.

### 4. Risk Engine and Decision Logic
The `RiskEngine` maps continuous probabilities into operational thresholds:
- Probability < 0.35: **Low Risk** (Decision: `APPROVE`)
- 0.35 <= Probability < 0.70: **Medium Risk** (Decision: `MANUAL REVIEW`)
- Probability >= 0.70: **High Risk** (Decision: `BLOCK TRANSACTION`)

---

## High-Level Pipeline

```
Transaction Data Stream (Synthetic / Historical)
                 |
                 v
   Feature Validation & Preprocessing
      (StandardScaler for Time & Amount)
                 |
                 v
      XGBoost Inference Engine
   (Continuous Fraud Probability Calculation)
                 |
                 v
   Risk Engine & Explainability Module
 (Tier Assignment: Approve / Review / Block)
                 |
                 v
 Background Telemetry & Ring Buffer Update
(Sliding Window Stats, KPIs, Alert Tracking)
                 |
                 v
   Server-Sent Events (SSE) Dispatcher
  (/api/stream/sse Push Stream Protocol)
                 |
                 v
    Operations Single-Page Application
 (Real-Time Charts, Incident Feed, Audit Logs)
```

> **Implementation Note**: Detailed hyperparameter search logs, cross-validation splits, and model export pipelines are encapsulated within the offline training scripts. Serialized inference artifacts are stored directly within the `Models/` directory for production deployment.

---

## Experimental Results

Models were evaluated on a stratified test holdout representing the extreme 0.172% fraud distribution. Metric prioritization focused on optimizing the Area Under the Precision-Recall Curve (PR-AUC) and minimizing False Negatives (Recall on Fraud) while maintaining acceptable False Positive rates.

| Model Architecture | Precision | Recall (Fraud) | F1-Score | ROC-AUC | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Production)** | **0.874** | **0.816** | **0.844** | **0.978** | **2.1 ms** |
| LightGBM | 0.852 | 0.796 | 0.823 | 0.972 | 1.8 ms |
| Random Forest | 0.881 | 0.765 | 0.819 | 0.959 | 14.6 ms |
| Logistic Regression | 0.612 | 0.622 | 0.617 | 0.924 | 0.8 ms |

---

## Repository Structure

```
CreditCardFraudv2/
|-- Models/
|   |-- xgboost.pkl             # Primary production gradient boosted classifier
|   |-- lightgbm.pkl            # Benchmark LightGBM model
|   |-- random_forest.pkl       # Benchmark Random Forest model
|   |-- logistic_regression.pkl # Baseline linear classifier
|   |-- scaler.pkl              # Fitted StandardScaler for numerical features
|   `-- feature_columns.pkl     # Schema list of required feature names
|-- data/
|   `-- creditcard.csv.gz       # Gzip-compressed European cardholders dataset
|-- static/
|   |-- css/
|   |   `-- style.css           # Custom dark cyber design system and layout rules
|   |-- js/
|   |   `-- app.js              # SPA controller, SSE consumer, charts & audio logic
|   `-- index.html              # HTML5 application structure and modals
|-- app.py                      # Core ML engines, data structures, and monitoring logic
|-- server.py                   # FastAPI REST API, SSE streaming, and static server
|-- monitoring_service.py       # Thread-safe producer daemon for background ingestion
|-- transaction_generator.py    # Synthetic realistic stream generator
|-- requirements.txt            # Pinned production Python dependencies
|-- .gitignore                  # Git tracking rules (excludes uncompressed 150MB CSV)
|-- PROJECT_DOCUMENTATION.md    # Detailed technical specification and audit manual
`-- README.md                   # Project overview and documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd CreditCardFraudv2
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
python server.py
```
Or start via Uvicorn directly:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Once initialized, open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Cloud Deployment (Render / Native Python)

The application includes automated port binding (`os.environ.get("PORT", 8000)`) and host configuration (`0.0.0.0`), allowing deployment on platforms like Render or Railway without containerization:

1. Push your repository to GitHub.
2. In [Render](https://render.com), create a **New Web Service** connected to your repository.
3. Configure the following deployment parameters:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Deploy the service.
