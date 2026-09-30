"""
===============================================================================
FraudShield AI - Credit Card Fraud Detection Dashboard
Single-File Streamlit Application: app.py
Features:
 - Uses native Streamlit border containers with custom CSS to ensure that
   titles, Plotly charts, donut charts, alert cards, and dataframes are 100%
   contained INSIDE dark rounded card boxes (#0E1326).
 - Baseline dashboard metrics set to realistic volumes (12,547 total, 12,420 legit,
   127 fraud, 34 high risk) which increment dynamically in real-time.
===============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import deque, Counter

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import threading
from streamlit_autorefresh import st_autorefresh
from monitoring_service import MonitoringService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("FraudShieldAI")


# =============================================================================
# SECTION 1: CONFIGURATION & DOMAIN MODELS
# =============================================================================
class Config:
    MODEL_PATH = Path("Models/xgboost.pkl")
    SCALER_PATH = Path("Models/scaler.pkl")
    FEATURE_COLUMNS_PATH = Path("Models/feature_columns.pkl")
    DATASET_PATH = Path("data/creditcard.csv.gz") if Path("data/creditcard.csv.gz").exists() else Path("data/creditcard.csv")

    LOW_RISK = 0.20
    MEDIUM_RISK = 0.50
    HIGH_RISK = 0.85

    MAX_HISTORY = 1000
    MODEL_NAME = "XGBoost"
    VERSION = "1.0"
    DEFAULT_SPEED = 1


@dataclass
class PredictionResult:
    prediction: int
    probability: float
    confidence: float
    latency_ms: float


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str
    amount: float
    time: float
    features: Dict[str, float]
    merchant: str = ""
    location: str = ""
    device: str = ""

    prediction: Optional[int] = None
    probability: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0
    risk_level: str = ""
    decision: str = ""
    status: str = ""
    recommendation: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Alert:
    alert_id: str
    transaction_id: str
    timestamp: str
    amount: float
    probability: float
    severity: str
    status: str = "Open"


@dataclass
class DashboardStats:
    processed: int = 0
    approved: int = 0
    review: int = 0
    fraud: int = 0
    fraud_rate: float = 0
    average_probability: float = 0
    average_latency: float = 0
    average_amount: float = 0
    critical_alerts: int = 0


# =============================================================================
# SECTION 2: BACKEND UTILITIES & DATA PREPROCESSING
# =============================================================================
class Utils:
    @staticmethod
    def transaction_id():
        return f"TXN-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def alert_id():
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%d %B %Y %I:%M:%S %p")

    @staticmethod
    def safe_divide(a, b):
        return 0.0 if b == 0 else a / b


class ModelArtifacts:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load()

    def verify(self):
        for attr, default_path in [
            ("MODEL_PATH", Config.MODEL_PATH),
            ("SCALER_PATH", Config.SCALER_PATH),
            ("FEATURE_COLUMNS_PATH", Config.FEATURE_COLUMNS_PATH)
        ]:
            if not default_path.exists():
                fallback = Path("models") / default_path.name
                if fallback.exists():
                    setattr(Config, attr, fallback)

    def load(self):
        self.verify()
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.feature_columns = joblib.load(Config.FEATURE_COLUMNS_PATH)


class InputValidator:
    @staticmethod
    def validate(amount, time_value, features):
        if amount < 0 or time_value < 0 or len(features) != 28:
            raise ValueError("Invalid transaction parameters.")


class TransactionBuilder:
    @staticmethod
    def create(amount, time_value, features, transaction_id=None, timestamp=None, merchant="", location="", device=""):
        InputValidator.validate(amount, time_value, features)
        return Transaction(
            transaction_id=transaction_id or Utils.transaction_id(),
            timestamp=timestamp or Utils.timestamp(),
            amount=amount,
            time=time_value,
            features=features,
            merchant=merchant,
            location=location,
            device=device,
        )


class DataPreprocessor:
    def __init__(self, artifacts: ModelArtifacts):
        self.scaler = artifacts.scaler
        self.columns = artifacts.feature_columns

    def transform(self, transaction: Transaction):
        row = {"Time": transaction.time, "Amount": transaction.amount}
        row.update(transaction.features)
        df = pd.DataFrame([row])
        df = df[self.columns]
        
        scaler_cols = getattr(self.scaler, 'feature_names_in_', None)
        if scaler_cols is not None:
            df[scaler_cols] = self.scaler.transform(df[scaler_cols])
        else:
            try:
                df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])
            except Exception:
                try:
                    df[["Amount"]] = self.scaler.transform(df[["Amount"]])
                except Exception:
                    pass
        return df


# =============================================================================
# SECTION 3: INFERENCE & RECOMMENDATION ENGINES
# =============================================================================
class PredictionEngine:
    def __init__(self, artifacts, preprocessor):
        self.model = artifacts.model
        self.preprocessor = preprocessor

    def predict(self, transaction: Transaction) -> PredictionResult:
        df = self.preprocessor.transform(transaction)
        start = time.perf_counter()
        prediction = int(self.model.predict(df)[0])
        probabilities = self.model.predict_proba(df)[0]
        latency = (time.perf_counter() - start) * 1000

        result = PredictionResult(
            prediction=prediction,
            probability=float(probabilities[1]),
            confidence=float(max(probabilities)),
            latency_ms=round(latency, 3)
        )
        transaction.prediction = result.prediction
        transaction.probability = result.probability
        transaction.confidence = result.confidence
        transaction.latency_ms = result.latency_ms
        return result


class RiskEngine:
    @staticmethod
    def risk(prob):
        if prob < Config.LOW_RISK: return "Low"
        elif prob < Config.MEDIUM_RISK: return "Medium"
        return "High"

    @staticmethod
    def decision(prob):
        if prob >= Config.HIGH_RISK: return "Blocked"
        elif prob >= Config.MEDIUM_RISK: return "Under Review"
        return "Approved"

    @staticmethod
    def status(decision):
        return {
            "Approved": "Genuine",
            "Under Review": "Suspicious",
            "Blocked": "Fraud Detected"
        }[decision]


class RecommendationEngine:
    @staticmethod
    def generate(decision):
        if decision == "Approved": return ["Approve payment", "Continue passive monitoring"]
        if decision == "Under Review": return ["Hold payment temporarily", "Verify identity"]
        return ["Block payment", "Notify customer", "Raise alert"]


class ReasonEngine:
    @staticmethod
    def generate(prob):
        if prob >= 0.85: return "High probability of anomaly detected."
        if prob >= 0.50: return "Suspicious transaction characteristics."
        return "Normal genuine transaction behavior."


class ReportGenerator:
    @staticmethod
    def build(transaction: Transaction):
        transaction.risk_level = RiskEngine.risk(transaction.probability)
        transaction.decision = RiskEngine.decision(transaction.probability)
        transaction.status = RiskEngine.status(transaction.decision)
        transaction.recommendation = RecommendationEngine.generate(transaction.decision)
        transaction.reason = ReasonEngine.generate(transaction.probability)

        return {
            "Transaction": asdict(transaction),
            "Summary": {
                "Model": Config.MODEL_NAME,
                "Fraud Probability": round(transaction.probability * 100, 2),
                "Risk": transaction.risk_level,
                "Decision": transaction.decision,
            },
        }


class FraudAnalysisEngine:
    def __init__(self, artifacts, preprocessor):
        self.predictor = PredictionEngine(artifacts, preprocessor)

    def analyze(self, transaction: Transaction):
        self.predictor.predict(transaction)
        return ReportGenerator.build(transaction)


# =============================================================================
# SECTION 4: ALERTS, QUEUES & HISTORY
# =============================================================================
class AlertManager:
    def __init__(self):
        self.alerts = []
        self.lock = threading.Lock()

    def create_alert(self, transaction: Transaction):
        if transaction.probability < Config.MEDIUM_RISK:
            return None
        alert = Alert(
            alert_id=Utils.alert_id(),
            transaction_id=transaction.transaction_id,
            timestamp=Utils.timestamp(),
            amount=transaction.amount,
            probability=transaction.probability,
            severity="High" if transaction.probability >= Config.HIGH_RISK else "Medium"
        )
        with self.lock:
            self.alerts.append(alert)
        return alert

    def dataframe(self):
        with self.lock:
            alerts_copy = [asdict(a) for a in self.alerts]
        return pd.DataFrame(alerts_copy)


class HistoryManager:
    def __init__(self):
        self.history = []
        self.lock = threading.Lock()

    def add(self, transaction: Transaction):
        with self.lock:
            self.history.append(transaction)
            if len(self.history) > Config.MAX_HISTORY:
                self.history.pop(0)

    def dataframe(self):
        with self.lock:
            history_copy = list(self.history)
        rows = []
        for t in history_copy:
            r = {
                "Transaction ID": t.transaction_id,
                "Time": t.timestamp.split(" ")[-2] + " " + t.timestamp.split(" ")[-1] if " " in t.timestamp else t.timestamp,
                "Amount (₹)": f"₹{t.amount:,.2f}",
                "Probability": f"{t.probability*100:.1f}%",
                "Risk Level": t.risk_level,
                "Status": t.decision,
                "Location": getattr(t, "location", "Mumbai, IN")
            }
            rows.append(r)
        return pd.DataFrame(rows)


class LiveStatistics:
    def __init__(self):
        self.stats = DashboardStats()
        self.tx_timestamps = deque()
        self.lock = threading.Lock()

    def update(self, transaction: Transaction, alert_created=False):
        with self.lock:
            s = self.stats
            s.processed += 1
            if transaction.decision == "Approved": s.approved += 1
            elif transaction.decision == "Under Review": s.review += 1
            else: s.fraud += 1

            if alert_created: s.critical_alerts += 1
            s.fraud_rate = round((s.fraud / s.processed) * 100, 2)

            now = time.time()
            self.tx_timestamps.append(now)

    def get_tpm(self):
        now = time.time()
        cutoff = now - 60.0
        with self.lock:
            while self.tx_timestamps and self.tx_timestamps[0] < cutoff:
                self.tx_timestamps.popleft()
            return len(self.tx_timestamps)

    def summary(self):
        with self.lock:
            res = asdict(self.stats)
        res["tpm"] = self.get_tpm()
        return res


class MonitoringState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.current_index = 0
        self.speed = Config.DEFAULT_SPEED

    def start(self): self.running = True; self.paused = False
    def pause(self): self.paused = True
    def stop(self): self.running = False; self.paused = False; self.current_index = 0
    def set_speed(self, speed): self.speed = speed


# =============================================================================
# SECTION 5: SYNTHETIC TRANSACTION GENERATOR
# =============================================================================
class TransactionGenerator:
    def __init__(
        self,
        dataset_path=None,
        fraud_rate=0.10,          # 10% default demo fraud rate
        noise_level=0.03,
        random_state=42
    ):
        if dataset_path is None: dataset_path = Config.DATASET_PATH
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            fallback = Path("Dataset/creditcard.csv")
            if fallback.exists(): dataset_path = fallback

        random.seed(random_state)
        np.random.seed(random_state)

        self.df = pd.read_csv(dataset_path)
        self.legit = self.df[self.df["Class"] == 0].reset_index(drop=True)
        self.fraud = self.df[self.df["Class"] == 1].reset_index(drop=True)

        self.fraud_rate = fraud_rate
        self.noise_level = noise_level
        self.current_time = datetime.now()

        self.merchants = ["Amazon", "Flipkart", "Uber", "Zomato", "Swiggy", "Myntra", "Netflix", "Paytm"]
        self.locations = ["Mumbai, MH", "Bangalore, KA", "Delhi, DL", "Hyderabad, TG", "Chennai, TN"]
        self.devices = ["Mobile", "Desktop", "POS Terminal"]

    def set_fraud_rate(self, rate): self.fraud_rate = max(0.0, min(rate, 1.0))

    def generate_transaction(self):
        is_fraud = random.random() < self.fraud_rate
        base = self.fraud.sample(1).iloc[0] if is_fraud and len(self.fraud) > 0 else self.legit.sample(1).iloc[0]
        
        row = base.copy()
        for col in row.index:
            if col == "Class": continue
            if col == "Amount": row[col] = max(100.0, row[col] * np.random.uniform(5.0, 50.0))
            elif col == "Time": row[col] = max(0.0, row[col] + np.random.randint(-60, 60))
            else: row[col] += np.random.normal(0, self.df[col].std() * self.noise_level)

        self.current_time += timedelta(seconds=random.randint(1, 5))

        transaction = {
            "Transaction_ID": "TXN" + str(random.randint(239900, 239999)),
            "Timestamp": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Merchant": random.choice(self.merchants),
            "Location": random.choice(self.locations),
            "Device": random.choice(self.devices),
            "Actual_Class": int(row["Class"])
        }
        transaction["model_input"] = {col: float(row[col]) for col in self.df.columns if col != "Class"}
        return transaction


# =============================================================================
# SECTION 6: LIVE MONITORING & DASHBOARD ENGINES
# =============================================================================
class LiveMonitoringEngine:
    def __init__(self, dataset_path=Config.DATASET_PATH):
        self.generator = TransactionGenerator(dataset_path)
        self.artifacts = ModelArtifacts()
        self.preprocessor = DataPreprocessor(self.artifacts)
        self.analysis = FraudAnalysisEngine(self.artifacts, self.preprocessor)
        self.alerts = AlertManager()
        self.history = HistoryManager()
        self.statistics = LiveStatistics()
        self.state = MonitoringState()
        self.latest_result = None

    def _seed_initial_data(self):
        for _ in range(12):
            tx_data = self.generator.generate_transaction()
            model_input = tx_data["model_input"]
            tx = TransactionBuilder.create(
                amount=float(model_input["Amount"]),
                time_value=float(model_input["Time"]),
                features={f"V{i}": float(model_input[f"V{i}"]) for i in range(1, 29)},
                transaction_id=tx_data["Transaction_ID"],
                timestamp=tx_data["Timestamp"],
                merchant=tx_data["Merchant"],
                location=tx_data["Location"],
                device=tx_data["Device"]
            )
            self.analysis.analyze(tx)
            self.alerts.create_alert(tx)
            self.history.add(tx)
            self.statistics.update(tx)

    def process_next(self):
        raw_tx = self.generator.generate_transaction()
        model_input = raw_tx["model_input"]

        transaction = TransactionBuilder.create(
            amount=float(model_input["Amount"]),
            time_value=float(model_input["Time"]),
            features={f"V{i}": float(model_input[f"V{i}"]) for i in range(1, 29)},
            transaction_id=raw_tx["Transaction_ID"],
            timestamp=raw_tx["Timestamp"],
            merchant=raw_tx["Merchant"],
            location=raw_tx["Location"],
            device=raw_tx["Device"]
        )

        report = self.analysis.analyze(transaction)
        alert = self.alerts.create_alert(transaction)
        self.history.add(transaction)
        self.statistics.update(transaction, alert_created=alert is not None)
        self.state.current_index += 1

        res = {
            "transaction": transaction,
            "report": report,
            "alert": alert,
            "actual_class": raw_tx["Actual_Class"]
        }
        self.latest_result = res
        return res

    def dashboard_snapshot(self):
        stats = self.statistics.summary()
        stats["transactions_processed"] = stats.get("processed", 0)
        stats["fraud_detected"] = stats.get("fraud", 0)
        stats["manual_review"] = stats.get("review", 0)
        stats["approved"] = stats.get("approved", 0)
        return {
            "kpis": stats,
            "statistics": stats,
            "history": self.history.dataframe(),
            "alerts": self.alerts.dataframe()
        }

    def reset(self):
        self.state.stop()
        self.alerts = AlertManager()
        self.history = HistoryManager()
        self.statistics = LiveStatistics()
        self.latest_result = None


class FraudShieldSystem:
    def __init__(self, dataset_path=None):
        self.monitor = LiveMonitoringEngine(dataset_path or Config.DATASET_PATH)

    def process_next(self): return self.monitor.process_next()
    def dashboard(self): return self.monitor.dashboard_snapshot()
    def reset(self): self.monitor.reset()


DEFAULT_GENUINE_V = {
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
}

DEFAULT_FRAUD_V = {
    "V1": -3.043541, "V2": -3.157307, "V3": 1.088463, "V4": 2.288644,
    "V5": 1.359805, "V6": -1.064823, "V7": 0.325574, "V8": -0.067794,
    "V9": -0.270953, "V10": -0.838587, "V11": -0.414575, "V12": -0.503141,
    "V13": 0.676502, "V14": -1.692029, "V15": 2.000635, "V16": 0.666780,
    "V17": -0.339247, "V18": -0.472149, "V19": 0.683500, "V20": 2.102339,
    "V21": 0.661696, "V22": 0.435477, "V23": 1.375966, "V24": -0.293803,
    "V25": 0.279798, "V26": -0.145362, "V27": -0.252773, "V28": 0.035764
}


# =============================================================================
# SECTION 7: STREAMLIT UI CONFIGURATION & STYLING
# =============================================================================
def configure_page():
    st.set_page_config(
        page_title="Credit Card Fraud Detection Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99999;
        }

        /* SIDEBAR TOGGLE BUTTON (SMALL >> ICON ALWAYS VISIBLE) */
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[aria-label*="sidebar"],
        button[aria-label*="Sidebar"] {
            visibility: visible !important;
            display: flex !important;
            background: #0E1326 !important;
            color: #818CF8 !important;
            border: 1px solid #1A223D !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
            transition: all 0.2s ease !important;
        }
        button[data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="collapsedControl"]:hover,
        button[aria-label*="sidebar"]:hover {
            background: #1E1B4B !important;
            border-color: #6366F1 !important;
            color: #C084FC !important;
        }

        /* =============================================================================
           STREAM CONTROLLER & SETTINGS - EXACT TARGET REFERENCE MATCH
           ============================================================================= */

        /* COMMON BUTTON SIZING & FLEX CENTER */
        [data-testid="stElementContainer"]:has(div.st-key-btn_ctrl_start) button,
        [data-testid="stElementContainer"]:has(div.st-key-btn_ctrl_pause) button,
        [data-testid="stElementContainer"]:has(div.st-key-btn_ctrl_resume) button,
        [data-testid="stElementContainer"]:has(div.st-key-btn_ctrl_stop) button,
        [data-testid="stElementContainer"]:has(div.st-key-btn_ctrl_reset) button,
        div.st-key-btn_ctrl_start button,
        div.st-key-btn_ctrl_pause button,
        div.st-key-btn_ctrl_resume button,
        div.st-key-btn_ctrl_stop button,
        div.st-key-btn_ctrl_reset button {
            width: 100% !important;
            height: 48px !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 8px !important;
            margin: 0 !important;
            padding: 0 12px !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            transition: all 200ms ease-in-out !important;
        }

        /* 1. START BUTTON (PRIMARY ELECTRIC BLUE) */
        div.st-key-btn_ctrl_start button:not(:disabled) {
            background: linear-gradient(180deg, #1E40AF 0%, #1D4ED8 100%) !important;
            background-color: #1D4ED8 !important;
            border: 1px solid #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 14px rgba(37, 99, 235, 0.45) !important;
        }
        div.st-key-btn_ctrl_start button:not(:disabled) * {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* 2. PAUSE BUTTON (DARK NAVY / SLATE) */
        div.st-key-btn_ctrl_pause button:not(:disabled) {
            background: #0D1527 !important;
            background-color: #0D1527 !important;
            border: 1px solid #1E293B !important;
            color: #CBD5E1 !important;
            box-shadow: none !important;
        }
        div.st-key-btn_ctrl_pause button:not(:disabled) * {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
        }

        /* 3. RESUME BUTTON (DARK GREEN WITH EMERALD ACCENT) */
        div.st-key-btn_ctrl_resume button:not(:disabled) {
            background: #04271D !important;
            background-color: #04271D !important;
            border: 1px solid #059669 !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 10px rgba(5, 150, 105, 0.3) !important;
        }
        div.st-key-btn_ctrl_resume button:not(:disabled) * {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* 4. STOP BUTTON (DARK RED WITH CRIMSON ACCENT) */
        div.st-key-btn_ctrl_stop button:not(:disabled) {
            background: #2A0910 !important;
            background-color: #2A0910 !important;
            border: 1px solid #991B1B !important;
            color: #F87171 !important;
            box-shadow: 0 0 10px rgba(153, 27, 27, 0.3) !important;
        }
        div.st-key-btn_ctrl_stop button:not(:disabled) * {
            color: #F87171 !important;
            font-weight: 600 !important;
        }

        /* 5. RESET BUTTON (DARK PURPLE WITH VIOLET ACCENT) */
        div.st-key-btn_ctrl_reset button:not(:disabled) {
            background: #121026 !important;
            background-color: #121026 !important;
            border: 1px solid #3B2D54 !important;
            color: #C4B5FD !important;
            box-shadow: 0 0 10px rgba(59, 45, 84, 0.3) !important;
        }
        div.st-key-btn_ctrl_reset button:not(:disabled) * {
            color: #C4B5FD !important;
            font-weight: 600 !important;
        }

        /* DISABLED BUTTONS (INACTIVE TRANSPARENT DARK NAVY WITH SLATE BORDER AND MUTED TEXT) */
        div.st-key-btn_ctrl_start button:disabled,
        div.st-key-btn_ctrl_pause button:disabled,
        div.st-key-btn_ctrl_resume button:disabled,
        div.st-key-btn_ctrl_stop button:disabled {
            background: #090E1A !important;
            background-color: #090E1A !important;
            border: 1px solid #162032 !important;
            color: #334155 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }
        div.st-key-btn_ctrl_start button:disabled *,
        div.st-key-btn_ctrl_pause button:disabled *,
        div.st-key-btn_ctrl_resume button:disabled *,
        div.st-key-btn_ctrl_stop button:disabled * {
            color: #334155 !important;
            font-weight: 500 !important;
        }

        /* TRANSACTION SPEED SELECTBOX (DARK NAVY MATCHING TARGET IMAGE) */
        div[data-testid="stSelectbox"] label {
            color: #94A3B8 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stSelectbox"] > div[data-baseweb="select"],
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[role="combobox"] {
            background-color: #0D1527 !important;
            background: #0D1527 !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }
        div[data-testid="stSelectbox"] div[role="combobox"] * {
            color: #F8FAFC !important;
            background-color: transparent !important;
        }

        /* CYAN BLUE SLIDER (EXACT MATCH FOR TARGET IMAGE) */
        div[data-testid="stSlider"] label {
            color: #94A3B8 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:first-child {
            background: #1E293B !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
            background-color: #00B0FF !important;
            border: 2px solid #80D8FF !important;
            box-shadow: 0 0 12px rgba(0, 176, 255, 0.7) !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[aria-valuenow] {
            background: linear-gradient(90deg, #0091EA 0%, #00B0FF 100%) !important;
            background-color: #00B0FF !important;
        }
        div[data-testid="stSlider"] div[data-testid="stWidgetLabel"] + div *,
        div[data-testid="stSlider"] span[data-testid="stMarkdownContainer"],
        div[data-testid="stSlider"] div[style*="color"],
        div[data-testid="stSlider"] div[data-testid="stTickBar"] + div * {
            color: #00B0FF !important;
            font-weight: 700 !important;
        }

        /* CLEAN INLINE TEXT LINK BUTTONS FOR DASHBOARD TITLE BARS (VIEW ALL ALERTS / TRANSACTIONS) */
        div.st-key-btn_goto_alerts button,
        div.st-key-btn_goto_transactions button {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            min-height: unset !important;
            height: auto !important;
            width: auto !important;
            float: right !important;
            justify-content: flex-end !important;
        }

        div.st-key-btn_goto_alerts button *,
        div.st-key-btn_goto_alerts button p {
            color: #38BDF8 !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            background: transparent !important;
        }

        div.st-key-btn_goto_transactions button *,
        div.st-key-btn_goto_transactions button p {
            color: #0066FF !important;
            font-size: 0.8rem !important;
            font-weight: 700 !important;
            background: transparent !important;
        }

        div.st-key-btn_goto_alerts button:hover *,
        div.st-key-btn_goto_alerts button:hover p {
            color: #7DD3FC !important;
            text-decoration: underline !important;
        }

        div.st-key-btn_goto_transactions button:hover *,
        div.st-key-btn_goto_transactions button:hover p {
            color: #38BDF8 !important;
            text-decoration: underline !important;
        }
        
        /* REMOVE PLOTLY MODEBAR ICONS (ZOOM, PAN, FULLSCREEN, DOWNLOAD) */
        .modebar-container,
        .modebar,
        div[data-testid="stPlotlyChart"] .modebar,
        div[data-testid="stPlotlyChart"] .modebar-container {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }

        /* DARK BACKGROUND */
        .stApp {
            background-color: #070A14;
            color: #F8FAFC;
        }

        [data-testid="stSidebar"] {
            background-color: #0B0F1C;
            border-right: 1px solid #161D33;
        }




        /* Dashboard Main Title */
        .dashboard-main-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin-bottom: 2px;
        }
        
        .dashboard-main-sub {
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 500;
            margin-bottom: 20px;
        }

        /* NATIVE BORDER CONTAINERS STYLING FOR UNIFIED DARK CARDS (#0E1326) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #0E1326 !important;
            border: 1px solid #1A223D !important;
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        }

        /* TOP 6 KPI CARDS GRID */
        .kpi-card-mockup {
            background-color: #0E1326;
            border: 1px solid #1A223D;
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 110px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }

        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .kpi-label {
            font-size: 0.76rem;
            color: #94A3B8;
            font-weight: 600;
        }

        .kpi-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.0rem;
        }

        .kpi-val-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-top: 8px;
        }

        .kpi-val {
            font-size: 1.65rem;
            font-weight: 800;
            color: #FFFFFF;
        }

        .kpi-trend {
            font-size: 0.72rem;
            font-weight: 700;
        }

        .box-title-bar {
            font-size: 1.05rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* RECENT ALERT STACK ITEM CARDS INSIDE BOX */
        .alert-card-item {
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            position: relative;
            background: #080C1A;
            border: 1px solid #161D33;
        }
        
        .alert-card-item.high {
            background: linear-gradient(135deg, rgba(255, 23, 68, 0.08) 0%, rgba(8, 12, 26, 1) 100%);
            border: 1px solid rgba(255, 23, 68, 0.3);
        }
        
        .alert-card-item.medium {
            background: linear-gradient(135deg, rgba(255, 145, 0, 0.08) 0%, rgba(8, 12, 26, 1) 100%);
            border: 1px solid rgba(255, 145, 0, 0.3);
        }

        .alert-card-item.low {
            background: linear-gradient(135deg, rgba(0, 200, 83, 0.08) 0%, rgba(8, 12, 26, 1) 100%);
            border: 1px solid rgba(0, 200, 83, 0.3);
        }

        .badge-pill {
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .badge-high { background: #FF1744; color: #FFFFFF; }
        .badge-medium { background: #FF9100; color: #000000; }
        .badge-low { background: #00C853; color: #FFFFFF; }

        .stButton>button {
            border-radius: 10px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def init_session_state():
    if "backend" not in st.session_state:
        st.session_state.backend = FraudShieldSystem()

    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False

    if "stream_paused" not in st.session_state:
        st.session_state.stream_paused = False

    if "manual_result" not in st.session_state:
        st.session_state.manual_result = None

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    if "v_features" not in st.session_state:
        st.session_state.v_features = DEFAULT_GENUINE_V.copy()


def build_dark_chart_layout():
    return dict(
        title=dict(text="", font=dict(size=12, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=10, b=30),
        height=260,
        xaxis=dict(
            gridcolor="#1A223D",
            zerolinecolor="#1A223D",
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#94A3B8", size=11)
        ),
        yaxis=dict(
            gridcolor="#1A223D",
            zerolinecolor="#1A223D",
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#94A3B8", size=11)
        ),
        legend=dict(font=dict(color="#CBD5E1", size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#161D33", font_color="#FFFFFF", font_family="Plus Jakarta Sans")
    )


def init_session_state():
    if "backend" not in st.session_state:
        st.session_state.backend = FraudShieldSystem()
    if "monitor_service" not in st.session_state:
        st.session_state.monitor_service = MonitoringService(st.session_state.backend)
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    if "stream_paused" not in st.session_state:
        st.session_state.stream_paused = False


# =============================================================================
# SECTION 8: SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar_navigation():
    nav_options = [
        "Dashboard",
        "Live Monitoring",
        "Alerts",
        "Transactions"
    ]

    if hasattr(st, "query_params") and "page" in st.query_params:
        qp_val = str(st.query_params.get("page", "")).lower()
        if qp_val == "alerts":
            target_page = "Alerts"
        elif qp_val == "transactions":
            target_page = "Transactions"
        elif qp_val == "dashboard":
            target_page = "Dashboard"
        elif qp_val == "live":
            target_page = "Live Monitoring"
        else:
            target_page = "Dashboard"
        st.session_state.selected_page = target_page
        st.query_params.clear()

    if "selected_page" not in st.session_state or st.session_state.selected_page not in nav_options:
        st.session_state.selected_page = "Dashboard"

    # Get dynamic badge counts
    alert_count = 0
    try:
        hist_df = st.session_state.backend.monitor.history.dataframe()
        if not hist_df.empty:
            prob_num = pd.to_numeric(hist_df["Probability"].astype(str).str.replace("%", ""), errors="coerce").fillna(0.0)
            status_str = hist_df["Status"].astype(str).str.lower()
            alert_count = len(hist_df[(status_str.str.contains("blocked|fraud", regex=True)) | (prob_num >= 50.0)])
    except Exception:
        pass

    is_streaming = st.session_state.get("is_streaming", False)
    is_paused = st.session_state.get("stream_paused", False)
    current_page = st.session_state.selected_page

    with st.sidebar:
        # ── TITLE SECTION ──
        st.markdown(
            """
            <div style="text-align: left; padding: 12px 0 20px 0;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em;">
                    FraudShield <span style="color: #A855F7;">AI</span>
                </div>
                <div style="font-size: 0.62rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 2px;">Smart Fraud Detection</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── WORKSPACE SECTION ──
        st.markdown(
            """<div style="font-size: 0.62rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; padding: 0 0 10px 4px; margin-top: 4px;">Workspace</div>""",
            unsafe_allow_html=True
        )

        # Dashboard button
        if st.button("Dashboard", key="nav_btn_dashboard", use_container_width=True):
            st.session_state.selected_page = "Dashboard"
            st.rerun()

        # Live Monitoring button
        if st.button("Live Monitoring", key="nav_btn_live", use_container_width=True):
            st.session_state.selected_page = "Live Monitoring"
            st.rerun()

        # ── MONITORING SECTION ──
        st.markdown(
            """<div style="font-size: 0.62rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; padding: 14px 0 10px 4px;">Monitoring</div>""",
            unsafe_allow_html=True
        )

        # Alerts button
        if st.button("Alerts", key="nav_btn_alerts", use_container_width=True):
            st.session_state.selected_page = "Alerts"
            st.rerun()

        # Transactions button
        if st.button("Transactions", key="nav_btn_transactions", use_container_width=True):
            st.session_state.selected_page = "Transactions"
            st.rerun()

        # ── LIVE / ALERT BADGES (injected via CSS pseudo-elements) ──
        live_badge_color = "#00E676" if (is_streaming and not is_paused) else "#64748B"
        live_badge_text = "LIVE" if (is_streaming and not is_paused) else "OFF"
        live_badge_bg = "rgba(0, 230, 118, 0.15)" if (is_streaming and not is_paused) else "rgba(100, 116, 139, 0.15)"
        live_badge_border = "rgba(0, 230, 118, 0.4)" if (is_streaming and not is_paused) else "rgba(100, 116, 139, 0.3)"

        alert_badge_display = "flex" if alert_count > 0 else "none"

        # Determine active states
        dash_active = "true" if current_page == "Dashboard" else "false"
        live_active = "true" if current_page == "Live Monitoring" else "false"
        alerts_active = "true" if current_page == "Alerts" else "false"
        tx_active = "true" if current_page == "Transactions" else "false"

        # Icons as data URIs (inline SVG)
        st.markdown(
            f"""
            <style>
            /* ── SIDEBAR NAV BUTTON BASE ── */
            div.st-key-nav_btn_dashboard button,
            div.st-key-nav_btn_live button,
            div.st-key-nav_btn_alerts button,
            div.st-key-nav_btn_transactions button {{
                background: transparent !important;
                background-color: transparent !important;
                border: 1px solid transparent !important;
                border-radius: 10px !important;
                color: #94A3B8 !important;
                font-weight: 600 !important;
                font-size: 0.88rem !important;
                height: 44px !important;
                min-height: 44px !important;
                padding: 0 14px 0 44px !important;
                text-align: left !important;
                justify-content: flex-start !important;
                margin-bottom: 2px !important;
                transition: all 0.2s ease !important;
                position: relative !important;
                box-shadow: none !important;
            }}
            div.st-key-nav_btn_dashboard button *,
            div.st-key-nav_btn_live button *,
            div.st-key-nav_btn_alerts button *,
            div.st-key-nav_btn_transactions button * {{
                color: inherit !important;
                font-weight: inherit !important;
            }}

            /* ── HOVER STATE ── */
            div.st-key-nav_btn_dashboard button:hover,
            div.st-key-nav_btn_live button:hover,
            div.st-key-nav_btn_alerts button:hover,
            div.st-key-nav_btn_transactions button:hover {{
                background: rgba(99, 102, 241, 0.08) !important;
                color: #C7D2FE !important;
            }}

            /* ── ICONS VIA ::before ── */
            div.st-key-nav_btn_dashboard button::before {{
                content: '';
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                width: 18px;
                height: 18px;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394A3B8' stroke-width='2'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
                background-size: contain;
                background-repeat: no-repeat;
            }}
            div.st-key-nav_btn_live button::before {{
                content: '';
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                width: 18px;
                height: 18px;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394A3B8' stroke-width='2'%3E%3Cpath d='M5.636 18.364a9 9 0 010-12.728M18.364 5.636a9 9 0 010 12.728M8.464 15.536a5 5 0 010-7.072M15.536 8.464a5 5 0 010 7.072M12 12h.01'/%3E%3C/svg%3E");
                background-size: contain;
                background-repeat: no-repeat;
            }}
            div.st-key-nav_btn_alerts button::before {{
                content: '';
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                width: 18px;
                height: 18px;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394A3B8' stroke-width='2'%3E%3Cpath d='M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9'/%3E%3C/svg%3E");
                background-size: contain;
                background-repeat: no-repeat;
            }}
            div.st-key-nav_btn_transactions button::before {{
                content: '';
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                width: 18px;
                height: 18px;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394A3B8' stroke-width='2'%3E%3Cpath d='M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'/%3E%3C/svg%3E");
                background-size: contain;
                background-repeat: no-repeat;
            }}

            /* ── LIVE BADGE on Live Monitoring ── */
            div.st-key-nav_btn_live button::after {{
                content: '{live_badge_text}';
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 0.58rem;
                font-weight: 800;
                color: {live_badge_color};
                background: {live_badge_bg};
                border: 1px solid {live_badge_border};
                padding: 2px 7px;
                border-radius: 6px;
                letter-spacing: 0.06em;
            }}

            /* ── ALERT COUNT BADGE on Alerts ── */
            div.st-key-nav_btn_alerts button::after {{
                content: '{alert_count}';
                display: {alert_badge_display};
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 0.65rem;
                font-weight: 800;
                color: #FFFFFF;
                background: #EF4444;
                width: 22px;
                height: 22px;
                border-radius: 50%;
                align-items: center;
                justify-content: center;
            }}

            /* ── ACTIVE STATE: Dashboard ── */
            div.st-key-nav_btn_dashboard button[data-active="{dash_active}"] {{}}
            {"div.st-key-nav_btn_dashboard button { background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%) !important; color: #FFFFFF !important; border: 1px solid #6366F1 !important; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important; }" if current_page == "Dashboard" else ""}
            {"div.st-key-nav_btn_dashboard button::before { background-image: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23C7D2FE' stroke-width='2'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E\") !important; }" if current_page == "Dashboard" else ""}

            /* ── ACTIVE STATE: Live Monitoring ── */
            {"div.st-key-nav_btn_live button { background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%) !important; color: #FFFFFF !important; border: 1px solid #6366F1 !important; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important; }" if current_page == "Live Monitoring" else ""}
            {"div.st-key-nav_btn_live button::before { background-image: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23C7D2FE' stroke-width='2'%3E%3Cpath d='M5.636 18.364a9 9 0 010-12.728M18.364 5.636a9 9 0 010 12.728M8.464 15.536a5 5 0 010-7.072M15.536 8.464a5 5 0 010 7.072M12 12h.01'/%3E%3C/svg%3E\") !important; }" if current_page == "Live Monitoring" else ""}

            /* ── ACTIVE STATE: Alerts ── */
            {"div.st-key-nav_btn_alerts button { background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%) !important; color: #FFFFFF !important; border: 1px solid #6366F1 !important; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important; }" if current_page == "Alerts" else ""}
            {"div.st-key-nav_btn_alerts button::before { background-image: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23C7D2FE' stroke-width='2'%3E%3Cpath d='M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9'/%3E%3C/svg%3E\") !important; }" if current_page == "Alerts" else ""}

            /* ── ACTIVE STATE: Transactions ── */
            {"div.st-key-nav_btn_transactions button { background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%) !important; color: #FFFFFF !important; border: 1px solid #6366F1 !important; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important; }" if current_page == "Transactions" else ""}
            {"div.st-key-nav_btn_transactions button::before { background-image: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23C7D2FE' stroke-width='2'%3E%3Cpath d='M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'/%3E%3C/svg%3E\") !important; }" if current_page == "Transactions" else ""}
            </style>
            """,
            unsafe_allow_html=True
        )

        # ── SYSTEM STATUS CARD ──
        st.markdown("<hr style='border-color: #161D33; margin: 20px 0 15px 0;'>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background-color: #0E1326; border: 1px solid #1A223D; border-radius: 12px; padding: 14px; font-size: 0.8rem;">
                <div style="color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; font-size: 0.62rem; letter-spacing: 0.1em;">
                    SYSTEM STATUS
                </div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #00C853; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #00C853; display: inline-block;"></span>
                    Operational
                </div>
                <div style="color: #94A3B8; font-size: 0.72rem;">
                    All systems running smoothly
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        return st.session_state.selected_page


def style_transaction_df(df: pd.DataFrame):
    if df.empty:
        return df

    def highlight_status(val):
        s = str(val).lower()
        if "approved" in s or "genuine" in s:
            return "color: #00E676; font-weight: 800; background-color: rgba(0, 230, 118, 0.12);"
        elif "review" in s or "suspicious" in s or "unapproved" in s:
            return "color: #FFAB00; font-weight: 800; background-color: rgba(255, 171, 0, 0.15);"
        elif "blocked" in s or "fraud" in s:
            return "color: #FF1744; font-weight: 800; background-color: rgba(255, 23, 68, 0.18);"
        return ""

    def highlight_risk(val):
        s = str(val).lower()
        if "low" in s:
            return "color: #00E676; font-weight: 700;"
        elif "med" in s:
            return "color: #FFAB00; font-weight: 700;"
        elif "high" in s:
            return "color: #FF1744; font-weight: 700;"
        return ""

    styler = df.style
    if "Status" in df.columns:
        styler = styler.map(highlight_status, subset=["Status"])
    if "Risk Level" in df.columns:
        styler = styler.map(highlight_risk, subset=["Risk Level"])
    return styler


def render_page_header(title: str, subtitle: str, icon: str = ""):
    is_streaming = st.session_state.get("is_streaming", False)
    is_paused = st.session_state.get("stream_paused", False)
    now_time = datetime.now().strftime("%H:%M:%S")

    if is_streaming and not is_paused:
        status_dot = "#00E676"
        status_label = "Monitoring Active"
        status_text_color = "#00E676"
    elif is_paused:
        status_dot = "#FFAB00"
        status_label = "Monitoring Paused"
        status_text_color = "#FFAB00"
    else:
        status_dot = "#FF1744"
        status_label = "Monitoring Stopped"
        status_text_color = "#FF1744"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; padding-bottom: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.07);">
            <div>
                <h1 style="font-size: 1.95rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 35%, #C084FC 70%, #F472B6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em; filter: drop-shadow(0 2px 10px rgba(168, 85, 247, 0.3));">
                    {title}
                </h1>
                <div style="font-size: 0.88rem; color: #94A3B8; font-weight: 500; margin-top: 4px; display: flex; align-items: center; gap: 8px;">
                    <span>{subtitle}</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #0E1326; border: 1px solid #1A223D; border-radius: 12px; padding: 7px 16px; display: flex; flex-direction: column; align-items: flex-end; gap: 2px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: {status_dot}; box-shadow: 0 0 10px {status_dot};"></span>
                        <span style="font-weight: 700; color: {status_text_color};">{status_label}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #64748B;">Last Updated: {now_time}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 1: DASHBOARD PAGE (CHARTS & TABLES 100% CONTAINED INSIDE BOXES)
# =============================================================================
def render_dashboard_page():
    # Main Dashboard Title Header (Vibrant & Visually Appealing)
    render_page_header(
        "Credit Card Fraud Detection Dashboard",
        "AI-powered Real-Time Transaction Monitoring System",
        """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>"""
    )

    dash_snapshot = st.session_state.backend.dashboard()
    kpis = dash_snapshot.get("kpis", dash_snapshot.get("statistics", {}))
    hist_df = dash_snapshot.get("history", pd.DataFrame())

    # 100% DYNAMIC COUNTS (STARTS FROM 0 ON APP START)
    total_tx = kpis.get('transactions_processed', 0)
    approved_tx = kpis.get('approved', 0)
    fraud_tx = kpis.get('fraud_detected', 0)

    app_pct = Utils.safe_divide(approved_tx, total_tx) * 100
    fraud_pct = Utils.safe_divide(fraud_tx, total_tx) * 100

    # CARD 4: TRANSACTIONS / MINUTE (TPM) STATES
    is_streaming = st.session_state.get("is_streaming", False)
    is_paused = st.session_state.get("stream_paused", False)

    if is_streaming and not is_paused:
        tpm_count = kpis.get('tpm', 0)
        tpm_text = f"{tpm_count:,}"
        tpm_sub = "Live Stream"
        tpm_sub_color = "#38BDF8"
        tpm_sub_bg = "rgba(56, 189, 248, 0.15)"
        tpm_sub_border = "1px solid rgba(56, 189, 248, 0.3)"
    elif is_paused:
        tpm_text = "0"
        tpm_sub = "Paused"
        tpm_sub_color = "#FBBF24"
        tpm_sub_bg = "rgba(245, 158, 11, 0.15)"
        tpm_sub_border = "1px solid rgba(245, 158, 11, 0.3)"
    else:
        tpm_text = "0"
        tpm_sub = "Offline"
        tpm_sub_color = "#94A3B8"
        tpm_sub_bg = "rgba(148, 163, 184, 0.15)"
        tpm_sub_border = "1px solid rgba(148, 163, 184, 0.3)"

    # ROW 1: TOP 4 KPI CARDS GRID
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card-mockup" style="background: linear-gradient(135deg, #0E1326 0%, rgba(30, 58, 138, 0.4) 100%); border: 1px solid rgba(59, 130, 246, 0.4); box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);">
                <div class="kpi-header">
                    <span class="kpi-label" style="color: #93C5FD;">Total Transactions</span>
                    <div class="kpi-icon" style="background: rgba(59, 130, 246, 0.25); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); display: flex; align-items: center; justify-content: center;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2"><rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg></div>
                </div>
                <div class="kpi-val-row">
                    <span class="kpi-val">{total_tx:,}</span>
                    <span class="kpi-trend" style="color: #60A5FA; background: rgba(59, 130, 246, 0.15); padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.3);">Live Stream</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card-mockup" style="background: linear-gradient(135deg, #0E1326 0%, rgba(6, 78, 59, 0.4) 100%); border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);">
                <div class="kpi-header">
                    <span class="kpi-label" style="color: #6EE7B7;">Legitimate Transactions</span>
                    <div class="kpi-icon" style="background: rgba(16, 185, 129, 0.25); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); display: flex; align-items: center; justify-content: center;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg></div>
                </div>
                <div class="kpi-val-row">
                    <span class="kpi-val">{approved_tx:,}</span>
                    <span class="kpi-trend" style="color: #34D399; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3);">{app_pct:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card-mockup" style="background: linear-gradient(135deg, #0E1326 0%, rgba(153, 27, 27, 0.4) 100%); border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);">
                <div class="kpi-header">
                    <span class="kpi-label" style="color: #FCA5A5;">Fraudulent Transactions</span>
                    <div class="kpi-icon" style="background: rgba(239, 68, 68, 0.25); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); display: flex; align-items: center; justify-content: center;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
                </div>
                <div class="kpi-val-row">
                    <span class="kpi-val">{fraud_tx:,}</span>
                    <span class="kpi-trend" style="color: #F87171; background: rgba(239, 68, 68, 0.15); padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.3);">{fraud_pct:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card-mockup" style="background: linear-gradient(135deg, #0E1326 0%, rgba(14, 165, 233, 0.4) 100%); border: 1px solid rgba(56, 189, 248, 0.4); box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);">
                <div class="kpi-header">
                    <span class="kpi-label" style="color: #7DD3FC;">Transactions / Minute</span>
                    <div class="kpi-icon" style="background: rgba(56, 189, 248, 0.25); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); display: flex; align-items: center; justify-content: center;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div>
                </div>
                <div class="kpi-val-row">
                    <span class="kpi-val">{tpm_text}</span>
                    <span class="kpi-trend" style="color: {tpm_sub_color}; background: {tpm_sub_bg}; padding: 2px 8px; border-radius: 6px; border: {tpm_sub_border};">{tpm_sub}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: 3 NATIVE BORDER CONTAINERS (EXACT UNIFIED DARK CARDS)
    r2_col1, r2_col2, r2_col3 = st.columns([1.6, 1.4, 1.2])

    with r2_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="box-title-bar">
                    <span>Fraud Detection Timeline</span>
                    <span style="font-size: 0.75rem; color: #94A3B8; background: #080C1A; padding: 4px 10px; border-radius: 6px; border: 1px solid #1A223D;">Today</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if not hist_df.empty:
                df_chart = hist_df.copy()
                df_chart["Probability_Num"] = pd.to_numeric(df_chart["Probability"].astype(str).str.replace("%", ""), errors="coerce").fillna(0.0)
                df_chart["Tx_Index"] = range(1, len(df_chart) + 1)
                fig_frauds = px.line(
                    df_chart,
                    x="Tx_Index",
                    y="Probability_Num",
                    markers=True,
                    color_discrete_sequence=["#FF1744"]
                )
                fig_frauds.update_traces(
                    line=dict(width=3),
                    marker=dict(size=7, color="#FF1744", line=dict(width=2, color="#FFFFFF"))
                )
                fig_frauds.update_layout(build_dark_chart_layout())
                st.plotly_chart(fig_frauds, use_container_width=True, config={'displayModeBar': False})
            else:
                hours = ["12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM"]
                frauds_count = [8, 18, 11, 25, 21, 40, 28, 17]
                df_chart = pd.DataFrame({"Hour": hours, "Frauds": frauds_count})
                fig_frauds = px.line(df_chart, x="Hour", y="Frauds", markers=True, color_discrete_sequence=["#FF1744"])
                fig_frauds.update_traces(line=dict(width=3), marker=dict(size=7, color="#FF1744", line=dict(width=2, color="#FFFFFF")))
                fig_frauds.update_layout(build_dark_chart_layout())
                st.plotly_chart(fig_frauds, use_container_width=True, config={'displayModeBar': False})

    with r2_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="box-title-bar">
                    <span>Transaction Distribution</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            donut_data = pd.DataFrame({
                "Category": ["Legitimate Transactions", "Fraudulent Transactions"],
                "Count": [approved_tx, fraud_tx]
            })
            
            fig_donut = px.pie(
                donut_data,
                names="Category",
                values="Count",
                hole=0.72,
                color="Category",
                color_discrete_map={
                    "Legitimate Transactions": "#00E676",
                    "Fraudulent Transactions": "#FF1744"
                }
            )
            
            fig_layout = build_dark_chart_layout()
            fig_layout["height"] = 340
            fig_layout["margin"] = dict(l=10, r=10, t=10, b=10)
            fig_layout["legend"] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color="#CBD5E1", size=11),
                bgcolor="rgba(0,0,0,0)"
            )
            fig_layout["annotations"] = [
                dict(
                    text=f"<span style='font-size:2rem; font-weight:800; color:#FFFFFF;'>{total_tx:,}</span><br><span style='font-size:0.85rem; font-weight:600; color:#94A3B8;'>Total</span>",
                    x=0.5, y=0.44,
                    showarrow=False,
                    font=dict(family="Inter, sans-serif")
                )
            ]
            fig_donut.update_traces(
                domain=dict(x=[0.05, 0.95], y=[0.0, 0.88]),
                textposition='outside',
                textinfo='percent',
                textfont_size=14
            )
            fig_donut.update_layout(fig_layout)
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    with r2_col3:
        with st.container(border=True):
            alert_title_col, alert_link_col = st.columns([2, 1])
            with alert_title_col:
                st.markdown("""<div style="font-size: 1rem; font-weight: 800; color: #F8FAFC; padding: 4px 0;">Recent Alerts</div>""", unsafe_allow_html=True)
            with alert_link_col:
                if st.button("View All Alerts", key="btn_goto_alerts", use_container_width=True):
                    st.session_state.selected_page = "Alerts"
                    st.rerun()

            fraud_alerts = pd.DataFrame()
            if not hist_df.empty:
                prob_num = pd.to_numeric(hist_df["Probability"].astype(str).str.replace("%", ""), errors="coerce").fillna(0.0)
                status_str = hist_df["Status"].astype(str).str.lower()
                fraud_alerts = hist_df[(status_str.str.contains("blocked|fraud", regex=True)) | (prob_num >= 50.0)]

            if not fraud_alerts.empty:
                recent_fraud = fraud_alerts.tail(2).iloc[::-1]  # Latest 2, newest first
                st.markdown('<div style="max-height: 250px; overflow-y: auto;">', unsafe_allow_html=True)
                for idx, alert_row in recent_fraud.iterrows():
                    raw_prob = str(alert_row.get("Probability", 0)).replace("%", "")
                    p_val = float(pd.to_numeric(raw_prob, errors="coerce") or 0.0)
                    tx_id = alert_row.get("Transaction ID", f"TXN-{idx}")
                    amt_str = str(alert_row.get("Amount (₹)", alert_row.get("Amount ($)", "₹0.00")))
                    time_str = str(alert_row.get("Time", "Just now"))

                    st.markdown(
                        f"""
                        <div class="alert-card-item high">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span class="badge-pill badge-high">FRAUD DETECTED</span>
                                <span style="font-size: 0.68rem; color: #64748B;">{time_str}</span>
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">Transaction ID: <strong style="color: #FFFFFF;">{tx_id}</strong></div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">Probability: <strong style="color: #FF1744;">{p_val:.1f}%</strong></div>
                            <div style="font-size: 0.78rem; color: #94A3B8; display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                <span>Amount: <strong style="color: #FFFFFF;">{amt_str}</strong></span>
                                <span class="badge-pill badge-high">Blocked</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 36px 12px; color: #94A3B8;">
                        <div style="font-size: 0.95rem; font-weight: 700; color: #00E676; margin-bottom: 4px;">No Active Fraud Alerts</div>
                        <div style="font-size: 0.78rem; color: #64748B;">All monitored transactions are currently legitimate.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 3: RECENT TRANSACTIONS CONTAINER (LATEST 10 TRANSACTIONS)
    with st.container(border=True):
        tx_title_col, tx_link_col = st.columns([3, 1.2])
        with tx_title_col:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
                    <span style="font-size: 1rem; font-weight: 800; color: #F8FAFC;">Recent Transactions</span>
                    <span style="font-size: 0.7rem; font-weight: 700; color: #00C853; background: rgba(0, 200, 83, 0.15); padding: 3px 8px; border-radius: 12px; display: flex; align-items: center; gap: 4px;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #00C853;"></span> Live
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with tx_link_col:
            if st.button("View All Transactions", key="btn_goto_transactions", use_container_width=True):
                st.session_state.selected_page = "Transactions"
                st.rerun()

        if not hist_df.empty:
            recent_10 = hist_df.tail(10).iloc[::-1].reset_index(drop=True)
            st.dataframe(style_transaction_df(recent_10), use_container_width=True, height=260, hide_index=True)
        else:
            st.caption("Live transaction feed initialization...")


def add_activity_log(event_text: str):
    if "activity_logs" not in st.session_state:
        st.session_state.activity_logs = []
    now_str = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{now_str}  {event_text}"
    if not st.session_state.activity_logs or st.session_state.activity_logs[-1] != log_entry:
        st.session_state.activity_logs.append(log_entry)
        if len(st.session_state.activity_logs) > 10:
            st.session_state.activity_logs.pop(0)


# =============================================================================
# PAGE 2: LIVE MONITORING PAGE
# =============================================================================
def render_live_monitoring_page():
    render_page_header(
        "Live Stream Monitoring Controls",
        "Real-time Telemetry & Stream Ingestion Controller",
        """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.93 4.93a10 10 0 0 1 14.14 0"/><path d="M7.76 7.76a6 6 0 0 1 8.48 0"/><circle cx="12" cy="12" r="2"/></svg>"""
    )

    monitor_svc = st.session_state.get("monitor_service", None)
    is_running = monitor_svc.is_running if monitor_svc else False
    is_paused = monitor_svc.is_paused if monitor_svc else False

    can_start = not is_running and not is_paused
    can_pause = is_running and not is_paused
    can_resume = is_paused
    can_stop = is_running or is_paused

    with st.container(border=True):
        if is_running and not is_paused:
            status_badge_html = """
            <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; background: rgba(6, 95, 70, 0.25); border: 1px solid #065F46; padding: 4px 14px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span> Monitoring Active
            </span>
            """
        elif is_paused:
            status_badge_html = """
            <span style="font-size: 0.75rem; font-weight: 700; color: #F59E0B; background: rgba(120, 53, 15, 0.25); border: 1px solid #78350F; padding: 4px 14px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #F59E0B; box-shadow: 0 0 8px #F59E0B;"></span> Monitoring Paused
            </span>
            """
        else:
            status_badge_html = """
            <span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; background: rgba(55, 65, 81, 0.25); border: 1px solid #374151; padding: 4px 14px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #64748B;"></span> Monitoring Standby
            </span>
            """

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                    Stream Controller & Settings
                </div>
                {status_badge_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        c_start, c_pause, c_resume, c_stop, c_reset = st.columns(5)
        with c_start:
            if st.button("Start", disabled=not can_start, key="btn_ctrl_start", use_container_width=True):
                st.session_state.monitor_service.start()
                st.session_state.is_streaming = True
                st.session_state.stream_paused = False
                add_activity_log("Monitoring Started")
                st.rerun()

        with c_pause:
            if st.button("Pause", disabled=not can_pause, key="btn_ctrl_pause", use_container_width=True):
                st.session_state.monitor_service.pause()
                st.session_state.stream_paused = True
                add_activity_log("Monitoring Paused")
                st.rerun()

        with c_resume:
            if st.button("Resume", disabled=not can_resume, key="btn_ctrl_resume", use_container_width=True):
                st.session_state.monitor_service.resume()
                st.session_state.stream_paused = False
                add_activity_log("Monitoring Resumed")
                st.rerun()

        with c_stop:
            if st.button("Stop", disabled=not can_stop, key="btn_ctrl_stop", use_container_width=True):
                st.session_state.show_stop_confirm = True

        with c_reset:
            if st.button("Reset", key="btn_ctrl_reset", use_container_width=True):
                st.session_state.show_reset_confirm = True

        # CONFIRMATION DIALOG: STOP MONITORING
        if st.session_state.get("show_stop_confirm", False):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FF1744; margin-bottom: 4px;">
                        Stop Monitoring?
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                        Monitoring will stop immediately.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col_cancel, col_confirm = st.columns(2)
                with col_cancel:
                    if st.button("Cancel Stop", key="btn_cancel_stop", use_container_width=True):
                        st.session_state.show_stop_confirm = False
                        st.rerun()
                with col_confirm:
                    if st.button("Confirm Stop", key="btn_confirm_stop", type="primary", use_container_width=True):
                        st.session_state.monitor_service.stop()
                        st.session_state.is_streaming = False
                        st.session_state.stream_paused = False
                        st.session_state.show_stop_confirm = False
                        add_activity_log("Monitoring Stopped")
                        st.rerun()

        # CONFIRMATION DIALOG: RESET ENGINE
        if st.session_state.get("show_reset_confirm", False):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FF1744; margin-bottom: 4px;">
                        Reset Monitoring Engine?
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                        This will clear monitoring history, clear alerts, and reset counters. This action cannot be undone.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col_cancel, col_confirm = st.columns(2)
                with col_cancel:
                    if st.button("Cancel Reset", key="btn_cancel_reset", use_container_width=True):
                        st.session_state.show_reset_confirm = False
                        st.rerun()
                with col_confirm:
                    if st.button("Confirm Reset", key="btn_confirm_reset", type="primary", use_container_width=True):
                        st.session_state.monitor_service.reset()
                        st.session_state.is_streaming = False
                        st.session_state.stream_paused = False
                        st.session_state.latest_result = None
                        st.session_state.show_reset_confirm = False
                        st.session_state.activity_logs = []
                        add_activity_log("Engine Reset - History Cleared")
                        st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #162032; margin: 18px 0 16px 0;'>", unsafe_allow_html=True)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            current_spd = float(getattr(st.session_state.backend.monitor.state, "speed", 1.0))
            if current_spd >= 5.0:
                default_spd_idx = 2
            elif current_spd >= 2.0:
                default_spd_idx = 1
            else:
                default_spd_idx = 0

            speed_choice = st.selectbox(
                "Transaction Speed",
                ["1x Normal", "2x Speed", "5x Fast"],
                index=default_spd_idx,
                key="speed_select_control"
            )
            s_val = 1.0 if "1x" in speed_choice else (2.0 if "2x" in speed_choice else 5.0)
            if current_spd != s_val:
                st.session_state.backend.monitor.state.set_speed(s_val)
                add_activity_log(f"Transaction Speed set to {speed_choice.split()[0]}x")

        with s_col2:
            current_rate = int(getattr(st.session_state.backend.monitor.generator, "fraud_rate", 0.10) * 100)
            sim_rate = st.slider(
                "Fraud Simulation Rate (%)",
                min_value=1,
                max_value=50,
                value=current_rate,
                key="sim_rate_slider_control"
            )
            if current_rate != sim_rate:
                st.session_state.backend.monitor.generator.set_fraud_rate(sim_rate / 100.0)
                add_activity_log(f"Fraud Rate changed to {sim_rate}%")


# =============================================================================
# PAGE 3: ALERTS PAGE
# =============================================================================
def render_alerts_page():
    render_page_header(
        "Fraud Alerts",
        "AI-detected Fraud Telemetry & High-Risk Security Logs",
        """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>"""
    )

    hist_df = st.session_state.backend.monitor.history.dataframe()

    fraud_alerts = pd.DataFrame()
    if not hist_df.empty:
        prob_num = pd.to_numeric(hist_df["Probability"].astype(str).str.replace("%", ""), errors="coerce").fillna(0.0)
        status_str = hist_df["Status"].astype(str).str.lower()
        fraud_alerts = hist_df[(status_str.str.contains("blocked|fraud", regex=True)) | (prob_num >= 50.0)]

    c_header, c_download = st.columns([3.2, 1.2])
    with c_header:
        total_alert_count = len(fraud_alerts) if not fraud_alerts.empty else 0
        st.markdown(
            f"""
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                All Fraud Alerts ({total_alert_count:,})
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">
                Real-time security ledger of flagged suspicious and blocked transaction records.
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_download:
        if not fraud_alerts.empty:
            csv_data = fraud_alerts.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Alerts",
                data=csv_data,
                file_name="fraud_alerts.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="btn_download_alerts_csv"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        if not fraud_alerts.empty:
            alerts_display = fraud_alerts.iloc[::-1].reset_index(drop=True)
            alerts_display.index = alerts_display.index + 1
            st.dataframe(style_transaction_df(alerts_display), use_container_width=True, height=540)
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 48px 12px; color: #94A3B8;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #00E676; margin-bottom: 4px;">No Active Fraud Alerts</div>
                    <div style="font-size: 0.85rem; color: #64748B;">All monitored transactions are currently legitimate.</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# PAGE 4: TRANSACTIONS PAGE
# =============================================================================
def render_transactions_page():
    render_page_header(
        "Transaction History",
        "Complete Real-Time Transaction Ingestion Ledger",
        """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>"""
    )

    full_hist_df = st.session_state.backend.monitor.history.dataframe()

    c_header, c_download = st.columns([3.2, 1.2])
    with c_header:
        total_tx_count = len(full_hist_df) if not full_hist_df.empty else 0
        st.markdown(
            f"""
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                All Monitored Transactions ({total_tx_count:,})
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">
                Complete ledger of credit card stream telemetry processed by FraudShield AI.
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_download:
        if not full_hist_df.empty:
            csv_data = full_hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Transactions",
                data=csv_data,
                file_name="transactions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="btn_download_transactions_csv"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        if not full_hist_df.empty:
            tx_display = full_hist_df.copy().reset_index(drop=True)
            tx_display.index = tx_display.index + 1
            st.dataframe(style_transaction_df(tx_display), use_container_width=True, height=560)
        else:
            st.caption("No transactions logged in current session.")


# =============================================================================
# SECTION 9: MAIN ENTRY POINT WITH ROUTER
# =============================================================================
def main():
    configure_page()
    init_session_state()
    
    selected_page = render_sidebar_navigation()

    if "Dashboard" in selected_page:
        render_dashboard_page()
    elif "Live Monitoring" in selected_page:
        render_live_monitoring_page()
    elif "Alerts" in selected_page:
        render_alerts_page()
    elif "Transactions" in selected_page:
        render_transactions_page()

    # STREAMLIT-AUTOREFRESH PASSIVE UI REDRAW (NO ST.RERUN() LOOP)
    # Background Monitoring Thread in monitoring_service.py is the ONLY component generating transactions.
    monitor_svc = st.session_state.get("monitor_service", None)
    if monitor_svc and monitor_svc.is_running and not monitor_svc.is_paused:
        st_autorefresh(interval=1000, key="global_monitor_refresh")


if __name__ == "__main__":
    main()
