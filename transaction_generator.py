import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from backend_part1 import Config


class TransactionGenerator:
    """
    Generates realistic synthetic transactions compatible with the trained
    XGBoost model while simulating a live banking transaction stream.
    """

    def __init__(
        self,
        dataset_path=None,
        fraud_rate=0.10,          # 10% for demo
        noise_level=0.03,
        random_state=42,
        debug=False
    ):
        if dataset_path is None:
            dataset_path = Path("data/creditcard.csv.gz") if Path("data/creditcard.csv.gz").exists() else Path("data/creditcard.csv")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            if Path("data/creditcard.csv.gz").exists():
                dataset_path = Path("data/creditcard.csv.gz")
            elif Path("data/creditcard.csv").exists():
                dataset_path = Path("data/creditcard.csv")
            elif Path("Dataset/creditcard.csv").exists():
                dataset_path = Path("Dataset/creditcard.csv")

        self.debug = debug

        random.seed(random_state)
        np.random.seed(random_state)

        self.df = pd.read_csv(dataset_path)

        self.legit = self.df[self.df["Class"] == 0].reset_index(drop=True)
        self.fraud = self.df[self.df["Class"] == 1].reset_index(drop=True)

        if self.debug:
            print("=" * 50)
            print("Transaction Generator Initialized")
            print(f"Legitimate Pool : {len(self.legit)}")
            print(f"Fraud Pool      : {len(self.fraud)}")
            print("=" * 50)

        self.fraud_rate = fraud_rate
        self.noise_level = noise_level

        self.current_time = datetime.now()

        self.merchants = [
            "Amazon",
            "Flipkart",
            "Apple",
            "Netflix",
            "Steam",
            "Uber",
            "Google Play",
            "Microsoft",
            "Spotify",
            "PayPal",
            "Walmart",
            "Best Buy",
            "Target",
            "eBay",
            "Costco"
        ]

        self.locations = [
            "New York",
            "London",
            "Delhi",
            "Mumbai",
            "Berlin",
            "Tokyo",
            "Singapore",
            "Paris",
            "Sydney",
            "Dubai",
            "Toronto",
            "San Francisco"
        ]

        self.devices = [
            "Mobile",
            "Desktop",
            "Tablet",
            "POS Terminal"
        ]

    # -------------------------------------------------------

    def set_fraud_rate(self, rate):
        """
        Change fraud rate during runtime.

        Example:
            generator.set_fraud_rate(0.20)
        """

        self.fraud_rate = max(0.0, min(rate, 1.0))

    # -------------------------------------------------------

    def _sample_base_transaction(self):

        if random.random() < self.fraud_rate:

            if self.debug:
                print("Generated FRAUD")

            return self.fraud.sample(1).iloc[0]

        else:

            if self.debug:
                print("Generated LEGIT")

            return self.legit.sample(1).iloc[0]

    # -------------------------------------------------------

    def _perturb(self, row):

        row = row.copy()

        for col in row.index:

            if col == "Class":
                continue

            if col == "Amount":

                factor = np.random.uniform(0.95, 1.05)
                row[col] *= factor
                row[col] = max(0.0, row[col])

            elif col == "Time":

                row[col] += np.random.randint(-60, 60)
                row[col] = max(0.0, row[col])

            else:

                std = self.df[col].std()

                noise = np.random.normal(
                    0,
                    std * self.noise_level
                )

                row[col] += noise

        return row

    # -------------------------------------------------------

    def generate_transaction(self):

        base = self._sample_base_transaction()

        row = self._perturb(base)

        self.current_time += timedelta(
            milliseconds=random.randint(300, 1500)
        )

        transaction = {

            "Transaction_ID":
                "TXN-" + uuid.uuid4().hex[:12].upper(),

            "Timestamp":
                self.current_time.strftime("%Y-%m-%d %H:%M:%S"),

            "Merchant":
                random.choice(self.merchants),

            "Location":
                random.choice(self.locations),

            "Device":
                random.choice(self.devices),

            "Actual_Class":
                int(row["Class"])
        }

        model_input = {}

        for col in self.df.columns:

            if col == "Class":
                continue

            model_input[col] = float(row[col])

        transaction["model_input"] = model_input

        return transaction
