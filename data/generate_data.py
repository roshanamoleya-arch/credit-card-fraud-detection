"""
generate_data.py
-----------------
Generates a synthetic but realistic credit-card-transaction dataset for
fraud detection. The feature set and fraud-generating logic are modeled
on well-known public fraud-detection datasets (transaction distance,
purchase-price ratios, chip/pin/online-order flags) so the resulting
data has realistic correlations between features and the fraud label.

Run:
    python generate_data.py --n_samples 150000 --fraud_rate 0.006 --seed 42

Output:
    data/transactions.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_samples: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_block(n, fraud: bool):
        if not fraud:
            distance_from_home = rng.gamma(shape=1.5, scale=8, size=n)
            distance_from_last_transaction = rng.gamma(shape=1.2, scale=5, size=n)
            ratio_to_median_purchase_price = rng.gamma(shape=3.0, scale=0.5, size=n)
            repeat_retailer = rng.binomial(1, 0.85, size=n)
            used_chip = rng.binomial(1, 0.65, size=n)
            used_pin_number = rng.binomial(1, 0.45, size=n)
            online_order = rng.binomial(1, 0.35, size=n)
            amount = np.round(rng.lognormal(mean=3.6, sigma=0.9, size=n), 2)
            hour = rng.normal(loc=14, scale=4.5, size=n) % 24
        else:
            distance_from_home = rng.gamma(shape=2.0, scale=60, size=n)
            distance_from_last_transaction = rng.gamma(shape=2.0, scale=45, size=n)
            ratio_to_median_purchase_price = rng.gamma(shape=4.0, scale=3.0, size=n)
            repeat_retailer = rng.binomial(1, 0.25, size=n)
            used_chip = rng.binomial(1, 0.15, size=n)
            used_pin_number = rng.binomial(1, 0.05, size=n)
            online_order = rng.binomial(1, 0.85, size=n)
            amount = np.round(rng.lognormal(mean=5.2, sigma=1.1, size=n), 2)
            hour = rng.normal(loc=3, scale=4.0, size=n) % 24

        merchant_category = rng.choice(
            ["grocery", "electronics", "travel", "restaurant", "online_retail",
             "gas_station", "entertainment", "gambling", "jewelry", "other"],
            size=n,
            p=([0.18, 0.12, 0.06, 0.14, 0.20, 0.12, 0.08, 0.02, 0.03, 0.05]
               if not fraud else
               [0.05, 0.20, 0.10, 0.05, 0.30, 0.03, 0.05, 0.12, 0.08, 0.02]),
        )

        card_present = rng.binomial(1, 0.7 if not fraud else 0.2, size=n)

        return pd.DataFrame({
            "amount": amount,
            "hour_of_day": np.round(hour, 2),
            "distance_from_home": np.round(distance_from_home, 2),
            "distance_from_last_transaction": np.round(distance_from_last_transaction, 2),
            "ratio_to_median_purchase_price": np.round(ratio_to_median_purchase_price, 3),
            "repeat_retailer": repeat_retailer,
            "used_chip": used_chip,
            "used_pin_number": used_pin_number,
            "online_order": online_order,
            "card_present": card_present,
            "merchant_category": merchant_category,
            "is_fraud": int(fraud),
        })

    legit_df = make_block(n_legit, fraud=False)
    fraud_df = make_block(n_fraud, fraud=True)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{100000+i}" for i in range(len(df))])

    # inject a little label noise to mimic real-world imperfect labeling (0.2%)
    noise_idx = rng.choice(df.index, size=int(0.002 * len(df)), replace=False)
    df.loc[noise_idx, "is_fraud"] = 1 - df.loc[noise_idx, "is_fraud"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud-detection dataset")
    parser.add_argument("--n_samples", type=int, default=150_000)
    parser.add_argument("--fraud_rate", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()

    df = generate_dataset(args.n_samples, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['is_fraud'].sum():,} fraud, "
          f"{df['is_fraud'].mean()*100:.3f}% fraud rate) to {args.out}")


if __name__ == "__main__":
    main()
