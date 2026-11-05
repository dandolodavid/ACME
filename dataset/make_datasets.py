"""
Synthetic datasets with a known ground truth, used to test ACME.

Run `python dataset/make_datasets.py` to (re)generate the CSV files next to this script.

- regression.csv      y = 4*x1 - 2*x2 + 1.5*c + 0*x3 + noise
                      expected ranking: x1 > x2 > c > x3 (x3 is pure noise);
                      x1 pushes the prediction up, x2 pushes it down; c is categorical {0, 1, 2}
- classification.csv  class depends only on x1: 0 if x1 < -0.5, 2 if x1 > 0.5, else 1
                      x2 and x3 are noise; P(class 2) increases with x1, P(class 0) decreases
- anomaly.csv         4 standard normal features; 2% of rows shifted by +6 only in a1
                      (anomaly_type=1) and 2% only in a2 (anomaly_type=2).
                      anomaly_type is ground truth, not a model feature
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N = 2000
SEED = 0


def make_regression(rng):
    df = pd.DataFrame({'x1': rng.normal(size=N), 'x2': rng.normal(size=N),
                       'x3': rng.normal(size=N), 'c': rng.integers(0, 3, N)})
    df['y'] = 4 * df['x1'] - 2 * df['x2'] + 1.5 * df['c'] + rng.normal(scale=0.1, size=N)
    return df


def make_classification(rng):
    df = pd.DataFrame({'x1': rng.normal(size=N), 'x2': rng.normal(size=N), 'x3': rng.normal(size=N)})
    df['y'] = np.digitize(df['x1'], [-0.5, 0.5])
    return df


def make_anomaly(rng):
    df = pd.DataFrame(rng.normal(size=(N, 4)), columns=['a1', 'a2', 'a3', 'a4'])
    df['anomaly_type'] = 0
    n = N // 50
    df.loc[:n - 1, 'a1'] += 6
    df.loc[:n - 1, 'anomaly_type'] = 1
    df.loc[n:2 * n - 1, 'a2'] += 6
    df.loc[n:2 * n - 1, 'anomaly_type'] = 2
    return df


def main():
    rng = np.random.default_rng(SEED)
    make_regression(rng).to_csv(HERE / 'regression.csv', index=False)
    make_classification(rng).to_csv(HERE / 'classification.csv', index=False)
    make_anomaly(rng).to_csv(HERE / 'anomaly.csv', index=False)


if __name__ == '__main__':
    main()
