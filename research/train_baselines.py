from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---- Config ----
EXCHANGE = "binance"
SYMBOL   = "BTCUSDT"
RESAMPLE = "1s"

K_FOLDS = 5
EMBARGO_STEPS = 10
TAKER_FEE_BPS = 5.0          # per-side fee; total = 2*fee
PROB_THRESH   = 0.55         # minimum prob to take a directional trade
CLIP_SPREAD_BPS = 15.0       # cap insane spreads (defensive)
CLIP_RET_BPS    = 50.0       # cap |ret_1s| at 50 bps (defensive)

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_GLOB = str(BASE_DIR / "data" / "features" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*" / f"features_resample-{RESAMPLE}.parquet")

logger.remove()
logger.add(lambda m: print(m, end=""))

def load_all_features() -> pl.DataFrame:
    df = pl.read_parquet(FEATURES_GLOB).sort("ts")

    # Defensive clipping against glitches/outliers
    df = df.with_columns([
        pl.when(pl.col("spread_bps").abs() > CLIP_SPREAD_BPS).then(CLIP_SPREAD_BPS).otherwise(pl.col("spread_bps")).alias("spread_bps"),
        pl.when((pl.col("ret_1s")*1e4).abs() > CLIP_RET_BPS).then(pl.lit(np.sign(0)*0)).otherwise(pl.col("ret_1s")).alias("ret_1s"),
    ])

    # Add forward spread for exit leg cost
    df = df.with_columns(spread_bps_fwd1 = pl.col("spread_bps").shift(-1))

    # Simple leakage-safe features
    df = df.with_columns([
        (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_lag1"),
        (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_lag2"),
        (pl.col("mid") / pl.col("mid").shift(5) - 1.0).alias("ret_lag5"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias("d_spread_bps"),
    ])

    df = df.drop_nulls(["ret_lag1","ret_lag2","ret_lag5","d_spread_bps","dir_1s","ret_1s","spread_bps","spread_bps_fwd1"])
    return df

def make_Xy(df: pl.DataFrame):
    feats = ["ret_lag1","ret_lag2","ret_lag5","spread_bps","d_spread_bps"]
    X = df.select(feats).to_numpy()
    y = df.select("dir_1s").to_numpy().ravel().astype(int)     # {-1,0,1}
    r = df.select("ret_1s").to_numpy().ravel()
    spr = df.select("spread_bps").to_numpy().ravel()
    spr1 = df.select("spread_bps_fwd1").to_numpy().ravel()
    return X, y, r, spr, spr1, feats

def purged_kfold_indices(n: int, k: int, embargo: int):
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    starts, s = [], 0
    for fs in sizes: starts.append(s); s += fs
    ends = [st + fs for st, fs in zip(starts, sizes)]
    folds = []
    for i in range(k):
        te = np.arange(starts[i], ends[i])
        left = max(0, starts[i] - embargo)
        right = min(n, ends[i] + embargo)
        tr = np.concatenate([np.arange(0, left), np.arange(right, n)]) if (left>0 or right<n) else np.array([], int)
        folds.append((tr, te))
    return folds

def choose_with_threshold(pipe, X, classes, thresh=0.55):
    """Return {-1,0,1}: only trade if max(p(up),p(down)) >= thresh."""
    proba = pipe.predict_proba(X)   # shape (n, n_classes)
    # map class -> column index
    idx = {c:i for i,c in enumerate(classes)}
    p_up   = proba[:, idx.get(1, 0)]   if 1 in idx else np.zeros(len(X))
    p_down = proba[:, idx.get(-1, 0)]  if -1 in idx else np.zeros(len(X))
    best = np.maximum(p_up, p_down)
    sign = np.where(p_up >= p_down, 1, -1)
    preds = np.where(best >= thresh, sign, 0).astype(int)
    return preds, p_up, p_down, best

def evaluate_net_bps(ret_1s, spread_bps_t, spread_bps_t1, preds, fee_bps):
    """
    Mid-return PnL adjusted by crossing the spread at entry and exit:
    net_bps = 1e4*ret_1s*pred - (0.5*spread_t + 0.5*spread_t1 + 2*fee_bps) * I(trade)
    """
    trade = (preds != 0).astype(float)
    gross_bps = 1e4 * ret_1s * preds
    spread_cost = 0.5*spread_bps_t + 0.5*spread_bps_t1
    net_bps = gross_bps - (spread_cost + 2.0*fee_bps) * trade
    return {
        "trade_rate": float(trade.mean()),
        "avg_gross_bps": float(np.nanmean(gross_bps[trade==1]) if trade.sum() else 0.0),
        "avg_net_bps": float(np.nanmean(net_bps[trade==1]) if trade.sum() else 0.0),
        "total_net_bps": float(np.nansum(net_bps)),
    }

def main():
    df = load_all_features()
    X, y, ret_1s, spr, spr1, feat_cols = make_Xy(df)
    n = len(y)
    logger.info(f"Samples: {n:,} | Features: {feat_cols} | fee_bps={TAKER_FEE_BPS} | prob_thresh={PROB_THRESH}\n")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, solver="lbfgs", class_weight="balanced")),  # multinomial by default
    ])

    folds = purged_kfold_indices(n, K_FOLDS, EMBARGO_STEPS)
    hits, metrics_all = [], []
    for i, (tr, te) in enumerate(folds, 1):
        if tr.size == 0 or te.size == 0: continue
        pipe.fit(X[tr], y[tr])
        preds, p_up, p_dn, best = choose_with_threshold(pipe, X[te], pipe.named_steps["clf"].classes_, PROB_THRESH)

        valid = preds != 0
        hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
        hits.append(hit)

        m = evaluate_net_bps(ret_1s[te], spr[te], spr1[te], preds, TAKER_FEE_BPS)
        logger.info(f"Fold {i}: hit={hit:.3f} | trade_rate={m['trade_rate']:.3f} | "
                    f"avg_gross_bps={m['avg_gross_bps']:.2f} | avg_net_bps={m['avg_net_bps']:.2f}\n")
        metrics_all.append(m)

    if metrics_all:
        avg_hit = float(np.nanmean(hits))
        avg_tr  = float(np.mean([m["trade_rate"] for m in metrics_all]))
        avg_g   = float(np.mean([m["avg_gross_bps"] for m in metrics_all]))
        avg_n   = float(np.mean([m["avg_net_bps"] for m in metrics_all]))
        tot_n   = float(np.sum([m["total_net_bps"] for m in metrics_all]))
        logger.info(f"CV avg → hit={avg_hit:.3f} | trade_rate={avg_tr:.3f} | "
                    f"avg_gross_bps={avg_g:.2f} | avg_net_bps={avg_n:.2f} | total_net_bps={tot_n:.2f}\n")
    else:
        logger.info("No folds evaluated.\n")

if __name__ == "__main__":
    main()
