from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

# ---- Config ----
EXCHANGE = "binance"
SYMBOL   = "BTCUSDT"
EVERY    = "100ms"      # or "200ms" if you switched
K_FOLDS = 5
EMBARGO_STEPS = 20      # if 200ms, 10–20 is fine
TAKER_FEE_BPS = 5.0
THRESH_GRID = [0.60, 0.65, 0.70]     # prob gate
EDGE_GRID   = [0.00, 0.05, 0.10]     # micro-edge gate in bps (|micro_edge_bps| >= cut)

BASE_DIR = Path(__file__).resolve().parents[1]
FEAT_GLOB = str(
    BASE_DIR / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"
    / "date=*" / f"features_tob_resample-{EVERY}.parquet"
)

logger.remove(); logger.add(lambda m: print(m, end=""))

# ---- Data ----
def load_features() -> pl.DataFrame:
    df = pl.read_parquet(FEAT_GLOB).sort("ts")
    # clip outliers (positional args for clip)
    df = df.with_columns([
        pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
        pl.col("ret_1s").clip(-0.005, 0.005).alias("ret_1s"),  # 50 bps cap
    ])
    # forward spread for exit leg: 1s ahead
    step = 10 if EVERY == "100ms" else 5 if EVERY == "200ms" else 10
    df = df.with_columns(spread_bps_fwd1 = pl.col("spread_bps").shift(-step))
    # feature set
    df = df.with_columns([
        (((pl.col("microprice") - pl.col("mid")) / pl.col("mid")) * 1e4).alias("micro_edge_bps"),
        (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias("d_spread_bps"),
        pl.col("ofi").rolling_sum(window_size=3,  min_samples=1).alias("ofi_s3"),
        pl.col("ofi").rolling_sum(window_size=10, min_samples=1).alias("ofi_s10"),
        (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
        (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
        (pl.col("microprice") / pl.col("microprice").shift(1) - 1.0).alias("ret_micro_lag1"),
    ])
    need = [
        "dir_1s","ret_1s","spread_bps","spread_bps_fwd1","imb","ofi","micro_edge_bps",
        "d_imb","d_spread_bps","ofi_s3","ofi_s10","ret_mid_lag1","ret_mid_lag2","ret_micro_lag1"
    ]
    return df.drop_nulls(need)

def make_Xy(df: pl.DataFrame):
    feats = [
        "imb","ofi","micro_edge_bps","d_imb","d_spread_bps","ofi_s3","ofi_s10",
        "spread_bps","ret_mid_lag1","ret_mid_lag2","ret_micro_lag1"
    ]
    X = df.select(feats).to_numpy()
    y = df.select("dir_1s").to_numpy().ravel().astype(int)   # {-1,0,1}
    r = df.select("ret_1s").to_numpy().ravel()
    s0 = df.select("spread_bps").to_numpy().ravel()
    s1 = df.select("spread_bps_fwd1").to_numpy().ravel()
    edge = df.select("micro_edge_bps").to_numpy().ravel()
    return X, y, r, s0, s1, edge, feats

# ---- CV ----
def purged_kfold_indices(n, k, embargo):
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    starts, s = [], 0
    for fs in sizes: starts.append(s); s += fs
    ends = [a+b for a,b in zip(starts, sizes)]
    folds = []
    for i in range(k):
        te = np.arange(starts[i], ends[i])
        left, right = max(0, starts[i]-embargo), min(n, ends[i]+embargo)
        tr = np.concatenate([np.arange(0,left), np.arange(right,n)]) if (left>0 or right<n) else np.array([], int)
        folds.append((tr, te))
    return folds

# ---- Eval ----
def pick_with_gates(model, X, classes, thresh, edge_arr, edge_cut):
    proba = model.predict_proba(X)
    idx = {c:i for i,c in enumerate(classes)}
    p_up   = proba[:, idx.get(1, 0)]  if 1 in idx else np.zeros(len(X))
    p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(X))
    best = np.maximum(p_up, p_down)
    sign = np.where(p_up >= p_down, 1, -1)
    preds = np.where(best >= thresh, sign, 0).astype(int)
    # micro-edge gate (in bps)
    preds = np.where(np.abs(edge_arr) >= edge_cut, preds, 0)
    return preds

def eval_net_bps(ret_1s, spr0, spr1, preds, fee_bps):
    trade = (preds != 0).astype(float)
    gross_bps = 1e4 * ret_1s * preds
    net_bps = gross_bps - (0.5*spr0 + 0.5*spr1 + 2.0*fee_bps) * trade
    return (float(trade.mean()),
            float(np.nanmean(gross_bps[trade==1]) if trade.sum() else 0.0),
            float(np.nanmean(net_bps[trade==1]) if trade.sum() else 0.0),
            float(np.nansum(net_bps)))

def main():
    df = load_features()
    X, y, r1, s0, s1, edge, feats = make_Xy(df)
    n = len(y)
    logger.info(f"Samples: {n:,} | Features: {feats} | fee_bps={TAKER_FEE_BPS}\n")

    clf = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=0.0,
        random_state=42,
    )
    folds = purged_kfold_indices(n, K_FOLDS, EMBARGO_STEPS)

    for thr in THRESH_GRID:
        for ecut in EDGE_GRID:
            hit_list, tr_list, g_list, n_list = [], [], [], []
            for (tr, te) in folds:
                if tr.size==0 or te.size==0: continue
                clf.fit(X[tr], y[tr])
                preds = pick_with_gates(clf, X[te], clf.classes_, thr, edge[te], ecut)
                valid = preds != 0
                hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
                tr, g, n, tot = eval_net_bps(r1[te], s0[te], s1[te], preds, TAKER_FEE_BPS)
                hit_list.append(hit); tr_list.append(tr); g_list.append(g); n_list.append(n)
            if hit_list:
                print(f"[thr={thr:.2f}, edge≥{ecut:.2f}bps] "
                      f"hit={np.nanmean(hit_list):.3f} | trade_rate={np.mean(tr_list):.3f} | "
                      f"avg_gross_bps={np.mean(g_list):.2f} | avg_net_bps={np.mean(n_list):.2f}\n")

if __name__ == "__main__":
    main()
