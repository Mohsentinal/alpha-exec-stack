# run_pipeline.py
from pathlib import Path
import subprocess, time, sys, datetime, os, signal

ROOT = Path(__file__).parent
PY   = sys.executable  # uses your PyCharm/venv interpreter
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

# -------- New: sensible defaults you can override per run --------
os.environ.setdefault("RESAMPLE_MS", "200")  # try "100" for faster grid
os.environ.setdefault("FWD_SECS",    "2")    # maker fill horizon in seconds
# -----------------------------------------------------------------

DURATION_MIN = 30   # how long to collect data before training
PRINT_EVERY  = 60   # status print cadence (seconds)

def spawn(script_relpath, log_name):
    """Start a process with a logfile (Windows-friendly)."""
    log = open(LOGS / log_name, "a", buffering=1, encoding="utf-8")
    p = subprocess.Popen(
        [PY, str(ROOT / script_relpath)],
        cwd=ROOT,
        stdout=log, stderr=log,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    return p, log

def run(cmd_relpath):
    subprocess.run([PY, str(ROOT / cmd_relpath)], cwd=ROOT, check=True)

def terminate(p):
    try:
        p.send_signal(signal.CTRL_BREAK_EVENT)  # soft stop on Windows
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

if __name__ == "__main__":
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[runner] starting pipeline at {ts}")
    print(f"[runner] config: RESAMPLE_MS={os.environ.get('RESAMPLE_MS')}ms | FWD_SECS={os.environ.get('FWD_SECS')}")

    # 1) start ingestors (parallel)
    p1, log1 = spawn("ingest/binance_bookticker_ingest.py", f"bookticker_{ts}.log")
    p2, log2 = spawn("ingest/binance_trades_ingest.py",    f"trades_{ts}.log")
    print(f"[runner] ingestors up. PIDs: bookticker={p1.pid} trades={p2.pid}")

    # 2) wait while collecting
    secs = DURATION_MIN * 60
    for s in range(0, secs, PRINT_EVERY):
        time.sleep(min(PRINT_EVERY, secs - s))
        left = secs - (s + PRINT_EVERY if s + PRINT_EVERY < secs else secs)
        print(f"[runner] collecting… ~{left}s left")

    # 3) stop ingestors (gracefully if possible)
    print("[runner] stopping ingestors…")
    terminate(p1); terminate(p2)
    log1.close(); log2.close()

    # 4) build features + join with trades
    print("[runner] build features (TOB)…")
    run("research/build_features_tob.py")

    print("[runner] join TOB with trades features…")
    run("research/build_features_tob_trades.py")

    # 5) train/evaluate
    print("[runner] train GBM (taker)…")
    run("research/train_tob_gbm.py")

    print("[runner] evaluate maker entries…")
    run("research/train_tob_maker.py")

    print("[runner] done. logs in ./logs, data in ./data, results printed above.")
