"""Landsat Satellite land-cover classification: soft-gated MoE baseline vs class-balanced, calibrated LightGBM.

Stand-in for LULUCF remote-sensing categories (paper dataset not stated).
Uses the official UCI Statlog Landsat train/test split, fetched via the ucimlrepo-free HTTP route with
the current UCI hosting (archive.ics.uci.edu/dataset/146) and a fallback to the OpenML mirror if the
legacy path returns 404.
"""
import io, json, os, sys, time, warnings
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
SEED = 42
SEEDS = [42, 43, 44]
MAX_ROWS = 200_000
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/landsat/"
URLS = {"train": BASE + "sat.trn", "test": BASE + "sat.tst"}
# Fallback: the same UCI Landsat files mirrored under the dataset page's static host
ALT_URLS = {"train": "https://archive.ics.uci.edu/static/public/146/data.csv",
            "test": None}
OUT = os.path.dirname(os.path.abspath(__file__))


def get(url):
    """HTTP GET with timeout=60 and retry/backoff on 429/5xx."""
    for attempt in range(5):
        r = requests.get(url, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"failed to download {url}")


def load_split(name):
    """Return (X, y) for the official train/test split, using the legacy whitespace files."""
    df = pd.read_csv(io.StringIO(get(URLS[name])), sep=r"\s+", header=None)
    return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(int)


def load_all():
    """Load official train and test; if the legacy files are unavailable, fall back to the
    OpenML copy of the same Statlog Landsat data, using its own split as train/test."""
    try:
        X_tr, y_tr = load_split("train")
        X_te, y_te = load_split("test")
        return X_tr, y_tr, X_te, y_te, "UCI Statlog (Landsat Satellite) official split"
    except requests.exceptions.HTTPError as e:
        print(f"Legacy UCI files unavailable ({e}); falling back to OpenML 'satimage'.")
        import openml
        ds = openml.datasets.get_dataset(182)  # satimage (Statlog Landsat)
        X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
        X = X.values.astype(float); y = pd.Series(y).astype("category").cat.codes.values.astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        return X_tr, y_tr, X_te, y_te, "OpenML satimage (Statlog Landsat), stratified 70/30 split"


print("Downloading UCI Statlog Landsat data...")
X_tr, y_tr, X_te, y_te, split_desc = load_all()
# Stratified subsample of training rows if very large (guard for runtime)
if len(X_tr) > MAX_ROWS:
    X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_ROWS, stratify=y_tr, random_state=SEED)
n_rows = len(X_tr) + len(X_te)
print(f"Split: {split_desc}")
print(f"Train {X_tr.shape}, test {X_te.shape}, classes {sorted(set(y_tr))}")

# Shared preprocessing: standardise using training statistics only
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
classes = np.unique(y_tr)


class SoftGatedMoE:
    """Soft-gated mixture of experts: logistic-regression gate over K small MLP experts."""

    def __init__(self, n_experts=4, seed=SEED):
        self.K, self.seed = n_experts, seed

    def fit(self, X, y):
        rs = check_random_state(self.seed)
        self.experts = [MLPClassifier((32,), max_iter=300, early_stopping=True,
                                      random_state=self.seed + k).fit(X, y) for k in range(self.K)]
        # Gate is trained on expert-agnostic targets: which expert is most accurate per sample
        correct = np.stack([e.predict(X) == y for e in self.experts], axis=1)
        # Pick a random-tie-broken best expert per sample as gate label
        best = np.array([rs.choice(np.flatnonzero(r == r.max())) for r in correct.astype(int)])
        self.gate = LogisticRegression(max_iter=1000, random_state=self.seed).fit(X, best)
        return self

    def predict_proba(self, X):
        g = self.gate.predict_proba(X)  # n x K
        P = np.stack([e.predict_proba(X) for e in self.experts], axis=1)  # n x K x C
        return np.einsum("nk,nkc->nc", g, P)

    def predict(self, X):
        return classes[self.predict_proba(X).argmax(1)]


def improved(seed):
    """Class-balanced LightGBM, small CV grid on train only, then sigmoid calibration."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    grid = GridSearchCV(LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=2, verbose=-1),
                        {"n_estimators": [200, 400], "num_leaves": [15, 31], "learning_rate": [0.05]},
                        scoring="f1_macro", cv=cv, n_jobs=1).fit(Xs_tr, y_tr)
    print(f"  LightGBM CV best params (seed {seed}): {grid.best_params_}")
    cal = CalibratedClassifierCV(grid.best_estimator_, method="sigmoid", cv=3).fit(Xs_tr, y_tr)
    return cal


def evaluate(model):
    """Shared evaluation: macro-F1 (main), accuracy and worst-class recall (secondary)."""
    p = model.predict(Xs_te)
    rec = recall_score(y_te, p, average=None, labels=classes)
    return {"f1": f1_score(y_te, p, average="macro"), "acc": accuracy_score(y_te, p),
            "worst_recall": rec.min()}


base_res, imp_res = [], []
for s in SEEDS:
    print(f"Seed {s}: fitting baseline MoE...")
    base_res.append(evaluate(SoftGatedMoE(seed=s).fit(Xs_tr, y_tr)))
    print(f"Seed {s}: fitting improved LightGBM...")
    imp_res.append(evaluate(improved(s)))
    print(f"  baseline F1={base_res[-1]['f1']:.4f}  improved F1={imp_res[-1]['f1']:.4f}")

agg = lambda rs, k: float(np.mean([r[k] for r in rs]))
sd = lambda rs, k: float(np.std([r[k] for r in rs]))
b, i = agg(base_res, "f1"), agg(imp_res, "f1")
metrics = {
    "baseline_score": b, "improved_score": i, "improvement": i - b,
    "baseline_score_std": sd(base_res, "f1"), "improved_score_std": sd(imp_res, "f1"),
    "baseline_f1": b, "improved_f1": i,
    "baseline_accuracy": agg(base_res, "acc"), "improved_accuracy": agg(imp_res, "acc"),
    "baseline_worst_class_recall": agg(base_res, "worst_recall"),
    "improved_worst_class_recall": agg(imp_res, "worst_recall"),
}
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("Non-finite metrics; cannot produce meaningful results")

results = {
    "question": "Can a class-balanced, cross-validated gradient-boosting model with per-band attribution-ready features beat a standard mixture-of-experts-style baseline at classifying land cover categories from multispectral pixel data?",
    "method": f"{split_desc}; soft-gated MoE baseline vs class-balanced LightGBM with CV grid and sigmoid calibration; seeds 42-44; macro-F1 on test set.",
    "dataset": {"name": "UCI Statlog (Landsat Satellite)", "source_url": "https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite",
                "n_rows": int(n_rows), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "macro_f1", "higher_is_better": True,
    "baseline": "Soft-gated mixture of 4 small MLP experts with logistic-regression gating (scikit-learn stand-in for MoE).",
    "improved": "Class-balanced LightGBM with 3-fold CV grid search on training data and sigmoid probability calibration, same standardised inputs.",
    "metrics": metrics,
    "notes": "Stand-in dataset, not the paper's LULUCF data; the paper's MoE architecture is unspecified so the baseline is a generic approximation. Test set used only once for final evaluation. Training subsampled only if above 200,000 rows (not triggered here). Results reflect this single dataset and split.",
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Optional chart: baseline vs improved macro-F1 with std error bars
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Baseline MoE", "Improved LightGBM"], [b, i],
       yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], capsize=6, color=["gray", "tab:blue"])
ax.set_ylabel("Test macro-F1 (mean ± std, 3 seeds)"); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "macro_f1_comparison.png"), dpi=120)
print("Wrote results.json and figures/macro_f1_comparison.png")
print(json.dumps(metrics, indent=2))
