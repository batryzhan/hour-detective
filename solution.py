# %% [markdown]
# # Hack Club "hour detectives": EDA, validation, distillation, blending
#
# This notebook is for local work (Ryzen 7 / RTX 4050). Features and models are **imported from
# `predict.py`**, so everything checked here is exactly what runs on the server.
# Put `train.csv` next to this notebook.
#
# **v2 changes.** Sections 5–7 are new. They review the 1st-place public repo
# (`craisined/hour-detective`), reproduce its main trick (training on teacher "soft labels"),
# show that its validation leaks, and test the trick honestly with nested CV.
#
# 1. Target and metric
# 2. EDA
# 3. Baseline: OOF of every model + blend
# 4. Feature importance
# 5. What the 1st-place repo does
# 6. Soft-label distillation: leaky vs nested CV
# 7. (optional, GPU) TabPFN teacher → `soft_labels.csv` for `predict.py`
# 8. Blending, post-processing, error analysis
# 9. Decision checklist

# %%
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_DIR = "."
sys.path.insert(0, os.path.abspath("."))
import predict as P

N_FOLDS = int(os.environ.get("N_FOLDS", 10))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
y = np.log1p(train["hours_spent"].values)
print(train.shape)
for lib in ["lightgbm", "catboost", "xgboost", "optuna", "tabpfn"]:
    try:
        print(f"{lib:9s}", __import__(lib).__version__)
    except ImportError:
        print(f"{lib:9s} not installed")

# %% [markdown]
# ## 1. Target and metric
#
# `root_mean_squared_log_error` = RMSE of `log1p`. The provided `log_hours` is `ln(hours)` **without**
# the +1, computed from unrounded hours, so we train on `log1p(hours_spent)` and never on `log_hours`.

# %%
h, lh = train["hours_spent"], train["log_hours"]
print("max |ln(h)    - log_hours| =", round((np.log(h) - lh).abs().max(), 4))
print("max |log1p(h) - log_hours| =", round((np.log1p(h) - lh).abs().max(), 4))
print("constant baseline RMSLE    =", round(P.rmse(np.full_like(y, y.mean()), y), 4))
print("buckets:", train["hours_bucket"].value_counts().to_dict())

# %% [markdown]
# ## 2. EDA

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(train.hours_spent, bins=80); ax[0].set_title("hours_spent")
ax[1].hist(y, bins=60); ax[1].set_title("log1p(hours_spent)")
sns.boxplot(data=train, x="project_category", y=y, ax=ax[2]); ax[2].set_title("log1p(hours) by category")
plt.tight_layout(); plt.show()

# %%
for c in ["page_weight_kb", "num_js_files_loaded", "pct_code_in_first_commit", "pct_primary_language"]:
    m = train[c].isna()
    print(f"{c:26s} missing: mean log1p y = {y[m].mean():.3f} vs present {y[~m].mean():.3f} (n={m.sum()})")

# %%
Xtr, _, cats = P.build_features(train, train.drop(columns=P.LABELS).iloc[:0])
vocab = {c: sorted(Xtr[c].astype(str).unique()) for c in cats}
Xc = P.to_codes_fixed(Xtr, cats, vocab)
num = [c for c in Xtr.columns if c not in cats]
corr = Xtr[num].corrwith(pd.Series(y), method="spearman").dropna()
top = corr.abs().sort_values(ascending=False).head(25).index
plt.figure(figsize=(8, 8)); corr[top].sort_values().plot.barh()
plt.title("Spearman with log1p(hours): top 25"); plt.tight_layout(); plt.show()
print("features:", Xtr.shape[1])

# %% [markdown]
# ## 3. Baseline: OOF of every model in `predict.py`
#
# 10-fold `StratifiedKFold` on `hours_bucket`. CatBoost is the slow one (about 40 s per fit with the
# current settings). Set `N_FOLDS=5` in the environment for a quicker run.

# %%
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=P.SEED).split(train, train["hours_bucket"]))
models = P.make_models(cats, vocab)
oof = {}
for name, fn in models:
    t = time.time(); o = np.zeros(len(y))
    for a, b in folds:
        o[b] = fn(Xtr.iloc[a], y[a], Xtr.iloc[b])
    oof[name] = o
    print(f"{name:10s} OOF RMSLE {P.rmse(o, y):.4f}  ({time.time()-t:.0f}s)")
OOF = pd.DataFrame(oof)
w = P.nnls_weights(OOF.values, y)
blend = OOF.values @ w
print("blend weights:", dict(zip(OOF.columns, w.round(3))))
print("BLEND OOF RMSLE:", round(P.rmse(blend, y), 4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
scores = pd.Series({n: P.rmse(OOF[n], y) for n in OOF} | {"blend": P.rmse(blend, y)}).sort_values()
scores.plot.barh(ax=ax[0], color=["#1b9e77" if i == "blend" else "#2a7ae2" for i in scores.index])
ax[0].set_xlim(scores.min() - 0.01, scores.max() + 0.005); ax[0].set_title("OOF RMSLE (lower is better)")
sns.heatmap(OOF.corr(), annot=True, fmt=".2f", cmap="Blues", vmin=0.9, vmax=1, ax=ax[1])
ax[1].set_title("Correlation of OOF predictions"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Feature importance

# %%
try:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(learning_rate=0.02, n_estimators=1400, num_leaves=24, min_child_samples=25,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=3,
                          importance_type="gain", verbose=-1, random_state=0).fit(Xc, y, categorical_feature=cats)
    imp, title = pd.Series(m.feature_importances_, Xc.columns), "LightGBM gain"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    a, b = folds[0]
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=20, random_state=0,
                                      categorical_features=[c in cats for c in Xc.columns]).fit(Xc.iloc[a], y[a])
    r = permutation_importance(m, Xc.iloc[b], y[b], n_repeats=3, random_state=0, n_jobs=-1)
    imp, title = pd.Series(r.importances_mean, Xc.columns), "Permutation importance (HGB, fold 0)"
plt.figure(figsize=(8, 9)); imp.sort_values().tail(30).plot.barh(); plt.title(title); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. What the 1st-place repo does (`craisined/hour-detective`)
#
# | Part | What it does |
# |---|---|
# | Teacher | `create_labels.py`: **TabPFN** regressor, 5-fold OOF predictions of `log1p(hours)` → column `tabpfn_log_hours` |
# | Students | XGBoost and CatBoost, each 5 fold models, tuned with Optuna (30 trials) |
# | **Key trick** | Students train on `pct * tabpfn_oof + (1 - pct) * log1p(hours)`. Optuna picked `pct ≈ 0.93`, i.e. almost pure teacher labels |
# | Features | Mostly raw columns + a few logs (`span × active days`, `non_first_commit_code`, `incremental_lines_added`); XGBoost drops `has_custom_domain` |
# | Validation | Early stopping on the validation fold; teacher and students use **the same folds and seed** |
# | `predict.py` | Loads 5 pre-trained CatBoost models from JSON and averages them (XGBoost is loaded but not used in the final line) |
# | Their numbers | README: about 0.74 on their validation; leaderboard 0.7328 |
#
# **The leak.** For a row in the student's training folds, its teacher label came from a TabPFN model
# trained on the *other* folds, which include the student's current validation fold. So the validation
# labels sneak into the training targets. The more weight on the teacher (`pct` → 1), the bigger the
# leak, which is why Optuna drove `pct` to 0.93. Early stopping on the validation fold adds a little
# more optimism.
#
# **Why it can still work.** Labels here are very noisy (similar projects get very different hours).
# A smooth teacher prediction is a denoised target, and trees fitted to it can generalise better.
# The question is only *how much*, measured without the leak. Section 6 tests that.
#
# **Leaderboard context.** 0.7328 (1st) vs 0.7458 (ours) on 500 hidden rows. The standard error of an
# RMSLE on 500 rows is about 0.02, so this gap is within noise. Our honest CV is about 0.65 but our
# leaderboard score is about 0.75, so the hidden set looks harder than `train.csv`.

# %% [markdown]
# ## 6. Soft-label distillation: leaky vs nested CV
#
# * **Leaky (repo-style):** teacher OOF computed once on the same outer folds, then used as targets.
# * **Nested (honest):** for each outer fold, the teacher OOF is recomputed **only inside the outer
#   training rows**. The outer validation rows never influence any training target.
#
# Teacher = TabPFN if you set `TEACHER = "tabpfn"` (section 7), otherwise our HGB+ExtraTrees blend.
# Student = LightGBM if installed, else HGB. 5 outer × 5 inner folds (about 5 min with the default teacher).

# %%
mdict = dict(models)
student_name = "lgbm" if "lgbm" in mdict else "hgb"
student = mdict[student_name]
TEACHER = os.environ.get("TEACHER", "ensemble")   # "ensemble" or "tabpfn"


def teacher_ensemble(Xa, ya, Xb):
    return 0.5 * mdict["hgb"](Xa, ya, Xb) + 0.5 * mdict["extratrees"](Xa, ya, Xb)


def teacher_tabpfn(Xa, ya, Xb):
    from tabpfn import TabPFNRegressor
    cols = [c for c in Xc.columns if not c.startswith("log_")]   # skip log copies: TabPFN doesn't need them
    A, B = P.to_codes_fixed(Xa, cats, vocab)[cols], P.to_codes_fixed(Xb, cats, vocab)[cols]
    m = TabPFNRegressor(random_state=0)   # uses CUDA automatically if available
    m.fit(A.values.astype(np.float32), ya)
    return m.predict(B.values.astype(np.float32))


teacher = teacher_tabpfn if TEACHER == "tabpfn" else teacher_ensemble
print("teacher:", TEACHER, "| student:", student_name)

# %%
ALPHAS = [0.0, 0.3, 0.6, 0.9]
outer = list(StratifiedKFold(5, shuffle=True, random_state=1).split(train, train["hours_bucket"]))
buckets = train["hours_bucket"].values

t_leak = np.zeros(len(y))
for a, b in outer:
    t_leak[b] = teacher(Xtr.iloc[a], y[a], Xtr.iloc[b])
print("teacher OOF RMSLE:", round(P.rmse(t_leak, y), 4))

nested = {al: np.zeros(len(y)) for al in ALPHAS}
leaky = {al: np.zeros(len(y)) for al in ALPHAS}
t0 = time.time()
for k, (a, b) in enumerate(outer):
    Xa, ya = Xtr.iloc[a].reset_index(drop=True), y[a]
    t_in = np.zeros(len(ya))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=2).split(Xa, buckets[a]):
        t_in[j] = teacher(Xa.iloc[i], ya[i], Xa.iloc[j])
    for al in ALPHAS:
        nested[al][b] = student(Xa, al * t_in + (1 - al) * ya, Xtr.iloc[b])
        leaky[al][b] = student(Xa, al * t_leak[a] + (1 - al) * ya, Xtr.iloc[b])
    print(f"outer fold {k} done ({time.time()-t0:.0f}s)")

res = pd.DataFrame({"nested (honest)": {al: P.rmse(nested[al], y) for al in ALPHAS},
                    "leaky (repo-style)": {al: P.rmse(leaky[al], y) for al in ALPHAS}})
res.index.name = "alpha (teacher weight)"
print(res.round(4))
ax = res.plot(marker="o", figsize=(7, 4)); ax.set_ylabel("OOF RMSLE"); ax.set_title("Distillation: honest vs leaky CV")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Our result (teacher = HGB+ET, student = HGB, run before writing this notebook):**
#
# | alpha | nested (honest) | leaky (repo-style) |
# |---|---|---|
# | 0.0 | 0.6718 | 0.6718 |
# | 0.3 | 0.6641 | 0.6479 |
# | 0.6 | **0.6632** | 0.6314 |
# | 0.9 | 0.6695 | 0.6234 |
#
# * Honest gain for a single student: about **0.009** at alpha 0.6. Real but modest; our blend of
#   several models already captures most of this smoothing.
# * The leaky curve keeps "improving" as alpha → 1 (down to 0.623). That gain is fake: it is the leak,
#   and it explains why Optuna chose 0.93 in the repo.
# * Choose alpha from the **nested** column only. A stronger teacher (TabPFN) is the only way this trick
#   could beat our current blend.

# %% [markdown]
# ## 7. (optional, GPU) TabPFN teacher → `soft_labels.csv`
#
# ```bash
# pip install tabpfn          # pulls in torch; the RTX 4050 is used automatically
# ```
# The first run downloads model weights from Hugging Face; newer TabPFN versions may ask you to log in
# and accept the licence once. TabPFN does **not** run on the server; we only ship its predictions.
#
# Steps:
# 1. Set `TEACHER = "tabpfn"` (or run with the env var `TEACHER=tabpfn`) and re-run section 6.
#    Note the best **nested** alpha.
# 2. Set `WRITE_SOFT_LABELS = True` below and run the cell: it writes `soft_labels.csv` (5-fold OOF
#    teacher predictions for every train row).
# 3. Commit `soft_labels.csv` next to `predict.py` and set `DISTILL_ALPHA` in `predict.py` to that alpha.
#    `predict.py` joins by `project_id`; rows without a soft label keep the true target, so it is safe
#    even if the organisers' `train.csv` differs.

# %%
WRITE_SOFT_LABELS = False
if WRITE_SOFT_LABELS:
    soft = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=P.SEED).split(train, buckets):
        soft[b] = teacher(Xtr.iloc[a], y[a], Xtr.iloc[b])
    print("teacher OOF RMSLE:", round(P.rmse(soft, y), 4))
    pd.DataFrame({"project_id": train["project_id"], "soft_log1p": soft.round(5)}).to_csv("soft_labels.csv", index=False)
    print("wrote soft_labels.csv")

# %% [markdown]
# **Caveat for `predict.py`.** Its internal 5-fold OOF (used only to pick blend weights) becomes slightly
# optimistic once soft labels are on, for the same reason as above. The test predictions themselves are
# not affected: no test label is ever used. Trust section 6, not the `predict.py` log, for the gain.

# %% [markdown]
# ## 8. Blending, post-processing, error analysis

# %%
from sklearn.linear_model import LinearRegression
st = np.zeros(len(y))
for a, b in folds:
    st[b] = LinearRegression(positive=True).fit(OOF.values[a], y[a]).predict(OOF.values[b])
per_fold = [P.rmse(blend[b], y[b]) for _, b in folds]
print(f"NNLS blend {P.rmse(blend, y):.4f} | positive-LR stack {P.rmse(st, y):.4f} | "
      f"per fold mean {np.mean(per_fold):.4f} std {np.std(per_fold):.4f}")


def score(pl):
    return root_mean_squared_log_error(train.hours_spent, np.clip(np.expm1(pl), 0.1, 140.7))


mu = blend.mean()
print("shift:  ", {s: round(score(blend + s), 4) for s in [-0.05, -0.02, 0, 0.02, 0.05]})
print("stretch:", {k: round(score(mu + k * (blend - mu)), 4) for k in [0.95, 1.0, 1.05, 1.1]})

# %%
err = train.assign(pred=np.expm1(blend), res=blend - y)
print(err.groupby("hours_bucket")["res"].agg(mean="mean", rmse=lambda r: np.sqrt(np.mean(r ** 2)), n="count").round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(y, blend, s=3, alpha=0.3); ax[0].plot([0, 5], [0, 5], "r--")
ax[0].set_xlabel("true log1p"); ax[0].set_ylabel("predicted log1p"); ax[0].set_title("OOF: predicted vs true")
sns.boxplot(data=err, x="hours_bucket", y="res", order=P.BUCKETS, ax=ax[1]); ax[1].axhline(0, color="r", ls="--")
ax[1].set_title("Residual by bucket (log1p)"); plt.tight_layout(); plt.show()

# %% [markdown]
# Typical picture: small projects are over-predicted and 100+ hour projects under-predicted (regression to
# the mean). Post-processing changes the score by less than 0.001, so `predict.py` only clips to
# `[0.1, 140.7]`.
#
# ## 9. Decision checklist
#
# 1. Keep `DISTILL_ALPHA = 0.0` unless section 6 with `TEACHER = "tabpfn"` beats alpha 0 by
#    **more than about 0.01 nested**. Smaller gains are within fold noise (std about 0.02).
# 2. If it does: write `soft_labels.csv`, commit it, set `DISTILL_ALPHA`, then run `predict.py` on a
#    simulated split (`make_sim.py`) and time it.
# 3. Don't tune on the leaderboard: 500 hidden rows give about ±0.02 noise, and the public run is only 20 rows.