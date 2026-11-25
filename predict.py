"""
Hack Club "hour detectives" — predict hours_spent.

Run:  python predict.py
Reads train.csv and test.csv next to this file, writes predictions.csv
(project_id,hours_spent).

Design
------
* Target: the metric is sklearn's root_mean_squared_log_error = RMSE of log1p(y).
  NOTE: the provided `log_hours` column is ln(hours) (no +1), so we do NOT train on it;
  we train every model on y = log1p(hours_spent) with plain RMSE loss.
* Everything is trained inside this script (~1-3 min on CPU, <600 MB RAM), so there are
  no pickles and no library-version mismatches.
* Models: LightGBM / CatBoost / XGBoost when importable + always-available sklearn models
  (HistGradientBoosting, ExtraTrees, Ridge). Any model that fails is simply skipped.
* Blend weights: non-negative least squares on 5-fold out-of-fold predictions computed
  right here, so the blend adapts to whichever libraries are actually present.
* Post-processing: clip to [0.1, 140.7] (the observed label range).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_FOLDS = 5
Y_MIN, Y_MAX = 0.1, 140.7
TIME_BUDGET_S = 420          # stop adding OOF work after this (limit is 600 s)
N_THREADS = max(1, min(4, os.cpu_count() or 1))
USE_GROUP_AGGS = False      # train+test group stats: no CV gain in our ablation (0.667 vs 0.669)

LABELS = ["hours_spent", "log_hours", "hours_bucket"]
BUCKETS = ["<10", "10-30", "30-100", "100+"]

ORDINAL_MAPS = {
    "author_prior_projects": {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "author_account_age_years": {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "commit_cadence": {"sporadic": 0, "daily": 1, "frequent": 2},
}
CAT_COLS = ["primary_language", "framework", "hosting_type", "project_category"]
BOOL_COLS = [
    "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend",
    "has_database", "has_custom_art", "readme_has_install_instructions", "has_license",
    "has_docs_folder", "url_is_live", "is_mobile_responsive", "has_custom_domain",
    "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial", "is_forked_from_template",
]
# heavy-tailed counts -> log1p copies help linear models and stabilise ratios
LOG_COLS = [
    "num_commits", "active_commit_days", "project_span_days", "lines_added_total",
    "lines_deleted_total", "median_lines_per_commit", "max_lines_in_one_commit",
    "lines_of_code", "num_source_files", "num_directories", "num_dependencies",
    "num_test_files", "avg_file_length", "num_image_files", "num_audio_files",
    "num_3d_model_files", "num_font_files", "asset_size_mb", "readme_word_count",
    "page_weight_kb", "load_time_ms", "description_word_count", "stars", "forks",
    "num_issues", "num_pull_requests", "num_releases", "num_branches", "num_merge_commits",
]
GROUP_KEYS = ["project_category", "primary_language", "framework", "author_prior_projects"]
GROUP_VALS = ["lines_of_code", "num_commits", "active_commit_days", "asset_size_mb",
              "readme_word_count", "lines_added_total"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
def _to_bool(s: pd.Series) -> pd.Series:
    m = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0,
         1: 1.0, 0: 0.0}
    return s.map(lambda v: m.get(v, np.nan) if not (isinstance(v, float) and np.isnan(v)) else np.nan).astype(float)


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Returns (X_train, X_test, cat_feature_names). Uses train+test together for
    label-free statistics (group aggregates, category codes)."""
    full = pd.concat([train.drop(columns=[c for c in LABELS if c in train.columns]), test],
                     axis=0, ignore_index=True)
    n_tr = len(train)
    out = pd.DataFrame(index=full.index)

    # ---- raw numeric columns
    raw_num = [c for c in full.columns
               if c not in CAT_COLS + BOOL_COLS + list(ORDINAL_MAPS) + ["project_id"]]
    for c in raw_num:
        out[c] = _num(full, c)
    for c in BOOL_COLS:
        out[c] = _to_bool(full[c]) if c in full.columns else np.nan
    for c, mp in ORDINAL_MAPS.items():
        out[c] = full[c].astype("object").map(mp).astype(float) if c in full.columns else np.nan

    # ---- categoricals: integer codes (train+test vocabulary), rare -> "rare"
    cat_names = []
    for c in CAT_COLS:
        s = full[c].astype("object").where(full[c].notna(), "__nan__").astype(str) \
            if c in full.columns else pd.Series("__nan__", index=full.index)
        vc = s.value_counts()
        s = s.where(s.map(vc) >= 10, "rare")
        out[c + "_cat"] = s.values
        cat_names.append(c + "_cat")
        out[c + "_freq"] = s.map(s.value_counts()).astype(float).values / len(s)

    e = 1e-5
    g = out  # shorthand
    nc, acd, span = g["num_commits"], g["active_commit_days"], g["project_span_days"]
    loc, added, deleted = g["lines_of_code"], g["lines_added_total"], g["lines_deleted_total"]
    files, dirs = g["num_source_files"], g["num_directories"]
    pfc = g["pct_code_in_first_commit"]

    # ---- 1. git velocity & consistency
    g["commits_per_active_day"] = nc / (acd + e)
    g["active_days_ratio"] = acd / (span + 1)
    g["loc_per_commit"] = loc / (nc + 1)
    g["loc_per_active_day"] = loc / (acd + 1)
    g["added_per_active_day"] = added / (acd + 1)
    g["net_lines_ratio"] = loc / (added + 1)
    g["net_added"] = added - deleted
    g["first_commit_loc_estimate"] = added * pfc
    g["non_first_commit_lines"] = added * (1 - pfc)
    g["log_non_first_commit_lines"] = np.log1p(g["non_first_commit_lines"].clip(lower=0))
    g["non_generic_commits"] = nc * (1 - g["pct_generic_commit_msgs"])
    g["log_non_generic_commits"] = np.log1p(g["non_generic_commits"].clip(lower=0))
    g["merge_ratio"] = g["num_merge_commits"] / (nc + 1)
    g["branches_per_author"] = g["num_branches"] / (g["num_commit_authors"] + 1)
    g["late_night_commits_count"] = nc * g["pct_commits_late_night"]
    g["weekend_commits_count"] = nc * g["pct_commits_weekend"]
    g["max_commit_share"] = g["max_lines_in_one_commit"] / (added + 1)
    g["median_to_max_commit"] = g["median_lines_per_commit"] / (g["max_lines_in_one_commit"] + 1)
    g["commit_msg_effort"] = nc * g["avg_commit_msg_length"] * (1 - g["pct_generic_commit_msgs"])
    g["span_minus_active"] = span - acd
    g["commits_per_author"] = nc / (g["num_commit_authors"] + e)
    g["single_day_project"] = ((acd <= 1) | (span <= 0)).astype(float)
    g["one_shot_upload"] = ((pfc >= 0.9) & (nc <= 3)).astype(float)
    g["sqrt_acd_x_log_loc"] = np.sqrt(acd.clip(lower=0)) * np.log1p(loc.clip(lower=0))

    # ---- 2. code complexity
    g["avg_files_per_dir"] = files / (dirs + 1)
    g["code_density"] = loc / (files * g["max_dir_depth"] + 1)
    g["comment_lines_total"] = loc * g["comment_line_ratio"]
    g["test_coverage_proxy"] = g["num_test_files"] / (files + 1)
    g["engineering_maturity_score"] = g[["has_tests", "has_ci_config", "has_dockerfile",
                                         "has_backend", "has_database", "has_license",
                                         "has_docs_folder"]].fillna(0).sum(axis=1)
    g["deps_per_loc"] = g["num_dependencies"] / (loc + 1)
    g["loc_per_language"] = loc / (g["num_languages"] + 1)
    g["secondary_lang_loc"] = loc * (1 - g["pct_primary_language"])

    # ---- 3. assets, live site, presentation
    g["total_asset_files"] = (g["num_image_files"] + g["num_audio_files"]
                              + g["num_3d_model_files"] + g["num_font_files"])
    g["log_total_asset_files"] = np.log1p(g["total_asset_files"])
    g["avg_asset_size_kb"] = g["asset_size_mb"] * 1024 / (g["total_asset_files"] + 1)
    g["assets_per_kloc"] = g["total_asset_files"] / (loc / 1000 + 1)
    g["readme_richness"] = g["readme_word_count"] * (
        1 + 0.2 * g["readme_num_headings"] + 0.3 * g["readme_num_images"]
        + 0.3 * g["readme_num_code_blocks"])
    g["log_readme_richness"] = np.log1p(g["readme_richness"])
    g["live_site_complexity"] = g["page_weight_kb"] / (g["num_js_files_loaded"] + 1)
    g["is_real_app_proxy"] = (g["url_is_live"].fillna(0) * (1 + g["has_custom_domain"].fillna(0))
                              * np.log1p(g["num_internal_pages"].fillna(0))
                              * (1 + g["is_mobile_responsive"].fillna(0)))
    g["community_score"] = np.log1p(g["stars"] + 2 * g["forks"] + g["num_issues"]
                                    + g["num_pull_requests"] + g["num_releases"])

    # ---- 4. anti-fraud / template signals
    g["template_flags"] = g[["is_clone_or_tutorial", "is_forked_from_template"]].fillna(0).sum(axis=1)
    g["template_x_first_commit"] = g["template_flags"] * pfc.fillna(0.5)
    g["ai_wrapper"] = g["mentions_ai_api"].fillna(0) * (g["deps_per_loc"] * 1000)
    g["custom_art_x_assets"] = g["has_custom_art"].fillna(0) * g["log_total_asset_files"]
    g["is_missing_git"] = ((nc.fillna(0) <= 0) | pfc.isna()).astype(float)
    g["is_missing_code"] = g["pct_primary_language"].isna().astype(float)
    g["is_missing_live_site"] = g["page_weight_kb"].isna().astype(float)
    g["is_missing_js"] = g["num_js_files_loaded"].isna().astype(float)
    g["is_missing_readme"] = (g["readme_word_count"].fillna(0) <= 0).astype(float)
    g["row_null_count"] = full.drop(columns=["project_id"]).isna().sum(axis=1).astype(float).values

    # ---- log copies
    for c in LOG_COLS:
        if c in g:
            g["log_" + c] = np.log1p(g[c].clip(lower=0))

    # ---- 5. group aggregates on train+test (label-free)
    keyframe = pd.DataFrame({
        "project_category": out["project_category_cat"],
        "primary_language": out["primary_language_cat"],
        "framework": out["framework_cat"],
        "author_prior_projects": out["author_prior_projects"].fillna(-1),
    })
    for k in (GROUP_KEYS if USE_GROUP_AGGS else []):
        for v in GROUP_VALS:
            lv = np.log1p(g[v].clip(lower=0))
            grp = lv.groupby(keyframe[k])
            med = grp.transform("median")
            std = grp.transform("std")
            g[f"{v}_minus_med_by_{k}"] = lv - med
            g[f"{v}_z_by_{k}"] = (lv - grp.transform("mean")) / (std + 1e-3)
            g[f"{v}_rank_by_{k}"] = grp.rank(pct=True)

    g = g.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in g.columns if c not in cat_names]
    g[num_cols] = g[num_cols].astype(np.float32)
    return g.iloc[:n_tr].reset_index(drop=True), g.iloc[n_tr:].reset_index(drop=True), cat_names


def to_codes(X: pd.DataFrame, cat_names) -> pd.DataFrame:
    X = X.copy()
    for c in cat_names:
        X[c] = pd.Categorical(X[c]).codes.astype(np.float32)
    return X


# --------------------------------------------------------------------------------------
# Models — each is a (name, fit_predict(Xtr, ytr, Xte) -> pred) pair
# --------------------------------------------------------------------------------------
def make_models(cat_names, cat_vocab):
    models = []

    # LightGBM
    try:
        import lightgbm as lgb

        def fit_lgb(Xtr, ytr, Xte):
            Xtr, Xte = to_codes_fixed(Xtr, cat_names, cat_vocab), to_codes_fixed(Xte, cat_names, cat_vocab)
            m = lgb.LGBMRegressor(
                objective="regression", learning_rate=0.02, n_estimators=1400,
                num_leaves=24, min_child_samples=25, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
                random_state=SEED, n_jobs=N_THREADS, verbose=-1)
            m.fit(Xtr, ytr, categorical_feature=cat_names)
            return m.predict(Xte)
        models.append(("lgbm", fit_lgb))
    except Exception as ex:  # noqa: BLE001
        log(f"lightgbm unavailable: {ex!r}")

    # CatBoost
    try:
        from catboost import CatBoostRegressor

        def fit_cb(Xtr, ytr, Xte):
            Xtr, Xte = Xtr.copy(), Xte.copy()
            for c in cat_names:
                Xtr[c] = Xtr[c].astype(str)
                Xte[c] = Xte[c].astype(str)
            # 600 iters @ lr 0.08 ~ same quality as 1500 @ 0.04, ~2.5x faster (keeps 6 fits < 10 min)
            m = CatBoostRegressor(
                loss_function="RMSE", iterations=600, learning_rate=0.08, depth=6,
                l2_leaf_reg=5, random_strength=1.0, bootstrap_type="Bernoulli",
                subsample=0.8, random_seed=SEED, thread_count=N_THREADS, verbose=0,
                allow_writing_files=False)
            m.fit(Xtr, ytr, cat_features=cat_names)
            return m.predict(Xte)
        models.append(("catboost", fit_cb))
    except Exception as ex:  # noqa: BLE001
        log(f"catboost unavailable: {ex!r}")

    # XGBoost
    try:
        import xgboost as xgb

        def fit_xgb(Xtr, ytr, Xte):
            Xtr, Xte = to_codes_fixed(Xtr, cat_names, cat_vocab), to_codes_fixed(Xte, cat_names, cat_vocab)
            m = xgb.XGBRegressor(
                n_estimators=1200, learning_rate=0.02, max_depth=5, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=5.0,
                gamma=0.0, tree_method="hist", objective="reg:squarederror",
                random_state=SEED, n_jobs=N_THREADS, verbosity=0)
            m.fit(Xtr, ytr)
            return m.predict(Xte)
        models.append(("xgboost", fit_xgb))
    except Exception as ex:  # noqa: BLE001
        log(f"xgboost unavailable: {ex!r}")

    # sklearn — always present
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    def fit_hgb(Xtr, ytr, Xte):
        Xtr, Xte = to_codes_fixed(Xtr, cat_names, cat_vocab), to_codes_fixed(Xte, cat_names, cat_vocab)
        cat_mask = [c in cat_names for c in Xtr.columns]
        m = HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=900, max_leaf_nodes=20, min_samples_leaf=25,
            l2_regularization=3.0, max_features=0.5, categorical_features=cat_mask,
            random_state=SEED)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    models.append(("hgb", fit_hgb))

    def fit_et(Xtr, ytr, Xte):
        Xtr, Xte = to_codes_fixed(Xtr, cat_names, cat_vocab), to_codes_fixed(Xte, cat_names, cat_vocab)
        med = Xtr.median()
        m = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.35,
                                n_jobs=N_THREADS, random_state=SEED)
        m.fit(Xtr.fillna(med).fillna(0), ytr)
        return m.predict(Xte.fillna(med).fillna(0))
    models.append(("extratrees", fit_et))

    def fit_ridge(Xtr, ytr, Xte):
        num = [c for c in Xtr.columns if c not in cat_names]
        A = pd.get_dummies(Xtr[cat_names].astype(str), dtype=np.float32)
        B = pd.get_dummies(Xte[cat_names].astype(str), dtype=np.float32).reindex(columns=A.columns, fill_value=0)
        med = Xtr[num].median()
        xa = np.hstack([Xtr[num].fillna(med).fillna(0).values, Xtr[num].isna().values * 1.0])
        xb = np.hstack([Xte[num].fillna(med).fillna(0).values, Xte[num].isna().values * 1.0])
        qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED)
        xa = np.hstack([qt.fit_transform(xa), A.values])
        xb = np.hstack([qt.transform(xb), B.values])
        m = Ridge(alpha=30.0)
        m.fit(xa, ytr)
        return m.predict(xb)
    models.append(("ridge", fit_ridge))
    return models


def to_codes_fixed(X: pd.DataFrame, cat_names, cat_vocab) -> pd.DataFrame:
    X = X.copy()
    for c in cat_names:
        X[c] = pd.Categorical(X[c], categories=cat_vocab[c]).codes.astype(np.float32)
    return X


# --------------------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------------------
def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimising RMSE of P @ w vs y."""
    try:
        from scipy.optimize import minimize
        k = P.shape[1]
        f = lambda w: np.mean((P @ w - y) ** 2)
        res = minimize(f, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        w = np.clip(res.x, 0, None)
        return w / w.sum() if w.sum() > 0 else np.full(k, 1 / k)
    except Exception:  # noqa: BLE001
        return np.full(P.shape[1], 1 / P.shape[1])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def stratified_folds(train: pd.DataFrame, y: np.ndarray, n_splits: int):
    from sklearn.model_selection import StratifiedKFold
    strat = train["hours_bucket"].astype(str).values if "hours_bucket" in train else \
        pd.qcut(y, 10, labels=False, duplicates="drop")
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(np.zeros(len(y)), strat))


# --------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(HERE, "train.csv"))
    test = pd.read_csv(os.path.join(HERE, "test.csv"))
    log(f"train {train.shape}, test {test.shape}, threads={N_THREADS}")

    train = train[train["hours_spent"].notna()].reset_index(drop=True)
    y = np.log1p(train["hours_spent"].clip(Y_MIN, Y_MAX).values.astype(float))
    fallback = float(np.expm1(np.mean(y)))

    try:
        Xtr, Xte, cat_names = build_features(train, test)
        cat_vocab = {c: sorted(pd.concat([Xtr[c], Xte[c]]).astype(str).unique()) for c in cat_names}
        log(f"features: {Xtr.shape[1]}")
        models = make_models(cat_names, cat_vocab)
        folds = stratified_folds(train, y, N_FOLDS)

        oof, test_preds, names = [], [], []
        for name, fn in models:
            ts = time.time()
            try:
                o = np.zeros(len(y))
                if time.time() - t0 < TIME_BUDGET_S:
                    for tr_idx, va_idx in folds:
                        o[va_idx] = fn(Xtr.iloc[tr_idx], y[tr_idx], Xtr.iloc[va_idx])
                else:
                    o[:] = np.nan  # out of time: no OOF, still fit full model
                p = fn(Xtr, y, Xte)
                if not np.all(np.isfinite(p)):
                    raise ValueError("non-finite predictions")
                oof.append(o); test_preds.append(p); names.append(name)
                msg = f"OOF RMSLE {rmse(o, y):.4f}" if np.all(np.isfinite(o)) else "no OOF"
                log(f"{name:10s} {msg}  ({time.time() - ts:.1f}s)")
            except Exception as ex:  # noqa: BLE001
                log(f"{name} FAILED: {ex!r} — skipped")

        if not names:
            raise RuntimeError("no model succeeded")
        O, T = np.column_stack(oof), np.column_stack(test_preds)
        ok = np.all(np.isfinite(O), axis=0)
        w = np.zeros(len(names))
        if ok.any():
            w[ok] = nnls_weights(O[:, ok], y)
            log(f"blend OOF RMSLE {rmse(O[:, ok] @ w[ok], y):.4f}  weights "
                + ", ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))
        else:
            w[:] = 1 / len(names)
        pred_log = T @ w
        pred = np.clip(np.expm1(pred_log), Y_MIN, Y_MAX)
    except Exception as ex:  # noqa: BLE001  — never leave the organisers without a file
        log(f"PIPELINE FAILED ({ex!r}); writing constant fallback")
        pred = np.full(len(test), fallback)

    pred = np.where(np.isfinite(pred), pred, fallback)
    pd.DataFrame({"project_id": test["project_id"].astype(str),
                  "hours_spent": np.round(pred, 3)}).to_csv(os.path.join(HERE, "predictions.csv"), index=False)
    log(f"wrote predictions.csv ({len(test)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()