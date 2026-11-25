# CLAUDE.md: Hack Club "Hour Detective" competition briefing

Read this first. It has the full competition rules, data description, the current state of our solution,
what we have already tested, and what to do next. Put it in the repo root; Claude Code reads it automatically.

## 1. The competition

Hack Club Haggle, Week 1: "Hour Detective". Some projects claim far more hours than anyone worked. The goal is
to predict how long a project really took from its code and metadata, so inflated claims can be spotted.

- **Task:** predict `hours_spent` (hours a reviewer approved) for 500 hidden Hack Club projects.
- **Data given:** `train.csv`, 4,500 real projects (git history, code, art, README, live site, author profile).
- **Submission:** a GitHub repo with `predict.py` at the top level. Organisers run it on `test.csv` (500 hidden rows).
- **Metric:** RMSLE, lower is better.
  ```python
  from sklearn.metrics import root_mean_squared_log_error
  root_mean_squared_log_error(actual, predicted)
  ```
  This is RMSE of `log1p(hours)`. A constant guess scores about 0.97 (we measured 0.9623). Being 2x too high costs the
  same on a 10 h project as on a 100 h one.
- **Leaderboard (as of Nov 25):** 1st craisin 0.7328, 2nd playnmaker123 0.7403, 3rd Henry 0.7452, 4th Khaled 0.7455,
  **5th Batyrzhan (us) 0.7458**. The top 5 are within 0.013. With 500 rows the score noise is about ±0.02, so
  these gaps are not meaningful.

## 2. Run environment (hard limits)

Organisers put their own `train.csv` and `test.csv` next to `predict.py` and run `python predict.py`. It must write
`predictions.csv`:

```
project_id,hours_spent
3f9a1c0b2e,12.5
```

One row per row of `test.csv`. It runs twice: first on 20 public projects (shown on the ship page), then on the 500
hidden ones (the score).

- No internet. **10 minutes** and **2 GB RAM** per run. **CPU only.**
- Python 3.13 with pandas 3.1.1, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, NumPy 2.5.5.
- Anything else goes in `requirements.txt` (installed with internet first, 5 minutes max). Models saved to disk must
  be saved with the same scikit-learn version, or pin it in `requirements.txt`.
- `test.csv` has the same columns as `train.csv` minus the three label columns (`hours_spent`, `log_hours`,
  `hours_bucket`). Never use those as features.
- Our own `train.csv` must **not** be committed (organisers replace it). `.gitignore` has `train.csv`, `test.csv`,
  `predictions.csv`, `__pycache__/`, `.ipynb_checkpoints/`. `train.csv` was committed earlier and must be untracked
  with `git rm --cached train.csv` and `git rm -r --cached __pycache__`.

## 3. Rules

- Allowed: any Python library, AI help while building, sharing code publicly.
- **Not allowed:** other datasets, scraping, lookups. Only `train.csv`. Do not try to identify the people behind rows
  (no names, links or dates are given).
- Fraud: "You're here to catch it, not do it."

## 4. Data description

One row per project. Empty cell = could not be measured (no README, dead link, no git history). Big counts are
rounded. Stars, forks, internal pages and hours are capped. Some categories come from an AI reading the project
(`project_category`, `is_multiplayer`, `is_clone_or_tutorial`, sometimes `has_custom_art`, `mentions_ai_api`,
`readme_has_install_instructions`).

**Labels (train only):** `hours_spent` (capped at 140.7), `log_hours`, `hours_bucket` (`<10`, `10-30`, `30-100`, `100+`).

**Git history** (default branch only, committer's local time): `project_id`, `num_commits`, `active_commit_days`,
`project_span_days`, `commit_cadence` (frequent/daily/sporadic), `pct_commits_weekend`, `pct_commits_late_night`
(22:00 to 05:59), `lines_added_total`, `lines_deleted_total` (no lockfiles/build output), `churn_ratio`,
`median_lines_per_commit`, `max_lines_in_one_commit`, `pct_code_in_first_commit` (near 1 = uploaded at once),
`num_commit_authors`, `avg_commit_msg_length`, `pct_generic_commit_msgs`, `num_branches`, `num_merge_commits`.

**Code:** `lines_of_code`, `num_source_files`, `num_directories`, `max_dir_depth`, `num_languages`, `primary_language`,
`pct_primary_language`, `num_dependencies`, `framework`, `uses_game_engine`, `has_tests`, `num_test_files`,
`has_ci_config`, `has_dockerfile`, `has_backend`, `has_database`, `avg_file_length`, `comment_line_ratio`.

**Art and sound:** `num_image_files`, `num_audio_files`, `num_3d_model_files`, `num_font_files`, `asset_size_mb`,
`has_custom_art`.

**README:** `readme_word_count` (0 if none), `readme_num_headings`, `readme_num_images`, `readme_num_code_blocks`,
`readme_has_install_instructions`, `has_license`, `has_docs_folder`.

**Live site** (fetched without running JavaScript, so app-style sites look smaller than they are): `url_is_live`,
`hosting_type`, `page_weight_kb`, `num_js_files_loaded`, `num_internal_pages` (up to 21), `is_mobile_responsive`,
`has_custom_domain`, `load_time_ms` (noisy).

**The project:** `description_word_count`, `project_category` (game/website/tool/bot/app/other), `is_multiplayer`,
`mentions_ai_api`, `is_clone_or_tutorial`, `stars` (up to 50), `forks` (up to 50), `num_issues`,
`num_pull_requests`, `num_releases`, `is_forked_from_template`.

**The author:** `author_prior_projects` (0, 1-2, 3-5, 6+), `author_account_age_years` (<1, 1-2, 3-5, 6+).

### Facts we verified on train.csv

- `log_hours = ln(hours)` **without +1**, from unrounded hours (max gap to `ln(hours_spent)` is 1.25 on tiny
  projects). The metric uses `log1p`, so **train on `log1p(hours_spent)`, never on `log_hours`.**
- 4,500 rows; 24 rows sit at the 140.7 cap. Buckets: `<10` 2985, `10-30` 1120, `30-100` 332, `100+` 63.
  Median 6 h, mean 12.1 h.
- Missing values carry signal: no live site (`page_weight_kb` NaN) has mean log1p(y) 2.33 vs 1.91 when present;
  missing git/code info has 1.64 vs 2.06.
- Values like `has_custom_art`, `is_mobile_responsive` etc. are mixed bool/NaN objects; `primary_language` and
  `framework` contain `none`/`other` strings.

## 5. Our solution (repo layout)

| File | Purpose |
|---|---|
| `predict.py` | The submission. Trains everything at run time (no saved models), writes `predictions.csv`. Also imported by the notebook. |
| `solution.ipynb` / `solution.py` | Local EDA, CV, importance, distillation test, blending. `solution.py` is the same notebook as `# %%` cells. |
| `requirements.txt` | Essentially empty: server libraries are preinstalled. |
| `requirements-dev.txt` | Notebook deps (optuna, matplotlib, seaborn, jupyter, and the pinned libs). |
| `README.md`, `REPORT.md`, `report/*.png` | Docs and charts. |

### How `predict.py` works

1. Target `y = log1p(hours_spent)`, RMSE loss everywhere. Output `expm1`, clipped to `[0.1, 140.7]`.
2. Features (`build_features`, about 153 columns): raw numerics; booleans to 0/1; ordinal maps for
   `author_prior_projects`, `author_account_age_years`, `commit_cadence`; categorical codes for `primary_language`,
   `framework`, `hosting_type`, `project_category` (rare levels merged); engineered ratios (commits per active day,
   LOC per commit, non-first-commit lines, non-generic commits, engineering maturity score, asset totals, README
   richness, template/clone flags, missing-value flags, `row_null_count`); `log1p` copies of heavy-tailed counts.
   Group aggregates exist but are off (`USE_GROUP_AGGS = False`, no gain).
3. Models: LightGBM, CatBoost (600 it, lr 0.08, depth 6), XGBoost, HistGradientBoosting, ExtraTrees, Ridge. Any model
   that fails to import or fit is skipped.
4. Blend: non-negative weights (SLSQP) fitted on 5-fold OOF predictions computed inside `predict.py`.
5. Safety: if the whole pipeline fails, a constant prediction is still written. `TIME_BUDGET_S = 420` skips OOF work
   for later models if time runs short (the full fit still happens).
6. Optional soft-label hook: `DISTILL_ALPHA = 0.0` (off). If `soft_labels.csv` (`project_id,soft_log1p`) sits next
   to `predict.py` and alpha > 0, models train on `alpha*soft + (1-alpha)*log1p(hours)`. With it off, output is
   byte-identical to the previous version (verified).

### Results so far (OOF RMSLE, 10-fold StratifiedKFold on `hours_bucket`)

| Model | OOF |
|---|---|
| Constant | 0.9623 |
| Ridge | 0.6898 |
| HistGradientBoosting | 0.6643 |
| ExtraTrees | 0.6640 |
| LightGBM | 0.6626 |
| CatBoost | 0.6615 |
| XGBoost | 0.6611 |
| **Blend** | **0.6526** (fold std about 0.02) |

Blend weights: ExtraTrees 0.34, LightGBM 0.21, HGB 0.18, CatBoost 0.15, Ridge 0.12, XGBoost 0.00.
On a simulated 500-row holdout (sklearn models only) we got about 0.68.
**Leaderboard: 0.7458.** CV is about 0.65 but the hidden score is about 0.75, so the hidden set looks harder than
`train.csv`. Everyone on the leaderboard shows a similar gap.

## 6. What we tried

- Group aggregates over train+test by category/language/framework/author: no gain, disabled.
- Stacking with OOF `hours_bucket` probabilities: worse (0.680 vs 0.665).
- Post-processing (shift, stretch, lower clip): changes under 0.001, noise. Only the clip is kept.
- **1st-place repo (`craisined/hour-detective`) analysis.** Their trick is distillation: a TabPFN regressor produces
  5-fold OOF predictions, then CatBoost/XGBoost train on `0.93*tabpfn + 0.07*true`. Their `predict.py` loads 5
  pre-trained CatBoost JSON models and averages them. Their validation **leaks**: teacher and students use the same
  folds and seed, so validation labels reach the training targets, which is why Optuna pushed the weight to 0.93.
  With a nested (leak-free) test using an HGB+ExtraTrees teacher and HGB student:
  alpha 0.0 gives 0.6718, alpha 0.3 gives 0.6641, **alpha 0.6 gives 0.6632**, alpha 0.9 gives 0.6695.
  The leaky version showed 0.6234 at alpha 0.9, which is fake. Honest gain is about 0.009 for one model; our blend
  already gets most of it. A stronger teacher (TabPFN) is the only thing that could beat the current blend.
- GPU: not useful. The data is tiny, and the server is CPU-only anyway.

## 7. Known issues and open items

1. **DONE (Nov 25).** Full run verified on Python 3.13 + server library versions (venv): 4 threads ~195 s,
   1 thread ~450 s, peak RAM ~420 MB, holdout RMSLE 0.693. CatBoost was 290 of 410 s because of categorical
   CTRs; switched to `one_hot_max_size=255` (10-fold OOF 0.6652 vs 0.6674, 4x faster). Model order is now
   ridge, hgb, extratrees, lgbm, catboost, xgboost; whole models are skipped past `TIME_BUDGET_S` or when the
   estimated finish passes `HARD_LIMIT_S=540`; a constant predictions.csv is written first. `HD_THREADS` env
   overrides the thread count for local tests. The `DISTILL_ALPHA` hook described in section 5 is NOT in
   predict.py. Old note: **Full run not yet verified with the boosting libraries.** The sandbox here only had sklearn, so LightGBM,
   CatBoost and XGBoost ran only inside the user's notebook, never through a full `predict.py` run. Do a full timed
   run: `time python predict.py` in a folder with `train.csv` and a label-free `test.csv`. Expected about 5 minutes;
   anything above 6 to 7 minutes risks the 10-minute limit. If slow, cut CatBoost `iterations` to about 400.
2. **Local Python is 3.14**, the server is 3.13. Test on 3.13 (`py -3.13`) if wheels are missing.
3. Make a simulated split to test locally (`train.csv` has labels; split 4,000/500, drop the three label columns from
   the 500 for `test.csv`, score `predictions.csv` against the held-out `hours_spent`).
4. DONE: README fixed (153 features, converted from UTF-16 to UTF-8, dead REPORT.md links removed).
5. DONE: `train.csv`/`__pycache__` untracked earlier; `sim/answers.csv` (labels) and `sim/predict.py` untracked too.
6. Optional experiment: TabPFN teacher on the RTX 4050 (`pip install tabpfn`; set `TEACHER = "tabpfn"` in notebook
   section 6). Adopt it only if the nested CV beats alpha 0 by more than about 0.01, then write `soft_labels.csv`,
   commit it, and set `DISTILL_ALPHA` in `predict.py`. TabPFN itself must never run on the server.
7. Other ideas not yet tried: several seeds per model, tuned LightGBM/XGBoost parameters (Optuna, accept only gains
   above fold noise), a small MLP for diversity, dropping the zero-weight XGBoost to save time.

## 8. Working rules for Claude Code

- Never leak labels: nothing derived from `hours_spent`, `log_hours` or `hours_bucket` may be a feature. In CV, any
  target-derived feature (teacher predictions, target encodings) must be computed inside the training fold only.
- Judge changes by OOF over 10 folds and by fold std. Differences below about 0.003 are noise; do not tune on the
  leaderboard (500 rows, about ±0.02 noise; the public run is only 20 rows).
- Keep `predict.py` on the preinstalled libraries, CPU only, no network, no extra files besides `train.csv` and
  `test.csv` (an optional `soft_labels.csv` is safe because rows are joined by `project_id`).
- Always keep the constant-prediction fallback so `predictions.csv` is written even if something fails.
- After any change to `predict.py`: run it on a simulated split, check 500 rows and the columns
  `project_id,hours_spent`, and time it.