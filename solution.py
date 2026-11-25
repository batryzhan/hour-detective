# %% [markdown]
# # Hack Club "hour detectives" — EDA, валидация, тюнинг, блендинг
#
# Ноутбук для локальной работы (Ryzen 7 / RTX 4050). Признаки и модели **импортируются из
# `predict.py`**, поэтому всё, что мы проверяем здесь, ровно то же самое, что запустится на
# сервере. Положите `train.csv` рядом с ноутбуком (или поменяйте `DATA_DIR`).
#
# План:
# 1. Связь `hours_spent` ↔ `log_hours` и точная формула метрики
# 2. EDA: распределение таргета, пропуски, сильнейшие признаки, категории
# 3. Признаки + абляция (что реально помогает)
# 4. 10-fold StratifiedKFold по `hours_bucket`, OOF всех моделей
# 5. Feature importance
# 6. (опционально) Optuna для LightGBM
# 7. Блендинг (неотрицательные веса) и пост-процессинг
# 8. Анализ ошибок

# %%
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_log_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_DIR = "."                      # где лежит train.csv
sys.path.insert(0, os.path.abspath("."))  # predict.py лежит рядом
import predict as P

N_FOLDS = int(os.environ.get("N_FOLDS", 10))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(train.shape)
for lib in ["lightgbm", "catboost", "xgboost", "optuna"]:
    try:
        print(lib, __import__(lib).__version__)
    except ImportError:
        print(lib, "— не установлен")

# %% [markdown]
# ## 1. Как связаны `hours_spent` и `log_hours`, и что считает метрика
#
# `sklearn.metrics.root_mean_squared_log_error` = `sqrt(mean((log1p(y) - log1p(ŷ))²))`.
# Проверяем, что такое `log_hours`.

# %%
h, lh = train["hours_spent"], train["log_hours"]
print("max |ln(h)    - log_hours| =", (np.log(h) - lh).abs().max().round(4))
print("max |log1p(h) - log_hours| =", (np.log1p(h) - lh).abs().max().round(4))
print("max |exp(log_hours) - h|   =", (np.exp(lh) - h).abs().max().round(4))
print(train.loc[(np.log(h) - lh).abs() > 0.05, ["hours_spent", "log_hours"]].head())

# %% [markdown]
# **Вывод.** `log_hours = ln(hours)` **без +1**, и посчитан от неокруглённых часов
# (`hours_spent` округлён до 0.1, поэтому на 0.1–0.8 ч расхождения заметны).
# Метрика же работает с `log1p`. Поэтому:
# * учим все модели на **`y = log1p(hours_spent)`** с RMSE-лоссом — тогда RMSE в этом
#   пространстве *в точности* равен RMSLE;
# * на инференсе `expm1` и клип в `[0.1, 140.7]` (реальный диапазон меток).
#
# `log_hours` как таргет использовать не нужно: он смещён относительно метрики, особенно
# на маленьких проектах.

# %%
y = np.log1p(train["hours_spent"].values)
const = np.full_like(y, y.mean())
print("Константа (mean log1p):", round(P.rmse(const, y), 4), " — в условии «about 0.97»")
print("Бакеты:", train["hours_bucket"].value_counts().to_dict())
print("Строк на потолке 140.7:", int((train.hours_spent >= 140.7).sum()))

# %% [markdown]
# ## 2. EDA

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(train.hours_spent, bins=80); ax[0].set_title("hours_spent")
ax[1].hist(y, bins=60); ax[1].set_title("log1p(hours_spent)")
sns.boxplot(data=train, x="project_category", y=np.log1p(train.hours_spent), ax=ax[2])
ax[2].set_title("log1p(hours) по категориям"); plt.tight_layout(); plt.show()

# %%
na = train.isna().mean().sort_values(ascending=False)
na = na[na > 0]
print(na.round(3).to_string())
# Несёт ли пропуск сигнал?
for c in ["page_weight_kb", "num_js_files_loaded", "pct_code_in_first_commit", "pct_primary_language"]:
    m = train[c].isna()
    print(f"{c:28s} missing: mean log1p y = {y[m].mean():.3f}  vs present {y[~m].mean():.3f}  (n={m.sum()})")

# %%
Xtr, _, cats = P.build_features(train, train.drop(columns=P.LABELS).iloc[:0])
num = [c for c in Xtr.columns if c not in cats]
corr = Xtr[num].corrwith(pd.Series(y), method="spearman").dropna()
top = corr.abs().sort_values(ascending=False).head(30).index
plt.figure(figsize=(8, 9))
corr[top].sort_values().plot.barh()
plt.title("Spearman с log1p(hours): топ-30 признаков"); plt.tight_layout(); plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, c in zip(axes.ravel(), ["log_active_commit_days", "log_num_commits", "log_lines_of_code",
                               "log_non_first_commit_lines", "pct_code_in_first_commit", "log_project_span_days"]):
    a.scatter(Xtr[c], y, s=3, alpha=0.3); a.set_xlabel(c); a.set_ylabel("log1p(h)")
plt.tight_layout(); plt.show()

# %%
for c in ["author_prior_projects", "commit_cadence", "framework", "hosting_type", "is_clone_or_tutorial"]:
    print(train.assign(ly=y).groupby(c, dropna=False)["ly"].agg(["mean", "count"]).sort_values("count", ascending=False).head(10).round(3), "\n")

# %% [markdown]
# ## 3. Валидация и абляция признаков
#
# 10-fold `StratifiedKFold` по `hours_bucket`. Для быстрой абляции используем HistGB (есть
# везде). Сравниваем: только сырые колонки / + инженерные / + групповые агрегаты.

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor

folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=P.SEED)
             .split(train, train["hours_bucket"]))
vocab = {c: sorted(Xtr[c].astype(str).unique()) for c in cats}

def cv_hgb(X, seed=0):
    Xc = P.to_codes_fixed(X, [c for c in cats if c in X], vocab)
    oof = np.zeros(len(y))
    for a, b in folds:
        m = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=900, max_leaf_nodes=20,
                                          min_samples_leaf=25, l2_regularization=3, max_features=0.5,
                                          categorical_features=[c in cats for c in Xc.columns], random_state=seed)
        m.fit(Xc.iloc[a], y[a]); oof[b] = m.predict(Xc.iloc[b])
    return P.rmse(oof, y)

raw = [c for c in train.columns if c not in P.LABELS + ["project_id"]]
raw_feats = [c for c in Xtr.columns if c in raw or c.replace("_cat", "") in raw]
P.USE_GROUP_AGGS = True
Xg, _, _ = P.build_features(train, train.drop(columns=P.LABELS).iloc[:0])
P.USE_GROUP_AGGS = False
ablation = {
    "raw only": cv_hgb(Xtr[raw_feats]),
    "raw + engineered (prod)": cv_hgb(Xtr),
    "+ group aggregates": cv_hgb(Xg),
}
print(pd.Series(ablation).round(4))

# %% [markdown]
# В нашем прогоне (5 фолдов) разница была ~0.002 — в пределах шума: деревья сами находят
# отношения вроде `loc / commits`. Групповые агрегаты по train+test не помогли, поэтому в
# `predict.py` они выключены (`USE_GROUP_AGGS = False`). Инженерные признаки оставляем: они
# заметно помогают Ridge и ExtraTrees и добавляют разнообразия ансамблю.
#
# **Стекинг с OOF-вероятностями `hours_bucket`** мы тоже проверили: стало хуже (0.690 против 0.669).
# Бакеты — это просто огрублённый таргет, новой информации классификатор не несёт.
# Проверка ниже.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
Xc = P.to_codes_fixed(Xtr, cats, vocab)
yb = train.hours_bucket.map({b: i for i, b in enumerate(P.BUCKETS)}).values
proba = np.zeros((len(y), 4))
for a, b in folds:
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=20,
                                         categorical_features=[c in cats for c in Xc.columns], random_state=0)
    proba[b] = clf.fit(Xc.iloc[a], yb[a]).predict_proba(Xc.iloc[b])
Xs = Xtr.copy(); Xs[[f"pb{i}" for i in range(4)]] = proba
print("с вероятностями бакетов:", round(cv_hgb(Xs), 4))

# %% [markdown]
# ## 4. OOF всех моделей ансамбля (точно те же, что в predict.py)

# %%
models = P.make_models(cats, vocab)
oof = {}
for name, fn in models:
    t = time.time(); o = np.zeros(len(y))
    for a, b in folds:
        o[b] = fn(Xtr.iloc[a], y[a], Xtr.iloc[b])
    oof[name] = o
    print(f"{name:10s} OOF RMSLE {P.rmse(o, y):.4f}  ({time.time()-t:.0f}s)")
OOF = pd.DataFrame(oof)
print(OOF.corr().round(3))

# %% [markdown]
# ## 5. Feature importance

# %%
try:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(learning_rate=0.02, n_estimators=1400, num_leaves=24, min_child_samples=25,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_alpha=0.5,
                          reg_lambda=3, importance_type="gain", verbose=-1, random_state=0)
    m.fit(Xc, y, categorical_feature=cats)
    imp = pd.Series(m.feature_importances_, Xc.columns)
    title = "LightGBM gain"
except ImportError:
    from sklearn.inspection import permutation_importance
    a, b = folds[0]
    m = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=600, max_leaf_nodes=20,
                                      min_samples_leaf=25, categorical_features=[c in cats for c in Xc.columns],
                                      random_state=0).fit(Xc.iloc[a], y[a])
    r = permutation_importance(m, Xc.iloc[b], y[b], n_repeats=3, random_state=0,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = pd.Series(r.importances_mean, Xc.columns)
    title = "Permutation importance (HistGB, fold 0)"
plt.figure(figsize=(8, 10)); imp.sort_values().tail(35).plot.barh(); plt.title(title); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. (опционально) Optuna для LightGBM
#
# `pip install optuna`. ~50–100 trials занимают 10–30 минут на Ryzen 7. Лучшие параметры
# вручную перенесите в `fit_lgb` в `predict.py` и перепроверьте OOF. Не перетюнивайте:
# разница < 0.003 на 4.5k строках — это шум, смотрите на std по фолдам.

# %%
RUN_OPTUNA = False
if RUN_OPTUNA:
    import optuna, lightgbm as lgb

    def objective(trial):
        params = dict(
            objective="regression", n_estimators=3000, learning_rate=0.02, verbose=-1, random_state=0,
            num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0, 0.2))
        o = np.zeros(len(y)); iters = []
        for a, b in folds[:5]:
            m = lgb.LGBMRegressor(**params)
            m.fit(Xc.iloc[a], y[a], eval_set=[(Xc.iloc[b], y[b])], categorical_feature=cats,
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            o[b] = m.predict(Xc.iloc[b]); iters.append(m.best_iteration_)
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        idx = np.concatenate([b for _, b in folds[:5]])
        return P.rmse(o[idx], y[idx])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=80, show_progress_bar=True)
    print(study.best_value, study.best_params, study.best_trial.user_attrs)

# %% [markdown]
# ## 7. Блендинг и пост-процессинг

# %%
names = list(OOF.columns)
w = P.nnls_weights(OOF.values, y)
blend = OOF.values @ w
print("weights:", dict(zip(names, w.round(3))))
print("blend OOF RMSLE:", round(P.rmse(blend, y), 4))
per_fold = [P.rmse(blend[b], y[b]) for _, b in folds]
print("по фолдам: mean %.4f  std %.4f" % (np.mean(per_fold), np.std(per_fold)))

# Альтернатива — Ridge-стекинг с положительными коэффициентами и свободным intercept
from sklearn.linear_model import LinearRegression
st = np.zeros(len(y))
for a, b in folds:
    st[b] = LinearRegression(positive=True).fit(OOF.values[a], y[a]).predict(OOF.values[b])
print("positive-LR stack OOF:", round(P.rmse(st, y), 4))

# %%
# Пост-процессинг: сдвиг, растяжение вокруг среднего, клип (всё в log1p-пространстве)
def score(pl):
    return root_mean_squared_log_error(train.hours_spent, np.clip(np.expm1(pl), 0.1, 140.7))
print("blend + clip:", round(score(blend), 4))
for s in [-0.05, -0.02, 0, 0.02, 0.05]:
    print(f"shift {s:+.2f}: {score(blend + s):.4f}")
mu = blend.mean()
for k in [0.95, 1.0, 1.05, 1.1, 1.15]:
    print(f"stretch {k:.2f}: {score(mu + k*(blend-mu)):.4f}")
for lo in [0.1, 0.5, 1.0]:
    print(f"clip lo {lo}: {root_mean_squared_log_error(train.hours_spent, np.clip(np.expm1(blend), lo, 140.7)):.4f}")

# %% [markdown]
# Если растяжение `k > 1` даёт стабильный выигрыш по фолдам (а не только на всём OOF), его можно
# добавить в `predict.py`. Деревья с RMSE-лоссом «поджимают» прогнозы к среднему, и небольшое
# растяжение иногда помогает. Проверяйте на фолдах, а не на одном числе.

# %% [markdown]
# ## 8. Анализ ошибок

# %%
res = blend - y
df_err = train.assign(pred=np.expm1(blend), res=res)
print(df_err.groupby("hours_bucket")["res"].agg(["mean", lambda r: np.sqrt(np.mean(r**2)), "count"])
      .rename(columns={"<lambda_0>": "rmse"}).round(3))
plt.figure(figsize=(6, 6))
plt.scatter(y, blend, s=3, alpha=0.3); plt.plot([0, 5], [0, 5], "r--")
plt.xlabel("true log1p"); plt.ylabel("pred log1p"); plt.title("OOF: прогноз vs факт"); plt.show()
print(df_err.loc[res.abs().sort_values(ascending=False).index[:10] if isinstance(res, pd.Series) else np.argsort(-np.abs(res))[:10],
             ["hours_spent", "pred", "num_commits", "active_commit_days", "lines_of_code", "project_category"]])

# %% [markdown]
# **Типичная картина:** модель переоценивает очень маленькие проекты (0.1–1 ч) и недооценивает
# 100+ — регрессия к среднему. Часть проектов с одинаковыми метаданными получила очень разные
# часы от ревьюеров: это неустранимый шум, и он задаёт потолок ~0.65 RMSLE.
#
# ## 9. Финальная проверка `predict.py`
# Симуляция боевого запуска: отложите 500 строк, удалите метки, положите `train.csv`/`test.csv`
# в отдельную папку вместе с `predict.py`, запустите `python predict.py` и посчитайте RMSLE.
