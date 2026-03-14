# GenAI-Genesis Codebase Summary

Quick reference for the AML (Anti–Money Laundering) project.

---

## Entry points

| What | How |
|------|-----|
| **Run IBM AML pipeline** | `python run_pipeline.py` |
| **Use another CSV** | `python run_pipeline.py --file HI-Medium_Trans.csv` or `--csv "path/to/file.csv"` |
| **Download IBM data** | `python download_ibm_data.py` (needs `kagglehub` + Kaggle auth) |

---

## Main pipeline (IBM AML RF)

Flow: **run_pipeline.py** → **src.pipeline.runner** → data → features → model → metrics + predictions CSV.

1. **run_pipeline.py**  
   Adds project root to `sys.path`, imports and calls `src.pipeline.runner.main()`.

2. **src/pipeline/runner.py**  
   - Parses `--file` and `--csv`.
   - Loads transaction CSV via `src.data.ibm_loader`.
   - Builds feature matrix via `src.features.engine.build_model_matrix`.
   - Trains Random Forest via `src.model.train.train_random_forest`.
   - Evaluates and prints metrics; writes test predictions to `outputs/rf_eval_predictions_<timestamp>.csv`.

3. **src/data/ibm_loader.py**  
   - `get_dataset_path(file_name=None)`: resolves path by looking in **Data/** then **kagglehub_cache/.../versions/8**.
   - `load_transactions(file_name=None, csv_path=None)`: reads CSV; if `csv_path` is set, uses that path directly.

4. **src/features/engine.py**  
   - Expects IBM AML schema (see `src.config.IBM_REQUIRED_COLUMNS`).
   - `validate_ibm_schema(df)` then `build_model_matrix(df)` → (X, y).
   - Derives Hour, DayOfWeek, Day, Month from `Timestamp`; factorizes categoricals; returns `MODEL_FEATURE_COLUMNS` + target `Is Laundering`.

5. **src/model/train.py**  
   - `stratified_downsample(X, y)` to cap rows (config: `MAX_ROWS`).
   - `train_random_forest(X, y)`: train/test split, `RandomForestClassifier` (config: `RF_N_ESTIMATORS`, `class_weight="balanced_subsample"`), returns model + splits.
   - `evaluate_model(model, X_test, y_test)`: classification report, confusion matrix, ROC-AUC, PR-AUC.

6. **src/utils/logging.py**  
   - `setup_logging()`: file log under `outputs/logs/`, console at INFO.

---

## Config (src/config.py)

| Key | Purpose |
|-----|--------|
| `PROJECT_ROOT` | Project root (parent of `src`). |
| `DATA_DIR` | `Data/` — first place to look for CSV. |
| `DATASET_DIR` | Kaggle cache path for IBM dataset. |
| `DEFAULT_DATASET_FILE` | `"HI-Small_Trans.csv"`. |
| `OUTPUT_DIR` / `LOG_DIR` | `outputs/`, `outputs/logs/`. |
| `IBM_REQUIRED_COLUMNS` | Columns required in raw IBM CSV. |
| `MODEL_FEATURE_COLUMNS` | Features used for the RF model. |
| `RANDOM_STATE`, `MAX_ROWS`, `TEST_SIZE`, `RF_N_ESTIMATORS` | Training/sampling settings. |

---

## Other modules (not used by run_pipeline.py)

- **tests/test_scorer.py**  
  Tests for aggregation, context scoring, risk buckets. Imports `src.agents.base`, `src.aggregation.scorer`, `src.rules.aml_rules` and config keys (e.g. `SCORE_WEIGHT_AGENT`, `RULE_CATEGORIES`) that **do not exist** in the current codebase — tests are for a different/planned AML scoring stack.

- **tests/test_sparse_agent.py**  
  Present in repo; purpose not summarized here.

---

## Data layout

- **IBM pipeline** expects a single CSV with columns: `Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`.
- CSV is looked up in: **Data/** then **kagglehub_cache/.../ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8**.
- **download_ibm_data.py** downloads via kagglehub and copies `HI-Small_Trans.csv` into **Data/**.

---

## Dependencies (requirements.txt)

- pandas, scikit-learn, tqdm, kagglehub

Install in the project venv:  
`pip install -r requirements.txt`
