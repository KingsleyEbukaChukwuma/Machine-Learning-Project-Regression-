# Apartment Price Regression Project

## Project Goal

The goal of this project is to build a robust regression pipeline that explains apartment prices and generates reliable forecasts on a hold-out test set.  

We work with **156,454 training rows** and **39,114 test rows**. Each record contains:

- **Basic attributes**: size (`dim_m2`), number of rooms (`n_rooms`), floor number, total floors, and year built.
- **Geographic distances**: distance to city center (`dist_centre`) and to amenities (schools, clinics, restaurants, universities, etc.).
- **High-cardinality identifiers**: anonymized location code (`loc_code`), object type (`obj_type`), building material (`build_mat`).
- **Simulated indices**: market volatility, infrastructure quality, crime rate, popularity index, green space ratio, maintenance cost, and a global economic index.
- **Booleans**: whether the apartment has parking, a balcony, an elevator, security features, and storage.

Several columns have substantial missing values — notably `cond_class`, `build_mat`, and `has_lift`. Addressing these gaps is crucial before fitting any model.

---

## Approach

###  Exploratory Data Analysis (EDA)

- Quantified missingness and visualized patterns for key columns.
- Examined numeric distributions to detect skewness and outliers.
- Plotted residuals and errors by apartment size bins and by month (`src_month`) to uncover heteroscedasticity and periods of volatile pricing.

###  Feature Engineering

- Created boolean “_missing” flags and filled numeric gaps with medians.
- Converted string booleans (`"True"`, `"False"`) into proper Python boolean types.
- Added derived features: building `age`, relative floor (`rel_floor`), average room size (`room_size_avg`), and log-transformed distances (`log_dist_*`).
- Used frequency encoding for high-cardinality columns (`loc_code_freq`, `obj_type_freq`, `build_mat_freq`).
- Calculated a 3-month rolling median of price per `loc_code` (shifted by one month) to capture local trends.
- Added simple seasonal flags (`is_december`, `is_q1`).

###  Modeling and Hyperparameter Tuning

- Started with ElasticNet baseline to confirm linear relationships.
- Explored SGDRegressor and stubbed KNN/SVR, but these were impractical at full scale.
- Selected **HistGradientBoostingRegressor (HGB)** as the final model.
- Used `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=3)` for time-aware tuning. Sampled 30,000 rows to manage runtime.
- Best configuration: `learning_rate ≈ 0.0787`, `max_iter = 200`, `max_leaf_nodes = 15`, `min_samples_leaf = 10`. CV RMSE ≈ 97,413.

### Evaluation and Prediction

- In-sample R²: 0.9544.
- Hold-out validation (5,000 rows): RMSE ≈ 93,061 (naive mean predictor RMSE ≈ 429,000).
- Residual analysis revealed larger errors for large apartments (> 100 m²) and certain months (Dec 2023, Mar 2024, Apr 2024).
- Permutation importance confirmed top features: `market_volatility`, `loc_code_freq`, and `dim_m2`.

---


## Key Metrics

- **In-sample R²**: 0.9544
- **Time-split OOF R²**: 0.3951
- **Hold-out RMSE (5,000 rows)**: 93,061
- **Test-set predictions**: Generated for all 39,114 apartments.

---


##  How to Run

1. Clone this repository.
2. Install requirements:
