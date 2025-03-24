import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, MultiTaskLasso, MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from math import sqrt
from sklearn.impute import SimpleImputer

# Load data
# daily_metrics = pd.read_csv('../src/data/processed/daily_metrics_with_clusters_sklearn.csv', parse_dates=['date'])
# dcl_prices = pd.read_csv('../src/data/processed/gb_dcl_prices_gbp_per_mw_per_hour.csv', parse_dates=['dtutc'])
#
# # Prepare DCL prices data
# dcl_prices['date'] = dcl_prices['dtutc'].dt.date
# dcl_prices['time_block'] = dcl_prices['dtutc'].dt.hour // 4 + 1
# dcl_prices_pivot = dcl_prices.pivot_table(index='date', columns='time_block', values='dcl_price',
#                                           aggfunc='first').reset_index()
# dcl_prices_pivot.columns = ['date'] + [f'price_{i}' for i in range(1, 7)]
#
# # Merge data
# daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
# dcl_prices_pivot['date'] = pd.to_datetime(dcl_prices_pivot['date'])
#
# data = pd.merge(daily_metrics, dcl_prices_pivot, on='date', how='inner')
def analyze_feature_importance(self, model):
    """Analyze feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance
    return None


# def add_features(self, data):
#     """Add additional features to the data"""
#     # Add UK holidays flag
#     uk_holidays = holidays.UK()
#     data['is_holiday'] = data['date'].dt.date.apply(lambda x: x in uk_holidays)
#
#     # Add day of week features
#     data['day_of_week'] = data['date'].dt.dayofweek
#
#     # Add month features
#     data['month'] = data['date'].dt.month
#
#     # Add season features
#     data['season'] = data['date'].dt.month % 12 // 3 + 1
#
#     return data
#
# # After loading and merging data
# data = add_features(data)




# Define features and target
feature_cols = [
    'demand_National Demand Forecast (NDF) - GB (MW)_mean',
    'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_mean',
    'solar_solar_fc_meteo_mw_mean',
    'wind_wind_fc_meteo_mw_mean',
    'is_weekend',
    'cluster'
]
target_cols = [f'price_{i}' for i in range(1, 7)]

X = data[feature_cols]
y = data[target_cols]

# Define date ranges
train_end_date = pd.Timestamp('2024-07-31')
test_start_date = pd.Timestamp('2024-08-01')
test_end_date = pd.Timestamp('2024-10-31')

# Split data
X_train = X[data['date'] <= train_end_date]
y_train = y[data['date'] <= train_end_date]
X_test = X[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
y_test = y[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]

# Define models
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet', ElasticNet()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor()),
    ('KNN', KNeighborsRegressor()),
    ('SVR', SVR()),
    ('XGBoost', XGBRegressor()),
    ('LightGBM', LGBMRegressor()),
    ('Neural Network', MLPRegressor()),
    ('Kernel Ridge', KernelRidge())
]

# Function to evaluate a model
imputer = SimpleImputer(strategy='mean')
y_train = imputer.fit_transform(y_train)
y_test = imputer.transform(y_test)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    return rmse


# Evaluate all models
results = []
for name, model in models:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(model))
    ])
    rmse = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    results.append({
        'Model': name,
        'RMSE': rmse,
        'Total RMSE': np.sum(rmse)
    })

# Sort results by total RMSE
results_df = pd.DataFrame(results).sort_values('Total RMSE')
print("Machine Learning Models Results:")
print(results_df)

# Select the best model
best_model_name = results_df.iloc[0]['Model']
best_model = dict(models)[best_model_name]

# Create a custom param_grid based on the best model type
if isinstance(best_model, LinearRegression):
    param_grid = {}  # Linear Regression doesn't have hyperparameters to tune
elif isinstance(best_model, Ridge):
    param_grid = {'model__estimator__alpha': [0.1, 1.0, 10.0]}
elif isinstance(best_model, Lasso):
    param_grid = {'model__estimator__alpha': [0.1, 1.0, 10.0]}
elif isinstance(best_model, ElasticNet):
    param_grid = {'model__estimator__alpha': [0.1, 1.0, 10.0], 'model__estimator__l1_ratio': [0.2, 0.5, 0.8]}
elif isinstance(best_model, RandomForestRegressor):
    param_grid = {
        'model__estimator__n_estimators': [50, 100, 200],
        'model__estimator__max_depth': [None, 10, 20],
        'model__estimator__min_samples_split': [2, 5, 10]
    }
elif isinstance(best_model, GradientBoostingRegressor):
    param_grid = {
        'model__estimator__n_estimators': [50, 100, 200],
        'model__estimator__learning_rate': [0.01, 0.1, 0.2],
        'model__estimator__max_depth': [3, 5, 7]
    }
elif isinstance(best_model, KNeighborsRegressor):
    param_grid = {
        'model__estimator__n_neighbors': [3, 5, 7],
        'model__estimator__weights': ['uniform', 'distance']
    }
elif isinstance(best_model, SVR):
    param_grid = {
        'model__estimator__C': [0.1, 1, 10],
        'model__estimator__kernel': ['linear', 'rbf'],
        'model__estimator__gamma': ['scale', 'auto']
    }
elif isinstance(best_model, XGBRegressor):
    param_grid = {
        'model__estimator__n_estimators': [50, 100, 200],
        'model__estimator__learning_rate': [0.01, 0.1, 0.2],
        'model__estimator__max_depth': [3, 5, 7]
    }
elif isinstance(best_model, LGBMRegressor):
    param_grid = {
        'model__estimator__n_estimators': [50, 100, 200],
        'model__estimator__learning_rate': [0.01, 0.1],
        'model__estimator__num_leaves': [15, 31],  # Reduce number of leaves for small datasets
        'model__estimator__max_depth': [3, 5],  # Limit depth for small datasets
        'model__estimator__min_data_in_leaf': [5, 10],  # Minimum samples per leaf
        'model__estimator__feature_fraction': [0.8, 1.0]  # Fraction of features used per iteration
    }
elif isinstance(best_model, MLPRegressor):
    param_grid = {
        'model__estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__estimator__alpha': [0.0001, 0.001, 0.01],
        'model__estimator__learning_rate': ['constant', 'adaptive']
    }
elif isinstance(best_model, KernelRidge):
    param_grid = {
        'model__estimator__alpha': [0.1, 1.0, 10.0],
        'model__estimator__kernel': ['linear', 'rbf', 'poly']
    }
else:
    param_grid = {}  # Default empty grid if model type is not recognized

# Hyperparameter tuning for the best model
param_grid = {
    'model__estimator__C': [0.1, 1, 10],
    'model__estimator__kernel': ['linear', 'rbf'],
    'model__estimator__gamma': ['scale', 'auto']
}


def rolling_window_evaluation(X, y, model, window_size=60, step=1):
    n_samples = len(X)
    n_windows = max(1, (n_samples - window_size) // step)
    results = []

    for i in range(n_windows):
        train_start = i * step
        train_end = train_start + window_size
        test_idx = train_end

        X_train, y_train = X.iloc[train_start:train_end], y.iloc[train_start:train_end]
        X_test, y_test = X.iloc[test_idx:test_idx + 1], y.iloc[test_idx:test_idx + 1]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        results.append(rmse)

    return np.mean(results, axis=0)


# In the model evaluation loop
for name, model in models:
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(model) if not isinstance(model, (
            MultiTaskLasso, MultiTaskElasticNet, RegressorChain)) else model)
        ])
        rmse = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
        rolling_rmse = rolling_window_evaluation(X, y, pipeline)
        results.append({
            'Model': name,
            'RMSE': rmse,
            'Total RMSE': np.sum(rmse),
            'Rolling RMSE': rolling_rmse,
            'Total Rolling RMSE': np.sum(rolling_rmse)
        })
    except Exception as e:
        print(f"Error evaluating {name}: {str(e)}")

# Models that account for output correlations
correlation_aware_models = [
    ('MultiTaskLasso', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', MultiTaskLasso())
    ]), {'model__alpha': [0.01, 0.1, 1.0]}),

    ('MultiTaskElasticNet', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', MultiTaskElasticNet())
    ]), {'model__alpha': [0.01, 0.1, 1.0], 'model__l1_ratio': [0.2, 0.5, 0.8]}),

    ('RegressorChain (Ridge)', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RegressorChain(Ridge(), random_state=42))
    ]), {'model__base_estimator__alpha': [0.1, 1.0, 10.0]})
]
# Include correlation-aware models in the evaluation
models += correlation_aware_models
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MultiOutputRegressor(best_model))
])

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print("Best model RMSE:", rmse)
print("Total RMSE:", np.sum(rmse))

# Feature importance
if hasattr(best_model['model'].estimators_[0], 'feature_importances_'):
    importances = np.mean([est.feature_importances_ for est in best_model['model'].estimators_], axis=0)
    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    print("Feature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

# Load reference day performance results
ref_day_results = pd.read_csv('../src/data/processed/reference_day_performance_dcl.csv')
euclidean_results = pd.read_csv('../src/data/processed/euclidean_reference_day_performance_dcl.csv')

# Calculate overall metrics for reference methods
ref_day_total_rmse = ref_day_results['rmse'].sum()
euclidean_total_rmse = euclidean_results['rmse'].sum()

print("\nReference Day Methods Results:")
print(f"K-means Reference Day Total RMSE: {ref_day_total_rmse:.2f}")
print(f"Euclidean Reference Day Total RMSE: {euclidean_total_rmse:.2f}")

# Compare all methods
all_methods = pd.DataFrame({
    'Method': ['Best ML Model'] + [r['Model'] for r in results] + ['K-means Reference', 'Euclidean Reference'],

    'Total RMSE': [np.sum(rmse)] + [r['Total RMSE'] for r in results] + [ref_day_total_rmse, euclidean_total_rmse]
})
all_methods = all_methods.sort_values('Total RMSE')

print("\nComparison of All Methods:")
print(all_methods)

# Save results
all_methods.to_csv('../src/data/processed/dcl_price_forecast_results.csv', index=False)
print("\nResults saved to: ../src/data/processed/dcl_price_forecast_results.csv")
