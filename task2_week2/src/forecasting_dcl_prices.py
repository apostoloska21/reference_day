# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import holidays


# Load data
def load_data():
    daily_metrics = pd.read_csv('../src/data/processed/daily_metrics_with_clusters_sklearn.csv', parse_dates=['date'])
    dcl_prices = pd.read_csv('../src/data/processed/gb_dcl_prices_gbp_per_mw_per_hour.csv', parse_dates=['dtutc'])

    # Prepare DCL prices data
    dcl_prices['date'] = dcl_prices['dtutc'].dt.date
    dcl_prices['time_block'] = dcl_prices['dtutc'].dt.hour // 4 + 1
    dcl_prices_pivot = dcl_prices.pivot_table(index='date', columns='time_block', values='dcl_price',
                                              aggfunc='first').reset_index()
    dcl_prices_pivot.columns = ['date'] + [f'price_{i}' for i in range(1, 7)]

    # Merge data
    data = pd.merge(daily_metrics, dcl_prices_pivot, on='date', how='inner')
    return data


# Feature engineering
def add_features(data):
    uk_holidays = holidays.UK()
    data['is_holiday'] = data['date'].dt.date.apply(lambda x: x in uk_holidays)
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['season'] = data['date'].dt.month % 12 // 3 + 1
    return data


# Define features and targets
def prepare_features_and_targets(data):
    feature_cols = [
        'demand_National Demand Forecast (NDF) - GB (MW)_mean',
        'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_mean',
        'solar_solar_fc_meteo_mw_mean',
        'wind_wind_fc_meteo_mw_mean',
        'is_weekend',
        'cluster',
        'is_holiday',
        'day_of_week',
        'month',
        'season'
    ]
    target_cols = [f'price_{i}' for i in range(1, 7)]
    X = data[feature_cols]
    y = data[target_cols]
    return X, y, feature_cols, target_cols


# Split data
def split_data(X, y, data):
    train_end_date = pd.Timestamp('2024-07-31')
    test_start_date = pd.Timestamp('2024-08-01')
    test_end_date = pd.Timestamp('2024-10-31')

    X_train = X[data['date'] <= train_end_date]
    y_train = y[data['date'] <= train_end_date]
    X_test = X[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
    y_test = y[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]

    return X_train, y_train, X_test, y_test


# Define models
def create_models():
    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge()),
        ('Lasso Regression', Lasso()),
        ('ElasticNet', ElasticNet()),
        ('Random Forest', RandomForestRegressor(random_state=42)),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
        ('KNN', KNeighborsRegressor()),
        ('SVR', SVR()),
        ('XGBoost', XGBRegressor(random_state=42)),
        ('LightGBM', LGBMRegressor(random_state=42)),
        ('Neural Network', MLPRegressor(random_state=42)),
        ('Kernel Ridge', KernelRidge())
    ]

    correlation_aware_models = [
        ('MultiTaskLasso', MultiTaskLasso(alpha=0.1, max_iter=10000)),
        ('MultiTaskElasticNet', MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ('RegressorChain (Ridge)', RegressorChain(Ridge(alpha=1.0)))
    ]

    return models + correlation_aware_models


# Evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
    return rmse


# Rolling window evaluation
def rolling_window_evaluation(X, y, model, window_size=60, step=1):
    n_samples = len(X)
    n_windows = max(1, (n_samples - window_size) // step)
    results = []
    for i in range(n_windows):
        train_start = i * step
        train_end = train_start + window_size
        X_train_win, y_train_win = X.iloc[train_start:train_end], y.iloc[train_start:train_end]
        X_test_win, y_test_win = X.iloc[train_end:train_end + 1], y.iloc[train_end:train_end + 1]
        model.fit(X_train_win, y_train_win)
        y_pred_win = model.predict(X_test_win)
        rmse_win = np.sqrt(mean_squared_error(y_test_win, y_pred_win))
        results.append(rmse_win)
    return np.mean(results)


# Evaluate all models
def evaluate_all_models(X_train, y_train, X_test, y_test, models):
    results = []
    for name, model in models:
        try:
            if not isinstance(model, (MultiTaskLasso, MultiTaskElasticNet, RegressorChain)):
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', MultiOutputRegressor(model))
                ])
            else:
                pipeline = model
            rmse = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
            rolling_rmse = rolling_window_evaluation(X_train, y_train, pipeline)
            results.append({
                'Model': name,
                'RMSE': rmse,
                'Total RMSE': np.sum(rmse),
                'Rolling RMSE': rolling_rmse
            })
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
    return results


# Feature selection
def select_features(X_train, y_train, X_test, method='mutual_info', k=10):
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        raise ValueError("Invalid feature selection method")

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


# Hyperparameter tuning
def tune_hyperparameters(model, X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


# Main execution
def main():
    # Load and prepare data
    data = load_data()
    data = add_features(data)
    X, y, feature_cols, target_cols = prepare_features_and_targets(data)
    X_train, y_train, X_test, y_test = split_data(X, y, data)

    # Feature selection
    X_train_selected, X_test_selected, selector = select_features(X_train, y_train, X_test)

    # Create and evaluate models
    models = create_models()
    results = evaluate_all_models(X_train_selected, y_train, X_test_selected, y_test, models)

    # Display results
    results_df = pd.DataFrame(results)
    print("Model Evaluation Results:")
    print(results_df.sort_values('Total RMSE'))

    # Select best model
    best_model = results_df.loc[results_df['Total RMSE'].idxmin(), 'Model']
    print(f"\nBest Model: {best_model}")

    # Hyperparameter tuning for best model
    best_model_instance = dict(models)[best_model]
    tuned_model = tune_hyperparameters(best_model_instance, X_train_selected, y_train)

    # Final evaluation
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(tuned_model))
    ])
    final_rmse = evaluate_model(final_pipeline, X_train_selected, y_train, X_test_selected, y_test)

    print("\nFinal RMSE per time block:", final_rmse)
    print("Total RMSE:", np.sum(final_rmse))

    # Visualize results
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Total RMSE', data=results_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Comparison - Total RMSE')
    plt.tight_layout()
    plt.show()

    # Save results
    results_df.to_csv('../src/data/processed/dcl_price_forecast_results.csv', index=False)


if __name__ == "__main__":
    main()

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error
# from sklearn.multioutput import MultiOutputRegressor, RegressorChain
# from sklearn.linear_model import (
#     LinearRegression, Ridge, Lasso, ElasticNet,
#     MultiTaskLasso, MultiTaskElasticNet
# )
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.impute import SimpleImputer
# import logging
# import warnings
# import os
# from datetime import datetime
#
# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# warnings.filterwarnings('ignore')
#
# def load_data():
#     """Load and prepare the data for modeling."""
#     try:
#         # Define file paths
#         daily_metrics_path = os.path.join('data', 'processed', 'daily_metrics_with_clusters_sklearn.csv')
#         dcl_prices_path = os.path.join('data', 'processed', 'gb_dcl_prices_gbp_per_mw_per_hour.csv')
#
#         # Load data
#         daily_metrics = pd.read_csv(daily_metrics_path, parse_dates=['date'])
#         dcl_prices = pd.read_csv(dcl_prices_path, parse_dates=['dtutc'])
#
#         # Prepare DCL prices data
#         dcl_prices['date'] = dcl_prices['dtutc'].dt.date
#         dcl_prices['time_block'] = dcl_prices['dtutc'].dt.hour // 4 + 1
#         dcl_prices_pivot = dcl_prices.pivot_table(
#             index='date',
#             columns='time_block',
#             values='dcl_price',
#             aggfunc='first'
#         ).reset_index()
#         dcl_prices_pivot.columns = ['date'] + [f'price_{i}' for i in range(1, 7)]
#
#         # Merge data
#         daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
#         dcl_prices_pivot['date'] = pd.to_datetime(dcl_prices_pivot['date'])
#         data = pd.merge(daily_metrics, dcl_prices_pivot, on='date', how='inner')
#
#         return data
#
#     except Exception as e:
#         logging.error(f"Error loading data: {str(e)}")
#         raise
#
# def prepare_features_and_targets(data):
#     """Prepare feature matrix and target variables."""
#     # Define features
#     feature_cols = [
#         'demand_National Demand Forecast (NDF) - GB (MW)_mean',
#         'price_Price average forecast ECMWF ENS United Kingdom day-ahead (£/MWh)_mean',
#         'solar_solar_fc_meteo_mw_mean',
#         'wind_wind_fc_meteo_mw_mean',
#         'is_weekend',
#         'cluster'
#     ]
#     target_cols = [f'price_{i}' for i in range(1, 7)]
#
#     # Add holiday flag if available
#     if 'is_holiday' in data.columns:
#         feature_cols.append('is_holiday')
#
#     X = data[feature_cols]
#     y = data[target_cols]
#
#     return X, y, feature_cols, target_cols
#
# def split_data(X, y, data):
#     """Split data into training and test sets."""
#     train_end_date = pd.Timestamp('2024-07-31')
#     test_start_date = pd.Timestamp('2024-08-01')
#     test_end_date = pd.Timestamp('2024-10-31')
#
#     X_train = X[data['date'] <= train_end_date]
#     y_train = y[data['date'] <= train_end_date]
#     X_test = X[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
#     y_test = y[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
#
#     return X_train, y_train, X_test, y_test
#
# def create_model_list():
#     """Create list of models to evaluate."""
#     models = [
#         ('Linear Regression', LinearRegression()),
#         ('Ridge Regression', Ridge()),
#         ('Lasso Regression', Lasso()),
#         ('ElasticNet', ElasticNet()),
#         ('Random Forest', RandomForestRegressor(random_state=42)),
#         ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
#         ('KNN', KNeighborsRegressor()),
#         ('XGBoost', XGBRegressor(random_state=42)),
#         ('LightGBM', LGBMRegressor(random_state=42)),
#         ('Neural Network', MLPRegressor(random_state=42)),
#         ('Kernel Ridge', KernelRidge())
#     ]
#     return models
#
# def evaluate_model(model, X_train, y_train, X_test, y_test):
#     """Evaluate a single model."""
#     try:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
#         return rmse
#     except Exception as e:
#         logging.error(f"Error evaluating model: {str(e)}")
#         return None
#
# def create_param_grid(model):
#     """Create parameter grid for model tuning."""
#     if isinstance(model, LinearRegression):
#         return {}
#     elif isinstance(model, Ridge):
#         return {'model__estimator__alpha': [0.1, 1.0, 10.0]}
#     elif isinstance(model, RandomForestRegressor):
#         return {
#             'model__estimator__n_estimators': [50, 100, 200],
#             'model__estimator__max_depth': [None, 10, 20],
#             'model__estimator__min_samples_split': [2, 5, 10]
#         }
#     # Add more parameter grids for other models as needed
#     return {}
#
# def analyze_feature_importance(model, feature_cols):
#     """Analyze feature importance for the model."""
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#         feature_importance = pd.DataFrame({
#             'feature': feature_cols,
#             'importance': importances
#         }).sort_values('importance', ascending=False)
#         return feature_importance
#     return None
#
# def main():
#     """Main execution function."""
#     try:
#         # Load data
#         logging.info("Loading data...")
#         data = load_data()
#
#         # Prepare features and targets
#         logging.info("Preparing features and targets...")
#         X, y, feature_cols, target_cols = prepare_features_and_targets(data)
#
#         # Split data
#         logging.info("Splitting data...")
#         X_train, y_train, X_test, y_test = split_data(X, y, data)
#
#         # Handle missing values
#         imputer = SimpleImputer(strategy='mean')
#         y_train = imputer.fit_transform(y_train)
#         y_test = imputer.transform(y_test)
#
#         # Create and evaluate models
#         logging.info("Evaluating models...")
#         models = create_model_list()
#         results = []
#
#         for name, model in models:
#             try:
#                 logging.info(f"Evaluating {name}...")
#                 pipeline = Pipeline([
#                     ('scaler', StandardScaler()),
#                     ('model', MultiOutputRegressor(model))
#                 ])
#                 rmse = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
#                 if rmse is not None:
#                     results.append({
#                         'Model': name,
#                         'RMSE': rmse,
#                         'Total RMSE': np.sum(rmse)
#                     })
#             except Exception as e:
#                 logging.error(f"Error with {name}: {str(e)}")
#                 continue
#
#         # Create results DataFrame
#         results_df = pd.DataFrame(results).sort_values('Total RMSE')
#
#         # Save results
#         output_dir = os.path.join('data', 'processed')
#         os.makedirs(output_dir, exist_ok=True)
#         results_path = os.path.join(output_dir, 'dcl_price_forecast_results.csv')
#         results_df.to_csv(results_path, index=False)
#
#         logging.info(f"Results saved to: {results_path}")
#         logging.info("\nTop 5 Models:")
#         print(results_df.head())
#
#     except Exception as e:
#         logging.error(f"Error in main execution: {str(e)}")
#         raise
#
# if __name__ == "__main__":
#     main()
