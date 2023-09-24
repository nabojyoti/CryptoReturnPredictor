import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# Support Vector Regression (SVR)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# XGBoost Regression
# import xgboost as xgb

# # LightGBM Regression
# import lightgbm as lgb

# # CatBoost Regression
# import catboost as catb

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def feature_engineering(data: pd.DataFrame, prediction_interval: int = 1) -> pd.DataFrame:
    """
    Perform feature engineering on financial data.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data with columns like 'datetime', 'open', 'high', 'low', 'close', etc.
    - prediction_interval: int 
        Time duration for prediction. 
        NOTE: The data is in 15 mins interval. hence , prediction_interval is 2 for 30 mins.

    Returns:
    - pd.DataFrame
        The DataFrame with additional features engineered from the input data.

    Example:
    engineered_data = feature_engineering(df)
    """
    # Calculate the percentage change in 'close' as the actual return
    data['label'] = data['close'].pct_change(periods = prediction_interval)

    # Historical Volatility (e.g., 10-period and 30-period rolling standard deviation of returns)
    data['Volatility_10'] = data['label'].rolling(window=10).std()
    data['Volatility_30'] = data['label'].rolling(window=30).std()

    # Lagged Returns
    data['Lagged_Return_1'] = data['label'].shift(1)
    data['Lagged_Return_2'] = data['label'].shift(2)
    data['Lagged_Return_3'] = data['label'].shift(3)
    data['Lagged_Return_4'] = data['label'].shift(4)
    data['Lagged_Return_5'] = data['label'].shift(5)
    data['Lagged_Return_7'] = data['label'].shift(7)
    data['Lagged_Return_9'] = data['label'].shift(9)



    # Bollinger Bands
    window = 20  # Adjust the window size as needed
    data['SMA_Bollinger'] = data['close'].rolling(window=window).mean()
    data['Upper_Bollinger'] = data['SMA_Bollinger'] + 2 * data['close'].rolling(window=window).std()
    data['Lower_Bollinger'] = data['SMA_Bollinger'] - 2 * data['close'].rolling(window=window).std()

    # Stochastic Oscillator (14-period)
    def calculate_stochastic(data, window):
        min_low = data['low'].rolling(window=window).min()
        max_high = data['high'].rolling(window=window).max()
        stochastic = ((data['close'] - min_low) / (max_high - min_low)) * 100
        return stochastic

    data['Stochastic_Oscillator_14'] = calculate_stochastic(data, window=14)

    # Ichimoku Cloud (We can adjust the periods as needed)
    data['Tenkan_Sen'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
    data['Kijun_Sen'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)
    data['Senkou_Span_B'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

    # Fourier Transform-based Features (only dominant frequency)
    def calculate_dominant_frequency(data, window):
        fft_result = np.fft.fft(data)
        fft_freqs = np.fft.fftfreq(len(fft_result))

        # Find the dominant frequency
        dominant_frequency = np.abs(fft_freqs[np.argmax(np.abs(fft_result))])

        return dominant_frequency

    window = 100  # Adjust the window size as needed
    data['Dominant_Frequency'] = data['close'].rolling(window=window).apply(
        lambda x: calculate_dominant_frequency(x, window), raw=False)
    
    data.dropna(inplace = True)

    return data

def calculate_feature_importance(model, X, y, method='coef', num_features=None):
    """
    Calculate feature importances using various methods.

    Parameters:
    - model: A regression model (e.g., LinearRegression, Lasso, RandomForestRegressor).
    - X: Feature matrix.
    - y: Target variable.
    - method: Feature importance method ('coef', 'importances', 'permutation', 'rfe', 'select_k_best').
    - num_features: Number of top features to select (only for 'select_k_best' method).

    Returns:
    - Feature importances.
    """

    if method == 'coef':
        if isinstance(model, (LinearRegression, Lasso)):
            model.fit(X, y)
            return model.coef_
        else:
            raise ValueError("Method 'coef' is only applicable to linear models.")

    elif method == 'importances':
        if isinstance(model, RandomForestRegressor):
            model.fit(X, y)
            return model.feature_importances_
        else:
            raise ValueError("Method 'importances' is only applicable to tree-based models.")

    elif method == 'permutation':
        if isinstance(model, (LinearRegression, RandomForestRegressor)):
            result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
            return result.importances_mean
        else:
            raise ValueError("Method 'permutation' is only applicable to linear and tree-based models.")

    elif method == 'rfe':
        if isinstance(model, LinearRegression):
            rfe = RFE(model, n_features_to_select=num_features)
            rfe.fit(X, y)
            return rfe.ranking_
        else:
            raise ValueError("Method 'rfe' is only applicable to linear models.")

    elif method == 'select_k_best':
        selector = SelectKBest(score_func=f_regression, k=num_features)
        X_new = selector.fit_transform(X, y)
        return selector.scores_

    else:
        raise ValueError("Invalid method. Choose from 'coef', 'importances', 'permutation', 'rfe', 'select_k_best'.")

def get_top_features(feature_importance:list, feature_names:list, num_features=10) -> list:
    """
    Get the top N features based on their importance scores.

    Args:
        feature_importance (list): A list of feature importance scores.
        feature_names (list): A list of feature names.
        num_features (int): The number of top features to choose.

    Returns:
        list: A list of tuples containing feature names and their importance scores.

    Example:
        top_features = get_top_features(feature_importance, feature_names, num_features=12)
    """
    # Sort feature importance in descending order and get the top N indices
    top_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:num_features]

    # Extract the top feature names and their importance scores
    top_feature_names = [feature_names[i] for i in top_indices]
    top_feature_scores = [feature_importance[i] for i in top_indices]

    # Create a list of tuples containing feature names and their importance scores
    top_features = list(zip(top_feature_names, top_feature_scores))

    return top_features


def hyperparameter_tuning(model, param_grid, X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning for a machine learning model.

    Args:
        model: The machine learning model (e.g., RandomForestRegressor, LinearRegression).
        param_grid (dict): The parameter grid for hyperparameter tuning.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.

    Returns:
        best_model: The best model with tuned hyperparameters.
        best_params: The best hyperparameters.
        evaluation_metrics: A dictionary containing evaluation metrics.

    Example:
        best_rf_model, best_rf_params, rf_metrics = hyperparameter_tuning(
            RandomForestRegressor(),
            {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]},
            X_train, y_train, X_test, y_test
        )
    """
    # Create the GridSearchCV object for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test data
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics (e.g., MSE, R-squared, RMSE)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Create a dictionary of evaluation metrics
    evaluation_metrics = {
        "Mean Squared Error (MSE)": mse,
        "R-squared (R2)": r2,
        "Root Mean Squared Error (RMSE)": rmse
    }

    return best_model, best_params, evaluation_metrics


