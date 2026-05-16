import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================

def load_and_preprocess(filepath):
    """Load earthquake data and perform initial preprocessing"""
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    
    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0])
    df.set_index('Timestamp', inplace=True)
    
    # Rename columns for clarity (assuming first row is header)
    df.columns = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Magnitude_type', 
                  'No_of_Stations', 'Gap', 'Close', 'RMS', 'SRC', 'EventID']
    
    # Remove any rows with critical missing values
    critical_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'No_of_Stations']
    df = df.dropna(subset=critical_cols)
    
    # Filter unrealistic values
    df = df[(df['Depth'] >= 0) & (df['Depth'] <= 700)]  # Realistic depth range
    df = df[(df['Magnitude'] > 0) & (df['Magnitude'] < 10)]  # Realistic magnitude
    
    return df

def engineer_features(df):
    """Create new features from existing data"""
    df_copy = df.copy()
    
    # Log transform depth (earthquake depth is often log-normally distributed)
    df_copy['Log_Depth'] = np.log1p(df_copy['Depth'])
    
    # Quality metrics
    df_copy['Stations_per_Gap'] = df_copy['No_of_Stations'] / (df_copy['Gap'] + 1)
    df_copy['Quality_Score'] = df_copy['No_of_Stations'] / (df_copy['RMS'] + 0.01)
    df_copy['Coverage_Quality'] = df_copy['No_of_Stations'] / (df_copy['Gap'] * df_copy['RMS'] + 1)
    
    # Temporal features
    df_copy['Hour'] = df_copy.index.hour
    df_copy['Day_of_week'] = df_copy.index.dayofweek
    df_copy['Month'] = df_copy.index.month
    df_copy['Year'] = df_copy.index.year
    
    # Spatial features - distance from reference point (e.g., San Francisco)
    sf_lat, sf_lon = 37.7749, -122.4194
    df_copy['Dist_from_SF'] = np.sqrt((df_copy['Latitude'] - sf_lat)**2 + 
                                       (df_copy['Longitude'] - sf_lon)**2)
    
    # Rolling statistics (requires sorting by time)
    df_copy = df_copy.sort_index()
    df_copy['Magnitude_rolling_mean_30'] = df_copy['Magnitude'].rolling(window=30, min_periods=1).mean()
    df_copy['Magnitude_rolling_std_30'] = df_copy['Magnitude'].rolling(window=30, min_periods=1).std()
    
    # Days since last event
    df_copy['Days_since_last'] = df_copy.index.to_series().diff().dt.days.fillna(0)
    
    # Interaction features
    df_copy['Depth_Stations_interaction'] = df_copy['Depth'] * df_copy['No_of_Stations']
    df_copy['Gap_RMS_interaction'] = df_copy['Gap'] * df_copy['RMS']
    
    return df_copy

# ============================================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================================================

def prepare_features(df):
    """Select and encode features for modeling"""
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Magnitude_type_encoded'] = le.fit_transform(df['Magnitude_type'].fillna('unknown'))
    
    # Select features (after engineering)
    feature_cols = [
        'Depth', 'Log_Depth', 'No_of_Stations', 'Gap', 'Close', 'RMS',
        'Stations_per_Gap', 'Quality_Score', 'Coverage_Quality',
        'Hour', 'Day_of_week', 'Month', 'Year',
        'Dist_from_SF', 'Magnitude_rolling_mean_30', 'Magnitude_rolling_std_30',
        'Days_since_last', 'Depth_Stations_interaction', 'Gap_RMS_interaction',
        'Magnitude_type_encoded', 'Latitude', 'Longitude'
    ]
    
    X = df[feature_cols]
    y = df['Magnitude']
    
    return X, y, feature_cols

# ============================================================================
# STEP 3: BUILD XGBOOST MODEL
# ============================================================================

class EarthquakeMagnitudePredictor:
    """XGBoost model for earthquake magnitude prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def create_ts_cv(self, n_splits=5):
        """Create time series cross-validation splits"""
        return TimeSeriesSplit(n_splits=n_splits)
    
    def train(self, X, y, use_tuning=True):
        """Train the XGBoost model"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_cols = X.columns
        
        if use_tuning:
            # Hyperparameter tuning with RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 300, 500, 700],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'reg_alpha': [0, 0.1, 1, 10],
                'reg_lambda': [0, 0.1, 1, 10],
                'min_child_weight': [1, 3, 5]
            }
            
            tscv = self.create_ts_cv(n_splits=5)
            
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                         random_state=42,
                                         n_jobs=-1)
            
            random_search = RandomizedSearchCV(
                xgb_model, param_dist, n_iter=50, 
                cv=tscv, scoring='neg_mean_squared_error',
                verbose=1, random_state=42, n_jobs=-1
            )
            
            random_search.fit(X_scaled, y)
            self.model = random_search.best_estimator_
            print(f"Best parameters: {random_search.best_params_}")
            
        else:
            # Default model
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions),
            'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        return metrics, predictions
    
    def get_feature_importance(self):
        """Get SHAP values for interpretability"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.scaler.transform(X_train))
        
        # Create feature importance plot
        shap.summary_plot(shap_values, X_train, feature_names=self.feature_cols)
        
        return shap_values

# ============================================================================
# STEP 4: TRAINING PIPELINE
# ============================================================================

# Load and preprocess
print("Loading data...")
df = load_and_preprocess('Earthquake_data_processed.xlsx')
print(f"Loaded {len(df)} earthquakes")

print("Engineering features...")
df = engineer_features(df)

print("Preparing features...")
X, y, feature_cols = prepare_features(df)

# Time-based split (respect temporal order)
split_idx = int(len(X) * 0.7)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train model
print("\nTraining XGBoost model...")
predictor = EarthquakeMagnitudePredictor()
predictor.train(X_train, y_train, use_tuning=True)

# Evaluate
print("\nEvaluating model...")
metrics, predictions = predictor.evaluate(X_test, y_test)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
print("="*50)

# ============================================================================
# STEP 5: LIGHTGBM ALTERNATIVE (for comparison)
# ============================================================================

class LightGBMPredictor:
    """LightGBM model for comparison"""
    
    def train(self, X, y):
        self.model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

# Train LightGBM for comparison
print("\nTraining LightGBM for comparison...")
lgb_predictor = LightGBMPredictor()
lgb_predictor.train(X_train, y_train)
lgb_predictions = lgb_predictor.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_predictions))
print(f"LightGBM RMSE: {lgb_rmse:.4f}")

# ============================================================================
# STEP 6: ERROR ANALYSIS
# ============================================================================

# Analyze prediction errors
errors = y_test.values - predictions
error_df = pd.DataFrame({
    'True_Magnitude': y_test,
    'Predicted_Magnitude': predictions,
    'Error': errors,
    'Absolute_Error': np.abs(errors)
})

print("\nError Statistics:")
print(error_df['Absolute_Error'].describe())

# Identify worst predictions
worst_predictions = error_df.nlargest(10, 'Absolute_Error')
print("\nWorst Predictions (highest error):")
print(worst_predictions)

# ============================================================================
# STEP 7: SAVE MODEL FOR PRODUCTION
# ============================================================================

import joblib

# Save model and preprocessing objects
joblib.dump(predictor.model, 'xgb_earthquake_model.pkl')
joblib.dump(predictor.scaler, 'scaler.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')

print("\nModel saved successfully!")

# ============================================================================
# PREDICTION EXAMPLE
# ============================================================================

def predict_new_earthquake(depth, stations, gap, close, rms, mag_type, lat, lon):
    """Predict magnitude for a new detected earthquake"""
    
    # Create feature dictionary
    new_data = pd.DataFrame({
        'Depth': [depth],
        'No_of_Stations': [stations],
        'Gap': [gap],
        'Close': [close],
        'RMS': [rms],
        'Magnitude_type': [mag_type],
        'Latitude': [lat],
        'Longitude': [lon]
    })
    
    # Engineer features (simplified version for single prediction)
    new_data['Log_Depth'] = np.log1p(new_data['Depth'])
    new_data['Stations_per_Gap'] = new_data['No_of_Stations'] / (new_data['Gap'] + 1)
    new_data['Quality_Score'] = new_data['No_of_Stations'] / (new_data['RMS'] + 0.01)
    new_data['Dist_from_SF'] = np.sqrt((new_data['Latitude'] - 37.7749)**2 + 
                                       (new_data['Longitude'] + 122.4194)**2)
    
    # Encode magnitude type
    le = LabelEncoder()
    le.fit(['Mx', 'ML', 'Md'])
    new_data['Magnitude_type_encoded'] = le.transform([mag_type])[0]
    
    # Fill missing features with defaults
    default_features = {
        'Coverage_Quality': 1.0,
        'Hour': 12,
        'Day_of_week': 3,
        'Month': 6,
        'Year': 2024,
        'Magnitude_rolling_mean_30': 3.0,
        'Magnitude_rolling_std_30': 0.5,
        'Days_since_last': 10,
        'Depth_Stations_interaction': depth * stations,
        'Gap_RMS_interaction': gap * rms
    }
    
    for col, val in default_features.items():
        if col not in new_data.columns:
            new_data[col] = val
    
    # Ensure all feature columns are present
    feature_cols_loaded = joblib.load('feature_columns.pkl')
    for col in feature_cols_loaded:
        if col not in new_data.columns:
            new_data[col] = 0
    
    new_data = new_data[feature_cols_loaded]
    
    # Scale and predict
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_earthquake_model.pkl')
    
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    
    return prediction

# Example prediction
example_pred = predict_new_earthquake(
    depth=10.5, 
    stations=15, 
    gap=120, 
    close=25, 
    rms=0.08, 
    mag_type='ML', 
    lat=36.5, 
    lon=-121.3
)
print(f"\nExample prediction for new earthquake: {example_pred:.2f} ergs")