# missforest_imputer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

class ModifiedMissForest:
    """
    Custom MissForest for streamflow imputation with spatial/hydrological weighting.
    """
    def __init__(self, distance_matrix, connectivity, max_iter=10, n_estimators=100, random_state=42,
                 distance_weighting_type='inverse', decay_rate=0.1, temporal_feature_columns=None):
        self.distance_matrix = distance_matrix
        self.connectivity = connectivity
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.col_means = {}
        self.col_names = None
        self.discharge_columns = None
        self.temporal_feature_columns = temporal_feature_columns if temporal_feature_columns is not None else []
        self.site_to_idx = None
        self.distance_weighting_type = distance_weighting_type
        self.decay_rate = decay_rate

    def _calculate_weights(self, target_station, station_predictors):
        """Calculates blended spatial and hydrological weights for predictors."""
        if target_station not in self.distance_matrix.index:
            return pd.Series(0.0, index=station_predictors)

        actual_predictors_dist = [p for p in station_predictors if p in self.distance_matrix.columns]
        if not actual_predictors_dist:
            return pd.Series(0.0, index=station_predictors)

        distances = self.distance_matrix.loc[target_station, actual_predictors_dist]
        dist_weights = 1 / (distances + 1e-9) if self.distance_weighting_type == 'inverse' else np.exp(-self.decay_rate * distances)
        dist_weights = dist_weights.fillna(0).replace([np.inf, -np.inf], 0)

        actual_predictors_conn = [p for p in station_predictors if p in self.connectivity.columns]
        connectivity_weights = self.connectivity.loc[target_station, actual_predictors_conn] if target_station in self.connectivity.index and actual_predictors_conn else pd.Series(0.0, index=actual_predictors_conn)

        aligned_dist_weights = dist_weights.reindex(station_predictors, fill_value=0.0)
        aligned_connectivity = connectivity_weights.reindex(station_predictors, fill_value=0.0)

        aligned_dist_weights = aligned_dist_weights / aligned_dist_weights.sum() if aligned_dist_weights.sum() > 0 else pd.Series(0.0, index=station_predictors)
        aligned_connectivity = aligned_connectivity / aligned_connectivity.sum() if aligned_connectivity.sum() > 0 else pd.Series(0.0, index=station_predictors)

        alpha = 0.5
        blended_weights = alpha * aligned_dist_weights + (1 - alpha) * aligned_connectivity

        return blended_weights / blended_weights.sum() if blended_weights.sum() > 0 else pd.Series(0.0, index=station_predictors)

    def fit(self, X_incomplete):
        """Trains RandomForest models iteratively to impute missing values."""
        X = X_incomplete.copy()
        self.col_names = X.columns.tolist()
        self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
        self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}

        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()
            X[col] = X[col].fillna(self.col_means[col])

        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        total_original_nans = original_missing_mask.sum().sum()
        
        if total_original_nans == 0:
            for col_name in self.discharge_columns:
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X[temporal_predictors]
                
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                     self.models[col_name] = None
                     continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
            return self

        for iteration in range(self.max_iter):
            X_prev = X.copy()
            discharge_cols_order = np.random.RandomState(self.random_state + iteration).permutation(self.discharge_columns)

            for col_name in discharge_cols_order:
                y_known = X_incomplete.loc[~original_missing_mask[col_name], col_name]
                if y_known.empty:
                    self.models[col_name] = None
                    continue

                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X[temporal_predictors]
                
                X_train_for_model = X_predictors_combined.loc[y_known.index]
                y_train_for_model = y_known.loc[X_train_for_model.index]

                if X_train_for_model.empty or y_train_for_model.empty or (X_train_for_model == 0).all().all():
                    self.models[col_name] = None
                    continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_train_for_model, y_train_for_model)
                self.models[col_name] = model

                missing_in_col = original_missing_mask[col_name]
                if missing_in_col.any():
                    X_predict_for_model = X_predictors_combined.loc[missing_in_col]
                    if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                        X.loc[X_predict_for_model.index, col_name] = model.predict(X_predict_for_model)
                    else:
                        X.loc[missing_in_col, col_name] = self.col_means[col_name]

            current_imputed_vals = X[self.discharge_columns][original_missing_mask].stack().values
            prev_imputed_vals = X_prev[self.discharge_columns][original_missing_mask].stack().values

            change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals) if current_imputed_vals.size > 0 else 0.0
            if change_norm < 1e-6:
                break
        return self

    def transform(self, X_incomplete):
        """Imputes missing values in new data using trained models."""
        X_imp = X_incomplete.copy()

        if self.col_names is None: # Untrained imputer fallback
            self.col_names = X_incomplete.columns.tolist()
            self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
            self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}
            for col in self.discharge_columns:
                self.col_means[col] = X_incomplete[col].mean()
            return X_imp[self.discharge_columns].fillna(self.col_means)

        for col in self.discharge_columns:
            X_imp[col] = X_imp[col].fillna(self.col_means[col])

        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        if not missing_mask_new_data.sum().sum():
            return X_imp

        for iteration in range(self.max_iter):
            X_prev_imp = X_imp.copy()

            for col_name in self.discharge_columns:
                if col_name not in self.models or self.models[col_name] is None:
                    continue

                missing_in_col = missing_mask_new_data[col_name]
                if not missing_in_col.any(): continue

                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X_imp[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X_imp[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X_imp[temporal_predictors]

                X_predict_for_model = X_predictors_combined.loc[missing_in_col]

                if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                    X_imp.loc[X_predict_for_model.index, col_name] = self.models[col_name].predict(X_predict_for_model)
                else:
                    X_imp.loc[missing_in_col, col_name] = self.col_means[col_name]

            current_imputed_vals = X_imp[self.discharge_columns][missing_mask_new_data].stack().values
            prev_imputed_vals = X_prev_imp[self.discharge_columns][missing_mask_new_data].stack().values

            change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals) if current_imputed_vals.size > 0 else 0.0
            if change_norm < 1e-6:
                break
        return X_imp
