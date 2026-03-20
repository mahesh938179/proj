# predictor/ml_engine/feature_selection.py
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    def __init__(self, n_pca_components=10, importance_threshold=0.01):
        self.n_pca_components = n_pca_components
        self.importance_threshold = importance_threshold
        self.xgb_model = None
        self.pca = None
        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.n_original_features = None

    def xgboost_feature_importance(self, X, y):
        self.n_original_features = X.shape[1]
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        self.xgb_model.fit(X, y)

        importances = self.xgb_model.feature_importances_
        important_mask = importances >= self.importance_threshold
        self.selected_features = np.where(important_mask)[0]

        if len(self.selected_features) == 0:
            self.selected_features = np.argsort(importances)[-10:]

        print(f"    XGBoost selected {len(self.selected_features)} / {X.shape[1]} features")
        return X[:, self.selected_features], importances

    def apply_pca(self, X):
        n_components = min(self.n_pca_components, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_components)
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"    PCA: {n_components} components, {explained:.2%} variance explained")
        return X_pca

    def fit_transform(self, X, y):
        print("    Running XGBoost feature importance...")
        X_important, importances = self.xgboost_feature_importance(X, y)
        print("    Applying PCA...")
        X_final = self.apply_pca(X_important)
        return X_final, importances

    def transform(self, X):
        """Transform new data using fitted selector - with safety checks"""
        if self.selected_features is None:
            raise ValueError("FeatureSelector not fitted yet. Call fit_transform first.")

        max_idx = self.selected_features.max()

        if X.shape[1] <= max_idx:
            # Pad with zeros if new data has fewer features
            required_cols = max_idx + 1
            padded = np.zeros((X.shape[0], required_cols))
            padded[:, :X.shape[1]] = X
            X = padded

        X_selected = X[:, self.selected_features]
        X_scaled = self.scaler.transform(X_selected)
        X_pca = self.pca.transform(X_scaled)
        return X_pca