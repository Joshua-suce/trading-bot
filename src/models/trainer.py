# Model trainer — orchestrates feature preparation, XGBoost training, ensemble building
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import settings
from src.models.ensemble import ModelEnsemble
from src.models.feature_engineer import FeatureEngineer

XGBModel: Any = None
try:
    from src.models.classifier import XGBoostClassifier as XGBModel

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBModel = None


class ModelTrainer:
    def __init__(self, model_dir: str | None = None, lookback: int | None = None):
        configured_dir = Path(model_dir or settings.model_dir).expanduser()
        if not configured_dir.is_absolute():
            configured_dir = Path(__file__).resolve().parents[2] / configured_dir
        self.model_dir = configured_dir.resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_engineer = FeatureEngineer(
            lookback=lookback or settings.feature_lookback
        )

    # Full pipeline: indicators → features → drop NaN → matrix + labels
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        from src.indicators.compute import compute_all_indicators

        df_ind = compute_all_indicators(df)
        df_feat = self.feature_engineer.create_features(df_ind)
        df_feat = df_feat.dropna()
        X = self.feature_engineer.get_feature_matrix(df_feat)
        y = df_feat["target_direction"].values
        return X, y

    # Train an XGBoost classifier and optionally save it to disk
    def train_xgb(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        save: bool = True,
    ):
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available, skipping XGB training")
            return None
        logger.info("Preparing data for XGBoost training...")
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = XGBModel()
        model.metadata = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
        }
        model.train(X_train, y_train, eval_set=(X_test, y_test))

        if save:
            path = str(self.model_dir / f"xgb_{symbol.upper()}_{timeframe}.json")
            model.save(path)
        return model

    # Train the full ensemble (XGBoost + optional LSTM)
    def train_ensemble(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        save: bool = True,
    ) -> ModelEnsemble:
        xgb = self.train_xgb(
            df,
            symbol=symbol,
            timeframe=timeframe,
            save=save,
        )
        ensemble = ModelEnsemble(xgb_model=xgb)
        return ensemble
