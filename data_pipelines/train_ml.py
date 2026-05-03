import argparse
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ProbaCalibrator:
    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes
        self.calibrators = [IsotonicRegression(out_of_bounds="clip") for _ in range(n_classes)]
        self._fitted = False

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "ProbaCalibrator":
        assert raw_proba.shape[1] == self.n_classes
        for cls_idx in range(self.n_classes):
            # Binary: is this sample's true class == cls_idx?
            binary_labels = (y_true == cls_idx).astype(float)
            self.calibrators[cls_idx].fit(raw_proba[:, cls_idx], binary_labels)
        self._fitted = True
        return self

    def predict_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        assert self._fitted, "Calibrator not fitted."
        calibrated = np.stack(
            [self.calibrators[i].predict(raw_proba[:, i]) for i in range(self.n_classes)],
            axis=1,
        )
        # Re-normalize: Isotonic per-class outputs don't sum to 1.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Guard division by zero
        return calibrated / row_sums


def evaluate_calibration(proba: np.ndarray, y_true: np.ndarray, label: str) -> None:
    ll = log_loss(y_true, proba)
    logger.info(f"{label} — Log-Loss (calibration): {ll:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=True, help="Path to XGBoost .json model")
    parser.add_argument("--val-features",  required=True, help="Validation features (.parquet)")
    parser.add_argument("--val-labels",    required=True, help="Validation labels (.parquet)")
    parser.add_argument("--output",        required=True, help="Output path for calibrator .pkl")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    logger.info(f"Model loaded: {args.model}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    # Modelin eğitildiği sütun isimlerini ve tam sıralamasını booster'dan alıyoruz
    expected_features = model.get_booster().feature_names
    logger.info(f"Model yüklendi. Beklenen öznitelikler: {expected_features}")

    # ── Load validation data ─────────────────────────────────────────────────
    X_val_raw = pd.read_csv(args.val_features)
    y_val_raw = pd.read_csv(args.val_labels)

    # Sadece modelin beklediği sütunları, tam olarak o sıralamayla seçiyoruz
    # Bu işlem 'close', 'timestamp' gibi fazlalıkları otomatik olarak filtreler
    try:
        X_val = X_val_raw[expected_features]
    except KeyError as e:
        missing_cols = set(expected_features) - set(X_val_raw.columns)
        logger.error(f"Kritik Hata: CSV içinde beklenen bazı sütunlar eksik: {missing_cols}")
        return

    # Etiketleri yükle (Sayısal olan son sütunu al)
    y_val = y_val_raw.select_dtypes(include=[np.number]).iloc[:, -1].values.astype(int)

    logger.info(f"Filtrelenmiş özellik seti şekli: {X_val.shape}")

    logger.info(f"Validation set: {len(X_val)} samples | Class distribution: "
                f"{dict(zip(*np.unique(y_val, return_counts=True)))}")

    # ── Raw proba before calibration ─────────────────────────────────────────
    raw_proba = model.predict_proba(X_val)
    # Modelin gerçekte kaç sınıf döndürdüğünü dinamik olarak al
    actual_n_classes = raw_proba.shape[1] 
    logger.info(f"Model {actual_n_classes} sınıflı olasılık döndürüyor.")
    
    evaluate_calibration(raw_proba, y_val, "BEFORE calibration")

    # ── Fit calibrator ───────────────────────────────────────────────────────
    # n_classes=3 yerine actual_n_classes kullan
    calibrator = ProbaCalibrator(n_classes=actual_n_classes)
    calibrator.fit(raw_proba, y_val)
    logger.info(f"{actual_n_classes} sınıflı Isotonic calibrator eğitildi.")
    # ── Evaluate calibrated output ───────────────────────────────────────────
    calibrated_proba = calibrator.predict_proba(raw_proba)
    evaluate_calibration(calibrated_proba, y_val, "AFTER calibration")

    # ── Save ─────────────────────────────────────────────────────────────────
    # args.output yolunun varlığını ve yazılabilirliğini doğrular
    import joblib
    import os
    
    output_path = args.output
    joblib.dump(calibrator, output_path)
    
    if os.path.exists(output_path):
        logger.info(f"SUCCESS: Calibrator saved at {output_path}")
    else:
        logger.error(f"FAILURE: Could not save calibrator to {output_path}")

if __name__ == "__main__":
    main()