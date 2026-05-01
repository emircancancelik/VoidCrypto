import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PKL_PATH = os.path.join(DATA_DIR, "ml_ready_features.pkl")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "dl_price_action.json")

class DLModelTrainer:
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None

    def load_and_split_data(self) -> tuple:
        print("[*] Loading optimized PKL data...")
        df = pd.read_pickle(self.data_path)

        y = df['target']
        X = df.drop(columns=['target'])
        split_idx = int(len(df) * 0.8)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[*] Training Set: {X_train.shape[0]} rows | Test Set: {X_test.shape[0]} rows")
        return X_train, X_test, y_train, y_test

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("[*] Initializing XGBoost Classifier...")
        
        neg_class_count = (y_train == 0).sum()
        pos_class_count = (y_train == 1).sum()
        scale_weight = neg_class_count / pos_class_count

        self.model = xgb.XGBClassifier(
            n_estimators=300,            
            max_depth=5,                 
            learning_rate=0.05,
            subsample=0.8,               
            colsample_bytree=0.8,        
            scale_pos_weight=scale_weight, 
            tree_method='hist',          
            eval_metric='aucpr',         
            random_state=42
        )

        print("[*] Training in progress... (This might take a minute on M4)")
        self.model.fit(X_train, y_train)
        print("[+] Training completed.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        print("\n[*] Evaluating Model on Future (Unseen) Data...")
        predictions = self.model.predict(X_test)
        predict_proba = self.model.predict_proba(X_test)[:, 1]
        confidence_threshold = 0.65
        high_conf_preds = (predict_proba >= confidence_threshold).astype(int)

        print("-" * 50)
        print("STANDARD PREDICTIONS (Threshold: 0.50)")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
        
        print("-" * 50)
        print(f"HIGH CONFIDENCE PREDICTIONS (Threshold: {confidence_threshold})")
        print(classification_report(y_test, high_conf_preds))
    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.model.save_model(self.model_output_path)
        
        model_size_mb = os.path.getsize(self.model_output_path) / (1024 * 1024)
        print(f"\n[+] Ultra-lightweight model saved to {self.model_output_path}")
        print(f"[*] Model Size: {model_size_mb:.2f} MB (Ideal for KEDA Scale-to-Zero)")

if __name__ == "__main__":
    if not os.path.exists(PKL_PATH):
        print(f"[-] Error: ML ready features not found at {PKL_PATH}. Run the ETL script first.")
    else:
        trainer = DLModelTrainer(PKL_PATH, MODEL_OUTPUT_PATH)
        X_train, X_test, y_train, y_test = trainer.load_and_split_data()
        
        trainer.train_xgboost(X_train, y_train)
        trainer.evaluate_model(X_test, y_test)
        trainer.save_model()