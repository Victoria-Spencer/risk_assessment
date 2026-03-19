# 公共配置：统一管理路径、特征列等，避免重复代码
import os

# 模型保存路径（训练后生成）
MODEL_DIR = "./model_files"
MODEL_PATH = os.path.join(MODEL_DIR, "risk_rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "risk_scaler.joblib")

# 特征列（和请求参数对应）
FEATURE_COLS = ['total_risk_score', 'occupation_risk_level', 'age', 'insure_amount', 'has_history_disease']

# 随机种子（保证结果可复现）
RANDOM_SEED = 42