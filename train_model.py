import numpy as np
import pandas as pd
import os
import joblib  # 用于保存/加载sklearn模型（比pickle更高效）
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 导入公共配置
from config import MODEL_DIR, MODEL_PATH, SCALER_PATH, FEATURE_COLS, RANDOM_SEED


# ========== 1. 生成模拟训练数据 ==========
def generate_simulate_data(n_samples=10000):
    """生成模拟风控训练数据：特征 + 标签（风险概率0-1）"""
    np.random.seed(RANDOM_SEED)

    # 特征：和请求参数一一对应
    data = {
        "total_risk_score": np.random.randint(0, 101, n_samples),
        "occupation_risk_level": np.random.randint(0, 5, n_samples),
        "age": np.random.randint(0, 121, n_samples),
        "insure_amount": np.random.uniform(1000, 1000000, n_samples),  # 保额1千-100万
        "has_history_disease": np.random.choice([True, False], n_samples)
    }
    df = pd.DataFrame(data)

    # 构造标签（风险概率）：基于特征的非线性组合
    df['risk_probability'] = (
            df['total_risk_score'] / 100 * 0.5 +  # 总分占50%
            df['occupation_risk_level'] / 4 * 0.3 +  # 职业等级占30%
            df['has_history_disease'].astype(int) * 0.15 +  # 病史占15%
            np.clip(df['age'] / 120 * 0.05, 0, 0.05)  # 年龄占5%
    )
    # 防止概率超过1（边界处理）
    df['risk_probability'] = np.clip(df['risk_probability'], 0.0, 1.0)

    return df


# ========== 2. 训练模型并保存 ==========
def train_and_save_model():
    """训练随机森林模型，保存模型和标准化器到指定路径"""
    # 1. 生成数据
    df = generate_simulate_data()
    X = df[FEATURE_COLS]
    # 转换布尔值为数值（True=1，False=0）
    X['has_history_disease'] = X['has_history_disease'].astype(int)
    y = df['risk_probability']  # 标签：风险概率

    # 2. 数据拆分（训练集80%，测试集20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 3. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 5. 打印模型精度
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"模型训练完成 | 训练集R²：{train_score:.4f} | 测试集R²：{test_score:.4f}")

    # 6. 创建模型保存目录（如果不存在）
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 7. 保存模型和标准化器
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"模型已保存到：{MODEL_PATH}")
    print(f"标准化器已保存到：{SCALER_PATH}")


# ========== 执行训练 ==========
if __name__ == "__main__":
    train_and_save_model()