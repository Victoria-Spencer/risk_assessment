import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ========== 配置项（和API层保持一致） ==========
MODEL_PATH = "models/risk_model.pkl"  # 模型保存路径
SCALER_PATH = "models/risk_scaler.pkl"  # 标准化器保存路径
FEATURE_COLS = [  # 特征列（和API层构造的特征顺序一致）
    "total_risk_score",
    "occupation_risk_level",
    "age",
    "insure_amount",
    "has_history_disease"
]
RANDOM_SEED = 42  # 固定随机种子，保证训练结果可复现


# ========== 生成模拟训练数据（可替换为真实业务数据） ==========
def generate_train_data(n_samples=10000):
    """生成贴合业务逻辑的风控训练数据（特征和风险标签强相关）"""
    np.random.seed(RANDOM_SEED)

    # 构造特征（模拟真实业务分布）
    total_risk_score = np.random.randint(0, 101, n_samples)
    occupation_risk_level = np.random.randint(0, 5, n_samples)
    age = np.random.randint(0, 121, n_samples)
    insure_amount = np.random.uniform(1000, 1000000, n_samples)
    has_history_disease = np.random.choice([0, 1], n_samples)

    # 构造风险标签（特征和风险强相关：总分越高/职业等级越高/年龄越大/有病史，风险越高）
    risk_score = (
                         total_risk_score * 0.4 +  # 总分权重40%
                         occupation_risk_level * 10 +  # 职业等级权重（0-4→0-40）
                         (age >= 60).astype(int) * 15 +  # 高龄加15分
                         has_history_disease * 20  # 有病史加20分
                 ) / 100  # 归一化到0-1
    # 按风险分数生成标签（概率和分数正相关）
    risk_label = np.random.binomial(1, risk_score, n_samples)

    # 组装数据框
    df = pd.DataFrame({
        "total_risk_score": total_risk_score,
        "occupation_risk_level": occupation_risk_level,
        "age": age,
        "insure_amount": insure_amount,
        "has_history_disease": has_history_disease,
        "risk_label": risk_label
    })
    return df


# ========== 训练模型并保存 ==========
def train_and_save_model():
    # 1. 生成/加载数据
    df = generate_train_data()
    X = df[FEATURE_COLS]
    y = df["risk_label"]

    # 2. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 3. 特征标准化（仅拟合训练集，避免数据泄露）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练逻辑回归模型（输出0-1概率）
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    # 5. 创建模型保存目录
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 6. 保存模型和标准化器
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # 输出训练结果
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"✅ 模型训练完成！")
    print(f"📌 训练集准确率：{train_acc:.4f}")
    print(f"📌 测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    train_and_save_model()