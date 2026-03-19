import numpy as np
import pandas as pd
import yaml
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

warnings.filterwarnings('ignore')


# ========== 读取配置文件 ==========
def load_config(config_path="config.yaml"):
    """加载配置文件，返回配置字典"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 校验核心配置是否存在
    required_keys = ["model.model_path", "model.scaler_path", "model.feature_cols", "model.random_seed"]
    for key in required_keys:
        keys = key.split(".")
        val = config
        for k in keys:
            val = val.get(k)
            if val is None:
                raise KeyError(f"配置文件缺失核心字段：{key}")

    return config


# 加载配置（全局生效）
CONFIG = load_config()


# ========== 生成模拟训练数据（可替换为真实业务数据） ==========
def generate_train_data():
    """生成贴合业务逻辑的风控训练数据（特征和风险标签强相关）"""
    n_samples = CONFIG["train"]["n_samples"]
    random_seed = CONFIG["model"]["random_seed"]
    np.random.seed(random_seed)

    # 构造特征（模拟真实业务分布）
    total_risk_score = np.random.randint(0, 101, n_samples)
    occupation_risk_level = np.random.randint(0, 5, n_samples)
    age = np.random.randint(0, 121, n_samples)
    insure_amount = np.random.uniform(1000, 1000000, n_samples)
    has_history_disease = np.random.choice([0, 1], n_samples)

    # 构造风险分数（特征和风险强相关）
    risk_score = (
                         total_risk_score * 0.4 +  # 总分权重40%（0-40）
                         occupation_risk_level * 10 +  # 职业等级权重（0-4→0-40）
                         (age >= 60).astype(int) * 15 +  # 高龄加15分
                         has_history_disease * 20  # 有病史加20分
                 ) / 100  # 归一化到0-1

    # 关键修复：将risk_score裁剪到[0,1]范围，避免p超出合法区间
    risk_score = np.clip(risk_score, 0.0, 1.0)

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
    # 1. 读取配置项
    model_path = CONFIG["model"]["model_path"]
    scaler_path = CONFIG["model"]["scaler_path"]
    feature_cols = CONFIG["model"]["feature_cols"]
    random_seed = CONFIG["model"]["random_seed"]
    test_size = CONFIG["train"]["test_size"]

    # 2. 生成/加载数据
    df = generate_train_data()
    X = df[feature_cols]
    y = df["risk_label"]

    # 3. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # 4. 特征标准化（仅拟合训练集，避免数据泄露）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练逻辑回归模型（输出0-1概率）
    model = LogisticRegression(random_state=random_seed)
    model.fit(X_train_scaled, y_train)

    # 6. 创建模型保存目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 7. 保存模型和标准化器
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # 输出训练结果
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"✅ 模型训练完成！")
    print(f"📌 训练集准确率：{train_acc:.4f}")
    print(f"📌 测试集准确率：{test_acc:.4f}")
    print(f"📌 模型保存路径：{os.path.abspath(model_path)}")
    print(f"📌 标准化器保存路径：{os.path.abspath(scaler_path)}")


if __name__ == "__main__":
    train_and_save_model()