import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV  # 新增GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 导入公共配置
from config import MODEL_DIR, MODEL_PATH, SCALER_PATH, FEATURE_COLS, RANDOM_SEED


# ========== 1. 优化模拟训练数据（方案1核心） ==========
def generate_simulate_data(n_samples=10000):
    """
    优化点：
    1. 特征不再纯随机，贴合真实风控业务分布（高风险职业占比低、病史占比低等）
    2. total_risk_score由其他特征计算而来，而非随机
    3. 标签加入非线性交互项，发挥随机森林优势
    """
    np.random.seed(RANDOM_SEED)

    # 1. 特征：带业务逻辑的非纯随机生成
    data = {
        # 职业风险等级：高等级（3-4）占比25%，低等级（0-2）占比75%（贴合真实风控）
        "occupation_risk_level": np.random.choice(
            [0, 1, 2, 3, 4],
            size=n_samples,
            p=[0.2, 0.2, 0.2, 0.15, 0.25]
        ),
        # 年龄：符合真实人口分布（18-50岁占主流）
        "age": np.concatenate([
            np.random.randint(0, 18, int(n_samples * 0.1)),  # 少儿10%
            np.random.randint(18, 36, int(n_samples * 0.3)),  # 青年30%
            np.random.randint(36, 51, int(n_samples * 0.3)),  # 中年30%
            np.random.randint(51, 66, int(n_samples * 0.15)),  # 中老年15%
            np.random.randint(66, 121, int(n_samples * 0.15))  # 老年15%
        ]),
        # 保额：80%是1千-50万（主流），20%是50万-100万（高保额）
        "insure_amount": np.random.choice(
            [np.random.uniform(1000, 500000), np.random.uniform(500000, 1000000)],
            size=n_samples,
            p=[0.8, 0.2]
        ),
        # 病史：有病史的仅占15%（真实场景中低占比）
        "has_history_disease": np.random.choice(
            [True, False],
            size=n_samples,
            p=[0.15, 0.85]
        )
    }
    df = pd.DataFrame(data)

    # 2. 风险总分：由其他特征计算（而非随机），贴合业务逻辑
    df['total_risk_score'] = (
            df['occupation_risk_level'] * 20 +  # 职业等级0-4 → 0-80分
            np.clip(df['age'] // 10, 0, 10) +  # 年龄//10 → 0-12分
            df['has_history_disease'].astype(int) * 8  # 有病史加8分
    )
    df['total_risk_score'] = np.clip(df['total_risk_score'], 0, 100)  # 限制0-100分

    # 3. 风险概率标签：加入非线性交互项（高龄+高保额风险翻倍）
    df['risk_probability'] = (
            df['total_risk_score'] / 100 * 0.4 +  # 总分占40%
            df['occupation_risk_level'] / 4 * 0.25 +  # 职业占25%
            df['has_history_disease'].astype(int) * 0.2 +  # 病史占20%
            # 非线性交互项：年龄×保额（体现高年龄+高保额的额外风险）
            np.clip((df['age'] / 120) * (df['insure_amount'] / 1000000) * 0.15, 0, 0.15)
    )
    df['risk_probability'] = np.clip(df['risk_probability'], 0.0, 1.0)  # 边界处理

    # 打乱数据顺序（避免分组生成的顺序影响模型）
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ========== 2. 训练模型并保存（整合方案2+方案3） ==========
def train_and_save_model():
    """
    优化点：
    1. 增加特征交互项（方案2）
    2. 网格搜索调优超参数（方案3）
    3. 保留随机森林回归模型，不替换
    """
    # 1. 生成优化后的模拟数据
    df = generate_simulate_data()

    # 2. 基础特征提取 + 增加交互特征（方案2核心）
    X = df[FEATURE_COLS].copy()
    # 新增交互特征：年龄×职业风险（非线性关联）
    X['age_occupation_interact'] = X['age'] * X['occupation_risk_level']
    # 新增交互特征：保额×病史（高保额+有病史风险更高）
    X['amount_disease_interact'] = X['insure_amount'] * X['has_history_disease'].astype(int)
    # 转换布尔值为数值
    X['has_history_disease'] = X['has_history_disease'].astype(int)

    y = df['risk_probability']  # 标签：风险概率

    # 3. 数据拆分（训练集80%，测试集20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 网格搜索调优超参数（方案3核心）
    print("🔍 开始网格搜索最优参数...")
    param_grid = {
        'n_estimators': [100, 200, 300],  # 决策树数量
        'max_depth': [8, 12, 16],  # 单棵树最大深度
        'min_samples_split': [2, 5, 10],  # 节点分裂最小样本数
        'min_samples_leaf': [1, 2, 4]  # 叶节点最小样本数
    }
    # 网格搜索+5折交叉验证（用R²评分）
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    # 获取最优模型
    best_model = grid_search.best_estimator_
    print(f"✅ 最优参数：{grid_search.best_params_}")

    # 6. 模型评估
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"📌 训练集R²：{train_score:.4f}")
    print(f"📌 测试集R²：{test_score:.4f}")

    # 7. 创建模型保存目录
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 8. 保存最优模型和标准化器
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"📌 模型已保存到：{MODEL_PATH}")
    print(f"📌 标准化器已保存到：{SCALER_PATH}")


# ========== 执行训练 ==========
if __name__ == "__main__":
    train_and_save_model()