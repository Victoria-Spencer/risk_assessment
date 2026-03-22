import numpy as np
import pandas as pd
import os
import joblib
import yaml
import shutil
import time
from filelock import FileLock
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# ========== 加载配置文件 ==========
def load_config(config_path="config.yaml"):
    """加载配置文件，返回配置字典（含异常处理）"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"加载配置失败：{str(e)}")


# 全局配置
CONFIG = load_config()
MODEL_DIR = os.path.dirname(CONFIG["model"]["model_path"])
MODEL_PATH = CONFIG["model"]["model_path"]
SCALER_PATH = CONFIG["model"]["scaler_path"]
FEATURE_COLS = CONFIG["model"]["feature_cols"]
RANDOM_SEED = CONFIG["model"]["random_seed"]
N_SAMPLES = CONFIG["train"]["n_samples"]
TEST_SIZE = CONFIG["train"]["test_size"]


# ========== 安全保存文件函数 ==========
def safe_save_file(file_path, obj, backup_suffix=".bak"):
    """安全保存文件：检测占用+备份旧文件"""
    max_retry = 3
    retry_count = 0

    while retry_count < max_retry:
        try:
            # 备份旧文件
            if os.path.exists(file_path):
                backup_path = file_path + backup_suffix
                shutil.copy2(file_path, backup_path)
                print(f"📌 已备份旧文件到：{backup_path}")

            # 保存新文件
            joblib.dump(obj, file_path)
            print(f"📌 文件保存成功：{file_path}")
            return True
        except PermissionError as e:
            retry_count += 1
            print(f"⚠️ 文件被占用，{retry_count}次重试（共{max_retry}次）...")
            time.sleep(1)
    raise RuntimeError(f"❌ 保存文件失败：{file_path}（请关闭API服务后重试）")


# ========== 生成模拟数据（仅优化分布，完全保留原有非线性计算） ==========
def generate_simulate_data(n_samples=10000):
    np.random.seed(RANDOM_SEED)
    # 1. 保留原有特征生成逻辑，仅优化保额分布以贴合你的风险等级区间（不破坏非线性）
    data = {
        "occupation_risk_level": np.random.choice(
            [0, 1, 2, 3, 4],
            size=n_samples,
            p=[0.2, 0.2, 0.2, 0.15, 0.25]
        ),
        "age": np.concatenate([
            np.random.randint(0, 18, int(n_samples * 0.1)),
            np.random.randint(18, 36, int(n_samples * 0.3)),
            np.random.randint(36, 51, int(n_samples * 0.3)),
            np.random.randint(51, 66, int(n_samples * 0.15)),
            np.random.randint(66, 121, int(n_samples * 0.15))
        ]),
        # 优化：保额分布贴合你的风险等级区间（仅调整分布，不改变数值类型）
        "insure_amount": np.concatenate([
            np.random.uniform(0, 100000, int(n_samples * 0.2)),  # 低风险区间
            np.random.uniform(100000.01, 500000, int(n_samples * 0.3)),  # 较低风险
            np.random.uniform(500000.01, 1000000, int(n_samples * 0.2)),  # 中风险
            np.random.uniform(1000000.01, 2000000, int(n_samples * 0.15)),  # 高风险
            np.random.uniform(2000000.01, 5000000, int(n_samples * 0.15))  # 极高风险
        ]),
        "has_history_disease": np.random.choice(
            [True, False],
            size=n_samples,
            p=[0.15, 0.85]
        )
    }
    df = pd.DataFrame(data)

    # 2. 完全保留你原有total_risk_score计算逻辑（无修改）
    df['total_risk_score'] = (
            df['occupation_risk_level'] * 20 +
            np.clip(df['age'] // 10, 0, 10) +
            df['has_history_disease'].astype(int) * 8
    )
    df['total_risk_score'] = np.clip(df['total_risk_score'], 0, 100)

    # 3. 完全保留你原有risk_probability的非线性计算逻辑（核心！无任何修改）
    df['risk_probability'] = (
            df['total_risk_score'] / 100 * 0.4 +
            df['occupation_risk_level'] / 4 * 0.25 +
            df['has_history_disease'].astype(int) * 0.2 +
            np.clip((df['age'] / 120) * (df['insure_amount'] / 1000000) * 0.15, 0, 0.15)
    )
    df['risk_probability'] = np.clip(df['risk_probability'], 0.0, 1.0)

    # 4. 打乱数据顺序（原有逻辑）
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ========== 训练模型（完全保留原有逻辑，无修改） ==========
def train_and_save_model():
    # 加文件锁，避免多进程重复训练
    lock_file = os.path.join(MODEL_DIR, "train.lock")
    lock = FileLock(lock_file)

    with lock:
        # 1. 生成数据
        df = generate_simulate_data(n_samples=N_SAMPLES)

        # 2. 特征处理（保留原有非线性交互特征）
        X = df[FEATURE_COLS].copy()
        X['age_occupation_interact'] = X['age'] * X['occupation_risk_level']  # 非线性交互项
        X['amount_disease_interact'] = X['insure_amount'] * X['has_history_disease'].astype(int)  # 非线性交互项
        X['has_history_disease'] = X['has_history_disease'].astype(int)
        y = df['risk_probability']

        # 3. 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # 4. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. 网格搜索
        print("🔍 开始网格搜索最优参数...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=2),
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=2
        )
        grid_search.fit(X_train_scaled, y_train)

        # 6. 评估
        best_model = grid_search.best_estimator_
        print(f"✅ 最优参数：{grid_search.best_params_}")
        train_score = best_model.score(X_train_scaled, y_train)
        test_score = best_model.score(X_test_scaled, y_test)
        print(f"📌 训练集R²：{train_score:.6f}")
        print(f"📌 测试集R²：{test_score:.6f}")

        # 7. 安全保存模型
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        safe_save_file(MODEL_PATH, best_model)
        safe_save_file(SCALER_PATH, scaler)


if __name__ == "__main__":
    train_and_save_model()

# 你的年龄/保额风险等级映射（仅用于API侧展示，不影响训练逻辑）
age_risk_map = {
    (0, 17): "较低风险",
    (18, 35): "低风险",
    (36, 50): "中风险",
    (51, 65): "高风险",
    (66, 120): "极高风险"
}

sum_insured_risk_map = {
    (0.00, 100000.00): "低风险",
    (100000.01, 500000.00): "较低风险",
    (500000.01, 1000000.00): "中风险",
    (1000000.01, 2000000.00): "高风险",
    (2000000.01, 999999999.99): "极高风险"
}