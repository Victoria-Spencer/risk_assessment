from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os
import yaml
import warnings

warnings.filterwarnings('ignore')

# ========== 新增：年龄/保额风险映射（核心修改） ==========
# 年龄风险映射（阈值区间: 风险等级）
age_risk_map = {
    (0, 17): "较低风险",
    (18, 35): "低风险",
    (36, 50): "中风险",
    (51, 65): "高风险",
    (66, 120): "极高风险"
}

# 保额风险映射（阈值区间: 风险等级）
sum_insured_risk_map = {
    (0.00, 100000.00): "低风险",
    (100000.01, 500000.00): "较低风险",
    (500000.01, 1000000.00): "中风险",
    (1000000.01, 2000000.00): "高风险",
    (2000000.01, 999999999.99): "极高风险"
}

# 新增：通用风险等级匹配函数
def get_risk_level(value, risk_map):
    """
    根据数值和风险映射表，获取对应的风险等级
    :param value: 待判断的数值（年龄/保额）
    :param risk_map: 风险映射表（age_risk_map/sum_insured_risk_map）
    :return: 风险等级字符串，若未匹配到返回"未知风险"
    """
    for (min_val, max_val), level in risk_map.items():
        if min_val <= value <= max_val:
            return level
    return "未知风险"

# ========== 加载配置文件 ==========
def load_config(config_path="config.yaml"):
    """加载配置文件，返回配置字典（含异常处理）"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 校验API必需的配置字段
        required_api_keys = ["model.model_path", "model.scaler_path", "model.feature_cols", "api.host", "api.port"]
        for key in required_api_keys:
            keys = key.split(".")
            val = config
            for k in keys:
                val = val.get(k)
                if val is None:
                    raise KeyError(f"配置文件缺失核心字段：{key}")

        return config
    except Exception as e:
        raise RuntimeError(f"加载配置失败：{str(e)}")

# 全局配置（启动时加载）
CONFIG = load_config()

# ========== 1. 初始化 FastAPI 应用 ==========
app = FastAPI(title="风控风险计算API", version="1.0")

# ========== 2. 加载模型和标准化器 ==========
def load_model_and_scaler():
    """加载训练好的模型和标准化器"""
    try:
        # 从配置读取路径
        model_path = CONFIG["model"]["model_path"]
        scaler_path = CONFIG["model"]["scaler_path"]

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}（请先运行train_model.py训练模型）")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在：{scaler_path}（请先运行train_model.py训练模型）")

        # 加载模型和标准化器
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ 模型和标准化器加载成功！")
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{str(e)}")

# 初始化模型（启动API时加载）
risk_model, risk_scaler = load_model_and_scaler()

# ========== 3. 定义请求/响应模型 ==========
class RiskDecisionPythonRequest(BaseModel):
    trace_id: str = Field(..., description="轨迹ID（唯一标识）")
    total_risk_score: int = Field(..., ge=0, le=100, description="风险因子总分（0-100）")
    occupation_risk_level: int = Field(..., ge=0, le=4, description="职业风险等级（0-4）")
    age: int = Field(..., ge=0, le=120, description="投保年龄（0-120）")
    insure_amount: float = Field(..., gt=0, description="投保保额（元）")
    has_history_disease: bool = Field(..., description="是否有既往病史")

    # 自定义校验：保额不能为0
    @validator('insure_amount')
    def insure_amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('投保保额必须大于0')
        return v

# 响应模型：仅返回概率（3位小数）+ 分析文本
class RiskDecisionPythonResponse(BaseModel):
    python_risk_probability: float = Field(..., ge=0.0, le=1.0, description="Python侧风险概率（0-1，保留3位小数）")
    python_risk_analysis: str = Field(..., description="Python侧风险分析（供Java端整合原因）")

# ========== 4. 辅助函数：生成风险分析文本（核心修改） ==========
def generate_risk_analysis(request: RiskDecisionPythonRequest, risk_prob: float) -> str:
    """根据参数和风险概率，生成结构化的风险分析"""
    analysis_parts = []
    # 1. 总分相关
    if request.total_risk_score >= 80:
        analysis_parts.append(f"风险因子总分{request.total_risk_score}（极高）")
    elif request.total_risk_score >= 50:
        analysis_parts.append(f"风险因子总分{request.total_risk_score}（中高）")
    else:
        analysis_parts.append(f"风险因子总分{request.total_risk_score}（低/中等）")

    # 2. 职业风险相关
    occupation_level_map = {0: "低", 1: "较低", 2: "中", 3: "高", 4: "极高"}
    analysis_parts.append(
        f"职业风险等级{request.occupation_risk_level}（{occupation_level_map[request.occupation_risk_level]}）")

    # 3. 病史相关
    if request.has_history_disease:
        analysis_parts.append("存在既往病史")
    else:
        analysis_parts.append("无既往病史")

    # 4. 年龄风险（核心修改：替换原简单判断为风险等级映射）
    age_risk_level = get_risk_level(request.age, age_risk_map)
    analysis_parts.append(f"投保年龄{request.age}岁（{age_risk_level}）")

    # 5. 保额风险（核心修改：替换原简单判断为风险等级映射）
    amount_risk_level = get_risk_level(request.insure_amount, sum_insured_risk_map)
    analysis_parts.append(f"投保保额{request.insure_amount:.2f}元（{amount_risk_level}）")

    # 风险概率展示（3位小数）
    analysis = "; ".join(analysis_parts) + f"；计算得出风险概率{risk_prob:.3f}"
    return analysis

# ========== 5. 核心API接口 ==========
@app.post("/risk/calculate", response_model=RiskDecisionPythonResponse)
async def calculate_risk(request: RiskDecisionPythonRequest):
    """
    风控风险计算接口（供Java端调用）
    输入：Java端传递的风控参数
    输出：Python侧风险概率（3位小数） + 风险分析
    """
    try:
        # 1. 构造基础特征数组（匹配配置的特征列顺序）
        feature_cols = CONFIG["model"]["feature_cols"]
        base_features = np.array([
            request.total_risk_score,
            request.occupation_risk_level,
            request.age,
            request.insure_amount,
            1 if request.has_history_disease else 0
        ], dtype=np.float64)

        # 2. 补充训练时的交互特征（关键：和训练侧保持一致）
        age_occupation_interact = request.age * request.occupation_risk_level
        amount_disease_interact = request.insure_amount * (1 if request.has_history_disease else 0)

        # 合并基础特征+交互特征
        full_features = np.concatenate([base_features, [age_occupation_interact, amount_disease_interact]]).reshape(1, -1)

        # 3. 特征标准化（仅transform，避免数据泄露）
        features_scaled = risk_scaler.transform(full_features)

        # 4. 模型预测风险概率（回归模型用predict，而非predict_proba）
        risk_prob_raw = risk_model.predict(features_scaled)[0]  # 取第一个（唯一）预测值
        risk_prob = round(float(risk_prob_raw), 3)  # 转float+保留3位小数
        print(f"📌 模型预测风险概率：{risk_prob}")

        # 5. 生成风险分析
        risk_analysis = generate_risk_analysis(request, risk_prob)

        # 6. 返回结果
        return RiskDecisionPythonResponse(
            python_risk_probability=risk_prob,
            python_risk_analysis=risk_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python侧风险计算失败：{str(e)}")

# ========== 6. 启动API服务 ==========
if __name__ == "__main__":
    import uvicorn

    # 从配置读取监听IP和端口
    api_host = CONFIG["api"]["host"]
    api_port = CONFIG["api"]["port"]

    # 启动服务
    print(f"🚀 API服务启动中，监听地址：{api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port)