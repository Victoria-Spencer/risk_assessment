from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os
import yaml
import warnings

warnings.filterwarnings('ignore')


# ========== 加载配置文件（核心修改：替换硬编码） ==========
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


# ========== 2. 加载真实模型和标准化器（使用配置文件路径） ==========
def load_model_and_scaler():
    """加载训练好的真实模型和标准化器"""
    try:
        # 从配置读取路径
        model_path = CONFIG["model"]["model_path"]
        scaler_path = CONFIG["model"]["scaler_path"]

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}（请先运行train_model.py训练模型）")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在：{scaler_path}（请先运行train_model.py训练模型）")

        # 加载真实模型和标准化器
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ 真实模型和标准化器加载成功！")
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{str(e)}")


# 初始化模型（启动API时加载）
risk_model, risk_scaler = load_model_and_scaler()


# ========== 3. 定义请求/响应模型（匹配 Java 端 DTO） ==========
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


# ========== 4. 辅助函数：生成风险分析文本 ==========
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

    # 4. 保额/年龄相关
    if request.insure_amount >= 1000000:
        analysis_parts.append(f"投保保额{request.insure_amount:.2f}元（高保额）")
    if request.age >= 50:
        analysis_parts.append(f"投保年龄{request.age}岁（高龄）")

    # 风险概率展示（3位小数）
    analysis = "; ".join(analysis_parts) + f"；计算得出风险概率{risk_prob:.3f}"
    return analysis


# ========== 5. 核心API接口（动态预测） ==========
@app.post("/risk/calculate", response_model=RiskDecisionPythonResponse)
async def calculate_risk(request: RiskDecisionPythonRequest):
    """
    风控风险计算接口（供Java端调用）
    输入：Java端传递的风控参数
    输出：Python侧风险概率（3位小数） + 风险分析
    """
    try:
        # 1. 构造特征数组（从配置读取特征列顺序，避免硬编码）
        feature_cols = CONFIG["model"]["feature_cols"]
        features = np.array([
            request.total_risk_score,
            request.occupation_risk_level,
            request.age,
            request.insure_amount,
            1 if request.has_history_disease else 0
        ], dtype=np.float64).reshape(1, -1)  # 转二维数组（sklearn要求）

        # 2. 特征标准化（使用训练好的scaler，仅transform）
        features_scaled = risk_scaler.transform(features)

        # 3. 模型预测风险概率（关键：用predict_proba取正例概率，而非固定值）
        # predict_proba返回 [负例概率, 正例概率]，取[1]即风险概率
        risk_prob_raw = risk_model.predict_proba(features_scaled)[0][1]
        print(f"📌 模型原始预测概率：{risk_prob_raw}")  # 打印真实预测值（不再固定）
        risk_prob = round(risk_prob_raw, 3)  # 保留3位小数

        # 4. 生成风险分析
        risk_analysis = generate_risk_analysis(request, risk_prob)

        # 5. 返回结果
        return RiskDecisionPythonResponse(
            python_risk_probability=risk_prob,
            python_risk_analysis=risk_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python侧风险计算失败：{str(e)}")


# ========== 6. 启动API服务（使用配置文件的IP/端口） ==========
if __name__ == "__main__":
    import uvicorn

    # 从配置读取监听IP和端口
    api_host = CONFIG["api"]["host"]
    api_port = CONFIG["api"]["port"]

    # 启动服务
    print(f"🚀 API服务启动中，监听地址：{api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port)