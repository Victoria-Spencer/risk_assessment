from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# 导入公共配置（如果没有实际配置文件，可先注释，不影响核心逻辑）
# from config import MODEL_PATH, SCALER_PATH, FEATURE_COLS

# ========== 1. 初始化 FastAPI 应用 ==========
app = FastAPI(title="风控风险计算API", version="1.0")


# ========== 2. 加载预训练的模型和标准化器（模拟加载，实际需替换为真实路径） ==========
def load_model_and_scaler():
    """加载训练好的模型和标准化器（示例：模拟加载，实际需替换为真实逻辑）"""
    try:
        # 模拟加载（实际需替换为：model = joblib.load(MODEL_PATH)）
        # 这里为了代码可运行，临时返回空对象，实际使用时需删除模拟逻辑
        class MockModel:
            def predict(self, X):
                # 模拟预测结果（示例值，实际由真实模型输出）
                return np.array([0.9767])

        class MockScaler:
            def transform(self, X):
                return X

        model = MockModel()
        scaler = MockScaler()
        print("模型和标准化器加载成功！")
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


# 【修改1】响应模型：删除python_risk_score，概率改为保留3位小数
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
    if request.insure_amount >= 500000:
        analysis_parts.append(f"投保保额{request.insure_amount:.2f}元（高保额）")
    if request.age >= 60:
        analysis_parts.append(f"投保年龄{request.age}岁（高龄）")

    # 【修改2】风险概率展示改为3位小数
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
        # 构造特征数组（匹配模型输入）
        features = np.array([
            request.total_risk_score,
            request.occupation_risk_level,
            request.age,
            request.insure_amount,
            1 if request.has_history_disease else 0
        ]).reshape(1, -1)  # 转二维数组（sklearn要求）

        # 特征标准化
        features_scaled = risk_scaler.transform(features)

        # 模型预测风险概率
        risk_prob = float(risk_model.predict(features_scaled)[0])
        # 【修改3】四舍五入改为3位小数（核心修改）
        risk_prob = round(risk_prob, 3)

        # 生成风险分析
        risk_analysis = generate_risk_analysis(request, risk_prob)

        # 返回结果（删除python_risk_score字段）
        return RiskDecisionPythonResponse(
            python_risk_probability=risk_prob,
            python_risk_analysis=risk_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python侧风险计算失败：{str(e)}")


# ========== 6. 启动API服务 ==========
if __name__ == "__main__":
    import uvicorn

    # 监听所有网卡，端口8000（可根据需要修改host/port）
    uvicorn.run(app, host="10.22.209.68", port=8000)