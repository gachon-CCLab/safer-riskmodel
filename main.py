# main.py ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import numpy as np

# ────────────────────────────────────────────────────────────
# 1. TFT 모델 정의 (기존과 동일)
# ────────────────────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        return self.softmax(self.fc2(torch.relu(self.fc1(x))))

class TemporalFusionTransformer(nn.Module):
    def __init__(self, hidden_size, lstm_layers, dropout, output_size,
                 attention_head_size, static_input_size, sequence_input_size):
        super().__init__()
        self.static_vsn   = VariableSelectionNetwork(static_input_size, hidden_size)
        self.sequence_vsn = VariableSelectionNetwork(sequence_input_size, hidden_size)
        self.lstm      = nn.LSTM(sequence_input_size, hidden_size,
                                 num_layers=lstm_layers, dropout=dropout,
                                 batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=attention_head_size,
                                               batch_first=True)
        self.static_fc = nn.Linear(static_input_size, hidden_size)
        self.fc        = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid   = nn.Sigmoid()
    def forward(self, x_static, x_seq):
        s_w   = self.static_vsn(x_static)
        s_out = self.static_fc(x_static * s_w)
        q_w   = self.sequence_vsn(x_seq)
        lstm_out,_ = self.lstm(x_seq * q_w)
        attn_out,_ = self.attention(lstm_out, lstm_out, lstm_out)
        fused = torch.cat((s_out, attn_out[:, -1, :]), dim=1)
        return self.sigmoid(self.fc(fused))

# ────────────────────────────────────────────────────────────
# 2. 모델 로드
# ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalFusionTransformer(
            hidden_size=16, lstm_layers=2, dropout=0.1, output_size=1,
            attention_head_size=4, static_input_size=10, sequence_input_size=26
        ).to(device)

ckpt = torch.load("tft_model.pth", map_location=device)
sd   = ckpt["model_state_dict"]
for k in list(sd):
    if k in model.state_dict() and model.state_dict()[k].shape != sd[k].shape:
        del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()

# ────────────────────────────────────────────────────────────
# 3. 컬럼 리스트 (모델이 학습 시 사용한 정확한 순서)
# ────────────────────────────────────────────────────────────
static_vars = [
    "MED_SH","BIS_sum","BPRS_AFF","PH_tx_status","convicted",
    "CTQ_PN","CTQ_EN","BPRS_POS","CS_status_1_score_cal","DIG_cat"
]
sequence_vars = [
    "Daily_Entropy","Normalized_Daily_Entropy","Eight_Hour_Entropy",
    "Normalized_Eight_Hour_Entropy","Location_Variability","place",
    "first_TOTAL_ACCELERATION","last_TOTAL_ACCELERATION","mean_TOTAL_ACCELERATION",
    "median_TOTAL_ACCELERATION","max_TOTAL_ACCELERATION","min_TOTAL_ACCELERATION",
    "std_TOTAL_ACCELERATION","nunique_TOTAL_ACCELERATION","first_HEARTBEAT",
    "last_HEARTBEAT","mean_HEARTBEAT","median_HEARTBEAT","max_HEARTBEAT",
    "min_HEARTBEAT","std_HEARTBEAT","nunique_HEARTBEAT","delta_DISTANCE",
    "delta_SLEEP","delta_STEP","delta_CALORIES"
]

# ────────────────────────────────────────────────────────────
# 4. Pydantic 입력 스키마
#    - 두 개 딕셔너리(static, sequence)로 컬럼명을 키로 보냄
# ────────────────────────────────────────────────────────────
class InputData(BaseModel):
    static:   Dict[str, float] = Field(..., description="10 static feature key-value pairs")
    sequence: Dict[str, float] = Field(..., description="26 sequence feature key-value pairs")

# ────────────────────────────────────────────────────────────
# 5. FastAPI 서버
# ────────────────────────────────────────────────────────────
app = FastAPI(title="TFT Predictor (컬럼명 기반 JSON)")

def dict_to_ordered_vector(src: Dict[str, float], ordered_keys: list[str]) -> np.ndarray:
    """
    src 딕셔너리에서 ordered_keys 순서대로 값을 뽑아 float32 벡터 생성.
    키 누락 시 오류.
    """
    try:
        return np.asarray([float(src[k]) for k in ordered_keys], dtype=np.float32)
    except KeyError as e:
        missing = e.args[0]
        raise HTTPException(
            status_code=422,
            detail=f"Missing required column: {missing}"
        )

@app.post("/predict")
def predict_api(data: InputData):
    # ① static / sequence 길이 검증 & 순서대로 벡터화
    if len(data.static) != 10 or len(data.sequence) != 26:
        raise HTTPException(status_code=400,
                            detail="Exactly 10 static columns and 26 sequence columns required.")

    static_vec = dict_to_ordered_vector(data.static, static_vars)      # (10,)
    seq_vec    = dict_to_ordered_vector(data.sequence, sequence_vars)  # (26,)

    # ② 텐서 변환
    static_tensor = torch.tensor(static_vec, dtype=torch.float32, device=device).unsqueeze(0)        # (1,10)
    seq_tensor    = torch.tensor(seq_vec,   dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)  # (1,1,26)

    # ③ 추론
    with torch.no_grad():
        prob = model(static_tensor, seq_tensor).item()
    label = int(prob >= 0.5)

    return {"probability": round(prob, 4), "label": label}

# ────────────────────────────────────────────────────────────
# 6. 실행 안내
# ────────────────────────────────────────────────────────────
"""
$ uvicorn main:app --reload
  → Swagger: http://localhost:8000/docs
"""
