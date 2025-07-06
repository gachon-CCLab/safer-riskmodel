# main.py ──────────────────────────────────────────────────────────────
"""
TFT 예측 → MySQL 저장 → JSON 형태로 조회 (+ 외부 rwd/data API 연동)
────────────────────────────────────────────────────────────────────────
필수 패키지
  pip install "sqlalchemy>=2.0" pymysql python-dotenv fastapi uvicorn \
              torch numpy pydantic requests
.env 예시
  DB_HOST=210.102.181.208
  DB_PORT=40011
  DB_USER=root
  DB_PW=
  DB_NAME=safer
  MODEL_CKPT=./tft_model.pth
  RWD_BASE=http://210.102.181.208:40011   # ← rwd/data 서비스 주소
"""

# ───────────────────────── DB·기본 모듈 ─────────────────────────
import os, numpy as np, torch, torch.nn as nn, requests
from urllib.parse import quote_plus
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import uvicorn

load_dotenv()

# inference + (선택) DB 저장 함수 
def run_inference(static_dict: Dict[str,float], seq_dict: Dict[str,float]):
    static_v = to_vec(static_dict, static_vars)
    seq_v    = to_vec(seq_dict,    sequence_vars)

    st = torch.tensor(static_v, device=device).unsqueeze(0)
    sq = torch.tensor(seq_v, device=device).unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        prob_val = model(st, sq).item()
    label_val = int(prob_val >= 0.5)
    return round(prob_val, 4), label_val

# DB URL 빌드 
def build_db_url() -> str:
    user, host, port, db = (
        os.environ["DB_USER"],
        os.environ["DB_HOST"],
        os.environ["DB_PORT"],
        os.environ["DB_NAME"],
    )
    pw = os.getenv("DB_PW", "")
    cred = f"{user}:{quote_plus(pw)}@" if pw else f"{user}@"
    return f"mysql+pymysql://{cred}{host}:{port}/{db}?charset=utf8mb4"

engine = create_engine(build_db_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# TFT 아키텍처 정의 
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(input_size, hidden_size), nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):  # (B,S)
        return self.softmax(self.fc2(torch.relu(self.fc1(x))))

class TemporalFusionTransformer(nn.Module):
    def __init__(self, hidden_size, lstm_layers, dropout, output_size,
                 attn_heads, static_in, seq_in):
        super().__init__()
        self.static_vsn   = VariableSelectionNetwork(static_in, hidden_size)
        self.sequence_vsn = VariableSelectionNetwork(seq_in, hidden_size)
        self.lstm   = nn.LSTM(seq_in, hidden_size, num_layers=lstm_layers,
                              dropout=dropout, batch_first=True)
        self.attn   = nn.MultiheadAttention(hidden_size, attn_heads, batch_first=True)
        self.static_fc = nn.Linear(static_in, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_static, x_seq):                    # x_seq:(B,1,S)
        s = self.static_fc(x_static * self.static_vsn(x_static))         # (B,H)
        lstm,_ = self.lstm(x_seq * self.sequence_vsn(x_seq))             # (B,1,H)
        attn,_ = self.attn(lstm, lstm, lstm)                              # (B,1,H)
        fused = torch.cat((s, attn[:, -1, :]), dim=1)                    # (B,2H)
        return self.sigmoid(self.fc(fused))                              # (B,1)

# 모델 로드 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalFusionTransformer(
    hidden_size=16, lstm_layers=2, dropout=0.1, output_size=1,
    attn_heads=4, static_in=10, seq_in=26).to(device)

ckpt = torch.load(os.getenv("MODEL_CKPT", "./app/tft_model.pth"), map_location=device)
sd   = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
for k in list(sd):
    if k in model.state_dict() and model.state_dict()[k].shape != sd[k].shape:
        del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()

# 컬럼 순서 
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

# Pydantic 스키마
class InputData(BaseModel):
    static:   Dict[str, float] = Field(..., min_items=10, max_items=10)
    sequence: Dict[str, float] = Field(..., min_items=26, max_items=26)

# ───────────────────── FastAPI ─────────────────────

print("safer-riskmodel")

app = FastAPI(title="TFT Predictor + rwd/data 연동")

def to_vec(src: Dict[str,float], order: List[str]) -> np.ndarray:
    try:
        return np.asarray([float(src[k]) for k in order], dtype=np.float32)
    except KeyError as e:
        raise HTTPException(422, f"Missing column: {e.args[0]}")


@app.post("/predict")
def predict_json(data: InputData):
    prob, label = run_inference(data.static, data.sequence)
    return {"probability": prob, "label": label}

# predict-by-id : rwd/data에서 조회 → 예측 
RWD_BASE = os.getenv("RWD_BASE", "http://210.102.181.208:40011")
#  /rwd/data : DB → static/sequence JSON
# ───────────────────── FastAPI ─────────────────────
INTERNAL_FLAG = RWD_BASE in ("", "internal", "self")          # 편의 플래그
@app.get("/rwd/data")
def rwd_data(id: str = Query(..., description="환자 이름(식별자)")):
    if not INTERNAL_FLAG:
        try:
            r = requests.get(f"{RWD_BASE}/rwd/data", params={"id": id}, timeout=5)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(502, f"외부 rwd/data 호출 실패: {e}")
        return r.json()

    
    select_sql = text(
        f"SELECT {', '.join(static_vars + sequence_vars)} "
        "FROM sensor_pred_flat "
        "WHERE `이름` = :rid"
    )
    with SessionLocal() as db:
        row = db.execute(select_sql, {"rid": id}).mappings().first()
        if row is None:
            raise HTTPException(404, "이름이 DB에 없음")
        return {
            "static":   {k: row[k] for k in static_vars},
            "sequence": {k: row[k] for k in sequence_vars},
        }

@app.get("/predict-by-id")
def predict_by_id(id: str = Query(..., description="환자 이름(식별자)")):
    try:
        r = requests.get(f"{RWD_BASE}/rwd/data", params={"id": id}, timeout=5)
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"rwd/data 요청 실패: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"rwd/data 오류: {r.text}")
    payload = r.json()
    if "static" not in payload or "sequence" not in payload:
        raise HTTPException(500, "rwd/data 응답 형식 오류")

    prob, label = run_inference(payload["static"], payload["sequence"])
    return {"probability": prob, "label": label}

# ───────── 헬스체크 ─────────
@app.get("/")
def root():
    return {"msg": "TFT Predictor is up"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)