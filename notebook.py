"""
pip install pandas sqlalchemy pymysql fastapi uvicorn
"""

import json, pathlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text

# ───────────────────────────────────────────────
# 0. DB ENGINE
# ───────────────────────────────────────────────
DB_URL = "mysql+pymysql://safer:Str0ng!Pass@210.102.181.208:40011/safer_db?charset=utf8mb4"
engine = create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)

# ───────────────────────────────────────────────
# 1. CSV → DB  (한 번에 적재)
# ───────────────────────────────────────────────
def load_csv_to_db(path: str):
    df = pd.read_csv(path)                # json_col, probability, label 3열 있다고 가정
    df["payload"] = df["json_col"]        # JSON 문자열 그대로
    df["src_file"] = pathlib.Path(path).name
    insert_df = df[["src_file","payload","probability","label"]]

    insert_df.to_sql(
        "sensor_payload", engine, if_exists="append",
        index=False, dtype={"payload": "JSON"},
        method="multi", chunksize=1000
    )
    print(f"{path}: {len(insert_df)} rows inserted")

# 예시 실행
load_csv_to_db("merged_data_m1_dong.csv")
load_csv_to_db("merged_data_m1_yong.csv")
load_csv_to_db("merged_data_m1_seoul.csv")

# ───────────────────────────────────────────────
# 2. REST API → DB
# ───────────────────────────────────────────────
app = FastAPI(title="SAFER JSON Ingest API")

class PayloadReq(BaseModel):
    payload: dict
    probability: float
    label: int

@app.post("/ingest")
def ingest(req: PayloadReq):
    with engine.begin() as conn:
        sql = text("""
            INSERT INTO sensor_payload (src_file, payload, probability, label)
            VALUES (:src, :pl, :prob, :lbl)
        """)
        conn.execute(sql, {
            "src": "api",
            "pl" : json.dumps(req.payload),
            "prob": req.probability,
            "lbl" : req.label
        })
    return {"status": "ok"}

# uvicorn main:app --host 0.0.0.0 --port 8080
