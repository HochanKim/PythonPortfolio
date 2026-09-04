"""
SQLite 적재 계층
: 설비 데이터는 "같은 시각, 같은 설비"가 유일해야 하기 때문에
"machine_id", "ts" 값에 UNIQUE 제약을 걸고 UPSERT로 넣습니다.
=> 중복 전송이 와도 DB가 알아서 막아주는데 파이썬에서 막는 것보다 확실합니다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# 'data' 폴더 아래에 DB 파일 생성
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sensors.db"

# 스키마 생성
# => 'sensor_raw' 테이블 생성 후 각각의 column 생성
SCHEMA = """
  CREATE TABLE IF NOT EXISTS sensor_raw (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT  NOT NULL,
    ts    TEXT  NOT NULL,   -- IS08601 문자열
    type    TEXT,
    air_temp_k    REAL,
    rot_speed_rpm    REAL,  
    torque_nm    REAL,  
    process_temp_k  REAL,
    tool_wear_min    REAL,  
    vibration_mms REAL,
    current_a    REAL,  
    humidity_pct REAL,
    machine_failure INTEGER,
    collected_at  TEXT,
    UNIQUE (machine_id, ts)   -- * 중복 방어선
  );
  CREATE INDEX IF NOT EXISTS ix_sensor_ts       ON sensor_raw (ts);
  CREATE INDEX IF NOT EXISTS ix_sensor_machine  ON sensor_raw (machine_id, ts);
  
  CREATE TABLE IF NOT EXISTS collect_log (
    id  INTEGER PRIMARY KEY AUTOINCREMENT,  
    run_at  TEXT,
    window_start  TEXT,  
    window_end  TEXT,  
    rows_received  INTEGER,  
    rows_inserted  INTEGER,  
    rows_skipped  INTEGER,  
    note  TEXT
  );
  
"""
# -- 수집 이력(추적성): run_at, window_start/end, rows_received/interted/skipped,
# note CREATE TABLE IF NOT EXISTS collect_log (...);

COLUMNS = [
    "machine_id",
    "ts",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
    "machine_failure",
    "collected_at",
]


# DB 연결 함수
def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    # DB 설계도 실행
    con.executescript(SCHEMA)
    return con


# DB에 데이터를 보내는 함수 생성
def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    # (insert된 행 수, 중복으로 건너뛴 행 수)를 돌려주는 함수
    df = df.reindex(columns=COLUMNS)
    # 쿼리문 연결 (sensor_raw 테이블 내 자료 전부 가져오기)
    before = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    sql = (
        # 파일 주입 시 에러는 무시(IGNORE)
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' * len(COLUMNS))})"
    )
    con.executemany(
        sql, df.where(pd.notna(df), None).itertuples(index=False, name=None)
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    inserted = after - before
    return inserted, len(df) - inserted


# 로그 남기기용 함수 생성
def log_run(con, window_start, window_end, received, inserted, skipped, note=""):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end, "
        " rows_received, rows_inserted, rows_skipped, note)"
        "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (str(window_start), str(window_end), received, inserted, skipped, note),
    )
    con.commit()  # 저장 확정


# DB에 저장된 데이터를 읽어오는 함수
def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sensor_raw ORDER BY ts, machine_id", con)


# csv 파일 내보내기
from pathlib import Path

print(Path(__file__).resolve())
print(Path(__file__).resolve().parents[1])


# 파일 맨 아래(또는 별도 파일)에 추가해서 직접 확인
if __name__ == "__main__":
    print("DB 경로:", DB_PATH)
    con = connect()
    print("연결 및 생성 완료. 실제 파일 존재?", DB_PATH.exists())

# DB 파일 가져오기
con = connect("data/sensors.db")
df = read_all(con)
print(df)

# df.to_csv("db_확인용.csv", index=False, encoding="utf-8-sig")
