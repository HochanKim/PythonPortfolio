# 주기 수집을 위한 파일
# => GitHum Actions가 하루 한 번씩 이 파일을 실행합니다 / ex. python src/collector.py --minutes 1440

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db as dbmod
from simulator import (
    sample_window,
)  # simulator.py의 함수(메서드) sample_window 가져오기

ROOT = Path(__file__).resolve().parent[1]
HIST = ROOT / "data" / "history"


# 센서 소스 호출부
# => 실제 현장이면 여기에 REST API가 호출됩니다.
def fetch(minutes, end=None):
    """
    ex)
      r = requests.get(API, params={...}, timeout=10)
      r.raise_for_status()
      return pd.DataFrame(r.json()["items"])
    """
    return sample_window(n_minutes=minutes, end=end)


# collector 파일의 메인 함수
def main() -> int:
    # 명령줄 옵션을 정의하고 해석(파싱)하는 도구를 담은 객체 'ap'
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=1440, help="수집할 구간 길이 (분)")
    ap.add_argument("--end", default=None, help="구간 끝 시각 (기본: 현재 시각)")
    ap.add_argument("--db", default=str(dbmod.DB_PATH))
    ap.add_argument("--hist", default=str(HIST), help="일별 CSV 저장 폴더")
    args = ap.parse_args()

    hist = Path(args.hist)
    hist.mkdir(parents=True, exist_ok=True)

    try:
        raw = fetch(args.minutes, args.end)
    except Exception as e:
        # 수집 실패 오류를 위한 예외처리
        print(f"[ERROR] 수집 실패: {type(e).__name__}: {e}")
        return 1

    if raw.empty:
        # 받은 데이터가 없으면 0으로 return
        print("[WARN] 받은 데이터가 0건입니다. 종료합니다.")
        return 0

    w_start, w_end = raw["ts"].min(), raw["ts"].max()
    tag = pd.Timestamp(w_end).strftime("%Y-%m-%d")
    csv_path = hist / f"{tag}.csv"

    # 같은 날 여러번 돌아도 안전하게
    # => 기존 파일과 합쳐서 중복 제거
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        raw = pd.concat([old, raw], ignore_index=True)
    raw = raw.drop_duplicates(subset=["machine_id", "ts"], keep="last")
    raw.to_csv(csv_path, index=False)

    con = dbmod.connect(args.db)
    inserted, skipped = dbmod.upsert(con, raw)
    dbmod.log_run(
        con, w_start, w_end, len(raw), inserted, skipped, note=f"csv={csv_path.name}"
    )
    total = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    con.close()

    print(f"[OK] window {w_start} ~ {w_end}")
    print(f"     받은 행 {len(raw):,} / DB 신규 {inserted:,} / 중복 스킵 {skipped:,}")
    print(f"     CSV  {csv_path}")
    print(f"     DB 누적 {total:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
