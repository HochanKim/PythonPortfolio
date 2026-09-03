"""
설비 센서 시뮬레이터 (물리 기반)
CNC 밀링 설비 3대를 1분 단위로 시뮬레이션합니다
- truth: 오염 없는 참값
- observed: 현장에서 실제로 받은 오염된 데이터

"""

from __future__ import annotations
import numpy as np
import pandas as pd

# CNC 밀링 설비 3대 정보
MACHINES = {
    # machine_id:
    #   {type => 품질등급, osf_limit => 과부하 한계, tool_life => 공구 교체주기 (분/minute)}
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}

# 관측 오염 강도 (기본값 = "현장급")
POLLUTION = {
    "dropout_rate": 0.015,  # 통신 끊김
    "dropout_len": (3, 40),  # 끊김 길이(분)
    "nan_rate": 0.008,  # 개별 센서값만 NaN
    "spike_rate": 0.004,  # 센서 튐 (전기 노이즈)
    "dup_rate": 0.006,  # 같은 레코드 중복 전송
    "ts_jitter_rate": 0.05,  # 타임스탬프 흔들림
    "unit_mix_rate": 0.1,  # 단위 혼재 (K 대신 섭씨)
    "drift_per_day": 0.35,  # 온도 센서 드리프트 (K/day)
}


def _simulate_one(
    # rng: np.random.Generator => 'np.random.default_rng(seed=42)'으로 값 전달
    machine_id: str,
    n_minutes: int,
    start: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    spec = MACHINES[machine_id]  # 설비 스펙
    ts = pd.date_range(start, periods=n_minutes, freq="min")

    # 공정 부하: 주간에 높고 야간에 낮음 (일 주기)
    hour = ts.hour + ts.minute / 60.0
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # 공기 온도: 계절/일교차 + 랜덤워크
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))
    air = air + rng.normal(0, 0.15, n_minutes)

    # 공구 마모: 누적되다가 교체하면 0으로 변경 (벡터화 불가 -> 루프)
    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty
    wear, acc = np.zeros(n_minutes), rng.uniform(0, 60)
    limit = tool_life * rng.uniform(0.9, 1.15)  # 교체 시점은 정비반 재량
    for i in range(n_minutes):
        acc += wear_rate[i]  # 부하가 클수록 빨리 닳음
        if acc > limit:
            acc, limit = 0.0, spec["tool_life"] * rng.uniform(0.9, 1.15)
        wear[i] = acc

    # 회전수: 부하에 반비례 (무거운 절삭일수록 저속)
    rpm = np.clip(2860 - 1500 * duty + rng.normal(0, 45, n_minutes), 1150, 2900)
    rpm = np.clip(rpm, 1150, 2900)

    # 토크: 부하에 비례, 마모되면 저항이 증가
    torque = np.clip(
        10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes), 3, 80
    )
    torque = np.clip(torque, 3, 80)

    # 냉각(HVAC) 이상: 가끔 공장 내 공조에 문제가 발생하면 실내가 더워질 수 있음
    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail

    # 공정 온도 = 공기온도 + 절삭열(전력) + 마모분
    power_w = torque * rpm * 2 * np.pi / 60.0
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear + rng.normal(0, 0.12, n_minutes)
    proc = proc - 6.0 * hvac_fail  # 온도차(방열 이력)가 줄어듦
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # 진동: 마모에 3제곱으로 반응 (후반에 급격히)
    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / spec["tool_life"]) ** 3  # 마모에 3제곱 반응
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    # 전류: 전력 / (380V x  √3 × 역률 0.85)
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # 습도: 온도와 약한 음의 관계
    humid = 55 - 1.8 * (air - 298) + rng.normal(0, 2.5, n_minutes)
    humid = np.clip(humid, 15, 95)

    df = pd.DataFrame(
        {
            "ts": ts,
            "machine_id": machine_id,
            "type": spec["type"],
            "air_temp_k": air,
            "process_temp_k": proc,
            "rot_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "vibration_mms": vib,
            "current_a": current,
            "humidity_pct": humid,
        }
    )

    # ------------------------------------------------------------------
    # 고장 라벨 (AI4I 2020 정의 그대로)
    # ------------------------------------------------------------------

    # Tool Wear Failure (공구 마모 고장)
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
    # Heat Dissipation Failure (방열 실패)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    # Power Failure (전력 이상)
    pwf = (power_w < 3500) | (power_w > 9000)
    # Overstrain Failure (과부하)
    osf = (wear * torque) > spec["osf_limit"]
    # Random Failure (원인 불명)
    rnf = rng.random(n_minutes) < 0.0002

    df["twf"] = twf.astype(int)
    df["hdf"] = hdf.astype(int)
    df["pwf"] = pwf.astype(int)
    df["osf"] = osf.astype(int)
    df["rnf"] = rnf.astype(int)
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


# 오염 없는 참값을 생성하기 위한 함수 생성
def simulate_truth(
    # 시뮬레이션 총 분, 시작하는 날(시각), 난수 시드
    n_minutes: int = 1440,
    start: str | pd.Timestamp = "2026-09-02",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]
    out = pd.concat(parts, ignore_index=True)
    # 참값 생성 후 인덱스 번호 재정립
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


# print(
#     _simulate_one(
#         # 3개월 (90일) 단위로 시뮬레이션 체크
#         "CNC-01",
#         1440 * 90,
#         "2026-09-02 12:00",
#         np.random.default_rng(seed=42),
#     )
# )

# 14일치 샘플
truth = simulate_truth(n_minutes=1440 * 14, start="2024-01-01", seed=42)
print("설비 수 :", truth["machine_id"].nunique())
print("기간 :", truth["ts"].min(), "~", truth["ts"].max())
print("행 수 :", f"{len(truth):,}")
print()

# 고장 모드 분포
modes = truth[["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]].sum()
print(pd.DataFrame({"건수": modes, "비율(%)": (modes / len(truth) * 100).round(3)}))
print()

# 센서 요약
cols = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "power_w",
]
print(truth[cols].describe().loc[["mean", "std", "min", "50%", "max"]].round(2))

# pd.date_range("2024-01-01", periods=10, freq="T")
# => FutureWarning: 'T' is deprecated and will be removed in a future version, please use 'min' instead.
# => "'T'라는 표기는 지금은 쓸 수 있지만 앞으로 사라질 예정(deprecated)이다. 'min'으로 바꿔서 써라."

pd.date_range("2024-01-01", periods=10, freq="min")
# => pandas 최신 버전 적용
