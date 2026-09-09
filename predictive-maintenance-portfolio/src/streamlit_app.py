from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fonts

# ↓↓↓ 디버그용 임시 코드 (확인 후 지워도 됨)
st.write("폰트 파일 존재?", fonts.FONT_PATH.exists())
st.write(
    "폰트 파일 크기(바이트):",
    fonts.FONT_PATH.stat().st_size if fonts.FONT_PATH.exists() else "없음",
)
st.write("적용된 font.family:", plt.rcParams["font.family"])


# ============================================================
# 시뮬레이션 로직 (기존 코드 그대로 + rng.integer 오타를 rng.integers로 수정)
# ============================================================
MACHINES = {
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}


def _simulate_one(machine_id, n_minutes, start, rng):
    spec = MACHINES[machine_id]
    ts = pd.date_range(start, periods=n_minutes, freq="min")
    hour = ts.hour + ts.minute / 60.0
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))
    air = air + rng.normal(0, 0.15, n_minutes)

    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty
    wear, acc = np.zeros(n_minutes), rng.uniform(0, 60)
    limit = tool_life * rng.uniform(0.9, 1.15)
    for i in range(n_minutes):
        acc += wear_rate[i]
        if acc > limit:
            acc, limit = 0.0, spec["tool_life"] * rng.uniform(0.9, 1.15)
        wear[i] = acc

    rpm = np.clip(2860 - 1500 * duty + rng.normal(0, 45, n_minutes), 1150, 2900)
    torque = np.clip(
        10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes), 3, 80
    )

    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail

    power_w = torque * rpm * 2 * np.pi / 60.0
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear + rng.normal(0, 0.12, n_minutes)
    proc = proc - 6.0 * hvac_fail
    proc = proc + rng.normal(0, 0.12, n_minutes)

    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / spec["tool_life"]) ** 3
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

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

    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power_w < 3500) | (power_w > 9000)
    osf = (wear * torque) > spec["osf_limit"]
    rnf = rng.random(n_minutes) < 0.0002

    df["twf"], df["hdf"], df["pwf"] = twf.astype(int), hdf.astype(int), pwf.astype(int)
    df["osf"], df["rnf"] = osf.astype(int), rnf.astype(int)
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


def simulate_truth(n_minutes=1440, start="2026-09-02", seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


# ============================================================
# Streamlit 화면 구성
# ============================================================
st.set_page_config(page_title="CNC 설비 시뮬레이터", layout="wide")
st.title("🏭 CNC 설비 시뮬레이션 대시보드")

# --- 사이드바: 파라미터를 사용자가 직접 조절 ---
st.sidebar.header("시뮬레이션 설정")
일수 = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=1, max_value=100, value=14)
시작일 = st.sidebar.date_input("시작일", value=pd.Timestamp("2026-09-01"))
시드값 = st.sidebar.number_input("시드(seed)", min_value=0, value=42, step=1)

# --- 시뮬레이션 실행 (파라미터가 바뀔 때마다 자동 재실행) ---
truth = simulate_truth(n_minutes=1440 * 일수, start=str(시작일), seed=시드값)

# --- 질문하신 코드의 3줄 요약 정보를 화면 위쪽에 카드 형태로 표시 ---
col1, col2, col3 = st.columns(3)
col1.metric("설비 수", f"{truth['machine_id'].nunique()}대")
col2.metric("기간", f"{일수}일")
col3.metric("행 수", f"{len(truth):,}행")

st.caption(f"기간: {truth['ts'].min()} ~ {truth['ts'].max()}")

st.divider()

# --- 설비 선택 후 추이 그래프 ---
설비선택 = st.selectbox("설비 선택", sorted(truth["machine_id"].unique()))
지표선택 = st.multiselect(
    "확인할 지표",
    ["air_temp_k", "process_temp_k", "vibration_mms", "tool_wear_min", "torque_nm"],
    default=["air_temp_k", "process_temp_k"],
)

대상 = truth[truth["machine_id"] == 설비선택].copy()
대상["날짜"] = 대상["ts"].dt.date
일별평균 = 대상.groupby("날짜")[지표선택].mean()

st.subheader(f"{설비선택} 일별 평균 추이")
st.line_chart(일별평균)

st.subheader("고장 발생 현황")
고장열 = ["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]
고장요약 = truth.groupby("machine_id")[고장열].sum()
st.dataframe(고장요약)

st.subheader("원본 데이터 미리보기")
st.dataframe(truth.head(100))

# --- 다운로드 버튼 ---
csv = truth.to_csv(index=False).encode("utf-8-sig")
st.download_button("전체 데이터 CSV로 다운로드", csv, "시뮬레이션_결과.csv", "text/csv")


# 그래프 호출
st.subheader(f"{설비선택} 이동시 참값 — 마모 누적과 교체, 그리고 고장 시점")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 1) 공구 마모 (톱니 모양)
axes[0].plot(대상["ts"], 대상["tool_wear_min"], color="steelblue", linewidth=0.8)
axes[0].set_ylabel("공구 마모(분)")

# 2) 토크
axes[1].plot(대상["ts"], 대상["torque_nm"], color="seagreen", linewidth=0.6)
axes[1].set_ylabel("토크(Nm)")

# 3) 진동
axes[2].plot(대상["ts"], 대상["vibration_mms"], color="peru", linewidth=0.6)
axes[2].set_ylabel("진동(mm/s)")
axes[2].set_xlabel("시각")

# 고장 발생 시점을 빨간 띠로 표시 (모든 서브플롯에 공통 적용)
고장시점 = 대상.loc[대상["machine_failure"] == 1, "ts"]
for ax in axes:
    for t in 고장시점:
        ax.axvspan(t, t + pd.Timedelta(minutes=1), color="red", alpha=0.3)

fig.suptitle(f"{설비선택} 이동시 참값 — 마모 누적과 교체, 그리고 고장 시점")
plt.tight_layout()

st.pyplot(fig)
