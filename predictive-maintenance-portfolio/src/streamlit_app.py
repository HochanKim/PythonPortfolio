from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fonts

import altair as alt

# ============================================================
# 시뮬레이션 로직 (기존 코드 그대로 + rng.integer 오타를 rng.integers로 수정)
# ============================================================

# CNC 설비 3대의 시뮬레이션 설정
# type: 설비 구분 코드
# osf_limit: 공구 마모량 x (토크가 이 값을 넘으면 과부하 고장으로 판정)
# tool_life: 공구 교체 기준을 정할 때 사용하는 기준 수명
MACHINES = {
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}


# 설비 한 대의 센서값과 고장 여부를 1분 간격으로 생성
# machine_id: 설비 이름 / n_minutes: 생성할 시간 길이(분)
# start: 시작 시각 / rng: 난수 생성기
# 반환값(return): 시간별 센서값과 고장 표시가 담긴 DataFrame
def _simulate_one(machine_id, n_minutes, start, rng):
    spec = MACHINES[machine_id]

    # 시작 시각부터 1분 간격으로 시간 목록 생성
    ts = pd.date_range(start, periods=n_minutes, freq="min")

    # 시각을 소수 형태로 변환: ex. 09:30 -> 9.5
    hour = ts.hour + ts.minute / 60.0

    # 하루 주기로 변하는 가상의 가동 부하에 무작위 변동을 추가
    # 부하는 0.05 ~ 1.0 범위로 제한
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # 외기 온도(air): 하루 주기 변화 + 누적 변동 + 순간적인 잡음을 합쳐 생성
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))
    air = air + rng.normal(0, 0.15, n_minutes)

    tool_life = spec["tool_life"]

    # 가동 부하가 높을수록 공구 마모가 더 빠르게 누적된다.
    wear_rate = 1.0 + 0.6 * duty

    # 초기 마모량과 교체 기준에 무작위 차이를 줘서 사용 중인 공구를 모사
    wear, acc = np.zeros(n_minutes), rng.uniform(0, 60)
    limit = tool_life * rng.uniform(0.9, 1.15)

    for i in range(n_minutes):
        # 매 분(minutes) 마모량 누적
        acc += wear_rate[i]

        # 교체 기준을 넘으면 새 공구로 교체한 것으로 처리
        if acc > limit:
            acc, limit = 0.0, spec["tool_life"] * rng.uniform(0.9, 1.15)

        wear[i] = acc

    # 가동 부하가 커질수록 회전속도는 낮아지고 토크는 높아지도록 설정
    rpm = np.clip(2860 - 1500 * duty + rng.normal(0, 45, n_minutes), 1150, 2900)

    # 토크(torque)에는 공구 마모의 영향도 반영
    torque = np.clip(
        10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes), 3, 80
    )

    # 임의의 시간 구간을 냉방 이상 구간으로 지정
    hvac_fail = np.zeros(n_minutes, dtype=bool)

    # 해당 구간에서는 외기 온도와 공정 온도 계산을 조정
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail

    # 토크와 회전속도(rpm)로 기계적 동력(W) 계산
    power_w = torque * rpm * 2 * np.pi / 60.0

    # 공정 온도(proc): 외기 온도(air), 동력, 공구 마모, 잡음의 영향을 반영
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear + rng.normal(0, 0.12, n_minutes)
    proc = proc - 6.0 * hvac_fail
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # 진동: 회전속도와 공구 마모가 커질수록 증가하도록 설정
    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / spec["tool_life"]) ** 3
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    # 동력을 바탕으로 전류를 단순 추정하고 잡음을 추가
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # 외기 온도가 높아질수록 습도가 낮아지는 가상의 관계 적용
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

    # 공구 마모 고장: 마모량이 200~240인 구간에서 확률적으로 발생
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)

    # 열 방출 고장: '공정-외기 온도'가 8.6보다 작고 회전속도가 1380보다 낮으면 발생
    hdf = ((proc - air) < 8.6) & (rpm < 1380)

    # 동력 이상: 계산된 동력이 각각 지정 범위를 벗어나면 발생
    pwf = (power_w < 3500) | (power_w > 9000)

    # 과부하 고장: '마모량 * 토크'가 설비별 한계를 넘으면 발생
    osf = (wear * torque) > spec["osf_limit"]

    # 무작위 고장: 각 시점에 낮은 확률로 발생
    rnf = rng.random(n_minutes) < 0.0002

    # 각 고장 여부를 True/False에서 1/0으로 변환하여 저장
    df["twf"], df["hdf"], df["pwf"] = twf.astype(int), hdf.astype(int), pwf.astype(int)
    df["osf"], df["rnf"] = osf.astype(int), rnf.astype(int)

    # 다섯 가지 고장 중 하나라도 발생하면 전체 고장 표시를 1로 설정
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


# 설비별 데이터를 생성한 뒤 하나의 표로 합쳐 반환
def simulate_truth(n_minutes=1440, start="2026-09-02", seed=42):
    # 같은 입력과 시드로 실행하면 같은 난수 기반 데이터를 생성
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)

    # MACHINES에 등록된 각 설비에 대해 시뮬레이션 실행
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]

    # 설비별 표를 세로로 연결
    out = pd.concat(parts, ignore_index=True)

    # 시각과 설비 이름 순으로 정렬하고 행 번호 재설정
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


# ============================================================
# Streamlit 화면 구성
# ============================================================
# 브라우저 탭 제목을 설정하고 화면을 넓게 사용
st.set_page_config(page_title="CNC 설비 시뮬레이터", layout="wide")

# 대시보드 제목 설정
st.title("🏭 CNC 설비 시뮬레이션 대시보드")

# --- 사이드바: 파라미터를 사용자가 직접 조절 ---
# => 사이드바에 일수(dates), 시작일(start_date), 시드값(seed_v)를 입력받음
st.sidebar.header("시뮬레이션 설정")
# 일수
dates = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=1, max_value=100, value=14)
# 시작일
start_date = st.sidebar.date_input("시작일", value=pd.Timestamp("2026-09-01"))
# 시드값
seed_v = st.sidebar.number_input("시드(seed)", min_value=0, value=42, step=1)

# --- 시뮬레이션 실행 (파라미터가 바뀔 때마다 자동 재실행) ---
# => 일수를 분으로 변환하여 전체 설비 데이터 생성
# => 현재 구조에서는 화면 입력이 변경되어 스크립트가 재실행될 때 다시 계산
truth = simulate_truth(n_minutes=1440 * dates, start=str(start_date), seed=seed_v)

# --- 질문하신 코드의 3줄 요약 정보를 화면 위쪽에 카드 형태로 표시 ---
# => 전체 설비 수, 시뮬레이션 기간, 생성된 데이터 행 수를 카드로 표시
col1, col2, col3 = st.columns(3)
col1.metric("설비 수", f"{truth['machine_id'].nunique()}대")
col2.metric("기간", f"{dates}일")
col3.metric("행 수", f"{len(truth):,}행")

st.caption(f"기간: {truth['ts'].min()} ~ {truth['ts'].max()}")

st.divider()

# --- 설비 선택 후 추이 그래프 ---
select_machine = st.selectbox("설비 선택", sorted(truth["machine_id"].unique()))
select_indicator = st.multiselect(
    "확인할 지표",
    ["air_temp_k", "process_temp_k", "vibration_mms", "tool_wear_min", "torque_nm"],
    default=["air_temp_k", "process_temp_k"],
)

# 선택한 설비의 데이터만 추출
object_one = truth[truth["machine_id"] == select_machine].copy()

# 시각에서 날짜만 추출하여 날짜별 집계 기준으로 사용
object_one["날짜"] = object_one["ts"].dt.date

# 선택한 센서 지표를 날짜별로 묶어 평균 계산
avg_of_days = object_one.groupby("날짜")[select_indicator].mean()

st.subheader(f"{select_machine} 일별 평균 추이")

# 날짜별 평균값을 선 그래프로 표시
avg_of_days_long = avg_of_days.reset_index().melt(
    id_vars="날짜", var_name="지표", value_name="값"
)

chart = (
    alt.Chart(avg_of_days_long)
    .mark_line()
    .encode(
        x=alt.X("날짜:T", title="날짜"),
        # alt.Scale(zero=False): 핵심 수정 부분
        y=alt.Y("온도(K):Q", scale=alt.Scale(zero=False), title="온도(K)"),
        color=alt.Color("지표:N", title="지표"),
    )
)
st.altair_chart(chart, use_container_width=True)

# 설비별로 각 고장 조건이 참인 행 수를 합산
# => 1분 간격 데이터이므로 고장이 표시된 분 수에 해당하며,
#    서로 독립적인 고장 사건의 횟수를 의미하지는 않음
st.subheader("고장 발생 현황")
# 고장열
breakdown_columns = ["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]
# 고장요약
breakdown_summary = truth.groupby("machine_id")[breakdown_columns].sum()
st.dataframe(breakdown_summary)

st.subheader("원본 데이터 미리보기")
# 생성된 전체 데이터 중 첫 100행을 화면에 표시
st.dataframe(truth.head(100))

# --- 다운로드 버튼 ---
# => 전체 데이터를 CSV로 변환
csv = truth.to_csv(index=False).encode(
    "utf-8-sig"
)  # Excel에서 한글을 읽기 쉽도록 저장 ("utf-8-sig")

# 사용자가 전체 데이터를 CSV 파일로 내려받을 수 있도록 버튼 표시
st.download_button("전체 데이터 CSV로 다운로드", csv, "시뮬레이션_결과.csv", "text/csv")


# 그래프 호출
st.subheader(f"{select_machine} 이동시 참값 — 마모 누적과 교체, 그리고 고장 시점")

# 같은 시간축을 공유하는 그래프 3개 생성
# => 위에서부터 공구 마모, 토크, 진동을 표시
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 1) 공구 마모 (톱니 모양)
axes[0].plot(
    object_one["ts"], object_one["tool_wear_min"], color="steelblue", linewidth=0.8
)
axes[0].set_ylabel("공구 마모(분)")

# 2) 토크
axes[1].plot(object_one["ts"], object_one["torque_nm"], color="seagreen", linewidth=0.6)
axes[1].set_ylabel("토크(Nm)")

# 3) 진동
axes[2].plot(object_one["ts"], object_one["vibration_mms"], color="peru", linewidth=0.6)
axes[2].set_ylabel("진동(mm/s)")
axes[2].set_xlabel("시각")

# 고장 발생 시점을 빨간 띠로 표시 (모든 서브플롯에 공통 적용)
# => 선택한 설비에서 고장으로 표시된 시각 추출
breakdown_time_point = object_one.loc[object_one["machine_failure"] == 1, "ts"]

# 각 고장 시각부터 1분 동안을 모든 그래프에 빨간 띠로 표시
for ax in axes:
    for t in breakdown_time_point:
        ax.axvspan(t, t + pd.Timedelta(minutes=1), color="red", alpha=0.3)

fig.suptitle(f"{select_machine} 이동시 참값 — 마모 누적과 교체, 그리고 고장 시점")

# 그래프 간격을 자동 조정하고 Streamlit 화면에 출력
plt.tight_layout()
st.pyplot(fig)
