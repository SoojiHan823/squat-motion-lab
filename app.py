import io
import os
import tempfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import mediapipe as mp

from PIL import Image
from scipy.signal import savgol_filter
from streamlit_image_coordinates import streamlit_image_coordinates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="Squat Biomechanics Lab",
    layout="wide"
)

st.title("Squat Biomechanics Lab")
st.caption("시상면 스쿼트 동작분석 | 조건별 3회 | 평균 ± SD")

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_PATH = "/tmp/pose_landmarker_full.task"

if "results" not in st.session_state:
    st.session_state.results = {}

@st.cache_resource
def get_model_path():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            MODEL_URL, MODEL_PATH
        )
    return MODEL_PATH

def read_video(file_bytes):
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as f:
        f.write(file_bytes)
        filename = f.name

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(image)

    cap.release()
    os.remove(filename)

    if not frames or fps <= 0:
        raise ValueError("영상을 읽을 수 없습니다.")

    return frames, float(fps)

def first_frame(file_bytes):
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as f:
        f.write(file_bytes)
        filename = f.name

    cap = cv2.VideoCapture(filename)
    ok, frame = cap.read()
    cap.release()
    os.remove(filename)

    if not ok:
        raise ValueError("첫 프레임을 읽을 수 없습니다.")

    return frame

def pick_point(frame, key):
    h, w = frame.shape[:2]
    shown_w = min(w, 650)
    shown_h = int(h * shown_w / w)

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ).resize((shown_w, shown_h))

    response = streamlit_image_coordinates(
        image, key=key
    )

    if response is None:
        return None

    return np.array([
        response["x"] * w / shown_w,
        response["y"] * h / shown_h
    ], dtype=float)

def calculate_angle(a, b, c):
    u = a - b
    v = c - b

    numerator = np.sum(u * v, axis=1)
    denominator = (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
    )

    cosine = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan),
        where=denominator > 1e-10
    )

    return np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    )

def clean_signal(values, fps):
    s = pd.Series(values, dtype=float)

    s = s.interpolate(
        limit=3, limit_area="inside"
    )

    values = s.to_numpy()
    valid = np.isfinite(values)
    result = values.copy()

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]
    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    window = max(5, int(fps * 0.15) | 1)

    for a, b in zip(starts, ends):
        length = b - a + 1
        win = min(
            window,
            length if length % 2 else length - 1
        )

        if win >= 5:
            result[a:b+1] = savgol_filter(
                values[a:b+1], win, 2
            )

    return result

def calculate_velocity(values, time):
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]
    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    for a, b in zip(starts, ends):
        if b > a:
            result[a:b+1] = np.gradient(
                values[a:b+1], time[a:b+1]
            )

    return result

def track_barbell(frames, initial):
    result = np.full((len(frames), 2), np.nan)
    result[0] = initial

    prev = cv2.cvtColor(
        frames[0], cv2.COLOR_BGR2GRAY
    )
    prev_point = np.array(
        [[initial]], dtype=np.float32
    )

    for i in range(1, len(frames)):
        current = cv2.cvtColor(
            frames[i], cv2.COLOR_BGR2GRAY
        )

        next_point, status, error = (
            cv2.calcOpticalFlowPyrLK(
                prev, current, prev_point, None,
                winSize=(31, 31),
                maxLevel=3
            )
        )

        if (
            next_point is None
            or status is None
            or status[0, 0] != 1
        ):
            break

        point = next_point.reshape(2)
        last = prev_point.reshape(2)

        if np.linalg.norm(point - last) > 80:
            break

        result[i] = point
        prev_point = next_point.reshape(1, 1, 2)
        prev = current

    return result

def analyze_video(
    file_bytes, fps_override,
    scale, bar_point, side
):
    frames, native_fps = read_video(file_bytes)
    fps = fps_override if fps_override > 0 else native_fps

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=get_model_path()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    landmarks = np.full(
        (len(frames), 33, 2), np.nan
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as detector:

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            prediction = detector.detect_for_video(
                image, int(round(i * 1000 / fps))
            )

            if not prediction.pose_landmarks:
                continue

            pose = prediction.pose_landmarks[0]

            for j, p in enumerate(pose):
                if p.visibility >= 0.5:
                    landmarks[i, j] = [
                        p.x * w, p.y * h
                    ]

    time = np.arange(len(frames)) / fps

    ids = {
        "left": [11, 23, 25, 27, 31],
        "right": [12, 24, 26, 28, 32]
    }

    shoulder, hip, knee, ankle, toe = ids[side]

    points = {}
    for j in [shoulder, hip, knee, ankle, toe]:
        points[j] = np.column_stack([
            clean_signal(landmarks[:, j, 0], fps),
            clean_signal(landmarks[:, j, 1], fps)
        ])

    pelvis = np.nanmean(
        landmarks[:, [23, 24], :],
        axis=1
    )

    for axis in range(2):
        pelvis[:, axis] = clean_signal(
            pelvis[:, axis], fps
        )

    barbell = track_barbell(frames, bar_point)

    for axis in range(2):
        barbell[:, axis] = clean_signal(
            barbell[:, axis], fps
        )

    def displacement(p):
        # 영상 X 오른쪽 양수 / 수직 위쪽 양수
        # 최초 유효 프레임을 기준으로 변위 계산
        valid = np.isfinite(p).all(axis=1)
        if not valid.any():
            return np.full_like(p, np.nan)
        origin = p[np.where(valid)[0][0]]
        return np.column_stack([
            (p[:, 0] - origin[0]) * scale,
            -(p[:, 1] - origin[1]) * scale
        ])

    pelvis_d = displacement(pelvis)
    bar_d = displacement(barbell)

    knee_flex = 180 - calculate_angle(
        points[hip], points[knee], points[ankle]
    )
    hip_angle = calculate_angle(
        points[shoulder], points[hip],
        points[knee]
    )
    ankle_angle = calculate_angle(
        points[knee], points[ankle],
        points[toe]
    )

    trunk = points[shoulder] - points[hip]
    trunk_angle = np.degrees(
        np.arctan2(trunk[:, 0], -trunk[:, 1])
    )

    df = pd.DataFrame({
        "frame": np.arange(len(frames)),
        "time_s": time,
        "knee_flexion_deg": knee_flex,
        "hip_segment_deg": hip_angle,
        "ankle_segment_deg": ankle_angle,
        "trunk_inclination_deg": trunk_angle,
        "pelvis_x_m": pelvis_d[:, 0],
        "pelvis_y_m": pelvis_d[:, 1],
        "bar_x_m": bar_d[:, 0],
        "bar_y_m": bar_d[:, 1]
    })

    for name in ["pelvis", "bar"]:
        for axis in ["x", "y"]:
            col = f"{name}_{axis}_m"
            df[f"{name}_v{axis}_ms"] = (
                calculate_velocity(
                    df[col].to_numpy(), time
                )
            )

    vertical = df.pelvis_y_m.to_numpy()

    if np.isfinite(vertical).sum() < 5:
        raise ValueError(
            "골반 좌표가 충분히 검출되지 않았습니다."
        )

    bottom = int(np.nanargmin(vertical))

    if bottom < 2 or bottom > len(df) - 3:
        raise ValueError(
            "최저점 검출 실패. 영상 구간을 확인하세요."
        )

    # 하강 0~50%, 상승 50~100%
    phase = np.empty(len(df))
    phase[:bottom+1] = np.linspace(
        0, 50, bottom+1
    )
    phase[bottom:] = np.linspace(
        50, 100, len(df)-bottom
    )
    df["phase_pct"] = phase

    def data_range(values):
        v = np.asarray(values, dtype=float)
        return (
            np.nanmax(v) - np.nanmin(v)
            if np.isfinite(v).any()
            else np.nan
        )

    def abs_max(values):
        v = np.asarray(values, dtype=float)
        return (
            np.nanmax(np.abs(v))
            if np.isfinite(v).any()
            else np.nan
        )

    summary = {
        "max_knee_flexion_deg": np.nanmax(knee_flex),
        "knee_ROM_deg": data_range(knee_flex),
        "hip_segment_ROM_deg": data_range(hip_angle),
        "ankle_segment_ROM_deg": data_range(ankle_angle),
        "max_trunk_inclination_deg": abs_max(trunk_angle),
        "pelvis_vertical_range_cm":
            data_range(pelvis_d[:, 1]) * 100,
        "pelvis_AP_range_cm":
            data_range(pelvis_d[:, 0]) * 100,
        "bar_vertical_range_cm":
            data_range(bar_d[:, 1]) * 100,
        "bar_AP_range_cm":
            data_range(bar_d[:, 0]) * 100,
        "max_pelvis_vertical_speed_ms":
            abs_max(df.pelvis_vy_ms),
        "max_pelvis_AP_speed_ms":
            abs_max(df.pelvis_vx_ms),
        "max_bar_vertical_speed_ms":
            abs_max(df.bar_vy_ms),
        "max_bar_AP_speed_ms":
            abs_max(df.bar_vx_ms),
        "descent_time_s": time[bottom],
        "ascent_time_s": time[-1] - time[bottom],
        "total_time_s": time[-1]
    }

    return df, summary

# ==============================
# 1. 실험 조건
# ==============================

st.header("1. 실험 조건 설정")

number_conditions = st.selectbox(
    "비교할 조건 수",
    [2, 3],
    index=1
)

condition_names = []

for i in range(number_conditions):
    name = st.text_input(
        f"조건 {chr(65+i)} 이름",
        value=f"조건 {chr(65+i)}",
        key=f"name_{i}"
    )
    condition_names.append(name)

side = st.radio(
    "분석할 다리",
    ["right", "left"],
    format_func=lambda x:
        "오른쪽" if x == "right" else "왼쪽",
    horizontal=True
)

fps_override = st.number_input(
    "영상 FPS (0이면 영상 정보 사용)",
    min_value=0.0,
    value=0.0,
    step=10.0
)

st.info(
    "각 영상에는 스쿼트 1회만 포함하고 "
    "시작과 종료를 선 자세로 촬영하세요."
)

# ==============================
# 2. 거리 보정
# ==============================

st.header("2. 바벨 기준 길이 보정")

st.write(
    "보정 영상에서 실제 길이를 알고 있는 "
    "바벨 구간의 두 점을 선택하세요."
)

calibration_file = st.file_uploader(
    "보정 이미지 업로드",
    type=["jpg", "jpeg", "png"],
    key="calibration"
)

actual_length = st.number_input(
    "선택한 구간의 실제 길이 (m)",
    min_value=0.01,
    value=1.0,
    step=0.05
)

if calibration_file is not None:
    cal_image = cv2.imdecode(
        np.frombuffer(
            calibration_file.getvalue(),
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    st.write("보정점을 클릭하세요.")
    click = pick_point(cal_image, "cal_click")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("첫 번째 점 A 저장"):
            if click is not None:
                st.session_state.cal_A = click

    with c2:
        if st.button("두 번째 점 B 저장"):
            if click is not None:
                st.session_state.cal_B = click

    if (
        "cal_A" in st.session_state
        and "cal_B" in st.session_state
    ):
        pixels = np.linalg.norm(
            st.session_state.cal_A
            - st.session_state.cal_B
        )

        if pixels > 0:
            scale = actual_length / pixels
            st.session_state.scale = scale
            st.session_state.cal_size = (
                cal_image.shape[1],
                cal_image.shape[0]
            )
            st.success(
                f"보정 완료: {scale:.6f} m/pixel"
            )

st.warning(
    "보정 길이는 촬영 시상면과 평행해야 합니다. "
    "보정 이미지와 동작 영상의 해상도, 줌, "
    "카메라 위치가 같아야 합니다."
)

# ==============================
# 3. 영상 업로드 및 분석
# ==============================

st.header("3. 조건별 스쿼트 영상 분석")

for ci, condition in enumerate(condition_names):
    st.subheader(f"조건 {chr(65+ci)}: {condition}")

    videos = st.file_uploader(
        "스쿼트 영상 3개를 업로드하세요.",
        type=["mp4", "mov"],
        accept_multiple_files=True,
        key=f"videos_{ci}"
    )

    if len(videos) != 3:
        st.info("이 조건의 영상 3개를 선택하세요.")
        continue

    for trial, video in enumerate(videos, 1):
        with st.expander(
            f"{condition} - {trial}회차",
            expanded=False
        ):
            try:
                frame = first_frame(video.getvalue())
            except Exception as exc:
                st.error(str(exc))
                continue

            st.write("영상에서 바벨 추적점을 클릭하세요.")
            bar_point = pick_point(
                frame,
                f"bar_click_{ci}_{trial}"
            )

            if bar_point is not None:
                st.write(
                    f"선택 좌표: "
                    f"X={bar_point[0]:.1f}, "
                    f"Y={bar_point[1]:.1f}"
                )

            if st.button(
                "이 영상 분석하기",
                key=f"analyze_{ci}_{trial}"
            ):
                if "scale" not in st.session_state:
                    st.error("먼저 거리 보정을 완료하세요.")
                    continue

                if bar_point is None:
                    st.error("바벨 위치를 클릭하세요.")
                    continue

                current_size = (
                    frame.shape[1], frame.shape[0]
                )
                if current_size != st.session_state.cal_size:
                    st.error(
                        "보정 이미지와 영상의 해상도가 "
                        "다릅니다. 동일 해상도로 보정하세요."
                    )
                    continue

                with st.spinner("운동학 분석 중..."):
                    try:
                        df, summary = analyze_video(
                            video.getvalue(),
                            fps_override,
                            st.session_state.scale,
                            bar_point,
                            side
                        )

                        summary["condition"] = condition
                        summary["trial"] = trial

                        st.session_state.results[
                            (ci, trial)
                        ] = (df, summary)

                        st.success("분석 완료!")

                    except Exception as exc:
                        st.error(f"분석 오류: {exc}")

# ==============================
# 4. 결과표
# ==============================

st.header("4. 운동학 분석 결과")

results = st.session_state.results

valid_results = [
    (key, df, summary)
    for key, (df, summary) in results.items()
    if key[0] < number_conditions
]

if valid_results:
    summaries = pd.DataFrame([
        s for _, _, s in valid_results
    ])

    st.subheader("반복별 분석 결과")
    st.dataframe(
        summaries.round(3),
        use_container_width=True
    )

    variables = [
        c for c in summaries.select_dtypes(
            include="number"
        ).columns if c != "trial"
    ]

    grouped = summaries.groupby(
        "condition"
    )[variables].agg(["mean", "std", "count"])

    st.subheader("조건별 평균 ± SD")
    st.dataframe(
        grouped.round(3),
        use_container_width=True
    )

    # ==============================
    # 5. 평균 ± SD 막대그래프
    # ==============================

    st.header("5. 조건별 평균 ± SD 그래프")

    selected_metric = st.selectbox(
        "비교할 운동학 변수",
        variables,
        key="summary_metric"
    )

    fig = go.Figure()

    for condition in condition_names:
        vals = summaries.loc[
            summaries.condition == condition,
            selected_metric
        ].dropna().to_numpy()

        if not len(vals):
            continue

        mean = vals.mean()
        sd = (
            vals.std(ddof=1)
            if len(vals) >= 2 else 0
        )

        fig.add_trace(go.Bar(
            x=[condition],
            y=[mean],
            name=condition,
            error_y=dict(
                type="data",
                array=[sd],
                visible=True
            ),
            text=[
                f"{mean:.2f} ± {sd:.2f}"
            ],
            textposition="outside"
        ))

        fig.add_trace(go.Scatter(
            x=[condition] * len(vals),
            y=vals,
            mode="markers",
            marker=dict(
                color="black",
                symbol="circle-open",
                size=10
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=f"{selected_metric}: Mean ± SD",
        xaxis_title="Condition",
        yaxis_title=selected_metric,
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==============================
    # 6. 시간 정규화 그래프
    # ==============================

    st.header("6. 시간 정규화 그래프")

    available_curves = [
        "knee_flexion_deg",
        "hip_segment_deg",
        "ankle_segment_deg",
        "trunk_inclination_deg",
        "pelvis_x_m",
        "pelvis_y_m",
        "pelvis_vx_ms",
        "pelvis_vy_ms",
        "bar_x_m",
        "bar_y_m",
        "bar_vx_ms",
        "bar_vy_ms"
    ]

    curve_metric = st.selectbox(
        "시간에 따른 분석 변수",
        available_curves
    )

    phase_grid = np.linspace(0, 100, 101)
    curve_fig = go.Figure()

    colors = [
        ("rgb(45,110,210)", "rgba(45,110,210,.15)"),
        ("rgb(30,150,100)", "rgba(30,150,100,.15)"),
        ("rgb(220,135,40)", "rgba(220,135,40,.15)")
    ]

    for ci, condition in enumerate(condition_names):
        curves = []

        for key, df, summ in valid_results:
            if key[0] != ci:
                continue

            x = df.phase_pct.to_numpy()
            y = df[curve_metric].to_numpy()

            good = np.isfinite(x) & np.isfinite(y)

            if good.mean() < 0.90:
                continue

            curves.append(
                np.interp(
                    phase_grid, x[good], y[good]
                )
            )

        if not curves:
            continue

        curves = np.array(curves)
        mean_curve = curves.mean(axis=0)

        sd_curve = (
            curves.std(axis=0, ddof=1)
            if len(curves) > 1
            else np.zeros_like(mean_curve)
        )

        line_color, fill_color = colors[ci]

        curve_fig.add_trace(go.Scatter(
            x=phase_grid,
            y=mean_curve + sd_curve,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        curve_fig.add_trace(go.Scatter(
            x=phase_grid,
            y=mean_curve - sd_curve,
            mode="lines",
            fill="tonexty",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False
        ))

        curve_fig.add_trace(go.Scatter(
            x=phase_grid,
            y=mean_curve,
            mode="lines",
            name=f"{condition} (n={len(curves)})",
            line=dict(
                color=line_color, width=3
            )
        ))

    curve_fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="gray"
    )

    curve_fig.update_layout(
        title=f"{curve_metric}: Mean ± SD",
        xaxis_title=(
            "Squat phase (%) "
            "- 0: Start, 50: Bottom, 100: End"
        ),
        yaxis_title=curve_metric,
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        curve_fig,
        use_container_width=True
    )

    # ==============================
    # 7. 바벨 궤적
    # ==============================

    st.header("7. 바벨 전후·수직 이동 궤적")

    trajectory_fig = go.Figure()

    for key, df, summ in valid_results:
        trajectory_fig.add_trace(go.Scatter(
            x=df.bar_x_m * 100,
            y=df.bar_y_m * 100,
            mode="lines",
            name=(
                f"{summ['condition']} "
                f"{summ['trial']}회"
            )
        ))

    trajectory_fig.update_layout(
        xaxis_title="전후 변위 (cm)",
        yaxis_title="수직 변위 (cm)",
        template="plotly_white"
    )

    st.plotly_chart(
        trajectory_fig,
        use_container_width=True
    )

    # ==============================
    # 8. Excel 다운로드
    # ==============================

    st.header("8. 분석 결과 다운로드")

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        summaries.to_excel(
            writer, "Trial_Results", index=False
        )

        grouped.to_excel(
            writer, "Mean_SD"
        )

        for key, df, summ in valid_results:
            sheet = f"C{key[0]+1}_T{key[1]}"
            df.to_excel(
                writer,
                sheet_name=sheet,
                index=False
            )

    st.download_button(
        "Excel 분석 결과 다운로드",
        data=excel_buffer.getvalue(),
        file_name="Squat_Biomechanics_Results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

else:
    st.info(
        "먼저 각 조건의 영상을 업로드하고 "
        "분석 버튼을 눌러주세요."
    )
