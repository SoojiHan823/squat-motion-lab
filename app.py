
import io
import os
import hashlib
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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 기본 설정
# ==========================================

st.set_page_config(
    page_title="Squat Biomechanics Lab",
    layout="wide"
)

st.title("Squat Biomechanics Lab")
st.caption(
    "시상면 2D 스쿼트 분석 | 영상별 거리 보정 | "
    "조건별 평균 ± 표준편차"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_FILE = "/tmp/squat_pose_landmarker.task"

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "points" not in st.session_state:
    st.session_state.points = {}

@st.cache_resource
def prepare_model():
    if not os.path.isfile(MODEL_FILE):
        urllib.request.urlretrieve(
            MODEL_URL, MODEL_FILE
        )
    return MODEL_FILE

# ==========================================
# 영상 처리
# ==========================================

@st.cache_data(show_spinner=False, max_entries=12)
def read_video(video_bytes, extension):
    with tempfile.NamedTemporaryFile(
        suffix=extension, delete=False
    ) as f:
        f.write(video_bytes)
        filename = f.name

    frames = []
    cap = cv2.VideoCapture(filename)

    try:
        if not cap.isOpened():
            raise ValueError("영상 파일을 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            if len(frames) > 1200:
                raise ValueError(
                    "영상이 너무 깁니다. "
                    "20초 내외의 짧은 영상으로 준비하세요."
                )
    finally:
        cap.release()
        os.remove(filename)

    if not frames:
        raise ValueError("영상 프레임이 없습니다.")

    return frames, float(fps)

def select_point(frame, key, marks=None):
    image = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    ).copy()

    if marks:
        for label, pt, color in marks:
            if pt is None:
                continue
            p = tuple(np.asarray(pt, dtype=int))
            cv2.circle(image, p, 6, color, -1)
            cv2.putText(
                image, label,
                (p[0]+8, p[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

    h, w = image.shape[:2]
    shown_w = min(w, 600)
    shown_h = max(1, round(h*shown_w/w))

    display_image = Image.fromarray(image).resize(
        (shown_w, shown_h)
    )

    clicked = streamlit_image_coordinates(
        display_image, key=key
    )

    if clicked is None:
        return None

    return np.array([
        clicked["x"]*w/shown_w,
        clicked["y"]*h/shown_h
    ], dtype=float)

# ==========================================
# 수치 계산
# ==========================================

def smooth_signal(values, fps):
    values = pd.Series(
        values, dtype=float
    ).interpolate(
        limit=3,
        limit_area="inside"
    ).to_numpy()

    valid = np.isfinite(values)
    result = values.copy()

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]
    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    window = max(5, int(fps*0.15) | 1)

    for a, b in zip(starts, ends):
        n = b-a+1
        win = min(
            window, n if n%2 else n-1
        )

        if win >= 5:
            result[a:b+1] = savgol_filter(
                values[a:b+1], win, 2
            )

    return result

def joint_angle(a, b, c):
    u = a-b
    v = c-b

    numerator = np.sum(u*v, axis=1)
    denominator = (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
    )

    cosine = np.divide(
        numerator, denominator,
        out=np.full(len(numerator), np.nan),
        where=denominator>1e-10
    )

    return np.degrees(
        np.arccos(np.clip(cosine,-1,1))
    )

def differentiate(values, time):
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]
    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    for a, b in zip(starts, ends):
        if b-a >= 2:
            result[a:b+1] = np.gradient(
                values[a:b+1],
                time[a:b+1]
            )
    return result

def safe_range(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).any():
        return np.nan
    return float(np.nanmax(a)-np.nanmin(a))

def safe_max(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).any():
        return np.nan
    return float(np.nanmax(a))

def safe_absmax(values):
    return safe_max(np.abs(values))

# ==========================================
# MediaPipe 관절 추출
# ==========================================

@st.cache_data(show_spinner=False, max_entries=12)
def detect_landmarks(video_bytes, extension, fps):
    frames, _ = read_video(
        video_bytes, extension
    )

    landmarks = np.full(
        (len(frames),33,2), np.nan
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=prepare_model()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as detector:

        for i, frame in enumerate(frames):
            h,w = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            result = detector.detect_for_video(
                image, int(round(i*1000/fps))
            )

            if not result.pose_landmarks:
                continue

            for j, p in enumerate(
                result.pose_landmarks[0]
            ):
                if p.visibility >= 0.5:
                    landmarks[i,j] = [
                        p.x*w, p.y*h
                    ]

    return landmarks

# ==========================================
# 바벨 추적
# ==========================================

def track_barbell(frames, start_point):
    result = np.full(
        (len(frames),2), np.nan
    )

    result[0] = start_point

    prev_gray = cv2.cvtColor(
        frames[0], cv2.COLOR_BGR2GRAY
    )

    prev_pt = np.array(
        [[start_point]], dtype=np.float32
    )

    for i in range(1,len(frames)):
        current_gray = cv2.cvtColor(
            frames[i], cv2.COLOR_BGR2GRAY
        )

        new_pt, status, error = (
            cv2.calcOpticalFlowPyrLK(
                prev_gray,
                current_gray,
                prev_pt,
                None,
                winSize=(31,31),
                maxLevel=3
            )
        )

        if new_pt is None or status is None:
            break

        if status[0,0] != 1:
            break

        point = new_pt.reshape(2)

        if np.linalg.norm(
            point-prev_pt.reshape(2)
        ) > 80:
            break

        result[i] = point
        prev_pt = new_pt.reshape(1,1,2)
        prev_gray = current_gray

    return result

# ==========================================
# 운동학 분석 엔진
# ==========================================

def analyze_squat(
    video_bytes, extension,
    fps, scale, side,
    start, bottom, end, bar_point
):
    frames,_ = read_video(
        video_bytes,extension
    )

    if not 0 <= start < bottom < end < len(frames):
        raise ValueError("프레임 순서를 확인하세요.")

    pose = detect_landmarks(
        video_bytes,extension,fps
    )[start:end+1].copy()

    selected_frames = frames[start:end+1]

    n = len(selected_frames)
    time = np.arange(n)/fps

    ids = {
        "left":[11,23,25,27,31],
        "right":[12,24,26,28,32]
    }

    shoulder,hip,knee,ankle,toe = ids[side]

    p = {}
    for j in [shoulder,hip,knee,ankle,toe]:
        p[j] = np.column_stack([
            smooth_signal(pose[:,j,0],fps),
            smooth_signal(pose[:,j,1],fps)
        ])

    pelvis = np.full((n,2),np.nan)

    both = np.isfinite(
        pose[:,[23,24],:]
    ).all(axis=(1,2))

    pelvis[both] = (
        pose[both,23,:]
        + pose[both,24,:]
    )/2

    for axis in range(2):
        pelvis[:,axis] = smooth_signal(
            pelvis[:,axis],fps
        )

    bar = track_barbell(
        selected_frames,
        np.asarray(bar_point,dtype=float)
    )

    for axis in range(2):
        bar[:,axis] = smooth_signal(
            bar[:,axis],fps
        )

    def displacement(points):
        result = np.full_like(
            points,np.nan
        )
        good = np.isfinite(points).all(axis=1)

        if not good.any():
            return result

        origin = points[np.where(good)[0][0]]

        result[:,0] = (
            points[:,0]-origin[0]
        )*scale

        result[:,1] = -(
            points[:,1]-origin[1]
        )*scale

        return result

    pelvis_d = displacement(pelvis)
    bar_d = displacement(bar)

    knee_flexion = 180-joint_angle(
        p[hip],p[knee],p[ankle]
    )

    hip_angle = joint_angle(
        p[shoulder],p[hip],p[knee]
    )

    ankle_angle = joint_angle(
        p[knee],p[ankle],p[toe]
    )

    trunk = p[shoulder]-p[hip]

    trunk_angle = np.degrees(
        np.arctan2(
            trunk[:,0],-trunk[:,1]
        )
    )

    df = pd.DataFrame({
        "frame":np.arange(start,end+1),
        "time_s":time,
        "knee_flexion_deg":knee_flexion,
        "hip_segment_deg":hip_angle,
        "ankle_segment_deg":ankle_angle,
        "trunk_inclination_deg":trunk_angle,
        "pelvis_AP_m":pelvis_d[:,0],
        "pelvis_vertical_m":pelvis_d[:,1],
        "bar_AP_m":bar_d[:,0],
        "bar_vertical_m":bar_d[:,1]
    })

    for name in ["pelvis","bar"]:
        for axis in ["AP","vertical"]:
            position = f"{name}_{axis}_m"
            velocity = f"{name}_{axis}_velocity_ms"

            df[velocity] = differentiate(
                df[position].to_numpy(),time
            )

    bottom_local = bottom-start

    phase = np.empty(n)
    phase[:bottom_local+1] = np.linspace(
        0,50,bottom_local+1
    )
    phase[bottom_local:] = np.linspace(
        50,100,n-bottom_local
    )

    df["phase_pct"] = phase

    summary = {
        "max_knee_flexion_deg":
            safe_max(knee_flexion),
        "knee_ROM_deg":
            safe_range(knee_flexion),
        "hip_segment_ROM_deg":
            safe_range(hip_angle),
        "ankle_segment_ROM_deg":
            safe_range(ankle_angle),
        "max_trunk_inclination_deg":
            safe_absmax(trunk_angle),

        "pelvis_AP_range_cm":
            safe_range(pelvis_d[:,0])*100,
        "pelvis_vertical_range_cm":
            safe_range(pelvis_d[:,1])*100,
        "bar_AP_range_cm":
            safe_range(bar_d[:,0])*100,
        "bar_vertical_range_cm":
            safe_range(bar_d[:,1])*100,

        "max_pelvis_AP_velocity_ms":
            safe_absmax(
                df.pelvis_AP_velocity_ms
            ),
        "max_pelvis_vertical_velocity_ms":
            safe_absmax(
                df.pelvis_vertical_velocity_ms
            ),
        "max_bar_AP_velocity_ms":
            safe_absmax(
                df.bar_AP_velocity_ms
            ),
        "max_bar_vertical_velocity_ms":
            safe_absmax(
                df.bar_vertical_velocity_ms
            ),

        "descent_time_s":bottom_local/fps,
        "ascent_time_s":(end-bottom)/fps,
        "total_time_s":(end-start)/fps,

        "bar_tracking_pct":
            np.isfinite(bar[:,0]).mean()*100,
        "pelvis_valid_pct":
            np.isfinite(pelvis[:,0]).mean()*100
    }

    return df,summary

# ==========================================
# 웹앱 1. 조건 설정
# ==========================================

st.header("1. 실험 조건 설정")

condition_count = st.selectbox(
    "비교 조건 수",
    [2,3],
    index=1
)

conditions = []

for i in range(condition_count):
    conditions.append(
        st.text_input(
            f"조건 {chr(65+i)} 이름",
            value=f"조건 {chr(65+i)}",
            key=f"condition_{i}"
        )
    )

side = st.radio(
    "분석할 다리",
    ["right","left"],
    horizontal=True,
    format_func=lambda x:
        "오른쪽" if x=="right" else "왼쪽"
)

actual_length = st.number_input(
    "영상 속 바벨 A-B 실제 거리 (m)",
    min_value=0.01,
    value=1.0,
    step=0.05
)

fps_input = st.number_input(
    "FPS 직접 입력 (0이면 영상 정보 사용)",
    min_value=0.0,
    value=0.0,
    step=10.0
)

st.info(
    "각 영상 앞부분에 보정 장면이 있으며, "
    "분석할 스쿼트는 영상당 1회라고 가정합니다."
)

# ==========================================
# 웹앱 2. 영상별 분석
# ==========================================

st.header("2. 영상 업로드 및 분석")

current_results = []

for ci in range(condition_count):
    st.subheader(
        f"조건 {chr(65+ci)} — {conditions[ci]}"
    )

    for trial in range(1,4):
        slot = f"{ci}_{trial}"

        with st.expander(
            f"{trial}회차 분석",
            expanded=False
        ):
            uploaded = st.file_uploader(
                "스쿼트 동영상 업로드",
                type=["mp4","mov"],
                key=f"upload_{slot}"
            )

            if uploaded is None:
                st.info("영상을 업로드하세요.")
                continue

            raw = uploaded.getvalue()
            extension = os.path.splitext(
                uploaded.name
            )[1].lower() or ".mp4"

            digest = hashlib.sha256(
                raw
            ).hexdigest()[:12]

            identity = f"{slot}_{digest}"

            try:
                frames,video_fps = read_video(
                    raw,extension
                )
            except Exception as e:
                st.error(str(e))
                continue

            fps = (
                fps_input
                if fps_input>0 else video_fps
            )

            st.caption(
                f"전체 프레임: {len(frames)} | "
                f"FPS: {fps:.2f}"
            )

            # ---- A. 거리 보정 ----

            st.markdown("#### A. 바벨 거리 보정")

            cal_frame = st.slider(
                "바벨 길이가 보이는 프레임",
                0,len(frames)-1,0,
                key=f"cal_{identity}"
            )

            point_base = f"{identity}_{cal_frame}"
            Akey = f"A_{point_base}"
            Bkey = f"B_{point_base}"

            A = st.session_state.points.get(Akey)
            B = st.session_state.points.get(Bkey)

            clicked = select_point(
                frames[cal_frame],
                f"cal_click_{point_base}",
                [
                    ("A",A,(255,80,0)),
                    ("B",B,(0,180,255))
                ]
            )

            left,right = st.columns(2)

            with left:
                if st.button(
                    "A점 저장",
                    key=f"save_A_{point_base}"
                ):
                    if clicked is not None:
                        st.session_state.points[Akey] = clicked
                        st.rerun()

            with right:
                if st.button(
                    "B점 저장",
                    key=f"save_B_{point_base}"
                ):
                    if clicked is not None:
                        st.session_state.points[Bkey] = clicked
                        st.rerun()

            scale = None

            if A is not None and B is not None:
                distance = np.linalg.norm(A-B)

                if distance > 1:
                    scale = actual_length/distance
                    st.success(
                        f"보정 거리 {distance:.1f}px | "
                        f"{scale:.6f} m/pixel"
                    )

            # ---- B. 분석 구간 ----

            st.markdown("#### B. 스쿼트 구간 선택")

            if len(frames)<5:
                st.warning("영상 길이가 너무 짧습니다.")
                continue

            start,end = st.slider(
                "시작 ~ 종료 프레임",
                0,len(frames)-1,
                (0,len(frames)-1),
                key=f"range_{identity}"
            )

            if end-start<4:
                st.warning("분석 구간을 넓혀주세요.")
                continue

            bottom = st.slider(
                "스쿼트 최저점 프레임",
                start+1,end-1,
                value=(start+end)//2,
                key=f"bottom_{identity}"
            )

            preview_frame = st.select_slider(
                "자세 확인 프레임",
                options=range(start,end+1),
                value=bottom,
                key=f"preview_{identity}"
            )

            st.image(
                cv2.cvtColor(
                    frames[preview_frame],
                    cv2.COLOR_BGR2RGB
                ),
                width=480,
                caption=f"프레임 {preview_frame}"
            )

            # ---- C. 바벨 추적점 ----

            st.markdown("#### C. 바벨 추적점 선택")

            st.caption(
                "스쿼트 시작 프레임에서 "
                "바벨의 동일한 물리적 점을 클릭하세요."
            )

            bar_key = f"bar_{identity}_{start}"
            bar_point = st.session_state.points.get(
                bar_key
            )

            bar_click = select_point(
                frames[start],
                f"bar_click_{identity}_{start}",
                [("Bar",bar_point,(255,90,0))]
            )

            if st.button(
                "바벨 추적점 저장",
                key=f"save_bar_{identity}_{start}"
            ):
                if bar_click is not None:
                    st.session_state.points[
                        bar_key
                    ] = bar_click
                    st.rerun()

            if bar_point is not None:
                st.success(
                    f"바벨 X={bar_point[0]:.1f}, "
                    f"Y={bar_point[1]:.1f}"
                )

            # ---- D. 실행 ----

            ready = (
                scale is not None
                and bar_point is not None
            )

            settings = (
                digest, fps, scale,
                start,bottom,end,side,
                tuple(bar_point)
                if bar_point is not None else None
            )

            if st.button(
                "이 영상 운동학 분석",
                key=f"run_{identity}",
                disabled=not ready,
                type="primary"
            ):
                try:
                    with st.spinner("영상 분석 중..."):
                        df,metrics = analyze_squat(
                            raw,extension,
                            fps,scale,side,
                            start,bottom,end,bar_point
                        )

                    st.session_state.analysis_results[
                        slot
                    ] = {
                        "settings":settings,
                        "data":df,
                        "metrics":metrics
                    }

                    st.success("분석 완료!")

                except Exception as e:
                    st.error(
                        f"분석 오류: {type(e).__name__}: {e}"
                    )

            saved = st.session_state.analysis_results.get(
                slot
            )

            if saved and saved["settings"] == settings:
                st.info(
                    f"바벨 추적 성공률: "
                    f"{saved['metrics']['bar_tracking_pct']:.1f}%"
                )

                current_results.append({
                    "condition":conditions[ci],
                    "condition_id":ci+1,
                    "trial":trial,
                    "data":saved["data"],
                    "metrics":saved["metrics"]
                })

# ==========================================
# 웹앱 3. 결과 비교
# ==========================================

st.header("3. 조건별 운동학 결과")

if not current_results:
    st.info(
        "영상별 분석을 완료하면 "
        "조건 비교 결과가 나타납니다."
    )
    st.stop()

rows = []

for item in current_results:
    row = {
        "condition":item["condition"],
        "condition_id":item["condition_id"],
        "trial":item["trial"]
    }
    row.update(item["metrics"])
    rows.append(row)

results_df = pd.DataFrame(rows)

numeric = [
    c for c in results_df.select_dtypes(
        include="number"
    ).columns
    if c not in ["condition_id","trial"]
]

st.subheader("개별 시행 운동학 지표")
st.dataframe(
    results_df.round(3),
    width="stretch"
)

statistics = results_df.groupby(
    ["condition_id","condition"]
)[numeric].agg(["mean","std","count"])

st.subheader("조건별 평균 ± SD")
st.dataframe(
    statistics.round(3),
    width="stretch"
)

st.caption(
    "SD는 조건별 반복 시행에 대한 표본 표준편차입니다. "
    "한 조건에 시행이 1개만 있으면 SD는 계산되지 않습니다."
)

# ==========================================
# 웹앱 4. 막대그래프
# ==========================================

st.header("4. 조건별 평균 ± SD 그래프")

selected_metric = st.selectbox(
    "비교 지표 선택",
    numeric
)

fig = go.Figure()
colors = ["royalblue","seagreen","darkorange"]

for ci,name in enumerate(conditions):
    values = results_df.loc[
        results_df.condition_id == ci+1,
        selected_metric
    ].dropna().to_numpy()

    if not len(values):
        continue

    mean = values.mean()
    sd = (
        values.std(ddof=1)
        if len(values)>1
        else np.nan
    )

    fig.add_trace(go.Bar(
        x=[name],
        y=[mean],
        name=name,
        marker_color=colors[ci],
        error_y=dict(
            type="data",
            array=[sd if np.isfinite(sd) else 0],
            visible=True
        ),
        text=[
            f"{mean:.2f} ± {sd:.2f}"
            if np.isfinite(sd)
            else f"{mean:.2f} (SD 없음)"
        ],
        textposition="outside"
    ))

    fig.add_trace(go.Scatter(
        x=[name]*len(values),
        y=values,
        mode="markers",
        marker=dict(
            color="black",
            symbol="circle-open",
            size=10
        ),
        showlegend=False
    ))

fig.update_layout(
    title=f"{selected_metric} — Mean ± SD",
    template="plotly_white",
    height=480
)

st.plotly_chart(
    fig,width="stretch"
)

# ==========================================
# 웹앱 5. 시간 정규화 그래프
# ==========================================

st.header("5. 스쿼트 주기 정규화 곡선")

curve_columns = [
    c for c in current_results[0]["data"].columns
    if c not in ["frame","time_s","phase_pct"]
]

selected_curve = st.selectbox(
    "곡선 변수 선택",
    curve_columns
)

phase_grid = np.linspace(0,100,101)
curve_fig = go.Figure()

fills = [
    "rgba(65,105,225,0.16)",
    "rgba(46,139,87,0.16)",
    "rgba(255,140,0,0.16)"
]

for ci,name in enumerate(conditions):
    curves = []

    for item in current_results:
        if item["condition_id"] != ci+1:
            continue

        df = item["data"]
        x = df.phase_pct.to_numpy()
        y = df[selected_curve].to_numpy()

        good = np.isfinite(x)&np.isfinite(y)

        if good.sum()<5 or good.mean()<0.9:
            continue

        curves.append(
            np.interp(
                phase_grid,
                x[good],y[good]
            )
        )

    if not curves:
        continue

    curves = np.asarray(curves)
    mean_curve = curves.mean(axis=0)

    sd = (
        curves.std(axis=0,ddof=1)
        if len(curves)>1
        else np.zeros_like(mean_curve)
    )

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,y=mean_curve+sd,
        line=dict(width=0),
        showlegend=False
    ))

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,y=mean_curve-sd,
        fill="tonexty",
        fillcolor=fills[ci],
        line=dict(width=0),
        showlegend=False
    ))

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,y=mean_curve,
        name=f"{name} (n={len(curves)})",
        line=dict(
            color=colors[ci],
            width=3
        )
    ))

curve_fig.add_vline(
    x=50,
    line_dash="dash",
    line_color="gray"
)

curve_fig.update_layout(
    title=f"{selected_curve}: Mean ± SD",
    xaxis_title=(
        "Movement phase (%) "
        "0=시작, 50=최저점, 100=종료"
    ),
    yaxis_title=selected_curve,
    template="plotly_white",
    height=500
)

st.plotly_chart(
    curve_fig,width="stretch"
)

# ==========================================
# 웹앱 6. 이동 궤적
# ==========================================

st.header("6. 바벨 및 골반 이동 궤적")

target = st.radio(
    "분석 대상",
    ["bar","pelvis"],
    horizontal=True,
    format_func=lambda x:
        "바벨" if x=="bar" else "골반 중심"
)

trajectory = go.Figure()

for item in current_results:
    df = item["data"]

    trajectory.add_trace(go.Scatter(
        x=df[f"{target}_AP_m"]*100,
        y=df[f"{target}_vertical_m"]*100,
        mode="lines+markers",
        marker=dict(size=3),
        name=(
            f"{item['condition']} "
            f"{item['trial']}회"
        )
    ))

trajectory.update_layout(
    xaxis_title="화면 기준 전후 변위 (cm)",
    yaxis_title="수직 변위 (cm)",
    template="plotly_white"
)

trajectory.update_yaxes(
    scaleanchor="x",
    scaleratio=1
)

st.plotly_chart(
    trajectory,width="stretch"
)

# ==========================================
# 웹앱 7. Excel 다운로드
# ==========================================

st.header("7. 분석 데이터 다운로드")

buffer = io.BytesIO()

with pd.ExcelWriter(
    buffer,engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,"Trials",index=False
    )

    statistics.to_excel(
        writer,"Mean_SD"
    )

    for item in current_results:
        sheet = (
            f"C{item['condition_id']}_"
            f"T{item['trial']}"
        )
        item["data"].to_excel(
            writer,sheet,index=False
        )

st.download_button(
    "Excel 전체 결과 다운로드",
    data=buffer.getvalue(),
    file_name="Squat_Analysis_Results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    type="primary"
)

st.warning(
    "연구 해석 주의: 골반 중심은 전신 COM이 아니며, "
    "관절각도는 2D 투영각입니다. "
    "바벨과 골반의 거리·속도는 보정계수와 "
    "추적 정확도에 따라 오차가 발생할 수 있습니다."
)
