
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
from streamlit_image_coordinates import streamlit_image_coordinates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="Squat Biomechanics Lab",
    layout="wide"
)

st.title("Squat Biomechanics Lab")
st.caption(
    "2D 시상면 스쿼트 운동학 분석 | "
    "영상별 바벨 길이 보정 | 조건별 평균 ± SD"
)

MODEL_PATH = "/tmp/pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

if "results" not in st.session_state:
    st.session_state.results = {}

@st.cache_resource
def model_path():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

@st.cache_data(show_spinner=False, max_entries=12)
def load_video(file_bytes, suffix=".mp4"):
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    ) as f:
        f.write(file_bytes)
        path = f.name

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
        os.unlink(path)

    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("영상 읽기 실패 또는 FPS 정보 오류")

    return frames, float(fps)

def show_click_image(frame, key, marks=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    if marks:
        for label, point, color in marks:
            if point is not None:
                p = tuple(np.round(point).astype(int))
                cv2.circle(rgb, p, 7, color, -1)
                cv2.putText(
                    rgb, label, (p[0]+8, p[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )

    shown_w = min(w, 650)
    shown_h = round(h * shown_w / w)

    pil = Image.fromarray(rgb).resize(
        (shown_w, shown_h)
    )

    response = streamlit_image_coordinates(
        pil, key=key
    )

    if response is None:
        return None

    return np.array([
        response["x"] * w / shown_w,
        response["y"] * h / shown_h
    ], dtype=float)

def filter_signal(values, fps):
    arr = pd.Series(
        values, dtype=float
    ).interpolate(
        limit=3, limit_area="inside"
    ).to_numpy()

    good = np.isfinite(arr)
    starts = np.where(
        good & ~np.r_[False, good[:-1]]
    )[0]
    ends = np.where(
        good & ~np.r_[good[1:], False]
    )[0]

    result = arr.copy()
    window = max(5, int(fps * 0.15) | 1)

    for a, b in zip(starts, ends):
        length = b-a+1
        win = min(
            window,
            length if length % 2 else length-1
        )
        if win >= 5:
            result[a:b+1] = savgol_filter(
                arr[a:b+1], win, 2
            )
    return result

def angle_2d(a, b, c):
    u = a-b
    v = c-b
    dot = np.sum(u*v, axis=1)
    length = (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
    )
    cos = np.divide(
        dot, length,
        out=np.full(len(dot), np.nan),
        where=length > 1e-10
    )
    return np.degrees(
        np.arccos(np.clip(cos, -1, 1))
    )

def velocity(values, time):
    arr = np.asarray(values, dtype=float)
    result = np.full(len(arr), np.nan)

    good = np.isfinite(arr)
    starts = np.where(
        good & ~np.r_[False, good[:-1]]
    )[0]
    ends = np.where(
        good & ~np.r_[good[1:], False]
    )[0]

    for a, b in zip(starts, ends):
        if b > a:
            result[a:b+1] = np.gradient(
                arr[a:b+1], time[a:b+1]
            )
    return result

def get_range(values):
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return np.nan
    return float(
        np.nanmax(arr)-np.nanmin(arr)
    )

def get_max(values, absolute=False):
    arr = np.asarray(values, dtype=float)
    if absolute:
        arr = np.abs(arr)
    if not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmax(arr))

@st.cache_data(show_spinner=False, max_entries=12)
def extract_pose(file_bytes, suffix, fps):
    frames, native_fps = load_video(
        file_bytes, suffix
    )

    coords = np.full(
        (len(frames), 33, 2), np.nan
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=model_path()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
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

            result = detector.detect_for_video(
                image, int(round(i*1000/fps))
            )

            if not result.pose_landmarks:
                continue

            for j, p in enumerate(
                result.pose_landmarks[0]
            ):
                if p.visibility >= 0.5:
                    coords[i,j] = [
                        p.x*w, p.y*h
                    ]

    return coords

def track_bar(frames, start_point):
    points = np.full(
        (len(frames), 2), np.nan
    )
    points[0] = start_point

    prev_gray = cv2.cvtColor(
        frames[0], cv2.COLOR_BGR2GRAY
    )
    prev_pt = np.array(
        [[start_point]], dtype=np.float32
    )

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(
            frames[i], cv2.COLOR_BGR2GRAY
        )

        nxt, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pt, None,
            winSize=(31,31), maxLevel=3
        )

        if nxt is None or status is None:
            break
        if status[0,0] != 1:
            break

        p = nxt.reshape(2)
        if not np.isfinite(p).all():
            break
        if np.linalg.norm(
            p-prev_pt.reshape(2)
        ) > 80:
            break

        points[i] = p
        prev_pt = nxt.reshape(1,1,2)
        prev_gray = gray

    return points

def analyze(
    file_bytes, suffix, fps, scale,
    start, bottom, end, bar_point, side
):
    frames, _ = load_video(
        file_bytes, suffix
    )

    if not 0 <= start < bottom < end < len(frames):
        raise ValueError(
            "시작 < 최저점 < 종료 순서로 설정하세요."
        )

    coords = extract_pose(
        file_bytes, suffix, fps
    )[start:end+1].copy()

    segment_frames = frames[start:end+1]
    n = len(segment_frames)
    t = np.arange(n)/fps

    ids = {
        "left": [11,23,25,27,31],
        "right": [12,24,26,28,32]
    }
    sh, hip, knee, ankle, toe = ids[side]

    points = {}
    for j in [sh, hip, knee, ankle, toe]:
        points[j] = np.column_stack([
            filter_signal(coords[:,j,0], fps),
            filter_signal(coords[:,j,1], fps)
        ])

    # 골반 중심 = 좌우 고관절 중점
    pelvis = np.full((n,2), np.nan)
    both = np.isfinite(
        coords[:,[23,24],:]
    ).all(axis=(1,2))

    pelvis[both] = (
        coords[both,23,:]
        + coords[both,24,:]
    )/2

    for axis in range(2):
        pelvis[:,axis] = filter_signal(
            pelvis[:,axis], fps
        )

    bar = track_bar(
        segment_frames,
        np.asarray(bar_point, dtype=float)
    )
    for axis in range(2):
        bar[:,axis] = filter_signal(
            bar[:,axis], fps
        )

    def displacement(p):
        out = np.full_like(p, np.nan)
        good = np.isfinite(p).all(axis=1)

        if not good.any():
            return out

        first = np.where(good)[0][0]
        out[:,0] = (
            p[:,0]-p[first,0]
        )*scale
        out[:,1] = -(
            p[:,1]-p[first,1]
        )*scale
        return out

    pelvis_d = displacement(pelvis)
    bar_d = displacement(bar)

    knee_angle = 180-angle_2d(
        points[hip],points[knee],points[ankle]
    )
    hip_angle = angle_2d(
        points[sh],points[hip],points[knee]
    )
    ankle_angle = angle_2d(
        points[knee],points[ankle],points[toe]
    )

    trunk = points[sh]-points[hip]
    trunk_angle = np.degrees(
        np.arctan2(
            trunk[:,0],-trunk[:,1]
        )
    )

    df = pd.DataFrame({
        "frame": np.arange(start,end+1),
        "time_s": t,
        "knee_flexion_deg": knee_angle,
        "hip_segment_deg": hip_angle,
        "ankle_segment_deg": ankle_angle,
        "trunk_inclination_deg": trunk_angle,
        "pelvis_x_m": pelvis_d[:,0],
        "pelvis_y_m": pelvis_d[:,1],
        "bar_x_m": bar_d[:,0],
        "bar_y_m": bar_d[:,1]
    })

    for name in ["pelvis","bar"]:
        for axis in ["x","y"]:
            col = f"{name}_{axis}_m"
            df[f"{name}_v{axis}_ms"] = velocity(
                df[col].to_numpy(),t
            )

    bottom_relative = bottom-start

    # 하강과 상승을 각각 50%로 정규화
    phase = np.empty(n)
    phase[:bottom_relative+1] = np.linspace(
        0,50,bottom_relative+1
    )
    phase[bottom_relative:] = np.linspace(
        50,100,n-bottom_relative
    )
    df["phase_pct"] = phase

    summary = {
        "max_knee_flexion_deg":
            get_max(knee_angle),
        "knee_ROM_deg":
            get_range(knee_angle),
        "hip_segment_ROM_deg":
            get_range(hip_angle),
        "ankle_segment_ROM_deg":
            get_range(ankle_angle),
        "max_trunk_inclination_deg":
            get_max(trunk_angle,True),

        "pelvis_vertical_range_cm":
            get_range(pelvis_d[:,1])*100,
        "pelvis_AP_range_cm":
            get_range(pelvis_d[:,0])*100,
        "bar_vertical_range_cm":
            get_range(bar_d[:,1])*100,
        "bar_AP_range_cm":
            get_range(bar_d[:,0])*100,

        "max_pelvis_vertical_speed_ms":
            get_max(df.pelvis_vy_ms,True),
        "max_pelvis_AP_speed_ms":
            get_max(df.pelvis_vx_ms,True),
        "max_bar_vertical_speed_ms":
            get_max(df.bar_vy_ms,True),
        "max_bar_AP_speed_ms":
            get_max(df.bar_vx_ms,True),

        "descent_time_s":
            bottom_relative/fps,
        "ascent_time_s":
            (end-bottom)/fps,
        "total_time_s":
            (end-start)/fps,
        "bar_tracking_pct":
            np.isfinite(bar[:,0]).mean()*100,
        "pelvis_valid_pct":
            np.isfinite(pelvis[:,0]).mean()*100
    }

    return df, summary

# =========================================
# 학생 화면: 실험 조건
# =========================================

st.header("1. 비교 조건 설정")

num_conditions = st.selectbox(
    "비교할 조건 수", [2,3], index=1
)

names = []
for i in range(num_conditions):
    names.append(st.text_input(
        f"조건 {chr(65+i)} 이름",
        value=f"조건 {chr(65+i)}",
        key=f"condition_{i}"
    ))

side = st.radio(
    "분석할 다리 (카메라에 가까운 다리)",
    ["right","left"],
    horizontal=True,
    format_func=lambda x:
        "오른쪽" if x=="right" else "왼쪽"
)

length_m = st.number_input(
    "바벨 A-B 실제 길이 (m)",
    min_value=0.01,
    value=1.0,
    step=0.05
)

fps_setting = st.number_input(
    "FPS 직접 지정 (0 = 영상 FPS)",
    min_value=0.0,
    value=0.0,
    step=10.0
)

st.info(
    "각 영상의 앞부분에는 보정 장면이 있고, "
    "이후 스쿼트 동작 1회가 포함되어 있다고 가정합니다."
)

# =========================================
# 영상별 개별 보정 및 분석
# =========================================

st.header("2. 영상 업로드 및 개별 분석")

for ci in range(num_conditions):
    st.subheader(
        f"조건 {chr(65+ci)}: {names[ci]}"
    )

    uploads = st.file_uploader(
        "이 조건의 영상 3개 업로드",
        type=["mp4","mov"],
        accept_multiple_files=True,
        key=f"upload_{ci}"
    )

    if len(uploads) != 3:
        st.warning("영상 3개를 업로드해 주세요.")
        continue

    for trial, uploaded in enumerate(uploads,1):
        raw_bytes = uploaded.getvalue()
        fingerprint = hashlib.sha256(
            raw_bytes
        ).hexdigest()[:16]
        suffix = os.path.splitext(
            uploaded.name
        )[1].lower() or ".mp4"

        uid = f"{ci}_{trial}_{fingerprint}"

        with st.expander(
            f"{trial}회차: {uploaded.name}",
            expanded=False
        ):
            try:
                frames, native_fps = load_video(
                    raw_bytes, suffix
                )
            except Exception as e:
                st.error(str(e))
                continue

            fps = (
                fps_setting
                if fps_setting > 0
                else native_fps
            )

            st.write(
                f"전체 {len(frames)}프레임 | "
                f"영상 FPS: {native_fps:.2f}"
            )

            st.markdown("#### A. 바벨 길이 보정")

            cal_idx = st.slider(
                "보정 장면의 프레임 선택",
                0,len(frames)-1,0,
                key=f"cal_idx_{uid}"
            )

            # 다른 프레임으로 바꾸면 이전 보정점 제거
            frame_token = f"{uid}:{cal_idx}"
            token_key = f"token_{ci}_{trial}"

            if st.session_state.get(
                token_key
            ) != frame_token:
                st.session_state[token_key] = frame_token
                st.session_state.pop(f"A_{ci}_{trial}",None)
                st.session_state.pop(f"B_{ci}_{trial}",None)

            Akey = f"A_{ci}_{trial}"
            Bkey = f"B_{ci}_{trial}"
            A = st.session_state.get(Akey)
            B = st.session_state.get(Bkey)

            click = show_click_image(
                frames[cal_idx],
                f"cal_click_{uid}_{cal_idx}",
                [
                    ("A",A,(255,80,0)),
                    ("B",B,(0,180,255))
                ]
            )

            c1,c2,c3 = st.columns(3)

            with c1:
                if st.button(
                    "A점 저장",
                    key=f"save_A_{uid}"
                ) and click is not None:
                    st.session_state[Akey] = click
                    st.rerun()

            with c2:
                if st.button(
                    "B점 저장",
                    key=f"save_B_{uid}"
                ) and click is not None:
                    st.session_state[Bkey] = click
                    st.rerun()

            with c3:
                if st.button(
                    "보정 초기화",
                    key=f"reset_{uid}"
                ):
                    st.session_state.pop(Akey,None)
                    st.session_state.pop(Bkey,None)
                    st.rerun()

            scale = None
            A = st.session_state.get(Akey)
            B = st.session_state.get(Bkey)

            if A is not None and B is not None:
                distance = np.linalg.norm(A-B)

                if distance > 1:
                    scale = length_m/distance
                    st.success(
                        f"A-B: {distance:.1f}px | "
                        f"보정계수: {scale:.6f} m/pixel"
                    )
                else:
                    st.error("A점과 B점을 서로 다르게 선택하세요.")

            st.markdown("#### B. 스쿼트 분석 프레임")

            start,end = st.slider(
                "스쿼트 시작 및 종료 프레임",
                0,len(frames)-1,
                (0,len(frames)-1),
                key=f"range_{uid}"
            )

            if end-start < 4:
                st.warning("분석 구간이 너무 짧습니다.")
                continue

            bottom = st.slider(
                "스쿼트 최저점 프레임",
                start+1,end-1,
                min(max(
                    (start+end)//2,start+1
                ),end-1),
                key=f"bottom_{uid}"
            )

            preview = st.select_slider(
                "자세 확인",
                options=[
                    start,bottom,end
                ] if len(set([start,bottom,end]))==3
                else list(range(start,end+1)),
                value=bottom,
                key=f"preview_{uid}"
            )

            st.image(
                cv2.cvtColor(
                    frames[preview],
                    cv2.COLOR_BGR2RGB
                ),
                caption=f"프레임 {preview}",
                use_container_width=False,
                width=420
            )

            st.markdown("#### C. 바벨 추적점 지정")
            st.caption(
                "스쿼트 시작 프레임에서 바벨의 "
                "추적하기 쉬운 한 점을 클릭하세요."
            )

            bar_key = f"bar_{ci}_{trial}"
            bar_token_key = f"bar_token_{ci}_{trial}"
            current_bar_token = f"{uid}:{start}"

            if st.session_state.get(
                bar_token_key
            ) != current_bar_token:
                st.session_state[bar_token_key] = (
                    current_bar_token
                )
                st.session_state.pop(bar_key,None)

            bar_saved = st.session_state.get(bar_key)

            bar_click = show_click_image(
                frames[start],
                f"bar_click_{uid}_{start}",
                [("Bar",bar_saved,(255,80,0))]
            )

            if st.button(
                "바벨 추적점 저장",
                key=f"save_bar_{uid}"
            ) and bar_click is not None:
                st.session_state[bar_key] = bar_click
                st.rerun()

            if bar_saved is not None:
                st.success(
                    f"바벨 점: X={bar_saved[0]:.1f}, "
                    f"Y={bar_saved[1]:.1f}"
                )

            ready = (
                scale is not None
                and bar_saved is not None
            )

            if st.button(
                "이 영상 운동학 분석",
                key=f"analyze_{uid}",
                disabled=not ready,
                type="primary"
            ):
                with st.spinner("관절 및 바벨 분석 중..."):
                    try:
                        df, summary = analyze(
                            raw_bytes,suffix,fps,scale,
                            start,bottom,end,
                            bar_saved,side
                        )

                        summary.update({
                            "condition":names[ci],
                            "condition_id":ci+1,
                            "trial":trial,
                            "fps":fps,
                            "scale_m_per_px":scale,
                            "start_frame":start,
                            "bottom_frame":bottom,
                            "end_frame":end
                        })

                        st.session_state.results[
                            (ci,trial)
                        ] = {
                            "df":df,
                            "summary":summary,
                            "fingerprint":fingerprint,
                            "settings":(
                                fps,scale,start,bottom,
                                end,tuple(bar_saved),side
                            )
                        }
                        st.success("분석 완료")
                    except Exception as e:
                        st.error(f"분석 오류: {e}")

            saved = st.session_state.results.get(
                (ci,trial)
            )

            if saved and saved["fingerprint"]==fingerprint:
                st.write(
                    "마지막 분석의 바벨 추적 성공률:",
                    f"{saved['summary']['bar_tracking_pct']:.1f}%"
                )

# =========================================
# 결과표 및 조건별 평균 ± SD
# =========================================

st.header("3. 조건별 결과")

valid = []

for key, item in st.session_state.results.items():
    ci,trial = key
    if ci >= num_conditions:
        continue

    entry = item["summary"].copy()
    entry["condition"] = names[ci]
    valid.append((ci,trial,item["df"],entry))

if not valid:
    st.info("영상 분석을 완료하면 결과가 나타납니다.")
    st.stop()

summary_df = pd.DataFrame([
    r for ci,tr,df,r in valid
])

summary_df = summary_df.sort_values(
    ["condition_id","trial"]
)

st.subheader("개별 시행 결과")
st.dataframe(summary_df, use_container_width=True)

metric_columns = [
    c for c in summary_df.select_dtypes(
        include="number"
    ).columns
    if c not in [
        "condition_id","trial","fps",
        "scale_m_per_px",
        "start_frame","bottom_frame","end_frame"
    ]
]

grouped = summary_df.groupby(
    "condition",sort=False
)[metric_columns].agg(["mean","std","count"])

st.subheader("조건별 평균 ± 표준편차")
st.dataframe(grouped.round(3),use_container_width=True)

st.caption(
    "조건별 SD는 각 조건의 3회 시행값에 대한 "
    "표본 표준편차(ddof=1)입니다."
)

# =========================================
# 조건별 막대그래프
# =========================================

st.header("4. 평균 ± SD 막대그래프")

metric = st.selectbox(
    "비교할 변수",
    metric_columns
)

fig = go.Figure()
colors = ["royalblue","seagreen","darkorange"]

for ci,name in enumerate(names):
    values = summary_df.loc[
        summary_df.condition_id==ci+1,
        metric
    ].dropna().to_numpy()

    if not len(values):
        continue

    mean = values.mean()
    sd = (
        values.std(ddof=1)
        if len(values)>1 else np.nan
    )

    fig.add_trace(go.Bar(
        x=[name],y=[mean],
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
    title=f"{metric} — Mean ± SD",
    template="plotly_white",
    yaxis_title=metric,
    height=480
)
st.plotly_chart(fig,use_container_width=True)

# =========================================
# 시간 정규화 곡선
# =========================================

st.header("5. 시간 정규화 곡선 ± SD")

curve_options = [
    "knee_flexion_deg",
    "hip_segment_deg",
    "ankle_segment_deg",
    "trunk_inclination_deg",
    "pelvis_x_m","pelvis_y_m",
    "pelvis_vx_ms","pelvis_vy_ms",
    "bar_x_m","bar_y_m",
    "bar_vx_ms","bar_vy_ms"
]

curve_metric = st.selectbox(
    "시간 정규화 변수",curve_options
)

phase = np.linspace(0,100,101)
curve_fig = go.Figure()

fills = [
    "rgba(65,105,225,.16)",
    "rgba(46,139,87,.16)",
    "rgba(255,140,0,.16)"
]

for ci,name in enumerate(names):
    curves = []

    for k,tr,df,r in valid:
        if k != ci:
            continue

        x = df.phase_pct.to_numpy()
        y = df[curve_metric].to_numpy()
        good = np.isfinite(x)&np.isfinite(y)

        if good.mean()<0.9 or good.sum()<5:
            continue

        curves.append(
            np.interp(phase,x[good],y[good])
        )

    if not curves:
        continue

    curves = np.asarray(curves)
    avg = curves.mean(axis=0)
    sd = (
        curves.std(axis=0,ddof=1)
        if len(curves)>1
        else np.zeros_like(avg)
    )

    curve_fig.add_trace(go.Scatter(
        x=phase,y=avg+sd,
        line=dict(width=0),
        showlegend=False
    ))
    curve_fig.add_trace(go.Scatter(
        x=phase,y=avg-sd,
        line=dict(width=0),
        fill="tonexty",
        fillcolor=fills[ci],
        showlegend=False
    ))
    curve_fig.add_trace(go.Scatter(
        x=phase,y=avg,
        line=dict(color=colors[ci],width=3),
        name=f"{name} (n={len(curves)})"
    ))

curve_fig.add_vline(
    x=50,line_dash="dash",
    line_color="gray"
)

curve_fig.update_layout(
    title=f"{curve_metric} — Mean ± SD",
    xaxis_title=(
        "Squat phase (%) | "
        "0: 시작, 50: 최저점, 100: 종료"
    ),
    yaxis_title=curve_metric,
    template="plotly_white",
    height=500
)

st.plotly_chart(
    curve_fig,use_container_width=True
)

# =========================================
# 바벨 및 골반 이동 궤적
# =========================================

st.header("6. 전후·수직 이동 궤적")

target = st.radio(
    "추적 대상",
    ["bar","pelvis"],
    horizontal=True,
    format_func=lambda x:
        "바벨" if x=="bar" else "골반 중심"
)

trajectory = go.Figure()

for ci,tr,df,r in valid:
    trajectory.add_trace(go.Scatter(
        x=df[f"{target}_x_m"]*100,
        y=df[f"{target}_y_m"]*100,
        mode="lines+markers",
        marker=dict(size=3),
        name=f"{names[ci]} {tr}회"
    ))

trajectory.update_layout(
    xaxis_title="화면 기준 전후 변위 (cm)",
    yaxis_title="수직 변위 (cm)",
    template="plotly_white",
    height=500
)
trajectory.update_yaxes(scaleanchor="x",scaleratio=1)

st.plotly_chart(
    trajectory,use_container_width=True
)

# =========================================
# Excel 저장
# =========================================

st.header("7. Excel 결과 다운로드")

buffer = io.BytesIO()

with pd.ExcelWriter(
    buffer,engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,"Trial_Results",index=False
    )

    grouped.to_excel(
        writer,"Mean_SD"
    )

    for ci,tr,df,r in valid:
        df.to_excel(
            writer,
            sheet_name=f"C{ci+1}_T{tr}",
            index=False
        )

st.download_button(
    "전체 분석 결과 Excel 다운로드",
    data=buffer.getvalue(),
    file_name="Squat_Biomechanics_Results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)

st.info(
    "본 앱은 2D 영상 기반 탐색적 운동학 분석 도구입니다. "
    "골반 중심은 전신 COM이 아니며, 정확한 거리 분석은 "
    "보정 평면과 카메라 기하의 일치를 전제로 합니다."
)
