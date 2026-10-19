
import io
import os
import hashlib
import tempfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import mediapipe as mp

from PIL import Image
from scipy.signal import savgol_filter
from streamlit_image_coordinates import streamlit_image_coordinates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================
# 1. 앱 설정
# =====================================

st.set_page_config(
    page_title="Squat Biomechanics Lab",
    layout="wide"
)

st.title("Squat Biomechanics Lab")
st.caption(
    "시상면 2D 동작분석 | 조건별 3회 | "
    "관절각도 · 골반 및 바벨 운동학 · 평균 ± SD"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_FILE = "/tmp/squat_pose_landmarker.task"

if "results" not in st.session_state:
    st.session_state.results = {}

if "saved_points" not in st.session_state:
    st.session_state.saved_points = {}


# =====================================
# 2. MediaPipe 모델 준비
# =====================================

@st.cache_resource
def get_model_path():
    if not os.path.isfile(MODEL_FILE):
        urllib.request.urlretrieve(
            MODEL_URL, MODEL_FILE
        )
    return MODEL_FILE


# =====================================
# 3. 영상 읽기
# =====================================

@st.cache_data(
    show_spinner=False,
    max_entries=10
)
def read_video(data, extension):
    with tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False
    ) as f:
        f.write(data)
        path = f.name

    frames = []
    cap = cv2.VideoCapture(path)

    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

            if len(frames) > 900:
                raise ValueError(
                    "영상이 너무 깁니다. "
                    "15초 이내 영상으로 준비하세요."
                )
    finally:
        cap.release()
        os.unlink(path)

    if len(frames) < 5:
        raise ValueError(
            "유효한 영상 프레임이 부족합니다."
        )

    return frames, float(fps)


# =====================================
# 4. 영상에서 좌표 클릭
# =====================================

def select_point(frame, key, marks=None):
    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    ).copy()

    if marks:
        for label, point, color in marks:
            if point is None:
                continue

            p = tuple(
                np.asarray(point, dtype=int)
            )

            cv2.circle(
                rgb, p, 7, color, -1
            )

            cv2.putText(
                rgb, label,
                (p[0]+8, p[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

    height, width = rgb.shape[:2]

    shown_width = min(width, 620)
    shown_height = max(
        1,
        round(height*shown_width/width)
    )

    image = Image.fromarray(rgb).resize(
        (shown_width, shown_height)
    )

    result = streamlit_image_coordinates(
        image, key=key
    )

    if result is None:
        return None

    return np.array([
        result["x"]*width/shown_width,
        result["y"]*height/shown_height
    ], dtype=float)


# =====================================
# 5. 수치 계산 도구
# =====================================

def smooth_signal(values, fps):
    values = pd.Series(
        values, dtype=float
    ).interpolate(
        limit=3,
        limit_area="inside"
    ).to_numpy()

    result = values.copy()
    valid = np.isfinite(values)

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]

    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    window = max(
        5, int(fps*0.15) | 1
    )

    for a, b in zip(starts, ends):
        n = b-a+1

        win = min(
            window,
            n if n % 2 else n-1
        )

        if win >= 5:
            result[a:b+1] = savgol_filter(
                values[a:b+1],
                win, 2
            )

    return result


def angle_2d(a, b, c):
    u = a-b
    v = c-b

    dot = np.sum(
        u*v, axis=1
    )

    denom = (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
    )

    cosine = np.divide(
        dot,
        denom,
        out=np.full(
            len(dot), np.nan
        ),
        where=denom > 1e-10
    )

    return np.degrees(
        np.arccos(
            np.clip(cosine, -1, 1)
        )
    )


def velocity(values, time):
    values = np.asarray(
        values, dtype=float
    )

    result = np.full(
        len(values), np.nan
    )

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


def safe_max(x, absolute=False):
    x = np.asarray(x, dtype=float)

    if absolute:
        x = np.abs(x)

    if not np.isfinite(x).any():
        return np.nan

    return float(np.nanmax(x))


def safe_range(x):
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).any():
        return np.nan

    return float(
        np.nanmax(x)-np.nanmin(x)
    )


# =====================================
# 6. MediaPipe 랜드마크 추출
# =====================================

@st.cache_data(
    show_spinner=False,
    max_entries=10
)
def detect_pose(video_bytes, ext, fps):
    frames, native_fps = read_video(
        video_bytes, ext
    )

    coords = np.full(
        (len(frames),33,2),
        np.nan
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=get_model_path()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as detector:

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            result = detector.detect_for_video(
                image,
                int(round(i*1000/fps))
            )

            if not result.pose_landmarks:
                continue

            for j, p in enumerate(
                result.pose_landmarks[0]
            ):
                if p.visibility >= 0.5:
                    coords[i,j] = [
                        p.x*w,
                        p.y*h
                    ]

    return coords


# =====================================
# 7. 바벨 추적
# =====================================

def track_barbell(frames, initial):
    points = np.full(
        (len(frames),2), np.nan
    )

    points[0] = initial

    previous_gray = cv2.cvtColor(
        frames[0],
        cv2.COLOR_BGR2GRAY
    )

    previous_point = np.array(
        [[initial]],
        dtype=np.float32
    )

    for i in range(1,len(frames)):
        current_gray = cv2.cvtColor(
            frames[i],
            cv2.COLOR_BGR2GRAY
        )

        next_point, status, error = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_point,
                None,
                winSize=(31,31),
                maxLevel=3
            )
        )

        if next_point is None or status is None:
            break

        if status[0,0] != 1:
            break

        point = next_point.reshape(2)

        if not np.isfinite(point).all():
            break

        if np.linalg.norm(
            point-previous_point.reshape(2)
        ) > 80:
            break

        points[i] = point

        previous_gray = current_gray
        previous_point = next_point.reshape(
            1,1,2
        )

    return points


# =====================================
# 8. 운동학 분석
# =====================================

def analyze_squat(
    video_bytes,
    extension,
    fps,
    scale,
    side,
    start,
    bottom,
    end,
    squat_type,
    bar_point=None
):
    frames,_ = read_video(
        video_bytes, extension
    )

    if not (
        0 <= start < bottom < end < len(frames)
    ):
        raise ValueError(
            "프레임 순서는 시작 < 최저점 < 종료입니다."
        )

    pose = detect_pose(
        video_bytes,
        extension,
        fps
    )[start:end+1].copy()

    selected_frames = frames[start:end+1]

    n = len(selected_frames)
    time = np.arange(n)/fps

    joint_ids = {
        "left": [11,23,25,27,31],
        "right": [12,24,26,28,32]
    }

    shoulder,hip,knee,ankle,toe = joint_ids[side]

    points = {}

    for joint in [
        shoulder,hip,knee,ankle,toe
    ]:
        points[joint] = np.column_stack([
            smooth_signal(
                pose[:,joint,0],fps
            ),
            smooth_signal(
                pose[:,joint,1],fps
            )
        ])

    # 골반 중점
    pelvis = np.full((n,2),np.nan)

    valid_both = np.isfinite(
        pose[:,[23,24],:]
    ).all(axis=(1,2))

    pelvis[valid_both] = (
        pose[valid_both,23,:]
        + pose[valid_both,24,:]
    )/2

    for axis in range(2):
        pelvis[:,axis] = smooth_signal(
            pelvis[:,axis],fps
        )

    # 바벨 스쿼트일 때만 추적
    if squat_type == "바벨 스쿼트":
        if bar_point is None:
            raise ValueError(
                "바벨 추적점을 지정하세요."
            )

        bar = track_barbell(
            selected_frames,
            np.asarray(bar_point,dtype=float)
        )

        for axis in range(2):
            bar[:,axis] = smooth_signal(
                bar[:,axis],fps
            )
    else:
        bar = np.full((n,2),np.nan)

    def displacement(coords):
        result = np.full_like(
            coords,np.nan
        )

        valid = np.isfinite(
            coords
        ).all(axis=1)

        if not valid.any():
            return result

        first = np.where(valid)[0][0]
        origin = coords[first]

        result[:,0] = (
            coords[:,0]-origin[0]
        )*scale

        result[:,1] = -(
            coords[:,1]-origin[1]
        )*scale

        return result

    pelvis_d = displacement(pelvis)
    bar_d = displacement(bar)

    knee_flexion = 180-angle_2d(
        points[hip],
        points[knee],
        points[ankle]
    )

    hip_angle = angle_2d(
        points[shoulder],
        points[hip],
        points[knee]
    )

    ankle_angle = angle_2d(
        points[knee],
        points[ankle],
        points[toe]
    )

    trunk_vector = (
        points[shoulder]-points[hip]
    )

    trunk_angle = np.degrees(
        np.arctan2(
            trunk_vector[:,0],
            -trunk_vector[:,1]
        )
    )

    df = pd.DataFrame({
        "frame": np.arange(start,end+1),
        "time_s": time,
        "knee_flexion_deg": knee_flexion,
        "hip_segment_deg": hip_angle,
        "ankle_segment_deg": ankle_angle,
        "trunk_inclination_deg": trunk_angle,
        "pelvis_AP_m": pelvis_d[:,0],
        "pelvis_vertical_m": pelvis_d[:,1],
        "bar_AP_m": bar_d[:,0],
        "bar_vertical_m": bar_d[:,1]
    })

    for name in ["pelvis","bar"]:
        for axis in ["AP","vertical"]:
            position = f"{name}_{axis}_m"
            speed = f"{name}_{axis}_velocity_ms"

            df[speed] = velocity(
                df[position].to_numpy(),
                time
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
            safe_max(trunk_angle,True),

        "pelvis_AP_range_cm":
            safe_range(pelvis_d[:,0])*100,
        "pelvis_vertical_range_cm":
            safe_range(pelvis_d[:,1])*100,

        "bar_AP_range_cm":
            safe_range(bar_d[:,0])*100,
        "bar_vertical_range_cm":
            safe_range(bar_d[:,1])*100,

        "max_pelvis_AP_velocity_ms":
            safe_max(
                df.pelvis_AP_velocity_ms,True
            ),
        "max_pelvis_vertical_velocity_ms":
            safe_max(
                df.pelvis_vertical_velocity_ms,True
            ),

        "max_bar_AP_velocity_ms":
            safe_max(
                df.bar_AP_velocity_ms,True
            ),
        "max_bar_vertical_velocity_ms":
            safe_max(
                df.bar_vertical_velocity_ms,True
            ),

        "descent_time_s":bottom_local/fps,
        "ascent_time_s":(end-bottom)/fps,
        "total_time_s":(end-start)/fps,

        "pelvis_valid_pct":
            np.isfinite(
                pelvis[:,0]
            ).mean()*100,

        "bar_tracking_pct":
            (
                np.isfinite(
                    bar[:,0]
                ).mean()*100
                if squat_type=="바벨 스쿼트"
                else np.nan
            )
    }

    return df,summary


# =====================================
# 9. 학생 입력 화면
# =====================================

st.header("1. 실험 조건")

condition_count = st.selectbox(
    "비교할 조건 수",
    [2,3],
    index=1
)

condition_names = []
condition_types = []

for i in range(condition_count):
    col1,col2 = st.columns(2)

    with col1:
        name = st.text_input(
            f"조건 {chr(65+i)} 이름",
            value=f"조건 {chr(65+i)}",
            key=f"condition_name_{i}"
        )

    with col2:
        squat_type = st.selectbox(
            f"조건 {chr(65+i)} 운동 유형",
            ["맨몸 스쿼트","바벨 스쿼트"],
            key=f"condition_type_{i}"
        )

    condition_names.append(name)
    condition_types.append(squat_type)

side = st.radio(
    "분석할 다리",
    ["right","left"],
    horizontal=True,
    format_func=lambda x:
        "오른쪽" if x=="right" else "왼쪽"
)

fps_override = st.number_input(
    "FPS 직접 지정 (0이면 영상 FPS 사용)",
    min_value=0.0,
    value=0.0,
    step=10.0
)

st.info(
    "영상별로 알려진 길이의 두 점을 이용해 "
    "거리 보정합니다. 맨몸 스쿼트는 기준자를 "
    "사용할 수 있습니다."
)


# =====================================
# 10. 조건별 영상 분석
# =====================================

st.header("2. 영상 업로드 및 분석")

current_results = []

for ci in range(condition_count):
    condition = condition_names[ci]
    squat_type = condition_types[ci]

    st.subheader(
        f"조건 {chr(65+ci)}: {condition}"
    )

    for trial in range(1,4):
        slot = f"{ci}_{trial}"

        with st.expander(
            f"{trial}회차",
            expanded=False
        ):
            uploaded = st.file_uploader(
                "스쿼트 영상 업로드",
                type=["mp4","mov"],
                key=f"upload_{slot}"
            )

            if uploaded is None:
                st.info("영상을 업로드하세요.")
                continue

            video_bytes = uploaded.getvalue()

            digest = hashlib.sha256(
                video_bytes
            ).hexdigest()[:12]

            extension = os.path.splitext(
                uploaded.name
            )[1].lower() or ".mp4"

            uid = f"{slot}_{digest}"

            try:
                frames,native_fps = read_video(
                    video_bytes,extension
                )
            except Exception as e:
                st.error(str(e))
                continue

            fps = (
                fps_override
                if fps_override>0
                else native_fps
            )

            st.caption(
                f"전체 {len(frames)}프레임 | "
                f"FPS {fps:.2f}"
            )

            # -------------------------
            # A. 거리 보정
            # -------------------------

            st.markdown("#### A. 거리 보정")

            actual_length = st.number_input(
                "A-B 실제 길이 (m)",
                min_value=0.01,
                value=1.0,
                step=0.05,
                key=f"length_{uid}"
            )

            cal_frame = st.slider(
                "보정 프레임",
                0,len(frames)-1,0,
                key=f"cal_frame_{uid}"
            )

            Akey = f"A_{uid}_{cal_frame}"
            Bkey = f"B_{uid}_{cal_frame}"

            A = st.session_state.saved_points.get(Akey)
            B = st.session_state.saved_points.get(Bkey)

            click = select_point(
                frames[cal_frame],
                f"cal_click_{uid}_{cal_frame}",
                [
                    ("A",A,(255,80,0)),
                    ("B",B,(0,180,255))
                ]
            )

            c1,c2 = st.columns(2)

            with c1:
                if st.button(
                    "A점 저장",
                    key=f"save_A_{uid}"
                ) and click is not None:
                    st.session_state.saved_points[
                        Akey
                    ] = click
                    st.rerun()

            with c2:
                if st.button(
                    "B점 저장",
                    key=f"save_B_{uid}"
                ) and click is not None:
                    st.session_state.saved_points[
                        Bkey
                    ] = click
                    st.rerun()

            scale = None

            if A is not None and B is not None:
                px = np.linalg.norm(A-B)

                if px > 1:
                    scale = actual_length/px

                    st.success(
                        f"보정 완료: "
                        f"{scale:.6f} m/pixel"
                    )

            # -------------------------
            # B. 스쿼트 구간
            # -------------------------

            st.markdown("#### B. 스쿼트 구간")

            if len(frames)<6:
                st.warning("영상이 너무 짧습니다.")
                continue

            start,end = st.slider(
                "스쿼트 시작~종료 프레임",
                0,len(frames)-1,
                (0,len(frames)-1),
                key=f"range_{uid}"
            )

            if end-start<4:
                st.warning("구간을 넓혀주세요.")
                continue

            bottom = st.slider(
                "스쿼트 최저점 프레임",
                start+1,end-1,
                value=(start+end)//2,
                key=f"bottom_{uid}"
            )

            preview = st.select_slider(
                "영상 프레임 확인",
                options=range(start,end+1),
                value=bottom,
                key=f"preview_{uid}"
            )

            st.image(
                cv2.cvtColor(
                    frames[preview],
                    cv2.COLOR_BGR2RGB
                ),
                caption=f"프레임 {preview}",
                width=440
            )

            # -------------------------
            # C. 바벨 추적
            # -------------------------

            bar_point = None

            if squat_type=="바벨 스쿼트":
                st.markdown("#### C. 바벨 추적")

                bar_key = f"bar_{uid}_{start}"

                bar_point = (
                    st.session_state.saved_points
                    .get(bar_key)
                )

                bar_click = select_point(
                    frames[start],
                    f"bar_click_{uid}_{start}",
                    [(
                        "Bar",
                        bar_point,
                        (255,80,0)
                    )]
                )

                if st.button(
                    "바벨 추적점 저장",
                    key=f"save_bar_{uid}_{start}"
                ) and bar_click is not None:
                    st.session_state.saved_points[
                        bar_key
                    ] = bar_click
                    st.rerun()

            else:
                st.info(
                    "맨몸 스쿼트: 바벨 추적 단계 생략"
                )

            # -------------------------
            # D. 분석 버튼
            # -------------------------

            ready = (
                scale is not None
                and (
                    squat_type=="맨몸 스쿼트"
                    or bar_point is not None
                )
            )

            settings = (
                digest,fps,scale,start,bottom,end,
                side,squat_type,
                tuple(bar_point)
                if bar_point is not None else None
            )

            if st.button(
                "이 영상 분석",
                key=f"analyze_{uid}",
                disabled=not ready,
                type="primary"
            ):
                try:
                    with st.spinner("운동학 분석 중..."):
                        df,metrics = analyze_squat(
                            video_bytes,
                            extension,
                            fps,
                            scale,
                            side,
                            start,
                            bottom,
                            end,
                            squat_type,
                            bar_point
                        )

                    st.session_state.results[
                        slot
                    ] = {
                        "settings":settings,
                        "data":df,
                        "metrics":metrics
                    }

                    st.success("분석 완료!")

                except Exception as e:
                    st.error(
                        f"{type(e).__name__}: {e}"
                    )

            saved = st.session_state.results.get(
                slot
            )

            if (
                saved is not None
                and saved["settings"]==settings
            ):
                current_results.append({
                    "condition":condition,
                    "condition_id":ci+1,
                    "squat_type":squat_type,
                    "trial":trial,
                    "data":saved["data"],
                    "metrics":saved["metrics"]
                })


# =====================================
# 11. 결과표
# =====================================

st.header("3. 조건별 운동학 결과")

if not current_results:
    st.info(
        "영상 분석을 완료하면 "
        "조건별 결과가 표시됩니다."
    )
    st.stop()

rows = []

for item in current_results:
    row = {
        "condition":item["condition"],
        "condition_id":item["condition_id"],
        "squat_type":item["squat_type"],
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

st.subheader("반복별 분석 결과")
st.dataframe(results_df.round(3))

statistics = results_df.groupby(
    ["condition_id","condition"]
)[numeric].agg(["mean","std","count"])

st.subheader("조건별 평균 ± SD")
st.dataframe(statistics.round(3))


# =====================================
# 12. 막대그래프
# =====================================

st.header("4. 조건별 평균 ± SD 그래프")

metric = st.selectbox(
    "운동학 변수 선택",
    numeric
)

colors = [
    "royalblue",
    "seagreen",
    "darkorange"
]

fig = go.Figure()

for ci,condition in enumerate(condition_names):
    vals = results_df.loc[
        results_df.condition_id==ci+1,
        metric
    ].dropna().to_numpy()

    if not len(vals):
        continue

    mean = np.mean(vals)
    sd = (
        np.std(vals,ddof=1)
        if len(vals)>1
        else np.nan
    )

    fig.add_trace(go.Bar(
        x=[condition],
        y=[mean],
        name=condition,
        marker_color=colors[ci],
        error_y=dict(
            type="data",
            array=[
                sd if np.isfinite(sd) else 0
            ],
            visible=True
        ),
        text=[
            f"{mean:.2f} ± {sd:.2f}"
            if np.isfinite(sd)
            else f"{mean:.2f}"
        ],
        textposition="outside"
    ))

    fig.add_trace(go.Scatter(
        x=[condition]*len(vals),
        y=vals,
        mode="markers",
        marker=dict(
            size=10,
            color="black",
            symbol="circle-open"
        ),
        showlegend=False
    ))

fig.update_layout(
    title=f"{metric}: Mean ± SD",
    template="plotly_white",
    height=480
)

st.plotly_chart(fig)


# =====================================
# 13. 시간 정규화 곡선
# =====================================

st.header("5. 스쿼트 동작 곡선 ± SD")

curve_options = [
    c for c in current_results[0]["data"].columns
    if c not in ["frame","time_s","phase_pct"]
]

curve_metric = st.selectbox(
    "그래프 변수",
    curve_options
)

phase_grid = np.linspace(0,100,101)
curve_fig = go.Figure()

fills = [
    "rgba(65,105,225,0.16)",
    "rgba(46,139,87,0.16)",
    "rgba(255,140,0,0.16)"
]

for ci,condition in enumerate(condition_names):
    curves = []

    for item in current_results:
        if item["condition_id"]!=ci+1:
            continue

        df = item["data"]
        x = df.phase_pct.to_numpy()
        y = df[curve_metric].to_numpy()

        good = np.isfinite(x)&np.isfinite(y)

        if good.sum()<5 or good.mean()<0.9:
            continue

        curves.append(
            np.interp(
                phase_grid,
                x[good],
                y[good]
            )
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
        x=phase_grid,
        y=avg+sd,
        line=dict(width=0),
        showlegend=False
    ))

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,
        y=avg-sd,
        fill="tonexty",
        fillcolor=fills[ci],
        line=dict(width=0),
        showlegend=False
    ))

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,
        y=avg,
        mode="lines",
        name=f"{condition} (n={len(curves)})",
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
    title=f"{curve_metric}: Mean ± SD",
    xaxis_title=(
        "Squat phase (%) "
        "0=시작, 50=최저점, 100=종료"
    ),
    yaxis_title=curve_metric,
    template="plotly_white"
)

st.plotly_chart(curve_fig)


# =====================================
# 14. 이동 궤적
# =====================================

st.header("6. 전후·수직 이동 궤적")

target = st.radio(
    "분석 대상",
    ["pelvis","bar"],
    horizontal=True,
    format_func=lambda x:
        "골반 중심" if x=="pelvis" else "바벨"
)

trajectory = go.Figure()

for item in current_results:
    df = item["data"]

    if target=="bar" and (
        item["squat_type"]=="맨몸 스쿼트"
    ):
        continue

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
    xaxis_title="전후 변위 (cm)",
    yaxis_title="수직 변위 (cm)",
    template="plotly_white"
)

trajectory.update_yaxes(
    scaleanchor="x",
    scaleratio=1
)

st.plotly_chart(trajectory)


# =====================================
# 15. Excel 저장
# =====================================

st.header("7. 결과 다운로드")

buffer = io.BytesIO()

with pd.ExcelWriter(
    buffer,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Trials",
        index=False
    )

    statistics.to_excel(
        writer,
        sheet_name="Mean_SD"
    )

    for item in current_results:
        sheet = (
            f"C{item['condition_id']}"
            f"_T{item['trial']}"
        )

        item["data"].to_excel(
            writer,
            sheet_name=sheet,
            index=False
        )

st.download_button(
    "Excel 전체 결과 다운로드",
    data=buffer.getvalue(),
    file_name="Squat_Biomechanics_Results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)

st.warning(
    "골반 중심은 실제 전신 COM이 아니며 "
    "관절각도는 2D 영상 기반 추정값입니다. "
    "거리와 속도의 정확도는 영상 보정 조건과 "
    "랜드마크 추적 정확도에 영향을 받습니다."
)
