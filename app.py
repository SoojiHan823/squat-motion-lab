
import os
import io
import hashlib
import tempfile
import subprocess
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import mediapipe as mp
import imageio_ffmpeg

from PIL import Image
from scipy.signal import butter, sosfiltfilt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================================
# 1. 기본 설정
# =========================================

st.set_page_config(
    page_title="Squat Biomechanics Lab",
    layout="wide"
)

st.title("Squat Biomechanics Lab")
st.caption(
    "맨몸·바벨 스쿼트 | 운동학 분석 | "
    "Butterworth Filter | Mean ± SD"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_PATH = "/tmp/squat_pose_landmarker.task"
MAX_WIDTH = 640
MAX_FRAMES = 1200

for key in ["squat_files_v4", "squat_results_v4"]:
    if key not in st.session_state:
        st.session_state[key] = {}

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29),
    (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30),
    (30, 32), (28, 32)
]

JOINTS = {
    "left": (11, 23, 25, 27, 31),
    "right": (12, 24, 26, 28, 32)
}


# =========================================
# 2. 영상 파일 관리
# =========================================

def prepare_file(upload, slot):
    raw = upload.getvalue()

    if len(raw) > 40 * 1024 * 1024:
        raise ValueError(
            "파일이 40MB를 넘습니다. "
            "영상 길이나 용량을 줄여주세요."
        )

    digest = hashlib.sha256(raw).hexdigest()[:16]
    existing = st.session_state.squat_files_v4.get(slot)

    if existing and existing["digest"] == digest:
        return existing

    if existing:
        old_path = existing.get("path")
        if old_path and os.path.isfile(old_path):
            os.remove(old_path)

    suffix = os.path.splitext(upload.name)[1].lower()
    if suffix not in [".mp4", ".mov"]:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    ) as f:
        f.write(raw)
        path = f.name

    state = {
        "digest": digest,
        "path": path,
        "cal_frame": None,
        "A": None,
        "B": None,
        "cal_version": 0,
        "bar_version": 0,
        "bar_start": None,
        "bar_point": None
    }

    st.session_state.squat_files_v4[slot] = state
    st.session_state.squat_results_v4.pop(slot, None)
    return state


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not np.isfinite(fps) or fps <= 0 or count < 5:
        raise ValueError("영상 FPS 또는 프레임 정보 오류")

    if count > MAX_FRAMES:
        raise ValueError(
            "영상이 너무 깁니다. "
            "20초 이내의 짧은 영상으로 준비하세요."
        )

    return float(fps), count


def resize_frame(frame):
    h, w = frame.shape[:2]

    if w > MAX_WIDTH:
        new_h = max(1, round(h * MAX_WIDTH / w))
        frame = cv2.resize(
            frame,
            (MAX_WIDTH, new_h),
            interpolation=cv2.INTER_AREA
        )

    return frame


def get_frame(path, number):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(
            f"{number}번 프레임을 읽지 못했습니다."
        )

    return resize_frame(frame)


def pick_point(frame, key, marks=None):
    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    ).copy()

    for label, p, color in (marks or []):
        if p is None:
            continue

        x, y = np.round(p).astype(int)
        cv2.circle(rgb, (x, y), 6, color, -1)
        cv2.putText(
            rgb, label, (x+8, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb)

    click = streamlit_image_coordinates(
        pil, key=key
    )

    if click is None:
        return None

    return np.array([
        click["x"], click["y"]
    ], dtype=float)


# =========================================
# 3. 필터와 운동학 계산
# =========================================

def lowpass(values, fps, cutoff):
    """4차 Butterworth, zero-phase"""

    arr = pd.Series(
        values, dtype=float
    ).interpolate(
        limit=3, limit_area="inside"
    ).to_numpy()

    result = arr.copy()

    sos = butter(
        4, cutoff,
        fs=fps,
        btype="low",
        output="sos"
    )

    valid = np.isfinite(arr)

    starts = np.where(
        valid & ~np.r_[False, valid[:-1]]
    )[0]

    ends = np.where(
        valid & ~np.r_[valid[1:], False]
    )[0]

    for a, b in zip(starts, ends):
        segment = arr[a:b+1]

        if len(segment) < 20:
            continue

        try:
            result[a:b+1] = sosfiltfilt(
                sos, segment
            )
        except ValueError:
            pass

    return result


def angle_2d(a, b, c):
    u = a-b
    v = c-b

    dot = np.sum(u*v, axis=1)
    mag = (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
    )

    cosine = np.divide(
        dot, mag,
        out=np.full(len(dot), np.nan),
        where=mag > 1e-10
    )

    return np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    )


def derivative(values, time):
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
                values[a:b+1],
                time[a:b+1]
            )

    return result


def safe_range(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanmax(x)-np.nanmin(x))


def safe_max(x, absolute=False):
    x = np.asarray(x, dtype=float)
    if absolute:
        x = np.abs(x)

    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanmax(x))


@st.cache_resource
def model_file():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            MODEL_URL, MODEL_PATH
        )
    return MODEL_PATH


# =========================================
# 4. Skeleton 영상 + 실시간 처리 미리보기
# =========================================

def process_video(
    path, fps, start, end,
    bar_point, show_bar,
    preview_area, progress_area
):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    raw_out = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ).name

    browser_out = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ).name

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=model_file()
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    writer = None
    coords = []
    bars = []

    previous_gray = None
    previous_point = None

    total = end-start+1
    step = max(1, int(fps/4))

    try:
        with vision.PoseLandmarker.create_from_options(
            options
        ) as detector:

            for i in range(total):
                ok, original = cap.read()
                if not ok:
                    raise ValueError(
                        "분석 구간에서 영상 읽기 실패"
                    )

                frame = resize_frame(original)
                h, w = frame.shape[:2]

                if writer is None:
                    writer = cv2.VideoWriter(
                        raw_out,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h)
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            "분석 영상 저장 실패"
                        )

                rgb = np.ascontiguousarray(
                    cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect_for_video(
                    image,
                    int(round(i*1000/fps))
                )

                xy = np.full((33, 2), np.nan)

                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]

                    for j, p in enumerate(pose):
                        if (
                            p.visibility >= 0.5
                            and np.isfinite([p.x, p.y]).all()
                        ):
                            xy[j] = [p.x*w, p.y*h]

                coords.append(xy)

                # 바벨 Optical Flow
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )

                bar = np.array([np.nan, np.nan])

                if show_bar:
                    if i == 0:
                        bar = np.array(
                            bar_point, dtype=float
                        )
                        previous_point = np.array(
                            [[bar_point]], dtype=np.float32
                        )
                    elif previous_point is not None:
                        nxt, status, err = (
                            cv2.calcOpticalFlowPyrLK(
                                previous_gray,
                                gray,
                                previous_point,
                                None,
                                winSize=(31, 31),
                                maxLevel=3
                            )
                        )

                        if (
                            nxt is not None
                            and status is not None
                            and status[0, 0] == 1
                        ):
                            candidate = nxt.reshape(2)

                            if (
                                np.linalg.norm(
                                    candidate
                                    - previous_point.reshape(2)
                                ) < 60
                            ):
                                bar = candidate.copy()
                                previous_point = nxt.reshape(
                                    1, 1, 2
                                )
                            else:
                                previous_point = None
                        else:
                            previous_point = None

                previous_gray = gray
                bars.append(bar)

                # Skeleton 연결선
                for a, b in CONNECTIONS:
                    if (
                        np.isfinite(xy[a]).all()
                        and np.isfinite(xy[b]).all()
                    ):
                        p1 = tuple(xy[a].astype(int))
                        p2 = tuple(xy[b].astype(int))
                        cv2.line(
                            frame, p1, p2,
                            (0, 220, 0), 2
                        )

                # 관절점 표시
                for j, p in enumerate(xy):
                    if np.isfinite(p).all():
                        cv2.circle(
                            frame,
                            tuple(p.astype(int)),
                            3, (0, 0, 255), -1
                        )

                if np.isfinite(bar).all():
                    cv2.circle(
                        frame,
                        tuple(bar.astype(int)),
                        7, (255, 100, 0), -1
                    )
                    cv2.putText(
                        frame, "BAR",
                        tuple(
                            bar.astype(int)
                            + np.array([8, -8])
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 100, 0), 2
                    )

                cv2.putText(
                    frame,
                    f"Frame {start+i} | "
                    f"{i/fps:.2f}s",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                writer.write(frame)

                if i % step == 0 or i == total-1:
                    preview_area.image(
                        cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        ),
                        caption=(
                            f"관절 트래킹 중: "
                            f"{i+1}/{total}프레임"
                        )
                    )

                    progress_area.progress(
                        (i+1)/total
                    )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    coords = np.asarray(coords, dtype=float)
    bars = np.asarray(bars, dtype=float)

    # 브라우저 재생용 H.264 MP4
    try:
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-loglevel", "error",
            "-i", raw_out,
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            browser_out
        ], check=True, capture_output=True)

        video_out = browser_out
        os.remove(raw_out)

    except Exception:
        video_out = raw_out
        if os.path.exists(browser_out):
            os.remove(browser_out)

    return coords, bars, video_out


# =========================================
# 5. 운동학 지표 계산
# =========================================

def calculate_metrics(
    coords, bar, fps, cutoff,
    scale, side, direction,
    start, bottom, end, squat_type
):
    n = len(coords)
    time = np.arange(n)/fps

    sh, hip, knee, ankle, toe = JOINTS[side]

    essential = {
        11, 12, 23, 24, 25, 26,
        27, 28, 31, 32
    }

    filtered = {}

    for j in essential:
        filtered[j] = np.column_stack([
            lowpass(coords[:, j, 0], fps, cutoff),
            lowpass(coords[:, j, 1], fps, cutoff)
        ])

    knee_raw = 180-angle_2d(
        coords[:, hip],
        coords[:, knee],
        coords[:, ankle]
    )

    knee_flex = 180-angle_2d(
        filtered[hip],
        filtered[knee],
        filtered[ankle]
    )

    hip_segment = angle_2d(
        filtered[sh],
        filtered[hip],
        filtered[knee]
    )

    ankle_segment = angle_2d(
        filtered[knee],
        filtered[ankle],
        filtered[toe]
    )

    trunk = filtered[sh]-filtered[hip]
    trunk_angle = np.degrees(
        np.arctan2(
            trunk[:, 0],
            -trunk[:, 1]
        )
    )

    # 양쪽 고관절이 보이면 중점 사용
    # 그렇지 않으면 가까운 쪽 고관절 대체
    left_hip = filtered[23]
    right_hip = filtered[24]

    both = (
        np.isfinite(left_hip).all(axis=1)
        & np.isfinite(right_hip).all(axis=1)
    )

    pelvis = filtered[hip].copy()
    pelvis[both] = (
        left_hip[both]+right_hip[both]
    )/2

    smooth_bar = bar.copy()

    if squat_type == "바벨 스쿼트":
        for axis in range(2):
            smooth_bar[:, axis] = lowpass(
                bar[:, axis], fps, cutoff
            )

    # 영상 속 진행 방향을 +AP로 정의
    sign = 1 if direction == "오른쪽" else -1

    def position(points):
        result = np.full_like(points, np.nan)
        good = np.isfinite(points).all(axis=1)

        if not good.any() or scale is None:
            return result

        origin = points[np.where(good)[0][0]]

        result[:, 0] = (
            points[:, 0]-origin[0]
        )*scale*sign

        result[:, 1] = -(
            points[:, 1]-origin[1]
        )*scale

        return result

    pelvis_pos = position(pelvis)
    bar_pos = position(smooth_bar)

    df = pd.DataFrame({
        "frame": np.arange(start, end+1),
        "time_s": time,
        "knee_flexion_raw_deg": knee_raw,
        "knee_flexion_deg": knee_flex,
        "hip_segment_deg": hip_segment,
        "ankle_segment_deg": ankle_segment,
        "trunk_inclination_deg": trunk_angle,
        "pelvis_AP_m": pelvis_pos[:, 0],
        "pelvis_vertical_m": pelvis_pos[:, 1],
        "bar_AP_m": bar_pos[:, 0],
        "bar_vertical_m": bar_pos[:, 1]
    })

    for name in ["pelvis", "bar"]:
        for axis in ["AP", "vertical"]:
            col = f"{name}_{axis}_m"

            df[f"{name}_{axis}_velocity_ms"] = (
                derivative(
                    df[col].to_numpy(), time
                )
            )

    bottom_local = bottom-start

    phase = np.empty(n)
    phase[:bottom_local+1] = np.linspace(
        0, 50, bottom_local+1
    )
    phase[bottom_local:] = np.linspace(
        50, 100, n-bottom_local
    )
    df["phase_pct"] = phase

    summary = {
        "max_knee_flexion_deg":
            safe_max(knee_flex),
        "knee_ROM_deg":
            safe_range(knee_flex),
        "hip_segment_ROM_deg":
            safe_range(hip_segment),
        "ankle_segment_ROM_deg":
            safe_range(ankle_segment),
        "max_trunk_inclination_deg":
            safe_max(trunk_angle, True),

        "pelvis_AP_range_cm":
            safe_range(pelvis_pos[:, 0])*100,
        "pelvis_vertical_range_cm":
            safe_range(pelvis_pos[:, 1])*100,
        "bar_AP_range_cm":
            safe_range(bar_pos[:, 0])*100,
        "bar_vertical_range_cm":
            safe_range(bar_pos[:, 1])*100,

        "max_pelvis_AP_velocity_ms":
            safe_max(
                df.pelvis_AP_velocity_ms, True
            ),
        "max_pelvis_vertical_velocity_ms":
            safe_max(
                df.pelvis_vertical_velocity_ms, True
            ),
        "max_bar_AP_velocity_ms":
            safe_max(
                df.bar_AP_velocity_ms, True
            ),
        "max_bar_vertical_velocity_ms":
            safe_max(
                df.bar_vertical_velocity_ms, True
            ),

        "descent_time_s": bottom_local/fps,
        "ascent_time_s": (end-bottom)/fps,
        "total_time_s": (end-start)/fps,
        "cutoff_hz": cutoff,
        "pelvis_valid_pct":
            np.isfinite(pelvis[:, 0]).mean()*100,
        "bar_tracking_pct":
            np.isfinite(bar[:, 0]).mean()*100
            if squat_type == "바벨 스쿼트"
            else np.nan
    }

    return df, summary


# =========================================
# 6. 학생 실험 설정
# =========================================

st.header("1. 분석 조건 설정")

condition_count = st.selectbox(
    "비교 조건 수", [2, 3], index=1
)

condition_names = []
condition_types = []

for ci in range(condition_count):
    c1, c2 = st.columns(2)

    with c1:
        name = st.text_input(
            f"조건 {chr(65+ci)} 이름",
            value=f"조건 {chr(65+ci)}",
            key=f"name_{ci}"
        )

    with c2:
        kind = st.selectbox(
            f"조건 {chr(65+ci)} 운동 유형",
            ["맨몸 스쿼트", "바벨 스쿼트"],
            key=f"type_{ci}"
        )

    condition_names.append(name)
    condition_types.append(kind)

side = st.radio(
    "분석할 다리",
    ["right", "left"],
    horizontal=True,
    format_func=lambda v:
        "오른쪽" if v == "right" else "왼쪽"
)

direction = st.radio(
    "영상에서 참가자가 바라보는 방향",
    ["오른쪽", "왼쪽"],
    horizontal=True
)

st.subheader("저역통과 필터 설정")

cutoff = st.slider(
    "차단주파수 (Hz)",
    min_value=2.0,
    max_value=12.0,
    value=6.0,
    step=0.5
)

st.caption(
    "4차 Butterworth 저역통과 필터를 "
    "양방향으로 적용합니다. "
    "관절 좌표를 필터링한 후 각도와 속도를 계산합니다."
)

fps_override = st.number_input(
    "FPS 직접 입력 (0 = 영상 FPS)",
    min_value=0.0,
    value=0.0,
    step=10.0
)


# =========================================
# 7. 영상 업로드 및 보정
# =========================================

st.header("2. 영상별 보정 및 분석")

current = []

for ci in range(condition_count):
    st.subheader(
        f"조건 {chr(65+ci)} — {condition_names[ci]}"
    )

    for trial in range(1, 4):
        slot = f"{ci}_{trial}"

        with st.expander(f"{trial}회차 영상"):
            upload = st.file_uploader(
                "스쿼트 MP4 업로드",
                type=["mp4", "mov"],
                key=f"upload_{slot}"
            )

            if upload is None:
                continue

            try:
                state = prepare_file(upload, slot)
                native_fps, count = video_info(
                    state["path"]
                )
            except Exception as e:
                st.error(str(e))
                continue

            uid = f"{slot}_{state['digest']}"
            fps = (
                fps_override
                if fps_override > 0
                else native_fps
            )

            st.caption(
                f"{count}프레임 | {fps:.2f}fps"
            )

            if cutoff >= fps/2:
                st.error(
                    "차단주파수는 FPS의 절반보다 "
                    "작아야 합니다."
                )
                continue

            st.markdown("#### A. 거리 보정")

            use_scale = st.checkbox(
                "거리 보정 사용",
                value=True,
                key=f"scale_on_{uid}"
            )

            scale = None
            length_m = None
            cal_idx = None

            if use_scale:
                length_m = st.number_input(
                    "기준 길이 (m)",
                    min_value=0.01,
                    value=1.0,
                    step=0.05,
                    key=f"length_{uid}"
                )

                cal_idx = st.slider(
                    "보정 프레임",
                    0, count-1, 0,
                    key=f"cal_{uid}"
                )

                if state["cal_frame"] != cal_idx:
                    state["cal_frame"] = cal_idx
                    state["A"] = None
                    state["B"] = None
                    state["cal_version"] += 1

                cal_image = get_frame(
                    state["path"], cal_idx
                )

                click = pick_point(
                    cal_image,
                    f"calclick_{uid}_{cal_idx}_"
                    f"{state['cal_version']}",
                    [
                        ("A", state["A"], (255, 80, 0)),
                        ("B", state["B"], (0, 180, 255))
                    ]
                )

                a_col, b_col = st.columns(2)

                with a_col:
                    if st.button(
                        "A점 저장",
                        key=f"Abutton_{uid}"
                    ) and click is not None:
                        state["A"] = click.copy()
                        state["cal_version"] += 1
                        st.rerun()

                with b_col:
                    if st.button(
                        "B점 저장",
                        key=f"Bbutton_{uid}"
                    ) and click is not None:
                        state["B"] = click.copy()
                        state["cal_version"] += 1
                        st.rerun()

                if (
                    state["A"] is not None
                    and state["B"] is not None
                ):
                    px = np.linalg.norm(
                        state["A"]-state["B"]
                    )

                    if px > 1:
                        scale = length_m/px
                        st.success(
                            f"거리 보정 완료: "
                            f"{scale:.6f} m/pixel"
                        )

            st.markdown("#### B. 스쿼트 구간")

            start, end = st.slider(
                "시작 ~ 종료 프레임",
                0, count-1,
                (0, count-1),
                key=f"range_{uid}"
            )

            if end-start < 4:
                st.warning("구간을 넓혀주세요.")
                continue

            bottom = st.slider(
                "스쿼트 최저점",
                start+1, end-1,
                (start+end)//2,
                key=f"bottom_{uid}_{start}_{end}"
            )

            preview_idx = st.slider(
                "프레임 미리보기",
                start, end, bottom,
                key=f"preview_{uid}_{start}_{end}"
            )

            st.image(
                cv2.cvtColor(
                    get_frame(state["path"], preview_idx),
                    cv2.COLOR_BGR2RGB
                ),
                caption=f"프레임 {preview_idx}"
            )

            st.markdown("#### C. 바벨 추적점")

            is_barbell = (
                condition_types[ci] == "바벨 스쿼트"
            )

            if state["bar_start"] != start:
                state["bar_start"] = start
                state["bar_point"] = None
                state["bar_version"] += 1

            if is_barbell:
                bar_click = pick_point(
                    get_frame(state["path"], start),
                    f"barclick_{uid}_{start}_"
                    f"{state['bar_version']}",
                    [(
                        "BAR", state["bar_point"],
                        (255, 120, 0)
                    )]
                )

                if st.button(
                    "바벨 추적점 저장",
                    key=f"savebar_{uid}"
                ) and bar_click is not None:
                    state["bar_point"] = bar_click.copy()
                    state["bar_version"] += 1
                    st.rerun()
            else:
                st.info(
                    "맨몸 스쿼트: 바벨 추적은 생략합니다."
                )

            ready = (
                (not use_scale or scale is not None)
                and (
                    not is_barbell
                    or state["bar_point"] is not None
                )
            )

            settings = repr((
                state["digest"], fps, cutoff,
                use_scale, length_m, cal_idx,
                tuple(state["A"])
                    if state["A"] is not None else None,
                tuple(state["B"])
                    if state["B"] is not None else None,
                start, bottom, end, side, direction,
                condition_types[ci],
                tuple(state["bar_point"])
                    if state["bar_point"] is not None else None
            ))

            if st.button(
                "관절 트래킹 및 운동학 분석 시작",
                key=f"run_{uid}",
                disabled=not ready,
                type="primary"
            ):
                try:
                    progress = st.empty()
                    live = st.empty()

                    with st.spinner("AI 분석 중..."):
                        pose, bar, output_video = (
                            process_video(
                                state["path"],
                                fps, start, end,
                                state["bar_point"],
                                is_barbell,
                                live, progress
                            )
                        )

                        df, metrics = calculate_metrics(
                            pose, bar, fps, cutoff,
                            scale, side, direction,
                            start, bottom, end,
                            condition_types[ci]
                        )

                    previous = (
                        st.session_state.squat_results_v4
                        .get(slot)
                    )

                    if previous:
                        old = previous.get("video")
                        if (
                            old and old != output_video
                            and os.path.isfile(old)
                        ):
                            os.remove(old)

                    st.session_state.squat_results_v4[
                        slot
                    ] = {
                        "settings": settings,
                        "df": df,
                        "metrics": metrics,
                        "video": output_video
                    }

                    progress.empty()
                    live.empty()
                    st.success("분석 완료!")

                except Exception as e:
                    st.error(
                        f"분석 오류: "
                        f"{type(e).__name__}: {e}"
                    )

            saved = (
                st.session_state.squat_results_v4
                .get(slot)
            )

            if saved and saved["settings"] == settings:
                st.success("이 영상의 분석 결과가 준비됐습니다.")

                st.caption(
                    "영상에서 AI 관절점과 바벨 추적 위치를 "
                    "확인할 수 있습니다."
                )

                if st.checkbox(
                    "Skeleton 분석 영상 보기",
                    key=f"view_{uid}"
                ):
                    st.video(saved["video"])

                    with open(
                        saved["video"], "rb"
                    ) as f:
                        video_bytes = f.read()

                    st.download_button(
                        "Skeleton 영상 다운로드",
                        data=video_bytes,
                        file_name=(
                            f"Squat_C{ci+1}_T{trial}.mp4"
                        ),
                        mime="video/mp4",
                        key=f"download_video_{uid}"
                    )

                current.append({
                    "condition": condition_names[ci],
                    "condition_id": ci+1,
                    "trial": trial,
                    "type": condition_types[ci],
                    "df": saved["df"],
                    "metrics": saved["metrics"]
                })


# =========================================
# 8. 반복별 결과 / 평균 ± SD
# =========================================

st.header("3. 조건별 운동학 결과")

if not current:
    st.info(
        "먼저 영상을 분석하면 결과표가 표시됩니다."
    )
    st.stop()

rows = []

for item in current:
    row = {
        "condition": item["condition"],
        "condition_id": item["condition_id"],
        "trial": item["trial"],
        "type": item["type"]
    }
    row.update(item["metrics"])
    rows.append(row)

results_df = pd.DataFrame(rows)

numeric = [
    c for c in results_df.select_dtypes(
        include="number"
    ).columns
    if c not in [
        "condition_id", "trial", "cutoff_hz"
    ]
    and results_df[c].notna().any()
]

st.subheader("반복별 분석 결과")
st.dataframe(results_df.round(3))

stats_rows = []

for ci, name in enumerate(condition_names):
    subset = results_df[
        results_df.condition_id == ci+1
    ]

    for metric in numeric:
        vals = subset[metric].dropna()

        if len(vals) == 0:
            continue

        mean = vals.mean()
        sd = (
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan
        )

        stats_rows.append({
            "조건": name,
            "지표": metric,
            "N": len(vals),
            "평균": mean,
            "SD": sd,
            "평균 ± SD": (
                f"{mean:.3f} ± {sd:.3f}"
                if np.isfinite(sd)
                else f"{mean:.3f} (SD 없음)"
            )
        })

stats_df = pd.DataFrame(stats_rows)

st.subheader("조건별 평균 ± 표준편차")
st.dataframe(stats_df.round(3))


# =========================================
# 9. 조건별 평균 ± SD 막대그래프
# =========================================

st.header("4. 조건별 평균 ± SD 비교")

selected_metric = st.selectbox(
    "비교 지표",
    numeric
)

colors = [
    "royalblue",
    "seagreen",
    "darkorange"
]

fig = go.Figure()

for ci, condition in enumerate(condition_names):
    vals = results_df.loc[
        results_df.condition_id == ci+1,
        selected_metric
    ].dropna().to_numpy()

    if not len(vals):
        continue

    avg = vals.mean()
    sd = (
        vals.std(ddof=1)
        if len(vals) > 1
        else np.nan
    )

    fig.add_trace(go.Bar(
        x=[condition],
        y=[avg],
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
            f"{avg:.2f} ± {sd:.2f}"
            if np.isfinite(sd)
            else f"{avg:.2f}"
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
    title=f"{selected_metric}: Mean ± SD",
    template="plotly_white",
    height=480
)

st.plotly_chart(fig)


# =========================================
# 10. 스쿼트 곡선 평균 ± SD
# =========================================

st.header("5. 스쿼트 동작 곡선 ± SD")

available = [
    c for c in current[0]["df"].columns
    if c not in [
        "frame", "time_s", "phase_pct"
    ]
    and any(
        item["df"][c].notna().any()
        for item in current
    )
]

curve_metric = st.selectbox(
    "시간 정규화 그래프 변수",
    available
)

show_trials = st.checkbox(
    "개별 시행 곡선 표시",
    value=True
)

phase_grid = np.linspace(0, 100, 101)
curve_fig = go.Figure()
curve_count = 0

fills = [
    "rgba(65,105,225,0.16)",
    "rgba(46,139,87,0.16)",
    "rgba(255,140,0,0.16)"
]

for ci, condition in enumerate(condition_names):
    curves = []

    for item in current:
        if item["condition_id"] != ci+1:
            continue

        df = item["df"]
        x = df.phase_pct.to_numpy(dtype=float)
        y = df[curve_metric].to_numpy(dtype=float)

        good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < 5:
            continue

        # 결측이 있는 부분은 NaN으로 유지
        curve = np.interp(
            phase_grid,
            x[good], y[good],
            left=np.nan,
            right=np.nan
        )

        curves.append(curve)

        if show_trials:
            curve_fig.add_trace(go.Scatter(
                x=phase_grid,
                y=curve,
                mode="lines",
                line=dict(
                    color=colors[ci],
                    width=1,
                    dash="dot"
                ),
                opacity=0.35,
                showlegend=False
            ))

    if not curves:
        continue

    curve_count += len(curves)
    curves = np.asarray(curves)

    valid_count = np.isfinite(curves).sum(axis=0)

    avg = np.full(101, np.nan)
    sd = np.full(101, np.nan)

    for j in range(101):
        values = curves[:, j]
        values = values[np.isfinite(values)]

        if len(values) >= 1:
            avg[j] = values.mean()

        if len(values) >= 2:
            sd[j] = values.std(ddof=1)

    if len(curves) >= 2:
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
            name=f"{condition} ± SD"
        ))

    curve_fig.add_trace(go.Scatter(
        x=phase_grid,
        y=avg,
        mode="lines",
        name=f"{condition} 평균 (n={len(curves)})",
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
        "Squat cycle (%) "
        "0=시작, 50=최저점, 100=종료"
    ),
    yaxis_title=curve_metric,
    template="plotly_white",
    height=500
)

if curve_count:
    st.plotly_chart(curve_fig)
    st.caption(
        f"그래프에 포함된 유효 시행: {curve_count}회. "
        "조건별 2회 이상일 때 SD를 표시합니다."
    )
else:
    st.warning(
        "선택한 변수에 유효한 좌표가 부족합니다. "
        "다른 변수를 선택하거나 영상 추적 상태를 확인하세요."
    )


# =========================================
# 11. 원본과 필터 결과 비교
# =========================================

st.header("6. 필터 적용 전후 비교")

selected_trial = st.selectbox(
    "비교할 영상",
    range(len(current)),
    format_func=lambda i: (
        f"{current[i]['condition']} "
        f"{current[i]['trial']}회"
    )
)

selected_df = current[selected_trial]["df"]

filter_fig = go.Figure()

filter_fig.add_trace(go.Scatter(
    x=selected_df.time_s,
    y=selected_df.knee_flexion_raw_deg,
    name="필터 적용 전",
    line=dict(color="gray", width=1)
))

filter_fig.add_trace(go.Scatter(
    x=selected_df.time_s,
    y=selected_df.knee_flexion_deg,
    name=f"필터 적용 후 ({cutoff} Hz)",
    line=dict(color="royalblue", width=3)
))

filter_fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Knee flexion (deg)",
    template="plotly_white"
)

st.plotly_chart(filter_fig)


# =========================================
# 12. 바벨 / 골반 이동 궤적
# =========================================

st.header("7. 전후·수직 이동 궤적")

target = st.radio(
    "분석 대상",
    ["pelvis", "bar"],
    horizontal=True,
    format_func=lambda x:
        "골반 대표점" if x == "pelvis"
        else "바벨"
)

trajectory = go.Figure()
trajectory_count = 0

for item in current:
    df = item["df"]
    x = df[f"{target}_AP_m"]*100
    y = df[f"{target}_vertical_m"]*100

    if not (
        np.isfinite(x).any()
        and np.isfinite(y).any()
    ):
        continue

    trajectory_count += 1

    trajectory.add_trace(go.Scatter(
        x=x, y=y,
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
    scaleanchor="x", scaleratio=1
)

if trajectory_count:
    st.plotly_chart(trajectory)
else:
    st.info("선택한 대상에 거리 보정 데이터가 없습니다.")


# =========================================
# 13. Excel 다운로드
# =========================================

st.header("8. 결과 다운로드")

buffer = io.BytesIO()

with pd.ExcelWriter(
    buffer, engine="openpyxl"
) as writer:
    results_df.to_excel(
        writer, "Trials", index=False
    )

    stats_df.to_excel(
        writer, "Mean_SD", index=False
    )

    for item in current:
        ci = item["condition_id"]
        tr = item["trial"]

        item["df"].to_excel(
            writer,
            sheet_name=f"C{ci}_T{tr}",
            index=False
        )

st.download_button(
    "전체 분석 결과 Excel 다운로드",
    data=buffer.getvalue(),
    file_name="Squat_Analysis_Results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)

st.warning(
    "이 프로그램은 2D 영상 기반 운동학 분석 도구입니다. "
    "골반 대표점은 전신 COM이 아닙니다. "
    "관절각도, 거리, 속도 및 바벨 추적 결과는 "
    "영상 품질과 보정 정확도에 따라 오차가 발생합니다."
)
