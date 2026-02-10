import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
import os

# --- 웹사이트 설정 ---
st.set_page_config(page_title="게임 킬 장면 컷편집기", page_icon="✂️")
st.title("✂️ 게임 킬 장면 자동 컷편집기 (강화판)")
st.markdown("""
**💡 꿀팁:**
1. **반드시 영상 파일을 재생시키고, 그 화면을 캡처**해서 아이콘으로 쓰세요. (해상도 일치 필수!)
2. 인식이 안 되면 왼쪽 사이드바에서 **민감도**를 조절하세요.
""")

# --- 사이드바: 설정 옵션 ---
st.sidebar.header("⚙️ 설정")
threshold = st.sidebar.slider(
    "민감도 (기본값: 0.7)", 
    min_value=0.4, 
    max_value=0.9, 
    value=0.7, 
    step=0.05,
    help="못 찾으면 숫자를 낮추세요(0.5~0.6). 엉뚱한 걸 자르면 높이세요."
)

use_grayscale = st.sidebar.checkbox("흑백 모드로 찾기 (추천)", value=True, help="색깔을 무시하고 모양만 봅니다. 인식률이 좋습니다.")

# 1. 파일 업로드
uploaded_video = st.file_uploader("1. 게임 영상 파일", type=["mp4", "mov", "avi", "mkv"])
uploaded_icon = st.file_uploader("2. 킬 로그 이미지", type=["png", "jpg", "jpeg"])

# 임시 파일 저장 함수
def save_uploaded_file(uploaded_file):
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"파일 저장 오류: {e}")
        return None

# --- 메인 로직 ---
if st.button("🚀 컷편집 시작!"):
    if uploaded_video and uploaded_icon:
        st.info("영상을 분석 중입니다... ☕ 잠시만 기다려주세요.")
        
        video_path = save_uploaded_file(uploaded_video)
        icon_path = save_uploaded_file(uploaded_icon)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 준비
            cap = cv2.VideoCapture(video_path)
            # 이미지 읽기
            if use_grayscale:
                icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
            else:
                icon = cv2.imread(icon_path, cv2.IMREAD_COLOR)

            if icon is None:
                st.error("이미지를 읽을 수 없습니다.")
                cap.release()
            else:
                timestamps = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: fps = 30.0
                
                frame_idx = 0
                
                # 2. 영상 스캔
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 5프레임마다 검사
                    if frame_idx % 5 == 0:
                        try:
                            # 흑백 모드 변환
                            if use_grayscale:
                                search_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            else:
                                search_frame = frame

                            # 매칭 시작
                            result = cv2.matchTemplate(search_frame, icon, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                            
                            if max_val >= threshold:
                                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                                
                                # 중복 방지 (3초 쿨타임)
                                if not timestamps or (current_time - timestamps[-1] > 3):
                                    timestamps.append(current_time)
                                    print(f"Found at {current_time}s (Accuracy: {max_val:.2f})")
                        except Exception as e:
                            pass
                            
                    frame_idx += 1
                    if frame_idx % 100 == 0:
                        prog = int((frame_idx / total_frames) * 50)
                        progress_bar.progress(min(50, prog))
                        status_text.text(f"🔍 킬 로그 찾는 중... ({int(frame_idx/total_frames*100)}%)")
                
                cap.release()
                
                # 3. 결과 처리
                if timestamps:
                    status_text.text(f"✂️ {len(timestamps)}개의 킬 장면을 자르고 있습니다...")
                    clip = VideoFileClip(video_path)
                    clips = []
                    
                    for idx, t in enumerate(timestamps):
                        start = max(0, t - 2) # 킬 2초 전 (여유 있게 수정)
                        end = min(clip.duration, t + 2) # 킬 2초 후
                        sub = clip.subclip(start, end)
                        clips.append(sub)
                        
                        prog = 50 + int((idx / len(timestamps)) * 40)
                        progress_bar.progress(min(90, prog))
                    
                    final_clip = concatenate_videoclips(clips)
                    
                    output_path = tempfile.mktemp(suffix=".mp4")
                    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True)
                    
                    progress_bar.progress(100)
                    status_text.success(f"🎉 편집 완료! {len(timestamps)}개의 킬 장면을 합쳤습니다.")
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 영상 다운로드",
                            data=file,
                            file_name="kill_highlight.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("😭 킬 장면을 하나도 못 찾았습니다.")
                    st.warning("""
                    **해결 방법:**
                    1. 동영상 파일을 재생하고 **일시정지 한 상태에서 캡처**했나요? (해상도가 다르면 못 찾습니다)
                    2. 왼쪽 설정에서 **'민감도'를 0.5 ~ 0.6**으로 낮추고 다시 해보세요.
                    """)
                    
        except Exception as e:
            st.error(f"오류: {e}")
        finally:
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(icon_path): os.remove(icon_path)
