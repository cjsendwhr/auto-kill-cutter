import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
import os

# --- 웹사이트 설정 ---
st.set_page_config(page_title="게임 킬 장면 컷편집기", page_icon="✂️")
st.title("✂️ 게임 킬 장면 자동 컷편집기")
st.markdown("""
**사용법:**
1. 게임 녹화 영상과 **'킬 로그 이미지'**를 업로드하세요.
2. 프로그램이 킬 로그가 뜬 시간을 찾아 **앞뒤 1초씩(총 2초)** 자동으로 잘라줍니다.
""")

# --- 사이드바: 설정 옵션 ---
st.sidebar.header("⚙️ 설정")
threshold = st.sidebar.slider(
    "민감도 설정 (기본값: 0.8)", 
    min_value=0.5, 
    max_value=0.99, 
    value=0.8, 
    step=0.01,
    help="킬 장면을 잘 못 찾으면 숫자를 낮추고(0.6~0.7), 엉뚱한 장면을 자르면 숫자를 높이세요(0.85~0.9)."
)

# 1. 파일 업로드
uploaded_video = st.file_uploader("1. 게임 영상 파일 (MP4)", type=["mp4", "mov", "avi"])
uploaded_icon = st.file_uploader("2. 킬 로그 이미지 (PNG, JPG)", type=["png", "jpg", "jpeg"])

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
        st.info("영상을 분석 중입니다... (영상 길이에 따라 시간이 걸립니다)")
        
        # 파일 저장
        video_path = save_uploaded_file(uploaded_video)
        icon_path = save_uploaded_file(uploaded_icon)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. OpenCV로 킬 장면 시간(Timestamp) 찾기
            cap = cv2.VideoCapture(video_path)
            icon = cv2.imread(icon_path, cv2.IMREAD_COLOR)
            
            # 이미지 읽기 실패 시 예외 처리
            if icon is None:
                st.error("이미지 파일을 읽을 수 없습니다. 다른 이미지로 시도해주세요.")
                cap.release()
            else:
                timestamps = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: fps = 30.0 # 기본값 방어
                
                frame_idx = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 5프레임마다 검사 (속도 최적화)
                    if frame_idx % 5 == 0:
                        # 템플릿 매칭 (이미지 찾기)
                        try:
                            result = cv2.matchTemplate(frame, icon, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                            
                            # 설정한 민감도보다 높으면 '킬'로 인식
                            if max_val >= threshold:
                                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                                
                                # 중복 방지 (이전 킬 장면과 3초 이내면 무시)
                                if not timestamps or (current_time - timestamps[-1] > 3):
                                    timestamps.append(current_time)
                        except Exception as e:
                            # 이미지 크기가 영상보다 클 경우 등 에러 무시
                            pass
                            
                    frame_idx += 1
                    # 진행률 표시 (전체의 50%까지는 분석 단계)
                    if frame_idx % 100 == 0:
                        prog = int((frame_idx / total_frames) * 50)
                        progress_bar.progress(min(50, prog))
                        status_text.text(f"분석 중... {frame_idx}/{total_frames} 프레임")
                
                cap.release()
                
                # 2. MoviePy로 영상 자르기
                if timestamps:
                    status_text.text(f"🔫 총 {len(timestamps)}개의 킬 장면 발견! 자르는 중...")
                    clip = VideoFileClip(video_path)
                    clips = []
                    
                    for idx, t in enumerate(timestamps):
                        # 킬 발생 시점 기준: 앞 1초 ~ 뒤 1초 (총 2초)
                        start = max(0, t - 1)
                        end = min(clip.duration, t + 1)
                        
                        sub = clip.subclip(start, end)
                        clips.append(sub)
                        
                        # 진행률 (50% ~ 90%)
                        prog = 50 + int((idx / len(timestamps)) * 40)
                        progress_bar.progress(min(90, prog))
                    
                    # 조각 영상 합치기
                    final_clip = concatenate_videoclips(clips)
                    
                    # 결과 파일 저장
                    output_path = tempfile.mktemp(suffix=".mp4")
                    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True)
                    
                    progress_bar.progress(100)
                    status_text.success("🎉 편집 완료!")
                    
                    # 다운로드 버튼
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 하이라이트 영상 다운로드",
                            data=file,
                            file_name="kill_highlight.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("킬 장면을 찾지 못했습니다. 왼쪽 사이드바에서 '민감도'를 낮춰서(0.6~0.7) 다시 시도해보세요!")
                    
        except Exception as e:
            st.error(f"오류 발생: {e}")
            
        finally:
            # 임시 파일 삭제 (청소)
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(icon_path): os.remove(icon_path)
            
    else:
        st.warning("영상과 이미지 파일을 모두 업로드해주세요.")