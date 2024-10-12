# 필요한 모듈 및 라이브러리 임포트
from fastapi import FastAPI, HTTPException, Body
import os
import cv2
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe as mp
from ShazamAPI import Shazam
import hashlib
import httpx
import shutil
import io

# FastAPI 애플리케이션 생성
app = FastAPI()

# 다운로드 및 데이터 저장 경로 설정
download_path = "/app/data/"

# Mediapipe 라이브러리를 이용한 포즈 추출 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
selected_keypoints = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# HTTP 요청 타임아웃 설정 (초 단위)
timeout_in_seconds = 600

# HTTP 클라이언트 설정, 게이트웨이(gateway.py)의 URL
gateway = httpx.AsyncClient(base_url="http://gateway-service:8000", timeout=httpx.Timeout(timeout_in_seconds))

# YouTube 다운로드 및 키포인트 추출 경로 설정, 서비스 경로 핸들러
@app.post("/download_youtube")
async def download_youtube(youtube_data: dict = Body(...)):
    # 유튜브 URL 가져오기
    youtube_url = youtube_data.get("youtube_url")
    if not youtube_url:
        raise HTTPException(status_code=400, detail="youtube_url 필드가 필요합니다")

    # URL 해시 생성
    h_youtube_url = hashlib.sha1(youtube_url.encode()).hexdigest()[-10:]

    # 데이터가 이미 존재하는지 확인, 해시 변환된 URL로 비교.
    response = await gateway.post("/check_existing_data_mariadb", json={"url": h_youtube_url})
    response.raise_for_status()
    existing_data = response.json()

    if existing_data:
        # 이미 데이터가 존재할 경우 정보 반환
        video_path = existing_data["video_path"]
        audio_path = existing_data["audio_path"]
        song_info = {"title": existing_data["title"], "artist": existing_data["artist"]}

        response = await gateway.post("/find_data_mongodb", json={
            "youtube_url": h_youtube_url,
            "title": existing_data["title"],
            "artist": existing_data["artist"]
        })
        res = response.json()

        return {"message": "이미 해당 URL의 데이터가 존재합니다.",
                "h_youtube_url": h_youtube_url,
                "title": song_info["title"],  # 음악 제목
                "artist": song_info["artist"],  # 아티스트 이름
                "keypoints":  res["keypoints"]}
    else:
        # 데이터가 없는 경우 다운로드 및 추출 프로세스 수행
        yt = YouTube(youtube_url)  # pytube를 사용하여 YouTube 객체 생성
        song_info = recognize_song_info(yt, h_youtube_url)  # 음악 정보 인식 함수 호출
        folder_name = f"{h_youtube_url}_{song_info['title']}_{song_info['artist']}"  # 폴더 이름 생성
        folder_path = os.path.join(download_path, folder_name)  # 폴더 경로 생성
        os.makedirs(folder_path)  # 생성한 폴더 경로에 폴더 생성
        video_filename = f"{folder_name}.mp4"  # 비디오 파일 이름 생성
        audio_filename = f"{folder_name}.mp3"  # 오디오 파일 이름 생성
        video_path = os.path.join(folder_path, video_filename)  # 비디오 파일 경로 생성
        audio_path = os.path.join(folder_path, audio_filename)  # 오디오 파일 경로 생성

        stream = yt.streams.get_highest_resolution()  # 가장 높은 해상도의 스트림 가져오기
        stream.download(output_path=folder_path, filename=video_filename)  # 비디오 다운로드

        video_clip = VideoFileClip(video_path)  # VideoFileClip 객체 생성
        audio_clip = video_clip.audio  # 비디오에서 오디오 추출
        audio_clip.write_audiofile(audio_path)  # 오디오 파일로 저장

        new_video_path = change_frame_rate_and_save(folder_path, video_path, target_fps=30)

        keypoints_list, boxsizes_list = process_keypoints(new_video_path)  # 비디오에서 키포인트 추출

        # 데이터 객체 생성
        data = {
            "h_youtube_url": h_youtube_url,  # 해시 처리한 YouTube URL
            "title": song_info["title"],  # 음악 제목
            "artist": song_info["artist"], # 아티스트 이름
            "keypoints": keypoints_list, # 키포인트 리스트를 데이터에 포함
            "boxsizes": boxsizes_list
        }

        # MongoDB 및 MariaDB에 데이터 저장
        await gateway.post("/insert_new_data_mongodb", json=data)
        await insert_new_data_via_http(h_youtube_url, video_path, audio_path, song_info["title"],
                                       song_info["artist"])

        delete_file_or_folder(folder_path)

        return {"message": "다운로드 및 키포인트 추출이 완료되었습니다.",
                "h_youtube_url": h_youtube_url,
                "title": song_info["title"],  # 음악 제목
                "artist": song_info["artist"],  # 아티스트 이름
                "keypoints": keypoints_list}


# HTTP 클라이언트를 통한 MariaDB에 데이터 삽입
async def insert_new_data_via_http(h_youtube_url, video_path, audio_path, title, artist):
    data = {
        "h_youtube_url": h_youtube_url,  # 해시 처리한 YouTube URL
        "video_path": video_path,  # 다운로드한 비디오 파일 경로
        "audio_path": audio_path,  # 추출한 오디오 파일 경로
        "title": title,  # 음악 제목
        "artist": artist  # 아티스트 이름
    }

    # HTTP 클라이언트를 비동기적으로 생성하여 데이터 삽입 요청 전송
    async with httpx.AsyncClient() as client:
        response = await gateway.post("/insert_new_data_mariadb", json=data)  # 게이트웨이를 통해 데이터 삽입 요청 전송
        response.raise_for_status()  # 응답 상태 코드 확인


def delete_file_or_folder(path):
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                print(f"File {path} deleted successfully.")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Folder {path} and its contents deleted successfully.")
        else:
            print(f"Path {path} not found. Skipping deletion.")
    except Exception as e:
        print(f"An error occurred while deleting: {e}")


# 30 프레임 변환

def change_frame_rate_and_save(folder_path,video_path, target_fps=30):
    # 영상을 다운로드하고 파일 경로 획득

    # 영상 불러오기
    clip = VideoFileClip(video_path)

    # fps를 목표 프레임 속도로 변경
    clip_30fps = clip.set_fps(target_fps)

    # 저장 경로 설정
    video_filename = f"_30fps.mp4"
    changed_path = os.path.join(folder_path, video_filename)

    # 변경된 영상 저장
    clip_30fps.write_videofile(changed_path)

    print(f"변환 완료: {changed_path}")
    return changed_path


# 비디오에서 키포인트 추출
def process_keypoints(video_path):
    keypoints_list = []  # 추출된 키포인트 리스트 초기화
    boxsizes_list = []
    cap = cv2.VideoCapture(video_path)  # 비디오 파일을 열고 캡처 객체 생성
    frames = []  # 비디오의 프레임들을 저장할 리스트 초기화

    while cap.isOpened():
        ret, frame = cap.read()  # 비디오 프레임 읽기
        if not ret:
            break
        frames.append(frame)  # 프레임 저장

    for frame in frames:
        keypoints, boxsizes = get_keypoints_and_boxsize(frame)  # 이미지에서 키포인트 추출
        keypoints_list.append(keypoints)  # 추출한 키포인트 리스트에 추가
        boxsizes_list.append(boxsizes)

    cap.release()  # 캡처 객체 종료
    return keypoints_list, boxsizes_list  # 추출된 키포인트 리스트 반환


# Get keypoints data & bounded box size from 1 frame
def get_keypoints_and_boxsize(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    keypoints = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in selected_keypoints:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if xmin == 0:
                xmin, ymin, zmin = landmark.x, landmark.y, landmark.z

            else:
                xmin, xmax, ymin, ymax, zmin, zmax = min(xmin, landmark.x), max(xmax, landmark.x), min(ymin,
                                                                                                       landmark.y), max(
                    ymax, landmark.y), min(zmin, landmark.z), max(zmax, landmark.z)

    boxsize = (xmin, xmax, ymin, ymax, zmin, zmax)
    boxsize = [boxsize[2 * i + 1] - boxsize[2 * i] for i in range(3)]

    return keypoints, boxsize

# 이미지에서 포즈 키포인트 추출
def get_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 공간 변환 (OpenCV의 BGR을 Mediapipe의 RGB로 변환)
    results = pose.process(image_rgb)  # 포즈 추출 수행
    keypoints = []  # 추출한 키포인트들을 저장할 리스트 초기화

    if results.pose_landmarks:  # 포즈 랜드마크가 존재하는 경우
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in selected_keypoints:  # 선택된 키포인트만 골라내기
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])  # x, y, z, 가시성 값 저장

    return keypoints  # 추출한 키포인트 리스트 반환


# 음악 정보 추출 (ShazamAPI 사용)
async def recognize_song_info(youtube_obj, url):
    try:
        audio_stream = youtube_obj.streams.filter(only_audio=True).first()
        audio_content = io.BytesIO()
        audio_stream.stream_to_buffer(audio_content)  # 오디오 스트림을 메모리로 바로 다운로드
        audio_content.seek(0)

        song_info = await get_song_info(audio_content, url)
        logging.info("음악 정보 추출 완료.")
    except Exception as e:
        logging.error(f"음악 정보 추출 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="음악 정보 추출 중 오류가 발생했습니다.")

    return song_info

# ShazamAPI를 이용하여 음악 정보 추출
def get_song_info(audio_filepath, url):
    mp3_file_content_to_recognize = open(audio_filepath, 'rb').read()  # 오디오 파일을 바이트 형식으로 읽기
    shazam = Shazam(mp3_file_content_to_recognize)  # Shazam 객체 생성
    try:
        response = shazam.recognizeSong()  # 음악 정보 인식 요청
        first_item = next(response)  # 첫 번째 결과 아이템 가져오기

        if 'track' in first_item[1]:  # 'track' 정보가 있는 경우
            title = first_item[1]['track'].get('title', 'Unknown')  # 음악 제목 가져오기
            artist = first_item[1]['track'].get('subtitle', 'Unknown')  # 아티스트 이름 가져오기
        else:
            title = "Unknown_" + str(url)  # 정보가 없는 경우 "Unknown"과 URL로 대체
            artist = "Unknown_" + str(url)

        return {"title": title, "artist": artist}  # 추출된 음악 정보 반환

    except StopIteration:
        return {"title": "Unknown", "artist": "Unknown"}  # 인식 실패 시 "Unknown" 정보 반환