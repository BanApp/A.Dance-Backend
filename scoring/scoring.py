# 필요한 모듈 및 라이브러리 임포트
import mediapipe as mp # Mediapipe 라이브러리를 임포트합니다.
import httpx # HTTP 클라이언트를 위한 httpx 라이브러리를 임포트합니다.
import numpy as np # 배열 및 수학 연산을 위한 NumPy 라이브러리를 임포트합니다.
import cv2 # OpenCV 라이브러리를 임포트합니다.
from fastapi import FastAPI, UploadFile, Body, File
import shutil # 파일 복사 및 삭제를 위한 shutil 모듈을 임포트합니다.
import os # 운영 체제 관련 작업을 위한 os 모듈을 임포트합니다.
import json # JSON 데이터 처리를 위한 모듈을 임포트합니다.

# FastAPI 애플리케이션 생성
app = FastAPI()

# Mediapipe 라이브러리를 이용한 포즈 추출 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 사용할 키포인트 이름 및 선택된 키포인트 인덱스 목록 설정
keypoint_names = [
    "nose", "left eye (inner)", "left eye", "left eye (outer)", "right eye (inner)",
    "right eye", "right eye (outer)", "left ear", "right ear", "mouth (left)",
    "mouth (right)", "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left pinky", "right pinky", "left index",
    "right index", "left thumb", "right thumb", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle", "left heel",
    "right heel", "left foot index", "right foot index"
]

selected_keypoints = [0,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

connections = [
    (0,1), (0,2), # Nose to Ears
    (3,5), (4,6), # Shoulders to Elbows
    (5,7), (6,8), # Elbows to Wrists
    (9,11), (10,12), # Hips to Knees
    (11,13), (12,14), # Knees to Ankles
    (3, 4), (4, 10), (10, 9), (9, 3) # body
]

# HTTP 요청 타임아웃 설정 (초 단위)
timeout_in_seconds = 600

# HTTP 클라이언트 설정, 게이트웨이(gateway.py)의 URL
gateway = httpx.AsyncClient(base_url="http://gateway-service:8000", timeout=httpx.Timeout(timeout_in_seconds))

download_path = "/app/data/"

oks_cnt = [[] for _ in range(11)]
pck_cnt = [[] for _ in range(11)]


# YouTube 다운로드 및 키포인트 추출 경로 설정, 서비스 경로 핸들러
@app.post("/scoring")
async def Scoring(youtube_data: str = Body(...), video_file: UploadFile = File(...)):

    # 유튜브 URL 가져오기
    youtube_data = json.loads(youtube_data)
    youtube_url = youtube_data.get("youtube_url")
    title = youtube_data.get("title")
    artist = youtube_data.get("artist")
    username = youtube_data.get("username")

    # video_path는 업로드된 파일의 경로로 설정
    video_path = f"{video_file.filename}"  # 업로드된 파일이 저장될 경로

    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    # MongoDB에 데이터 찾기 (gateway로 요청)
    response = await gateway.post("/find_data_mongodb", json={
        "youtube_url": youtube_url,
        "title": title,
        "artist": artist})

    # MongoDB에서 받아온 데이터 파싱
    answer = response.json()
    answer_keypoints = answer["keypoints"]
    box_size = answer["boxsizes"]
    frame_cnt = 0

    # 업로드된 동영상 파일을 열기
    cap1 = cv2.VideoCapture(video_path)

    # 각 프레임에서의 OKS 및 pck의 리스트
    oks_list = []
    pck_list = []

    # 사용자의 키포인트 리스트
    user_keypoints = []

    # 동영상의 모든 프레임을 처리
    while cap1.isOpened() and frame_cnt <= len(answer_keypoints)-1:
        ret1, frame = cap1.read()
        frame1 = cv2.flip(frame, 1)

        if ret1:
            # 현재 프레임에서 사용자의 키포인트 및 바운딩 박스 크기 가져오기
            user_key, _ = get_keypoints_and_boxsize(frame1)
            user_keypoints.append(user_key)

            # 만약 정답 키포인트와 사용자 키포인트가 존재하면 점수 계산
            if len(answer_keypoints[frame_cnt]) > 0 and len(user_key) > 0:
                oks_percent, pck_percent = weighted_similarity(np.array(answer_keypoints[frame_cnt]), np.array(user_key),
                                                               box_size[frame_cnt])  # Calculate Scores from each frame
                oks_cnt[int(oks_percent / 10)].append(frame_cnt)
                pck_cnt[int(pck_percent / 10)].append(frame_cnt)

                oks_list.append(oks_percent)
                pck_list.append(pck_percent)
        else:
            break
        frame_cnt = frame_cnt + 1

    # MariaDB에 점수 데이터 삽입을 위해 데이터 준비
    score_data = {
        "username": username,
        "title": title,
        "artist": artist,
        "score": np.mean(pck_list) # 평균 PCK 점수 계산
    }

    # MongoDB에 데이터 삽입 (gateway로 요청)
    response = await gateway.post("/insert_new_score_data_mariadb", json=score_data)

    oks_answer = np.mean(oks_list)
    pck_answer = np.mean(pck_list)
    print("oks =", oks_answer, "pck =", pck_answer)

    # JSON 응답에 넣을 데이터를 딕셔너리로 만듦 (값들을 float로 변환)
    response_data = {
        "oks_30": oks_answer,
        "pck_30": pck_answer,
        "oks_frame_score": oks_list,
        "pck_frame_score": pck_list
    }
    delete_file_or_folder(video_path)
    cap1.release()

    return response_data


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


# 정답 프레임에서 키포인트 데이터 및 바운딩 박스 크기를 가져오는 함수
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


# OKS 값 계산 함수
def oks(gt, preds, idx, boxsize):
    sigmas = np.array([.026, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
    dx = gt[0] - preds[0]
    dy = gt[1] - preds[1]
    bbox_gt = boxsize[0] ** 2 + boxsize[1] ** 2
    kp_c = sigmas[idx]
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * (bbox_gt) * (kp_c ** 2)))


# PCK 값 계산 함수
def pck(gt, preds, threshold):
    dx = gt[0] - preds[0]
    dy = gt[1] - preds[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    return 1.0 if distance < threshold else 0.0


# 가중치가 적용된 유사도 계산 함수
def weighted_similarity(keypoints1, keypoints2, boxsize):

    oks_list = []
    pck_list = []
    for i in range(len(keypoints1)):
        oks_list.append(oks(keypoints1[i][:3], keypoints2[i][:3], i, boxsize))
        pck_list.append(pck(keypoints1[i][:3], keypoints2[i][:3], 0.1))

    return (np.mean(oks_list)) * 100, (np.mean(pck_list)) * 100


# 키포인트 리스트의 평균 좌표 계산 함수
def mean_value_of_keypoints(keypoints):
    mean_of_keypoints = np.zeros_like(keypoints[0])
    for key in keypoints:
        mean_of_keypoints += key

    mean_of_keypoints /= len(keypoints)
    return mean_of_keypoints