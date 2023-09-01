import httpx
from fastapi import FastAPI,UploadFile, HTTPException, Form
import json

# FastAPI 애플리케이션 생성
app = FastAPI()

# HTTP 요청 타임아웃 설정 (초 단위)
timeout_in_seconds = 600

# 게이트웨이 클래스 정의
class Gateway:
    def __init__(self):
        # 유튜브 영상 다운로드 및 키포인트 추출 서비스(download_youtube.py)
        self.client_8001 = httpx.AsyncClient(base_url="http://download-youtube-service:8001", timeout=httpx.Timeout(timeout_in_seconds))

        # 안무 채점 서비스(scoring.py)
        self.client_8003 = httpx.AsyncClient(base_url="http://scoring-service:8003", timeout=httpx.Timeout(timeout_in_seconds))

        # mariadb 접근 서비스(mariadb_connector.py)
        self.client_8004 = httpx.AsyncClient(base_url="http://mariadb-connector-service:8004", timeout=httpx.Timeout(timeout_in_seconds))

        # mongodb 접근 서비스(mongodb_connector.py)
        self.client_8005 = httpx.AsyncClient(base_url="http://mongodb-connector-service:8005", timeout=httpx.Timeout(timeout_in_seconds))


    # 유튜브 영상 다운로드 및 키포인트 추출 서비스 호출, 유튜브 원본 링크 전달 / 서비스 요청 메소드
    async def user_request_download_youtube(self, youtube_data: dict):
        response = await self.client_8001.post("/download_youtube", json=youtube_data)
        response.raise_for_status()
        return response.json()

    # MariaDB에 해쉬 변환된 유튜브 링크가 있는지 확인, 데이터 보유 유무 확인 / 서비스 요청 메소드
    async def check_existing_data_mariadb(self, url: dict):
        response = await self.client_8004.post("/check_existing_data_mariadb", json=url)  # 엔드포인트와 매개변수 설정
        response.raise_for_status()
        return response.json()

    # MariaDB에 새로운 데이터(영상 정보, 파일 경로 등) 넣기 / 서비스 요청 메소드
    async def insert_new_data_mariadb(self, data: dict):
        await self.client_8004.post("/insert_new_data_mariadb", json=data)

    # MariaDB에 새로운 데이터(영상 정보, 파일 경로 등) 넣기 / 서비스 요청 메소드
    async def insert_new_score_data_mariadb(self, data: dict):
        await self.client_8004.post("/insert_new_score_data_mariadb", json=data)

    # MongoDB에 새로운 데이터(영상 정보, 파일 경로, 원본 키포인트 값 등) 넣기 / 서비스 요청 메소드
    async def insert_new_data_into_mongodb(self, data: dict):
        await self.client_8005.post("/insert_new_data_mongodb", json=data)

    # MongoDB에 원본 키포인트 값 조회(해쉬변환 url, title, artist 정보를 보내고, 셋 다 일치하는 Document 전체를 가져옴) / 서비스 요청 메소드
    async def find_data_mongodb(self, data: dict):
        response = await self.client_8005.post("/find_data_mongodb", json=data)
        return response.json()

    # 인기곡 반환 서비스 요청 메소드
    async def get_hot_contents(self, num: int):
        response = await self.client_8004.get(f"/get_hot_contents/{num}")
        response.raise_for_status()
        return response.json()

    # 노래 제목으로 사용자 성적 조회(리더보드) 서비스 요청 메소드
    async def get_leaderboard(self, data: dict):
        response = await self.client_8004.post("/get_leaderboard", json=data)
        response.raise_for_status()
        return response.json()

    # 사용자 영상 안무 키포인트 추출 및 채점 / 서비스 요청 메소드
    async def scoring(self, youtube_data: dict, video_file: UploadFile):
        try:
            response = await self.client_8003.post(
                "/scoring",
                data={'youtube_data': json.dumps(youtube_data)},  # JSON 데이터를 문자열로 변환하여 전달
                files={'video_file': (video_file.filename, video_file.file, video_file.content_type)}  # 파일
            )
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail="Gateway <scoring> Error")

    async def close(self):
        await self.client_8001.aclose()
        await self.client_8003.aclose()
        await self.client_8004.aclose()
        await self.client_8005.aclose()

# 게이트웨이 객체 생성
gateway = Gateway()

# 유튜브 영상 다운로드 및 키포인트 추출 서비스 경로 핸들러, 앱에서 직접 접근
@app.post("/api/user_request_download_youtube")
async def user_request_download_youtube(youtube_data: dict):
    try:
        response = await gateway.user_request_download_youtube(youtube_data)
        return response
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <user_request_download_youtube> Error")


# MariaDB에 해쉬 변환된 유튜브 링크가 있는지 확인, 데이터 보유 유무 확인 서비스 경로 핸들러
@app.post("/check_existing_data_mariadb")
async def check_existing_data(url: dict):
    try:
        existing_data = await gateway.check_existing_data_mariadb(url)
        return existing_data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Gateway <check_existing_data> Error")


# MariaDB에 새로운 데이터(영상 정보, 파일 경로 등) 넣기 서비스 경로 핸들러
@app.post("/insert_new_data_mariadb")
async def insert_new_data_mariadb(data: dict):
    try:
        await gateway.insert_new_data_mariadb(data)
        return {"message": "Data inserted successfully."}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <insert_new_data_mariadb> Error")


# MariaDB에 새로운 데이터(영상 정보, 파일 경로 등) 넣기 서비스 경로 핸들러
@app.post("/insert_new_score_data_mariadb")
async def insert_new_score_data_mariadb(data: dict):
    try:
        await gateway.insert_new_score_data_mariadb(data)
        return {"message": "Score Data inserted successfully."}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <insert_new_score_data_mariadb> Error")


# MongoDB에 새로운 데이터(영상 정보, 파일 경로, 원본 키포인트 값 등) 넣기 서비스 경로 핸들러
@app.post("/insert_new_data_mongodb")
async def insert_new_data_mongodb(data: dict):
    try:
        await gateway.insert_new_data_into_mongodb(data)
        return {"message": "Data inserted successfully."}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <insert_new_data_mongodb> Error")


# MongoDB에 원본 키포인트 값 조회(해쉬변환 url, title, artist 정보를 보내고, 셋 다 일치하는 Document 전체를 가져옴) 서비스 경로 핸들러
@app.post("/find_data_mongodb")
async def find_data_mongodb(data: dict):
    try:
        response = await gateway.find_data_mongodb(data)
        return response
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <find_data_mongodb> Error")


# 인기곡 반환 서비스 경로 핸들러
@app.get("/api/get_hot_contents/{num}")
async def get_hot_contents(num: int):
    try:
        response = await gateway.get_hot_contents(num)
        return response
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <get_hot_contents> Error")


# 노래 제목으로 사용자 성적 조회(리더보드) 서비스 경로 핸들러
@app.post("/api/get_leaderboard")
async def get_leaderboard(data: dict):
    try:
        title = data.get("title")
        num = data.get("num")
        if not title:
            raise HTTPException(status_code=400, detail="title field is required")
        if not num:
            raise HTTPException(status_code=400, detail="num field is required")
        response = await gateway.get_leaderboard(data)
        return response
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <get_leaderboard> Error")


# 사용자 영상 안무 키포인트 추출 및 채점 서비스 경로 핸들러
@app.post("/api/scoring")
async def scoring(data: str = Form(...), file: UploadFile = Form(...)):
    try:
        # data를 딕셔너리로 변환
        data_dict = json.loads(data)
        # 'data_dict' 딕셔너리와 'file' 업로드 파일을 사용하여 처리
        response = await gateway.scoring(data_dict, file)

        return response
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail="Gateway <scoring> Error")