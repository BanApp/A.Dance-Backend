import httpx
from fastapi import FastAPI, HTTPException
import motor.motor_asyncio
from contextlib import asynccontextmanager

app = FastAPI()

class MongoService:
    def __init__(self, host, port, database):
        self.mongo_host = host  # MongoDB 호스트 주소
        self.mongo_port = port  # MongoDB 포트 번호
        self.mongo_database = database  # MongoDB 데이터베이스 이름

    @asynccontextmanager
    async def get_db_connection(self):
        client = motor.motor_asyncio.AsyncIOMotorClient(
            host=self.mongo_host,  # MongoDB 호스트 주소 설정
            port=self.mongo_port  # MongoDB 포트 번호 설정
        )
        db = client[self.mongo_database]  # 지정된 데이터베이스와 연결
        try:
            yield db  # 데이터베이스 연결 객체를 반환하여 'with' 블록 안에서 사용 가능
        finally:
            client.close()  # 'with' 블록이 끝날 때 MongoDB 클라이언트 연결 닫기

    async def insert_new_data(self, data: dict):
        youtube_url = data.get("h_youtube_url")  # YouTube URL
        video_path = data.get("video_path")  # 비디오 파일 경로
        audio_path = data.get("audio_path")  # 오디오 파일 경로
        title = data.get("title")  # 음악 제목
        artist = data.get("artist")  # 아티스트 이름
        keypoints_list = data.get("keypoints")  # 키포인트 데이터를 가져옴
        boxsizes_list = data.get("boxsizes")

        # MongoDB 컬렉션에 데이터 삽입
        async with self.get_db_connection() as db:
            collection = db["keypoints"]  # 컬렉션 이름을 명시
            result = await collection.insert_one({
                "youtube_url": youtube_url,
                "video_path": video_path,
                "audio_path": audio_path,
                "title": title,
                "artist": artist,
                "boxsizes" : boxsizes_list,
                "keypoints": keypoints_list  # 키포인트 데이터 삽입
            })

        return {"message": "데이터가 성공적으로 삽입되었습니다", "inserted_id": str(result.inserted_id)}

    async def find_data_mongodb(self, query):
        async with self.get_db_connection() as db:
            collection = db["keypoints"]
            data = await collection.find_one(query)
            return data

# MongoDB 서비스 객체 생성
mongo = MongoService(host="mongodb-service", port=27017, database="key_points_db")

# MongoDB에 새로운 데이터(영상 정보, 파일 경로, 원본 키포인트 값 등) 넣기 서비스 경로 핸들러(gateway.py와 대응)
@app.post("/insert_new_data_mongodb")
async def insert_new_data(data: dict):
    await mongo.insert_new_data(data)  # MongoDB에 음원 정보, 키포인트 등 삽입
    return {"message": "데이터가 성공적으로 삽입되었습니다"}


from fastapi.responses import JSONResponse
from bson import json_util
import json
# MongoDB에 원본 키포인트 값 조회(해쉬변환 url, title, artist 정보를 보내고, 셋 다 일치하는 Document 전체를 가져옴) 서비스 경로 핸들러(gateway.py와 대응)
@app.post("/find_data_mongodb")
async def find_data(query: dict):
    answer_data = await mongo.find_data_mongodb(query)
    json_data = json.dumps(answer_data, default=json_util.default)  # ObjectId 직렬화 처리
    return JSONResponse(content=json.loads(json_data))