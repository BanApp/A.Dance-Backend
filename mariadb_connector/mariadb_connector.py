import mariadb
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

app = FastAPI()

class MariaService:
    def __init__(self, host, port, user, password, database):
        self.maria_host = host  # MariaDB 호스트 주소 설정
        self.maria_port = port  # MariaDB 포트 번호 설정
        self.maria_user = user  # MariaDB 사용자 이름 설정
        self.maria_password = password  # MariaDB 비밀번호 설정
        self.maria_database = database  # MariaDB 데이터베이스 이름 설정

    @asynccontextmanager
    async def get_db_connection(self):
        conn = mariadb.connect(
            host=self.maria_host,
            port=self.maria_port,
            user=self.maria_user,
            password=self.maria_password,
            database=self.maria_database
        )
        cursor = conn.cursor(dictionary=True)
        try:
            yield conn, cursor  # 데이터베이스 커넥션과 커서 반환하여 'with' 블록 안에서 사용 가능
        finally:
            cursor.close()  # 커서 닫기
            conn.close()  # 데이터베이스 연결 닫기

    async def check_existing_data(self, url):
        async with self.get_db_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM music_info WHERE url=?", (url,)) # music_info에서 url 값이 같은 row 조회
            existing_data = cursor.fetchone() # 데이터베이스 쿼리 결과 가져오기
            return existing_data

    # 음원, 링크, 비디오 정보 등등..
    async def insert_new_data(self, url, video_path, audio_path, title, artist):
        async with self.get_db_connection() as (conn, cursor):
            cursor.execute(
                "INSERT INTO music_info (url, video_path, audio_path, title, artist) VALUES (?, ?, ?, ?, ?)",
                (url, video_path, audio_path, title, artist)
            )
            conn.commit()  # 데이터베이스에 변경사항을 커밋하여 새 데이터를 삽입합니다.

    # 사용자 점수 데이터
    async def insert_new_score_data(self, username, title, artist, score):
        async with self.get_db_connection() as (conn, cursor):
            cursor.execute(
                "INSERT INTO user_scores (username, title, artist, score) VALUES (?, ?, ?, ?)",
                (username, title, artist, score)
            )
            conn.commit()  # 데이터베이스에 변경사항을 커밋하여 새 데이터를 삽입합니다.

    async def get_hot_contents(self, num: int):
        async with self.get_db_connection() as (conn, cursor):
            query = (
                "SELECT title, artist "
                "FROM user_scores "
                "GROUP BY title, artist "
                "ORDER BY COUNT(*) DESC "
                f"LIMIT {num}"
            )
            cursor.execute(query)
            hot_contents = cursor.fetchall()
            return [{"title": entry["title"], "artist": entry["artist"]} for entry in hot_contents]

    async def get_leaderboard(self, title, num):
        async with self.get_db_connection() as (conn, cursor):
            query = (
                "SELECT username, score "
                "FROM user_scores "
                "WHERE title = %s "
                "ORDER BY score DESC "
                f"LIMIT {num}"  # num 개 만큼의 결과를 조회
            )
            cursor.execute(query, (title,))
            leaderboard_results = cursor.fetchall()

            leaderboard = [{"username": entry["username"], "score": entry["score"]} for entry in leaderboard_results]
            return leaderboard

# 해당 이름은 쿠버네티스 클러스터 내에서 MariaDB 서비스의 DNS 이름으로 사용됩니다.
mariadb_service_host = "mariadb-service"  # MariaDB 서비스의 DNS 이름
mariadb_service_port = 3306  # MariaDB 서비스의 포트 번호

# MariaService 클래스의 인스턴스를 생성할 때 쿠버네티스 서비스 정보를 전달합니다.
database = MariaService(
    host=mariadb_service_host,
    port=mariadb_service_port,
    user="",
    password="",
    database="music"
)

# MariaDB에 해쉬 변환된 유튜브 링크가 있는지 확인, 데이터 보유 유무 확인 서비스 경로 핸들러(gateway.py 와 대응)
@app.post("/check_existing_data_mariadb")
async def check_existing_data(request_data: dict):
    url = request_data.get("url")  # JSON에서 "url" 값을 추출
    if not url:
        raise HTTPException(status_code=400, detail="JSON에 'url' 필드가 필요합니다")
    existing_data = await database.check_existing_data(url)  # MariaDB로부터 데이터 조회
    return existing_data


# MariaDB에 새로운 데이터(영상 정보, 파일 경로 등) 넣기 서비스 경로 핸들러(gateway.py 와 대응)
@app.post("/insert_new_data_mariadb")
async def insert_new_data(data: dict):
    url = data.get("h_youtube_url")
    video_path = data.get("video_path")
    audio_path = data.get("audio_path")
    title = data.get("title")
    artist = data.get("artist")
    await database.insert_new_data(url, video_path, audio_path, title, artist)  # MariaDB에 데이터 삽입
    return {"message": "Data inserted successfully"}

# MariaDB에 새로운 점수 데이터 넣기 서비스 경로 핸들러(gateway.py 와 대응)
@app.post("/insert_new_score_data_mariadb")
async def insert_new_score_data_mariadb(data: dict):
    username = data.get("username")
    title = data.get("title")
    artist = data.get("artist")
    score = data.get("score")
    await database.insert_new_score_data(username, title, artist, score)  # MariaDB에 데이터 삽입
    return {"message": "Score Data inserted successfully"}


@app.get("/get_hot_contents/{num}")
async def get_hot_contents(num: int):
    hot_contents = await database.get_hot_contents(num)  # MariaDB에서 인기 컨텐츠 조회
    return {"hot_contents": hot_contents}


@app.post("/get_leaderboard")
async def get_leaderboard(data: dict):
    title = data.get("title")
    num = data.get("num")
    leaderboard = await database.get_leaderboard(title,num)  # MariaDB에서 곡별 사용자 성적 num개 조회
    return {"leaderboard": leaderboard}