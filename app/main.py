from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import audio, websocket
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI 아바타를 통한 스피치 학습 코칭 시스템",
    version="1.0.0"
)

# CORS 설정
origins = [
    "http://localhost:3000",  # React 개발 서버
    "http://127.0.0.1:3000",  # React 개발 서버 (대체 주소)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# API 라우터 등록
app.include_router(audio.router, prefix=settings.API_V1_STR + "/audio", tags=["audio"])

# Socket.IO 애플리케이션 등록
app.mount("/socket.io", websocket.socket_app)

@app.get("/")
async def root():
    return {"message": "AI 아바타 스피치 학습 코치 API 서버"} 