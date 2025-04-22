from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import audio
from app.api.v1.websocket import socket_app  # socketio.ASGIApp(sio)

app = FastAPI(
    title="AI 아바타 스피치 학습 코치 시스템",
    description="AI 아바타를 통한 스피치 학습 코칭 시스템",
    version="1.0.0"
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# API 라우터 등록
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])

# Socket.IO 애플리케이션 등록
app.mount("/ws", socket_app)

@app.get("/")
async def root():
    return {"message": "FastAPI 메인 서버 정상 작동 중"}
