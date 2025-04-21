from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import io
import soundfile as sf
import asyncio
from typing import List
import json
from app.services.speech_analysis import speech_analyzer

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 오디오 데이터 수신
            data = await websocket.receive_bytes()
            
            # 바이트 데이터를 numpy 배열로 변환
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # 음성 분석 수행
            analysis_result = await speech_analyzer.analyze_speech(audio_data)
            
            # 분석 결과를 JSON 문자열로 변환하여 전송
            await manager.send_personal_message(json.dumps(analysis_result), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client disconnected")
    except Exception as e:
        error_message = json.dumps({
            "status": "error",
            "message": str(e)
        })
        await manager.send_personal_message(error_message, websocket) 