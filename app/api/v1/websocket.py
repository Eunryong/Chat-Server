import socketio
import numpy as np
import json
from app.services.pronunciation_analyzer import pronunciation_analyzer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Socket.IO 서버 생성
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=['http://localhost:3000'],
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)

# 연결 관리
class ConnectionManager:
    def __init__(self):
        self.reference_texts = {}
        self.audio_buffers = {}
        self.buffer_size = 5

    def set_reference_text(self, sid, text):
        self.reference_texts[sid] = text
        logger.info(f"참조 텍스트 설정: {text}")

    def add_audio_chunk(self, sid, audio_data):
        if sid not in self.audio_buffers:
            self.audio_buffers[sid] = []
        self.audio_buffers[sid].append(audio_data)
        if len(self.audio_buffers[sid]) > self.buffer_size:
            self.audio_buffers[sid].pop(0)

    def get_audio_buffer(self, sid):
        if sid in self.audio_buffers and self.audio_buffers[sid]:
            return np.concatenate(self.audio_buffers[sid])
        return np.array([])

manager = ConnectionManager()

# Socket.IO 이벤트 핸들러
@sio.event
async def connect(sid, environ):
    logger.info(f"새로운 클라이언트 연결: {sid}")
    await sio.emit('connect_response', {'status': 'connected'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"클라이언트 연결 종료: {sid}")
    if sid in manager.reference_texts:
        del manager.reference_texts[sid]
    if sid in manager.audio_buffers:
        del manager.audio_buffers[sid]

@sio.event
async def reference(sid, data):
    try:
        text = data.get('text')
        if text:
            manager.set_reference_text(sid, text)
            await sio.emit('reference_response', {
                'status': 'success',
                'message': '참조 텍스트 설정 완료',
                'text': text
            }, room=sid)
    except Exception as e:
        logger.error(f"참조 텍스트 설정 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def audio(sid, data):
    try:
        audio_data = np.array(data.get('data'), dtype=np.float32)
        manager.add_audio_chunk(sid, audio_data)
        
        if len(manager.audio_buffers.get(sid, [])) >= manager.buffer_size:
            reference_text = manager.reference_texts.get(sid)
            if reference_text is None:
                await sio.emit('error', {
                    'message': '참조 텍스트가 설정되지 않았습니다'
                }, room=sid)
                return
            
            audio_buffer = manager.get_audio_buffer(sid)
            analysis_result = pronunciation_analyzer.analyze_pronunciation(
                audio_buffer,
                reference_text
            )
            
            await sio.emit('analysis_result', analysis_result, room=sid)
    except Exception as e:
        logger.error(f"오디오 처리 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

# Socket.IO 애플리케이션 생성
socket_app = socketio.ASGIApp(sio) 