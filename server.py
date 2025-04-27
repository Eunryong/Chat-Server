import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stt_server.log')
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Socket.IO 서버 생성
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=['*'],
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    allow_upgrades=True,
    transports=['websocket', 'polling'],
    max_http_buffer_size=1e8,
    async_handlers=True,
    always_connect=True,
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1000,
    reconnection_delay_max=5000
)

# Socket.IO 애플리케이션 생성
socket_app = socketio.ASGIApp(
    sio,
    socketio_path='socket.io'
)

# FastAPI에 Socket.IO 애플리케이션 마운트
app.mount("/", socket_app)

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

@sio.on('audio_stream_data')
async def handle_audio_stream(sid, data):
    try:
        stream_id = data.get('stream_id')
        if not stream_id:
            logger.error(f"[스트림 ID 없음] sid: {sid}")
            await sio.emit('stream_error', {'stream_id': stream_id, 'message': '스트림 ID가 없습니다.'}, room=sid)
            return

        if sid not in audio_streams:
            logger.error(f"[스트림 없음] sid: {sid}, stream_id: {stream_id}")
            await sio.emit('stream_error', {'stream_id': stream_id, 'message': '스트림이 초기화되지 않았습니다.'}, room=sid)
            return

        if not audio_streams[sid].get('reference_text'):
            logger.warning(f"[참조 텍스트 없음] sid: {sid}")
            await sio.emit('stream_error', {'stream_id': stream_id, 'message': '참조 텍스트가 설정되지 않았습니다.'}, room=sid)
            return

        # 바이너리 데이터 처리
        audio_data = data.get('data')
        if isinstance(audio_data, dict) and '_placeholder' in audio_data:
            # Socket.IO 바이너리 데이터 처리
            binary_data = await sio.get_binary_data(audio_data['num'])
            if binary_data:
                audio_data = binary_data
            else:
                logger.error(f"[바이너리 데이터 없음] sid: {sid}")
                return

        # 오디오 데이터 처리
        audio_streams[sid]['buffer'].extend(audio_data)
        
        # 버퍼가 충분히 쌓였을 때 처리
        if len(audio_streams[sid]['buffer']) >= CHUNK_SIZE:
            audio_chunk = audio_streams[sid]['buffer'][:CHUNK_SIZE]
            audio_streams[sid]['buffer'] = audio_streams[sid]['buffer'][CHUNK_SIZE:]
            
            # 오디오 데이터 처리 및 STT 수행
            try:
                # 바이너리 데이터를 numpy 배열로 변환
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # 오디오 데이터 정규화
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                # STT 모델에 입력
                input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
                
                # 추론 수행
                with torch.no_grad():
                    logits = model(input_values).logits
                
                # CTC 디코딩
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
                
                # 결과 전송
                await sio.emit('transcription', {
                    'stream_id': stream_id,
                    'text': transcription,
                    'reference_text': audio_streams[sid]['reference_text']
                }, room=sid)
                
            except Exception as e:
                logger.error(f"[STT 처리 오류] sid: {sid}, error: {str(e)}")
                await sio.emit('stream_error', {'stream_id': stream_id, 'message': f'STT 처리 중 오류가 발생했습니다: {str(e)}'}, room=sid)
                
    except Exception as e:
        logger.error(f"[오디오 스트림 처리 오류] sid: {sid}, error: {str(e)}")
        await sio.emit('stream_error', {'stream_id': stream_id, 'message': f'오디오 스트림 처리 중 오류가 발생했습니다: {str(e)}'}, room=sid) 