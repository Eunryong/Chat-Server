import socketio
import numpy as np
import json
from app.services.pronunciation_analyzer import pronunciation_analyzer
import logging
import asyncio
from collections import deque
import base64
import wave
import io
from datetime import datetime
import librosa
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버 주소
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

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

# Socket.IO 서버 생성
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=["http://localhost:3000"],  # React 개발 서버 주소
    cors_credentials=False,
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

# WebRTC 설정
ICE_SERVERS = [
    {'urls': 'stun:stun.l.google.com:19302'},
    {'urls': 'stun:stun1.l.google.com:19302'},
    {'urls': 'stun:stun2.l.google.com:19302'},
    {'urls': 'stun:stun3.l.google.com:19302'},
    {'urls': 'stun:stun4.l.google.com:19302'},
]

# 연결 관리
class ConnectionManager:
    def __init__(self):
        self.reference_texts = {}
        self.audio_buffers = {}
        self.buffer_size = 5
        self.avatar_states = {}
        self.peer_connections = {}
        self.speech_buffers = {}
        self.audio_streams = {}  # 오디오 스트림 관리
        self.stream_buffers = {}  # 스트림 버퍼 관리
        self.stream_processors = {}  # 스트림 처리 태스크 관리
        self.sample_rate = 16000  # STT 모델 샘플레이트
        self.buffer_duration = 0.5  # 버퍼 지속 시간 (초)
        self.stt_files = {}  # STT 파일 처리 상태 관리
        self.processing_tasks = {}  # 처리 중인 태스크 관리
        self.webrtc_sample_rate = 48000  # WebRTC 샘플레이트
        self.max_buffer_size = 48000 * 3  # 최대 버퍼 크기 (3초)
        self.min_buffer_duration = 0.1  # 최소 100ms
        self.optimal_buffer_duration = 0.5  # 최적 500ms
        self.max_buffer_duration = 1.0  # 최대 1초
        self.ice_servers = ICE_SERVERS

    def set_reference_text(self, sid, text):
        self.reference_texts[sid] = text
        logger.info(f"참조 텍스트 설정: {text}")

    def add_audio_stream(self, sid, stream_id):
        """새로운 오디오 스트림 추가"""
        if sid not in self.audio_streams:
            self.audio_streams[sid] = {}
            self.stream_buffers[sid] = {}
            self.stream_processors[sid] = {}
            self.processing_tasks[sid] = {}
        
        self.audio_streams[sid][stream_id] = {
            'active': True,
            'start_time': datetime.now(),
            'total_samples': 0
        }
        
        # 스트림 버퍼 초기화 (WebRTC 샘플레이트 기준)
        buffer_size = int(self.webrtc_sample_rate * self.buffer_duration)
        self.stream_buffers[sid][stream_id] = deque(maxlen=buffer_size)
        
        logger.info(f"[오디오 스트림 추가] sid: {sid}, stream_id: {stream_id}, 버퍼 크기: {buffer_size}")

    def remove_audio_stream(self, sid, stream_id):
        """오디오 스트림 제거"""
        if sid in self.audio_streams and stream_id in self.audio_streams[sid]:
            del self.audio_streams[sid][stream_id]
            if sid in self.stream_buffers and stream_id in self.stream_buffers[sid]:
                del self.stream_buffers[sid][stream_id]
            if sid in self.stream_processors and stream_id in self.stream_processors[sid]:
                self.stream_processors[sid][stream_id].cancel()
                del self.stream_processors[sid][stream_id]
            if sid in self.processing_tasks and stream_id in self.processing_tasks[sid]:
                self.processing_tasks[sid][stream_id].cancel()
                del self.processing_tasks[sid][stream_id]
            logger.info(f"[오디오 스트림 제거] sid: {sid}, stream_id: {stream_id}")

    def add_audio_chunk(self, sid, stream_id, audio_data):
        """오디오 청크 추가 및 처리"""
        try:
            if sid not in self.stream_buffers or stream_id not in self.stream_buffers[sid]:
                self.add_audio_stream(sid, stream_id)

            # 1. 바이너리 데이터를 int16 numpy 배열로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 0 데이터 체크
            if np.all(audio_array == 0):
                logger.warning("[오디오 데이터] 모든 샘플이 0입니다.")
                return {"ready": False, "error": "오디오 데이터가 없습니다."}

            # 2. Stereo인 경우 Mono로 다운믹싱
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1).astype(np.int16)

            # 3. float32로 정규화
            max_abs = np.max(np.abs(audio_array))
            if max_abs > 0:
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                logger.warning("[오디오 정규화] 최대 절대값이 0입니다.")
                return {"ready": False, "error": "오디오 데이터가 없습니다."}

            # 4. 오디오 데이터를 버퍼에 추가
            buffer = self.stream_buffers[sid][stream_id]
            buffer.extend(audio_array)

            # 5. 버퍼가 최대 크기를 초과하면 오래된 데이터 제거
            if len(buffer) > self.max_buffer_size:
                excess = len(buffer) - self.max_buffer_size
                for _ in range(excess):
                    buffer.popleft()

            # 6. 버퍼가 충분히 쌓였는지 확인 (최소 0.1초)
            min_required_samples = int(self.webrtc_sample_rate * 0.1)  # 100ms
            optimal_required_samples = int(self.webrtc_sample_rate * 0.5)  # 500ms

            if len(buffer) >= min_required_samples:
                # 7. 처리할 데이터 크기 결정
                process_size = min(len(buffer), optimal_required_samples)
                process_data = np.array(list(buffer)[:process_size], dtype=np.float32)
                
                # 8. 처리된 데이터만큼 버퍼에서 제거
                for _ in range(process_size):
                    buffer.popleft()

                # 9. 48kHz → 16kHz 리샘플링
                resampled_data = librosa.resample(process_data, orig_sr=self.webrtc_sample_rate, target_sr=self.sample_rate)
                logger.info(f"[STT 처리] 데이터 크기: {len(resampled_data)}")

                # 10. 스트림 통계 업데이트
                self.audio_streams[sid][stream_id]['total_samples'] += len(resampled_data)

                # 11. 추론 준비 완료 상태로 반환
                return {"ready": True, "audio": resampled_data}

            return {"ready": False}

        except Exception as e:
            logger.error(f"[오디오 청크 처리 오류] {str(e)}", exc_info=True)
            return {"ready": False, "error": str(e)}
            
    def update_avatar_state(self, sid, state):
        self.avatar_states[sid] = state
        logger.info(f"아바타 상태 업데이트: {state}")

    def add_peer_connection(self, sid, peer_id):
        if sid not in self.peer_connections:
            self.peer_connections[sid] = set()
        self.peer_connections[sid].add(peer_id)
        logger.info(f"피어 연결 추가: {sid} -> {peer_id}")
        
        # ICE 서버 정보 전송
        if socketRef.current:
            socketRef.current.emit('ice_servers', {
                'servers': self.ice_servers
            }, room=sid)

    def remove_peer_connection(self, sid, peer_id):
        if sid in self.peer_connections:
            self.peer_connections[sid].discard(peer_id)
            logger.info(f"피어 연결 제거: {sid} -> {peer_id}")

    async def process_audio_stream(self, sid, stream_id, audio_data):
        """오디오 스트림 처리"""
        try:
            # 오디오 데이터 처리
            result = self.add_audio_chunk(sid, stream_id, audio_data)
            
            if result["ready"]:
                # 참조 텍스트 확인
                reference_text = self.reference_texts.get(sid)
                if reference_text is None:
                    logger.warning(f"[참조 텍스트 없음] sid: {sid}")
                    return
                
                logger.info(f"[STT 처리 시작] 데이터 크기: {len(result['audio'])}")
                
                # STT 처리
                analysis_result = pronunciation_analyzer.analyze_pronunciation(
                    result['audio'],
                    reference_text
                )
                
                logger.info(f"[STT 결과] 전사: {analysis_result.get('transcription', '')}")
                logger.info(f"[STT 결과] 유사도 점수: {analysis_result.get('similarity_score', 0.0)}")
                
                # 결과 전송
                await sio.emit('stream_analysis_result', {
                    'stream_id': stream_id,
                    'result': analysis_result,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
                # STT 결과만 별도로 전송
                if 'transcription' in analysis_result and analysis_result['transcription']:
                    await sio.emit('stt-result', {
                        'text': analysis_result['transcription']
                    }, room=sid)
                    
        except Exception as e:
            logger.error(f"[오디오 스트림 처리 오류] {str(e)}", exc_info=True)
            await sio.emit('stream_error', {
                'stream_id': stream_id,
                'message': str(e)
            }, room=sid)

    async def process_stt_file(self, sid, file_id, audio_data):
        try:
            logger.info(f"[STT 파일 처리 시작] sid: {sid}, file_id: {file_id}")
            start_time = datetime.now()
            
            # Base64 디코딩
            logger.info(f"[Base64 디코딩] 파일 크기: {len(audio_data)} bytes")
            audio_bytes = base64.b64decode(audio_data)
            
            # WAV 파일 파싱
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                # 오디오 파라미터 확인
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logger.info(f"[WAV 파일 정보] 채널: {n_channels}, 샘플 크기: {sample_width}, "
                          f"프레임 레이트: {frame_rate}, 프레임 수: {n_frames}")
                
                # 오디오 데이터 읽기
                audio_frames = wav_file.readframes(n_frames)
                
                # numpy 배열로 변환
                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    raise ValueError(f"지원하지 않는 샘플 크기: {sample_width}")
                
                audio_array = np.frombuffer(audio_frames, dtype=dtype)
                logger.info(f"[오디오 데이터 변환] 배열 크기: {audio_array.shape}, 데이터 타입: {dtype}")
                logger.debug(f"[STT 디버그] 추출된 오디오 배열 샘플 일부: {audio_array[:10]}")
                
                # 모노로 변환 (필요한 경우)
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
                    logger.info(f"[모노 변환] 변환 후 배열 크기: {audio_array.shape}")
                
                # float32로 정규화
                audio_array = audio_array.astype(np.float32) / np.iinfo(dtype).max
                logger.info(f"[정규화] 최소값: {audio_array.min()}, 최대값: {audio_array.max()}")
                
                # STT 모델 샘플레이트(16kHz)로 리샘플링
                if frame_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=frame_rate, target_sr=16000)
                    logger.info(f"[리샘플링] {frame_rate}Hz → 16000Hz 변환 완료")
                
                # STT 처리
                reference_text = self.reference_texts.get(sid)
                if reference_text is None:
                    logger.error(f"[참조 텍스트 오류] sid: {sid}에 대한 참조 텍스트가 없습니다")
                    await sio.emit('stt_file_error', {
                        'file_id': file_id,
                        'message': '참조 텍스트가 설정되지 않았습니다'
                    }, room=sid)
                    return
                
                logger.info(f"[STT 처리 시작] 참조 텍스트: {reference_text}")
                # 발음 분석 수행
                analysis_result = pronunciation_analyzer.analyze_pronunciation(
                    audio_array,
                    reference_text
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"[STT 처리 완료] 처리 시간: {processing_time:.2f}초")
                logger.info(f"[STT 결과] 전사: {analysis_result.get('transcription', '')}")
                logger.info(f"[발음 분석] 유사도 점수: {analysis_result.get('similarity_score', 0.0)}")
                
                # 결과 전송
                await sio.emit('stt_file_result', {
                    'file_id': file_id,
                    'status': 'completed',
                    'result': {
                        'transcription': analysis_result.get('transcription', ''),
                        'similarity_score': analysis_result.get('similarity_score', 0.0),
                        'overall_score': analysis_result.get('overall_score', 0.0),
                        'phoneme_analysis': analysis_result.get('phoneme_analysis', []),
                        'processing_time': processing_time
                    }
                }, room=sid)
                
        except Exception as e:
            logger.error(f"[STT 파일 처리 오류] {str(e)}", exc_info=True)
            await sio.emit('stt_file_error', {
                'file_id': file_id,
                'message': str(e)
            }, room=sid)

manager = ConnectionManager()

# Socket.IO 이벤트 핸들러
@sio.event
async def connect(sid, environ):
    """클라이언트 연결"""
    try:
        logger.info(f"새로운 클라이언트 연결: {sid}")
        logger.info(f"연결 환경: {environ}")
        
        # 연결 응답 전송
        await sio.emit('connect_response', {
            'status': 'connected',
            'message': '연결이 성공적으로 설정되었습니다',
            'sid': sid
        }, room=sid)
        
        # 연결 상태 로깅
        logger.info(f"클라이언트 {sid} 연결 완료")
        
    except Exception as e:
        logger.error(f"[연결 설정 오류] {str(e)}", exc_info=True)
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def disconnect(sid):
    """클라이언트 연결 종료"""
    try:
        logger.info(f"클라이언트 연결 종료: {sid}")
        
        # 모든 스트림 정리
        if sid in manager.audio_streams:
            for stream_id in list(manager.audio_streams[sid].keys()):
                manager.remove_audio_stream(sid, stream_id)
            del manager.audio_streams[sid]
        
        # 모든 태스크 정리
        if sid in manager.processing_tasks:
            for task in manager.processing_tasks[sid].values():
                task.cancel()
            del manager.processing_tasks[sid]
            
        logger.info(f"클라이언트 {sid} 연결 종료 처리 완료")
            
    except Exception as e:
        logger.error(f"[연결 종료 처리 오류] {str(e)}", exc_info=True)

@sio.event
async def reference(sid, data):
    """참조 텍스트 설정"""
    try:
        text = data.get('text')
        if text:
            manager.set_reference_text(sid, text)
            await sio.emit('reference_response', {
                'status': 'success',
                'message': '참조 텍스트 설정 완료',
                'text': text
            }, room=sid)
        else:
            await sio.emit('reference_response', {
                'status': 'error',
                'message': '참조 텍스트가 제공되지 않았습니다'
            }, room=sid)
    except Exception as e:
        logger.error(f"참조 텍스트 설정 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def audio_stream_start(sid, data):
    """오디오 스트림 시작"""
    try:
        stream_id = data.get('stream_id')
        logger.info(f"[오디오 스트림 시작] sid: {sid}, stream_id: {stream_id}")
        
        manager.add_audio_stream(sid, stream_id)
        await sio.emit('stream_status', {
            'stream_id': stream_id,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"[오디오 스트림 시작 오류] {str(e)}", exc_info=True)
        await sio.emit('stream_error', {
            'stream_id': stream_id,
            'message': str(e)
        }, room=sid)

@sio.event
async def audio_stream_data(sid, data):
    """오디오 스트림 데이터 수신"""
    try:
        stream_id = data.get('stream_id')
        logger.info(f"[오디오 스트림 데이터 수신] sid: {sid}, stream_id: {stream_id}")
        
        # 참조 텍스트 확인
        if sid not in manager.reference_texts:
            logger.warning(f"[참조 텍스트 없음] sid: {sid}")
            await sio.emit('stream_error', {
                'stream_id': stream_id,
                'message': '참조 텍스트가 설정되지 않았습니다'
            }, room=sid)
            return
            
        # 바이너리 데이터 처리
        binary_data = None
        if isinstance(data.get('data'), dict) and data['data'].get('_placeholder'):
            # Socket.IO 바이너리 데이터 처리
            binary_data = await sio.get_binary_data()
            logger.info(f"[바이너리 데이터 수신] 크기: {len(binary_data) if binary_data else 0}")
        else:
            binary_data = data.get('data')
            
        if not binary_data:
            logger.error("[오디오 데이터 누락] 바이너리 데이터를 받지 못했습니다")
            return
            
        if not isinstance(binary_data, bytes):
            logger.error(f"[오디오 데이터 형식 오류] 바이너리 데이터가 아닙니다: {type(binary_data)}")
            return
            
        # 스트림 처리
        await manager.process_audio_stream(sid, stream_id, binary_data)
            
    except Exception as e:
        logger.error(f"[오디오 스트림 데이터 처리 오류] {str(e)}", exc_info=True)
        await sio.emit('stream_error', {
            'stream_id': stream_id,
            'message': str(e)
        }, room=sid)

@sio.event
async def audio_stream_end(sid, data):
    """오디오 스트림 종료"""
    try:
        stream_id = data.get('stream_id')
        logger.info(f"[오디오 스트림 종료] sid: {sid}, stream_id: {stream_id}")
        
        manager.remove_audio_stream(sid, stream_id)
        await sio.emit('stream_status', {
            'stream_id': stream_id,
            'status': 'ended',
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"[오디오 스트림 종료 오류] {str(e)}", exc_info=True)
        await sio.emit('stream_error', {
            'stream_id': stream_id,
            'message': str(e)
        }, room=sid)

@sio.event
async def avatarState(sid, data):
    try:
        manager.update_avatar_state(sid, data)
        await sio.emit('avatarStateUpdate', data, skip_sid=sid)
    except Exception as e:
        logger.error(f"아바타 상태 업데이트 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

# WebRTC 시그널링 이벤트
@sio.event
async def offer(sid, data):
    try:
        logger.info(f"Offer 수신 from {sid}: {data}")
        target_sid = data.get('target')
        if target_sid:
            await sio.emit('offer', {
                'from': sid,
                'offer': data.get('offer'),
                'ice_servers': ICE_SERVERS
            }, room=target_sid)
            logger.info(f"Offer 전달 완료 from {sid} to {target_sid}")
        else:
            await sio.emit('offer', {
                'from': sid,
                'offer': data.get('offer'),
                'ice_servers': ICE_SERVERS
            }, skip_sid=sid)
            logger.info(f"Offer 전달 완료 from {sid}")
    except Exception as e:
        logger.error(f"Offer 전달 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def answer(sid, data):
    try:
        logger.info(f"Answer 수신 from {sid}: {data}")
        target_sid = data.get('target')
        if target_sid:
            await sio.emit('answer', {
                'from': sid,
                'answer': data.get('answer')
            }, room=target_sid)
            logger.info(f"Answer 전달 완료 from {sid} to {target_sid}")
        else:
            await sio.emit('answer', {
                'from': sid,
                'answer': data.get('answer')
            }, skip_sid=sid)
            logger.info(f"Answer 전달 완료 from {sid}")
    except Exception as e:
        logger.error(f"Answer 전달 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def ice_candidate(sid, data):
    try:
        logger.info(f"ICE 후보 수신 from {sid}: {data}")
        target_sid = data.get('target')
        if target_sid:
            await sio.emit('ice-candidate', {
                'from': sid,
                'candidate': data.get('candidate')
            }, room=target_sid)
            logger.info(f"ICE 후보 전달 완료 from {sid} to {target_sid}")
        else:
            await sio.emit('ice-candidate', {
                'from': sid,
                'candidate': data.get('candidate')
            }, skip_sid=sid)
            logger.info(f"ICE 후보 전달 완료 from {sid}")
    except Exception as e:
        logger.error(f"ICE 후보 전달 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

# WebRTC 상태 이벤트
@sio.event
async def webrtc_state(sid, data):
    try:
        state = data.get('state')
        peer_id = data.get('peer_id')
        logger.info(f"WebRTC 상태 업데이트 from {sid}: {state} (peer: {peer_id})")
        
        if state == 'connected':
            manager.add_peer_connection(sid, peer_id)
        elif state == 'disconnected':
            manager.remove_peer_connection(sid, peer_id)
            
        await sio.emit('webrtc_state_update', {
            'from': sid,
            'state': state,
            'peer_id': peer_id
        }, skip_sid=sid)
    except Exception as e:
        logger.error(f"WebRTC 상태 업데이트 중 오류: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def stt_file_upload(sid, data):
    try:
        file_id = data.get('file_id')
        audio_data = data.get('audio_data')
        
        logger.info(f"[파일 업로드 시작] sid: {sid}, file_id: {file_id}")
        
        if not file_id or not audio_data:
            logger.error(f"[파일 업로드 오류] 필수 데이터 누락 - file_id: {file_id}, audio_data: {bool(audio_data)}")
            await sio.emit('stt_file_error', {
                'file_id': file_id,
                'message': '필수 데이터가 누락되었습니다'
            }, room=sid)
            return
        
        # 파일 처리 상태 업데이트
        if sid not in manager.stt_files:
            manager.stt_files[sid] = {}
        
        manager.stt_files[sid][file_id] = {
            'status': 'processing',
            'progress': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"[파일 처리 상태] sid: {sid}, file_id: {file_id} - 처리 시작")
        
        # 처리 시작 알림
        await sio.emit('stt_file_status', {
            'file_id': file_id,
            'status': 'processing',
            'progress': 0
        }, room=sid)
        
        # 비동기 처리 시작
        task = asyncio.create_task(
            manager.process_stt_file(sid, file_id, audio_data)
        )
        
        # 태스크 저장
        if sid not in manager.processing_tasks:
            manager.processing_tasks[sid] = {}
        manager.processing_tasks[sid][file_id] = task
        
    except Exception as e:
        logger.error(f"[파일 업로드 처리 오류] {str(e)}", exc_info=True)
        await sio.emit('stt_file_error', {
            'file_id': file_id,
            'message': str(e)
        }, room=sid)

@sio.event
async def stt_file_cancel(sid, data):
    try:
        file_id = data.get('file_id')
        logger.info(f"[파일 처리 취소 요청] sid: {sid}, file_id: {file_id}")
        
        if sid in manager.processing_tasks and file_id in manager.processing_tasks[sid]:
            # 처리 중인 태스크 취소
            manager.processing_tasks[sid][file_id].cancel()
            del manager.processing_tasks[sid][file_id]
            
            # 파일 상태 업데이트
            if sid in manager.stt_files and file_id in manager.stt_files[sid]:
                manager.stt_files[sid][file_id]['status'] = 'cancelled'
                logger.info(f"[파일 처리 취소 완료] sid: {sid}, file_id: {file_id}")
                await sio.emit('stt_file_status', {
                    'file_id': file_id,
                    'status': 'cancelled'
                }, room=sid)
                
    except Exception as e:
        logger.error(f"[파일 취소 처리 오류] {str(e)}", exc_info=True)
        await sio.emit('error', {'message': str(e)}, room=sid)

# Socket.IO 애플리케이션 생성
socket_app = socketio.ASGIApp(
    sio,
    socketio_path='socket.io'
)

# FastAPI에 Socket.IO 애플리케이션 마운트
app.mount("/", socket_app) 