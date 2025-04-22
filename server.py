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