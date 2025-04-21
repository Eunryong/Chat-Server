import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import jamo
from typing import Dict, List, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum

class PhonemeType(Enum):
    CONSONANT = "consonant"
    VOWEL = "vowel"
    FINAL = "final"

@dataclass
class PhonemeAnalysis:
    phoneme: str
    type: PhonemeType
    score: float
    feedback: str
    position: int

class PronunciationAnalyzer:
    def __init__(self):
        # 한국어 음성 인식을 위한 모델 로드
        self.model_name = "kresnik/wav2vec2-large-xlsr-korean"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        
        # GPU 사용 가능시 GPU로 이동
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 한국어 음소 매핑
        self.jamo_map = {
            'ㄱ': 'g', 'ㄲ': 'kk', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄸ': 'tt',
            'ㄹ': 'r', 'ㅁ': 'm', 'ㅂ': 'b', 'ㅃ': 'pp', 'ㅅ': 's',
            'ㅆ': 'ss', 'ㅇ': 'ng', 'ㅈ': 'j', 'ㅉ': 'jj', 'ㅊ': 'ch',
            'ㅋ': 'k', 'ㅌ': 't', 'ㅍ': 'p', 'ㅎ': 'h',
            'ㅏ': 'a', 'ㅐ': 'ae', 'ㅑ': 'ya', 'ㅒ': 'yae', 'ㅓ': 'eo',
            'ㅔ': 'e', 'ㅕ': 'yeo', 'ㅖ': 'ye', 'ㅗ': 'o', 'ㅘ': 'wa',
            'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅛ': 'yo', 'ㅜ': 'u', 'ㅝ': 'wo',
            'ㅞ': 'we', 'ㅟ': 'wi', 'ㅠ': 'yu', 'ㅡ': 'eu', 'ㅢ': 'ui',
            'ㅣ': 'i'
        }
        
        # 발음 평가 기준
        self.pronunciation_rules = {
            'ㄱ': {'strength': 0.7, 'duration': 0.1},
            'ㄲ': {'strength': 0.9, 'duration': 0.15},
            'ㅋ': {'strength': 0.8, 'duration': 0.12},
            # ... 다른 음소에 대한 규칙 추가
        }
    
    def analyze_pronunciation(self, audio_data: np.ndarray, reference_text: str) -> Dict[str, Any]:
        """
        오디오 데이터의 발음을 분석하여 정확도를 평가
        
        Args:
            audio_data: numpy 배열 형태의 오디오 데이터
            reference_text: 참조 텍스트 (정확한 발음)
            
        Returns:
            발음 분석 결과
        """
        try:
            # 음성 인식
            input_values = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # 음소 단위로 분리
            reference_jamo = self._text_to_jamo(reference_text)
            transcription_jamo = self._text_to_jamo(transcription)
            
            # 음소별 상세 분석
            phoneme_analysis = self._analyze_phonemes(audio_data, reference_jamo, transcription_jamo)
            
            # 발음 유사도 계산
            similarity_score = self._calculate_similarity(reference_jamo, transcription_jamo)
            
            # 음절 강세 분석
            stress_score = self._analyze_stress(audio_data)
            
            return {
                "overall_score": (similarity_score + stress_score) / 2,
                "similarity_score": similarity_score,
                "stress_score": stress_score,
                "transcription": transcription,
                "reference": reference_text,
                "phoneme_analysis": [{
                    "phoneme": p.phoneme,
                    "type": p.type.value,
                    "score": p.score,
                    "feedback": p.feedback,
                    "position": p.position
                } for p in phoneme_analysis]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "overall_score": 0.0
            }
    
    def _text_to_jamo(self, text: str) -> List[str]:
        """한글 텍스트를 음소(자모) 단위로 분리"""
        result = []
        for char in text:
            if '\uAC00' <= char <= '\uD7A3':  # 한글 범위
                # 초성, 중성, 종성 분리
                code = ord(char) - 0xAC00
                jong = code % 28
                jung = ((code - jong) // 28) % 21
                cho = ((code - jong) // 28) // 21
                
                # 자모 변환
                if cho > 0:
                    result.append(jamo.h2j(char)[0])
                if jung > 0:
                    result.append(jamo.h2j(char)[1])
                if jong > 0:
                    result.append(jamo.h2j(char)[2])
            else:
                result.append(char)
        return result
    
    def _calculate_similarity(self, ref: List[str], trans: List[str]) -> float:
        """두 음소 시퀀스의 유사도 계산"""
        # Levenshtein 거리 기반 유사도 계산
        m, n = len(ref), len(trans)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == trans[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        max_len = max(m, n)
        return 1 - (dp[m][n] / max_len)
    
    def _analyze_stress(self, audio_data: np.ndarray) -> float:
        """음절 강세 분석"""
        # 음성 에너지 계산
        energy = librosa.feature.rms(y=audio_data)[0]
        
        # 강세 패턴 분석
        peaks = librosa.util.peak_pick(energy, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        
        # 강세 점수 계산 (예시)
        if len(peaks) > 0:
            return min(1.0, len(peaks) / len(audio_data) * 100)
        return 0.5
    
    def _analyze_phonemes(self, audio_data: np.ndarray, 
                         reference: List[str], 
                         transcription: List[str]) -> List[PhonemeAnalysis]:
        """음소별 상세 분석 수행"""
        analysis_results = []
        
        # 음성 신호를 프레임 단위로 분할
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        
        for i, (ref_phoneme, trans_phoneme) in enumerate(zip(reference, transcription)):
            # 음소 타입 결정
            phoneme_type = self._get_phoneme_type(ref_phoneme)
            
            # 음소별 점수 계산
            score = self._calculate_phoneme_score(
                ref_phoneme, 
                trans_phoneme, 
                frames[:, i] if i < frames.shape[1] else None
            )
            
            # 피드백 생성
            feedback = self._generate_phoneme_feedback(ref_phoneme, trans_phoneme, score)
            
            analysis_results.append(
                PhonemeAnalysis(
                    phoneme=ref_phoneme,
                    type=phoneme_type,
                    score=score,
                    feedback=feedback,
                    position=i
                )
            )
        
        return analysis_results
    
    def _get_phoneme_type(self, phoneme: str) -> PhonemeType:
        """음소의 타입을 결정"""
        if phoneme in ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']:
            return PhonemeType.CONSONANT
        elif phoneme in ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']:
            return PhonemeType.VOWEL
        else:
            return PhonemeType.FINAL
    
    def _calculate_phoneme_score(self, ref_phoneme: str, trans_phoneme: str, 
                               frame: np.ndarray = None) -> float:
        """음소별 점수 계산"""
        # 기본 유사도 점수
        base_score = 1.0 if ref_phoneme == trans_phoneme else 0.0
        
        if frame is not None and ref_phoneme in self.pronunciation_rules:
            # 음성 특성 분석
            rules = self.pronunciation_rules[ref_phoneme]
            
            # 에너지 분석
            energy = np.mean(np.abs(frame))
            energy_score = min(1.0, energy / rules['strength'])
            
            # 지속 시간 분석
            duration = len(frame) / 16000  # 초 단위
            duration_score = min(1.0, duration / rules['duration'])
            
            # 종합 점수 계산
            return (base_score + energy_score + duration_score) / 3
        
        return base_score
    
    def _generate_phoneme_feedback(self, ref_phoneme: str, trans_phoneme: str, 
                                 score: float) -> str:
        """음소별 피드백 생성"""
        if score > 0.8:
            return "발음이 정확합니다."
        elif score > 0.5:
            return f"{ref_phoneme}의 발음을 더 명확하게 발음해보세요."
        else:
            return f"{ref_phoneme}의 발음이 부정확합니다. 정확한 발음 방법을 연습해보세요."

# 싱글톤 인스턴스 생성
pronunciation_analyzer = PronunciationAnalyzer() 