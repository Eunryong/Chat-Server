from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Avatar Speech Coach"
    API_V1_STR: str = "/api/v1"
    
    # MongoDB 설정
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "speech_coach"
    
    # Redis 설정
    REDIS_URL: str = "redis://localhost:6379"
    
    # 음성 분석 모델 설정
    SPEECH_MODEL_NAME: str = "kresnik/wav2vec2-large-xlsr-korean"
    
    # JWT 설정
    SECRET_KEY: str = "your-secret-key"  # 실제 운영 환경에서는 환경 변수로 관리
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings() 