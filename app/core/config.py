from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI 아바타 스피치 학습 코치"
    
    # MongoDB 설정
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "speech_coach"
    
    # Redis 설정
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # JWT 설정
    SECRET_KEY: str = "your-secret-key-here"  # 실제 운영 환경에서는 환경 변수로 설정
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 