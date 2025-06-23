"""
Configuration settings for the RAG application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get project root directory
project_root = Path(__file__).parent.parent

class Settings:
    """Application settings and configuration"""
    
    # Directory paths
    PDF_DIRECTORY = str(project_root / "data" / "raw")
    PROCESSED_DOCUMENTS_DIR = str(project_root / "data" / "processed")
    CHUNKS_DIR = str(project_root / "data" / "chunks")
    KEYWORDS_DIR = str(project_root / "data" / "keywords")
    EMBEDDINGS_DIR = str(project_root / "data" / "embeddings")
    THEMES_DIR = str(project_root / "data" / "themes")
    
    # Debug settings
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # OCR Configuration
    USE_OCR = os.getenv("USE_OCR", "True").lower() == "true"
    OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract")
    OCR_DPI = int(os.getenv("OCR_DPI", "300"))
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
    ENHANCE_IMAGE = os.getenv("ENHANCE_IMAGE", "True").lower() == "true"
    
    # Text processing settings
    MIN_PARAGRAPH_LENGTH = int(os.getenv("MIN_PARAGRAPH_LENGTH", "50"))
    MAX_PARAGRAPH_LENGTH = int(os.getenv("MAX_PARAGRAPH_LENGTH", "2000"))
    
    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # Keyword extraction settings
    MAX_KEYWORDS = int(os.getenv("MAX_KEYWORDS", "20"))
    KEYWORD_MIN_SCORE = float(os.getenv("KEYWORD_MIN_SCORE", "0.3"))
    
    # Embedding settings  
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")  # Gemini's latest embedding model
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Theme identification settings
    NUM_TOPICS = int(os.getenv("NUM_TOPICS", "10"))
    THEME_ALGORITHM = os.getenv("THEME_ALGORITHM", "lda")
    
    def validate_config(self):
        """Validate configuration settings"""
        try:
            # Check if required directories can be created
            for directory in [
                self.PDF_DIRECTORY,
                self.PROCESSED_DOCUMENTS_DIR,
                self.CHUNKS_DIR,
                self.KEYWORDS_DIR,
                self.EMBEDDINGS_DIR,
                self.THEMES_DIR
            ]:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Validate numeric settings
            assert self.OCR_DPI > 0, "OCR_DPI must be positive"
            assert self.MIN_PARAGRAPH_LENGTH > 0, "MIN_PARAGRAPH_LENGTH must be positive"
            assert self.MAX_PARAGRAPH_LENGTH > self.MIN_PARAGRAPH_LENGTH, "MAX_PARAGRAPH_LENGTH must be greater than MIN_PARAGRAPH_LENGTH"
            assert self.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
            assert self.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
            assert self.MAX_RETRIES > 0, "MAX_RETRIES must be positive"
            assert self.MAX_KEYWORDS > 0, "MAX_KEYWORDS must be positive"
            assert 0 <= self.KEYWORD_MIN_SCORE <= 1, "KEYWORD_MIN_SCORE must be between 0 and 1"
            assert self.EMBEDDING_BATCH_SIZE > 0, "EMBEDDING_BATCH_SIZE must be positive"
            assert self.NUM_TOPICS > 0, "NUM_TOPICS must be positive"
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.PDF_DIRECTORY,
            self.PROCESSED_DOCUMENTS_DIR,
            self.CHUNKS_DIR,
            self.KEYWORDS_DIR,
            self.EMBEDDINGS_DIR,
            self.THEMES_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created/verified directory: {directory}")

# Create settings instance
settings = Settings()

# Validate configuration on import
if not settings.validate_config():
    print("⚠️  Configuration validation failed. Using default values.")