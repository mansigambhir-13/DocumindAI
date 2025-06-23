"""
Embeddings Generation Script
Generates vector embeddings for text chunks using sentence transformers
Aligns with RAG application pipeline structure
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import settings (with fallback if not available)
try:
    from config.settings import settings
    # Add embeddings settings if not present
    if not hasattr(settings, 'KEYWORDS_DIR'):
        settings.KEYWORDS_DIR = str(project_root / "data" / "keywords")
    if not hasattr(settings, 'EMBEDDINGS_DIR'):
        settings.EMBEDDINGS_DIR = str(project_root / "data" / "embeddings")
    if not hasattr(settings, 'EMBEDDING_MODEL'):
        settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    if not hasattr(settings, 'EMBEDDING_BATCH_SIZE'):
        settings.EMBEDDING_BATCH_SIZE = 32
except ImportError:
    # Fallback configuration if settings not available
    class Settings:
        KEYWORDS_DIR = str(project_root / "data" / "keywords")
        EMBEDDINGS_DIR = str(project_root / "data" / "embeddings")
        DEBUG = True
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        EMBEDDING_BATCH_SIZE = 32
        MAX_RETRIES = 3
        
        def validate_config(self):
            return True
        
        def create_directories(self):
            Path(self.EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    
    settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    def __init__(self):
        """Initialize embeddings generator"""
        # Fix paths to match project structure
        self.input_dir = Path(settings.KEYWORDS_DIR)
        if not self.input_dir.is_absolute():
            self.input_dir = project_root / self.input_dir
            
        self.output_dir = Path(settings.EMBEDDINGS_DIR)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.model_name = getattr(settings, 'EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2")
        
        # Check for Gemini preference
        self.use_gemini = bool(getattr(settings, 'GEMINI_API_KEY', ''))
        if self.use_gemini:
            self.model_name = "text-embedding-004"  # Latest Gemini embedding model
            print(f"🔄 Using Gemini embeddings: {self.model_name}")
        elif self.model_name.startswith('text-embedding'):
            # Fix OpenAI model names for sentence-transformers
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"⚠️  Detected OpenAI model name, switching to: {self.model_name}")
        
        self.batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 32)
        
        # Initialize model
        self.model = None
        self.gemini_client = None
        self.model_loaded = False
        
        logger.info(f"EmbeddingsGenerator initialized")
        logger.info(f"Input Directory (keywords): {self.input_dir}")
        logger.info(f"Output Directory (embeddings): {self.output_dir}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Batch size: {self.batch_size}")
    
    def load_model(self) -> bool:
        """Load the embedding model (Gemini or Sentence Transformers)"""
        try:
            if self.use_gemini:
                return self._load_gemini_model()
            else:
                return self._load_sentence_transformer_model()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"❌ Failed to load model: {e}")
            return False
    
    def _load_gemini_model(self) -> bool:
        """Load Gemini embedding model"""
        try:
            print(f"🔄 Loading Gemini embedding model: {self.model_name}")
            
            # Import and configure Gemini
            try:
                import google.generativeai as genai
            except ImportError:
                print("❌ google-generativeai not installed")
                print("💡 Install with: pip install google-generativeai")
                return False
            
            # Configure with API key
            gemini_api_key = getattr(settings, 'GEMINI_API_KEY', '')
            if not gemini_api_key:
                print("❌ GEMINI_API_KEY not found in settings")
                print("💡 Add GEMINI_API_KEY to your .env file")
                return False
            
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai
            self.model_loaded = True
            
            print(f"✅ Gemini model loaded successfully!")
            print(f"📏 Model: {self.model_name} (Latest Gemini embedding model)")
            print(f"🎯 Embedding dimension: 768")
            print(f"🚀 Optimized for document retrieval and semantic search")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {e}")
            print(f"❌ Failed to load Gemini model: {e}")
            print(f"💡 Make sure your GEMINI_API_KEY is valid")
            return False
    
    def _load_sentence_transformer_model(self) -> bool:
        """Load sentence transformer model"""
        try:
            print(f"🔄 Loading embedding model: {self.model_name}")
            
            # Try to import sentence transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("❌ sentence-transformers not installed")
                print("💡 Install with: pip install sentence-transformers")
                return False
            
            # Check if model name is valid for sentence-transformers
            valid_models = [
                "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions, fast
                "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions, better quality
                "sentence-transformers/all-MiniLM-L12-v2",  # 384 dimensions, good balance
                "all-MiniLM-L6-v2",  # Short version
                "all-mpnet-base-v2",  # Short version  
                "all-MiniLM-L12-v2"   # Short version
            ]
            
            # Add prefix if missing
            if not self.model_name.startswith('sentence-transformers/'):
                if any(self.model_name == model.split('/')[-1] for model in valid_models):
                    self.model_name = f"sentence-transformers/{self.model_name}"
            
            # Load model with error handling
            try:
                self.model = SentenceTransformer(self.model_name)
                self.model_loaded = True
                
                print(f"✅ Model loaded successfully!")
                print(f"📏 Model max sequence length: {self.model.max_seq_length}")
                return True
                
            except Exception as model_error:
                print(f"❌ Failed to load '{self.model_name}': {model_error}")
                print(f"🔄 Trying fallback model: sentence-transformers/all-MiniLM-L6-v2")
                
                # Try fallback model
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                self.model_loaded = True
                
                print(f"✅ Fallback model loaded successfully!")
                print(f"📏 Model max sequence length: {self.model.max_seq_length}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            print(f"❌ Failed to load sentence transformer model: {e}")
            print(f"💡 Available models to try:")
            print(f"   - sentence-transformers/all-MiniLM-L6-v2 (384d, fast)")
            print(f"   - sentence-transformers/all-mpnet-base-v2 (768d, better quality)")
            print(f"   - sentence-transformers/all-MiniLM-L12-v2 (384d, good balance)")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.model_loaded:
            logger.error("Model not loaded")
            return None
        
        try:
            # Clean and truncate text if needed
            text = text.strip()
            if not text:
                return None
            
            if self.use_gemini:
                # Use Gemini for embeddings
                response = self.gemini_client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = response['embedding']
            else:
                # Use sentence transformers
                embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts"""
        if not self.model_loaded:
            logger.error("Model not loaded")
            return [None] * len(texts)
        
        try:
            # Clean texts
            cleaned_texts = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cleaned_text = text.strip() if text else ""
                if cleaned_text:
                    cleaned_texts.append(cleaned_text)
                    text_indices.append(i)
            
            if not cleaned_texts:
                return [None] * len(texts)
            
            if self.use_gemini:
                # Generate embeddings using Gemini (one by one for now)
                embeddings = []
                for text in cleaned_texts:
                    try:
                        response = self.gemini_client.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(response['embedding'])
                        # Small delay to respect rate limits
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Failed to embed text: {e}")
                        embeddings.append(None)
            else:
                # Generate embeddings for valid texts using sentence transformers
                embeddings = self.model.encode(cleaned_texts, convert_to_tensor=False, batch_size=self.batch_size)
            
            # Map back to original positions
            result_embeddings = [None] * len(texts)
            for idx, embedding in zip(text_indices, embeddings):
                if embedding is not None:
                    if isinstance(embedding, np.ndarray):
                        result_embeddings[idx] = embedding.tolist()
                    else:
                        result_embeddings[idx] = embedding
            
            return result_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chunk to add embeddings"""
        try:
            text = chunk.get('text', '')
            
            if not text or len(text.strip()) < 10:
                # Keep chunk but with null embedding
                enhanced_chunk = chunk.copy()
                enhanced_chunk['embedding'] = None
                enhanced_chunk['embedding_model'] = self.model_name
                enhanced_chunk['embedding_generated_at'] = datetime.now().isoformat()
                return enhanced_chunk
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Update chunk with embedding
            enhanced_chunk = chunk.copy()
            enhanced_chunk['embedding'] = embedding
            enhanced_chunk['embedding_model'] = self.model_name
            enhanced_chunk['embedding_dimension'] = len(embedding) if embedding else None
            enhanced_chunk['embedding_generated_at'] = datetime.now().isoformat()
            
            return enhanced_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Return chunk with null embedding on error
            enhanced_chunk = chunk.copy()
            enhanced_chunk['embedding'] = None
            enhanced_chunk['embedding_model'] = self.model_name
            enhanced_chunk['embedding_generated_at'] = datetime.now().isoformat()
            enhanced_chunk['embedding_error'] = str(e)
            return enhanced_chunk
    
    def process_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of chunks efficiently"""
        try:
            # Extract texts for batch processing
            texts = []
            valid_indices = []
            
            for i, chunk in enumerate(chunks):
                text = chunk.get('text', '').strip()
                if text and len(text) >= 10:
                    texts.append(text)
                    valid_indices.append(i)
            
            # Generate embeddings for all valid texts at once
            if texts:
                embeddings = self.generate_embeddings_batch(texts)
            else:
                embeddings = []
            
            # Create enhanced chunks
            enhanced_chunks = []
            embedding_idx = 0
            
            for i, chunk in enumerate(chunks):
                enhanced_chunk = chunk.copy()
                
                if i in valid_indices and embedding_idx < len(embeddings):
                    embedding = embeddings[embedding_idx]
                    enhanced_chunk['embedding'] = embedding
                    enhanced_chunk['embedding_dimension'] = len(embedding) if embedding else None
                    embedding_idx += 1
                else:
                    enhanced_chunk['embedding'] = None
                    enhanced_chunk['embedding_dimension'] = None
                
                enhanced_chunk['embedding_model'] = self.model_name
                enhanced_chunk['embedding_generated_at'] = datetime.now().isoformat()
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return chunks with null embeddings on error
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = chunk.copy()
                enhanced_chunk['embedding'] = None
                enhanced_chunk['embedding_model'] = self.model_name
                enhanced_chunk['embedding_generated_at'] = datetime.now().isoformat()
                enhanced_chunk['embedding_error'] = str(e)
                enhanced_chunks.append(enhanced_chunk)
            return enhanced_chunks
    
    def process_all_chunks(self) -> bool:
        """Process all chunks to generate embeddings"""
        
        # Load chunks from keywords step
        keywords_file = self.input_dir / "all_chunks_with_keywords.json"
        
        if not keywords_file.exists():
            logger.error(f"Keywords file not found: {keywords_file}")
            print(f"❌ Keywords file not found: {keywords_file}")
            print(f"💡 Please run keywords.py first to create keywords!")
            return False
        
        # Load model first
        if not self.load_model():
            return False
        
        try:
            logger.info(f"Loading chunks from: {keywords_file}")
            
            with open(keywords_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                print("❌ No chunks found in input file")
                return False
            
            print(f"📁 Found {total_chunks} chunks to process")
            print(f"🧠 Using model: {self.model_name}")
            print(f"📦 Batch size: {self.batch_size}")
            
            # Estimate processing time
            estimated_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            estimated_time = estimated_batches * 2  # ~2 seconds per batch estimate
            print(f"⏱️  Estimated processing time: {estimated_time/60:.1f} minutes")
            
            # Process chunks in batches
            start_time = time.time()
            enhanced_chunks = []
            
            for i in range(0, total_chunks, self.batch_size):
                batch_end = min(i + self.batch_size, total_chunks)
                batch_chunks = chunks[i:batch_end]
                
                # Process batch
                batch_enhanced = self.process_chunks_batch(batch_chunks)
                enhanced_chunks.extend(batch_enhanced)
                
                # Progress reporting
                processed = len(enhanced_chunks)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_chunks - processed) / rate if rate > 0 else 0
                
                print(f"📊 Progress: {processed}/{total_chunks} ({processed/total_chunks*100:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update data structure
            enhanced_data = data.copy()
            enhanced_data['chunks'] = enhanced_chunks
            enhanced_data['metadata'].update({
                'embeddings_generated': True,
                'embedding_generation_completed_at': datetime.now().isoformat(),
                'embedding_model': self.model_name,
                'embedding_batch_size': self.batch_size,
                'processing_time_seconds': processing_time,
                'chunks_per_second': total_chunks / processing_time if processing_time > 0 else 0
            })
            
            # Save results
            output_file = self.output_dir / "all_chunks_with_embeddings.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            # Calculate statistics
            chunks_with_embeddings = sum(1 for chunk in enhanced_chunks if chunk.get('embedding') is not None)
            embedding_dimensions = [len(chunk.get('embedding', [])) for chunk in enhanced_chunks if chunk.get('embedding')]
            avg_dimension = sum(embedding_dimensions) / len(embedding_dimensions) if embedding_dimensions else 0
            
            # Save detailed statistics
            self._save_embedding_statistics(enhanced_chunks, enhanced_data['metadata'])
            
            print(f"\n✅ Embedding generation completed!")
            print(f"⏱️  Processing time: {processing_time/60:.1f} minutes")
            print(f"📊 Total chunks: {total_chunks:,}")
            print(f"🎯 Chunks with embeddings: {chunks_with_embeddings:,}")
            print(f"📈 Success rate: {chunks_with_embeddings/total_chunks*100:.1f}%")
            print(f"📏 Embedding dimension: {int(avg_dimension) if avg_dimension else 'N/A'}")
            print(f"🚀 Processing rate: {total_chunks/(processing_time/60):.1f} chunks/minute")
            print(f"📁 Results saved to: {output_file}")
            
            # Show sample results
            self._show_sample_results(enhanced_chunks)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            print(f"❌ Error processing chunks: {e}")
            return False
    
    def _save_embedding_statistics(self, enhanced_chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Save detailed embedding generation statistics"""
        
        # Calculate statistics
        total_chunks = len(enhanced_chunks)
        chunks_with_embeddings = sum(1 for chunk in enhanced_chunks if chunk.get('embedding') is not None)
        
        # Embedding dimension analysis
        embedding_dimensions = []
        embedding_sizes = []
        for chunk in enhanced_chunks:
            embedding = chunk.get('embedding')
            if embedding:
                dim = len(embedding)
                embedding_dimensions.append(dim)
                # Estimate size in bytes (assuming float32)
                embedding_sizes.append(dim * 4)
        
        # Text length analysis
        text_lengths = [len(chunk.get('text', '')) for chunk in enhanced_chunks]
        
        # Error analysis
        chunks_with_errors = sum(1 for chunk in enhanced_chunks if chunk.get('embedding_error'))
        error_types = {}
        for chunk in enhanced_chunks:
            error = chunk.get('embedding_error')
            if error:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Compile statistics
        stats = {
            "embedding_summary": {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "chunks_with_errors": chunks_with_errors,
                "success_rate": round(chunks_with_embeddings/total_chunks*100, 2) if total_chunks > 0 else 0,
                "processing_time_minutes": metadata.get('processing_time_seconds', 0) / 60,
                "chunks_per_minute": metadata.get('chunks_per_second', 0) * 60
            },
            "embedding_analysis": {
                "model_used": metadata.get('embedding_model', 'unknown'),
                "embedding_dimension": embedding_dimensions[0] if embedding_dimensions else None,
                "total_embeddings_generated": len(embedding_dimensions),
                "average_embedding_size_bytes": round(sum(embedding_sizes) / len(embedding_sizes), 2) if embedding_sizes else 0,
                "total_embedding_storage_mb": round(sum(embedding_sizes) / (1024 * 1024), 2) if embedding_sizes else 0
            },
            "text_analysis": {
                "average_text_length": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "total_text_characters": sum(text_lengths)
            },
            "error_analysis": {
                "error_count": chunks_with_errors,
                "error_rate": round(chunks_with_errors/total_chunks*100, 2) if total_chunks > 0 else 0,
                "error_types": error_types
            },
            "configuration": {
                "embedding_model": self.model_name,
                "batch_size": self.batch_size,
                "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown') if self.model else 'unknown'
            },
            "processed_at": datetime.now().isoformat()
        }
        
        # Save statistics
        stats_file = self.output_dir / "embedding_generation_summary.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Statistics saved to: {stats_file}")
    
    def _show_sample_results(self, enhanced_chunks: List[Dict[str, Any]]):
        """Show sample embedding generation results"""
        print(f"\n🔍 Sample Embedding Results:")
        
        sample_count = 0
        for chunk in enhanced_chunks:
            embedding = chunk.get('embedding')
            if embedding and sample_count < 2:
                text_preview = chunk.get('text', '')[:100] + "..." if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                keywords = chunk.get('keywords', [])
                
                print(f"\n   📄 Sample {sample_count + 1}:")
                print(f"   Document: {chunk.get('doc_name', 'Unknown')}")
                print(f"   Page: {chunk.get('page', 'N/A')}")
                print(f"   Keywords: {keywords}")
                print(f"   Embedding dimension: {len(embedding)}")
                print(f"   Embedding preview: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
                print(f"   Text: {text_preview}")
                sample_count += 1
        
        if sample_count == 0:
            print("   ⚠️  No embeddings found in sample chunks")

def main():
    """Main function to run embedding generation"""
    print(f"🧠 Embedding Generation for RAG Pipeline")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Input from keywords: {project_root / 'data' / 'keywords'}")
    print(f"📁 Output to embeddings: {project_root / 'data' / 'embeddings'}")
    print(f"=" * 60)
    
    # Initialize generator
    generator = EmbeddingsGenerator()
    
    # Check if input exists
    if not generator.input_dir.exists():
        print("❌ Keywords directory not found")
        print(f"💡 Please run keywords.py first to create keywords!")
        return
    
    keywords_file = generator.input_dir / "all_chunks_with_keywords.json"
    if not keywords_file.exists():
        print("❌ Keywords file not found")
        print(f"💡 Please run keywords.py first to create all_chunks_with_keywords.json!")
        return
    
    # Process chunks
    success = generator.process_all_chunks()
    
    if success:
        print(f"\n🎉 Embedding generation completed successfully!")
        print(f"🔄 Ready for next pipeline step: theme_identification.py")
    else:
        print(f"\n❌ Embedding generation failed")

if __name__ == "__main__":
    main()