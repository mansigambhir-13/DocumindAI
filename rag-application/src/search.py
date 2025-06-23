"""
Enhanced Hybrid Advanced Semantic Search & Answer Generation Script
Works with both local embeddings and Qdrant vector database
Improved error handling, fallback mechanisms, and better answer generation
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("[OK] Loaded .env file")
            return True
    except ImportError:
        pass
    
    env_files = [Path('.env'), project_root / '.env']
    for env_file in env_files:
        if env_file.exists():
            print(f"[INFO] Loading .env from: {env_file}")
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"').strip("'")
                print(f"[OK] Loaded environment variables")
                return True
            except Exception as e:
                print(f"[ERROR] Error reading {env_file}: {e}")
    return False

load_env_file()

@dataclass
class SearchConfig:
    """Configuration for semantic search"""
    
    # Local embeddings (primary method)
    EMBEDDINGS_DIR: str = str(project_root / "data" / "embeddings")
    
    # Qdrant settings (optional, for advanced users)
    COLLECTION_NAME: str = "document_chunks"
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', '6333'))
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Search settings
    TOP_K_CHUNKS: int = int(os.getenv('TOP_K_CHUNKS', '10'))
    FINAL_CHUNKS: int = int(os.getenv('FINAL_CHUNKS', '3'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    
    # Answer generation settings
    ANSWER_MODEL: str = os.getenv('ANSWER_MODEL', 'gpt-3.5-turbo')
    MAX_CONTEXT_LENGTH: int = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration"""
        if self.QDRANT_URL and self.QDRANT_API_KEY:
            return {"url": self.QDRANT_URL, "api_key": self.QDRANT_API_KEY}
        else:
            return {"host": self.QDRANT_HOST, "port": self.QDRANT_PORT}

# Initialize configuration
config = SearchConfig()

# Setup logging
def setup_logging() -> logging.Logger:
    """Setup logging for search operations"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "search.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class AdvancedTextProcessor:
    """Advanced text processing for better answer generation"""
    
    @staticmethod
    def extract_key_sentences(text: str, query_keywords: List[str], max_sentences: int = 3) -> List[str]:
        """Extract the most relevant sentences from text based on query keywords"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return [text[:200] + "..." if len(text) > 200 else text]
        
        # Score sentences based on keyword presence
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            for keyword in query_keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # Bonus for sentence length (not too short, not too long)
            if 50 <= len(sentence) <= 300:
                score += 0.5
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        return top_sentences if top_sentences else sentences[:max_sentences]
    
    @staticmethod
    def create_structured_summary(chunks: List[Dict[str, Any]], query: str) -> str:
        """Create a structured summary from multiple chunks"""
        query_keywords = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        query_keywords = [w for w in query_keywords if len(w) > 2]
        
        # Group chunks by document
        doc_groups = {}
        for chunk in chunks:
            doc_name = chunk.get('doc_name', 'Unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(chunk)
        
        summary_parts = []
        
        for doc_name, doc_chunks in doc_groups.items():
            doc_parts = []
            for chunk in doc_chunks:
                # Extract key sentences from this chunk
                key_sentences = AdvancedTextProcessor.extract_key_sentences(
                    chunk['text'], query_keywords, max_sentences=2
                )
                if key_sentences:
                    doc_parts.extend(key_sentences)
            
            if doc_parts:
                doc_summary = f"**From {doc_name}:**\n" + " ".join(doc_parts[:3])
                summary_parts.append(doc_summary)
        
        return "\n\n".join(summary_parts)

class HybridSemanticSearchEngine:
    """Enhanced hybrid semantic search engine with improved answer generation"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        
        # Initialize search backend (local or Qdrant)
        self.use_qdrant = False
        self.use_local = False
        
        # Local embeddings data
        self.chunks_data = None
        self.embeddings_matrix = None
        
        # Qdrant client
        self.qdrant_client = None
        
        # Embedding and answer generation clients
        self.embedding_client = None
        self.answer_client = None
        self.gemini_client = None
        self.anthropic_client = None
        self.embedding_service = None
        self.embedding_dim = None
        
        # Initialize search backend
        self._initialize_search_backend()
        self._initialize_embedding_service()
        self._initialize_answer_generation()
        
        logger.info("Enhanced HybridSemanticSearchEngine initialized successfully")
    
    def _initialize_search_backend(self):
        """Initialize search backend (try Qdrant first, fallback to local)"""
        
        # Try Qdrant first (for advanced users)
        if self._try_initialize_qdrant():
            self.use_qdrant = True
            print("[INFO] Using Qdrant vector database for search")
            return
        
        # Fallback to local embeddings
        if self._try_initialize_local():
            self.use_local = True
            print("[INFO] Using local embeddings for search")
            return
        
        raise ValueError("No search backend available. Please run embeddings.py first or set up Qdrant.")
    
    def _try_initialize_qdrant(self) -> bool:
        """Try to initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient
            
            qdrant_config = self.config.get_qdrant_config()
            
            if "url" in qdrant_config and qdrant_config["url"]:
                self.qdrant_client = QdrantClient(
                    url=qdrant_config["url"],
                    api_key=qdrant_config["api_key"],
                    timeout=60.0
                )
                logger.info(f"[OK] Connected to Qdrant Cloud")
            else:
                self.qdrant_client = QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config["port"],
                    timeout=30.0
                )
                logger.info(f"[OK] Connected to local Qdrant")
            
            # Verify collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.COLLECTION_NAME not in collection_names:
                logger.warning(f"Collection '{self.config.COLLECTION_NAME}' not found")
                return False
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.config.COLLECTION_NAME)
            self.embedding_dim = collection_info.config.params.vectors.size
            
            logger.info(f"[OK] Collection '{self.config.COLLECTION_NAME}' ready ({collection_info.points_count:,} points)")
            return True
            
        except Exception as e:
            logger.info(f"Qdrant not available: {e}")
            return False
    
    def _try_initialize_local(self) -> bool:
        """Try to initialize local embeddings"""
        try:
            embeddings_file = Path(self.config.EMBEDDINGS_DIR) / "all_chunks_with_embeddings.json"
            
            if not embeddings_file.exists():
                logger.warning(f"Local embeddings file not found: {embeddings_file}")
                return False
            
            logger.info(f"Loading embeddings from: {embeddings_file}")
            
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            if not chunks:
                logger.warning("No chunks found in embeddings file")
                return False
            
            # Extract chunks with valid embeddings
            valid_chunks = []
            embeddings_list = []
            
            for chunk in chunks:
                embedding = chunk.get('embedding')
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    valid_chunks.append(chunk)
                    embeddings_list.append(embedding)
            
            if not valid_chunks:
                logger.warning("No valid embeddings found")
                return False
            
            # Convert to numpy matrix for efficient similarity calculations
            self.embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            self.chunks_data = valid_chunks
            self.embedding_dim = self.embeddings_matrix.shape[1]
            
            logger.info(f"Loaded {len(valid_chunks)} chunks with embeddings")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Local embeddings not available: {e}")
            return False
    
    def _initialize_embedding_service(self):
        """Initialize embedding service with better fallbacks"""
        try:
            # Try Gemini first (matches your embeddings)
            if self.config.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                self.gemini_client = genai
                self.embedding_service = "gemini"
                self.config.EMBEDDING_MODEL = "text-embedding-004"
                
                # Test connection
                test_response = genai.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content="test query",
                    task_type="retrieval_query"
                )
                logger.info(f"[OK] Gemini embedding service ready: {self.config.EMBEDDING_MODEL}")
                return
            
            # Try OpenAI
            if (self.config.EMBEDDING_MODEL.startswith("text-embedding") and 
                self.config.OPENAI_API_KEY):
                
                import openai
                self.embedding_client = openai.OpenAI(
                    api_key=self.config.OPENAI_API_KEY,
                    timeout=60.0
                )
                self.embedding_service = "openai"
                
                # Test connection
                test_response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input="test query"
                )
                logger.info(f"[OK] OpenAI embedding service ready: {self.config.EMBEDDING_MODEL}")
                return
            
            # Fallback to sentence transformers for local queries
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_client = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_service = "sentence_transformers"
                logger.info("[OK] Sentence Transformers fallback ready")
                return
            except ImportError:
                pass
            
            raise ValueError("No embedding service configured")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize embedding service: {e}")
            raise
    
    def _initialize_answer_generation(self):
        """Initialize answer generation service with multiple providers"""
        self.available_answer_services = []
        
        # Try Anthropic (Claude)
        if self.config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
                self.available_answer_services.append("anthropic")
                logger.info("[OK] Anthropic (Claude) answer generation ready")
            except ImportError:
                logger.warning("[WARN] Anthropic library not installed")
            except Exception as e:
                logger.warning(f"[WARN] Anthropic setup failed: {e}")
        
        # Try Gemini
        if self.config.GEMINI_API_KEY:
            if not self.gemini_client:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                self.gemini_client = genai
            self.available_answer_services.append("gemini")
            logger.info("[OK] Gemini answer generation ready")
        
        # Try OpenAI
        if self.config.OPENAI_API_KEY:
            try:
                import openai
                self.answer_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                self.available_answer_services.append("openai")
                logger.info(f"[OK] OpenAI answer generation ready: {self.config.ANSWER_MODEL}")
            except Exception as e:
                logger.warning(f"[WARN] OpenAI setup failed: {e}")
        
        # Try Groq
        if self.config.GROQ_API_KEY:
            self.available_answer_services.append("groq")
            logger.info("[OK] Groq answer generation ready")
        
        if not self.available_answer_services:
            logger.warning("[WARN] No answer generation service configured - using fallback mode")
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for user query"""
        try:
            if self.embedding_service == "gemini":
                response = self.gemini_client.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
                return response['embedding']
            
            elif self.embedding_service == "openai":
                response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=query
                )
                return response.data[0].embedding
            
            elif self.embedding_service == "sentence_transformers":
                embedding = self.embedding_client.encode(query, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create query embedding: {e}")
            raise
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract keywords from user query for enhanced matching"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'what', 'when', 'where', 'why', 'how',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def perform_semantic_search_local(self, query: str, 
                                    top_k: Optional[int] = None,
                                    doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using local embeddings"""
        try:
            logger.info(f"[INFO] Performing local search for: '{query}'")
            
            # Create query embedding
            query_embedding = self.create_query_embedding(query)
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Calculate cosine similarities
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            
            # Calculate cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            
            # Extract query keywords for bonus scoring
            query_keywords = self.extract_keywords_from_query(query)
            
            # Create results with scores
            results = []
            for i, (chunk, similarity) in enumerate(zip(self.chunks_data, similarities)):
                
                # Apply document filter if specified
                if doc_filter and doc_filter.lower() not in chunk.get('doc_name', '').lower():
                    continue
                
                # Apply similarity threshold
                if similarity < self.config.SIMILARITY_THRESHOLD:
                    continue
                
                # Calculate keyword overlap bonus
                chunk_keywords = chunk.get('keywords', [])
                keyword_bonus = 0.0
                if chunk_keywords and query_keywords:
                    keyword_overlap = len(set(query_keywords) & set(chunk_keywords))
                    keyword_bonus = keyword_overlap / len(query_keywords) * 0.1  # 10% bonus max
                
                adjusted_score = float(similarity) + keyword_bonus
                
                result = {
                    "id": i,
                    "score": float(similarity),
                    "adjusted_score": adjusted_score,
                    "keyword_bonus": keyword_bonus,
                    "confidence": self._calculate_confidence(float(similarity)),
                    "chunk_id": chunk.get('chunk_id', f"chunk_{i}"),
                    "doc_name": chunk.get('doc_name', 'Unknown'),
                    "doc_id": chunk.get('doc_id', 'Unknown'),
                    "page": chunk.get('page', 'N/A'),
                    "chunk_number": chunk.get('chunk_number', i),
                    "text": chunk.get('text', ''),
                    "keywords": chunk.get('keywords', []),
                    "text_length": len(chunk.get('text', '')),
                    "embedding_model": chunk.get('embedding_model', self.config.EMBEDDING_MODEL)
                }
                
                results.append(result)
            
            # Sort by adjusted score (semantic + keyword matching)
            results.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
            # Limit results
            if top_k is None:
                top_k = self.config.TOP_K_CHUNKS
            
            results = results[:top_k]
            
            logger.info(f"[OK] Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Local search failed: {e}")
            return []
    
    def perform_semantic_search_qdrant(self, query: str, 
                                      top_k: Optional[int] = None,
                                      doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using Qdrant"""
        try:
            logger.info(f"[INFO] Performing Qdrant search for: '{query}'")
            
            # Create query embedding
            query_embedding = self.create_query_embedding(query)
            
            # Extract keywords for additional filtering
            query_keywords = self.extract_keywords_from_query(query)
            logger.info(f"[INFO] Extracted keywords: {query_keywords}")
            
            # Set search parameters
            if top_k is None:
                top_k = self.config.TOP_K_CHUNKS
            
            # Prepare filters
            search_filter = None
            if doc_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_filter))]
                )
            
            # Perform vector search
            try:
                # Try newer query_points method
                search_results = self.qdrant_client.query_points(
                    collection_name=self.config.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=self.config.SIMILARITY_THRESHOLD,
                    with_payload=True,
                    with_vectors=False
                )
                results = search_results.points
            except AttributeError:
                # Fallback to older search method
                results = self.qdrant_client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=self.config.SIMILARITY_THRESHOLD,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Process and enhance results
            processed_results = []
            for result in results:
                chunk_data = {
                    "id": result.id,
                    "score": result.score,
                    "confidence": self._calculate_confidence(result.score),
                    "chunk_id": result.payload.get("chunk_id", f"chunk_{result.id}"),
                    "doc_name": result.payload["doc_name"],
                    "doc_id": result.payload.get("doc_id", "unknown"),
                    "page": result.payload.get("page", "N/A"),
                    "chunk_number": result.payload.get("chunk_number", result.id),
                    "text": result.payload["text"],
                    "keywords": result.payload.get("keywords", []),
                    "text_length": len(result.payload.get("text", "")),
                    "embedding_model": result.payload.get("embedding_model", ""),
                }
                
                # Calculate keyword overlap bonus
                chunk_keywords = chunk_data["keywords"]
                if chunk_keywords and query_keywords:
                    keyword_overlap = len(set(query_keywords) & set(chunk_keywords))
                    keyword_bonus = keyword_overlap / len(query_keywords) * 0.1  # 10% bonus max
                    chunk_data["keyword_bonus"] = keyword_bonus
                    chunk_data["adjusted_score"] = chunk_data["score"] + keyword_bonus
                else:
                    chunk_data["keyword_bonus"] = 0.0
                    chunk_data["adjusted_score"] = chunk_data["score"]
                
                processed_results.append(chunk_data)
            
            # Sort by adjusted score (semantic + keyword matching)
            processed_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
            logger.info(f"[OK] Found {len(processed_results)} relevant chunks")
            return processed_results
            
        except Exception as e:
            logger.error(f"[ERROR] Qdrant search failed: {e}")
            return []
    
    def perform_semantic_search(self, query: str, 
                               top_k: Optional[int] = None,
                               doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using the best available method"""
        if self.use_qdrant:
            return self.perform_semantic_search_qdrant(query, top_k, doc_filter)
        elif self.use_local:
            return self.perform_semantic_search_local(query, top_k, doc_filter)
        else:
            raise ValueError("No search backend available")
    
    def select_best_chunks_for_answer(self, search_results: List[Dict[str, Any]], 
                                    query: str) -> List[Dict[str, Any]]:
        """Select the best chunks for answer generation using advanced ranking"""
        
        if not search_results:
            return []
        
        logger.info(f"[INFO] Selecting best chunks from {len(search_results)} candidates")
        
        # Take top percentage of chunks based on similarity score
        top_percent = 0.3  # Top 30% of results
        top_count = max(1, int(len(search_results) * top_percent))
        top_chunks = search_results[:top_count]
        
        # Advanced ranking considering multiple factors
        for chunk in top_chunks:
            rank_score = 0.0
            
            # 1. Base similarity score (70% weight)
            rank_score += chunk["adjusted_score"] * 0.7
            
            # 2. Text length bonus (longer chunks often have more context)
            text_length_norm = min(chunk["text_length"] / 1000, 1.0)  # Normalize to 1000 chars
            rank_score += text_length_norm * 0.1
            
            # 3. Keyword density bonus
            if chunk["keywords"]:
                query_keywords = self.extract_keywords_from_query(query)
                if query_keywords:
                    keyword_density = len(chunk["keywords"]) / len(query_keywords)
                    rank_score += min(keyword_density, 1.0) * 0.1
            
            # 4. Document diversity (prefer chunks from different documents)
            rank_score += 0.05  # Base diversity bonus
            
            chunk["final_rank_score"] = rank_score
        
        # Sort by final ranking score
        top_chunks.sort(key=lambda x: x["final_rank_score"], reverse=True)
        
        # Select final chunks ensuring diversity
        selected_chunks = []
        used_docs = set()
        
        for chunk in top_chunks:
            # Ensure document diversity
            if len(selected_chunks) < self.config.FINAL_CHUNKS:
                selected_chunks.append(chunk)
                used_docs.add(chunk["doc_name"])
            elif chunk["doc_name"] not in used_docs and len(selected_chunks) < self.config.FINAL_CHUNKS + 2:
                selected_chunks.append(chunk)
                used_docs.add(chunk["doc_name"])
        
        # Limit to final count
        selected_chunks = selected_chunks[:self.config.FINAL_CHUNKS]
        
        logger.info(f"[OK] Selected {len(selected_chunks)} best chunks for answer generation")
        return selected_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive answer using retrieved chunks with improved fallbacks"""
        
        if not context_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "confidence": "low",
                "sources": []
            }
        
        try:
            logger.info(f"[INFO] Generating answer using {len(context_chunks)} chunks")
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, chunk in enumerate(context_chunks, 1):
                context_part = f"[Source {i}]\n"
                context_part += f"Document: {chunk['doc_name']}\n"
                context_part += f"Page: {chunk['page']}, Chunk: {chunk['chunk_number']}\n"
                context_part += f"Content: {chunk['text']}\n"
                context_part += f"Relevance Score: {chunk['score']:.3f}\n"
                
                context_parts.append(context_part)
                
                sources.append({
                    "doc_name": chunk['doc_name'],
                    "page": chunk['page'],
                    "chunk_number": chunk['chunk_number'],
                    "chunk_id": chunk['chunk_id'],
                    "relevance_score": round(chunk['score'], 3),
                    "keywords": chunk.get('keywords', [])
                })
            
            # Combine context (limit length)
            full_context = "\n\n".join(context_parts)
            if len(full_context) > self.config.MAX_CONTEXT_LENGTH:
                # Truncate context to fit limit
                full_context = full_context[:self.config.MAX_CONTEXT_LENGTH] + "..."
            
            # Try to generate answer with available AI services
            answer = None
            service_used = None
            
            for service in self.available_answer_services:
                try:
                    answer = self._generate_answer_with_service(query, full_context, service)
                    if answer and len(answer.strip()) > 50:  # Valid answer
                        service_used = service
                        break
                except Exception as e:
                    logger.warning(f"Answer generation with {service} failed: {e}")
                    continue
            
            # If AI generation failed, create enhanced fallback
            if not answer or len(answer.strip()) < 50:
                logger.info("AI answer generation failed, using enhanced fallback")
                answer = self._create_enhanced_fallback_answer(query, context_chunks)
                service_used = "enhanced_fallback"
            
            # Calculate confidence based on chunk scores and coverage
            avg_score = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)
            confidence = self._calculate_answer_confidence(avg_score, len(context_chunks))
            
            logger.info(f"[OK] Answer generated with {confidence} confidence using {service_used}")
            
            return {
                "answer": answer.strip(),
                "confidence": confidence,
                "sources": sources,
                "context_chunks_used": len(context_chunks),
                "avg_relevance_score": round(avg_score, 3),
                "query": query,
                "service_used": service_used,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Answer generation failed: {e}")
            return {
                "answer": self._create_enhanced_fallback_answer(query, context_chunks),
                "confidence": "medium",
                "sources": sources if 'sources' in locals() else [],
                "service_used": "error_fallback"
            }
    
    def _generate_answer_with_service(self, query: str, context: str, service: str) -> str:
        """Generate answer using specified AI service"""
        
        # Create comprehensive prompt
        prompt = f"""You are an expert assistant that provides accurate, comprehensive answers based on provided context.

USER QUESTION: {query}

RELEVANT CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive and accurate answer based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state what information is missing
3. Include specific references to sources when making claims
4. Structure your answer clearly with key points
5. If there are conflicting information in sources, mention it
6. Be concise but thorough
7. Use bullet points or numbered lists when appropriate for clarity

ANSWER:"""

        if service == "anthropic":
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif service == "gemini":
            model = self.gemini_client.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        
        elif service == "openai":
            response = self.answer_client.chat.completions.create(
                model=self.config.ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        
        elif service == "groq":
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Groq API error: {response.status_code}")
        
        else:
            raise Exception(f"Unknown service: {service}")
    
    def _create_enhanced_fallback_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create an enhanced answer when AI generation is not available"""
        
        query_keywords = self.extract_keywords_from_query(query)
        
        # Create structured summary using the advanced text processor
        structured_summary = AdvancedTextProcessor.create_structured_summary(chunks, query)
        
        if structured_summary:
            answer = f"**Answer to: {query}**\n\n{structured_summary}"
        else:
            # Fallback to simple concatenation
            relevant_texts = []
            for i, chunk in enumerate(chunks, 1):
                text = chunk['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                relevant_texts.append(f"**Source {i} (from {chunk['doc_name']}, Page {chunk['page']}):**\n{text}")
            
            answer = f"**Based on the available documents, here's what I found regarding: {query}**\n\n"
            answer += "\n\n".join(relevant_texts)
        
        # Add source summary
        unique_docs = list(set(chunk['doc_name'] for chunk in chunks))
        answer += f"\n\n**Sources:** Information compiled from {len(chunks)} relevant sections across {len(unique_docs)} document(s): {', '.join(unique_docs)}"
        
        # Add relevance note
        avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
        answer += f"\n\n**Relevance Score:** {avg_score:.3f} (Higher scores indicate better relevance to your question)"
        
        return answer
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.7:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _calculate_answer_confidence(self, avg_score: float, num_chunks: int) -> str:
        """Calculate overall answer confidence"""
        base_confidence = avg_score
        
        # Bonus for multiple supporting chunks
        if num_chunks >= 3:
            base_confidence += 0.1
        elif num_chunks >= 2:
            base_confidence += 0.05
        
        if base_confidence >= 0.8:
            return "very_high"
        elif base_confidence >= 0.7:
            return "high"
        elif base_confidence >= 0.6:
            return "medium"
        elif base_confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def search_and_answer(self, query: str, 
                         doc_filter: Optional[str] = None,
                         detailed_output: bool = False) -> Dict[str, Any]:
        """Complete search and answer pipeline"""
        
        start_time = time.time()
        
        try:
            # Step 1: Perform semantic search
            search_results = self.perform_semantic_search(query, doc_filter=doc_filter)
            
            if not search_results:
                return {
                    "query": query,
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or check if the relevant documents have been processed.",
                    "confidence": "none",
                    "sources": [],
                    "search_results_count": 0,
                    "processing_time": time.time() - start_time,
                    "suggestions": [
                        "Try using different keywords",
                        "Check if documents are properly processed",
                        "Verify embeddings are generated"
                    ]
                }
            
            # Step 2: Select best chunks
            best_chunks = self.select_best_chunks_for_answer(search_results, query)
            
            # Step 3: Generate answer
            answer_result = self.generate_answer(query, best_chunks)
            
            # Step 4: Compile final result
            result = {
                "query": query,
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "sources": answer_result["sources"],
                "search_results_count": len(search_results),
                "chunks_used_for_answer": len(best_chunks),
                "processing_time": round(time.time() - start_time, 2),
                "service_used": answer_result.get("service_used", "unknown")
            }
            
            if detailed_output:
                result["all_search_results"] = search_results
                result["selected_chunks"] = best_chunks
                result["search_metadata"] = {
                    "backend": "qdrant" if self.use_qdrant else "local",
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "embedding_service": self.embedding_service,
                    "similarity_threshold": self.config.SIMILARITY_THRESHOLD,
                    "total_chunks_in_db": len(self.chunks_data) if self.use_local else "unknown",
                    "available_answer_services": self.available_answer_services
                }
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Search and answer pipeline failed: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "confidence": "error",
                "sources": [],
                "processing_time": time.time() - start_time,
                "error_details": str(e)
            }

def test_search_functionality():
    """Test the search functionality with diagnostic information"""
    print("=" * 60)
    print("TESTING ENHANCED SEMANTIC SEARCH ENGINE")
    print("=" * 60)
    
    try:
        # Initialize search engine
        print("[INFO] Initializing search engine...")
        search_engine = HybridSemanticSearchEngine(config)
        print("[OK] Search engine initialized successfully!")
        
        # Display configuration
        print(f"\n[CONFIG] Search Engine Status:")
        print(f"   Backend: {'Qdrant' if search_engine.use_qdrant else 'Local Embeddings'}")
        print(f"   Embedding Service: {search_engine.embedding_service}")
        print(f"   Available Answer Services: {search_engine.available_answer_services}")
        
        if search_engine.use_local:
            print(f"   Total chunks available: {len(search_engine.chunks_data):,}")
            print(f"   Embedding dimension: {search_engine.embedding_dim}")
        
        # Test with a sample query
        test_query = "What is SEBI's role in algorithmic trading?"
        print(f"\n[TEST] Testing with query: '{test_query}'")
        
        result = search_engine.search_and_answer(test_query, detailed_output=True)
        
        print(f"\n[RESULT] Query: {result['query']}")
        print(f"[RESULT] Confidence: {result['confidence']}")
        print(f"[RESULT] Service Used: {result.get('service_used', 'unknown')}")
        print(f"[RESULT] Processing Time: {result['processing_time']}s")
        print(f"[RESULT] Sources Found: {len(result['sources'])}")
        
        print(f"\n[ANSWER]")
        print("-" * 50)
        print(result['answer'])
        print("-" * 50)
        
        if result['sources']:
            print(f"\n[SOURCES]")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['doc_name']} (Page {source['page']}, Score: {source['relevance_score']})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print(f"\n[TROUBLESHOOTING]")
        print("1. Check if embeddings have been generated: run embeddings.py")
        print("2. Verify API keys are set in .env file")
        print("3. Install required packages: pip install sentence-transformers google-generativeai")
        return False

def interactive_search_session():
    """Enhanced interactive search session"""
    print("=" * 60)
    print("ENHANCED HYBRID SEMANTIC SEARCH & ANSWER GENERATION")
    print("Supports Local Embeddings, Qdrant, and Multiple AI Services")
    print("=" * 60)
    
    try:
        # Initialize search engine
        print("[INFO] Initializing enhanced search engine...")
        search_engine = HybridSemanticSearchEngine(config)
        print("[OK] Search engine ready!")
        
        print(f"\n[INFO] Knowledge Base Statistics:")
        if search_engine.use_qdrant:
            collection_info = search_engine.qdrant_client.get_collection(config.COLLECTION_NAME)
            print(f"   Backend: Qdrant Vector Database")
            print(f"   Collection: {config.COLLECTION_NAME}")
            print(f"   Total chunks: {collection_info.points_count:,}")
            print(f"   Vector dimension: {collection_info.config.params.vectors.size}")
        elif search_engine.use_local:
            print(f"   Backend: Local Embeddings")
            print(f"   Total chunks: {len(search_engine.chunks_data):,}")
            print(f"   Vector dimension: {search_engine.embedding_dim}")
        
        print(f"   Embedding service: {search_engine.embedding_service}")
        print(f"   Model: {search_engine.config.EMBEDDING_MODEL}")
        print(f"   Available AI services: {', '.join(search_engine.available_answer_services) if search_engine.available_answer_services else 'Enhanced Fallback Only'}")
        
        # Get document list for reference
        if search_engine.use_local:
            doc_names = list(set(chunk.get('doc_name', 'Unknown') for chunk in search_engine.chunks_data))
            print(f"   Unique documents: {len(doc_names)}")
        
        print(f"\n[INFO] Search Settings:")
        print(f"   Top-K retrieval: {config.TOP_K_CHUNKS}")
        print(f"   Final chunks for answer: {config.FINAL_CHUNKS}")
        print(f"   Similarity threshold: {config.SIMILARITY_THRESHOLD}")
        
        print(f"\n" + "=" * 60)
        print("Ask questions about your documents (type 'quit' to exit)")
        print("Commands:")
        print("  - 'help': Show commands")
        print("  - 'docs': List available documents (local mode only)")
        print("  - 'test': Run functionality test")
        print("  - 'stats': Show search statistics")
        print("  - 'detailed <question>': Get detailed search results")
        print("=" * 60)
        
        search_count = 0
        
        while True:
            try:
                query = input("\n[QUESTION] ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit']:
                    print("[INFO] Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\n[HELP] Available commands:")
                    print("  - Ask any question about your documents")
                    print("  - 'detailed <question>': Get detailed search results")
                    print("  - 'docs': List available documents")
                    print("  - 'test': Run functionality test")
                    print("  - 'stats': Show search statistics")
                    print("  - 'quit': Exit the program")
                    continue
                
                if query.lower() == 'test':
                    test_search_functionality()
                    continue
                
                if query.lower() == 'docs':
                    if search_engine.use_local:
                        doc_names = list(set(chunk.get('doc_name', 'Unknown') for chunk in search_engine.chunks_data))
                        print(f"\n[DOCS] Available documents ({len(doc_names)}):")
                        for i, doc_name in enumerate(doc_names[:20], 1):
                            print(f"   {i}. {doc_name}")
                        if len(doc_names) > 20:
                            print(f"   ... and {len(doc_names) - 20} more")
                    else:
                        print("\n[INFO] Document listing only available in local mode")
                    continue
                
                if query.lower() == 'stats':
                    print(f"\n[STATS] Session Statistics:")
                    print(f"   Searches performed: {search_count}")
                    print(f"   Backend: {'Qdrant' if search_engine.use_qdrant else 'Local'}")
                    print(f"   Embedding service: {search_engine.embedding_service}")
                    print(f"   Available AI services: {len(search_engine.available_answer_services)}")
                    continue
                
                # Check for detailed search request
                detailed = False
                if query.lower().startswith('detailed '):
                    detailed = True
                    query = query[9:]  # Remove 'detailed ' prefix
                
                search_count += 1
                print(f"\n[SEARCH] Processing question {search_count}...")
                
                # Perform search and answer
                result = search_engine.search_and_answer(query, detailed_output=detailed)
                
                # Display results
                print(f"\n[ANSWER] ({result['confidence']} confidence)")
                print("=" * 50)
                print(result['answer'])
                print("=" * 50)
                
                print(f"\n[SOURCES] ({len(result['sources'])} sources)")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['doc_name']} (Page {source['page']}, Chunk {source['chunk_number']})")
                    print(f"   Relevance: {source['relevance_score']:.3f}")
                    if source['keywords']:
                        print(f"   Keywords: {', '.join(source['keywords'][:5])}")
                
                print(f"\n[METADATA]")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Search results: {result['search_results_count']}")
                print(f"   Chunks used: {result['chunks_used_for_answer']}")
                print(f"   Service used: {result.get('service_used', 'unknown')}")
                
                if detailed and 'all_search_results' in result:
                    print(f"\n[DETAILED] All Search Results:")
                    for i, chunk in enumerate(result['all_search_results'][:10], 1):
                        print(f"{i}. Score: {chunk['score']:.3f} | {chunk['doc_name']} (Page {chunk['page']})")
                        print(f"   Text: {chunk['text'][:100]}...")
                
            except KeyboardInterrupt:
                print("\n[INFO] Search interrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
                continue
    
    except Exception as e:
        print(f"[ERROR] Failed to initialize search engine: {e}")
        print("\n[INFO] Troubleshooting tips:")
        print("   1. Make sure embeddings.py has been run successfully")
        print("   2. Check that all_chunks_with_embeddings.json exists")
        print("   3. For Qdrant: verify connection settings in .env")
        print("   4. For AI answers: configure API keys in .env")
        print("   5. Install required packages:")
        print("      pip install sentence-transformers google-generativeai anthropic")

def main():
    """Main entry point with multiple modes"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--interactive" or command == "-i":
            interactive_search_session()
            
        elif command == "--test" or command == "-t":
            test_search_functionality()
            
        elif command == "--help" or command == "-h":
            print("ENHANCED HYBRID SEMANTIC SEARCH & ANSWER GENERATION")
            print("=" * 50)
            print("Usage: python search.py [command]")
            print("\nCommands:")
            print("  --interactive, -i     Interactive search session (default)")
            print("  --test, -t           Run functionality test")
            print("  --help, -h           Show this help")
            print("\nDirect question:")
            print("  python search.py \"What is the main topic?\"")
            print("\nSetup Requirements:")
            print("  1. Run the complete RAG pipeline:")
            print("     python src/ocr.py")
            print("     python src/chunking.py") 
            print("     python src/keywords.py")
            print("     python src/embeddings.py")
            print("  2. Install dependencies:")
            print("     pip install sentence-transformers google-generativeai anthropic")
            print("  3. Optional Qdrant setup:")
            print("     Set QDRANT_URL and QDRANT_API_KEY for cloud")
            print("     Or run local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
            print("  4. Configure API keys in .env:")
            print("     GEMINI_API_KEY=your_key")
            print("     OPENAI_API_KEY=your_key")
            print("     ANTHROPIC_API_KEY=your_key")
            print("     GROQ_API_KEY=your_key")
            
        else:
            # Treat as a direct question
            query = " ".join(sys.argv[1:])
            print(f"[QUESTION] {query}")
            
            try:
                search_engine = HybridSemanticSearchEngine(config)
                result = search_engine.search_and_answer(query)
                
                print(f"\n[ANSWER] ({result['confidence']} confidence)")
                print("=" * 50)
                print(result['answer'])
                print("=" * 50)
                
                print(f"\n[SOURCES] ({len(result['sources'])} sources)")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['doc_name']} (Page {source['page']})")
                    
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
    
    else:
        # Default to interactive mode
        interactive_search_session()

if __name__ == "__main__":
    main()