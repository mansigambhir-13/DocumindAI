"""
Concise Theme Identifier - One Comprehensive Answer Per Theme
Groups all related content into distinct themes and provides ONE precise answer for each
No repetitive responses - just clean, comprehensive answers users actually want
Updated with consistent paths matching the search engine structure
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
from collections import Counter, defaultdict

# Add project root to path (consistent with search engine)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_env_file():
    """Load environment variables from .env file (consistent with search engine)"""
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

# Setup logging (consistent with search engine)
def setup_logging() -> logging.Logger:
    """Setup logging for theme identification operations"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "themes.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ConciseThemeIdentifier:
    def __init__(self):
        """
        Concise theme identifier that produces ONE comprehensive answer per theme
        No repetitive or fragmentary responses
        Uses paths consistent with the search engine structure
        """
        
        # Use consistent paths with search engine
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.keywords_dir = self.data_dir / "keywords"  # Updated to match search structure
        self.embeddings_dir = self.data_dir / "embeddings"  # Updated to match search structure
        self.themes_dir = self.data_dir / "themes"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories if they don't exist
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration (support multiple providers like search engine)
        self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        # Default to Groq for themes, but can fallback to others
        self.api_key = self.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        self.rate_limit_delay = 3.0
        
        # Initialize available services
        self.available_services = []
        if self.groq_api_key:
            self.available_services.append("groq")
        if self.gemini_api_key:
            self.available_services.append("gemini")
        if self.openai_api_key:
            self.available_services.append("openai")
        if self.anthropic_api_key:
            self.available_services.append("anthropic")
        
        # Load data using consistent file structure
        self.chunks_data = self._load_chunks()
        self.embeddings_cache = self._load_embeddings()
        
        print(f"✅ Loaded {len(self.chunks_data)} chunks")
        print(f"✅ Available AI services: {', '.join(self.available_services) if self.available_services else 'None (fallback mode)'}")
        if self.embeddings_cache is not None:
            print(f"✅ Loaded embeddings: {self.embeddings_cache.shape}")
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks data from consistent file structure"""
        
        # Try multiple possible locations for chunks data
        possible_files = [
            # Primary location (from embeddings process)
            self.embeddings_dir / "all_chunks_with_embeddings.json",
            
            # Keywords enhanced location
            self.keywords_dir / "all_chunks_with_keywords.json",
            
            # Legacy locations
            self.data_dir / "all_chunks_with_keywords.json",
            self.project_root / "all_chunks_with_keywords.json",
        ]
        
        for chunks_file in possible_files:
            if chunks_file.exists():
                try:
                    logger.info(f"Loading chunks from: {chunks_file}")
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    chunks = data.get('chunks', [])
                    if chunks:
                        print(f"[INFO] Loaded chunks from: {chunks_file}")
                        return chunks
                        
                except Exception as e:
                    logger.warning(f"Error loading chunks from {chunks_file}: {e}")
                    continue
        
        logger.warning("No chunks data found in any expected location")
        return []
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from consistent file structure"""
        
        # Try multiple possible locations for embeddings
        possible_locations = [
            # Primary location (from embeddings process)
            self.embeddings_dir / "all_chunks_with_embeddings.json",
            
            # Numpy cache files
            self.embeddings_dir / "embeddings_cache.npy",
            self.data_dir / "embeddings_cache.npy",
            self.project_root / "embeddings_cache.npy",
        ]
        
        # Try JSON file with embeddings first
        embeddings_json = self.embeddings_dir / "all_chunks_with_embeddings.json"
        if embeddings_json.exists():
            try:
                with open(embeddings_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get('chunks', [])
                if chunks and chunks[0].get('embedding'):
                    # Extract embeddings into numpy array
                    embeddings_list = []
                    for chunk in chunks:
                        embedding = chunk.get('embedding')
                        if embedding and isinstance(embedding, list):
                            embeddings_list.append(embedding)
                    
                    if embeddings_list:
                        embeddings_array = np.array(embeddings_list, dtype=np.float32)
                        logger.info(f"Loaded embeddings from JSON: {embeddings_array.shape}")
                        return embeddings_array
                        
            except Exception as e:
                logger.warning(f"Error loading embeddings from JSON: {e}")
        
        # Try numpy files
        for emb_file in possible_locations:
            if emb_file.suffix == '.npy' and emb_file.exists():
                try:
                    embeddings = np.load(emb_file)
                    if len(embeddings) == len(self.chunks_data):
                        logger.info(f"Loaded embeddings from: {emb_file}")
                        return embeddings
                except Exception as e:
                    logger.warning(f"Error loading embeddings from {emb_file}: {e}")
                    continue
        
        logger.warning("No compatible embeddings found")
        return None
    
    def check_setup(self) -> bool:
        """Check setup with better error messages"""
        issues = []
        
        if not self.available_services:
            issues.append("❌ No AI service API keys configured (GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)")
        
        if not self.chunks_data:
            issues.append("❌ No chunks data found - please run the RAG pipeline first")
            issues.append("   Expected files: data/embeddings/all_chunks_with_embeddings.json")
        
        if issues:
            print("\n".join(issues))
            print("\n[SETUP INSTRUCTIONS]")
            print("1. Run the complete RAG pipeline:")
            print("   python src/ocr.py")
            print("   python src/chunking.py")
            print("   python src/keywords.py") 
            print("   python src/embeddings.py")
            print("2. Configure at least one API key in .env file")
            return False
        
        print(f"✅ Ready: {len(self.chunks_data)} chunks, {len(self.available_services)} AI services")
        return True
    
    def identify_themes_concise(self, query: str, max_themes: int = 5) -> List[Dict]:
        """
        Identify themes and create ONE comprehensive answer per theme
        No redundancy, no repetition - just clean, distinct themes with complete answers
        """
        
        print(f"\n🎯 ANALYZING: '{query}'")
        
        # Step 1: Get all relevant content
        relevant_chunks = self._get_relevant_chunks(query)
        if not relevant_chunks:
            print("❌ No relevant content found")
            return []
        
        print(f"📊 Found {len(relevant_chunks)} relevant chunks")
        
        # Step 2: Group into distinct themes (using best available method)
        themes = self._group_into_themes(relevant_chunks, max_themes)
        if not themes:
            print("❌ No themes could be formed")
            return []
        
        print(f"🧩 Identified {len(themes)} distinct themes")
        
        # Step 3: Create ONE comprehensive answer per theme
        final_themes = []
        for i, theme_chunks in enumerate(themes, 1):
            print(f"   Generating answer for theme {i}...")
            
            comprehensive_theme = self._create_comprehensive_theme(query, theme_chunks, i)
            if comprehensive_theme:
                final_themes.append(comprehensive_theme)
            
            time.sleep(self.rate_limit_delay)
        
        print(f"✅ Generated {len(final_themes)} comprehensive theme answers")
        return final_themes
    
    def _get_relevant_chunks(self, query: str) -> List[Dict]:
        """Get chunks relevant to the query with improved matching"""
        
        query_words = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_keywords = query_words - stop_words
        
        relevant_chunks = []
        
        for chunk in self.chunks_data:
            # Calculate relevance using multiple factors
            chunk_keywords = set(chunk.get('keywords', []))
            text_words = set(chunk.get('text', '').lower().split())
            
            # Keyword matching
            keyword_match = len(query_keywords & chunk_keywords) / len(query_keywords) if query_keywords else 0
            
            # Text content matching
            text_match = len(query_keywords & text_words) / len(query_keywords) if query_keywords else 0
            
            # Document name matching (for doc-specific queries)
            doc_match = 0
            doc_name = chunk.get('doc_name', '').lower()
            if any(word in doc_name for word in query_keywords):
                doc_match = 0.2
            
            # Combined relevance score
            relevance = max(keyword_match, text_match) + doc_match
            
            if relevance >= 0.1:  # Minimum relevance threshold
                chunk['relevance_score'] = relevance
                relevant_chunks.append(chunk)
        
        # Sort by relevance and limit for processing efficiency
        relevant_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return relevant_chunks[:50]  # Increased limit for better coverage
    
    def _group_into_themes(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Group chunks into distinct themes using best available method"""
        
        # Try embeddings-based clustering if available
        if self.embeddings_cache is not None:
            try:
                return self._cluster_with_embeddings(chunks, max_themes)
            except Exception as e:
                logger.warning(f"Embeddings clustering failed: {e}")
        
        # Fallback to keyword-based grouping
        return self._group_by_keywords(chunks, max_themes)
    
    def _cluster_with_embeddings(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Cluster using embeddings with improved chunk matching"""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise Exception("scikit-learn not available for clustering")
        
        # Get embeddings for relevant chunks by matching chunk identifiers
        embeddings_subset = []
        valid_chunks = []
        
        for chunk in chunks:
            # Try different ways to match chunks with their embeddings
            chunk_found = False
            
            for i, orig_chunk in enumerate(self.chunks_data):
                # Match by multiple identifiers
                if (chunk.get('chunk_id') == orig_chunk.get('chunk_id') or
                    (chunk.get('doc_id') == orig_chunk.get('doc_id') and 
                     chunk.get('page') == orig_chunk.get('page') and
                     chunk.get('paragraph_number') == orig_chunk.get('paragraph_number'))):
                    
                    if i < len(self.embeddings_cache):
                        embeddings_subset.append(self.embeddings_cache[i])
                        valid_chunks.append(chunk)
                        chunk_found = True
                        break
            
            if not chunk_found:
                logger.debug(f"No embedding found for chunk: {chunk.get('chunk_id', 'unknown')}")
        
        if len(embeddings_subset) < 4:
            raise Exception(f"Not enough embeddings found ({len(embeddings_subset)})")
        
        # Perform clustering
        n_clusters = min(max_themes, max(2, len(embeddings_subset) // 3))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_subset)
        
        # Group by clusters
        clusters = defaultdict(list)
        for chunk, label in zip(valid_chunks, cluster_labels):
            clusters[label].append(chunk)
        
        # Return clusters with minimum size
        valid_clusters = [cluster for cluster in clusters.values() if len(cluster) >= 2]
        logger.info(f"Created {len(valid_clusters)} clusters from {len(valid_chunks)} chunks")
        return valid_clusters
    
    def _group_by_keywords(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Group chunks by keyword similarity with improved algorithm"""
        
        # Get most common keywords across all chunks
        all_keywords = []
        for chunk in chunks:
            all_keywords.extend(chunk.get('keywords', []))
        
        # Get top keywords but ensure variety
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common(max_themes * 3)]
        
        # Group by dominant keywords
        groups = []
        used_chunks = set()
        
        for keyword in top_keywords:
            if len(groups) >= max_themes:
                break
                
            group = []
            for i, chunk in enumerate(chunks):
                if i not in used_chunks and keyword in chunk.get('keywords', []):
                    group.append(chunk)
                    used_chunks.add(i)
            
            if len(group) >= 2:  # Minimum group size
                groups.append(group)
        
        # Add remaining high-relevance chunks as additional groups
        remaining = [chunk for i, chunk in enumerate(chunks) 
                    if i not in used_chunks and chunk.get('relevance_score', 0) >= 0.3]
        
        if remaining and len(remaining) >= 2:
            # Split remaining into smaller groups if too large
            if len(remaining) > 8:
                mid = len(remaining) // 2
                groups.append(remaining[:mid])
                groups.append(remaining[mid:])
            else:
                groups.append(remaining)
        
        logger.info(f"Created {len(groups)} keyword-based groups")
        return groups
    
    def _create_comprehensive_theme(self, query: str, theme_chunks: List[Dict], theme_id: int) -> Optional[Dict]:
        """
        Create ONE comprehensive theme with a complete answer
        This is the key function that produces clean, non-repetitive responses
        """
        
        if not theme_chunks:
            return None
        
        # Prepare all content from this theme
        all_texts = []
        citations = []
        common_keywords = []
        
        for chunk in theme_chunks:
            text = chunk.get('text', '')
            if text:
                all_texts.append(text)
                common_keywords.extend(chunk.get('keywords', []))
                
                citations.append({
                    "doc_name": chunk.get('doc_name', 'Unknown'),
                    "doc_id": chunk.get('doc_id'),
                    "page": chunk.get('page'),
                    "chunk_id": chunk.get('chunk_id'),
                    "paragraph_number": chunk.get('paragraph_number', chunk.get('para_id')),
                    "relevance_score": round(chunk.get('relevance_score', 0), 3)
                })
        
        # Get top keywords for this theme
        top_keywords = [kw for kw, count in Counter(common_keywords).most_common(5)]
        
        # Combine text content with length limit for API efficiency
        combined_content = "\n\n".join(all_texts[:8])  # Top 8 chunks
        if len(combined_content) > 3000:  # Truncate if too long
            combined_content = combined_content[:3000] + "..."
        
        # Try AI generation with available services
        ai_response = None
        service_used = None
        
        for service in self.available_services:
            try:
                ai_response = self._generate_with_service(query, combined_content, service)
                if ai_response and len(ai_response.strip()) > 50:
                    service_used = service
                    break
            except Exception as e:
                logger.warning(f"Theme generation with {service} failed: {e}")
                continue
        
        # Parse AI response or create fallback
        if ai_response:
            theme_data = self._parse_theme_response(ai_response, citations, theme_chunks, theme_id)
            if theme_data:
                theme_data['service_used'] = service_used
            else:
                theme_data = self._create_fallback_theme(theme_chunks, citations, theme_id)
        else:
            theme_data = self._create_fallback_theme(theme_chunks, citations, theme_id)
        
        if theme_data:
            # Add metadata
            theme_data['citations'] = citations
            theme_data['cluster_size'] = len(theme_chunks)
            theme_data['top_keywords'] = top_keywords
            theme_data['method'] = 'comprehensive_synthesis'
            return theme_data
        
        return None
    
    def _generate_with_service(self, query: str, content: str, service: str) -> Optional[str]:
        """Generate theme using specified AI service"""
        
        # Create comprehensive analysis prompt
        prompt = f"""You are analyzing documents to answer the query: "{query}"

THEME CONTENT (related passages about one specific aspect):
{content}

TASK: Create ONE comprehensive, definitive answer about this specific theme.

Requirements:
1. THEME_TITLE: Clear, specific title (4-6 words) that captures this distinct aspect
2. COMPREHENSIVE_ANSWER: One complete, detailed response that:
   - Directly addresses this aspect of the user's query
   - Synthesizes ALL the information from the passages above
   - Provides specific details, examples, and insights
   - Is complete enough that the user needs no additional explanation for this theme
   - Flows naturally like an expert explanation
   - Is 150-250 words

3. Keep it focused on ONE distinct theme - don't try to cover everything

JSON format:
{{
    "theme_title": "Specific Theme Title",
    "comprehensive_answer": "One complete, detailed explanation that fully covers this theme with specific details from the documents. This should be a thorough, standalone answer that synthesizes all the relevant information about this particular aspect of the query."
}}"""
        
        if service == "groq":
            return self._call_groq_api(prompt)
        elif service == "gemini":
            return self._call_gemini_api(prompt)
        elif service == "openai":
            return self._call_openai_api(prompt)
        elif service == "anthropic":
            return self._call_anthropic_api(prompt)
        
        return None
    
    def _call_groq_api(self, prompt: str) -> Optional[str]:
        """Call Groq API"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:
                print("   ⏳ Rate limit - waiting...")
                time.sleep(10)
                return self._call_groq_api(prompt)
            else:
                logger.warning(f"Groq API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
            return None
    
    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
            return None
    
    def _call_openai_api(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return None
    
    def _call_anthropic_api(self, prompt: str) -> Optional[str]:
        """Call Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.warning(f"Anthropic API call failed: {e}")
            return None
    
    def _parse_theme_response(self, response: str, citations: List[Dict], 
                             chunks: List[Dict], theme_id: int) -> Optional[Dict]:
        """Parse LLM response into theme structure"""
        
        try:
            # Try JSON parsing
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                
                if 'theme_title' in data and 'comprehensive_answer' in data:
                    return {
                        'theme_title': data['theme_title'],
                        'comprehensive_answer': data['comprehensive_answer'],
                        'confidence': 'high',
                        'answer_length': len(data['comprehensive_answer'].split())
                    }
            
            # Manual parsing fallback
            lines = response.split('\n')
            theme_title = f"Theme {theme_id}"
            comprehensive_answer = "Analysis of related document content."
            
            for line in lines:
                line = line.strip()
                if 'title' in line.lower() and ':' in line:
                    theme_title = line.split(':', 1)[1].strip().strip('"').strip("'")
                elif 'answer' in line.lower() and ':' in line:
                    comprehensive_answer = line.split(':', 1)[1].strip().strip('"').strip("'")
                elif len(line) > 100:  # Likely the main answer
                    comprehensive_answer = line
            
            return {
                'theme_title': theme_title,
                'comprehensive_answer': comprehensive_answer,
                'confidence': 'medium',
                'answer_length': len(comprehensive_answer.split())
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return None
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return None
    
    def _create_fallback_theme(self, chunks: List[Dict], citations: List[Dict], theme_id: int) -> Dict:
        """Create fallback theme when AI is unavailable"""
        
        # Extract key information
        documents = set(chunk.get('doc_name', 'Unknown') for chunk in chunks)
        keywords = []
        for chunk in chunks:
            keywords.extend(chunk.get('keywords', []))
        
        top_keywords = [kw for kw, count in Counter(keywords).most_common(3)]
        
        # Create basic comprehensive answer from actual content
        key_sentences = []
        for chunk in chunks[:3]:
            text = chunk.get('text', '')
            if text:
                # Extract first meaningful sentence
                sentences = text.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 30:
                        key_sentences.append(sentence.strip())
                        break
        
        comprehensive_answer = f"This theme covers {', '.join(top_keywords)} based on analysis of {len(chunks)} document segments. "
        
        if key_sentences:
            comprehensive_answer += "Key findings include: " + ". ".join(key_sentences[:2]) + ". "
        
        comprehensive_answer += f"The information is sourced from {len(documents)} document(s): {', '.join(list(documents)[:3])}."
        
        if len(documents) > 3:
            comprehensive_answer += f" Plus {len(documents) - 3} additional sources."
        
        return {
            'theme_title': f"Analysis: {', '.join(top_keywords[:2])}",
            'comprehensive_answer': comprehensive_answer,
            'confidence': 'medium',
            'answer_length': len(comprehensive_answer.split()),
            'service_used': 'fallback'
        }
    
    def display_concise_results(self, themes: List[Dict], query: str):
        """Display clean, concise results - one answer per theme"""
        
        if not themes:
            print("❌ No themes identified")
            return
        
        print(f"\n🎯 ANALYSIS RESULTS FOR: '{query}'")
        print(f"📊 {len(themes)} distinct themes identified")
        print("=" * 80)
        
        for i, theme in enumerate(themes, 1):
            theme_title = theme.get('theme_title', f'Theme {i}')
            comprehensive_answer = theme.get('comprehensive_answer', 'No answer available')
            answer_length = theme.get('answer_length', 0)
            citations_count = len(theme.get('citations', []))
            service_used = theme.get('service_used', 'unknown')
            
            print(f"\n🧩 THEME {i}: {theme_title}")
            print(f"\n💬 ANSWER:")
            print(f"{comprehensive_answer}")
            
            print(f"\n📊 Details: {answer_length} words | {citations_count} sources | {service_used}")
            
            # Show top sources
            citations = theme.get('citations', [])
            if citations:
                top_sources = citations[:3]
                sources_text = ", ".join([f"{c.get('doc_name', 'Unknown')}" for c in top_sources])
                print(f"📄 Key Sources: {sources_text}")
                if len(citations) > 3:
                    print(f"   + {len(citations) - 3} more sources")
            
            print("-" * 60)
        
        # Summary
        total_words = sum(theme.get('answer_length', 0) for theme in themes)
        total_sources = sum(len(theme.get('citations', [])) for theme in themes)
        
        print(f"\n📋 SUMMARY:")
        print(f"   🎯 {len(themes)} comprehensive answers")
        print(f"   📝 {total_words:,} total words")
        print(f"   📚 {total_sources} total sources")
        print(f"   ✅ Complete coverage of query aspects")
    
    def process_query_concise(self, query: str, max_themes: int = 5, save_results: bool = True) -> List[Dict]:
        """Process query and return concise, comprehensive themes"""
        
        start_time = time.time()
        
        # Identify themes
        themes = self.identify_themes_concise(query, max_themes)
        
        # Save results if requested
        if save_results and themes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            
            result_data = {
                "query": query,
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "method": "concise_comprehensive_themes",
                "total_themes": len(themes),
                "available_services": self.available_services,
                "themes": themes,
                "summary": {
                    "total_words": sum(theme.get('answer_length', 0) for theme in themes),
                    "total_sources": sum(len(theme.get('citations', [])) for theme in themes),
                    "avg_answer_length": round(sum(theme.get('answer_length', 0) for theme in themes) / len(themes), 1) if themes else 0,
                    "services_used": list(set(theme.get('service_used', 'unknown') for theme in themes))
                }
            }
            
            output_file = self.themes_dir / f"concise_analysis_{query_safe}_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved to: {output_file}")
        
        return themes

def test_theme_functionality():
    """Test the theme identification functionality"""
    print("🧪 TESTING THEME IDENTIFICATION FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Initialize theme identifier
        print("[INFO] Initializing theme identifier...")
        identifier = ConciseThemeIdentifier()
        
        if not identifier.check_setup():
            return False
        
        print("[OK] Theme identifier initialized successfully!")
        
        # Display configuration
        print(f"\n[CONFIG] Theme Identifier Status:")
        print(f"   Data directory: {identifier.data_dir}")
        print(f"   Chunks loaded: {len(identifier.chunks_data):,}")
        print(f"   Embeddings available: {'Yes' if identifier.embeddings_cache is not None else 'No'}")
        print(f"   Available AI services: {identifier.available_services}")
        
        # Test with a sample query
        test_query = "What are the main regulatory requirements for algorithmic trading?"
        print(f"\n[TEST] Testing with query: '{test_query}'")
        
        themes = identifier.process_query_concise(test_query, max_themes=3, save_results=False)
        
        if themes:
            print(f"\n[SUCCESS] Generated {len(themes)} themes")
            identifier.display_concise_results(themes, test_query)
            return True
        else:
            print("[WARNING] No themes generated - check data and API configuration")
            return False
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        logger.error(f"Theme test error: {e}")
        return False

def interactive_theme_session():
    """Enhanced interactive theme identification session"""
    print("🎯 CONCISE THEME IDENTIFIER")
    print("📝 One Comprehensive Answer Per Theme")
    print("🚫 No Repetition, No Fragments")
    print("=" * 50)
    
    # Initialize
    identifier = ConciseThemeIdentifier()
    
    if not identifier.check_setup():
        print("\n[TROUBLESHOOTING TIPS]")
        print("1. Run the complete RAG pipeline:")
        print("   python src/ocr.py")
        print("   python src/chunking.py")
        print("   python src/keywords.py")
        print("   python src/embeddings.py")
        print("2. Configure at least one API key in .env:")
        print("   GROQ_API_KEY=your_key")
        print("   GEMINI_API_KEY=your_key")
        print("   OPENAI_API_KEY=your_key")
        print("   ANTHROPIC_API_KEY=your_key")
        return
    
    print(f"\n✨ WHAT YOU GET:")
    print(f"   🎯 Clear, distinct themes")
    print(f"   💬 One comprehensive answer per theme")
    print(f"   📚 Complete source citations")
    print(f"   🚫 No repetitive content")
    print(f"   🤖 AI-powered analysis using: {', '.join(identifier.available_services)}")
    
    # Get document overview
    if identifier.chunks_data:
        doc_names = list(set(chunk.get('doc_name', 'Unknown') for chunk in identifier.chunks_data))
        print(f"\n📁 Available Documents ({len(doc_names)}):")
        for i, doc_name in enumerate(doc_names[:10], 1):
            print(f"   {i}. {doc_name}")
        if len(doc_names) > 10:
            print(f"   ... and {len(doc_names) - 10} more")
    
    print(f"\n" + "=" * 50)
    print("Commands:")
    print("  - Ask any question to get themed analysis")
    print("  - 'test': Run functionality test")
    print("  - 'docs': List all available documents")
    print("  - 'stats': Show system statistics")
    print("  - 'help': Show this help")
    print("  - 'quit': Exit")
    print("=" * 50)
    
    session_count = 0
    
    # Interactive loop
    while True:
        try:
            query = input(f"\n[QUERY] ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n[HELP] Available commands:")
                print("  - Ask any question about your documents")
                print("  - 'test': Run functionality test")
                print("  - 'docs': List all available documents")
                print("  - 'stats': Show system statistics")
                print("  - 'quit': Exit the program")
                continue
            
            if query.lower() == 'test':
                test_theme_functionality()
                continue
            
            if query.lower() == 'docs':
                if identifier.chunks_data:
                    doc_names = list(set(chunk.get('doc_name', 'Unknown') for chunk in identifier.chunks_data))
                    print(f"\n[DOCS] Available documents ({len(doc_names)}):")
                    for i, doc_name in enumerate(doc_names, 1):
                        print(f"   {i}. {doc_name}")
                else:
                    print("\n[INFO] No documents loaded")
                continue
            
            if query.lower() == 'stats':
                print(f"\n[STATS] System Statistics:")
                print(f"   Session analyses: {session_count}")
                print(f"   Total chunks: {len(identifier.chunks_data):,}")
                print(f"   Embeddings available: {'Yes' if identifier.embeddings_cache is not None else 'No'}")
                print(f"   AI services: {len(identifier.available_services)}")
                print(f"   Data directory: {identifier.data_dir}")
                continue
            
            session_count += 1
            print(f"\n[PROCESSING] Analysis {session_count}...")
            
            try:
                start_time = time.time()
                
                # Process query with theme identification
                themes = identifier.process_query_concise(query, max_themes=5)
                
                end_time = time.time()
                
                # Display results
                identifier.display_concise_results(themes, query)
                
                print(f"\n⏱️  Completed in {end_time - start_time:.1f} seconds")
                print("=" * 50)
                
            except Exception as e:
                print(f"❌ Analysis Error: {e}")
                logger.error(f"Query processing error: {e}")
                continue
            
        except KeyboardInterrupt:
            print("\n[INFO] Analysis interrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"❌ Session Error: {e}")
            logger.error(f"Session error: {e}")
            continue

def main():
    """Main function for concise theme identification"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--interactive" or command == "-i":
            interactive_theme_session()
            
        elif command == "--test" or command == "-t":
            test_theme_functionality()
            
        elif command == "--help" or command == "-h":
            print("CONCISE THEME IDENTIFIER")
            print("=" * 30)
            print("Usage: python themes.py [command]")
            print("\nCommands:")
            print("  --interactive, -i     Interactive theme session (default)")
            print("  --test, -t           Run functionality test")
            print("  --help, -h           Show this help")
            print("\nDirect question:")
            print("  python themes.py \"What are the main topics?\"")
            print("\nSetup Requirements:")
            print("  1. Run the complete RAG pipeline:")
            print("     python src/ocr.py")
            print("     python src/chunking.py") 
            print("     python src/keywords.py")
            print("     python src/embeddings.py")
            print("  2. Install dependencies:")
            print("     pip install scikit-learn google-generativeai anthropic")
            print("  3. Configure API keys in .env:")
            print("     GROQ_API_KEY=your_key")
            print("     GEMINI_API_KEY=your_key")
            print("     OPENAI_API_KEY=your_key")
            print("     ANTHROPIC_API_KEY=your_key")
            
        else:
            # Treat as a direct question
            query = " ".join(sys.argv[1:])
            print(f"[QUESTION] {query}")
            
            try:
                identifier = ConciseThemeIdentifier()
                if identifier.check_setup():
                    themes = identifier.process_query_concise(query)
                    identifier.display_concise_results(themes, query)
                else:
                    print("[ERROR] Setup incomplete - see --help for instructions")
                    
            except Exception as e:
                print(f"[ERROR] Theme analysis failed: {e}")
    
    else:
        # Default to interactive mode
        interactive_theme_session()

if __name__ == "__main__":
    main()