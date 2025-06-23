"""
Keywords Extraction Script
Extracts relevant keywords from chunked text using multiple strategies
Aligns with RAG application pipeline structure
"""

import json
import os
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import settings (with fallback if not available)
try:
    from config.settings import settings
    # Add API keys if not present in settings
    if not hasattr(settings, 'GROQ_API_KEY'):
        settings.GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    if not hasattr(settings, 'OPENAI_API_KEY'):
        settings.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
except ImportError:
    # Fallback configuration if settings not available
    class Settings:
        CHUNKS_DIR = str(project_root / "data" / "chunks")
        KEYWORDS_DIR = str(project_root / "data" / "keywords")
        DEBUG = True
        MAX_KEYWORDS = 20
        KEYWORD_MIN_SCORE = 0.3
        MAX_RETRIES = 3
        
        # API Keys (optional - will try local extraction first)
        GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        
        def validate_config(self):
            return True
        
        def create_directories(self):
            Path(self.KEYWORDS_DIR).mkdir(parents=True, exist_ok=True)
    
    settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self):
        """Initialize keyword extractor with multiple strategies"""
        # Fix paths to match project structure
        self.input_dir = Path(settings.CHUNKS_DIR)
        if not self.input_dir.is_absolute():
            self.input_dir = project_root / self.input_dir
            
        self.output_dir = Path(settings.KEYWORDS_DIR)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.max_keywords = settings.MAX_KEYWORDS
        self.min_score = settings.KEYWORD_MIN_SCORE
        
        # Initialize extraction methods
        self.use_api = bool(getattr(settings, 'GROQ_API_KEY', '') or getattr(settings, 'OPENAI_API_KEY', ''))
        
        logger.info(f"KeywordExtractor initialized")
        logger.info(f"Input Directory (chunks): {self.input_dir}")
        logger.info(f"Output Directory (keywords): {self.output_dir}")
        logger.info(f"API extraction available: {self.use_api}")
        
        # Load stop words
        self.stop_words = self._load_stop_words()
        
        # Common business/finance terms that should be kept
        self.important_terms = {
            'sebi', 'securities', 'exchange', 'board', 'regulatory', 'compliance',
            'investment', 'fund', 'mutual', 'equity', 'debt', 'portfolio',
            'returns', 'risk', 'financial', 'market', 'trading', 'investor',
            'shares', 'stocks', 'bonds', 'dividend', 'capital', 'revenue',
            'profit', 'loss', 'asset', 'liability', 'turnover', 'growth',
            'company', 'corporation', 'limited', 'private', 'public',
            'business', 'industry', 'sector', 'economy', 'economic'
        }
    
    def _load_stop_words(self) -> set:
        """Load comprehensive stop words list"""
        # Basic English stop words
        basic_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your',
            'this', 'these', 'they', 'them', 'their', 'there', 'then',
            'than', 'or', 'but', 'not', 'no', 'can', 'could', 'should',
            'may', 'might', 'must', 'shall', 'have', 'had', 'do', 'does',
            'did', 'get', 'got', 'go', 'going', 'come', 'came', 'said',
            'say', 'see', 'seen', 'make', 'made', 'take', 'taken', 'give',
            'given', 'know', 'known', 'think', 'thought', 'find', 'found',
            'tell', 'told', 'ask', 'asked', 'try', 'tried', 'need', 'needed',
            'want', 'wanted', 'use', 'used', 'work', 'worked', 'call', 'called'
        }
        
        # Add common document words
        document_words = {
            'page', 'pages', 'document', 'documents', 'file', 'files',
            'section', 'sections', 'chapter', 'chapters', 'part', 'parts',
            'paragraph', 'paragraphs', 'line', 'lines', 'text', 'content',
            'information', 'data', 'details', 'description', 'note', 'notes',
            'example', 'examples', 'following', 'above', 'below', 'next',
            'previous', 'first', 'last', 'number', 'numbers', 'date', 'dates',
            'time', 'times', 'year', 'years', 'month', 'months', 'day', 'days'
        }
        
        return basic_stop_words | document_words
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for keyword extraction"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep hyphens in words
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords_statistical(self, text: str) -> List[str]:
        """Extract keywords using statistical methods"""
        if not text or len(text.strip()) < 20:
            return []
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Split into words
        words = cleaned_text.split()
        
        # Filter words
        filtered_words = []
        for word in words:
            # Skip short words, numbers, and stop words
            if (len(word) >= 3 and 
                not word.isdigit() and 
                word not in self.stop_words and
                not word.startswith('-') and
                not word.endswith('-')):
                filtered_words.append(word)
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Extract phrases (2-3 words)
        phrases = []
        for i in range(len(words) - 1):
            if i < len(words) - 2:  # 3-word phrases
                phrase = ' '.join(words[i:i+3])
                if self._is_valid_phrase(phrase):
                    phrases.append(phrase)
            
            # 2-word phrases
            phrase = ' '.join(words[i:i+2])
            if self._is_valid_phrase(phrase):
                phrases.append(phrase)
        
        phrase_counts = Counter(phrases)
        
        # Combine and score keywords
        keywords = []
        
        # Add single words (weighted by frequency and importance)
        for word, count in word_counts.most_common(20):
            score = count
            if word in self.important_terms:
                score *= 2  # Boost important business terms
            keywords.append((word, score))
        
        # Add phrases (weighted higher than single words)
        for phrase, count in phrase_counts.most_common(10):
            score = count * 1.5  # Phrases get higher weight
            keywords.append((phrase, score))
        
        # Sort by score and return top keywords
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, score in keywords[:self.max_keywords]]
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if phrase is valid for keyword extraction"""
        words = phrase.split()
        
        # Skip if too many stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count > len(words) // 2:
            return False
        
        # Skip if all words are too short
        if all(len(word) < 3 for word in words):
            return False
        
        # Skip if contains only numbers
        if all(word.isdigit() for word in words):
            return False
        
        return True
    
    def extract_keywords_pattern_based(self, text: str) -> List[str]:
        """Extract keywords using pattern recognition"""
        keywords = []
        
        # Financial amounts
        financial_patterns = [
            r'rs\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh|thousand|million|billion))?',
            r'₹\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh|thousand|million|billion))?',
            r'\d+(?:,\d+)*(?:\.\d+)?\s*(?:crore|lakh|thousand|million|billion)',
            r'\d+(?:\.\d+)?\s*%'
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keywords.append(match.strip())
        
        # Company names (capitalized words)
        company_pattern = r'\b[A-Z][a-z]+\s+(?:Ltd|Limited|Inc|Corporation|Corp|Private|Pvt)\b'
        company_matches = re.findall(company_pattern, text)
        keywords.extend(company_matches)
        
        # Technical terms (words ending in specific suffixes)
        technical_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'acy', 'ence', 'ance']
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if (len(word) > 6 and 
                any(word.endswith(suffix) for suffix in technical_suffixes) and
                word not in self.stop_words):
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def extract_keywords_api(self, text: str) -> List[str]:
        """Extract keywords using API (if available)"""
        if not self.use_api or len(text.strip()) < 20:
            return []
        
        try:
            if getattr(settings, 'GROQ_API_KEY', ''):
                return self._extract_with_groq(text)
            elif getattr(settings, 'OPENAI_API_KEY', ''):
                return self._extract_with_openai(text)
        except Exception as e:
            logger.warning(f"API extraction failed: {e}")
        
        return []
    
    def _extract_with_groq(self, text: str) -> List[str]:
        """Extract keywords using Groq API"""
        try:
            import requests
            
            prompt = f"""Extract 5-8 relevant keywords from this text. Focus on:
- Key business terms
- Important entities (companies, people, places)
- Financial concepts
- Technical terminology

Text: {text[:1500]}

Return only a JSON array of keywords:
["keyword1", "keyword2", "keyword3"]"""

            headers = {
                "Authorization": f"Bearer {getattr(settings, 'GROQ_API_KEY', '')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 100
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return self._parse_api_response(content)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
        
        return []
    
    def _extract_with_openai(self, text: str) -> List[str]:
        """Extract keywords using OpenAI API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=getattr(settings, 'OPENAI_API_KEY', ''))
            
            prompt = f"""Extract 5-8 relevant keywords from this text. Focus on:
- Key business terms
- Important entities (companies, people, places)
- Financial concepts
- Technical terminology

Text: {text[:1500]}

Return only a JSON array of keywords:
["keyword1", "keyword2", "keyword3"]"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_api_response(content)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
        
        return []
    
    def _parse_api_response(self, response: str) -> List[str]:
        """Parse API response to extract keywords"""
        try:
            # Try to find JSON array in response
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                keywords = json.loads(json_str)
                
                # Clean and validate keywords
                cleaned_keywords = []
                for kw in keywords:
                    if isinstance(kw, str) and len(kw.strip()) > 1:
                        cleaned = kw.strip().lower()
                        if cleaned not in self.stop_words:
                            cleaned_keywords.append(cleaned)
                
                return cleaned_keywords[:8]  # Limit to 8 keywords
                
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract from comma-separated or line-separated text
        keywords = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Split by commas
                parts = line.split(',')
                for part in parts:
                    cleaned = re.sub(r'[^\w\s]', '', part).strip().lower()
                    if cleaned and len(cleaned) > 1 and cleaned not in self.stop_words:
                        keywords.append(cleaned)
        
        return keywords[:8]
    
    def extract_keywords_combined(self, text: str, doc_context: Dict[str, Any] = None) -> List[str]:
        """Combine multiple extraction methods for best results"""
        all_keywords = []
        
        # Method 1: Statistical extraction (always available)
        statistical_keywords = self.extract_keywords_statistical(text)
        all_keywords.extend(statistical_keywords)
        
        # Method 2: Pattern-based extraction
        pattern_keywords = self.extract_keywords_pattern_based(text)
        all_keywords.extend(pattern_keywords)
        
        # Method 3: API extraction (if available)
        if self.use_api:
            api_keywords = self.extract_keywords_api(text)
            all_keywords.extend(api_keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            kw_clean = kw.lower().strip()
            if kw_clean not in seen and len(kw_clean) > 1:
                seen.add(kw_clean)
                unique_keywords.append(kw_clean)
        
        # Limit to max keywords
        return unique_keywords[:self.max_keywords]
    
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chunk to extract keywords"""
        try:
            text = chunk.get('text', '')
            
            if not text or len(text.strip()) < 20:
                # Keep chunk but with empty keywords
                enhanced_chunk = chunk.copy()
                enhanced_chunk['keywords'] = []
                enhanced_chunk['keyword_count'] = 0
                enhanced_chunk['keywords_extracted_at'] = datetime.now().isoformat()
                return enhanced_chunk
            
            # Create context for better extraction
            doc_context = {
                'doc_id': chunk.get('doc_id', ''),
                'doc_name': chunk.get('doc_name', ''),
                'page': chunk.get('page', 1)
            }
            
            # Extract keywords
            keywords = self.extract_keywords_combined(text, doc_context)
            
            # Update chunk with keywords
            enhanced_chunk = chunk.copy()
            enhanced_chunk['keywords'] = keywords
            enhanced_chunk['keyword_count'] = len(keywords)
            enhanced_chunk['keywords_extracted_at'] = datetime.now().isoformat()
            enhanced_chunk['extraction_method'] = 'combined'
            
            return enhanced_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Return chunk with empty keywords on error
            enhanced_chunk = chunk.copy()
            enhanced_chunk['keywords'] = []
            enhanced_chunk['keyword_count'] = 0
            enhanced_chunk['keywords_extracted_at'] = datetime.now().isoformat()
            enhanced_chunk['extraction_error'] = str(e)
            return enhanced_chunk
    
    def process_all_chunks(self) -> bool:
        """Process all chunks to extract keywords"""
        
        # Load chunks from chunking step
        chunks_file = self.input_dir / "all_chunks.json"
        
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            print(f"❌ Chunks file not found: {chunks_file}")
            print(f"💡 Please run chunking.py first to create chunks!")
            return False
        
        try:
            logger.info(f"Loading chunks from: {chunks_file}")
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                print("❌ No chunks found in input file")
                return False
            
            print(f"📁 Found {total_chunks} chunks to process")
            print(f"🔍 Extraction methods available:")
            print(f"   ✅ Statistical keyword extraction")
            print(f"   ✅ Pattern-based extraction")
            print(f"   {'✅' if self.use_api else '❌'} API-based extraction")
            
            # Process chunks
            start_time = time.time()
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks, 1):
                enhanced_chunk = self.process_chunk(chunk)
                enhanced_chunks.append(enhanced_chunk)
                
                # Progress reporting
                if i % 50 == 0 or i == total_chunks:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total_chunks - i) / rate if rate > 0 else 0
                    print(f"📊 Progress: {i}/{total_chunks} ({i/total_chunks*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update data structure
            enhanced_data = data.copy()
            enhanced_data['chunks'] = enhanced_chunks
            enhanced_data['metadata'].update({
                'keywords_extracted': True,
                'keyword_extraction_completed_at': datetime.now().isoformat(),
                'extraction_methods_used': ['statistical', 'pattern_based'] + (['api'] if self.use_api else []),
                'processing_time_seconds': processing_time,
                'chunks_per_second': total_chunks / processing_time if processing_time > 0 else 0
            })
            
            # Save results
            output_file = self.output_dir / "all_chunks_with_keywords.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            # Calculate statistics
            chunks_with_keywords = sum(1 for chunk in enhanced_chunks if chunk.get('keywords'))
            total_keywords = sum(len(chunk.get('keywords', [])) for chunk in enhanced_chunks)
            avg_keywords_per_chunk = total_keywords / total_chunks if total_chunks > 0 else 0
            
            # Save detailed statistics
            self._save_keyword_statistics(enhanced_chunks, enhanced_data['metadata'])
            
            print(f"\n✅ Keyword extraction completed!")
            print(f"⏱️  Processing time: {processing_time/60:.1f} minutes")
            print(f"📊 Total chunks: {total_chunks:,}")
            print(f"🔑 Chunks with keywords: {chunks_with_keywords:,}")
            print(f"📈 Success rate: {chunks_with_keywords/total_chunks*100:.1f}%")
            print(f"🎯 Average keywords per chunk: {avg_keywords_per_chunk:.1f}")
            print(f"📝 Total keywords extracted: {total_keywords:,}")
            print(f"🚀 Processing rate: {total_chunks/(processing_time/60):.1f} chunks/minute")
            print(f"📁 Results saved to: {output_file}")
            
            # Show sample results
            self._show_sample_results(enhanced_chunks)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            print(f"❌ Error processing chunks: {e}")
            return False
    
    def _save_keyword_statistics(self, enhanced_chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Save detailed keyword extraction statistics"""
        
        # Calculate statistics
        total_chunks = len(enhanced_chunks)
        chunks_with_keywords = sum(1 for chunk in enhanced_chunks if chunk.get('keywords'))
        
        # Keyword frequency analysis
        keyword_counts = {}
        for chunk in enhanced_chunks:
            for keyword in chunk.get('keywords', []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Keyword count distribution
        keyword_count_dist = {}
        for chunk in enhanced_chunks:
            count = len(chunk.get('keywords', []))
            keyword_count_dist[count] = keyword_count_dist.get(count, 0) + 1
        
        # Text length vs keyword count analysis
        length_keyword_correlation = []
        for chunk in enhanced_chunks:
            text_length = len(chunk.get('text', ''))
            keyword_count = len(chunk.get('keywords', []))
            length_keyword_correlation.append({'text_length': text_length, 'keyword_count': keyword_count})
        
        # Compile statistics
        stats = {
            "extraction_summary": {
                "total_chunks": total_chunks,
                "chunks_with_keywords": chunks_with_keywords,
                "success_rate": round(chunks_with_keywords/total_chunks*100, 2) if total_chunks > 0 else 0,
                "total_unique_keywords": len(keyword_counts),
                "total_keywords_extracted": sum(keyword_counts.values()),
                "average_keywords_per_chunk": round(sum(keyword_counts.values())/total_chunks, 2) if total_chunks > 0 else 0,
                "processing_time_minutes": metadata.get('processing_time_seconds', 0) / 60
            },
            "keyword_analysis": {
                "top_keywords": [{"keyword": kw, "frequency": count} for kw, count in top_keywords],
                "keyword_count_distribution": keyword_count_dist,
                "extraction_methods_used": metadata.get('extraction_methods_used', [])
            },
            "quality_metrics": {
                "chunks_with_no_keywords": total_chunks - chunks_with_keywords,
                "chunks_with_1_to_5_keywords": sum(1 for chunk in enhanced_chunks if 1 <= len(chunk.get('keywords', [])) <= 5),
                "chunks_with_more_than_5_keywords": sum(1 for chunk in enhanced_chunks if len(chunk.get('keywords', [])) > 5),
                "average_keyword_length": round(sum(len(kw) for kw in keyword_counts.keys()) / len(keyword_counts), 2) if keyword_counts else 0
            },
            "configuration": {
                "max_keywords_per_chunk": self.max_keywords,
                "min_score_threshold": self.min_score,
                "api_extraction_enabled": self.use_api,
                "stop_words_count": len(self.stop_words),
                "important_terms_count": len(self.important_terms)
            },
            "processed_at": datetime.now().isoformat()
        }
        
        # Save statistics
        stats_file = self.output_dir / "keyword_extraction_summary.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Statistics saved to: {stats_file}")
    
    def _show_sample_results(self, enhanced_chunks: List[Dict[str, Any]]):
        """Show sample keyword extraction results"""
        print(f"\n🔍 Sample Keyword Extraction Results:")
        
        sample_count = 0
        for chunk in enhanced_chunks:
            keywords = chunk.get('keywords', [])
            if keywords and sample_count < 3:
                text_preview = chunk.get('text', '')[:150] + "..." if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
                print(f"\n   📄 Sample {sample_count + 1}:")
                print(f"   Document: {chunk.get('doc_name', 'Unknown')}")
                print(f"   Page: {chunk.get('page', 'N/A')}")
                print(f"   Keywords: {keywords}")
                print(f"   Text: {text_preview}")
                sample_count += 1
        
        if sample_count == 0:
            print("   ⚠️  No keywords found in sample chunks")

def main():
    """Main function to run keyword extraction"""
    print(f"🔍 Keyword Extraction for RAG Pipeline")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Input from chunks: {project_root / 'data' / 'chunks'}")
    print(f"📁 Output to keywords: {project_root / 'data' / 'keywords'}")
    print(f"=" * 60)
    
    # Initialize extractor
    extractor = KeywordExtractor()
    
    # Check if input exists
    if not extractor.input_dir.exists():
        print("❌ Chunks directory not found")
        print(f"💡 Please run chunking.py first to create chunks!")
        return
    
    chunks_file = extractor.input_dir / "all_chunks.json"
    if not chunks_file.exists():
        print("❌ Chunks file not found")
        print(f"💡 Please run chunking.py first to create all_chunks.json!")
        return
    
    # Process chunks
    success = extractor.process_all_chunks()
    
    if success:
        print(f"\n🎉 Keyword extraction completed successfully!")
        print(f"🔄 Ready for next pipeline step: embeddings.py")
    else:
        print(f"\n❌ Keyword extraction failed")

if __name__ == "__main__":
    main()