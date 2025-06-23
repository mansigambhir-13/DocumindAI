#!/usr/bin/env python3
"""
Quick Status Check for RAG Application
Checks the current state of your pipeline
"""

import os
import json
from pathlib import Path

def check_env_file():
    """Check .env configuration"""
    print("🔍 CHECKING .env FILE...")
    print("=" * 40)
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read and check API keys
    api_keys_found = []
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if 'API_KEY' in key and value and len(value) > 10:
                    api_keys_found.append(key)
    
    if api_keys_found:
        print(f"✅ API keys configured: {', '.join(api_keys_found)}")
        return True
    else:
        print("❌ No valid API keys found in .env")
        print("🔧 Add at least one API key:")
        print("   GEMINI_API_KEY=your_key_here")
        return False

def check_documents():
    """Check if documents exist"""
    print("\n📄 CHECKING DOCUMENTS...")
    print("=" * 40)
    
    # Check raw documents
    raw_dir = Path('data/raw')
    if raw_dir.exists():
        docs = list(raw_dir.glob('*.*'))
        if docs:
            print(f"✅ Found {len(docs)} document(s) in data/raw/:")
            for doc in docs[:5]:
                size_mb = doc.stat().st_size / (1024*1024)
                print(f"   - {doc.name} ({size_mb:.1f}MB)")
            if len(docs) > 5:
                print(f"   ... and {len(docs) - 5} more")
            return True
        else:
            print("❌ No documents found in data/raw/")
    else:
        print("❌ data/raw/ directory not found")
    
    return False

def check_pipeline_status():
    """Check pipeline processing status"""
    print("\n🔄 CHECKING PIPELINE STATUS...")
    print("=" * 40)
    
    pipeline_files = {
        'data/processed/': 'OCR Processing',
        'data/chunks/all_chunks.json': 'Text Chunking', 
        'data/keywords/': 'Keyword Extraction',
        'data/embeddings/all_chunks_with_embeddings.json': 'Embeddings Generation'
    }
    
    status = {}
    
    for file_path, step_name in pipeline_files.items():
        path = Path(file_path)
        
        if path.is_dir():
            # Check if directory has files
            files = list(path.glob('*.*'))
            if files:
                print(f"✅ {step_name}: {len(files)} files")
                status[step_name] = True
            else:
                print(f"❌ {step_name}: No output files")
                status[step_name] = False
        else:
            # Check if file exists
            if path.exists():
                size_mb = path.stat().st_size / (1024*1024)
                print(f"✅ {step_name}: Complete ({size_mb:.1f}MB)")
                
                # Validate JSON structure
                if path.suffix == '.json':
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        if 'chunks' in data:
                            print(f"   📊 {len(data['chunks'])} chunks available")
                    except:
                        print("   ⚠️ JSON file may be corrupted")
                        
                status[step_name] = True
            else:
                print(f"❌ {step_name}: Not completed")
                status[step_name] = False
    
    return status

def check_search_readiness():
    """Check if search is ready to work"""
    print("\n🔍 CHECKING SEARCH READINESS...")
    print("=" * 40)
    
    # Check embeddings file specifically
    embeddings_file = Path('data/embeddings/all_chunks_with_embeddings.json')
    
    if not embeddings_file.exists():
        print("❌ Embeddings file missing")
        print("🔧 Run: python src/embeddings.py")
        return False
    
    try:
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        if not chunks:
            print("❌ No chunks in embeddings file")
            return False
        
        # Check if chunks have embeddings
        valid_chunks = 0
        for chunk in chunks[:5]:  # Check first 5
            if chunk.get('embedding') and len(chunk['embedding']) > 0:
                valid_chunks += 1
        
        if valid_chunks == 0:
            print("❌ Chunks missing embeddings")
            return False
        
        print(f"✅ Search ready: {len(chunks)} chunks with embeddings")
        
        # Show sample chunk info
        sample = chunks[0]
        print(f"   📝 Sample: {sample.get('doc_name', 'Unknown')} - {len(sample.get('text', ''))} chars")
        print(f"   🔢 Embedding dim: {len(sample.get('embedding', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading embeddings: {e}")
        return False

def provide_next_steps(env_ok, docs_ok, pipeline_status, search_ready):
    """Provide specific next steps"""
    print("\n🎯 NEXT STEPS...")
    print("=" * 40)
    
    if not env_ok:
        print("1. 🔑 CONFIGURE API KEYS:")
        print("   Edit .env file and add:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("   Get free key: https://aistudio.google.com/app/apikey")
        
    if not docs_ok:
        print("2. 📁 ADD DOCUMENTS:")
        print("   Place your PDF/text files in data/raw/")
        
    if not all(pipeline_status.values()):
        print("3. 🔄 RUN MISSING PIPELINE STEPS:")
        missing_steps = [step for step, done in pipeline_status.items() if not done]
        
        if 'OCR Processing' in missing_steps:
            print("   python src/ocr.py")
        if 'Text Chunking' in missing_steps:
            print("   python src/chunking.py")
        if 'Keyword Extraction' in missing_steps:
            print("   python src/keywords.py")
        if 'Embeddings Generation' in missing_steps:
            print("   python src/embeddings.py")
            
    if env_ok and docs_ok and all(pipeline_status.values()) and search_ready:
        print("🎉 READY TO SEARCH!")
        print("   python src/search.py --interactive")
        print("   OR")
        print("   python src/search.py \"your question here\"")

def main():
    """Run complete status check"""
    print("📊 RAG APPLICATION STATUS CHECK")
    print("=" * 50)
    
    # Run all checks
    env_ok = check_env_file()
    docs_ok = check_documents()
    pipeline_status = check_pipeline_status()
    search_ready = check_search_readiness()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 50)
    print(f"Environment:      {'✅' if env_ok else '❌'}")
    print(f"Documents:        {'✅' if docs_ok else '❌'}")
    print(f"OCR Processing:   {'✅' if pipeline_status.get('OCR Processing', False) else '❌'}")
    print(f"Text Chunking:    {'✅' if pipeline_status.get('Text Chunking', False) else '❌'}")
    print(f"Keyword Extract:  {'✅' if pipeline_status.get('Keyword Extraction', False) else '❌'}")
    print(f"Embeddings:       {'✅' if pipeline_status.get('Embeddings Generation', False) else '❌'}")
    print(f"Search Ready:     {'✅' if search_ready else '❌'}")
    
    all_ready = (env_ok and docs_ok and 
                all(pipeline_status.values()) and search_ready)
    
    if all_ready:
        print("\n🎉 EVERYTHING IS READY!")
        print("Your RAG application should work perfectly now!")
    else:
        provide_next_steps(env_ok, docs_ok, pipeline_status, search_ready)

if __name__ == "__main__":
    main()