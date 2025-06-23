"""
RAG Pipeline Main Controller
Orchestrates the complete RAG pipeline: OCR -> Chunking -> Keywords -> Embeddings -> Search & Themes
Supports multiple execution modes: full pipeline, individual steps, interactive mode
Enhanced error handling, progress tracking, and comprehensive logging
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import subprocess
import importlib.util

# Add project root to path
project_root = Path(__file__).parent
# Ensure we're in the correct directory (not in src/)
if project_root.name == "src":
    project_root = project_root.parent
sys.path.append(str(project_root))

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging for the RAG pipeline"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class RAGPipelineController:
    """
    Main controller for the complete RAG pipeline
    Handles execution, monitoring, error recovery, and user interaction
    """
    
    def __init__(self):
        self.project_root = project_root
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Pipeline steps configuration
        self.pipeline_steps = {
            "ocr": {
                "name": "Document Processing & OCR",
                "script": "ocr.py",
                "description": "Extract text from PDFs and images using OCR",
                "input_dir": "data/raw",
                "output_dir": "data/processed",
                "output_file": "processing_summary.json",
                "estimated_time": "2-5 minutes per document"
            },
            "chunking": {
                "name": "Text Chunking",
                "script": "chunking.py", 
                "description": "Split documents into semantic chunks",
                "input_dir": "data/processed",
                "output_dir": "data/chunks",
                "output_file": "all_chunks.json",
                "estimated_time": "1-3 minutes"
            },
            "keywords": {
                "name": "Keyword Extraction",
                "script": "keywords.py",
                "description": "Extract keywords from text chunks",
                "input_dir": "data/chunks",
                "output_dir": "data/keywords", 
                "output_file": "all_chunks_with_keywords.json",
                "estimated_time": "2-5 minutes"
            },
            "embeddings": {
                "name": "Vector Embeddings Generation",
                "script": "embeddings.py",
                "description": "Generate vector embeddings for semantic search",
                "input_dir": "data/keywords",
                "output_dir": "data/embeddings",
                "output_file": "all_chunks_with_embeddings.json", 
                "estimated_time": "3-10 minutes"
            }
        }
        
        # Interactive tools
        self.interactive_tools = {
            "search": {
                "name": "Semantic Search & QA",
                "script": "search.py",
                "description": "Ask questions and get AI-powered answers",
                "mode": "--interactive"
            },
            "themes": {
                "name": "Theme Analysis", 
                "script": "themes.py",
                "description": "Identify themes and get comprehensive analysis",
                "mode": "--interactive"
            }
        }
        
        # Ensure directories exist
        self._create_directories()
        
        logger.info("RAG Pipeline Controller initialized")
    
    def _create_directories(self):
        """Create necessary directories for the pipeline"""
        directories = [
            "data/raw",
            "data/processed", 
            "data/chunks",
            "data/keywords",
            "data/embeddings",
            "data/themes",
            "logs",
            "src"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if required Python packages are installed"""
        required_packages = {
            'fitz': 'PyMuPDF',
            'pytesseract': 'pytesseract', 
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'sentence_transformers': 'sentence-transformers'
        }
        
        optional_packages = {
            'google.generativeai': 'google-generativeai',
            'openai': 'openai',
            'anthropic': 'anthropic',
            'qdrant_client': 'qdrant-client',
            'dotenv': 'python-dotenv'
        }
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for module, package in required_packages.items():
            try:
                importlib.import_module(module)
            except ImportError:
                missing_required.append(package)
        
        # Check optional packages
        for module, package in optional_packages.items():
            try:
                importlib.import_module(module)
            except ImportError:
                missing_optional.append(package)
        
        all_good = len(missing_required) == 0
        
        if missing_required:
            print(f"❌ Missing required packages: {', '.join(missing_required)}")
            print(f"💡 Install with: pip install {' '.join(missing_required)}")
        
        if missing_optional:
            print(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
            print(f"💡 For enhanced features: pip install {' '.join(missing_optional)}")
        
        return all_good, missing_required + missing_optional
    
    def check_environment(self) -> Dict[str, bool]:
        """Check environment setup and configuration"""
        env_status = {}
        
        # Check for .env file
        env_file = self.project_root / ".env"
        env_status["env_file"] = env_file.exists()
        
        # Check for API keys (optional but recommended)
        api_keys = {
            "GROQ_API_KEY": os.getenv('GROQ_API_KEY', ''),
            "GEMINI_API_KEY": os.getenv('GEMINI_API_KEY', ''),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY', ''),
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY', '')
        }
        
        env_status["api_keys"] = any(key for key in api_keys.values())
        env_status["api_details"] = {k: bool(v) for k, v in api_keys.items()}
        
        # Check input data with improved file detection
        raw_dir = self.project_root / "data" / "raw"
        
        # Debug: Print the actual path being checked
        logger.info(f"Checking for files in: {raw_dir}")
        logger.info(f"Directory exists: {raw_dir.exists()}")
        if raw_dir.exists():
            all_items = list(raw_dir.iterdir())
            logger.info(f"Items in directory: {len(all_items)}")
            for item in all_items[:5]:  # Show first 5 items
                logger.info(f"  - {item.name} ({'file' if item.is_file() else 'dir'})")
        
        try:
            if raw_dir.exists():
                # Get PDF files (both .pdf and .PDF)
                pdf_files = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.PDF"))
                
                # Get image files with case variations
                image_files = []
                image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']
                for ext in image_extensions:
                    image_files.extend(raw_dir.glob(f"*{ext}"))
                    image_files.extend(raw_dir.glob(f"*{ext.upper()}"))
                
                # Remove duplicates (in case of case-insensitive filesystems)
                pdf_files = list(set(pdf_files))
                image_files = list(set(image_files))
                
                total_files = len(pdf_files) + len(image_files)
                
                env_status["raw_data"] = total_files > 0
                env_status["file_counts"] = {
                    "pdf_files": len(pdf_files),
                    "image_files": len(image_files),
                    "total_files": total_files
                }
                
                # Debug info for troubleshooting
                if total_files > 0:
                    logger.info(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files in {raw_dir}")
                    # Show first few filenames for verification
                    sample_files = [f.name for f in pdf_files[:3]]
                    if sample_files:
                        logger.info(f"Sample PDF files: {sample_files}")
                else:
                    logger.warning(f"No files found in {raw_dir}")
                    
            else:
                logger.warning(f"Raw directory does not exist: {raw_dir}")
                env_status["raw_data"] = False
                env_status["file_counts"] = {"pdf_files": 0, "image_files": 0, "total_files": 0}
                
        except Exception as e:
            logger.error(f"Error checking files in {raw_dir}: {e}")
            env_status["raw_data"] = False
            env_status["file_counts"] = {"pdf_files": 0, "image_files": 0, "total_files": 0}
        
        return env_status
    
    def check_pipeline_status(self) -> Dict[str, Dict[str, any]]:
        """Check the completion status of each pipeline step"""
        status = {}
        
        for step_id, step_config in self.pipeline_steps.items():
            step_status = {
                "completed": False,
                "output_exists": False,
                "file_count": 0,
                "last_modified": None,
                "file_size": 0
            }
            
            # Check if output directory and file exist
            output_dir = self.project_root / step_config["output_dir"]
            output_file = output_dir / step_config["output_file"]
            
            if output_file.exists():
                step_status["output_exists"] = True
                step_status["last_modified"] = datetime.fromtimestamp(output_file.stat().st_mtime)
                step_status["file_size"] = output_file.stat().st_size
                
                # Try to validate the file content
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if step_id == "ocr":
                        step_status["completed"] = bool(data.get("successfully_processed", 0) > 0)
                        step_status["file_count"] = data.get("successfully_processed", 0)
                    elif "chunks" in data:
                        chunks = data.get("chunks", [])
                        step_status["completed"] = len(chunks) > 0
                        step_status["file_count"] = len(chunks)
                    else:
                        step_status["completed"] = True
                        
                except (json.JSONDecodeError, KeyError):
                    step_status["completed"] = False
            
            status[step_id] = step_status
        
        return status
    
    def run_pipeline_step(self, step_id: str, force: bool = False) -> Tuple[bool, str]:
        """Run a single pipeline step"""
        if step_id not in self.pipeline_steps:
            return False, f"Unknown step: {step_id}"
        
        step_config = self.pipeline_steps[step_id]
        script_path = self.src_dir / step_config["script"]
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        
        print(f"\n🔄 Running: {step_config['name']}")
        print(f"📄 Description: {step_config['description']}")
        print(f"⏱️  Estimated time: {step_config['estimated_time']}")
        print(f"📁 Input: {step_config['input_dir']}")
        print(f"📁 Output: {step_config['output_dir']}")
        
        start_time = time.time()
        
        try:
            # Run the Python script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {step_config['name']} completed successfully!")
                print(f"⏱️  Duration: {duration/60:.1f} minutes")
                
                # Show output summary if available
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    print(f"📊 Output summary:")
                    for line in output_lines[-10:]:  # Show last 10 lines
                        if line.strip():
                            print(f"   {line}")
                
                logger.info(f"Step {step_id} completed successfully in {duration:.1f}s")
                return True, f"Completed in {duration/60:.1f} minutes"
            else:
                print(f"❌ {step_config['name']} failed!")
                print(f"⚠️  Error output:")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[-5:]:  # Show last 5 error lines
                        if line.strip():
                            print(f"   {line}")
                
                logger.error(f"Step {step_id} failed: {result.stderr}")
                return False, f"Failed: {result.stderr[:200]}..."
                
        except subprocess.TimeoutExpired:
            error_msg = f"Step {step_id} timed out after 30 minutes"
            print(f"⏰ {error_msg}")
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running {step_id}: {str(e)}"
            print(f"💥 {error_msg}")
            logger.error(error_msg)
            return False, error_msg
    
    def run_full_pipeline(self, force: bool = False, skip_completed: bool = True) -> bool:
        """Run the complete RAG pipeline"""
        print("🚀 STARTING COMPLETE RAG PIPELINE")
        print("=" * 60)
        
        pipeline_start_time = time.time()
        completed_steps = []
        failed_steps = []
        
        # Check current status
        if not force and skip_completed:
            status = self.check_pipeline_status()
            print(f"\n📊 Current Pipeline Status:")
            for step_id, step_status in status.items():
                step_name = self.pipeline_steps[step_id]["name"]
                status_icon = "✅" if step_status["completed"] else "⏸️"
                print(f"   {status_icon} {step_name}: {'Completed' if step_status['completed'] else 'Pending'}")
        
        # Run each step
        for i, step_id in enumerate(self.pipeline_steps.keys(), 1):
            step_config = self.pipeline_steps[step_id]
            
            print(f"\n📍 STEP {i}/{len(self.pipeline_steps)}: {step_config['name']}")
            print("-" * 50)
            
            # Check if step should be skipped
            if not force and skip_completed:
                status = self.check_pipeline_status()
                if status[step_id]["completed"]:
                    print(f"⏭️  Skipping {step_config['name']} (already completed)")
                    print(f"   📄 Output: {status[step_id]['file_count']} items")
                    print(f"   📅 Last modified: {status[step_id]['last_modified']}")
                    completed_steps.append(step_id)
                    continue
            
            # Run the step
            success, message = self.run_pipeline_step(step_id, force)
            
            if success:
                completed_steps.append(step_id)
                print(f"✅ Step {i} completed: {message}")
            else:
                failed_steps.append(step_id)
                print(f"❌ Step {i} failed: {message}")
                
                # Ask user if they want to continue despite failure
                if i < len(self.pipeline_steps):
                    response = input(f"\n❓ Continue with remaining steps? (y/n): ").lower()
                    if response != 'y':
                        break
            
            # Small delay between steps
            time.sleep(2)
        
        # Pipeline summary
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        print(f"\n🏁 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total time: {total_duration/60:.1f} minutes")
        print(f"✅ Completed steps: {len(completed_steps)}/{len(self.pipeline_steps)}")
        print(f"❌ Failed steps: {len(failed_steps)}")
        
        if completed_steps:
            print(f"\n✅ Successfully completed:")
            for step_id in completed_steps:
                print(f"   • {self.pipeline_steps[step_id]['name']}")
        
        if failed_steps:
            print(f"\n❌ Failed steps:")
            for step_id in failed_steps:
                print(f"   • {self.pipeline_steps[step_id]['name']}")
        
        # Final status check
        final_status = self.check_pipeline_status()
        all_completed = all(status["completed"] for status in final_status.values())
        
        if all_completed:
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"🔍 Ready for semantic search and theme analysis")
            return True
        else:
            print(f"\n⚠️  PIPELINE PARTIALLY COMPLETED")
            print(f"💡 You can run failed steps individually or with --force")
            return False
    
    def launch_interactive_tool(self, tool_name: str) -> bool:
        """Launch an interactive tool (search or themes)"""
        if tool_name not in self.interactive_tools:
            print(f"❌ Unknown tool: {tool_name}")
            return False
        
        tool_config = self.interactive_tools[tool_name]
        script_path = self.src_dir / tool_config["script"]
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n🚀 Launching: {tool_config['name']}")
        print(f"📄 {tool_config['description']}")
        print("-" * 50)
        
        try:
            # Launch interactive tool
            subprocess.run(
                [sys.executable, str(script_path), tool_config["mode"]],
                cwd=self.project_root
            )
            return True
        except KeyboardInterrupt:
            print(f"\n👋 {tool_config['name']} session ended")
            return True
        except Exception as e:
            print(f"❌ Error launching {tool_name}: {e}")
            return False
    
    def show_status_dashboard(self):
        """Display comprehensive system status dashboard"""
        print("\n🎛️  RAG PIPELINE STATUS DASHBOARD")
        print("=" * 60)
        
        # Environment check
        env_status = self.check_environment()
        
        print(f"\n📁 Environment:")
        print(f"   📄 .env file: {'✅' if env_status['env_file'] else '❌'}")
        print(f"   🔑 API keys configured: {'✅' if env_status['api_keys'] else '⚠️'}")
        
        if env_status['api_keys']:
            for key, status in env_status['api_details'].items():
                status_icon = "✅" if status else "❌"
                print(f"      {status_icon} {key}")
        
        print(f"   📚 Input data: {'✅' if env_status['raw_data'] else '❌'}")
        file_counts = env_status['file_counts']
        print(f"      📄 PDF files: {file_counts['pdf_files']}")
        print(f"      🖼️  Image files: {file_counts['image_files']}")
        print(f"      📊 Total files: {file_counts['total_files']}")
        
        # Dependencies check
        deps_ok, missing = self.check_dependencies()
        print(f"\n📦 Dependencies: {'✅' if deps_ok else '⚠️'}")
        if missing:
            print(f"   ⚠️  Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
        
        # Pipeline status
        pipeline_status = self.check_pipeline_status()
        print(f"\n🔄 Pipeline Status:")
        
        for step_id, step_config in self.pipeline_steps.items():
            status = pipeline_status[step_id]
            status_icon = "✅" if status["completed"] else "⏸️"
            
            print(f"   {status_icon} {step_config['name']}")
            
            if status["output_exists"]:
                print(f"      📄 Items: {status['file_count']}")
                print(f"      📅 Modified: {status['last_modified'].strftime('%Y-%m-%d %H:%M') if status['last_modified'] else 'Unknown'}")
                print(f"      💾 Size: {status['file_size']/1024/1024:.1f} MB")
            else:
                print(f"      ❌ No output found")
        
        # Ready status
        all_completed = all(status["completed"] for status in pipeline_status.values())
        print(f"\n🎯 System Status: {'✅ Ready for Search & Analysis' if all_completed else '⚠️ Pipeline Incomplete'}")
        
        if all_completed:
            print(f"\n🔍 Available Tools:")
            for tool_name, tool_config in self.interactive_tools.items():
                print(f"   • {tool_config['name']}: {tool_config['description']}")
    
    def show_help(self):
        """Display comprehensive help information"""
        print("\n📚 RAG PIPELINE HELP")
        print("=" * 50)
        
        print(f"\n🎯 PURPOSE:")
        print(f"   Build a complete Retrieval-Augmented Generation (RAG) system")
        print(f"   Process documents → Extract knowledge → Enable AI-powered Q&A")
        
        print(f"\n🔄 PIPELINE STEPS:")
        for i, (step_id, step_config) in enumerate(self.pipeline_steps.items(), 1):
            print(f"   {i}. {step_config['name']}")
            print(f"      📄 {step_config['description']}")
            print(f"      ⏱️  {step_config['estimated_time']}")
        
        print(f"\n🛠️  COMMANDS:")
        print(f"   python main.py --full                 # Run complete pipeline")
        print(f"   python main.py --step ocr             # Run single step")
        print(f"   python main.py --search               # Launch search interface")
        print(f"   python main.py --themes               # Launch theme analysis")
        print(f"   python main.py --status               # Show system status")
        print(f"   python main.py --interactive          # Interactive menu")
        
        print(f"\n📁 SETUP REQUIREMENTS:")
        print(f"   1. Create data/raw/ directory")
        print(f"   2. Add your PDF/image files to data/raw/")
        print(f"   3. Install dependencies: pip install -r requirements.txt")
        print(f"   4. Configure API keys in .env file (optional but recommended)")
        
        print(f"\n🔑 RECOMMENDED API KEYS:")
        print(f"   GROQ_API_KEY=your_key        # For AI analysis")
        print(f"   GEMINI_API_KEY=your_key      # For embeddings & AI")
        print(f"   OPENAI_API_KEY=your_key      # Alternative AI provider")
        print(f"   ANTHROPIC_API_KEY=your_key   # Claude AI")
        
        print(f"\n💡 TIPS:")
        print(f"   • Start with --status to check your setup")
        print(f"   • Use --full to run the complete pipeline")
        print(f"   • Try --search for quick document Q&A")
        print(f"   • Use --themes for comprehensive analysis")

def main():
    """Main entry point with comprehensive CLI interface"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Controller - Build and interact with your RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                 # Run complete pipeline
  python main.py --step ocr             # Run OCR step only
  python main.py --search               # Launch search interface
  python main.py --themes               # Launch theme analysis
  python main.py --status               # Show system status
  python main.py --interactive          # Interactive menu mode
        """
    )
    
    # Main actions
    parser.add_argument('--full', action='store_true', help='Run the complete RAG pipeline')
    parser.add_argument('--step', choices=['ocr', 'chunking', 'keywords', 'embeddings'], help='Run a specific pipeline step')
    parser.add_argument('--search', action='store_true', help='Launch interactive search interface')
    parser.add_argument('--themes', action='store_true', help='Launch theme analysis interface')
    parser.add_argument('--status', action='store_true', help='Show system status dashboard')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive menu')
    
    # Options
    parser.add_argument('--force', action='store_true', help='Force re-run completed steps')
    parser.add_argument('--no-skip', action='store_true', help='Do not skip completed steps')
    parser.add_argument('--help-detailed', action='store_true', help='Show detailed help and setup guide')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = RAGPipelineController()
    
    # Handle detailed help
    if args.help_detailed:
        controller.show_help()
        return
    
    # Handle specific actions
    if args.status:
        controller.show_status_dashboard()
        return
    
    if args.step:
        success, message = controller.run_pipeline_step(args.step, args.force)
        if success:
            print(f"\n✅ Step completed successfully!")
        else:
            print(f"\n❌ Step failed: {message}")
        return
    
    if args.full:
        success = controller.run_full_pipeline(
            force=args.force, 
            skip_completed=not args.no_skip
        )
        
        if success:
            print(f"\n🎉 Pipeline completed! Ready for search and analysis.")
            print(f"💡 Try: python main.py --search")
        else:
            print(f"\n⚠️  Pipeline incomplete. Check errors above.")
        return
    
    if args.search:
        # Check if pipeline is ready
        status = controller.check_pipeline_status()
        if not status["embeddings"]["completed"]:
            print("❌ Pipeline not ready for search. Run --full first.")
            return
        
        controller.launch_interactive_tool("search")
        return
    
    if args.themes:
        # Check if pipeline is ready
        status = controller.check_pipeline_status()
        if not status["keywords"]["completed"]:
            print("❌ Pipeline not ready for themes. Run --full first.")
            return
        
        controller.launch_interactive_tool("themes")
        return
    
    if args.interactive:
        # Interactive menu mode
        print("🎛️  RAG PIPELINE INTERACTIVE MODE")
        print("=" * 40)
        
        while True:
            print(f"\n📋 MENU:")
            print(f"1. 📊 Show Status Dashboard")
            print(f"2. 🚀 Run Full Pipeline")
            print(f"3. 🔧 Run Individual Step")
            print(f"4. 🔍 Launch Search Interface")
            print(f"5. 🎯 Launch Theme Analysis")
            print(f"6. 📚 Show Help Guide")
            print(f"7. 🚪 Exit")
            
            try:
                choice = input(f"\n👉 Select option (1-7): ").strip()
                
                if choice == "1":
                    controller.show_status_dashboard()
                
                elif choice == "2":
                    skip_completed = input("Skip completed steps? (Y/n): ").lower() != 'n'
                    controller.run_full_pipeline(skip_completed=skip_completed)
                
                elif choice == "3":
                    print("\nAvailable steps:")
                    for i, step_id in enumerate(controller.pipeline_steps.keys(), 1):
                        step_name = controller.pipeline_steps[step_id]["name"]
                        print(f"   {i}. {step_id} - {step_name}")
                    
                    step_choice = input("Enter step name: ").strip().lower()
                    if step_choice in controller.pipeline_steps:
                        force = input("Force re-run? (y/N): ").lower() == 'y'
                        controller.run_pipeline_step(step_choice, force)
                    else:
                        print("❌ Invalid step name")
                
                elif choice == "4":
                    controller.launch_interactive_tool("search")
                
                elif choice == "5":
                    controller.launch_interactive_tool("themes")
                
                elif choice == "6":
                    controller.show_help()
                
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-7.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return
    
    # Default: show help and status
    print("🚀 RAG PIPELINE CONTROLLER")
    print("=" * 30)
    print("Welcome to the Retrieval-Augmented Generation Pipeline!")
    print("\n💡 Quick start:")
    print("   1. Add documents to data/raw/")
    print("   2. Run: python main.py --full")
    print("   3. Try: python main.py --search")
    print("\n📊 Current Status:")
    
    # Show quick status
    env_status = controller.check_environment()
    pipeline_status = controller.check_pipeline_status()
    
    file_count = env_status['file_counts']['total_files']
    completed_steps = sum(1 for status in pipeline_status.values() if status["completed"])
    total_steps = len(pipeline_status)
    
    print(f"   📁 Input files: {file_count}")
    print(f"   ✅ Completed steps: {completed_steps}/{total_steps}")
    
    if file_count == 0:
        print(f"\n⚠️  No input files found. Add PDFs/images to data/raw/")
    elif completed_steps == 0:
        print(f"\n💡 Ready to start: python main.py --full")
    elif completed_steps == total_steps:
        print(f"\n🎉 Ready for search and analysis!")
        print(f"   🔍 Search: python main.py --search")
        print(f"   🎯 Themes: python main.py --themes")
    else:
        print(f"\n🔄 Pipeline partially complete. Continue with: python main.py --full")
    
    print(f"\n📚 For help: python main.py --help-detailed")
    print(f"📊 For status: python main.py --status")

def create_sample_env_file():
    """Create a sample .env file with all configuration options"""
    env_content = """# RAG Pipeline Configuration
# Copy this file to .env and add your API keys

# =============================================================================
# AI SERVICE API KEYS (At least one recommended for enhanced features)
# =============================================================================

# Groq (Fast, free tier available) - Recommended for theme analysis
GROQ_API_KEY=your_groq_api_key_here

# Google Gemini (Good for embeddings and chat) - Recommended for embeddings
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI (GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude (High quality responses)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

# Choose embedding model (affects search quality)
# Options: text-embedding-004 (Gemini), text-embedding-3-small (OpenAI), 
#          sentence-transformers/all-MiniLM-L6-v2 (Local)
EMBEDDING_MODEL=text-embedding-004

# Batch size for embedding generation (adjust based on API limits)
EMBEDDING_BATCH_SIZE=32

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

# Number of chunks to retrieve for search
TOP_K_CHUNKS=10

# Number of final chunks used for answer generation
FINAL_CHUNKS=3

# Minimum similarity threshold for search results (0.0 to 1.0)
SIMILARITY_THRESHOLD=0.3

# Maximum context length for answer generation
MAX_CONTEXT_LENGTH=4000

# =============================================================================
# OPTIONAL: QDRANT VECTOR DATABASE (Advanced Users)
# =============================================================================

# Qdrant Cloud (if using cloud instance)
# QDRANT_URL=https://your-cluster.qdrant.tech
# QDRANT_API_KEY=your_qdrant_api_key

# Qdrant Local (if running locally)
# QDRANT_HOST=localhost
# QDRANT_PORT=6333

# Collection name in Qdrant
# COLLECTION_NAME=document_chunks

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# OCR Configuration
USE_OCR=true
OCR_ENGINE=tesseract
OCR_DPI=300
OCR_LANGUAGE=eng
ENHANCE_IMAGE=true

# Text Processing
MIN_PARAGRAPH_LENGTH=50
MAX_PARAGRAPH_LENGTH=2000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Keyword Extraction
MAX_KEYWORDS=20
KEYWORD_MIN_SCORE=0.3

# Retry Configuration
MAX_RETRIES=3

# Debug Mode (set to false for production)
DEBUG=true
"""
    
    env_file = project_root / ".env.sample"
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(f"📄 Created sample configuration: {env_file}")
    print(f"💡 Copy to .env and configure your API keys")

def setup_command():
    """Setup command to initialize the RAG pipeline environment"""
    print("🛠️  RAG PIPELINE SETUP")
    print("=" * 30)
    
    controller = RAGPipelineController()
    
    print("📁 Creating directory structure...")
    controller._create_directories()
    print("✅ Directories created")
    
    print("\n📄 Creating sample configuration...")
    create_sample_env_file()
    
    print("\n📦 Checking dependencies...")
    deps_ok, missing = controller.check_dependencies()
    
    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("✅ All dependencies installed")
    
    print("\n📚 Next steps:")
    print("1. Copy .env.sample to .env")
    print("2. Add your API keys to .env (at least one recommended)")
    print("3. Add PDF/image files to data/raw/")
    print("4. Run: python main.py --full")
    
    return True

if __name__ == "__main__":
    # Handle special setup command
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_command()
        sys.exit(0)
    
    # Handle version command
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        print("RAG Pipeline Controller v1.0")
        print("Enhanced document processing and semantic search system")
        sys.exit(0)
    
    # Run main controller
    main()