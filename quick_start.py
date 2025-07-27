"""
Quick Start Script for CCGL Analytics
Provides easy setup and execution of the warehouse data analysis platform
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_dependencies():
    """Install required Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_sample_data():
    """Create sample warehouse data"""
    print("🗄️ Creating sample data...")
    
    try:
        subprocess.check_call([
            sys.executable, "main.py", "--create-sample"
        ])
        print("✅ Sample data created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def run_basic_analysis():
    """Run basic data analysis"""
    print("📊 Running basic data analysis...")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Basic analysis completed successfully")
            print("\nAnalysis Summary:")
            # Extract summary from output
            lines = result.stdout.split('\n')
            in_summary = False
            for line in lines:
                if "=== CCGL Analytics Results Summary ===" in line:
                    in_summary = True
                if in_summary:
                    print(line)
            return True
        else:
            print(f"❌ Analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run analysis: {e}")
        return False


def run_mcp_demo():
    """Run MCP architecture demo"""
    print("🔄 Running MCP architecture demo...")
    
    try:
        result = subprocess.run([
            sys.executable, "main_mcp.py", 
            "--start-mcp-servers", "--demo-workflow", "--list-tools"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MCP demo completed successfully")
            print("\nMCP Demo Summary:")
            # Extract relevant output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in [
                    "Active servers:", "Available tools:", "Workflow completed", 
                    "Steps completed:", "Data Quality Score:"
                ]):
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"❌ MCP demo failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run MCP demo: {e}")
        return False


def check_environment():
    """Check system environment and requirements"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required. Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
    
    # Check if required files exist
    required_files = [
        "main.py", "main_mcp.py", "requirements.txt", 
        "ccgl_analytics/__init__.py", "config/mcp_config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    return True


def setup_environment():
    """Setup environment variables and configuration"""
    print("⚙️ Setting up environment...")
    
    # Create .env.example if it doesn't exist
    env_example = Path(".env.example")
    if not env_example.exists():
        env_content = """# CCGL Analytics Environment Variables

# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=ccgl_warehouse

# LLM API Keys (optional)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ZHIPU_API_KEY=your_zhipu_api_key_here

# Logging
LOG_LEVEL=INFO
"""
        with open(env_example, 'w') as f:
            f.write(env_content)
        print("✅ Created .env.example file")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Created logs directory")
    
    return True


def main():
    """Main quick start function"""
    parser = argparse.ArgumentParser(description='CCGL Analytics Quick Start')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--basic-only', action='store_true',
                       help='Run only basic analysis')
    parser.add_argument('--mcp-only', action='store_true',
                       help='Run only MCP demo')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup environment')
    
    args = parser.parse_args()
    
    print("🏭 CCGL Analytics Quick Start")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please resolve issues before continuing.")
        return 1
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return 1
    
    if args.setup_only:
        print("\n✅ Environment setup completed!")
        return 0
    
    # Step 3: Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("\n❌ Dependency installation failed.")
            return 1
    
    # Step 4: Create sample data
    if not create_sample_data():
        print("\n❌ Sample data creation failed.")
        return 1
    
    # Step 5: Run analysis
    success = True
    
    if not args.mcp_only:
        print("\n" + "=" * 50)
        success = run_basic_analysis() and success
    
    if not args.basic_only:
        print("\n" + "=" * 50)
        success = run_mcp_demo() and success
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 CCGL Analytics Quick Start completed successfully!")
        print("\nNext steps:")
        print("  1. Review the generated results in the 'results/' directory")
        print("  2. Explore the configuration files in the 'config/' directory")
        print("  3. Try running interactive MCP mode: python main_mcp.py --full-mcp --start-mcp-servers")
        print("  4. Check out the documentation in the 'docs/' directory")
        return 0
    else:
        print("⚠️ Quick start completed with some issues. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())