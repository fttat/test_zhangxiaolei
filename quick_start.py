#!/usr/bin/env python3
"""
CCGL Analytics System - Quick Start Script
One-command setup and launch for the entire system
"""

import os
import sys
import subprocess
import time
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import platform

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccgl_analytics.utils.logger import get_logger, setup_logging

class QuickStartManager:
    """Manager for quick start operations."""
    
    def __init__(self):
        """Initialize quick start manager."""
        self.logger = get_logger(__name__)
        self.system_info = self._get_system_info()
        self.setup_complete = False
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information.
        
        Returns:
            System information dictionary
        """
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor() or platform.machine(),
            'cwd': os.getcwd()
        }
    
    def display_welcome(self):
        """Display welcome message and system info."""
        print("🚀 CCGL Analytics System - Quick Start")
        print("=" * 60)
        print("Welcome to Centralized Control and Group Learning Analytics!")
        print("")
        print("📋 System Information:")
        print(f"  Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
        print(f"  Python: {self.system_info['python_version']}")
        print(f"  Working Directory: {self.system_info['cwd']}")
        print("")
        print("This script will help you set up and run CCGL Analytics System.")
        print("=" * 60)
        print("")
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites.
        
        Returns:
            True if all prerequisites are met
        """
        self.logger.info("Checking system prerequisites")
        print("🔍 Checking Prerequisites...")
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')[:2]))
        if python_version < (3, 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {platform.python_version()}")
            return False
        else:
            print(f"✅ Python {platform.python_version()} - OK")
        
        # Check for required files
        required_files = [
            'config.yml',
            'requirements.txt',
            'ccgl_analytics/__init__.py',
            'main.py',
            'main_mcp.py',
            'main_llm.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
            else:
                print(f"✅ {file_path} - Found")
        
        if missing_files:
            print("❌ Missing required files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        
        # Check for pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         check=True, capture_output=True)
            print("✅ pip - Available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
        
        print("✅ All prerequisites met")
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment and install dependencies.
        
        Returns:
            True if setup successful
        """
        self.logger.info("Setting up environment")
        print("\n📦 Setting Up Environment...")
        
        # Create directories
        directories = [
            'logs', 'reports', 'uploads', 'temp', 'data',
            'config', 'scripts', 'tests', 'docs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Copy .env.example to .env if .env doesn't exist
        if not Path('.env').exists() and Path('.env.example').exists():
            shutil.copy('.env.example', '.env')
            print("✅ Created .env from .env.example")
            print("   💡 Please edit .env with your actual configuration")
        
        # Install dependencies
        print("\n📚 Installing Dependencies...")
        print("   This may take a few minutes...")
        
        try:
            # Install core dependencies first
            core_deps = [
                'pyyaml>=6.0',
                'pandas>=2.0.0',
                'numpy>=1.24.0',
                'requests>=2.31.0'
            ]
            
            for dep in core_deps:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Installed: {dep}")
                else:
                    print(f"⚠️  Warning: Failed to install {dep}")
            
            # Install from requirements.txt
            if Path('requirements.txt').exists():
                print("📦 Installing from requirements.txt...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Requirements installed successfully")
                else:
                    print("⚠️  Some packages may have failed to install")
                    print("   You can continue, but some features may not work")
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        print("✅ Environment setup complete")
        self.setup_complete = True
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration files.
        
        Returns:
            True if configuration is valid
        """
        self.logger.info("Validating configuration")
        print("\n⚙️  Validating Configuration...")
        
        # Check config.yml
        try:
            with open('config.yml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['database', 'analysis', 'logging', 'mcp', 'llm']
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
                else:
                    print(f"✅ Configuration section: {section}")
            
            if missing_sections:
                print("⚠️  Missing configuration sections:")
                for section in missing_sections:
                    print(f"   - {section}")
                print("   Using default values")
            
            print("✅ Configuration file is valid")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def run_system_test(self) -> bool:
        """Run basic system test.
        
        Returns:
            True if test passes
        """
        self.logger.info("Running system test")
        print("\n🧪 Running System Test...")
        
        try:
            # Test basic import
            result = subprocess.run([
                sys.executable, '-c', 
                'import ccgl_analytics; print("Import successful")'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Package import test - PASSED")
            else:
                print("❌ Package import test - FAILED")
                print(f"   Error: {result.stderr}")
                return False
            
            # Test main.py help
            result = subprocess.run([
                sys.executable, 'main.py', '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Main program test - PASSED")
            else:
                print("❌ Main program test - FAILED")
                return False
            
            print("✅ System test completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ System test timed out")
            return False
        except Exception as e:
            print(f"❌ System test failed: {e}")
            return False
    
    def show_usage_examples(self):
        """Show usage examples and next steps."""
        print("\n🎯 Quick Start Complete!")
        print("=" * 60)
        print("Your CCGL Analytics System is ready to use!")
        print("")
        print("🚀 Usage Examples:")
        print("")
        
        print("1️⃣  Basic Data Analysis:")
        print("   python main.py -c config.yml -f your_data.csv --analysis quality,clustering")
        print("")
        
        print("2️⃣  MCP Distributed Architecture:")
        print("   python main_mcp.py --start-mcp-servers --interactive")
        print("")
        
        print("3️⃣  AI-Enhanced Interactive Mode:")
        print("   python main_llm.py --interactive")
        print("")
        
        print("4️⃣  Web Dashboard:")
        print("   ./scripts/run_web_analysis.sh")
        print("")
        
        print("📚 Available Commands:")
        print("   main.py           - Basic analysis with ML algorithms")
        print("   main_mcp.py       - Distributed MCP architecture") 
        print("   main_llm.py       - AI-enhanced natural language querying")
        print("   quick_start.py    - This setup script")
        print("")
        
        print("📖 Documentation:")
        print("   README.md         - Complete documentation")
        print("   config.yml        - System configuration")
        print("   .env              - Environment variables")
        print("")
        
        print("🔧 Configuration:")
        print("   - Edit .env with your database and API keys")
        print("   - Modify config.yml for advanced settings")
        print("   - Check logs/ directory for system logs")
        print("")
        
        print("🆘 Need Help?")
        print("   - Run any script with --help for detailed options")
        print("   - Check the logs/ directory for error messages")
        print("   - See README.md for complete documentation")
        print("")
        print("=" * 60)
        print("Happy analyzing! 🎉")
    
    def create_sample_data(self):
        """Create sample data file for testing."""
        print("\n📊 Creating Sample Data...")
        
        try:
            import pandas as pd
            import numpy as np
            
            # Create sample sales data
            np.random.seed(42)
            n_samples = 1000
            
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'sales_amount': np.random.normal(1000, 300, n_samples),
                'customer_id': np.random.randint(1, 200, n_samples),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
                'customer_age': np.random.randint(18, 80, n_samples),
                'discount_applied': np.random.choice([0, 5, 10, 15, 20], n_samples),
                'payment_method': np.random.choice(['Credit Card', 'Cash', 'Debit Card'], n_samples)
            })
            
            # Add some missing values and anomalies
            sample_data.loc[sample_data.index[:50], 'sales_amount'] = np.nan
            sample_data.loc[sample_data.index[100:105], 'sales_amount'] = 10000  # Anomalies
            
            # Save sample data
            sample_file = 'data/sample_sales_data.csv'
            sample_data.to_csv(sample_file, index=False)
            
            print(f"✅ Sample data created: {sample_file}")
            print(f"   Shape: {sample_data.shape[0]} rows, {sample_data.shape[1]} columns")
            print("   You can use this file to test the system:")
            print(f"   python main.py -f {sample_file} --analysis all")
            
        except ImportError:
            print("⚠️  pandas/numpy not available, skipping sample data creation")
        except Exception as e:
            print(f"❌ Failed to create sample data: {e}")
    
    def run_interactive_setup(self) -> bool:
        """Run interactive setup process.
        
        Returns:
            True if setup completed successfully
        """
        print("🔧 Interactive Setup Process")
        print("-" * 40)
        
        # Ask user preferences
        print("\n❓ Setup Questions:")
        
        # Create sample data?
        try:
            create_sample = input("Create sample data for testing? (y/n) [y]: ").strip().lower()
            if create_sample in ['', 'y', 'yes']:
                self.create_sample_data()
        except KeyboardInterrupt:
            print("\nSetup cancelled by user")
            return False
        
        # Show configuration notes
        print("\n📝 Configuration Notes:")
        print("   - Database: Edit .env with your database credentials")
        print("   - LLM APIs: Add your API keys to .env for AI features")
        print("   - MCP: Default ports are configured (8000-8004)")
        
        return True

def main():
    """Main entry point for quick start."""
    # Setup basic logging
    setup_logging(level='INFO', format_type='text')
    
    manager = QuickStartManager()
    
    try:
        # Display welcome
        manager.display_welcome()
        
        # Check prerequisites
        if not manager.check_prerequisites():
            print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
            return 1
        
        # Ask for interactive setup
        try:
            interactive = input("\nRun interactive setup? (y/n) [y]: ").strip().lower()
            if interactive not in ['', 'y', 'yes']:
                print("Skipping interactive setup")
            else:
                if not manager.run_interactive_setup():
                    return 1
        except KeyboardInterrupt:
            print("\nSetup cancelled by user")
            return 1
        
        # Setup environment
        if not manager.setup_environment():
            print("\n❌ Environment setup failed")
            return 1
        
        # Validate configuration
        if not manager.validate_configuration():
            print("\n⚠️  Configuration validation failed, but continuing...")
        
        # Run system test
        if not manager.run_system_test():
            print("\n❌ System test failed")
            print("   The system may still work, but some features might not function correctly")
            try:
                continue_anyway = input("Continue anyway? (y/n) [n]: ").strip().lower()
                if continue_anyway not in ['y', 'yes']:
                    return 1
            except KeyboardInterrupt:
                return 1
        
        # Show usage examples
        manager.show_usage_examples()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user. Goodbye!")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Quick start failed: {e}", exc_info=True)
        print(f"\n❌ Quick start failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())