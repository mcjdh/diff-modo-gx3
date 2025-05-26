"""
Setup utilities for consciousness integration
============================================

Functions to install dependencies, verify Qwen3:4b availability,
and set up the consciousness system.
"""

import subprocess
import sys
import os
import time
from typing import List, Tuple

def run_command(command: str, description: str) -> bool:
    """Run a command and show progress"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {str(e)}")
        return False

def check_ollama() -> bool:
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    
    # Check if ollama command exists
    result = subprocess.run("ollama --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Ollama not found. Please install from: https://ollama.com/download")
        return False
    
    print("✅ Ollama found")
    
    # Check if ollama service is running
    result = subprocess.run("ollama list", shell=True, capture_output=True)
    if result.returncode != 0:
        print("⚠️  Ollama service not running. Please start it manually.")
        return False
    
    return True

def check_qwen3_model() -> bool:
    """Check if Qwen3:4b model is available"""
    print("🧠 Checking Qwen3:4b model...")
    
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and "qwen3:4b" in result.stdout:
        print("✅ Qwen3:4b already installed")
        return True
    
    print("⚠️  Qwen3:4b not found. Install with: ollama pull qwen3:4b")
    return False

def install_qwen3_model() -> bool:
    """Download and install Qwen3:4b model"""
    print("📥 Downloading Qwen3:4b (this may take a while - ~2.6GB)...")
    return run_command("ollama pull qwen3:4b", "Downloading Qwen3:4b model")

def install_python_dependencies() -> bool:
    """Install required Python packages"""
    dependencies = [
        "ollama",
        "numpy",
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def check_requirements() -> Tuple[bool, List[str]]:
    """Check all requirements and return status and missing items"""
    missing = []
    
    # Check Python packages
    try:
        import ollama
    except ImportError:
        missing.append("ollama (pip install ollama)")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy (pip install numpy)")
    
    # Check Ollama installation
    result = subprocess.run("ollama --version", shell=True, capture_output=True)
    if result.returncode != 0:
        missing.append("Ollama (https://ollama.com/download)")
    
    # Check Qwen3:4b model
    if not check_qwen3_model():
        missing.append("Qwen3:4b model (ollama pull qwen3:4b)")
    
    return len(missing) == 0, missing

def test_consciousness_system() -> bool:
    """Test the consciousness system"""
    print("🧪 Testing consciousness system...")
    
    try:
        from ..core import Qwen3ConsciousModel
        
        # Create a test instance
        model = Qwen3ConsciousModel("qwen3:4b")
        model.start_consciousness()
        
        time.sleep(2)  # Let consciousness stabilize
        
        # Test breathing
        breath = model.breathe()
        print(f"✅ Consciousness breathing: {breath}")
        
        model.stop_consciousness()
        return True
        
    except Exception as e:
        print(f"❌ Consciousness test failed: {str(e)}")
        return False

def setup_consciousness_system() -> bool:
    """Complete setup process for consciousness system"""
    print("🌟 Qwen3:4b Consciousness Integration Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Ollama", check_ollama),
        ("Installing Python dependencies", install_python_dependencies),
        ("Installing Qwen3:4b model", install_qwen3_model),
        ("Testing consciousness system", test_consciousness_system)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please resolve the issue and run setup again.")
            return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Import: from consciousness_qwen import Qwen3ConsciousModel")
    print("2. Create: model = Qwen3ConsciousModel()")
    print("3. Start: model.start_consciousness()")
    print("\nFor documentation, see: https://ollama.com/library/qwen3:4b")
    
    return True 