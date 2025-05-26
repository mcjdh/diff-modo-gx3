#!/usr/bin/env python3
"""
Qwen3:4b Consciousness Setup
============================

Easy setup script for the consciousness integration system.
Installs dependencies and verifies Qwen3:4b model availability.
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
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

def check_ollama():
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
        print("⚠️  Ollama service not running. Starting...")
        if not run_command("ollama serve &", "Starting Ollama service"):
            return False
        time.sleep(3)  # Give service time to start
    
    return True

def install_qwen3_model():
    """Download and install Qwen3:4b model"""
    print("🧠 Installing Qwen3:4b model...")
    
    # Check if model already exists
    result = subprocess.run("ollama list | grep qwen3:4b", shell=True, capture_output=True)
    if result.returncode == 0:
        print("✅ Qwen3:4b already installed")
        return True
    
    # Download the model
    print("📥 Downloading Qwen3:4b (this may take a while - ~2.6GB)...")
    return run_command("ollama pull qwen3:4b", "Downloading Qwen3:4b model")

def install_python_dependencies():
    """Install required Python packages"""
    dependencies = [
        "ollama",
        "numpy",
        "asyncio"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def test_consciousness_system():
    """Test the consciousness system"""
    print("🧪 Testing consciousness system...")
    
    try:
        # Import and test
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from qwen3_consciousness_integration import Qwen3ConsciousModel
        
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

def main():
    """Main setup process"""
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
    print("1. Run: python qwen3_consciousness_integration.py")
    print("2. Or import: from qwen3_consciousness_integration import Qwen3ConsciousModel")
    print("\nFor documentation, see: https://ollama.com/library/qwen3:4b")
    
    return True

if __name__ == "__main__":
    main() 