#!/usr/bin/env python3
"""
Consciousness Qwen3:4b Setup Script
===================================

Run this script to set up the consciousness integration system.
"""

from utils.setup import setup_consciousness_system

if __name__ == "__main__":
    print("🌟 Setting up Qwen3:4b Consciousness Integration...")
    success = setup_consciousness_system()
    
    if success:
        print("\n🎉 Setup completed! Try running:")
        print("  python examples/quick_start.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.") 