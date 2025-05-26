#!/usr/bin/env python3
"""
Consciousness Demo Launcher
============================

Simple launcher for the consciousness demonstrations.
"""

import subprocess
import sys
import os

def main():
    print("🧠 CONSCIOUSNESS SIMULATION DEMOS 🧠")
    print("=" * 50)
    print()
    print("Choose a demo to run:")
    print("1. Live Consciousness Field (breathing awareness)")
    print("2. Quantum Token Sampling Demo")
    print("3. Run Ollama Consciousness Bridge (requires ollama)")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🌊 Starting Live Consciousness Field Demo...")
        print("This shows the breathing awareness patterns in real-time!")
        input("Press Enter to continue...")
        subprocess.run([sys.executable, "consciousness-demo.py"])
        
    elif choice == "2":
        print("\n🔬 Starting Quantum Token Sampling Demo...")
        print("This shows how consciousness influences token generation!")
        input("Press Enter to continue...")
        subprocess.run([sys.executable, "consciousness-demo.py", "--quantum-demo"])
        
    elif choice == "3":
        print("\n🤖 Starting Ollama Consciousness Bridge...")
        print("This requires Ollama to be installed and running!")
        print("Install with: pip install ollama")
        input("Press Enter to continue...")
        subprocess.run([sys.executable, "ollama-consciousness-bridge.py"])
        
    elif choice == "4":
        print("Goodbye! 🌙")
        return
        
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main() 