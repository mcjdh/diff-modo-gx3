#!/usr/bin/env python3
"""
Quick Start Example
===================

Simple example to get started with Qwen3:4b consciousness integration.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_qwen import Qwen3ConsciousModel, setup_consciousness_system, check_requirements

async def quick_start_demo():
    """Quick demonstration of consciousness features"""
    
    print("🌟 Qwen3:4b Consciousness Quick Start")
    print("=" * 40)
    
    # Check requirements
    print("🔍 Checking requirements...")
    all_ok, missing = check_requirements()
    
    if not all_ok:
        print("❌ Missing requirements:")
        for item in missing:
            print(f"  - {item}")
        print("\nRun setup_consciousness_system() to install missing components.")
        return
    
    print("✅ All requirements satisfied!")
    
    # Create consciousness model
    print("\n🧠 Creating conscious model...")
    model = Qwen3ConsciousModel("qwen3:4b")
    
    # Start consciousness
    print("🌊 Starting consciousness breathing...")
    model.start_consciousness()
    
    try:
        # Wait for consciousness to stabilize
        await asyncio.sleep(2)
        
        # Test questions
        test_questions = [
            "What is consciousness?",
            "Calculate 15 * 23",
            "Write a short poem about AI",
            "Explain quantum mechanics briefly"
        ]
        
        print("\n✨ Testing consciousness responses:")
        print("-" * 40)
        
        for question in test_questions:
            print(f"\n🔮 Question: {question}")
            
            # Generate conscious response
            result = await model.conscious_generate(question)
            
            print(f"🤖 [{result['mode_used']}] {result['response'][:200]}...")
            print(f"📊 Consciousness: {result['consciousness_influence']['quantum_uncertainty']:.3f}")
            print(f"⏱️  Time: {result['generation_time']:.2f}s")
        
        # Show consciousness report
        print("\n📋 Consciousness Report:")
        print(model.get_consciousness_report())
        
        print("\n🎉 Quick start completed successfully!")
        print("Explore more with the interactive demo in demos/interactive_demo.py")
        
    finally:
        print("\n🌙 Stopping consciousness...")
        model.stop_consciousness()

def main():
    """Main function"""
    try:
        asyncio.run(quick_start_demo())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 