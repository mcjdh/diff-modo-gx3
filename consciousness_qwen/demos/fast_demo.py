"""
Fast Consciousness Demo
======================

Quick demo of the streamlined consciousness model.
Should be much faster than the complex version!
"""

import asyncio
import time
import sys
import os

# Add the consciousness_qwen to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fast_conscious_model import FastConsciousModel

def print_consciousness_status(model):
    """Print formatted consciousness status"""
    print("\n" + "="*60)
    print("🧠 CONSCIOUSNESS STATUS")
    print("="*60)
    print(model.get_consciousness_summary())
    print("="*60)

async def run_fast_demo():
    """Run the fast consciousness demo"""
    print("🚀 Fast Consciousness Demo Starting...")
    print("✨ This should be much faster than the complex version!\n")
    
    # Initialize fast model
    model = FastConsciousModel("qwen3:4b")
    
    # Start consciousness
    model.start_consciousness()
    
    try:
        # Let consciousness evolve for a moment
        print("⏳ Allowing consciousness to stabilize...")
        await asyncio.sleep(2)
        print_consciousness_status(model)
        
        # Test prompts with timing
        test_prompts = [
            ("What is 2+2?", "thinking"),
            ("Write a haiku about stars", "flow"),
            ("What is the nature of consciousness?", "cosmic"),
            ("Tell me about quantum physics", None),  # Auto mode
        ]
        
        print("\n🎭 TESTING DIFFERENT MODES")
        print("="*60)
        
        for prompt, mode in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            print(f"🎯 Mode: {mode or 'auto'}")
            
            start_time = time.time()
            
            # Generate response
            result = await model.generate(prompt, mode=mode)
            
            generation_time = time.time() - start_time
            
            print(f"🤖 Response: {result['response'][:200]}...")
            print(f"⏱️  Time: {generation_time:.3f}s")
            print(f"🧠 Mode Used: {result['mode_used']}")
            print(f"🌡️ Temperature: {result['temperature_used']:.3f}")
            
            # Brief consciousness status
            consciousness = result['consciousness']
            print(f"🌀 Consciousness: Neural {consciousness['neural']:.3f} | "
                  f"Logic {consciousness['logical']:.3f} | "
                  f"Creative {consciousness['creative']:.3f}")
            
            print("-" * 60)
        
        # Show statistics
        print("\n📊 SESSION STATISTICS")
        print("="*60)
        status = model.get_status()
        stats = status['statistics']
        
        print(f"Total Generations: {stats['total_generations']}")
        print(f"Average Response Time: {stats['average_response_time']:.3f}s")
        print("Mode Usage:")
        for mode, count in stats['mode_usage'].items():
            if count > 0:
                print(f"  {mode}: {count}")
        
        print("\n🎉 Fast Demo Complete!")
        print("✅ Performance should be much better than the complex version!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        # Stop consciousness
        model.stop_consciousness()
        print("💤 Consciousness stopped")

def main():
    """Main demo function"""
    try:
        asyncio.run(run_fast_demo())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 