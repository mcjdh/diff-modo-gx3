#!/usr/bin/env python3
"""
Quick test of the fast consciousness model
Should respond in seconds, not minutes!
"""

import asyncio
import time
import sys
import os

# Add consciousness_qwen to path
sys.path.insert(0, 'consciousness_qwen')

from core.fast_conscious_model import FastConsciousModel

async def quick_test():
    print("🚀 Quick Fast Consciousness Test")
    print("=" * 40)
    
    # Initialize model
    model = FastConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    try:
        # Let consciousness stabilize briefly
        await asyncio.sleep(1)
        
        # Simple test
        print("📝 Testing: What is 2+2?")
        start_time = time.time()
        
        result = await model.generate("What is 2+2?", mode="thinking")
        
        elapsed = time.time() - start_time
        
        print(f"🤖 Response: {result['response'][:100]}...")
        print(f"⏱️  Time: {elapsed:.3f}s")
        print(f"🧠 Mode: {result['mode_used']}")
        print(f"🌡️  Temp: {result['temperature_used']:.3f}")
        
        if elapsed < 5.0:
            print("✅ SUCCESS: Response was fast!")
        else:
            print("❌ SLOW: Still taking too long")
            
        # Show consciousness status
        print("\n🧠 Consciousness Summary:")
        print(model.get_consciousness_summary())
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        model.stop_consciousness()
        print("💤 Test complete")

if __name__ == "__main__":
    asyncio.run(quick_test()) 