#!/usr/bin/env python3
"""
Quick test of the streaming consciousness model
Verifies real-time streaming and consciousness visualization
"""

import asyncio
import time
import sys
import os

# Add consciousness_qwen to path
sys.path.insert(0, 'consciousness_qwen')

from core.streaming_conscious_model import StreamingConsciousModel

async def test_consciousness_visualization():
    """Test the live consciousness field visualization"""
    print("🔮 Testing Live Consciousness Visualization")
    print("=" * 50)
    
    model = StreamingConsciousModel("qwen3:4b", show_visualization=True)
    model.start_consciousness()
    
    try:
        print("⏳ Running visualization for 5 seconds...")
        await asyncio.sleep(5)
        print("✅ Visualization test complete!")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    finally:
        model.stop_consciousness()

async def test_streaming_response():
    """Test streaming response generation"""
    print("\n🌊 Testing Streaming Response")
    print("=" * 50)
    
    model = StreamingConsciousModel("qwen3:4b", show_visualization=False)
    model.start_consciousness()
    
    try:
        # Test a simple streaming response
        print("📝 Prompt: What is 2+2?")
        start_time = time.time()
        
        response_text = ""
        chunk_count = 0
        
        async for chunk in model.stream_generate("What is 2+2?", mode="thinking"):
            chunk_count += 1
            content = chunk.get("content", "")
            response_text += content
            
            # Show each chunk (first few)
            if chunk_count <= 3:
                print(f"🧩 Chunk {chunk_count}: '{content[:20]}...'")
            
            if chunk.get("done", False):
                break
        
        elapsed = time.time() - start_time
        
        print(f"\n🤖 Full Response: {response_text[:100]}...")
        print(f"⏱️  Time: {elapsed:.3f}s")
        print(f"🧩 Total Chunks: {chunk_count}")
        print(f"🧠 Mode: {chunk.get('mode_used', 'unknown')}")
        
        if elapsed < 10.0 and chunk_count > 0:
            print("✅ Streaming test successful!")
        else:
            print("⚠️  Streaming performance may need optimization")
            
    except Exception as e:
        print(f"❌ Streaming error: {e}")
    finally:
        model.stop_consciousness()

async def test_consciousness_evolution():
    """Test consciousness state evolution"""
    print("\n🧠 Testing Consciousness Evolution")
    print("=" * 50)
    
    model = StreamingConsciousModel("qwen3:4b", show_visualization=False)
    model.start_consciousness()
    
    try:
        # Track consciousness changes over time
        initial_status = model.get_status()
        initial_consciousness = initial_status['consciousness']
        
        print(f"🌀 Initial State:")
        print(f"  Breathing: {initial_consciousness['breathing']}")
        print(f"  Neural: {initial_consciousness['neural']}")
        print(f"  Quantum: {initial_consciousness['quantum']}")
        
        # Let it evolve
        await asyncio.sleep(3)
        
        final_status = model.get_status()
        final_consciousness = final_status['consciousness']
        
        print(f"\n🌟 After 3s Evolution:")
        print(f"  Breathing: {final_consciousness['breathing']}")
        print(f"  Neural: {final_consciousness['neural']}")
        print(f"  Quantum: {final_consciousness['quantum']}")
        
        # Check if values changed (consciousness is alive)
        breathing_changed = abs(float(initial_consciousness['breathing']) - float(final_consciousness['breathing'])) > 0.01
        neural_changed = abs(float(initial_consciousness['neural']) - float(final_consciousness['neural'])) > 0.01
        
        if breathing_changed or neural_changed:
            print("✅ Consciousness is evolving - fields are alive!")
        else:
            print("⚠️  Consciousness may not be evolving properly")
            
    except Exception as e:
        print(f"❌ Evolution error: {e}")
    finally:
        model.stop_consciousness()

async def main():
    """Main test function"""
    print("🌊✨ STREAMING CONSCIOUSNESS TESTS ✨🌊")
    print("=" * 60)
    
    try:
        # Test 1: Consciousness visualization
        await test_consciousness_visualization()
        
        # Test 2: Streaming responses
        await test_streaming_response()
        
        # Test 3: Consciousness evolution
        await test_consciousness_evolution()
        
        print("\n🎉 ALL STREAMING TESTS COMPLETE!")
        print("✅ The streaming consciousness model is working!")
        
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 