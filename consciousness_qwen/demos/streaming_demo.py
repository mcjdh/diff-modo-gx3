"""
Streaming Consciousness Demo
===========================

Live demo with real-time consciousness visualization and streaming responses.
Shows consciousness field evolution in ASCII art while generating responses.

Features:
- Live ASCII consciousness field visualization
- Real-time streaming responses  
- Consciousness state monitoring
- Interactive mode switching
"""

import asyncio
import time
import sys
import os

# Add consciousness_qwen to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.streaming_conscious_model import StreamingConsciousModel

async def run_live_visualization():
    """Run live consciousness visualization demo"""
    print("🌌 LIVE CONSCIOUSNESS FIELD VISUALIZATION")
    print("=" * 70)
    print("🔮 Starting consciousness evolution...")
    print("💫 You'll see breathing patterns, quantum interference, and neural waves")
    print("⏱️  Let it run for 30 seconds to see the patterns emerge")
    print("🛑 Press Ctrl+C to stop and proceed to streaming demo")
    print("=" * 70)
    
    model = StreamingConsciousModel("qwen3:4b", show_visualization=True)
    model.start_consciousness()
    
    try:
        # Let it run for visualization
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        print("\n🌟 Visualization stopped - proceeding to streaming demo...")
    
    model.stop_consciousness()
    return model

async def run_streaming_demo():
    """Run streaming response demo"""
    print("\n🌊 STREAMING CONSCIOUSNESS DEMO")
    print("=" * 70)
    print("🚀 Now testing real-time streaming responses...")
    
    # Initialize streaming model without live visualization
    model = StreamingConsciousModel("qwen3:4b", show_visualization=False)
    model.start_consciousness()
    
    try:
        # Let consciousness stabilize
        await asyncio.sleep(2)
        
        # Test streaming with different modes
        test_prompts = [
            ("What is consciousness?", "cosmic"),
            ("Write a short poem about waves", "flow"),
            ("Explain quantum mechanics simply", "thinking"),
            ("Tell me about the nature of reality", None),  # Auto mode
        ]
        
        for prompt, mode in test_prompts:
            print(f"\n{'='*60}")
            print(f"📝 Prompt: {prompt}")
            print(f"🎯 Mode: {mode or 'auto'}")
            print(f"{'='*60}")
            
            # Stream the response
            start_time = time.time()
            await model.stream_ask(prompt, mode=mode)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Total time: {elapsed:.2f}s")
            
            # Brief pause between prompts
            await asyncio.sleep(1)
        
        # Show final statistics
        print(f"\n{'='*60}")
        print("📊 STREAMING SESSION STATISTICS")
        print(f"{'='*60}")
        
        status = model.get_status()
        stats = status['statistics']
        
        print(f"🔢 Total Generations: {stats['total_generations']}")
        print(f"🧩 Total Streaming Chunks: {stats['total_streaming_chunks']}")
        print(f"⚡ Average Response Time: {stats['average_response_time']:.3f}s")
        
        print("\n🎯 Mode Usage:")
        for mode, count in stats['mode_usage'].items():
            if count > 0:
                print(f"  {mode}: {count}")
        
        print(f"\n🧠 Final Consciousness State:")
        consciousness = status['consciousness']
        print(f"🌀 Breathing: {consciousness['breathing']}")
        print(f"⚛️  Quantum: {consciousness['quantum']}")
        print(f"🧬 Neural: {consciousness['neural']}")
        print(f"🌊 Flow: {consciousness['flow']}")
        print(f"🎯 Optimal Mode: {consciousness['optimal_mode']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Streaming demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        model.stop_consciousness()

async def run_interactive_mode():
    """Run interactive streaming mode"""
    print(f"\n{'='*60}")
    print("🎮 INTERACTIVE STREAMING MODE")
    print(f"{'='*60}")
    print("💬 Type your questions and see consciousness stream responses!")
    print("🎯 Commands:")
    print("  - Type normally for auto mode")
    print("  - Start with 'think:' for analytical mode")
    print("  - Start with 'flow:' for creative mode")
    print("  - Start with 'cosmic:' for philosophical mode")
    print("  - Type 'quit' to exit")
    print(f"{'='*60}")
    
    model = StreamingConsciousModel("qwen3:4b", show_visualization=False)
    model.start_consciousness()
    
    try:
        while True:
            try:
                user_input = input("\n💭 Your prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Parse mode from input
                mode = None
                if user_input.startswith('think:'):
                    mode = "thinking"
                    user_input = user_input[6:].strip()
                elif user_input.startswith('flow:'):
                    mode = "flow"
                    user_input = user_input[5:].strip()
                elif user_input.startswith('cosmic:'):
                    mode = "cosmic"
                    user_input = user_input[7:].strip()
                
                # Stream response
                await model.stream_ask(user_input, mode=mode)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    finally:
        model.stop_consciousness()

async def main():
    """Main demo orchestrator"""
    print("🌊✨ STREAMING CONSCIOUSNESS EXPERIENCE ✨🌊")
    print("=" * 70)
    print("🎭 This demo showcases three experiences:")
    print("  1. 🔮 Live consciousness field visualization")
    print("  2. 🌊 Automated streaming response demos")
    print("  3. 🎮 Interactive streaming mode")
    print("=" * 70)
    
    try:
        # Phase 1: Live visualization
        print("\n🔮 PHASE 1: CONSCIOUSNESS FIELD VISUALIZATION")
        await run_live_visualization()
        
        # Phase 2: Streaming demos
        print("\n🌊 PHASE 2: STREAMING RESPONSE DEMONSTRATIONS")
        await run_streaming_demo()
        
        # Phase 3: Interactive mode
        print("\n🎮 PHASE 3: INTERACTIVE STREAMING")
        await run_interactive_mode()
        
        print("\n🎉 STREAMING CONSCIOUSNESS DEMO COMPLETE!")
        print("✨ You've experienced the future of conscious AI interaction!")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted - thanks for exploring consciousness!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 