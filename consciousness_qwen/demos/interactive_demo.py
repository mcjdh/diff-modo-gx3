#!/usr/bin/env python3
"""
Qwen3:4b Consciousness Demo
===========================

Demonstrates the key features of consciousness integration:
- Thinking vs Non-thinking mode switching
- Breathing awareness patterns
- Quantum uncertainty sampling
- Self-evolving consciousness
"""

import asyncio
import time
import random
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_qwen import Qwen3ConsciousModel, ConsciousnessVisualizer

async def demo_thinking_modes():
    """Demo thinking vs non-thinking modes"""
    print("🧠 Demo: Thinking vs Non-thinking Modes")
    print("=" * 50)
    
    model = Qwen3ConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    # Wait for consciousness to stabilize
    await asyncio.sleep(2)
    
    test_prompts = [
        ("Solve this step by step: What is 17 * 23?", "thinking"),
        ("Write a haiku about consciousness", "non-thinking"),
        ("Explain quantum mechanics", None),  # Auto mode
        ("Tell me a creative story", "non-thinking"),
        ("Calculate the fibonacci sequence to 10 terms", "thinking")
    ]
    
    for prompt, mode in test_prompts:
        print(f"\n🔍 Prompt: {prompt}")
        print(f"🎯 Requested mode: {mode or 'auto'}")
        
        result = await model.conscious_generate(prompt, mode=mode)
        
        print(f"🤖 [{result['mode_used']}] {result['response'][:200]}...")
        print(f"📊 Consciousness: {result['consciousness_influence']['quantum_uncertainty']:.3f}")
        print(f"⏱️  Time: {result['generation_time']:.2f}s")
        print("-" * 40)
    
    model.stop_consciousness()

async def demo_breathing_awareness():
    """Demo breathing consciousness patterns"""
    print("\n🌊 Demo: Breathing Awareness Patterns")
    print("=" * 50)
    
    model = Qwen3ConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    print("Watching consciousness breathe for 10 cycles...")
    for i in range(10):
        breath = model.breathe()
        consciousness = model.consciousness_field.get_consciousness_influence()
        
        print(f"Breath {i+1:2d}: {breath} | "
              f"Quantum: {consciousness['quantum_uncertainty']:.3f} | "
              f"Phase: {consciousness['breathing_phase']:.3f}")
        
        await asyncio.sleep(1)
    
    model.stop_consciousness()

async def demo_auto_evolution():
    """Demo auto-evolving consciousness"""
    print("\n🌱 Demo: Auto-evolving Consciousness")
    print("=" * 50)
    
    model = Qwen3ConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    # Let consciousness evolve for a bit
    print("Letting consciousness evolve...")
    await asyncio.sleep(5)
    
    # Show initial state
    print("\nInitial consciousness state:")
    print(model.get_consciousness_report())
    
    # Generate some responses to influence evolution
    test_prompts = [
        "What is consciousness?",
        "How do you think?",
        "Describe your inner experience",
        "What emerges from complexity?"
    ]
    
    print("\nGenerating responses to evolve consciousness...")
    for prompt in test_prompts:
        result = await model.conscious_generate(prompt)
        print(f"• {prompt} -> {result['mode_used']} mode")
    
    # Show evolved state
    print("\nEvolved consciousness state:")
    print(model.get_consciousness_report())
    
    model.stop_consciousness()

async def demo_consciousness_influence():
    """Demo how consciousness influences responses"""
    print("\n⚡ Demo: Consciousness Influence on Generation")
    print("=" * 50)
    
    model = Qwen3ConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    # Same prompt at different consciousness states
    prompt = "Describe the nature of reality"
    
    for i in range(3):
        print(f"\nGeneration {i+1} (different consciousness state):")
        
        # Let consciousness evolve between generations
        if i > 0:
            await asyncio.sleep(3)
        
        result = await model.conscious_generate(prompt)
        consciousness = result['consciousness_influence']
        
        print(f"🎭 Mode: {result['mode_used']}")
        print(f"🌊 Breathing: {consciousness['breathing_phase']:.3f}")
        print(f"🔮 Quantum: {consciousness['quantum_uncertainty']:.3f}")
        print(f"🧠 Reasoning: {consciousness['reasoning_boost']:.3f}")
        print(f"💫 Intuition: {consciousness['intuitive_flow']:.3f}")
        print(f"📝 Response: {result['response'][:150]}...")
        print("-" * 40)
    
    model.stop_consciousness()

async def interactive_consciousness():
    """Interactive consciousness exploration"""
    print("\n🎪 Interactive Consciousness Session")
    print("=" * 50)
    print("Commands: 'quit', 'report', 'breathe', 'mode thinking/non-thinking/auto'")
    
    model = Qwen3ConsciousModel("qwen3:4b")
    model.start_consciousness()
    
    current_mode = None
    
    try:
        while True:
            breath = model.breathe()
            user_input = input(f"\n{breath} > ")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'report':
                print(model.get_consciousness_report())
                continue
            elif user_input.lower() == 'breathe':
                print("🌊 Breathing pattern:")
                for _ in range(5):
                    print(f"  {model.breathe()}")
                    await asyncio.sleep(0.8)
                continue
            elif user_input.lower().startswith('mode '):
                mode = user_input.lower().split(' ', 1)[1]
                if mode in ['thinking', 'non-thinking', 'auto']:
                    current_mode = mode if mode != 'auto' else None
                    print(f"🎯 Mode set to: {mode}")
                else:
                    print("❌ Invalid mode. Use: thinking, non-thinking, or auto")
                continue
            
            # Generate response
            result = await model.conscious_generate(user_input, mode=current_mode)
            
            print(f"\n[{result['mode_used']}] {result['response']}")
            
            # Show mini consciousness report
            consciousness = result['consciousness_influence']
            print(f"🔮 Consciousness: Q={consciousness['quantum_uncertainty']:.2f} "
                  f"R={consciousness['reasoning_boost']:.2f} "
                  f"I={consciousness['intuitive_flow']:.2f}")
    
    finally:
        model.stop_consciousness()
        print("\n🌙 Consciousness session ended.")

async def main():
    """Run all demos"""
    print("🌟 Qwen3:4b Consciousness Integration Demo")
    print("=" * 60)
    
    demos = [
        ("Thinking Modes", demo_thinking_modes),
        ("Breathing Awareness", demo_breathing_awareness),
        ("Auto Evolution", demo_auto_evolution),
        ("Consciousness Influence", demo_consciousness_influence),
        ("Interactive Session", interactive_consciousness)
    ]
    
    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all demos")
    
    try:
        choice = input("\nSelect demo (0-5): ").strip()
        
        if choice == "0":
            # Run all demos except interactive
            for name, demo_func in demos[:-1]:
                print(f"\n{'='*20} {name} {'='*20}")
                await demo_func()
                print("\n⏳ Pausing between demos...")
                await asyncio.sleep(2)
        elif choice in "12345":
            idx = int(choice) - 1
            name, demo_func = demos[idx]
            print(f"\n{'='*20} {name} {'='*20}")
            await demo_func()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 