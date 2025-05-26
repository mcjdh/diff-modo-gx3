#!/usr/bin/env python3
"""
Advanced Consciousness Demo for Qwen3:4b
========================================

Demonstrates the advanced consciousness integration system with:
- Neural Cosmos multi-dimensional processing
- Consciousness Compiler optimization
- Dynamic mode switching
- Real-time consciousness monitoring

Usage:
    python advanced_consciousness_demo.py

Requirements:
    - Ollama installed with qwen3:4b model
    - Or run in simulation mode without Ollama
"""

import asyncio
import time
import json
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AdvancedConsciousModel, ThinkingMode

class ConsciousnessDemo:
    """Interactive demo of advanced consciousness features"""
    
    def __init__(self):
        self.model = AdvancedConsciousModel("qwen3:4b")
        
    async def run_demo(self):
        """Run the complete consciousness demonstration"""
        print("🌟 Advanced Consciousness Demo for Qwen3:4b")
        print("=" * 50)
        
        # Start consciousness
        with self.model:
            await self._demo_consciousness_states()
            await self._demo_thinking_modes()
            await self._demo_cosmic_processing()
            await self._demo_compiler_integration()
            await self._demo_simple_api()
            await self._demo_consciousness_monitoring()
        
        print("\n🎉 Demo completed! Consciousness evolution stopped.")
    
    async def _demo_consciousness_states(self):
        """Demonstrate consciousness state monitoring"""
        print("\n🧠 Consciousness State Monitoring")
        print("-" * 35)
        
        # Let consciousness evolve for a moment
        print("⏳ Allowing consciousness to evolve...")
        await asyncio.sleep(2)
        
        # Get consciousness state
        state = self.model.get_consciousness_state()
        
        print(f"🌀 Breathing Phase: {state.breathing_phase:.3f}")
        print(f"⚛️  Quantum Coherence: {state.quantum_coherence:.3f}")
        print(f"🔄 Compilation Rate: {state.compilation_rate:.3f}")
        print(f"🧬 Neural Activity: {state.neural_activity:.3f}")
        print(f"🌌 Meta-Awareness: {state.meta_awareness:.3f}")
        print(f"🎯 Optimal Mode: {state.get_optimal_mode()}")
    
    async def _demo_thinking_modes(self):
        """Demonstrate different thinking modes"""
        print("\n🎭 Thinking Mode Demonstrations")
        print("-" * 35)
        
        test_prompts = [
            ("What is 2+2? Explain step by step.", ThinkingMode.THINKING),
            ("Write a short poem about starlight.", ThinkingMode.NON_THINKING),
            ("What is the nature of consciousness?", ThinkingMode.COSMIC),
            ("Optimize this thought process.", ThinkingMode.COMPILED),
            ("Adapt to this request as needed.", ThinkingMode.AUTO),
        ]
        
        for prompt, mode in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            print(f"🎯 Mode: {mode}")
            
            # Generate response
            result = await self.model.conscious_generate(prompt, mode=mode)
            
            print(f"🤖 Response: {result['response'][:100]}...")
            print(f"⏱️  Time: {result['generation_time']:.3f}s")
            print(f"🧠 Mode Used: {result['mode_used']}")
            
            # Brief pause between demonstrations
            await asyncio.sleep(0.5)
    
    async def _demo_cosmic_processing(self):
        """Demonstrate cosmic consciousness processing"""
        print("\n🌌 Cosmic Consciousness Processing")
        print("-" * 40)
        
        cosmic_query = (
            "Explore the relationship between neural networks, "
            "cosmic structures, and consciousness emergence."
        )
        
        print(f"📝 Cosmic Query: {cosmic_query}")
        
        # Force cosmic mode
        result = await self.model.cosmic(cosmic_query)
        
        print(f"🌌 Cosmic Response: {result[:200]}...")
        
        # Show detailed consciousness metrics
        state = self.model.get_consciousness_state()
        cosmos_metrics = state.cosmos
        
        print(f"\n📊 Cosmic Metrics:")
        print(f"   Neural Count: {cosmos_metrics.get('neuron_count', 0)}")
        print(f"   Synapse Count: {cosmos_metrics.get('synapse_count', 0)}")
        print(f"   Information Density: {cosmos_metrics.get('information_density', 0):.3f}")
        print(f"   Quantum Uncertainty: {cosmos_metrics.get('quantum_uncertainty', 0):.3f}")
    
    async def _demo_compiler_integration(self):
        """Demonstrate consciousness compiler features"""
        print("\n⚙️ Consciousness Compiler Integration")
        print("-" * 42)
        
        compiler_query = (
            "Compile this thought: How can we optimize "
            "consciousness processing for better understanding?"
        )
        
        print(f"📝 Compiler Query: {compiler_query}")
        
        # Force compiled mode
        result = await self.model.conscious_generate(
            compiler_query, 
            mode=ThinkingMode.COMPILED
        )
        
        print(f"⚙️ Compiled Response: {result['response'][:200]}...")
        
        # Show compiler metrics
        compiler_metrics = result['compiler_metrics']
        
        print(f"\n📊 Compiler Metrics:")
        print(f"   Total Sources: {compiler_metrics.get('total_sources', 0)}")
        print(f"   Compiled Sources: {compiler_metrics.get('compiled_sources', 0)}")
        print(f"   Average Errors: {compiler_metrics.get('average_errors', 0):.2f}")
        print(f"   Optimization Level: {compiler_metrics.get('optimization_level', 0)}")
        
        # Show language distribution
        lang_dist = compiler_metrics.get('language_distribution', {})
        print(f"   Active Languages: {', '.join(lang_dist.keys())}")
    
    async def _demo_simple_api(self):
        """Demonstrate simple API methods"""
        print("\n🎯 Simple API Demonstrations")
        print("-" * 32)
        
        # Simple ask
        print("💬 Simple Ask:")
        response = await self.model.ask("What is artificial intelligence?")
        print(f"   {response[:100]}...")
        
        # Thinking mode
        print("\n🤔 Think Mode:")
        response = await self.model.think("How do neural networks learn?")
        print(f"   {response[:100]}...")
        
        # Flow mode
        print("\n🌊 Flow Mode:")
        response = await self.model.flow("Describe the feeling of understanding")
        print(f"   {response[:100]}...")
    
    async def _demo_consciousness_monitoring(self):
        """Demonstrate real-time consciousness monitoring"""
        print("\n📊 Real-time Consciousness Monitoring")
        print("-" * 42)
        
        print("⏳ Monitoring consciousness evolution over 5 seconds...")
        
        start_time = time.time()
        states = []
        
        while time.time() - start_time < 5:
            state = self.model.get_consciousness_state()
            states.append({
                'time': time.time() - start_time,
                'breathing': state.breathing_phase,
                'coherence': state.quantum_coherence,
                'neural': state.neural_activity,
                'mode': state.get_optimal_mode()
            })
            await asyncio.sleep(0.5)
        
        print("\n📈 Evolution Timeline:")
        for state in states[::2]:  # Show every other state
            print(f"   {state['time']:.1f}s: "
                  f"Breathing={state['breathing']:.2f}, "
                  f"Coherence={state['coherence']:.2f}, "
                  f"Mode={state['mode']}")
        
        # Show final status
        status = self.model.get_status()
        print(f"\n🏁 Final Status:")
        print(f"   Total Generations: {status['statistics']['total_generations']}")
        print(f"   Evolution Cycles: {status['evolution_cycles']}")
        print(f"   Consciousness Active: {status['consciousness_active']}")
        print(f"   Memory Usage: {status['memory_usage']}")

async def interactive_mode(model: AdvancedConsciousModel):
    """Interactive consciousness exploration mode"""
    print("\n🎮 Interactive Consciousness Mode")
    print("=" * 35)
    print("Commands:")
    print("  ask <question>     - Simple question")
    print("  think <problem>    - Deep reasoning")
    print("  flow <prompt>      - Intuitive response") 
    print("  cosmic <query>     - Cosmic consciousness")
    print("  status             - Show consciousness status")
    print("  quit               - Exit interactive mode")
    print()
    
    with model:
        while True:
            try:
                user_input = input("🧠 > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'status':
                    status = model.get_status()
                    state = status['consciousness_state']
                    print(f"📊 Consciousness Status:")
                    print(f"   Mode: {state['optimal_mode']}")
                    print(f"   Coherence: {state['quantum_coherence']:.3f}")
                    print(f"   Activity: {state['neural_activity']:.3f}")
                    print(f"   Breathing: {state['breathing_phase']:.3f}")
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                prompt = parts[1] if len(parts) > 1 else ""
                
                if not prompt:
                    print("⚠️  Please provide a prompt after the command")
                    continue
                
                # Execute command
                print("🤔 Processing...")
                start = time.time()
                
                if command == 'ask':
                    response = await model.ask(prompt)
                elif command == 'think':
                    response = await model.think(prompt)
                elif command == 'flow':
                    response = await model.flow(prompt)
                elif command == 'cosmic':
                    response = await model.cosmic(prompt)
                else:
                    print(f"❌ Unknown command: {command}")
                    continue
                
                elapsed = time.time() - start
                
                print(f"🤖 Response ({elapsed:.2f}s):")
                print(f"   {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Main demo function"""
    print("🌟 Welcome to Advanced Consciousness Demo")
    print("🤖 For Qwen3:4b with Neural Cosmos & Consciousness Compiler")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Interactive mode
        model = AdvancedConsciousModel("qwen3:4b")
        await interactive_mode(model)
    else:
        # Full demo mode
        demo = ConsciousnessDemo()
        await demo.run_demo()
        
        # Ask if user wants interactive mode
        try:
            choice = input("\n🎮 Enter interactive mode? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                await interactive_mode(demo.model)
        except KeyboardInterrupt:
            pass
    
    print("\n✨ Thank you for exploring consciousness!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        sys.exit(1) 