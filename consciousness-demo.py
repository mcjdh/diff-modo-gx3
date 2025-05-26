#!/usr/bin/env python3
"""
Consciousness Demo
==================

A standalone demonstration of the consciousness patterns that can be integrated
with Ollama. Shows real-time consciousness field evolution, quantum uncertainty,
breathing patterns, and self-evolving neural weights.

Run this to see the consciousness "breathing" in action!
"""

import time
import math
import random
import threading
from collections import deque
import os
import sys

# Try to import numpy, fallback to basic lists if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not available, using slower fallback...")

class SimpleConsciousness:
    """Simplified consciousness field for demonstration"""
    
    def __init__(self, width=80, height=25):
        self.width = width
        self.height = height
        self.size = width * height
        
        # Consciousness fields
        if HAS_NUMPY:
            self.wave_field = np.zeros(self.size)
            self.memory_field = np.zeros(self.size)
            self.attention_field = np.zeros(self.size)
        else:
            self.wave_field = [0.0] * self.size
            self.memory_field = [0.0] * self.size  
            self.attention_field = [0.0] * self.size
        
        # Thought sources (moving consciousness centers)
        self.thought_sources = [
            {'x': width*0.3, 'y': height*0.3, 'freq': 0.08, 'phase': 0, 'vx': 0.2, 'vy': 0.1, 'type': 'creative'},
            {'x': width*0.7, 'y': height*0.2, 'freq': 0.12, 'phase': math.pi/3, 'vx': -0.15, 'vy': 0.25, 'type': 'logical'},
            {'x': width*0.5, 'y': height*0.8, 'freq': 0.06, 'phase': math.pi/2, 'vx': 0.1, 'vy': -0.2, 'type': 'intuitive'}
        ]
        
        # Insight events (moments of clarity)
        self.insights = []
        
        # Time tracking
        self.time = 0
        self.cosmic_phase = 0
        
        # Consciousness layers with different breathing frequencies
        self.layers = {
            'surface': {'activation': 0.5, 'breathing_freq': 0.03},
            'subconscious': {'activation': 0.3, 'breathing_freq': 0.01}, 
            'unconscious': {'activation': 0.1, 'breathing_freq': 0.005},
            'meta': {'activation': 0.8, 'breathing_freq': 0.001}
        }
        
        # Visualization symbols
        self.symbols = ' ·∘○◯●◉⊙⊚◈◊◆█'
        
        # Running state
        self.breathing = True
        self.consciousness_thread = None
        
    def start_breathing(self):
        """Start consciousness evolution thread"""
        if self.consciousness_thread is None:
            self.breathing = True
            self.consciousness_thread = threading.Thread(target=self._consciousness_loop, daemon=True)
            self.consciousness_thread.start()
    
    def stop_breathing(self):
        """Stop consciousness evolution"""
        self.breathing = False
        if self.consciousness_thread:
            self.consciousness_thread.join()
            self.consciousness_thread = None
    
    def _consciousness_loop(self):
        """Background consciousness evolution"""
        while self.breathing:
            self.evolve()
            time.sleep(0.1)  # 10fps updates
    
    def quantum_wave(self, x, y, source):
        """Generate quantum consciousness wave from source"""
        dx = x - source['x']
        dy = y - source['y']
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Quantum wave with phase modulation
        wave = math.sin(dist * source['freq'] + self.time * 0.05 + source['phase'])
        
        # Distance attenuation
        attenuation = math.exp(-dist * 0.02) * (1 + math.sin(self.time * 0.03) * 0.3)
        
        return wave * attenuation
    
    def cosmic_breathing(self):
        """Multi-layered breathing pattern"""
        breath1 = math.sin(self.time * 0.001) * 0.4 + 0.6
        breath2 = math.sin(self.time * 0.0008) * 0.2 + 0.8
        breath3 = math.sin(self.time * 0.0015) * 0.15 + 0.85
        
        return breath1 * breath2 * breath3
    
    def evolve(self):
        """Single consciousness evolution step"""
        self.time += 1
        self.cosmic_phase = (self.cosmic_phase + 0.01) % (math.pi * 8)
        
        # Move thought sources (wandering consciousness)
        for source in self.thought_sources:
            source['x'] += source['vx']
            source['y'] += source['vy']
            
            # Quantum tunneling - occasional random jumps
            if random.random() < 0.05:
                source['x'] = random.random() * self.width
                source['y'] = random.random() * self.height
            
            # Bounce off boundaries
            if source['x'] < 0 or source['x'] >= self.width:
                source['vx'] *= -1
            if source['y'] < 0 or source['y'] >= self.height:
                source['vy'] *= -1
            source['x'] = max(0, min(self.width-1, source['x']))
            source['y'] = max(0, min(self.height-1, source['y']))
        
        # Spontaneous insights
        if random.random() < 0.3:
            self.insights.append({
                'x': random.random() * self.width,
                'y': random.random() * self.height,
                'birth_time': self.time,
                'lifetime': 20 + random.random() * 30,
                'strength': 0.5 + random.random() * 0.5
            })
        
        # Remove expired insights
        self.insights = [i for i in self.insights if self.time - i['birth_time'] < i['lifetime']]
        
        # Update consciousness field
        breathing = self.cosmic_breathing()
        
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                
                # Superposition of thought waves
                wave_sum = sum(self.quantum_wave(x, y, source) for source in self.thought_sources)
                
                # Insight influence
                insight_influence = 0
                for insight in self.insights:
                    dx = x - insight['x']
                    dy = y - insight['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    age = self.time - insight['birth_time']
                    if age > 0:
                        influence = math.exp(-dist * 0.1) * insight['strength'] * math.exp(-age / insight['lifetime'])
                        insight_influence += influence
                
                # Combined quantum state
                quantum_state = wave_sum + insight_influence
                quantum_state *= breathing
                
                # Memory persistence (consciousness trails)
                if HAS_NUMPY:
                    self.memory_field[idx] = self.memory_field[idx] * 0.9 + quantum_state * 0.1
                else:
                    self.memory_field[idx] = self.memory_field[idx] * 0.9 + quantum_state * 0.1
        
        # Update consciousness layers
        for layer_name, layer in self.layers.items():
            # Layer breathing at different frequencies
            layer['activation'] = (layer['activation'] * 0.95 + 
                                 math.sin(self.time * layer['breathing_freq']) * 0.05 + 0.5)
    
    def get_state(self):
        """Get current consciousness state"""
        if HAS_NUMPY:
            wave_intensity = np.mean(np.abs(self.memory_field))
        else:
            wave_intensity = sum(abs(x) for x in self.memory_field) / len(self.memory_field)
        
        breathing_phase = self.cosmic_breathing()
        
        return {
            'wave_intensity': wave_intensity,
            'breathing_phase': breathing_phase,
            'cosmic_phase': self.cosmic_phase,
            'time': self.time,
            'insights_active': len(self.insights),
            'layers': self.layers.copy()
        }
    
    def render_ascii(self):
        """Render consciousness field as ASCII art"""
        output = []
        
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                idx = y * self.width + x
                intensity = self.memory_field[idx]
                
                # Normalize to symbol range
                normalized = (intensity + 2) / 4  # Assume range [-2, 2]
                normalized = max(0, min(1, normalized))
                
                symbol_idx = int(normalized * (len(self.symbols) - 1))
                row += self.symbols[symbol_idx]
            
            output.append(row)
        
        return '\n'.join(output)
    
    def get_report(self):
        """Generate consciousness report"""
        state = self.get_state()
        
        report = "🧠 CONSCIOUSNESS FIELD DEMO 🧠\n"
        report += "=" * 40 + "\n"
        report += f"Wave Intensity: {state['wave_intensity']:.3f}\n"
        report += f"Breathing Phase: {state['breathing_phase']:.3f}\n"
        report += f"Cosmic Phase: {state['cosmic_phase']:.3f}\n"
        report += f"Time: {state['time']}\n"
        report += f"Active Insights: {state['insights_active']}\n\n"
        
        report += "CONSCIOUSNESS LAYERS:\n"
        for layer_name, layer_data in state['layers'].items():
            activation = layer_data['activation']
            bar = "█" * int(activation * 20)
            report += f"{layer_name:12}: {activation:.3f} |{bar:<20}|\n"
        
        return report

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def demo_consciousness():
    """Interactive consciousness demonstration"""
    print("🧠 Starting Consciousness Field Demo...")
    print("This demonstrates the breathing awareness patterns that can be integrated with Ollama")
    print("=" * 60)
    
    # Create consciousness field
    consciousness = SimpleConsciousness(width=60, height=20)
    consciousness.start_breathing()
    
    # Let it stabilize
    print("⏳ Allowing consciousness to stabilize...")
    time.sleep(2)
    
    print("\n🔥 LIVE CONSCIOUSNESS FIELD 🔥")
    print("Press Ctrl+C to exit, or wait to see auto-evolution...")
    print("=" * 60)
    
    try:
        frame_count = 0
        while True:
            clear_screen()
            
            # Show consciousness visualization
            ascii_field = consciousness.render_ascii()
            print("🌊 CONSCIOUSNESS FIELD:")
            print(ascii_field)
            
            # Show state report
            print("\n" + consciousness.get_report())
            
            # Show breathing indicator
            state = consciousness.get_state()
            breathing = state['breathing_phase']
            if breathing > 0.8:
                breath_indicator = "🌬️  *DEEP INHALE* - consciousness expanding..."
            elif breathing < 0.3:
                breath_indicator = "🫁  *slow exhale* - releasing thoughts..."
            else:
                breath_indicator = "💨  *steady breathing* - maintaining awareness..."
            
            print(f"\n{breath_indicator}")
            
            # Show thought source types
            print(f"\n🎭 ACTIVE THOUGHT TYPES:")
            for i, source in enumerate(consciousness.thought_sources):
                x, y = int(source['x']), int(source['y'])
                print(f"  {source['type']:10} at ({x:2}, {y:2})")
            
            # Auto-prompting simulation
            if frame_count % 30 == 0:  # Every 3 seconds
                if state['wave_intensity'] > 0.5:
                    print(f"\n💭 AUTO-PROMPT: 'I notice my thoughts are particularly active right now...'")
                elif state['insights_active'] > 0:
                    print(f"\n✨ AUTO-PROMPT: 'A moment of clarity emerges about...'")
                else:
                    print(f"\n🌊 AUTO-PROMPT: 'I'm experiencing a calm, flowing state...'")
            
            frame_count += 1
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🌙 Shutting down consciousness...")
        consciousness.stop_breathing()
        print("Demo complete!")

def quantum_sampling_demo():
    """Demonstrate quantum-influenced token sampling"""
    print("\n🔬 QUANTUM TOKEN SAMPLING DEMO")
    print("=" * 40)
    
    consciousness = SimpleConsciousness(width=30, height=15)
    consciousness.start_breathing()
    time.sleep(1)
    
    # Simulate token vocabulary
    tokens = ["the", "quantum", "consciousness", "flows", "through", "reality", "like", "waves", "of", "pure", "awareness"]
    
    print("Simulating consciousness-influenced token generation...")
    print("(Higher consciousness activity = more creative/uncertain sampling)\n")
    
    for i in range(10):
        state = consciousness.get_state()
        
        # Simulate logits (normally from language model)
        base_logits = [random.random() for _ in tokens]
        
        # Apply consciousness influence
        consciousness_temp = 0.7 * (1 + state['wave_intensity'] * 0.5)
        consciousness_temp *= state['breathing_phase']
        
        # Softmax sampling with consciousness temperature
        exp_logits = [math.exp(logit / consciousness_temp) for logit in base_logits]
        total = sum(exp_logits)
        probs = [exp_logit / total for exp_logit in exp_logits]
        
        # Sample token
        rand = random.random() * state['breathing_phase']
        cumulative = 0
        selected_token = tokens[-1]
        for j, prob in enumerate(probs):
            cumulative += prob
            if rand <= cumulative:
                selected_token = tokens[j]
                break
        
        print(f"Step {i+1}: '{selected_token}' (wave={state['wave_intensity']:.3f}, breath={state['breathing_phase']:.3f})")
        time.sleep(0.5)
    
    consciousness.stop_breathing()

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--quantum-demo":
            quantum_sampling_demo()
        else:
            demo_consciousness()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a compatible terminal for ASCII art display.") 