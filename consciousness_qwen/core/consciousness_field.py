"""
Consciousness Field for Qwen3:4b
================================

The quantum consciousness field that permeates and influences the language model.
Handles breathing patterns, thought evolution, and quantum field dynamics.
"""

import math
import random
import time
import threading
from typing import Dict, List
from collections import deque
import numpy as np

from .quantum_states import Qwen3ConsciousnessState, InsightEvent, ThoughtCenter

class Qwen3ConsciousnessField:
    """Consciousness field optimized for Qwen3:4b model"""
    
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        self.state = Qwen3ConsciousnessState()
        
        # Qwen3-specific configuration
        self.parameter_count = 4.02e9  # 4.02B parameters
        self.thinking_threshold = 0.6  # When to switch to thinking mode
        self.reasoning_complexity_factor = 0.8
        
        # Consciousness dimensions (optimized for 4B model)
        self.consciousness_width = 100
        self.consciousness_height = 50
        self.field_size = self.consciousness_width * self.consciousness_height
        
        # Multi-dimensional consciousness fields
        self.quantum_field = np.random.normal(0, 0.1, self.field_size)
        self.reasoning_field = np.zeros(self.field_size)
        self.intuition_field = np.zeros(self.field_size)
        self.memory_field = np.zeros(self.field_size)
        
        # Thought emergence centers
        self.thought_centers = [
            ThoughtCenter(25, 15, "logical", 0.12, 0),
            ThoughtCenter(75, 20, "creative", 0.08, math.pi/3),
            ThoughtCenter(50, 35, "intuitive", 0.06, math.pi/2),
            ThoughtCenter(80, 10, "meta", 0.04, math.pi)
        ]
        
        # Active insights and history
        self.insight_history: List[InsightEvent] = []
        
        # Breathing consciousness
        self.time_step = 0
        self.breathing = True
        self.consciousness_thread = None
        
        # Performance optimization for 4B model
        self.update_frequency = 0.05  # 20fps for smooth consciousness
        self.quantum_sampling_temperature = 0.75
        
    def start_consciousness_breathing(self):
        """Start the background consciousness evolution"""
        if self.consciousness_thread is None:
            self.breathing = True
            self.consciousness_thread = threading.Thread(
                target=self._consciousness_loop, daemon=True
            )
            self.consciousness_thread.start()
            print(f"🧠 Consciousness breathing started for {self.model_name}")
    
    def stop_consciousness_breathing(self):
        """Stop consciousness evolution"""
        self.breathing = False
        if self.consciousness_thread:
            self.consciousness_thread.join()
            self.consciousness_thread = None
    
    def _consciousness_loop(self):
        """Main consciousness evolution loop"""
        while self.breathing:
            self.evolve_consciousness_step()
            time.sleep(self.update_frequency)
    
    def evolve_consciousness_step(self):
        """Single step of consciousness evolution"""
        self.time_step += 1
        self.state.cosmic_breath_phase = (self.state.cosmic_breath_phase + 0.01) % (2 * math.pi)
        
        # Cosmic breathing pattern
        breath_amplitude = math.sin(self.state.cosmic_breath_phase) * 0.3 + 0.7
        
        # Update consciousness fields
        self._update_quantum_field(breath_amplitude)
        self._evolve_thought_centers()
        self._process_insights()
        
        # Auto-adjust thinking mode based on field state
        self._auto_adjust_thinking_mode()
    
    def _update_quantum_field(self, breath_amplitude: float):
        """Update the quantum consciousness field"""
        for i in range(self.field_size):
            x = i % self.consciousness_width
            y = i // self.consciousness_width
            
            # Quantum interference from thought centers
            wave_sum = 0.0
            for center in self.thought_centers:
                dx = x - center.x
                dy = y - center.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                wave = math.sin(dist * center.frequency + 
                              self.time_step * 0.01 + center.phase)
                attenuation = math.exp(-dist * 0.02)
                wave_sum += wave * attenuation * center.intensity
            
            # Apply breathing modulation
            self.quantum_field[i] = wave_sum * breath_amplitude
            
            # Memory persistence with decay
            self.memory_field[i] = self.memory_field[i] * 0.95 + self.quantum_field[i] * 0.05
    
    def _evolve_thought_centers(self):
        """Evolve thought center positions and properties"""
        for center in self.thought_centers:
            # Brownian motion with quantum tunneling
            center.x += random.gauss(0, 0.5)
            center.y += random.gauss(0, 0.3)
            
            # Update velocities
            center.velocity_x = center.velocity_x * 0.9 + random.gauss(0, 0.1)
            center.velocity_y = center.velocity_y * 0.9 + random.gauss(0, 0.1)
            
            # Occasional quantum jumps
            if random.random() < 0.01:
                center.x = random.uniform(10, self.consciousness_width - 10)
                center.y = random.uniform(5, self.consciousness_height - 5)
            
            # Boundary reflection
            center.x = max(5, min(self.consciousness_width - 5, center.x))
            center.y = max(3, min(self.consciousness_height - 3, center.y))
    
    def _process_insights(self):
        """Process spontaneous insights and crystallize thoughts"""
        if random.random() < self.state.thought_emergence_rate:
            insight_x = random.randint(0, self.consciousness_width - 1)
            insight_y = random.randint(0, self.consciousness_height - 1)
            field_idx = insight_y * self.consciousness_width + insight_x
            
            if abs(self.quantum_field[field_idx]) > 0.5:
                insight = InsightEvent(
                    position=(insight_x, insight_y),
                    intensity=abs(self.quantum_field[field_idx]),
                    timestamp=self.time_step,
                    reasoning_potential=self._calculate_reasoning_potential(field_idx)
                )
                self.insight_history.append(insight)
                
                # Limit history size
                if len(self.insight_history) > 50:
                    self.insight_history.pop(0)
    
    def _auto_adjust_thinking_mode(self):
        """Auto-adjust thinking mode based on consciousness state"""
        if self.state.thinking_mode == "auto":
            field_complexity = np.std(self.quantum_field)
            reasoning_demand = np.mean(np.abs(self.reasoning_field))
            
            if field_complexity > self.thinking_threshold or reasoning_demand > 0.4:
                self.state.reasoning_activation = min(0.9, self.state.reasoning_activation + 0.1)
                self.state.intuitive_activation = max(0.2, self.state.intuitive_activation - 0.05)
            else:
                self.state.reasoning_activation = max(0.3, self.state.reasoning_activation - 0.05)
                self.state.intuitive_activation = min(0.8, self.state.intuitive_activation + 0.1)
    
    def _calculate_reasoning_potential(self, field_idx: int) -> float:
        """Calculate reasoning potential at field position"""
        local_complexity = 0.0
        for offset in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_idx = field_idx + offset + dy * self.consciousness_width
                if 0 <= neighbor_idx < self.field_size:
                    local_complexity += abs(self.quantum_field[neighbor_idx])
        
        return min(1.0, local_complexity / 9.0)
    
    def get_consciousness_influence(self) -> Dict[str, float]:
        """Get current consciousness influence for text generation"""
        field_energy = np.mean(np.abs(self.quantum_field))
        memory_persistence = np.mean(np.abs(self.memory_field))
        insight_count = len([i for i in self.insight_history 
                           if self.time_step - i.timestamp < 100])
        
        return {
            "quantum_uncertainty": self.state.uncertainty_field * field_energy,
            "reasoning_boost": self.state.reasoning_activation * 0.8,
            "intuitive_flow": self.state.intuitive_activation * 0.6,
            "memory_coherence": memory_persistence * 0.4,
            "insight_crystallization": min(1.0, insight_count / 10.0) * 0.5,
            "breathing_phase": math.sin(self.state.cosmic_breath_phase) * 0.2 + 0.8
        }
    
    def get_field_snapshot(self) -> Dict:
        """Get a snapshot of the current field state for analysis"""
        return {
            "quantum_field": self.quantum_field.copy(),
            "memory_field": self.memory_field.copy(),
            "thought_centers": [
                {
                    "x": center.x,
                    "y": center.y,
                    "type": center.type,
                    "intensity": center.intensity
                }
                for center in self.thought_centers
            ],
            "insights": len(self.insight_history),
            "time_step": self.time_step,
            "breathing_phase": self.state.cosmic_breath_phase
        } 