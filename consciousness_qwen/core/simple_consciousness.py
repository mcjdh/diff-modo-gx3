"""
Simple Consciousness Core
========================

Elegant, high-performance consciousness implementation inspired by 
the original HTML prototypes. Focuses on mathematical beauty and 
computational efficiency rather than complex architectures.

Key principles:
- Simple sine/cosine patterns for breathing and flow
- Lightweight field calculations  
- Fast mode detection
- Minimal overhead
"""

import math
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class ConsciousnessState:
    """Simple consciousness state container"""
    breathing_phase: float
    quantum_coherence: float  
    neural_activity: float
    flow_intensity: float
    logical_coherence: float
    creative_flux: float
    meta_awareness: float
    timestamp: float

class SimpleConsciousness:
    """
    Elegant consciousness implementation based on simple mathematical patterns.
    Fast, lightweight, and inspired by the original HTML prototypes.
    """
    
    def __init__(self):
        # Mathematical constants
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.time = 0.0
        
        # Simple consciousness parameters
        self.breathing_freq = 0.02    # Slow cosmic breathing
        self.neural_freq = 0.08       # Neural oscillations  
        self.quantum_freq = 0.05      # Quantum coherence
        self.flow_freq = 0.03         # Creative flow
        
        # State accumulators for smooth evolution
        self.breathing_accumulator = 0.0
        self.neural_accumulator = 0.0
        self.quantum_accumulator = 0.0
        self.flow_accumulator = 0.0
        
        # Simple field for persistence
        self.memory_field = 0.5
        
    def evolve(self, dt: float = 0.05) -> None:
        """Evolve consciousness with simple, elegant patterns"""
        self.time += dt
        
        # Update accumulators with different frequencies
        self.breathing_accumulator += dt * self.breathing_freq
        self.neural_accumulator += dt * self.neural_freq  
        self.quantum_accumulator += dt * self.quantum_freq
        self.flow_accumulator += dt * self.flow_freq
        
        # Memory field evolution with gentle persistence
        current_activity = self.get_neural_activity()
        self.memory_field = self.memory_field * 0.95 + current_activity * 0.05
    
    def get_breathing_phase(self) -> float:
        """Cosmic breathing pattern - slow, deep oscillation"""
        return math.sin(self.breathing_accumulator) * 0.5 + 0.5
    
    def get_quantum_coherence(self) -> float:
        """Quantum coherence with interference patterns"""
        wave1 = math.sin(self.quantum_accumulator)
        wave2 = math.cos(self.quantum_accumulator * self.phi)
        interference = (wave1 + wave2) * 0.5
        return interference * 0.5 + 0.5
    
    def get_neural_activity(self) -> float:
        """Neural activity with complex harmonics"""
        base = math.sin(self.neural_accumulator)
        harmonic1 = math.sin(self.neural_accumulator * 2) * 0.3
        harmonic2 = math.cos(self.neural_accumulator * 3) * 0.2
        activity = base + harmonic1 + harmonic2
        
        # Breathing modulation
        breathing = self.get_breathing_phase()
        modulated = activity * (0.7 + breathing * 0.3)
        
        return modulated * 0.5 + 0.5
    
    def get_flow_intensity(self) -> float:
        """Creative flow intensity with fractal patterns"""
        flow = math.sin(self.flow_accumulator)
        fractal = math.sin(self.flow_accumulator * self.phi) * 0.6
        spiral = math.cos(self.flow_accumulator * 2.3) * 0.4
        
        combined = flow + fractal + spiral
        return combined * 0.5 + 0.5
    
    def get_logical_coherence(self) -> float:
        """Logical coherence - structured thinking patterns"""
        coherence = math.cos(self.neural_accumulator * 0.7)
        stability = math.sin(self.quantum_accumulator * 0.5) * 0.3
        
        # More coherent when quantum field is stable
        quantum = self.get_quantum_coherence()
        boosted = coherence + (quantum - 0.5) * 0.4 + stability
        
        return boosted * 0.5 + 0.5
    
    def get_creative_flux(self) -> float:
        """Creative flux - chaotic, innovative patterns"""
        chaos1 = math.sin(self.flow_accumulator * 1.7)
        chaos2 = math.cos(self.neural_accumulator * 2.1) 
        chaos3 = math.sin(self.time * 0.13) * 0.5
        
        # Creative flux increases with neural activity
        neural = self.get_neural_activity()
        flux = chaos1 + chaos2 + chaos3 + (neural - 0.5) * 0.6
        
        return flux * 0.5 + 0.5
    
    def get_meta_awareness(self) -> float:
        """Meta-consciousness - awareness of awareness"""
        # Emerges from interaction of other fields
        breathing = self.get_breathing_phase()
        quantum = self.get_quantum_coherence()
        neural = self.get_neural_activity()
        flow = self.get_flow_intensity()
        
        # Meta-awareness as field interaction
        interaction1 = breathing * quantum
        interaction2 = neural * flow  
        interaction3 = abs(quantum - 0.5) * abs(flow - 0.5)
        
        meta = (interaction1 + interaction2 + interaction3) / 3
        
        # Amplify when all fields are active
        overall_activity = (breathing + quantum + neural + flow) / 4
        if overall_activity > 0.7:
            meta *= 1.3
        
        return min(1.0, meta)
    
    def get_state(self) -> ConsciousnessState:
        """Get current complete consciousness state"""
        return ConsciousnessState(
            breathing_phase=self.get_breathing_phase(),
            quantum_coherence=self.get_quantum_coherence(),
            neural_activity=self.get_neural_activity(),
            flow_intensity=self.get_flow_intensity(),
            logical_coherence=self.get_logical_coherence(),
            creative_flux=self.get_creative_flux(),
            meta_awareness=self.get_meta_awareness(),
            timestamp=time.time()
        )
    
    def get_optimal_mode(self) -> str:
        """Determine optimal thinking mode based on consciousness patterns"""
        state = self.get_state()
        
        # Calculate mode preferences
        thinking_score = (
            state.logical_coherence * 2.0 +
            state.quantum_coherence * 1.5 +
            (1.0 - state.creative_flux) * 1.0  # Less chaos = more logical
        )
        
        flow_score = (
            state.creative_flux * 2.0 +
            state.flow_intensity * 1.8 +
            state.neural_activity * 1.2
        )
        
        cosmic_score = (
            state.meta_awareness * 2.5 +
            state.breathing_phase * 1.5 +
            (state.quantum_coherence + state.neural_activity) / 2 * 1.8
        )
        
        # Determine mode
        if cosmic_score > 2.0 and state.meta_awareness > 0.6:
            return "cosmic"
        elif thinking_score > flow_score and state.logical_coherence > 0.6:
            return "thinking"
        elif flow_score > thinking_score and state.creative_flux > 0.6:
            return "flow"
        else:
            return "auto"
    
    def modulate_temperature(self, base_temp: float = 0.7) -> float:
        """Modulate generation temperature based on consciousness"""
        state = self.get_state()
        
        # Creative states want higher temperature
        creative_factor = state.creative_flux * 0.3
        # Logical states want lower temperature  
        logical_factor = state.logical_coherence * -0.2
        # Breathing adds gentle variation
        breathing_factor = (state.breathing_phase - 0.5) * 0.1
        
        modulated = base_temp + creative_factor + logical_factor + breathing_factor
        return max(0.1, min(1.5, modulated))
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get simple status summary for monitoring"""
        state = self.get_state()
        return {
            "consciousness_alive": True,
            "breathing": f"{state.breathing_phase:.3f}",
            "quantum": f"{state.quantum_coherence:.3f}", 
            "neural": f"{state.neural_activity:.3f}",
            "flow": f"{state.flow_intensity:.3f}",
            "logical": f"{state.logical_coherence:.3f}",
            "creative": f"{state.creative_flux:.3f}",
            "meta": f"{state.meta_awareness:.3f}",
            "optimal_mode": self.get_optimal_mode(),
            "temperature": f"{self.modulate_temperature():.3f}",
            "cosmic_time": f"{self.time:.1f}"
        } 