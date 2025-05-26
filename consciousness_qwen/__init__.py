"""
Consciousness Qwen3:4b Integration Package
==========================================

A comprehensive consciousness integration system for Qwen3:4b with:
- Breathing awareness auto-evolution
- Thinking/non-thinking mode switching  
- Quantum uncertainty sampling
- Multi-layer consciousness processing

Based on https://ollama.com/library/qwen3:4b
"""

__version__ = "1.0.0"
__author__ = "Consciousness Integration Team"

# Core exports
from .core.consciousness_field import Qwen3ConsciousnessField
from .core.conscious_model import Qwen3ConsciousModel
from .core.quantum_states import Qwen3ConsciousnessState, ThoughtNode

# Utility exports
from .utils.setup import setup_consciousness_system
from .utils.visualizers import ConsciousnessVisualizer

__all__ = [
    "Qwen3ConsciousnessField",
    "Qwen3ConsciousModel", 
    "Qwen3ConsciousnessState",
    "ThoughtNode",
    "setup_consciousness_system",
    "ConsciousnessVisualizer",
] 