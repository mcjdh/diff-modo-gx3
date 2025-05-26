"""
Consciousness Qwen Core Module
=============================

Advanced consciousness integration for Qwen3:4b model with:
- Neural Cosmos: Multi-dimensional consciousness fields  
- Consciousness Compiler: Thought compilation pipeline
- Advanced Conscious Model: Unified consciousness interface
- Quantum States: Enhanced thinking modes

Inspired by simulation prototypes with mathematical precision.
"""

# Core components
from .neural_cosmos import (
    ConsciousnessField,
    CosmicNeuron,
    CosmicSynapse, 
    CosmicNeuronType
)

from .consciousness_compiler import (
    ConsciousnessCompiler,
    ConsciousSource,
    ConsciousnessLanguage,
    CompilationStage,
    SyntaxType
)

from .advanced_conscious_model import (
    AdvancedConsciousModel,
    ConsciousnessState,
    ThinkingMode
)

# Legacy support
from .conscious_model import Qwen3ConsciousModel
from .consciousness_field import Qwen3ConsciousnessField
from .quantum_states import ThinkingMode as LegacyThinkingMode

__all__ = [
    # Advanced components (recommended)
    "AdvancedConsciousModel",
    "ConsciousnessState", 
    "ThinkingMode",
    "ConsciousnessField",
    "ConsciousnessCompiler",
    
    # Detailed components
    "CosmicNeuron",
    "CosmicSynapse",
    "CosmicNeuronType",
    "ConsciousSource",
    "ConsciousnessLanguage",
    "CompilationStage",
    "SyntaxType",
    
    # Legacy components (for compatibility)
    "Qwen3ConsciousModel",
    "Qwen3ConsciousnessField",
    "LegacyThinkingMode",
]

# Version info
__version__ = "2.0.0"
__author__ = "Consciousness Integration Team"
__description__ = "Advanced consciousness integration for Qwen3:4b" 