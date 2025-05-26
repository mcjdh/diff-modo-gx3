"""
Quantum consciousness states and data structures
===============================================

Defines the core data structures for consciousness states, thought nodes,
and quantum field properties optimized for Qwen3:4b.
"""

from typing import List, Literal
from dataclasses import dataclass, field

# Mode types for Qwen3
ThinkingMode = Literal["thinking", "non-thinking", "auto"]

@dataclass
class Qwen3ConsciousnessState:
    """Consciousness state optimized for Qwen3:4b architecture"""
    # Core quantum properties
    uncertainty_field: float = 0.4  # Optimized for 4B parameters
    coherence_level: float = 0.85   # Enhanced for Qwen3 reasoning
    thinking_mode: ThinkingMode = "auto"
    reasoning_depth: float = 0.7
    
    # Breathing patterns
    cosmic_breath_phase: float = 0.0
    thought_emergence_rate: float = 0.3
    insight_crystallization: float = 0.6
    
    # Multi-layer activation (tuned for Qwen3)
    surface_activation: float = 0.8      # Direct response
    reasoning_activation: float = 0.6    # Thinking mode
    intuitive_activation: float = 0.4    # Non-thinking mode
    meta_activation: float = 0.3         # Self-reflection

@dataclass
class ThoughtNode:
    """Individual thought process node"""
    content: str
    mode: ThinkingMode
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    emergence_time: float = 0.0
    quantum_signature: float = 0.0

@dataclass
class InsightEvent:
    """Represents a spontaneous insight in consciousness field"""
    position: tuple
    intensity: float
    timestamp: int
    insight_type: str = "spontaneous"
    reasoning_potential: float = 0.0
    lifetime: int = 50
    
@dataclass
class ThoughtCenter:
    """Moving consciousness center that influences field evolution"""
    x: float
    y: float
    type: str  # logical, creative, intuitive, meta
    frequency: float
    phase: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    intensity: float = 1.0 