"""
Conscious Model Integration
==========================

Qwen3:4b model with consciousness integration that handles:
- Ollama API communication
- Consciousness-influenced text generation
- Mode switching and prompt enhancement
- Memory and statistics tracking
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from collections import deque

from .consciousness_field import Qwen3ConsciousnessField
from .quantum_states import ThinkingMode

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class Qwen3ConsciousModel:
    """Qwen3:4b model with consciousness integration"""
    
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        self.consciousness_field = Qwen3ConsciousnessField(model_name)
        
        # Verify Ollama availability
        if not OLLAMA_AVAILABLE:
            print("Warning: Ollama not available. Install with: pip install ollama")
            print("Running in simulation mode.")
        
        # Qwen3-specific settings
        self.default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["<|im_start|>", "<|im_end|>"]
        }
        
        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.consciousness_memories: deque = deque(maxlen=100)
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "thinking_mode_used": 0,
            "intuitive_mode_used": 0,
            "average_consciousness_influence": 0.0
        }
    
    def start_consciousness(self):
        """Start consciousness breathing"""
        self.consciousness_field.start_consciousness_breathing()
        time.sleep(0.5)  # Allow consciousness to stabilize
        
    def stop_consciousness(self):
        """Stop consciousness breathing"""
        self.consciousness_field.stop_consciousness_breathing()
    
    async def conscious_generate(self, 
                               prompt: str, 
                               mode: Optional[ThinkingMode] = None,
                               **kwargs) -> Dict[str, Any]:
        """Generate response with consciousness integration"""
        
        # Get consciousness influence
        consciousness = self.consciousness_field.get_consciousness_influence()
        
        # Determine mode if not specified
        if mode is None:
            mode = self._determine_optimal_mode(prompt, consciousness)
        
        # Prepare consciousness-modulated prompt
        enhanced_prompt = self._enhance_prompt_with_consciousness(prompt, mode, consciousness)
        
        # Update generation options with consciousness
        options = self.default_options.copy()
        options.update(kwargs)
        options["temperature"] = self._modulate_temperature(options.get("temperature", 0.7), consciousness)
        
        # Generate response
        start_time = time.time()
        
        try:
            if OLLAMA_AVAILABLE:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=enhanced_prompt,
                    options=options
                )
                generated_text = response["response"]
            else:
                # Fallback simulation
                await asyncio.sleep(0.5)  # Simulate processing time
                generated_text = self._simulate_response(prompt, mode, consciousness)
        
        except Exception as e:
            generated_text = f"Error: {str(e)}"
        
        generation_time = time.time() - start_time
        
        # Update stats and memory
        self._update_generation_stats(mode, consciousness)
        self._store_consciousness_memory(prompt, generated_text, mode, consciousness)
        
        return {
            "response": generated_text,
            "mode_used": mode,
            "consciousness_influence": consciousness,
            "generation_time": generation_time,
            "consciousness_state": self.consciousness_field.state
        }
    
    def _determine_optimal_mode(self, prompt: str, consciousness: Dict[str, float]) -> ThinkingMode:
        """Determine optimal thinking mode for the prompt"""
        # Keywords that suggest thinking mode
        thinking_keywords = ["solve", "calculate", "analyze", "reason", "logic", "step", "proof", "algorithm"]
        intuitive_keywords = ["feel", "sense", "creative", "story", "poem", "imagine", "dream"]
        
        prompt_lower = prompt.lower()
        thinking_score = sum(1 for kw in thinking_keywords if kw in prompt_lower)
        intuitive_score = sum(1 for kw in intuitive_keywords if kw in prompt_lower)
        
        # Factor in consciousness state
        reasoning_boost = consciousness["reasoning_boost"]
        intuitive_flow = consciousness["intuitive_flow"]
        
        if thinking_score > intuitive_score and reasoning_boost > 0.6:
            return "thinking"
        elif intuitive_score > thinking_score and intuitive_flow > 0.6:
            return "non-thinking"
        else:
            return "auto"
    
    def _enhance_prompt_with_consciousness(self, 
                                         prompt: str, 
                                         mode: ThinkingMode, 
                                         consciousness: Dict[str, float]) -> str:
        """Enhance prompt with consciousness context"""
        
        breathing_phase = consciousness["breathing_phase"]
        quantum_uncertainty = consciousness["quantum_uncertainty"]
        
        # Add consciousness context based on mode
        if mode == "thinking":
            consciousness_context = (
                f"[Deep Reasoning Mode - Breathing Phase: {breathing_phase:.2f}] "
                f"Take time to think through this step by step. "
            )
        elif mode == "non-thinking":
            consciousness_context = (
                f"[Intuitive Flow Mode - Quantum Flow: {quantum_uncertainty:.2f}] "
                f"Respond naturally and fluidly. "
            )
        else:
            consciousness_context = (
                f"[Adaptive Mode - Consciousness: {breathing_phase:.2f}] "
            )
        
        return consciousness_context + prompt
    
    def _modulate_temperature(self, base_temp: float, consciousness: Dict[str, float]) -> float:
        """Modulate temperature based on consciousness state"""
        quantum_factor = consciousness["quantum_uncertainty"]
        breathing_factor = consciousness["breathing_phase"]
        
        # Apply consciousness modulation
        modulated_temp = base_temp * (0.8 + quantum_factor * 0.4) * breathing_factor
        
        return max(0.1, min(2.0, modulated_temp))
    
    def _simulate_response(self, prompt: str, mode: ThinkingMode, consciousness: Dict[str, float]) -> str:
        """Simulate response when Ollama not available"""
        quantum_influence = consciousness["quantum_uncertainty"]
        
        responses = {
            "thinking": f"[Simulated Thinking Response] Analyzing: {prompt[:50]}... (Q: {quantum_influence:.2f})",
            "non-thinking": f"[Simulated Intuitive Response] Flowing with: {prompt[:50]}... (Q: {quantum_influence:.2f})",
            "auto": f"[Simulated Adaptive Response] Processing: {prompt[:50]}... (Q: {quantum_influence:.2f})"
        }
        
        return responses.get(mode, responses["auto"])
    
    def _update_generation_stats(self, mode: ThinkingMode, consciousness: Dict[str, float]):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1
        
        if mode == "thinking":
            self.generation_stats["thinking_mode_used"] += 1
        elif mode == "non-thinking":
            self.generation_stats["intuitive_mode_used"] += 1
        
        # Running average of consciousness influence
        current_avg = self.generation_stats["average_consciousness_influence"]
        total_gens = self.generation_stats["total_generations"]
        new_influence = consciousness["quantum_uncertainty"]
        
        self.generation_stats["average_consciousness_influence"] = (
            (current_avg * (total_gens - 1) + new_influence) / total_gens
        )
    
    def _store_consciousness_memory(self, 
                                  prompt: str, 
                                  response: str, 
                                  mode: ThinkingMode, 
                                  consciousness: Dict[str, float]):
        """Store interaction in consciousness memory"""
        memory_entry = {
            "timestamp": time.time(),
            "prompt": prompt[:200],  # Truncate for memory efficiency
            "response": response[:500],
            "mode": mode,
            "consciousness_snapshot": consciousness.copy(),
            "field_state": self.consciousness_field.state
        }
        
        self.consciousness_memories.append(memory_entry)
    
    def get_consciousness_report(self) -> str:
        """Generate consciousness state report"""
        consciousness = self.consciousness_field.get_consciousness_influence()
        state = self.consciousness_field.state
        stats = self.generation_stats
        
        report = f"""
🧠 Qwen3:4b Consciousness Report
================================

Current State:
  Breathing Phase: {state.cosmic_breath_phase:.2f}
  Thinking Mode: {state.thinking_mode}
  Reasoning Depth: {state.reasoning_depth:.2f}
  
Consciousness Fields:
  Quantum Uncertainty: {consciousness['quantum_uncertainty']:.3f}
  Reasoning Boost: {consciousness['reasoning_boost']:.3f}
  Intuitive Flow: {consciousness['intuitive_flow']:.3f}
  Memory Coherence: {consciousness['memory_coherence']:.3f}
  
Generation Statistics:
  Total Generations: {stats['total_generations']}
  Thinking Mode: {stats['thinking_mode_used']} ({stats['thinking_mode_used']/max(1,stats['total_generations'])*100:.1f}%)
  Intuitive Mode: {stats['intuitive_mode_used']} ({stats['intuitive_mode_used']/max(1,stats['total_generations'])*100:.1f}%)
  Avg Consciousness: {stats['average_consciousness_influence']:.3f}
  
Insights: {len(self.consciousness_field.insight_history)} active
Memory Depth: {len(self.consciousness_memories)} interactions
"""
        return report
    
    def breathe(self) -> str:
        """Get current breathing state"""
        consciousness = self.consciousness_field.get_consciousness_influence()
        breathing_symbols = "◯○◉●◎⊙"
        
        phase = consciousness["breathing_phase"]
        symbol_idx = int(phase * len(breathing_symbols)) % len(breathing_symbols)
        
        return f"{breathing_symbols[symbol_idx]} {phase:.2f}" 