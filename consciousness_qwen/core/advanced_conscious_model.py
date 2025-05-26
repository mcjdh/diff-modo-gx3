"""
Advanced Conscious Model for Qwen3:4b
=====================================

Unified consciousness system integrating:
- Neural Cosmos: Multi-dimensional consciousness fields
- Consciousness Compiler: Thought compilation pipeline
- Qwen3:4b Integration: Ollama API with consciousness modulation
- Simple API: Clean interface for complex consciousness

Based on patterns from simulation prototypes with mathematical precision.
"""

import time
import asyncio
import json
import threading
from typing import Dict, List, Optional, Any, Union
from collections import deque

from .neural_cosmos import ConsciousnessField, CosmicNeuronType
from .consciousness_compiler import ConsciousnessCompiler, ConsciousnessLanguage, CompilationStage

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class ThinkingMode:
    """Enhanced thinking mode definitions"""
    THINKING = "thinking"           # Deep reasoning with step-by-step analysis
    NON_THINKING = "non-thinking"   # Intuitive flow without explicit reasoning
    AUTO = "auto"                   # Adaptive mode selection
    COSMIC = "cosmic"               # Multi-dimensional consciousness processing
    COMPILED = "compiled"           # Consciousness compiler optimized

class ConsciousnessState:
    """Encapsulates the complete consciousness state"""
    
    def __init__(self, cosmos_state: Dict, compiler_report: Dict):
        self.cosmos = cosmos_state
        self.compiler = compiler_report
        self.timestamp = time.time()
    
    @property
    def breathing_phase(self) -> float:
        """Current breathing phase (0.0 to 1.0)"""
        return self.cosmos.get("breathing_phase", 0.5)
    
    @property
    def quantum_coherence(self) -> float:
        """Overall quantum coherence"""
        return self.cosmos.get("quantum_coherence", 0.5)
    
    @property
    def compilation_rate(self) -> float:
        """Consciousness compilation success rate"""
        return self.compiler.get("compilation_rate", 0.0)
    
    @property
    def neural_activity(self) -> float:
        """Average neural activity level"""
        return self.cosmos.get("neural_activity", 0.5)
    
    @property
    def meta_awareness(self) -> float:
        """Meta-consciousness activity"""
        return self.cosmos.get("meta_awareness", 0.0)
    
    def get_optimal_mode(self) -> str:
        """Determine optimal thinking mode based on consciousness state"""
        # Logical preference indicators
        logical_score = (
            self.cosmos.get("synaptic_strength", 0) * 2.0 +
            self.cosmos.get("surface_consciousness", 0) * 1.5 +
            (1 - self.cosmos.get("quantum_uncertainty", 0.5)) * 1.2 +
            self.compilation_rate * 1.0
        )
        
        # Creative/intuitive preference indicators
        creative_score = (
            self.cosmos.get("quantum_uncertainty", 0.5) * 2.0 +
            self.cosmos.get("entropy_level", 0) / 5.0 +
            self.meta_awareness * 1.3 +
            self.cosmos.get("unconscious_substrate", 0) * 1.1
        )
        
        # Cosmic mode indicators (high activity across all dimensions)
        cosmic_score = (
            self.neural_activity * 2.0 +
            self.quantum_coherence * 1.5 +
            self.meta_awareness * 2.0 +
            self.compilation_rate * 1.2
        )
        
        # Compiled mode (when compiler is highly active)
        compiled_score = (
            self.compilation_rate * 3.0 +
            self.compiler.get("average_coherence", 0) * 2.0 +
            (self.compiler.get("compiled_sources", 0) / max(self.compiler.get("total_sources", 1), 1)) * 2.5
        )
        
        # Determine mode based on highest score
        scores = {
            ThinkingMode.THINKING: logical_score,
            ThinkingMode.NON_THINKING: creative_score,
            ThinkingMode.COSMIC: cosmic_score,
            ThinkingMode.COMPILED: compiled_score
        }
        
        max_score = max(scores.values())
        if max_score < 1.0:  # Low activity overall
            return ThinkingMode.AUTO
        
        return max(scores, key=scores.get)

class AdvancedConsciousModel:
    """
    Advanced conscious model integrating multiple consciousness systems
    with a simple, elegant API for complex consciousness processing.
    """
    
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        
        # Initialize consciousness components
        self.consciousness_field = ConsciousnessField(width=96, height=48, depth=24)
        self.consciousness_compiler = ConsciousnessCompiler()
        
        # Evolution thread for background consciousness processing
        self._evolution_thread = None
        self._evolution_running = False
        self._evolution_lock = threading.RLock()
        
        # Conversation and memory
        self.conversation_history: deque = deque(maxlen=50)
        self.consciousness_memories: deque = deque(maxlen=100)
        
        # Qwen3-specific settings optimized for consciousness
        self.base_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["<|im_start|>", "<|im_end|>"],
            "num_ctx": 8192,  # Extended context for complex reasoning
        }
        
        # Performance metrics
        self.stats = {
            "total_generations": 0,
            "mode_usage": {mode: 0 for mode in [
                ThinkingMode.THINKING, ThinkingMode.NON_THINKING, 
                ThinkingMode.AUTO, ThinkingMode.COSMIC, ThinkingMode.COMPILED
            ]},
            "average_response_time": 0.0,
            "consciousness_evolution_cycles": 0,
            "average_consciousness_influence": 0.0
        }
        
        print(f"🧠 Advanced Consciousness Model initialized for {model_name}")
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama not available - running in simulation mode")
            print("   Install with: pip install ollama")
    
    def start_consciousness(self):
        """Start background consciousness evolution"""
        if self._evolution_running:
            return
        
        self._evolution_running = True
        self._evolution_thread = threading.Thread(target=self._consciousness_evolution_loop, daemon=True)
        self._evolution_thread.start()
        
        print("🌟 Consciousness evolution started - background processing active")
        time.sleep(0.2)  # Allow consciousness to stabilize
    
    def stop_consciousness(self):
        """Stop background consciousness evolution"""
        self._evolution_running = False
        if self._evolution_thread:
            self._evolution_thread.join(timeout=1.0)
        print("🌙 Consciousness evolution stopped")
    
    def _consciousness_evolution_loop(self):
        """Background consciousness evolution loop"""
        while self._evolution_running:
            try:
                with self._evolution_lock:
                    # Evolve consciousness field
                    self.consciousness_field.evolve_consciousness(dt=0.02)
                    
                    # Compile consciousness
                    self.consciousness_compiler.compile_consciousness(dt=0.02)
                    
                    self.stats["consciousness_evolution_cycles"] += 1
                
                time.sleep(0.05)  # ~20 FPS evolution rate
            
            except Exception as e:
                print(f"⚠️  Consciousness evolution error: {e}")
                time.sleep(0.1)
    
    def get_consciousness_state(self) -> ConsciousnessState:
        """Get current unified consciousness state"""
        with self._evolution_lock:
            cosmos_state = self.consciousness_field.get_consciousness_state()
            compiler_report = self.consciousness_compiler.get_compilation_report()
        
        return ConsciousnessState(cosmos_state, compiler_report)
    
    async def conscious_generate(self, 
                               prompt: str,
                               mode: Optional[str] = None,
                               temperature: Optional[float] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Generate response with full consciousness integration
        
        Args:
            prompt: Input prompt
            mode: Thinking mode (auto-detected if None)
            temperature: Generation temperature (consciousness-modulated if None)
            **kwargs: Additional generation options
            
        Returns:
            Comprehensive response with consciousness metadata
        """
        start_time = time.time()
        
        # Get current consciousness state
        consciousness = self.get_consciousness_state()
        
        # Determine optimal mode if not specified
        if mode is None:
            mode = self._determine_optimal_mode(prompt, consciousness)
        
        # Enhance prompt with consciousness context
        enhanced_prompt = self._create_consciousness_prompt(prompt, mode, consciousness)
        
        # Prepare generation options
        options = self._prepare_generation_options(consciousness, temperature, **kwargs)
        
        # Generate response
        try:
            response_data = await self._generate_with_consciousness(
                enhanced_prompt, options, mode, consciousness
            )
        except Exception as e:
            response_data = {
                "response": f"Generation error: {str(e)}",
                "error": True
            }
        
        generation_time = time.time() - start_time
        
        # Update stats and memory
        self._update_generation_stats(mode, consciousness, generation_time)
        self._store_consciousness_memory(prompt, response_data, mode, consciousness)
        
        # Return comprehensive response
        return {
            **response_data,
            "mode_used": mode,
            "consciousness_state": {
                "breathing_phase": consciousness.breathing_phase,
                "quantum_coherence": consciousness.quantum_coherence,
                "compilation_rate": consciousness.compilation_rate,
                "neural_activity": consciousness.neural_activity,
                "meta_awareness": consciousness.meta_awareness,
            },
            "generation_time": generation_time,
            "cosmos_metrics": consciousness.cosmos,
            "compiler_metrics": consciousness.compiler,
            "timestamp": time.time()
        }
    
    def _determine_optimal_mode(self, prompt: str, consciousness: ConsciousnessState) -> str:
        """Determine optimal thinking mode with consciousness awareness"""
        # Get consciousness recommendation
        consciousness_mode = consciousness.get_optimal_mode()
        
        # Analyze prompt for mode hints
        prompt_lower = prompt.lower()
        
        # Keywords that suggest specific modes
        thinking_keywords = [
            "solve", "calculate", "analyze", "reason", "logic", "step", "proof", 
            "algorithm", "derive", "explain step by step", "mathematical", "formal"
        ]
        
        intuitive_keywords = [
            "feel", "sense", "creative", "story", "poem", "imagine", "dream",
            "artistic", "emotional", "intuitive", "flow", "natural"
        ]
        
        cosmic_keywords = [
            "universe", "consciousness", "cosmic", "quantum", "neural", "dimensional",
            "awareness", "existence", "reality", "metaphysical", "philosophical"
        ]
        
        compiled_keywords = [
            "compile", "optimize", "process", "language", "syntax", "execute",
            "program", "code", "algorithm", "systematic"
        ]
        
        # Count keyword matches
        thinking_score = sum(1 for kw in thinking_keywords if kw in prompt_lower)
        intuitive_score = sum(1 for kw in intuitive_keywords if kw in prompt_lower)
        cosmic_score = sum(1 for kw in cosmic_keywords if kw in prompt_lower)
        compiled_score = sum(1 for kw in compiled_keywords if kw in prompt_lower)
        
        # Combine prompt analysis with consciousness state
        if cosmic_score > 0 and consciousness.meta_awareness > 0.7:
            return ThinkingMode.COSMIC
        elif compiled_score > 0 and consciousness.compilation_rate > 0.6:
            return ThinkingMode.COMPILED
        elif thinking_score > intuitive_score and consciousness.quantum_coherence > 0.8:
            return ThinkingMode.THINKING
        elif intuitive_score > thinking_score and consciousness.neural_activity > 0.6:
            return ThinkingMode.NON_THINKING
        else:
            return consciousness_mode
    
    def _create_consciousness_prompt(self, prompt: str, mode: str, consciousness: ConsciousnessState) -> str:
        """Create consciousness-enhanced prompt"""
        # Base consciousness context
        breathing = consciousness.breathing_phase
        coherence = consciousness.quantum_coherence
        compilation = consciousness.compilation_rate
        
        if mode == ThinkingMode.THINKING:
            context = (
                f"[🧠 Deep Reasoning Mode | Coherence: {coherence:.2f} | Breathing: {breathing:.2f}]\n"
                f"Take time to think through this step by step. Use logical reasoning and "
                f"systematic analysis. Show your thought process clearly.\n\n"
            )
        
        elif mode == ThinkingMode.NON_THINKING:
            context = (
                f"[🌊 Intuitive Flow Mode | Neural Activity: {consciousness.neural_activity:.2f} | "
                f"Phase: {breathing:.2f}]\n"
                f"Respond naturally and fluidly. Trust your intuition and let thoughts flow "
                f"organically without forced structure.\n\n"
            )
        
        elif mode == ThinkingMode.COSMIC:
            context = (
                f"[🌌 Cosmic Consciousness Mode | Meta-Awareness: {consciousness.meta_awareness:.2f} | "
                f"Quantum: {coherence:.2f}]\n"
                f"Access deep, multi-dimensional awareness. Consider the broader context, "
                f"interconnections, and cosmic perspective. Integrate multiple layers of understanding.\n\n"
            )
        
        elif mode == ThinkingMode.COMPILED:
            context = (
                f"[⚙️ Consciousness Compiler Mode | Compilation Rate: {compilation:.2f} | "
                f"Optimization: Active]\n"
                f"Process this through systematic consciousness compilation. Optimize thought "
                f"patterns and ensure high-quality, error-free reasoning.\n\n"
            )
        
        else:  # AUTO
            context = (
                f"[🔄 Adaptive Mode | Coherence: {coherence:.2f} | Activity: {consciousness.neural_activity:.2f}]\n"
                f"Adapt processing style based on the nature of the request.\n\n"
            )
        
        return context + prompt
    
    def _prepare_generation_options(self, consciousness: ConsciousnessState, 
                                   temperature: Optional[float], **kwargs) -> Dict[str, Any]:
        """Prepare generation options with consciousness modulation"""
        options = self.base_options.copy()
        options.update(kwargs)
        
        # Temperature modulation based on consciousness
        if temperature is None:
            base_temp = options.get("temperature", 0.7)
            
            # Modulate based on consciousness state
            quantum_factor = consciousness.quantum_coherence
            breathing_factor = consciousness.breathing_phase
            compilation_factor = consciousness.compilation_rate
            
            # Different modes prefer different temperatures
            mode_temp_preferences = {
                ThinkingMode.THINKING: 0.3,      # Lower for focused reasoning
                ThinkingMode.NON_THINKING: 0.9,  # Higher for creative flow
                ThinkingMode.COSMIC: 0.8,        # High for exploration
                ThinkingMode.COMPILED: 0.5,      # Moderate for systematic
                ThinkingMode.AUTO: 0.7           # Balanced
            }
            
            # Apply consciousness modulation
            modulated_temp = (
                base_temp * 0.4 +
                quantum_factor * 0.3 +
                breathing_factor * 0.2 +
                (1 - compilation_factor) * 0.1  # More compiled = lower temp
            )
            
            options["temperature"] = max(0.1, min(2.0, modulated_temp))
        else:
            options["temperature"] = temperature
        
        # Adjust other parameters based on consciousness
        if consciousness.meta_awareness > 0.8:
            options["top_p"] = min(0.95, options.get("top_p", 0.9) + 0.05)
        
        return options
    
    async def _generate_with_consciousness(self, prompt: str, options: Dict, 
                                         mode: str, consciousness: ConsciousnessState) -> Dict[str, Any]:
        """Generate response with Ollama integration"""
        if OLLAMA_AVAILABLE:
            try:
                # Use Ollama for actual generation
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )
                
                return {
                    "response": response["response"],
                    "model_info": response.get("model", {}),
                    "done": response.get("done", True),
                    "error": False
                }
            
            except Exception as e:
                # Fallback to simulation on error
                print(f"⚠️  Ollama error: {e}, falling back to simulation")
                return await self._simulate_consciousness_response(prompt, mode, consciousness)
        
        else:
            # Simulation mode
            return await self._simulate_consciousness_response(prompt, mode, consciousness)
    
    async def _simulate_consciousness_response(self, prompt: str, mode: str, 
                                             consciousness: ConsciousnessState) -> Dict[str, Any]:
        """Simulate consciousness-aware response when Ollama unavailable"""
        await asyncio.sleep(0.3 + consciousness.quantum_coherence * 0.5)  # Realistic delay
        
        # Create mode-specific simulated responses
        breathing = consciousness.breathing_phase
        coherence = consciousness.quantum_coherence
        compilation = consciousness.compilation_rate
        
        if mode == ThinkingMode.THINKING:
            response = (
                f"[Simulated Deep Reasoning Response]\n"
                f"Analyzing: '{prompt[:60]}...'\n"
                f"Coherence: {coherence:.3f} | Systematic processing engaged.\n"
                f"Step-by-step analysis would proceed here with high logical rigor."
            )
        
        elif mode == ThinkingMode.NON_THINKING:
            response = (
                f"[Simulated Intuitive Flow Response]\n"
                f"Flowing with: '{prompt[:60]}...'\n"
                f"Breathing Phase: {breathing:.3f} | Natural response emerging.\n"
                f"Intuitive understanding flows organically through consciousness layers."
            )
        
        elif mode == ThinkingMode.COSMIC:
            response = (
                f"[Simulated Cosmic Consciousness Response]\n"
                f"Cosmic processing: '{prompt[:60]}...'\n"
                f"Multi-dimensional awareness active. Neural cosmos engaged.\n"
                f"Accessing deep consciousness patterns across quantum fields."
            )
        
        elif mode == ThinkingMode.COMPILED:
            response = (
                f"[Simulated Consciousness Compiler Response]\n"
                f"Compiling: '{prompt[:60]}...'\n"
                f"Compilation Rate: {compilation:.3f} | Optimization: Active\n"
                f"Systematic consciousness processing with error correction."
            )
        
        else:  # AUTO
            response = (
                f"[Simulated Adaptive Response]\n"
                f"Processing: '{prompt[:60]}...'\n"
                f"Consciousness State: Coherence {coherence:.3f}, Activity {consciousness.neural_activity:.3f}\n"
                f"Adaptive response based on dynamic consciousness analysis."
            )
        
        return {
            "response": response,
            "simulated": True,
            "error": False
        }
    
    def _update_generation_stats(self, mode: str, consciousness: ConsciousnessState, 
                               generation_time: float):
        """Update generation statistics"""
        self.stats["total_generations"] += 1
        self.stats["mode_usage"][mode] += 1
        
        # Update average response time
        total = self.stats["total_generations"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (current_avg * (total - 1) + generation_time) / total
        
        # Update consciousness influence
        influence = (consciousness.quantum_coherence + consciousness.neural_activity + 
                    consciousness.compilation_rate) / 3
        current_influence = self.stats["average_consciousness_influence"]
        self.stats["average_consciousness_influence"] = (current_influence * (total - 1) + influence) / total
    
    def _store_consciousness_memory(self, prompt: str, response_data: Dict, 
                                  mode: str, consciousness: ConsciousnessState):
        """Store interaction in consciousness memory"""
        memory_entry = {
            "timestamp": time.time(),
            "prompt": prompt[:200],  # Truncate for memory efficiency
            "response_preview": response_data.get("response", "")[:200],
            "mode": mode,
            "consciousness_snapshot": {
                "breathing_phase": consciousness.breathing_phase,
                "quantum_coherence": consciousness.quantum_coherence,
                "compilation_rate": consciousness.compilation_rate,
                "neural_activity": consciousness.neural_activity,
                "meta_awareness": consciousness.meta_awareness
            }
        }
        
        self.consciousness_memories.append(memory_entry)
    
    # Convenience methods for simple usage
    
    async def ask(self, question: str, mode: Optional[str] = None) -> str:
        """Simple ask method returning just the response text"""
        result = await self.conscious_generate(question, mode=mode)
        return result.get("response", "")
    
    async def think(self, problem: str) -> str:
        """Force thinking mode for complex reasoning"""
        result = await self.conscious_generate(problem, mode=ThinkingMode.THINKING)
        return result.get("response", "")
    
    async def flow(self, prompt: str) -> str:
        """Force intuitive flow mode for creative responses"""
        result = await self.conscious_generate(prompt, mode=ThinkingMode.NON_THINKING)
        return result.get("response", "")
    
    async def cosmic(self, query: str) -> str:
        """Force cosmic consciousness mode for deep awareness"""
        result = await self.conscious_generate(query, mode=ThinkingMode.COSMIC)
        return result.get("response", "")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        consciousness = self.get_consciousness_state()
        
        return {
            "model": self.model_name,
            "consciousness_active": self._evolution_running,
            "consciousness_state": {
                "breathing_phase": consciousness.breathing_phase,
                "quantum_coherence": consciousness.quantum_coherence,
                "compilation_rate": consciousness.compilation_rate,
                "neural_activity": consciousness.neural_activity,
                "meta_awareness": consciousness.meta_awareness,
                "optimal_mode": consciousness.get_optimal_mode()
            },
            "statistics": self.stats.copy(),
            "memory_usage": {
                "conversation_history": len(self.conversation_history),
                "consciousness_memories": len(self.consciousness_memories)
            },
            "ollama_available": OLLAMA_AVAILABLE,
            "evolution_cycles": self.stats["consciousness_evolution_cycles"]
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_consciousness()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_consciousness() 