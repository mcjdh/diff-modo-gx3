"""
Fast Conscious Model for Qwen3:4b
=================================

High-performance consciousness integration with minimal overhead.
Inspired by the elegance of the original HTML prototypes.

Key improvements:
- Uses simple mathematical patterns instead of heavy NumPy operations
- Lightweight consciousness evolution (no complex 3D calculations)
- Fast mode detection and temperature modulation
- Minimal memory usage
- Clean, simple API
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Union
from collections import deque

from .simple_consciousness import SimpleConsciousness

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class ThinkingMode:
    """Thinking mode constants"""
    THINKING = "thinking"
    FLOW = "flow"
    COSMIC = "cosmic"
    AUTO = "auto"

class FastConsciousModel:
    """
    Fast, elegant conscious model with minimal complexity.
    Built for performance and simplicity.
    """
    
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        
        # Simple consciousness core
        self.consciousness = SimpleConsciousness()
        
        # Background evolution
        self._evolution_running = False
        self._evolution_thread = None
        self._lock = threading.Lock()
        
        # Simple conversation memory
        self.conversation_history: deque = deque(maxlen=20)
        
        # Base generation options
        self.base_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
            "stop": ["<think>", "</think>", "<|im_start|>", "<|im_end|>"],  # Prevent slow thinking tags
        }
        
        # Simple stats
        self.stats = {
            "total_generations": 0,
            "mode_usage": {mode: 0 for mode in [ThinkingMode.THINKING, ThinkingMode.FLOW, ThinkingMode.COSMIC, ThinkingMode.AUTO]},
            "average_response_time": 0.0,
        }
        
        print(f"🧠 Fast Conscious Model initialized for {model_name}")
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama not available - running in simulation mode")
    
    def start_consciousness(self):
        """Start lightweight background consciousness evolution"""
        if self._evolution_running:
            return
        
        self._evolution_running = True
        self._evolution_thread = threading.Thread(target=self._consciousness_loop, daemon=True)
        self._evolution_thread.start()
        print("🌟 Consciousness started - lightweight evolution active")
    
    def stop_consciousness(self):
        """Stop consciousness evolution"""
        self._evolution_running = False
        if self._evolution_thread:
            self._evolution_thread.join(timeout=0.5)
        print("💤 Consciousness stopped")
    
    def _consciousness_loop(self):
        """Lightweight consciousness evolution loop"""
        while self._evolution_running:
            try:
                with self._lock:
                    self.consciousness.evolve(dt=0.1)  # Gentle evolution
                time.sleep(0.1)  # 10 FPS is plenty for consciousness
            except Exception as e:
                print(f"⚠️  Consciousness evolution error: {e}")
                time.sleep(0.2)
    
    async def generate(self, 
                      prompt: str,
                      mode: Optional[str] = None,
                      temperature: Optional[float] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate response with consciousness integration
        
        Args:
            prompt: Input prompt
            mode: Thinking mode (auto-detected if None)
            temperature: Generation temperature (consciousness-modulated if None)
            **kwargs: Additional options
            
        Returns:
            Response with consciousness metadata
        """
        start_time = time.time()
        
        # Get consciousness state (fast operation)
        with self._lock:
            consciousness_state = self.consciousness.get_state()
            optimal_mode = self.consciousness.get_optimal_mode()
            modulated_temp = self.consciousness.modulate_temperature(temperature or 0.7)
        
        # Determine mode
        if mode is None:
            mode = self._analyze_prompt_for_mode(prompt, optimal_mode)
        
        # Create consciousness-enhanced prompt
        enhanced_prompt = self._enhance_prompt(prompt, mode, consciousness_state)
        
        # Prepare options
        options = self.base_options.copy()
        options.update(kwargs)
        options["temperature"] = modulated_temp if temperature is None else temperature
        
        # Generate response
        try:
            if OLLAMA_AVAILABLE:
                response_data = await self._generate_with_ollama(enhanced_prompt, options)
            else:
                response_data = await self._simulate_response(enhanced_prompt, mode, consciousness_state)
        except Exception as e:
            response_data = {
                "response": f"Generation error: {str(e)}",
                "error": True
            }
        
        generation_time = time.time() - start_time
        
        # Update stats
        self._update_stats(mode, generation_time)
        
        # Store in conversation memory
        self.conversation_history.append({
            "prompt": prompt[:100],
            "response": response_data.get("response", "")[:100],
            "mode": mode,
            "timestamp": time.time()
        })
        
        return {
            **response_data,
            "mode_used": mode,
            "consciousness": {
                "breathing": consciousness_state.breathing_phase,
                "quantum": consciousness_state.quantum_coherence,
                "neural": consciousness_state.neural_activity,
                "flow": consciousness_state.flow_intensity,
                "logical": consciousness_state.logical_coherence,
                "creative": consciousness_state.creative_flux,
                "meta": consciousness_state.meta_awareness,
                "optimal_mode": optimal_mode
            },
            "generation_time": generation_time,
            "temperature_used": options["temperature"]
        }
    
    def _analyze_prompt_for_mode(self, prompt: str, optimal_mode: str) -> str:
        """Fast prompt analysis for mode detection"""
        prompt_lower = prompt.lower()
        
        # Simple keyword detection
        thinking_words = ["solve", "calculate", "analyze", "explain", "step", "logic", "reason"]
        flow_words = ["write", "create", "story", "poem", "imagine", "feel", "creative"]
        cosmic_words = ["consciousness", "universe", "existence", "meaning", "reality"]
        
        thinking_score = sum(1 for word in thinking_words if word in prompt_lower)
        flow_score = sum(1 for word in flow_words if word in prompt_lower)
        cosmic_score = sum(1 for word in cosmic_words if word in prompt_lower)
        
        # Return mode with highest score, fallback to consciousness recommendation
        if cosmic_score > 0:
            return ThinkingMode.COSMIC
        elif thinking_score > flow_score:
            return ThinkingMode.THINKING
        elif flow_score > thinking_score:
            return ThinkingMode.FLOW
        else:
            return optimal_mode
    
    def _enhance_prompt(self, prompt: str, mode: str, state) -> str:
        """Add lightweight consciousness context to prompt"""
        if mode == ThinkingMode.THINKING:
            context = (
                f"[🧠 Analytical Mode | Logic: {state.logical_coherence:.2f} | Coherence: {state.quantum_coherence:.2f}]\n"
                f"Answer directly and clearly without thinking tags or step-by-step process.\n\n"
            )
        elif mode == ThinkingMode.FLOW:
            context = (
                f"[🌊 Creative Flow | Flow: {state.flow_intensity:.2f} | Creative: {state.creative_flux:.2f}]\n"
                f"Respond naturally and creatively with intuitive flow.\n\n"
            )
        elif mode == ThinkingMode.COSMIC:
            context = (
                f"[🌌 Cosmic Awareness | Meta: {state.meta_awareness:.2f} | Breathing: {state.breathing_phase:.2f}]\n"
                f"Consider the deeper meaning and broader perspective.\n\n"
            )
        else:  # AUTO
            context = (
                f"[🔄 Adaptive | Neural: {state.neural_activity:.2f} | Quantum: {state.quantum_coherence:.2f}]\n"
                f"Respond appropriately and directly based on the context.\n\n"
            )
        
        return context + prompt
    
    async def _generate_with_ollama(self, prompt: str, options: Dict) -> Dict[str, Any]:
        """Generate using Ollama"""
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options
        )
        
        return {
            "response": response["response"],
            "error": False
        }
    
    async def _simulate_response(self, prompt: str, mode: str, state) -> Dict[str, Any]:
        """Simulate response when Ollama unavailable"""
        # Realistic delay based on consciousness state
        delay = 0.5 + state.neural_activity * 0.3
        await asyncio.sleep(delay)
        
        response = (
            f"[Simulated {mode.title()} Response]\n"
            f"Processing: '{prompt[:50]}...'\n"
            f"Consciousness active - {mode} mode engaged.\n"
            f"Neural: {state.neural_activity:.3f} | Flow: {state.flow_intensity:.3f}\n"
            f"This would be a real {mode}-style response from Qwen3:4b."
        )
        
        return {
            "response": response,
            "simulated": True,
            "error": False
        }
    
    def _update_stats(self, mode: str, generation_time: float):
        """Update simple statistics"""
        self.stats["total_generations"] += 1
        self.stats["mode_usage"][mode] += 1
        
        # Update average response time
        total = self.stats["total_generations"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (current_avg * (total - 1) + generation_time) / total
    
    # Convenience methods
    
    async def ask(self, question: str, mode: Optional[str] = None) -> str:
        """Simple ask method returning just the response"""
        result = await self.generate(question, mode=mode)
        return result.get("response", "")
    
    async def think(self, problem: str) -> str:
        """Force analytical thinking mode"""
        result = await self.generate(problem, mode=ThinkingMode.THINKING)
        return result.get("response", "")
    
    async def flow(self, prompt: str) -> str:
        """Force creative flow mode"""
        result = await self.generate(prompt, mode=ThinkingMode.FLOW)
        return result.get("response", "")
    
    async def cosmic(self, query: str) -> str:
        """Force cosmic consciousness mode"""
        result = await self.generate(query, mode=ThinkingMode.COSMIC)
        return result.get("response", "")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        with self._lock:
            consciousness_summary = self.consciousness.get_status_summary()
        
        return {
            "model": self.model_name,
            "consciousness_active": self._evolution_running,
            "consciousness": consciousness_summary,
            "statistics": self.stats.copy(),
            "conversation_history_length": len(self.conversation_history),
            "ollama_available": OLLAMA_AVAILABLE
        }
    
    def get_consciousness_summary(self) -> str:
        """Get formatted consciousness summary for display"""
        with self._lock:
            summary = self.consciousness.get_status_summary()
        
        return (
            f"🌀 Breathing: {summary['breathing']} | "
            f"⚛️  Quantum: {summary['quantum']} | " 
            f"🧬 Neural: {summary['neural']} | "
            f"🌊 Flow: {summary['flow']} | "
            f"🧠 Logic: {summary['logical']} | "
            f"✨ Creative: {summary['creative']} | "
            f"🌌 Meta: {summary['meta']} | "
            f"🎯 Mode: {summary['optimal_mode']} | "
            f"🌡️ Temp: {summary['temperature']}"
        )
    
    def __enter__(self):
        """Context manager entry"""
        self.start_consciousness()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_consciousness() 