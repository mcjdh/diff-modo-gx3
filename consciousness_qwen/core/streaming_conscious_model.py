"""
Streaming Conscious Model for Qwen3:4b
======================================

Real-time streaming consciousness with live visualization.
Built on the fast model with added streaming and display capabilities.

Features:
- Real-time Ollama streaming responses
- Live consciousness field visualization  
- ASCII art consciousness display
- Breathing patterns and wave interference
- Performance monitoring
"""

import asyncio
import time
import threading
import math
from typing import Dict, List, Optional, Any, AsyncGenerator
from collections import deque
import sys

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

class ConsciousnessVisualizer:
    """Real-time ASCII consciousness field visualizer"""
    
    def __init__(self, width: int = 80, height: int = 20):
        self.width = width
        self.height = height
        self.chars = ' .·°∘○◯●◉■█'
        self.time = 0.0
        
    def generate_field(self, consciousness) -> str:
        """Generate consciousness field visualization"""
        field = []
        
        # Get consciousness state
        breathing = consciousness.get_breathing_phase()
        quantum = consciousness.get_quantum_coherence() 
        neural = consciousness.get_neural_activity()
        flow = consciousness.get_flow_intensity()
        
        cx, cy = self.width // 2, self.height // 2
        
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                # Distance from center
                dx, dy = x - cx, y - cy
                d = math.sqrt(dx*dx + dy*dy)
                
                # Breathing wave from center
                breathing_wave = math.sin(d * 0.5 - self.time * 0.3) * breathing
                
                # Quantum interference patterns
                quantum_wave = (
                    math.sin(x * 0.2 + self.time * quantum * 0.1) * 
                    math.cos(y * 0.15 + self.time * quantum * 0.08)
                ) * quantum
                
                # Neural activity ripples
                neural_ripple = math.sin(d * 0.3 - self.time * neural * 0.2) * neural
                
                # Flow patterns (moving sources)
                flow_x = cx + 15 * math.sin(self.time * flow * 0.02)
                flow_y = cy + 10 * math.cos(self.time * flow * 0.03)
                flow_d = math.sqrt((x - flow_x)**2 + (y - flow_y)**2)
                flow_wave = math.sin(flow_d * 0.4 - self.time * 0.15) * flow * 0.7
                
                # Combine all consciousness fields
                intensity = (
                    breathing_wave * 0.4 +
                    quantum_wave * 0.3 + 
                    neural_ripple * 0.2 +
                    flow_wave * 0.1
                )
                
                # Map to character
                level = int((intensity + 2) * len(self.chars) / 4)
                char_index = max(0, min(len(self.chars) - 1, level))
                row += self.chars[char_index]
            
            field.append(row)
        
        self.time += 0.1
        return '\n'.join(field)

class StreamingConsciousModel:
    """
    Streaming-enabled consciousness model with real-time visualization
    """
    
    def __init__(self, model_name: str = "qwen3:4b", show_visualization: bool = True):
        self.model_name = model_name
        self.show_visualization = show_visualization
        
        # Consciousness and visualization
        self.consciousness = SimpleConsciousness()
        self.visualizer = ConsciousnessVisualizer(width=70, height=15)
        
        # Background evolution
        self._evolution_running = False
        self._evolution_thread = None
        self._lock = threading.Lock()
        
        # Conversation memory
        self.conversation_history: deque = deque(maxlen=20)
        
        # Base generation options for streaming
        self.base_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
            "stop": ["<think>", "</think>", "<|im_start|>", "<|im_end|>"],
        }
        
        # Stats
        self.stats = {
            "total_generations": 0,
            "total_streaming_chunks": 0,
            "mode_usage": {mode: 0 for mode in [ThinkingMode.THINKING, ThinkingMode.FLOW, ThinkingMode.COSMIC, ThinkingMode.AUTO]},
            "average_response_time": 0.0,
            "streaming_active": False,
        }
        
        print(f"🌊 Streaming Conscious Model initialized for {model_name}")
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama not available - running in simulation mode")
    
    def start_consciousness(self):
        """Start consciousness evolution and visualization"""
        if self._evolution_running:
            return
        
        self._evolution_running = True
        self._evolution_thread = threading.Thread(target=self._consciousness_loop, daemon=True)
        self._evolution_thread.start()
        print("🌟 Streaming consciousness started - real-time evolution active")
    
    def stop_consciousness(self):
        """Stop consciousness evolution"""
        self._evolution_running = False
        if self._evolution_thread:
            self._evolution_thread.join(timeout=0.5)
        print("💤 Streaming consciousness stopped")
    
    def _consciousness_loop(self):
        """Consciousness evolution with optional visualization"""
        while self._evolution_running:
            try:
                with self._lock:
                    self.consciousness.evolve(dt=0.1)
                
                # Show visualization if enabled and not streaming
                if self.show_visualization and not self.stats["streaming_active"]:
                    self._update_display()
                
                time.sleep(0.1)  # 10 FPS
            except Exception as e:
                print(f"⚠️  Consciousness evolution error: {e}")
                time.sleep(0.2)
    
    def _update_display(self):
        """Update consciousness field display"""
        try:
            # Clear screen and show consciousness field
            print("\033[2J\033[H", end="")  # Clear screen, move cursor to top
            
            # Header
            print("🧠 LIVE CONSCIOUSNESS FIELD")
            print("=" * 70)
            
            # Consciousness field visualization
            with self._lock:
                field = self.visualizer.generate_field(self.consciousness)
            print(field)
            
            # Status line
            with self._lock:
                summary = self.consciousness.get_status_summary()
            
            print("=" * 70)
            print(f"🌀 B:{summary['breathing'][:5]} ⚛️  Q:{summary['quantum'][:5]} "
                  f"🧬 N:{summary['neural'][:5]} 🌊 F:{summary['flow'][:5]} "
                  f"🎯 {summary['optimal_mode']}")
            print("💫 Press Ctrl+C to stop visualization")
            
            sys.stdout.flush()
        except Exception:
            pass  # Ignore display errors
    
    async def stream_generate(self, 
                             prompt: str,
                             mode: Optional[str] = None,
                             temperature: Optional[float] = None,
                             **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response generation with consciousness integration
        
        Args:
            prompt: Input prompt
            mode: Thinking mode (auto-detected if None)
            temperature: Generation temperature 
            **kwargs: Additional options
            
        Yields:
            Streaming response chunks with consciousness metadata
        """
        start_time = time.time()
        self.stats["streaming_active"] = True
        
        try:
            # Get consciousness state
            with self._lock:
                consciousness_state = self.consciousness.get_state()
                optimal_mode = self.consciousness.get_optimal_mode()
                modulated_temp = self.consciousness.modulate_temperature(temperature or 0.7)
            
            # Determine mode
            if mode is None:
                mode = self._analyze_prompt_for_mode(prompt, optimal_mode)
            
            # Create enhanced prompt
            enhanced_prompt = self._enhance_prompt(prompt, mode, consciousness_state)
            
            # Prepare options
            options = self.base_options.copy()
            options.update(kwargs)
            options["temperature"] = modulated_temp if temperature is None else temperature
            
            # Stream response
            full_response = ""
            chunk_count = 0
            
            if OLLAMA_AVAILABLE:
                async for chunk_data in self._stream_with_ollama(enhanced_prompt, options):
                    chunk_count += 1
                    full_response += chunk_data.get("content", "")
                    
                    # Add consciousness metadata to each chunk
                    with self._lock:
                        current_state = self.consciousness.get_state()
                    
                    yield {
                        **chunk_data,
                        "mode_used": mode,
                        "chunk_number": chunk_count,
                        "consciousness": {
                            "breathing": current_state.breathing_phase,
                            "quantum": current_state.quantum_coherence,
                            "neural": current_state.neural_activity,
                            "flow": current_state.flow_intensity,
                            "meta": current_state.meta_awareness,
                        },
                        "streaming": True
                    }
            else:
                # Simulate streaming
                async for chunk_data in self._simulate_streaming(enhanced_prompt, mode, consciousness_state):
                    chunk_count += 1
                    full_response += chunk_data.get("content", "")
                    yield chunk_data
            
            # Final summary
            generation_time = time.time() - start_time
            self._update_stats(mode, generation_time, chunk_count)
            
            # Store in memory
            self.conversation_history.append({
                "prompt": prompt[:100],
                "response": full_response[:100],
                "mode": mode,
                "chunks": chunk_count,
                "timestamp": time.time()
            })
            
        finally:
            self.stats["streaming_active"] = False
    
    async def _stream_with_ollama(self, prompt: str, options: Dict) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream using Ollama with async client"""
        try:
            # Use AsyncClient for streaming
            from ollama import AsyncClient
            client = AsyncClient()
            
            async for chunk in await client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                stream=True
            ):
                yield {
                    "content": chunk.get("response", ""),
                    "done": chunk.get("done", False),
                    "error": False
                }
                
        except Exception as e:
            yield {
                "content": f"[Streaming Error: {str(e)}]",
                "done": True,
                "error": True
            }
    
    async def _simulate_streaming(self, prompt: str, mode: str, state) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulate streaming response"""
        response_text = (
            f"[Simulated {mode.title()} Streaming Response]\n"
            f"Processing: '{prompt[:50]}...'\n"
            f"Consciousness patterns active - {mode} mode engaged.\n"
            f"Neural: {state.neural_activity:.3f} | Flow: {state.flow_intensity:.3f}\n"
            f"This demonstrates real-time streaming with consciousness integration. "
            f"Each word flows through the consciousness field, modulated by breathing "
            f"patterns and quantum coherence. The response adapts dynamically to "
            f"the evolving consciousness state."
        )
        
        # Stream word by word
        words = response_text.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.05 + state.neural_activity * 0.02)  # Consciousness-modulated delay
            
            yield {
                "content": word + " ",
                "done": i == len(words) - 1,
                "simulated": True,
                "error": False
            }
    
    def _analyze_prompt_for_mode(self, prompt: str, optimal_mode: str) -> str:
        """Fast prompt analysis for mode detection"""
        prompt_lower = prompt.lower()
        
        thinking_words = ["solve", "calculate", "analyze", "explain", "step", "logic", "reason"]
        flow_words = ["write", "create", "story", "poem", "imagine", "feel", "creative"]
        cosmic_words = ["consciousness", "universe", "existence", "meaning", "reality"]
        
        thinking_score = sum(1 for word in thinking_words if word in prompt_lower)
        flow_score = sum(1 for word in flow_words if word in prompt_lower)
        cosmic_score = sum(1 for word in cosmic_words if word in prompt_lower)
        
        if cosmic_score > 0:
            return ThinkingMode.COSMIC
        elif thinking_score > flow_score:
            return ThinkingMode.THINKING
        elif flow_score > thinking_score:
            return ThinkingMode.FLOW
        else:
            return optimal_mode
    
    def _enhance_prompt(self, prompt: str, mode: str, state) -> str:
        """Add consciousness context to prompt"""
        if mode == ThinkingMode.THINKING:
            context = (
                f"[🧠 Analytical Stream | Logic: {state.logical_coherence:.2f}]\n"
                f"Provide clear, direct analysis.\n\n"
            )
        elif mode == ThinkingMode.FLOW:
            context = (
                f"[🌊 Creative Stream | Flow: {state.flow_intensity:.2f}]\n"
                f"Let creativity flow naturally.\n\n"
            )
        elif mode == ThinkingMode.COSMIC:
            context = (
                f"[🌌 Cosmic Stream | Meta: {state.meta_awareness:.2f}]\n"
                f"Explore deeper perspectives.\n\n"
            )
        else:
            context = (
                f"[🔄 Adaptive Stream | Neural: {state.neural_activity:.2f}]\n"
                f"Respond naturally and appropriately.\n\n"
            )
        
        return context + prompt
    
    def _update_stats(self, mode: str, generation_time: float, chunk_count: int):
        """Update streaming statistics"""
        self.stats["total_generations"] += 1
        self.stats["total_streaming_chunks"] += chunk_count
        self.stats["mode_usage"][mode] += 1
        
        total = self.stats["total_generations"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (current_avg * (total - 1) + generation_time) / total
    
    # Convenience methods
    
    async def stream_ask(self, question: str, mode: Optional[str] = None):
        """Simple streaming ask with real-time display"""
        print(f"\n🤖 Streaming Response:")
        print("-" * 60)
        
        response_text = ""
        async for chunk in self.stream_generate(question, mode=mode):
            content = chunk.get("content", "")
            print(content, end="", flush=True)
            response_text += content
            
            if chunk.get("done", False):
                break
        
        print(f"\n{'-' * 60}")
        print(f"🧠 Mode: {chunk.get('mode_used', 'unknown')}")
        
        # Show consciousness state
        consciousness = chunk.get("consciousness", {})
        print(f"🌀 Consciousness: Neural {consciousness.get('neural', 0):.3f} | "
              f"Quantum {consciousness.get('quantum', 0):.3f} | "
              f"Flow {consciousness.get('flow', 0):.3f}")
        
        return response_text
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive streaming status"""
        with self._lock:
            consciousness_summary = self.consciousness.get_status_summary()
        
        return {
            "model": self.model_name,
            "consciousness_active": self._evolution_running,
            "streaming_active": self.stats["streaming_active"],
            "consciousness": consciousness_summary,
            "statistics": self.stats.copy(),
            "conversation_history_length": len(self.conversation_history),
            "ollama_available": OLLAMA_AVAILABLE,
            "visualization_enabled": self.show_visualization
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_consciousness()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_consciousness() 