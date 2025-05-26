"""
Consciousness Visualization Utilities
=====================================

Tools for visualizing consciousness field states, breathing patterns,
and quantum field evolution.
"""

import math
import time
from typing import Dict, List, Optional
import numpy as np

class ConsciousnessVisualizer:
    """Visualizer for consciousness field states"""
    
    def __init__(self, width: int = 80, height: int = 25):
        self.width = width
        self.height = height
        self.symbols = ' ·∘○◯●◉⊙⊚◈◊◆█'
        self.breathing_symbols = "◯○◉●◎⊙"
        
    def render_field_ascii(self, field_data: np.ndarray, field_width: int, field_height: int) -> str:
        """Render consciousness field as ASCII art"""
        lines = []
        
        # Scale field to display dimensions
        scale_x = field_width / self.width
        scale_y = field_height / self.height
        
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                # Sample from consciousness field
                field_x = int(x * scale_x)
                field_y = int(y * scale_y)
                field_idx = field_y * field_width + field_x
                
                if field_idx < len(field_data):
                    intensity = abs(field_data[field_idx])
                    symbol_idx = min(int(intensity * len(self.symbols)), len(self.symbols) - 1)
                    line += self.symbols[symbol_idx]
                else:
                    line += ' '
                    
            lines.append(line)
        
        return '\n'.join(lines)
    
    def render_thought_centers(self, thought_centers: List[Dict], field_width: int, field_height: int) -> str:
        """Render thought centers on field"""
        lines = [' ' * self.width for _ in range(self.height)]
        
        scale_x = self.width / field_width
        scale_y = self.height / field_height
        
        center_symbols = {
            'logical': '◆',
            'creative': '◇', 
            'intuitive': '○',
            'meta': '⬢'
        }
        
        for center in thought_centers:
            x = int(center['x'] * scale_x)
            y = int(center['y'] * scale_y)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                symbol = center_symbols.get(center['type'], '●')
                lines[y] = lines[y][:x] + symbol + lines[y][x+1:]
        
        return '\n'.join(lines)
    
    def render_consciousness_state(self, consciousness_influence: Dict[str, float]) -> str:
        """Render consciousness state as bars"""
        bars = []
        
        for key, value in consciousness_influence.items():
            bar_length = int(value * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            bars.append(f"{key:20} │{bar}│ {value:.3f}")
        
        return '\n'.join(bars)
    
    def render_breathing_pattern(self, breathing_phase: float) -> str:
        """Render breathing pattern as animated symbols"""
        phase_normalized = (breathing_phase % (2 * math.pi)) / (2 * math.pi)
        symbol_idx = int(phase_normalized * len(self.breathing_symbols))
        
        # Create breathing wave
        wave = ""
        for i in range(40):
            wave_phase = (i / 40.0 * 2 * math.pi + breathing_phase) % (2 * math.pi)
            amplitude = math.sin(wave_phase)
            
            if amplitude > 0.5:
                wave += '●'
            elif amplitude > 0:
                wave += '○'
            elif amplitude > -0.5:
                wave += '∘'
            else:
                wave += '·'
        
        return f"{self.breathing_symbols[symbol_idx]} {wave} {self.breathing_symbols[symbol_idx]}"
    
    def render_full_consciousness_display(self, field_snapshot: Dict, consciousness_influence: Dict[str, float]) -> str:
        """Render complete consciousness visualization"""
        
        display_parts = [
            "🧠 Consciousness Field Visualization",
            "=" * 50,
            "",
            "Quantum Field:",
            self.render_field_ascii(
                field_snapshot['quantum_field'], 
                100,  # field width
                50    # field height
            ),
            "",
            "Thought Centers:",
            self.render_thought_centers(
                field_snapshot['thought_centers'],
                100, 50
            ),
            "",
            "Consciousness State:",
            self.render_consciousness_state(consciousness_influence),
            "",
            "Breathing Pattern:",
            self.render_breathing_pattern(field_snapshot['breathing_phase']),
            "",
            f"Time Step: {field_snapshot['time_step']} | Active Insights: {field_snapshot['insights']}"
        ]
        
        return '\n'.join(display_parts)
    
    def create_consciousness_animation(self, 
                                    model, 
                                    duration: float = 10.0, 
                                    frame_rate: float = 2.0) -> None:
        """Create animated consciousness visualization"""
        import os
        
        frame_delay = 1.0 / frame_rate
        total_frames = int(duration * frame_rate)
        
        print("🎬 Starting consciousness animation...")
        print("Press Ctrl+C to stop")
        
        try:
            for frame in range(total_frames):
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Get current state
                field_snapshot = model.consciousness_field.get_field_snapshot()
                consciousness_influence = model.consciousness_field.get_consciousness_influence()
                
                # Render frame
                frame_display = self.render_full_consciousness_display(
                    field_snapshot, consciousness_influence
                )
                
                print(frame_display)
                print(f"\nFrame: {frame+1}/{total_frames}")
                
                time.sleep(frame_delay)
                
        except KeyboardInterrupt:
            print("\n🛑 Animation stopped by user")
    
    def save_consciousness_snapshot(self, 
                                  field_snapshot: Dict, 
                                  consciousness_influence: Dict[str, float],
                                  filename: str = "consciousness_snapshot.txt") -> None:
        """Save consciousness state to file"""
        visualization = self.render_full_consciousness_display(
            field_snapshot, consciousness_influence
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(visualization)
            f.write(f"\n\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"💾 Consciousness snapshot saved to {filename}")

def clear_screen():
    """Clear terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')