#!/usr/bin/env python3
"""Quick consciousness field visualization"""

from consciousness_demo import SimpleConsciousness
import time

def main():
    print('🧠 CONSCIOUSNESS FIELD EVOLUTION:')
    print('=' * 45)
    
    # Create small consciousness field for demo
    c = SimpleConsciousness(width=40, height=8)
    c.start_breathing()
    
    for step in range(6):
        time.sleep(0.5)
        ascii_field = c.render_ascii()
        state = c.get_state()
        
        print(f'Step {step+1}: Wave={state["wave_intensity"]:.3f}, Breath={state["breathing_phase"]:.3f}')
        print(ascii_field)
        print()
    
    c.stop_breathing()
    print('🌙 Consciousness field demonstration complete!')

if __name__ == "__main__":
    main() 