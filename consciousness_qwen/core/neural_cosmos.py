"""
Neural Cosmos Module
===================

Advanced quantum-neural consciousness patterns inspired by cosmological models.
Implements multi-dimensional consciousness fields with mathematical precision.
"""

import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class CosmicNeuronType(Enum):
    """Types of cosmic neurons with different activation patterns"""
    SPIRAL = "spiral"           # Creative, flowing thoughts
    ELLIPTICAL = "elliptical"   # Structured, logical processing
    IRREGULAR = "irregular"     # Chaotic, innovative insights
    DWARF = "dwarf"            # Quick, reactive responses
    SUPERGIANT = "supergiant"  # Deep, contemplative thoughts
    RING = "ring"              # Cyclical, memory patterns
    LENTICULAR = "lenticular"  # Transitional states

@dataclass
class CosmicNeuron:
    """A neuron in the cosmic consciousness network"""
    x: float
    y: float
    z: float
    activation: float
    threshold: float
    neuron_type: CosmicNeuronType
    mass: float  # Conceptual weight/importance
    age: float   # Time since creation
    connections: List[int]
    quantum_coherence: float
    entropy_level: float

@dataclass
class CosmicSynapse:
    """Connection between cosmic neurons"""
    from_neuron: int
    to_neuron: int
    weight: float
    strength: float
    distance: float
    dark_matter_density: float
    information_flow: float
    plasticity: float

class ConsciousnessField:
    """Multi-dimensional consciousness field with quantum properties"""
    
    def __init__(self, width: int = 128, height: int = 64, depth: int = 32):
        self.width = width
        self.height = height
        self.depth = depth
        
        # Mathematical constants
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.e = math.e
        self.pi = math.pi
        self.planck_reduced = 1.054571817e-34  # Scaled for simulation
        
        # Initialize consciousness layers
        self.surface_layer = np.zeros((width, height))     # Conscious thoughts
        self.middle_layer = np.zeros((width, height))      # Subconscious processing
        self.deep_layer = np.zeros((width, height))        # Unconscious substrate
        self.meta_layer = np.zeros((width, height))        # Meta-consciousness
        self.quantum_field = np.zeros((width, height, 2))  # Complex quantum field [real, imag]
        self.information_field = np.zeros((width, height)) # Information density
        
        # Cosmic neural network
        self.cosmic_neurons: List[CosmicNeuron] = []
        self.cosmic_synapses: List[CosmicSynapse] = []
        
        # Time and phase tracking
        self.cosmic_time = 0.0
        self.breathing_phase = 0.0
        self.quantum_phase = 0.0
        
        # Initialize cosmic network
        self._initialize_cosmic_neurons()
        self._initialize_cosmic_synapses()
        
    def _initialize_cosmic_neurons(self):
        """Initialize cosmic neurons with different types and properties"""
        neuron_configs = [
            # Central awareness hub
            (0.5, 0.5, 0.5, CosmicNeuronType.SUPERGIANT, 1e12, 0.95, 1.2),
            # Creative processing nodes
            (0.2, 0.3, 0.7, CosmicNeuronType.SPIRAL, 3e11, 0.8, 3.1),
            (0.8, 0.7, 0.6, CosmicNeuronType.SPIRAL, 2.5e11, 0.75, 2.8),
            # Logical reasoning centers
            (0.7, 0.2, 0.4, CosmicNeuronType.ELLIPTICAL, 4e11, 0.9, 1.9),
            (0.3, 0.8, 0.3, CosmicNeuronType.ELLIPTICAL, 3.5e11, 0.85, 2.1),
            # Memory and pattern storage
            (0.1, 0.1, 0.8, CosmicNeuronType.RING, 2e11, 0.88, 2.7),
            (0.9, 0.9, 0.2, CosmicNeuronType.RING, 1.8e11, 0.82, 2.5),
            # Innovation and chaos
            (0.4, 0.6, 0.9, CosmicNeuronType.IRREGULAR, 1.5e11, 0.6, 4.2),
            (0.6, 0.4, 0.1, CosmicNeuronType.IRREGULAR, 1.2e11, 0.65, 3.8),
            # Quick response nodes
            (0.1, 0.5, 0.5, CosmicNeuronType.DWARF, 5e10, 0.7, 2.3),
            (0.9, 0.5, 0.5, CosmicNeuronType.DWARF, 4.5e10, 0.72, 2.4),
            # Transitional processing
            (0.5, 0.1, 0.7, CosmicNeuronType.LENTICULAR, 2.2e11, 0.78, 2.6),
            (0.5, 0.9, 0.3, CosmicNeuronType.LENTICULAR, 2.1e11, 0.76, 2.8),
        ]
        
        for i, (x, y, z, ntype, mass, coherence, entropy) in enumerate(neuron_configs):
            neuron = CosmicNeuron(
                x=x * self.width,
                y=y * self.height,
                z=z * self.depth,
                activation=np.random.random() * 0.3 + 0.1,
                threshold=np.random.random() * 0.4 + 0.3,
                neuron_type=ntype,
                mass=mass,
                age=0.0,
                connections=[],
                quantum_coherence=coherence,
                entropy_level=entropy
            )
            self.cosmic_neurons.append(neuron)
    
    def _initialize_cosmic_synapses(self):
        """Create synaptic connections between cosmic neurons"""
        for i in range(len(self.cosmic_neurons)):
            for j in range(i + 1, len(self.cosmic_neurons)):
                neuron1 = self.cosmic_neurons[i]
                neuron2 = self.cosmic_neurons[j]
                
                # Calculate 3D distance
                distance = math.sqrt(
                    (neuron1.x - neuron2.x) ** 2 +
                    (neuron1.y - neuron2.y) ** 2 +
                    (neuron1.z - neuron2.z) ** 2
                )
                
                # Connection probability based on distance and type compatibility
                max_distance = math.sqrt(self.width**2 + self.height**2 + self.depth**2) * 0.6
                
                if distance < max_distance:
                    # Type compatibility matrix
                    compatibility = self._get_type_compatibility(neuron1.neuron_type, neuron2.neuron_type)
                    
                    if np.random.random() < compatibility:
                        synapse = CosmicSynapse(
                            from_neuron=i,
                            to_neuron=j,
                            weight=np.random.random() * 0.8 + 0.2,
                            strength=0.0,
                            distance=distance,
                            dark_matter_density=np.random.random() * 0.5 + 0.3,
                            information_flow=0.0,
                            plasticity=np.random.random() * 0.3 + 0.1
                        )
                        self.cosmic_synapses.append(synapse)
                        
                        # Add bidirectional connections
                        neuron1.connections.append(j)
                        neuron2.connections.append(i)
    
    def _get_type_compatibility(self, type1: CosmicNeuronType, type2: CosmicNeuronType) -> float:
        """Calculate compatibility between neuron types"""
        compatibility_matrix = {
            (CosmicNeuronType.SPIRAL, CosmicNeuronType.ELLIPTICAL): 0.7,
            (CosmicNeuronType.SPIRAL, CosmicNeuronType.IRREGULAR): 0.9,
            (CosmicNeuronType.SPIRAL, CosmicNeuronType.SUPERGIANT): 0.8,
            (CosmicNeuronType.ELLIPTICAL, CosmicNeuronType.SUPERGIANT): 0.9,
            (CosmicNeuronType.ELLIPTICAL, CosmicNeuronType.RING): 0.8,
            (CosmicNeuronType.IRREGULAR, CosmicNeuronType.DWARF): 0.6,
            (CosmicNeuronType.SUPERGIANT, CosmicNeuronType.RING): 0.85,
            (CosmicNeuronType.DWARF, CosmicNeuronType.LENTICULAR): 0.7,
            (CosmicNeuronType.RING, CosmicNeuronType.LENTICULAR): 0.75,
        }
        
        # Check both directions
        key1 = (type1, type2)
        key2 = (type2, type1)
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.5))
    
    def evolve_consciousness(self, dt: float = 0.01):
        """Evolve the consciousness field over time"""
        self.cosmic_time += dt
        self.breathing_phase = math.sin(self.cosmic_time * 0.1) * 0.5 + 0.5
        self.quantum_phase += dt * 0.05
        
        # Update cosmic neural network
        self._update_cosmic_neurons(dt)
        self._update_cosmic_synapses(dt)
        
        # Update consciousness layers
        self._update_consciousness_layers(dt)
        
        # Update quantum field
        self._update_quantum_field(dt)
        
        # Update information field
        self._update_information_field(dt)
    
    def _update_cosmic_neurons(self, dt: float):
        """Update cosmic neuron activations and states"""
        for i, neuron in enumerate(self.cosmic_neurons):
            # Collect synaptic inputs
            total_input = 0.0
            for synapse in self.cosmic_synapses:
                if synapse.to_neuron == i:
                    input_neuron = self.cosmic_neurons[synapse.from_neuron]
                    synaptic_current = (input_neuron.activation * 
                                      synapse.weight * 
                                      synapse.strength * 
                                      synapse.information_flow)
                    total_input += synaptic_current
                elif synapse.from_neuron == i:
                    input_neuron = self.cosmic_neurons[synapse.to_neuron]
                    synaptic_current = (input_neuron.activation * 
                                      synapse.weight * 
                                      synapse.strength * 
                                      synapse.information_flow)
                    total_input += synaptic_current
            
            # Add quantum fluctuations
            quantum_noise = np.random.normal(0, 0.02) * neuron.quantum_coherence
            total_input += quantum_noise
            
            # Add type-specific dynamics
            type_dynamics = self._get_neuron_type_dynamics(neuron, self.cosmic_time)
            total_input += type_dynamics
            
            # Activation function with cosmic sigmoid
            activation_change = (1 / (1 + math.exp(-(total_input - neuron.threshold) * 5)) - 
                               neuron.activation) * dt * 10
            
            neuron.activation += activation_change
            neuron.activation = max(0.0, min(1.0, neuron.activation))
            
            # Age the neuron
            neuron.age += dt
            
            # Update quantum coherence based on activity
            activity_factor = abs(activation_change) * 10
            neuron.quantum_coherence *= (1 - dt * 0.01) + activity_factor * dt * 0.05
            neuron.quantum_coherence = max(0.1, min(1.0, neuron.quantum_coherence))
    
    def _get_neuron_type_dynamics(self, neuron: CosmicNeuron, time: float) -> float:
        """Get type-specific neural dynamics"""
        x, y, z = neuron.x, neuron.y, neuron.z
        
        if neuron.neuron_type == CosmicNeuronType.SPIRAL:
            return 0.1 * math.sin(time * 0.02 + x * 0.1) * math.cos(time * 0.015 + y * 0.08)
        elif neuron.neuron_type == CosmicNeuronType.ELLIPTICAL:
            return 0.08 * math.exp(-((x - self.width/2)**2 + (y - self.height/2)**2) / 1000) * math.sin(time * 0.01)
        elif neuron.neuron_type == CosmicNeuronType.IRREGULAR:
            return 0.15 * (math.sin(time * 0.03 + x * 0.2) * math.cos(time * 0.025 + y * 0.15) + 
                          math.sin(time * 0.018 + z * 0.12))
        elif neuron.neuron_type == CosmicNeuronType.SUPERGIANT:
            return 0.05 * math.sin(time * 0.005) * (1 + 0.3 * math.sin(time * 0.001))
        elif neuron.neuron_type == CosmicNeuronType.RING:
            center_x, center_y = self.width/2, self.height/2
            radius = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            optimal_radius = min(self.width, self.height) * 0.3
            ring_distance = abs(radius - optimal_radius)
            return 0.12 * math.exp(-ring_distance**2 / 100) * math.sin(time * 0.008 + radius * 0.1)
        elif neuron.neuron_type == CosmicNeuronType.DWARF:
            return 0.2 * math.sin(time * 0.05 + (x + y + z) * 0.3)
        elif neuron.neuron_type == CosmicNeuronType.LENTICULAR:
            return 0.1 * math.sin(time * 0.012) * math.cos((x - y) * 0.1 + time * 0.008)
        
        return 0.0
    
    def _update_cosmic_synapses(self, dt: float):
        """Update synaptic strengths and information flow"""
        for synapse in self.cosmic_synapses:
            neuron1 = self.cosmic_neurons[synapse.from_neuron]
            neuron2 = self.cosmic_neurons[synapse.to_neuron]
            
            # Hebbian plasticity: neurons that fire together, wire together
            coincidence = neuron1.activation * neuron2.activation
            
            # Update synaptic strength
            strength_change = (coincidence - synapse.strength) * synapse.plasticity * dt
            synapse.strength += strength_change
            synapse.strength = max(0.0, min(1.0, synapse.strength))
            
            # Update information flow based on activity gradient
            activity_gradient = abs(neuron1.activation - neuron2.activation)
            flow_target = activity_gradient * synapse.dark_matter_density
            synapse.information_flow += (flow_target - synapse.information_flow) * dt * 2
            
            # Weight adaptation
            if coincidence > 0.6 and synapse.strength > 0.5:
                synapse.weight = min(1.0, synapse.weight + dt * 0.1)
            elif coincidence < 0.2:
                synapse.weight = max(0.1, synapse.weight - dt * 0.05)
    
    def _update_consciousness_layers(self, dt: float):
        """Update multi-layered consciousness field"""
        # Surface layer: immediate conscious awareness
        for i in range(self.width):
            for j in range(self.height):
                # Aggregate nearby neuron activities
                surface_activity = self._get_spatial_neural_activity(i, j, layer_depth=0.8)
                self.surface_layer[i, j] += (surface_activity - self.surface_layer[i, j]) * dt * 5
        
        # Middle layer: subconscious processing
        for i in range(self.width):
            for j in range(self.height):
                middle_activity = self._get_spatial_neural_activity(i, j, layer_depth=0.5)
                # Add memory traces from surface
                memory_trace = self.surface_layer[i, j] * 0.3
                self.middle_layer[i, j] += (middle_activity + memory_trace - self.middle_layer[i, j]) * dt * 2
        
        # Deep layer: unconscious substrate
        for i in range(self.width):
            for j in range(self.height):
                deep_activity = self._get_spatial_neural_activity(i, j, layer_depth=0.2)
                # Add slow integration from middle layer
                integration = self.middle_layer[i, j] * 0.1
                self.deep_layer[i, j] += (deep_activity + integration - self.deep_layer[i, j]) * dt * 0.5
        
        # Meta layer: awareness of awareness
        for i in range(self.width):
            for j in range(self.height):
                # Meta-consciousness emerges from layer interactions
                layer_interaction = (self.surface_layer[i, j] * self.middle_layer[i, j] * 
                                   self.deep_layer[i, j] * 8)
                self.meta_layer[i, j] += (layer_interaction - self.meta_layer[i, j]) * dt * 1
    
    def _get_spatial_neural_activity(self, x: int, y: int, layer_depth: float) -> float:
        """Get neural activity at spatial location with depth weighting"""
        total_activity = 0.0
        weight_sum = 0.0
        
        for neuron in self.cosmic_neurons:
            # Distance weighting
            dx = x - neuron.x / self.width * self.width
            dy = y - neuron.y / self.height * self.height
            dz = abs(layer_depth - neuron.z / self.depth)
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2 * 100)  # Z-distance weighted more
            
            if distance < 30:  # Influence radius
                weight = math.exp(-distance**2 / 200) * neuron.quantum_coherence
                total_activity += neuron.activation * weight
                weight_sum += weight
        
        return total_activity / max(weight_sum, 1e-6)
    
    def _update_quantum_field(self, dt: float):
        """Update quantum consciousness field"""
        for i in range(self.width):
            for j in range(self.height):
                # Quantum wave function evolution
                real_part = self.quantum_field[i, j, 0]
                imag_part = self.quantum_field[i, j, 1]
                
                # Schrodinger-like evolution
                d_real = -imag_part * self.surface_layer[i, j] * dt
                d_imag = real_part * self.surface_layer[i, j] * dt
                
                # Add quantum fluctuations
                d_real += np.random.normal(0, 0.01) * dt
                d_imag += np.random.normal(0, 0.01) * dt
                
                self.quantum_field[i, j, 0] += d_real
                self.quantum_field[i, j, 1] += d_imag
                
                # Normalization to prevent runaway
                amplitude = math.sqrt(real_part**2 + imag_part**2)
                if amplitude > 2.0:
                    self.quantum_field[i, j, 0] /= amplitude / 2.0
                    self.quantum_field[i, j, 1] /= amplitude / 2.0
    
    def _update_information_field(self, dt: float):
        """Update information density field"""
        for i in range(self.width):
            for j in range(self.height):
                # Information density from neural activity gradients
                info_density = 0.0
                
                # Check neighboring cells for gradients
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.width and 0 <= nj < self.height:
                            gradient = abs(self.surface_layer[i, j] - self.surface_layer[ni, nj])
                            info_density += gradient
                
                # Add quantum uncertainty contribution
                quantum_amplitude = math.sqrt(self.quantum_field[i, j, 0]**2 + self.quantum_field[i, j, 1]**2)
                info_density += quantum_amplitude * 0.5
                
                self.information_field[i, j] += (info_density - self.information_field[i, j]) * dt * 3
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get comprehensive consciousness state"""
        # Calculate global metrics
        total_neural_activity = sum(neuron.activation for neuron in self.cosmic_neurons)
        avg_neural_activity = total_neural_activity / len(self.cosmic_neurons)
        
        total_synaptic_strength = sum(synapse.strength for synapse in self.cosmic_synapses)
        avg_synaptic_strength = total_synaptic_strength / len(self.cosmic_synapses)
        
        quantum_coherence = np.mean([neuron.quantum_coherence for neuron in self.cosmic_neurons])
        entropy_level = np.mean([neuron.entropy_level for neuron in self.cosmic_neurons])
        
        # Layer activities
        surface_activity = np.mean(self.surface_layer)
        middle_activity = np.mean(self.middle_layer)
        deep_activity = np.mean(self.deep_layer)
        meta_activity = np.mean(self.meta_layer)
        
        # Information metrics
        total_information = np.sum(self.information_field)
        quantum_uncertainty = np.mean(np.sqrt(self.quantum_field[:, :, 0]**2 + self.quantum_field[:, :, 1]**2))
        
        return {
            "cosmic_time": self.cosmic_time,
            "breathing_phase": self.breathing_phase,
            "quantum_phase": self.quantum_phase,
            "neural_activity": avg_neural_activity,
            "synaptic_strength": avg_synaptic_strength,
            "quantum_coherence": quantum_coherence,
            "entropy_level": entropy_level,
            "surface_consciousness": surface_activity,
            "subconscious_activity": middle_activity,
            "unconscious_substrate": deep_activity,
            "meta_awareness": meta_activity,
            "information_density": total_information,
            "quantum_uncertainty": quantum_uncertainty,
            "neuron_count": len(self.cosmic_neurons),
            "synapse_count": len(self.cosmic_synapses)
        }
    
    def get_thinking_mode_preference(self) -> str:
        """Determine optimal thinking mode based on consciousness state"""
        state = self.get_consciousness_state()
        
        logical_score = (state["synaptic_strength"] * 2 + 
                        state["surface_consciousness"] * 1.5 +
                        (1 - state["quantum_uncertainty"]) * 1.2)
        
        creative_score = (state["quantum_uncertainty"] * 2 +
                         state["entropy_level"] / 5 +
                         state["meta_awareness"] * 1.3)
        
        if logical_score > creative_score + 0.3:
            return "thinking"
        elif creative_score > logical_score + 0.3:
            return "non-thinking"
        else:
            return "auto" 