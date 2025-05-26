# Prototype Files

⚠️ **These are outdated prototype files kept for reference only.**

These files contain the original implementations that were used to develop the organized consciousness system. They are **not functional** in the current codebase structure and should not be used directly.

## Files

- `original_setup.py` - Original setup script (superseded by `utils/setup.py`)
- `original_integration.py` - Original integration code (superseded by `core/conscious_model.py`)
- `consciousness-demo.py` - Original demo (superseded by `demos/interactive_demo.py`)
- `ollama-consciousness-bridge.py` - Original bridge (integrated into core modules)

## Current Usage

For current functionality, use the organized structure:

```bash
# Quick start
python examples/quick_start.py

# Interactive demo
python demos/interactive_demo.py

# Setup
from consciousness_qwen.utils import setup_consciousness_system
setup_consciousness_system()
```

## Purpose

These prototypes are preserved to:
- Show the evolution of the consciousness system
- Provide reference for understanding design decisions
- Maintain historical context for the project

**Do not run these files directly** - they will fail due to missing dependencies and outdated imports. 