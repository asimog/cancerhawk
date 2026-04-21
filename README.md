# CancerHawk

Open source autonomous cancer research simulation with multi-agent godmode agents.

## Local Setup

Run the setup script to install dependencies:

```bash
./setup.bat
```

This will:
- Install Ollama for local LLM inference
- Start Neo4j database via Docker
- Pull required models (qwen3.5:27b, nomic-embed-text)
- Configure MiroShark environment

## Running the Simulation

1. Start the local interface:
   - Open `index.html` in your browser
   - Features particle-based research visualization and pretext text interfaces

2. Run MiroShark simulation:
   ```bash
   cd MiroShark
   ./miroshark
   ```

3. Analyze CancerHawk data:
   - Use the exploratory notebook for TCGA cancer research
   - Integrate findings into MiroShark simulations

## Autonomous Operation

The system runs autonomous research cycles:
- Particle visualization updates every 10 minutes
- Simulation results committed to GitHub
- Godmode agents simulate cancer research breakthroughs

## Publishing

Code and results are published to https://github.com/asimog/cancerhawk