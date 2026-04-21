@echo off
echo Setting up CancerHawk local environment...

REM Install Ollama if not present
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing Ollama...
    winget install Ollama.Ollama
)

REM Start Ollama service
echo Starting Ollama...
ollama serve

REM Pull required models
echo Pulling Ollama models...
ollama pull qwen3.5:27b
ollama pull nomic-embed-text

REM Start Neo4j with Docker
echo Starting Neo4j...
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/miroshark neo4j:5.15-community

REM Configure MiroShark
echo Configuring MiroShark...
cd MiroShark
if not exist .env (
    copy .env.example .env
    echo Please edit .env with your API keys if needed
)

echo Setup complete! Run MiroShark with: cd MiroShark && ./miroshark