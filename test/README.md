# Resume Analysis Pipeline

## Overview
This project implements a multi-agent pipeline for analyzing resumes. It extracts information from PDF resumes, structures the data, and performs skill gap analysis.

## Features
- PDF resume text extraction
- Structured resume data parsing
- Skill gap analysis
- Configurable GPU/CPU processing
- REST API for resume analysis with streaming support
- Docker support for containerized deployment

## Installation

### Prerequisites
- Python 3.x
- Required Python packages (install via requirements.txt)

```bash
pip install -r requirements.txt
```

### Docker
You can also run the application using Docker:

#### Ubuntu/Linux
```bash
# Build the Docker image
docker build -t resume-analyzer .

# Run the container
docker run -p 8000:8000 resume-analyzer
```

#### Windows

Make sure that you have Docker installed on your Windows machine, and running. 

```bash
# Build the Docker image
docker build -t resume-analyzer .

# Run the container
docker run -p 8080:8000 resume-analyzer
```

##### Using WSL (Windows Subsystem for Linux)
For better performance on Windows, you can use WSL:

1. Install WSL if you haven't already:
   ```bash
   wsl --install
   ```

2. Open your WSL terminal and navigate to your project directory:
   ```bash
   cd /path/to/project
   ```

3. Build and run the Docker container:
   ```bash
   docker build -t resume-analyzer .
   docker run -p 8000:8000 resume-analyzer
   ```

#### Accessing the Application
After running the container, access the API at:
- API: http://localhost:8000
- Health check: http://localhost:8000/
- API documentation: http://localhost:8000/docs

## Usage

### CLI Mode
Run the main script with an optional flag to force CPU usage:

```bash
python main.py [--cpu]
```

#### Parameters
- `--cpu`: Force Ollama to use CPU instead of GPU

### API Mode
Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check endpoint |
| `/analyze-resume-graph` | POST | Upload and analyze a PDF resume (full graph) |
| `/analyze-resume-stream-steps` | POST | Stream results from each step of the analysis pipeline |
| `/debug-langgraph` | GET | Debug information about LangGraph configuration |

### Testing the Streaming API

You can test the streaming API using the provided test script:

```bash
python test_stream_api.py
```

This script connects to the streaming API endpoint and displays results from each agent as they become available.

#### Example Request Format

```json
{
  "resume_url": "https://example.com/path/to/resume.pdf"
}
```

#### Example Response

The streaming API returns JSON events as they occur:

```json
{"status": "started", "node": "resume_parser", "message": "Starting resume_parser..."}
{"status": "completed", "node": "resume_parser", "structured_resume": {"Name": "John Doe", "Skills": ["Python", "JavaScript"]}}
{"status": "started", "node": "gap_analyzer", "message": "Starting gap_analyzer..."}
{"status": "completed", "node": "gap_analyzer", "skill_gap_analysis": "Analysis of skills..."}
{"status": "complete", "message": "Analysis complete"}
```

## Project Structure
- `main.py`: Entry point for local CLI testing
- `app.py`: FastAPI application for REST API
- `test_stream_api.py`: Script to test the streaming API
- `core/`: Core functionality modules
  - `multi_agent_pipeline.py`: Contains the pipeline graph implementation
  - `utils.py`: Utility functions for PDF extraction and device configuration
- `agents/`: Contains the various agents used in the pipeline
- `Dockerfile`: Container configuration for deployment

## Output
The pipeline generates:
1. Generated prompts
2. Structured resume data
3. Skill gap analysis
4. (Optional) Enriched profile data

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]

