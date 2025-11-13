"""FastAPI application for sales transcript analysis."""
import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from src.api.models import (
    TextAnalysisRequest,
    AnalysisResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    InputType
)
from src.agent.transcript_analyzer import TranscriptAnalyzer
from src.agent.audio_processor import AudioProcessor
from src.agent.vector_store import MilvusVectorStore
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.utils.document_processor import DocumentProcessor


# Initialize configuration and logger
config = get_config()
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.get('fastapi.title', 'Sales Transcript Analysis API'),
    description=config.get('fastapi.description', 'API for analyzing sales conversations'),
    version=config.get('fastapi.version', '1.0.0')
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
transcript_analyzer = TranscriptAnalyzer()
audio_processor = AudioProcessor()

# Try to initialize Milvus, but continue without it if it fails
try:
    vector_store = MilvusVectorStore()
    MILVUS_ENABLED = True
    logger.info("Milvus vector store initialized successfully")
except Exception as e:
    vector_store = None
    MILVUS_ENABLED = False
    logger.warning(f"Milvus not available: {e}. Search functionality will be disabled.")

# Create temp directory for audio uploads
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - Web UI for file upload."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Transcript Analysis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .input-section {
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                color: #333;
                font-weight: 600;
                font-size: 1.1em;
            }
            select, textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            select:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                min-height: 200px;
                resize: vertical;
                font-family: 'Courier New', monospace;
            }
            .file-upload {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background: #f8f9ff;
            }
            .file-upload:hover {
                background: #e8ebff;
                border-color: #764ba2;
            }
            .file-upload input {
                display: none;
            }
            .file-info {
                margin-top: 15px;
                color: #667eea;
                font-weight: 600;
            }
            button {
                width: 100%;
                padding: 18px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .results {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9ff;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }
            .results h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .result-section {
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
            }
            .result-section h4 {
                color: #764ba2;
                margin-bottom: 10px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #667eea;
                font-size: 1.2em;
            }
            .error {
                background: #ffe0e0;
                border-left-color: #ff4444;
                color: #cc0000;
            }
            .hidden { display: none; }
            .supported-formats {
                text-align: center;
                color: #888;
                font-size: 0.9em;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Sales Transcript Analysis</h1>
            <p class="subtitle">AI-Powered Analysis with Azure OpenAI & Milvus</p>

            <div class="input-section">
                <label for="inputType">Select Input Type:</label>
                <select id="inputType" onchange="toggleInputType()">
                    <option value="text">Text Transcript</option>
                    <option value="file">Upload File (PDF, Word, CSV, Excel, Audio)</option>
                </select>
            </div>

            <div id="textInput" class="input-section">
                <label for="transcript">Paste Your Transcript:</label>
                <textarea id="transcript" placeholder="Sales Rep: Hi! I'm calling about our CRM solution.

Client: We need a system for 100 users with mobile access.

Sales Rep: Perfect! Our Enterprise plan supports that at $45 per user monthly.

Client: That fits our $5,000 budget. What's next?"></textarea>
            </div>

            <div id="fileInput" class="input-section hidden">
                <label>Upload File:</label>
                <div class="file-upload" onclick="document.getElementById('fileUpload').click()">
                    <input type="file" id="fileUpload" accept=".pdf,.docx,.doc,.csv,.xlsx,.xls,.txt,.mp3,.wav,.m4a,.ogg" onchange="handleFileSelect(event)">
                    <div>
                        <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                        <p style="font-size: 1.2em; color: #667eea; font-weight: 600;">Click to upload or drag & drop</p>
                        <p class="supported-formats">Supported: PDF, Word, CSV, Excel, TXT, MP3, WAV, M4A, OGG</p>
                    </div>
                    <div id="fileInfo" class="file-info"></div>
                </div>
            </div>

            <button onclick="analyzeTranscript()" id="analyzeBtn">🚀 Analyze Transcript</button>

            <div id="results" class="hidden"></div>
        </div>

        <script>
            let selectedFile = null;

            function toggleInputType() {
                const inputType = document.getElementById('inputType').value;
                const textInput = document.getElementById('textInput');
                const fileInput = document.getElementById('fileInput');

                if (inputType === 'text') {
                    textInput.classList.remove('hidden');
                    fileInput.classList.add('hidden');
                    selectedFile = null;
                } else {
                    textInput.classList.add('hidden');
                    fileInput.classList.remove('hidden');
                }
            }

            function handleFileSelect(event) {
                selectedFile = event.target.files[0];
                if (selectedFile) {
                    const fileInfo = document.getElementById('fileInfo');
                    const sizeMB = (selectedFile.size / (1024 * 1024)).toFixed(2);
                    fileInfo.innerHTML = `✅ Selected: ${selectedFile.name} (${sizeMB} MB)`;
                }
            }

            async function analyzeTranscript() {
                const inputType = document.getElementById('inputType').value;
                const resultsDiv = document.getElementById('results');
                const analyzeBtn = document.getElementById('analyzeBtn');

                // Show loading
                resultsDiv.className = 'results';
                resultsDiv.innerHTML = '<div class="loading">⏳ Analyzing... Please wait...</div>';
                resultsDiv.classList.remove('hidden');
                analyzeBtn.disabled = true;

                try {
                    let response;

                    if (inputType === 'text') {
                        const transcript = document.getElementById('transcript').value;
                        if (!transcript.trim()) {
                            throw new Error('Please enter a transcript');
                        }

                        response = await fetch('/analyze/text', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ transcript: transcript })
                        });
                    } else {
                        if (!selectedFile) {
                            throw new Error('Please select a file');
                        }

                        const formData = new FormData();
                        formData.append('file', selectedFile);

                        // Determine endpoint based on file type
                        const fileExt = selectedFile.name.split('.').pop().toLowerCase();
                        const audioFormats = ['mp3', 'wav', 'm4a', 'ogg'];
                        const endpoint = audioFormats.includes(fileExt) ? '/analyze/audio' : '/analyze/file';

                        response = await fetch(endpoint, {
                            method: 'POST',
                            body: formData
                        });
                    }

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Analysis failed');
                    }

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    resultsDiv.className = 'results error';
                    resultsDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                } finally {
                    analyzeBtn.disabled = false;
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');

                // Extract analysis from response
                const analysis = data.analysis || data;

                let html = '<h3>✅ Analysis Complete</h3>';

                // Requirements
                if (analysis.requirements && analysis.requirements.length > 0) {
                    html += '<div class="result-section"><h4>📋 Requirements</h4><ul>';
                    analysis.requirements.forEach(req => {
                        html += `<li><strong>${req.requirement}</strong> (Priority: ${req.priority})</li>`;
                    });
                    html += '</ul></div>';
                }

                // Recommendations
                if (analysis.recommendations && analysis.recommendations.length > 0) {
                    html += '<div class="result-section"><h4>💡 Recommendations</h4><ul>';
                    analysis.recommendations.forEach(rec => {
                        html += `<li><strong>${rec.product}</strong>: ${rec.rationale}</li>`;
                    });
                    html += '</ul></div>';
                }

                // Summary
                if (analysis.summary) {
                    html += '<div class="result-section"><h4>📝 Summary</h4>';
                    if (analysis.summary.overview) {
                        html += `<p><strong>Overview:</strong> ${analysis.summary.overview}</p>`;
                    }
                    if (analysis.summary.pain_points) {
                        const painPoints = Array.isArray(analysis.summary.pain_points)
                            ? analysis.summary.pain_points.join(', ')
                            : analysis.summary.pain_points;
                        html += `<p><strong>Pain Points:</strong> ${painPoints}</p>`;
                    }
                    if (analysis.summary.next_steps) {
                        const nextSteps = Array.isArray(analysis.summary.next_steps)
                            ? analysis.summary.next_steps.join(', ')
                            : analysis.summary.next_steps;
                        html += `<p><strong>Next Steps:</strong> ${nextSteps}</p>`;
                    }
                    html += '</div>';
                }

                // Action Items
                if (analysis.action_items && analysis.action_items.length > 0) {
                    html += '<div class="result-section"><h4>✅ Action Items</h4><ul>';
                    analysis.action_items.forEach(item => {
                        html += `<li><strong>${item.action}</strong> - ${item.owner} (${item.priority})</li>`;
                    });
                    html += '</ul></div>';
                }

                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.get('fastapi.version', '1.0.0'),
        services={
            "api": "running",
            "llm": "configured",
            "milvus": "connected"
        }
    )


@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_transcript(request: TextAnalysisRequest):
    """Analyze a text transcript.
    
    Args:
        request: Text analysis request containing the transcript
        
    Returns:
        Analysis results including requirements, recommendations, and summary
    """
    try:
        logger.info("Received text transcript analysis request")
        
        # Generate transcript ID if not provided
        transcript_id = request.transcript_id or str(uuid.uuid4())
        
        # Analyze transcript
        analysis_result = transcript_analyzer.analyze_transcript(request.transcript)
        
        # Check for errors in analysis
        if "error" in analysis_result:
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                transcript=request.transcript,
                error=analysis_result["error"],
                source_type=InputType.TEXT
            )
        
        # Store in database if requested
        if request.store_in_db:
            vector_store.store_transcript(
                transcript_id=transcript_id,
                transcript_text=request.transcript,
                analysis_result=analysis_result,
                source_type=InputType.TEXT
            )
        
        return AnalysisResponse(
            success=True,
            transcript_id=transcript_id,
            transcript=request.transcript,
            analysis=analysis_result,
            source_type=InputType.TEXT
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/audio", response_model=AnalysisResponse)
async def analyze_audio_transcript(
    file: UploadFile = File(..., description="Audio file (mp3, wav, m4a, ogg)"),
    transcript_id: Optional[str] = Form(None),
    store_in_db: bool = Form(True)
):
    """Analyze an audio file transcript.
    
    Args:
        file: Audio file upload
        transcript_id: Optional unique identifier
        store_in_db: Whether to store in database
        
    Returns:
        Analysis results including requirements, recommendations, and summary
    """
    temp_file_path = None
    
    try:
        logger.info(f"Received audio file analysis request: {file.filename}")
        
        # Generate transcript ID if not provided
        transcript_id = transcript_id or str(uuid.uuid4())
        
        # Save uploaded file temporarily
        file_extension = Path(file.filename).suffix
        temp_file_path = TEMP_DIR / f"{transcript_id}{file_extension}"
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe audio
        transcript_text = audio_processor.transcribe_audio(str(temp_file_path))
        
        if not transcript_text:
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                error="Failed to transcribe audio file",
                source_type=InputType.AUDIO
            )
        
        # Analyze transcript
        analysis_result = transcript_analyzer.analyze_transcript(transcript_text)
        
        # Check for errors in analysis
        if "error" in analysis_result:
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                transcript=transcript_text,
                error=analysis_result["error"],
                source_type=InputType.AUDIO
            )
        
        # Store in database if requested
        if store_in_db:
            vector_store.store_transcript(
                transcript_id=transcript_id,
                transcript_text=transcript_text,
                analysis_result=analysis_result,
                source_type=InputType.AUDIO
            )
        
        return AnalysisResponse(
            success=True,
            transcript_id=transcript_id,
            transcript=transcript_text,
            analysis=analysis_result,
            source_type=InputType.AUDIO
        )
        
    except Exception as e:
        logger.error(f"Error analyzing audio transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")


@app.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(..., description="Document file (PDF, Word, CSV, Excel, TXT)"),
    transcript_id: Optional[str] = Form(None),
    store_in_db: bool = Form(True)
):
    """Analyze a document file (PDF, Word, CSV, Excel, TXT).

    Args:
        file: Document file upload
        transcript_id: Optional unique identifier
        store_in_db: Whether to store in database

    Returns:
        Analysis results including requirements, recommendations, and summary
    """
    try:
        logger.info(f"Received file analysis request: {file.filename}")

        # Generate transcript ID if not provided
        transcript_id = transcript_id or str(uuid.uuid4())

        # Read file content
        file_content = await file.read()

        # Extract text from file
        try:
            transcript_text = DocumentProcessor.process_file(file.filename, file_content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")

        if not transcript_text or not transcript_text.strip():
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                error="No text could be extracted from the file",
                source_type=InputType.TEXT
            )

        logger.info(f"Extracted {len(transcript_text)} characters from {file.filename}")

        # Analyze the extracted text
        analysis_result = transcript_analyzer.analyze_transcript(transcript_text)

        # Store in vector database if requested
        if store_in_db and vector_store:
            try:
                vector_store.store_transcript(
                    transcript_id=transcript_id,
                    transcript_text=transcript_text,
                    metadata={
                        "source": "file_upload",
                        "filename": file.filename,
                        "file_type": Path(file.filename).suffix,
                        **analysis_result
                    }
                )
                logger.info(f"Stored file analysis in vector database: {transcript_id}")
            except Exception as e:
                logger.warning(f"Failed to store in vector database: {e}")
                analysis_result["storage_warning"] = "Analysis completed but not stored in database"

        return AnalysisResponse(
            success=True,
            transcript_id=transcript_id,
            transcript=transcript_text,
            analysis=analysis_result,
            source_type=InputType.TEXT
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_transcripts(request: SearchRequest):
    """Search for similar transcripts.
    
    Args:
        request: Search request with query text
        
    Returns:
        List of similar transcripts
    """
    try:
        logger.info(f"Searching for similar transcripts: {request.query}")
        
        results = vector_store.search_similar_transcripts(
            query_text=request.query,
            top_k=request.top_k
        )
        
        search_results = [
            SearchResult(
                transcript_id=r["transcript_id"],
                transcript_text=r["transcript_text"],
                analysis_result=r["analysis_result"],
                source_type=r["source_type"],
                timestamp=r["timestamp"],
                distance=r["distance"]
            )
            for r in results
        ]
        
        return SearchResponse(
            success=True,
            results=search_results,
            count=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching transcripts: {e}")
        return SearchResponse(
            success=False,
            results=[],
            count=0,
            error=str(e)
        )


@app.get("/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    """Retrieve a transcript by ID.
    
    Args:
        transcript_id: Transcript identifier
        
    Returns:
        Transcript data and analysis
    """
    try:
        result = vector_store.get_transcript_by_id(transcript_id)
        
        if result:
            return JSONResponse(content={
                "success": True,
                "data": result
            })
        else:
            raise HTTPException(status_code=404, detail="Transcript not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    host = config.get('fastapi.host', '0.0.0.0')
    port = config.get('fastapi.port', 8000)
    reload = config.get('fastapi.reload', True)
    
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)

