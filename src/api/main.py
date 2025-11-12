"""FastAPI application for sales transcript analysis."""
import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version=config.get('fastapi.version', '1.0.0'),
        services={
            "api": "running",
            "llm": "configured",
            "milvus": "connected"
        }
    )


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

