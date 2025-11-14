"""Audio processing module for transcribing audio files."""
import os
from pathlib import Path
from typing import Optional
from openai import AzureOpenAI
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger


class AudioProcessor:
    """Process audio files and convert to text transcripts."""

    def __init__(self):
        """Initialize the audio processor."""
        self.config = get_config()
        self.logger = setup_logger(__name__)

        # Configure Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.config.get('azure_openai.api_key'),
            api_version=self.config.get('azure_openai.api_version'),
            azure_endpoint=self.config.get('azure_openai.endpoint')
        )

        # Supported audio formats
        self.supported_formats = self.config.get('audio.supported_formats', ['mp3', 'wav', 'm4a', 'ogg'])
        self.max_file_size_mb = self.config.get('audio.max_file_size_mb', 25)
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file to text using Azure OpenAI Whisper.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text or None if transcription fails
        """
        self.logger.info(f"Starting audio transcription for: {audio_file_path}")
        
        try:
            # Validate file
            if not self._validate_audio_file(audio_file_path):
                return None
            
            # Read audio file
            with open(audio_file_path, 'rb') as audio_file:
                # Use Azure OpenAI Whisper
                deployment_name = self.config.get('audio.whisper_deployment', 'whisper-1')
                self.logger.info(f"Using Whisper deployment: {deployment_name}")

                response = self.client.audio.transcriptions.create(
                    model=deployment_name,
                    file=audio_file
                )

                transcript = response.text
                self.logger.info(f"Audio transcription completed successfully. Transcript length: {len(transcript)} chars")
                self.logger.info(f"Transcript preview: {transcript[:200]}...")
                return transcript

        except FileNotFoundError:
            self.logger.error(f"Audio file not found: {audio_file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error during audio transcription: {e}", exc_info=True)
            return None
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file format and size.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if valid, False otherwise
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check file extension
        file_extension = path.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            self.logger.error(f"Unsupported audio format: {file_extension}")
            return False
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            self.logger.error(f"File size ({file_size_mb:.2f}MB) exceeds maximum ({self.max_file_size_mb}MB)")
            return False
        
        return True
    
    def process_audio_to_analysis(self, audio_file_path: str) -> dict:
        """Process audio file and return transcript.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with transcript or error
        """
        transcript = self.transcribe_audio(audio_file_path)
        
        if transcript:
            return {
                "success": True,
                "transcript": transcript,
                "source": "audio",
                "file": audio_file_path
            }
        else:
            return {
                "success": False,
                "error": "Failed to transcribe audio file",
                "file": audio_file_path
            }

