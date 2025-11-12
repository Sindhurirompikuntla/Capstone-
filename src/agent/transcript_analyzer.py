"""Sales transcript analyzer using LiteLLM and Azure OpenAI."""
import json
import litellm
from typing import Dict, Any, Optional
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger


class TranscriptAnalyzer:
    """Analyze sales conversation transcripts using LLM."""
    
    def __init__(self):
        """Initialize the transcript analyzer."""
        self.config = get_config()
        self.logger = setup_logger(__name__)
        
        # Configure LiteLLM for Azure OpenAI
        self._setup_litellm()
    
    def _setup_litellm(self):
        """Configure LiteLLM with Azure OpenAI settings."""
        # Set Azure OpenAI configuration
        litellm.api_key = self.config.get('azure_openai.api_key')
        litellm.api_base = self.config.get('azure_openai.endpoint')
        litellm.api_version = self.config.get('azure_openai.api_version')
        
        # Set timeout and retries
        litellm.timeout = self.config.get('litellm.timeout', 60)
        litellm.num_retries = self.config.get('litellm.max_retries', 3)
        
        self.logger.info("LiteLLM configured for Azure OpenAI")
    
    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Analyze a sales conversation transcript.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing analysis results with requirements, recommendations, and summary
        """
        self.logger.info("Starting transcript analysis")
        
        try:
            # Get prompts
            system_prompt = self.config.get_prompt('system_prompt')
            analysis_prompt = self.config.get_prompt('analysis_prompt')
            
            # Format the analysis prompt with transcript
            user_prompt = analysis_prompt.format(transcript=transcript)
            
            # Call LiteLLM with Azure OpenAI
            deployment_name = self.config.get('azure_openai.deployment_name')
            model = f"azure/{deployment_name}"
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.get('azure_openai.temperature', 0.7),
                max_tokens=self.config.get('azure_openai.max_tokens', 2000),
                response_format={"type": "json_object"}
            )
            
            # Extract and parse response
            content = response.choices[0].message.content
            analysis_result = json.loads(content)
            
            self.logger.info("Transcript analysis completed successfully")
            return analysis_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._get_error_response("Failed to parse analysis results")
        except Exception as e:
            self.logger.error(f"Error during transcript analysis: {e}")
            return self._get_error_response(str(e))
    
    def extract_requirements(self, transcript: str) -> Dict[str, Any]:
        """Extract client requirements from transcript.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing extracted requirements
        """
        self.logger.info("Extracting requirements from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            requirements_prompt = self.config.get_prompt('requirements_extraction_prompt')
            
            user_prompt = requirements_prompt.format(transcript=transcript)
            
            deployment_name = self.config.get('azure_openai.deployment_name')
            model = f"azure/{deployment_name}"
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Requirements extraction completed")
            return {"requirements": content}
            
        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e}")
            return {"error": str(e)}
    
    def generate_recommendations(self, transcript: str) -> Dict[str, Any]:
        """Generate product recommendations based on transcript.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing recommendations
        """
        self.logger.info("Generating recommendations from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            recommendations_prompt = self.config.get_prompt('recommendations_prompt')
            
            user_prompt = recommendations_prompt.format(transcript=transcript)
            
            deployment_name = self.config.get('azure_openai.deployment_name')
            model = f"azure/{deployment_name}"
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Recommendations generation completed")
            return {"recommendations": content}
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {"error": str(e)}
    
    def generate_summary(self, transcript: str) -> Dict[str, Any]:
        """Generate summary of the conversation.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing summary
        """
        self.logger.info("Generating summary from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            summary_prompt = self.config.get_prompt('summary_prompt')
            
            user_prompt = summary_prompt.format(transcript=transcript)
            
            deployment_name = self.config.get('azure_openai.deployment_name')
            model = f"azure/{deployment_name}"
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Summary generation completed")
            return {"summary": content}
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response structure.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Error response dictionary
        """
        return {
            "error": error_message,
            "requirements": [],
            "recommendations": [],
            "summary": {
                "overview": "Analysis failed",
                "error": error_message
            },
            "key_points": [],
            "action_items": []
        }

