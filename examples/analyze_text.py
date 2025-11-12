"""Example script for analyzing text transcripts."""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.transcript_analyzer import TranscriptAnalyzer
from src.utils.logger import setup_logger


def main():
    """Analyze a text transcript."""
    # Setup logger
    logger = setup_logger(__name__)
    
    # Initialize analyzer
    logger.info("Initializing transcript analyzer...")
    analyzer = TranscriptAnalyzer()
    
    # Load sample transcript
    transcript_path = Path(__file__).parent.parent / "data" / "text" / "sample_transcript_1.txt"
    
    logger.info(f"Loading transcript from {transcript_path}")
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    
    # Analyze transcript
    logger.info("Analyzing transcript...")
    result = analyzer.analyze_transcript(transcript)
    
    # Display results
    print("\n" + "="*80)
    print("TRANSCRIPT ANALYSIS RESULTS")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "analysis_result.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

