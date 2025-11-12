"""Example script for using the FastAPI client."""
import requests
import json
from pathlib import Path


# API base URL
BASE_URL = "http://localhost:8000"


def check_health():
    """Check API health."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()


def analyze_text_transcript():
    """Analyze a text transcript via API."""
    # Load sample transcript
    transcript_path = Path(__file__).parent.parent / "data" / "text" / "sample_transcript_1.txt"
    
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    
    # Prepare request
    payload = {
        "transcript": transcript,
        "transcript_id": "example-001",
        "store_in_db": True
    }
    
    # Send request
    print("Analyzing text transcript...")
    response = requests.post(f"{BASE_URL}/analyze/text", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("Analysis successful!")
        print(json.dumps(result, indent=2))
        return result.get("transcript_id")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def analyze_audio_file():
    """Analyze an audio file via API."""
    # Path to audio file
    audio_path = Path(__file__).parent.parent / "data" / "audio" / "sample_conversation_1.mp3"
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        print("Please add an audio file to test this functionality.")
        return None
    
    # Prepare request
    files = {
        'file': open(audio_path, 'rb')
    }
    data = {
        'transcript_id': 'example-audio-001',
        'store_in_db': True
    }
    
    # Send request
    print("Analyzing audio file...")
    response = requests.post(f"{BASE_URL}/analyze/audio", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("Analysis successful!")
        print(json.dumps(result, indent=2))
        return result.get("transcript_id")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def search_transcripts(query):
    """Search for similar transcripts."""
    payload = {
        "query": query,
        "top_k": 3
    }
    
    print(f"\nSearching for: '{query}'")
    response = requests.post(f"{BASE_URL}/search", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['count']} results:")
        
        for i, item in enumerate(result['results'], 1):
            print(f"\nResult {i}:")
            print(f"  ID: {item['transcript_id']}")
            print(f"  Distance: {item['distance']:.4f}")
            print(f"  Preview: {item['transcript_text'][:150]}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def get_transcript(transcript_id):
    """Retrieve a specific transcript."""
    print(f"\nRetrieving transcript: {transcript_id}")
    response = requests.get(f"{BASE_URL}/transcript/{transcript_id}")
    
    if response.status_code == 200:
        result = response.json()
        print("Transcript retrieved successfully!")
        print(json.dumps(result, indent=2)[:500] + "...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def main():
    """Run example API calls."""
    print("="*80)
    print("Sales Transcript Analysis API - Example Client")
    print("="*80)
    print()
    
    # Check health
    check_health()
    
    # Analyze text transcript
    print("\n" + "-"*80)
    transcript_id = analyze_text_transcript()
    
    # Search for similar transcripts
    if transcript_id:
        print("\n" + "-"*80)
        search_transcripts("CRM system with mobile access")
        
        print("\n" + "-"*80)
        search_transcripts("security and compliance")
        
        # Retrieve specific transcript
        print("\n" + "-"*80)
        get_transcript(transcript_id)
    
    # Try audio analysis (if audio file exists)
    print("\n" + "-"*80)
    analyze_audio_file()
    
    print("\n" + "="*80)
    print("Example completed!")
    print("="*80)


if __name__ == "__main__":
    main()

