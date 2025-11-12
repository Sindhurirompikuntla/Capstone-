"""Quick test script to verify the setup is working."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.agent.transcript_analyzer import TranscriptAnalyzer
        from src.agent.audio_processor import AudioProcessor
        from src.agent.vector_store import MilvusVectorStore
        from src.utils.config_loader import get_config
        from src.utils.logger import setup_logger
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test if configuration loads correctly."""
    print("\nTesting configuration...")
    try:
        from src.utils.config_loader import get_config
        config = get_config()
        
        # Check critical config values
        endpoint = config.get('azure_openai.endpoint')
        api_key = config.get('azure_openai.api_key')
        
        if not endpoint or 'your-resource-name' in endpoint:
            print("⚠ Warning: Azure OpenAI endpoint not configured")
            print("  Please update config/.env with your Azure OpenAI credentials")
            return False
        
        if not api_key or 'your-azure-openai-api-key' in api_key:
            print("⚠ Warning: Azure OpenAI API key not configured")
            print("  Please update config/.env with your Azure OpenAI API key")
            return False
        
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_milvus():
    """Test Milvus connection."""
    print("\nTesting Milvus connection...")
    try:
        from src.agent.vector_store import MilvusVectorStore
        vector_store = MilvusVectorStore()
        print("✓ Milvus connection successful")
        vector_store.disconnect()
        return True
    except Exception as e:
        print(f"⚠ Milvus connection failed: {e}")
        print("  Make sure Milvus is running (docker ps)")
        print("  You can still use text analysis without Milvus")
        return False


def test_analyzer():
    """Test basic transcript analysis."""
    print("\nTesting transcript analyzer...")
    try:
        from src.agent.transcript_analyzer import TranscriptAnalyzer
        
        analyzer = TranscriptAnalyzer()
        
        # Simple test transcript
        test_transcript = """
        Sales Rep: Hello, I'm calling about our CRM solution.
        Client: Hi, we're looking for a system that can handle 100 users.
        Sales Rep: Great! Our enterprise plan supports that.
        Client: What's the pricing?
        Sales Rep: It's $50 per user per month.
        """
        
        print("  Analyzing sample transcript...")
        result = analyzer.analyze_transcript(test_transcript)
        
        if result and 'error' not in result:
            print("✓ Transcript analysis successful")
            print(f"  Found {len(result.get('requirements', []))} requirements")
            print(f"  Found {len(result.get('recommendations', []))} recommendations")
            return True
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Analyzer test failed: {e}")
        print("\nPossible issues:")
        print("  1. Check Azure OpenAI credentials in config/.env")
        print("  2. Verify deployment names match your Azure OpenAI deployments")
        print("  3. Ensure you have API quota available")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Sales Transcript Analysis Agent - Quick Test")
    print("="*70)
    
    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "Milvus": test_milvus(),
        "Analyzer": test_analyzer()
    }
    
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("="*70)
    
    all_critical_passed = results["Imports"] and results["Configuration"]
    
    if all_critical_passed:
        print("\n✓ Core components are working!")
        if not results["Milvus"]:
            print("⚠ Note: Milvus is not connected, but you can still use basic analysis")
        if results["Analyzer"]:
            print("\n🎉 Everything is working! You're ready to go!")
            print("\nNext steps:")
            print("  1. Run the API server: python run_api.py")
            print("  2. Try the examples: python examples/analyze_text.py")
            print("  3. Explore notebooks: jupyter notebook")
        else:
            print("\n⚠ Analyzer test failed. Please check your Azure OpenAI configuration.")
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")
        print("\nRefer to SETUP_GUIDE.md for detailed setup instructions.")
    
    print()


if __name__ == "__main__":
    main()

