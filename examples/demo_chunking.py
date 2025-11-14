"""Demonstration of Text Chunking with LangChain."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.text_chunker import TextChunker
from src.utils.logger import setup_logger

# Sample long sales transcript
SAMPLE_TRANSCRIPT = """
Sales Representative: Good morning! Thank you for taking the time to meet with me today. 
I understand you're looking for a comprehensive CRM solution for your growing business.

Client: Yes, that's correct. We've been using spreadsheets for the past few years, 
but as we've grown to 50 employees, it's becoming unmanageable. We need something more robust.

Sales Representative: I completely understand. Many of our clients were in the same position 
before switching to our platform. Can you tell me more about your specific pain points?

Client: Well, our main issues are tracking customer interactions across multiple team members, 
managing our sales pipeline, and generating reports for our quarterly reviews. We also need 
something that integrates with our existing email system.

Sales Representative: Those are all excellent points. Our CRM platform addresses each of those 
needs. Let me walk you through how we can help. First, regarding tracking customer interactions, 
our system provides a unified view of all customer touchpoints - emails, calls, meetings, and notes.

Client: That sounds promising. What about the sales pipeline management?

Sales Representative: Great question! Our pipeline management feature allows you to visualize 
your entire sales process from lead generation to closing. You can customize stages based on 
your specific workflow, set up automated reminders, and track conversion rates at each stage.

Client: Interesting. And the reporting capabilities?

Sales Representative: Our reporting module is quite comprehensive. You can generate custom reports 
on virtually any metric - sales performance, customer engagement, revenue forecasts, and more. 
Reports can be scheduled to run automatically and delivered to stakeholders via email.

Client: What about pricing? We have a budget of around $10,000 annually.

Sales Representative: Based on your team size of 50 employees and the features you need, 
I'd recommend our Professional plan at $150 per user per month. That would be $7,500 per month 
or $90,000 annually. However, we offer a 20% discount for annual commitments, bringing it down 
to $72,000 per year.

Client: That's significantly higher than our budget. Do you have any other options?

Sales Representative: I understand budget constraints are important. Let me see what we can do. 
For a company your size, we could potentially offer our Standard plan at $100 per user per month 
with a 25% discount for annual commitment. That would be $45,000 annually, which is closer to 
your budget range.

Client: That's still quite a bit more than we planned. What features would we lose with the 
Standard plan?

Sales Representative: The Standard plan includes all core CRM features - contact management, 
pipeline tracking, basic reporting, and email integration. The main differences from Professional 
are advanced analytics, custom workflows, and API access for third-party integrations.

Client: We definitely need the API access for our accounting software integration. Is there 
any way to add that to the Standard plan?

Sales Representative: Let me check with my manager. We might be able to create a custom package 
that includes the Standard plan features plus API access. Would you be available for a follow-up 
call next week to discuss this further?

Client: Yes, that would work. How about Tuesday at 2 PM?

Sales Representative: Perfect! I'll send you a calendar invite and come prepared with a custom 
proposal that fits your budget and requirements. In the meantime, I'll also arrange a demo 
account so you can explore the platform.

Client: That sounds great. I look forward to it.

Sales Representative: Excellent! Thank you for your time today. I'll follow up with the demo 
access by end of day and see you next Tuesday.
"""


def main():
    """Demonstrate chunking functionality."""
    logger = setup_logger(__name__)
    
    print("\n" + "=" * 80)
    print("TEXT CHUNKING DEMONSTRATION WITH LANGCHAIN")
    print("=" * 80)
    
    # Initialize chunker
    chunker = TextChunker()
    
    print(f"\n📄 Original Text Length: {len(SAMPLE_TRANSCRIPT)} characters")
    print(f"📄 Original Text Preview (first 200 chars):")
    print(f"   {SAMPLE_TRANSCRIPT[:200]}...\n")
    
    # Method 1: Recursive Character Splitter
    print("\n" + "-" * 80)
    print("METHOD 1: Recursive Character Text Splitter (Recommended)")
    print("-" * 80)
    
    recursive_chunks = chunker.chunk_text_recursive(SAMPLE_TRANSCRIPT)
    recursive_stats = chunker.get_chunk_stats(recursive_chunks)
    
    print(f"\n✓ Total Chunks: {recursive_stats['total_chunks']}")
    print(f"✓ Total Characters: {recursive_stats['total_characters']}")
    print(f"✓ Average Chunk Size: {recursive_stats['avg_chunk_size']} chars")
    print(f"✓ Min Chunk Size: {recursive_stats['min_chunk_size']} chars")
    print(f"✓ Max Chunk Size: {recursive_stats['max_chunk_size']} chars")
    
    print(f"\n📝 First Chunk (length: {len(recursive_chunks[0])}):")
    print(f"   {recursive_chunks[0][:300]}...")
    
    if len(recursive_chunks) > 1:
        print(f"\n📝 Second Chunk (length: {len(recursive_chunks[1])}):")
        print(f"   {recursive_chunks[1][:300]}...")
    
    # Method 2: Token-based Splitter
    print("\n\n" + "-" * 80)
    print("METHOD 2: Token-based Text Splitter")
    print("-" * 80)
    
    token_chunks = chunker.chunk_text_by_tokens(SAMPLE_TRANSCRIPT)
    token_stats = chunker.get_chunk_stats(token_chunks)
    
    print(f"\n✓ Total Chunks: {token_stats['total_chunks']}")
    print(f"✓ Average Chunk Size: {token_stats['avg_chunk_size']} chars")
    print(f"✓ Min/Max Size: {token_stats['min_chunk_size']}/{token_stats['max_chunk_size']} chars")
    
    # Method 3: Document Chunks with Metadata
    print("\n\n" + "-" * 80)
    print("METHOD 3: Document Chunks with Metadata")
    print("-" * 80)
    
    doc_chunks = chunker.chunk_documents(
        SAMPLE_TRANSCRIPT,
        metadata={
            'source': 'sales_call',
            'date': '2024-01-15',
            'client': 'ABC Corp'
        }
    )
    
    print(f"\n✓ Total Document Chunks: {len(doc_chunks)}")
    
    if doc_chunks:
        print(f"\n📋 First Document Chunk Info:")
        print(f"   Chunk Index: {doc_chunks[0]['chunk_index']}")
        print(f"   Total Chunks: {doc_chunks[0]['total_chunks']}")
        print(f"   Chunk Size: {doc_chunks[0]['chunk_size']} chars")
        print(f"   Metadata: {doc_chunks[0].get('metadata', {})}")
        print(f"   Text Preview: {doc_chunks[0]['text'][:200]}...")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

