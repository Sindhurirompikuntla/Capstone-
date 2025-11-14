# Text Chunking Implementation with LangChain

## Overview
This document describes the text chunking functionality implemented using **LangChain Text Splitters** integrated with **LiteLLM** for the Sales Transcript Analysis system.

---

## Architecture

### Components Updated

1. **`src/utils/text_chunker.py`** - New utility class for text chunking
2. **`src/agent/transcript_analyzer.py`** - Updated to use LangChain + LiteLLM with chunking
3. **`src/agent/chat_agent.py`** - Updated to use LangChain + LiteLLM
4. **`src/agent/sales_helper_agent.py`** - Updated to use LangChain + LiteLLM
5. **`src/agent/vector_store.py`** - Updated with chunking support and demonstration method
6. **`examples/demo_chunking.py`** - Demonstration script for chunking functionality

---

## Text Chunking Methods

### 1. Recursive Character Text Splitter (Recommended)
**Best for:** Maintaining semantic coherence in natural language text

**Features:**
- Tries to split on paragraphs first (`\n\n`)
- Falls back to sentences (`. `)
- Then words (` `)
- Finally characters
- Configurable chunk size: 2000 characters
- Overlap: 200 characters

**Usage:**
```python
from src.utils.text_chunker import TextChunker

chunker = TextChunker()
chunks = chunker.chunk_text_recursive(long_text)
```

**Example Output:**
```
✓ Total Chunks: 3
✓ Average Chunk Size: 1305 chars
✓ Min Chunk Size: 190 chars
✓ Max Chunk Size: 1871 chars
```

---

### 2. Token-based Text Splitter
**Best for:** Ensuring chunks fit within model token limits

**Features:**
- Splits based on token count (not characters)
- Chunk size: 1500 tokens (safe for 8192 token limit)
- Overlap: 150 tokens
- Prevents token limit errors in embeddings

**Usage:**
```python
chunks = chunker.chunk_text_by_tokens(long_text)
```

---

### 3. Character Text Splitter
**Best for:** Simple, predictable splitting

**Features:**
- Splits by character count
- Uses newline as separator
- Chunk size: 2000 characters
- Overlap: 200 characters

**Usage:**
```python
chunks = chunker.chunk_text_by_character(long_text)
```

---

### 4. Document Chunks with Metadata
**Best for:** Tracking chunk provenance and context

**Features:**
- Chunks text using recursive splitter
- Attaches metadata to each chunk
- Includes chunk index, total chunks, and size
- Custom metadata support

**Usage:**
```python
doc_chunks = chunker.chunk_documents(
    text,
    metadata={'source': 'sales_call', 'date': '2024-01-15'}
)
```

**Example Output:**
```python
{
    'text': 'Sales Representative: Good morning!...',
    'chunk_index': 0,
    'total_chunks': 3,
    'chunk_size': 1854,
    'metadata': {'source': 'sales_call', 'date': '2024-01-15'}
}
```

---

## LangChain Integration

### Chat Models
All components now use **`ChatLiteLLM`** from LangChain:

```python
from langchain_community.chat_models import ChatLiteLLM

llm = ChatLiteLLM(
    model=f"azure/{deployment_name}",
    temperature=0.7,
    max_tokens=1000
)
```

### Prompt Templates
Using **`PromptTemplate`** for structured prompts:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=config.get_prompt('chat_agent_prompt')
)
```

### Message Types
Using **`HumanMessage`** and **`SystemMessage`**:

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = llm.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
])
```

---

## Demonstration

### Running the Chunking Demo

```bash
python examples/demo_chunking.py
```

### Sample Output

```
================================================================================
TEXT CHUNKING DEMONSTRATION WITH LANGCHAIN
================================================================================

📄 Original Text Length: 3797 characters

--------------------------------------------------------------------------------
METHOD 1: Recursive Character Text Splitter (Recommended)
--------------------------------------------------------------------------------

✓ Total Chunks: 3
✓ Total Characters: 3915
✓ Average Chunk Size: 1305 chars
✓ Min Chunk Size: 190 chars
✓ Max Chunk Size: 1871 chars

📝 First Chunk (length: 1854):
   Sales Representative: Good morning! Thank you for taking the time...

--------------------------------------------------------------------------------
METHOD 2: Token-based Text Splitter
--------------------------------------------------------------------------------

✓ Total Chunks: 1
✓ Average Chunk Size: 3797 chars

--------------------------------------------------------------------------------
METHOD 3: Document Chunks with Metadata
--------------------------------------------------------------------------------

✓ Total Document Chunks: 3

📋 First Document Chunk Info:
   Chunk Index: 0
   Total Chunks: 3
   Chunk Size: 1854 chars
   Metadata: {'source': 'sales_call', 'date': '2024-01-15', 'client': 'ABC Corp'}
```

---

## Benefits

1. **Prevents Token Limit Errors**: Automatically handles large documents
2. **Maintains Context**: Overlap ensures continuity between chunks
3. **Semantic Coherence**: Recursive splitter preserves meaning
4. **Flexible**: Multiple splitting strategies for different use cases
5. **Metadata Tracking**: Know which chunk came from where
6. **LangChain Integration**: Unified interface with other LangChain components

---

## Updated Dependencies

```
langchain==0.3.0
langchain-community==0.3.0
langchain-core==0.3.0
langchain-text-splitters==0.3.0
litellm==1.17.9
```

---

## Server Logs

When the server starts, you'll see:

```
✓ Text chunker initialized with LangChain splitters
✓ LangChain + LiteLLM configured with Azure OpenAI deployment: gpt-4o
✓ Chat agent initialized successfully with LangChain + LiteLLM
```

---

## Next Steps

- Chunking is automatically used for long transcripts (>5000 chars)
- Statistics are logged for monitoring
- All functionality remains the same for end users
- Enhanced reliability for large documents

