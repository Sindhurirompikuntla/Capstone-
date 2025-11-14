# Agentic AI Architecture with LangChain

## Overview
The chat agent has been transformed from a simple LLM wrapper into a **true agentic AI system** using LangChain's **ReAct (Reasoning + Acting) framework**.

---

## What is Agentic AI?

**Agentic AI** refers to AI systems that can:
1. **Reason** about problems and decide what actions to take
2. **Use tools** to gather information or perform tasks
3. **Plan** multi-step solutions
4. **Adapt** based on observations from tool usage
5. **Maintain memory** of conversations and context

### Before (Simple LLM Wrapper):
```
User Question → LLM → Answer
```

### After (Agentic AI with ReAct):
```
User Question → Agent Reasoning → Tool Selection → Tool Execution → Observation → Reasoning → Final Answer
```

---

## Architecture Components

### 1. **LangChain ReAct Agent**
The core of the agentic system using the **ReAct pattern**:

- **Re**asoning: Agent thinks about what to do
- **Act**ing: Agent takes actions using tools
- **Observation**: Agent observes results
- **Iteration**: Repeats until it has enough information

<augment_code_snippet path="src/agent/chat_agent.py" mode="EXCERPT">
```python
# Create the ReAct agent
self.agent = create_react_agent(
    llm=self.llm,
    tools=self.tools,
    prompt=react_prompt
)

# Create agent executor
self.agent_executor = AgentExecutor(
    agent=self.agent,
    tools=self.tools,
    memory=self.memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)
```
</augment_code_snippet>

---

### 2. **Tools**
The agent has access to tools it can use to gather information:

#### **Tool: search_database**
Searches the Milvus vector database for relevant sales transcripts.

<augment_code_snippet path="src/agent/chat_agent.py" mode="EXCERPT">
```python
search_tool = Tool(
    name="search_database",
    func=search_database,
    description="Search the vector database for relevant sales transcripts, requirements, and past conversations."
)
```
</augment_code_snippet>

**When the agent uses this tool:**
- User asks: "What were the client's requirements?"
- Agent thinks: "I need to search the database"
- Agent uses: `search_database("client requirements")`
- Agent observes: Retrieved documents with requirements
- Agent answers: "The client required SAP integration and mobile app support."

---

### 3. **Conversation Memory**
Uses LangChain's `ConversationBufferMemory` to maintain context:

<augment_code_snippet path="src/agent/chat_agent.py" mode="EXCERPT">
```python
self.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)
```
</augment_code_snippet>

**Benefits:**
- Remembers previous questions and answers
- Can reference earlier parts of the conversation
- Maintains context across multiple turns

---

### 4. **ReAct Prompt Template**
Custom prompt that guides the agent's reasoning process:

```
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search_database]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer (crisp and short, 1-2 sentences max)
```

---

## How the Agent Works

### Example Interaction:

**User:** "What did the client say about pricing?"

**Agent's Internal Process:**
```
Thought: I need to search the database for information about pricing discussions
Action: search_database
Action Input: "client pricing discussion"
Observation: [Retrieved documents with pricing information]
Thought: I now have the information about pricing
Final Answer: The client mentioned a budget of $50,000 and requested flexible payment terms.
```

---

## Key Differences from Before

| Aspect | Before (Simple Wrapper) | After (Agentic AI) |
|--------|------------------------|-------------------|
| **Decision Making** | None - always calls LLM | Agent decides when to use tools |
| **Tool Usage** | Manual search in code | Agent autonomously uses tools |
| **Reasoning** | No reasoning process | Explicit reasoning steps |
| **Adaptability** | Fixed flow | Adapts based on observations |
| **Transparency** | Black box | Visible reasoning (verbose=True) |
| **Memory** | Simple list | LangChain ConversationBufferMemory |

---

## Benefits of Agentic Approach

1. **Autonomous Decision Making**: Agent decides when to search database vs. answer directly
2. **Multi-Step Reasoning**: Can break down complex questions into steps
3. **Tool Orchestration**: Can use multiple tools in sequence if needed
4. **Explainability**: Reasoning process is visible (when verbose=True)
5. **Extensibility**: Easy to add new tools (e.g., web search, calculations, API calls)
6. **Error Handling**: Built-in error recovery and parsing error handling
7. **Iteration Control**: Max iterations prevent infinite loops

---

## Code Structure

### Initialization
```python
class ChatAgent:
    def __init__(self):
        # 1. Initialize LLM
        self.llm = ChatLiteLLM(...)
        
        # 2. Initialize vector store
        self.vector_store = MilvusVectorStore()
        
        # 3. Initialize memory
        self.memory = ConversationBufferMemory(...)
        
        # 4. Create tools
        self.tools = self._create_tools()
        
        # 5. Create agent
        self.agent = self._create_agent()
        
        # 6. Create executor
        self.agent_executor = AgentExecutor(...)
```

### Chat Method
```python
def chat(self, user_message: str) -> Dict[str, Any]:
    # Agent executor handles everything
    result = self.agent_executor.invoke({
        "input": user_message
    })
    return {"answer": result["output"]}
```

---

## Future Extensibility

The agentic architecture makes it easy to add new capabilities:

### Potential New Tools:
1. **Web Search Tool**: Search the internet for current information
2. **Calculator Tool**: Perform calculations for pricing, ROI, etc.
3. **Email Tool**: Send follow-up emails to clients
4. **CRM Integration**: Query/update CRM systems
5. **Document Generator**: Create proposals, contracts
6. **Calendar Tool**: Schedule meetings

### Example: Adding a Calculator Tool
```python
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"

calc_tool = Tool(
    name="calculator",
    func=calculate,
    description="Perform mathematical calculations"
)
tools.append(calc_tool)
```

---

## Monitoring Agent Behavior

With `verbose=True`, you can see the agent's reasoning in logs:

```
🤖 Agent processing message: What's the total budget?
Thought: I need to search for budget information
Action: search_database
Action Input: budget
Observation: Client mentioned $50,000 budget
Thought: I now know the final answer
Final Answer: The total budget is $50,000.
✅ Agent response generated successfully
```

---

## Summary

✅ **Real Agentic AI**: Uses LangChain's ReAct framework, not just an LLM wrapper
✅ **Autonomous**: Agent decides when and how to use tools
✅ **Reasoning**: Explicit thought process visible
✅ **Memory**: Maintains conversation context
✅ **Extensible**: Easy to add new tools and capabilities
✅ **Production-Ready**: Error handling, iteration limits, parsing recovery

**The agent is now a true AI assistant that can reason, plan, and act autonomously!** 🤖

