"""LangChain-based Chat Agent with Conversation Memory."""
import json
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.agent.vector_store import MilvusVectorStore


class ChatAgent:
    """LangChain-based chat agent with conversation memory and vector store retrieval."""
    
    def __init__(self):
        """Initialize the chat agent."""
        self.config = get_config()
        self.logger = setup_logger(__name__)
        
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.get('azure_openai.endpoint'),
            api_key=self.config.get('azure_openai.api_key'),
            api_version=self.config.get('azure_openai.api_version'),
            deployment_name=self.config.get('azure_openai.deployment_name'),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize conversation memory (simple list-based)
        self.chat_history_list = []
        
        # Initialize vector store
        try:
            self.vector_store = MilvusVectorStore()
            self.db_enabled = True
            self.logger.info("Vector store initialized for chat agent")
        except Exception as e:
            self.vector_store = None
            self.db_enabled = False
            self.logger.warning(f"Vector store not available: {e}")
        
        # Chat prompt template
        self.chat_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=self.config.get_prompt('chat_agent_prompt')
        )
        
        self.logger.info("Chat agent initialized successfully")
    
    def chat(self, user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user message and return response.
        
        Args:
            user_message: User's message/question
            session_id: Optional session ID for tracking conversations
            
        Returns:
            Dictionary with response and metadata
        """
        self.logger.info(f"Processing chat message: {user_message[:100]}...")
        
        try:
            # Search vector store for relevant context
            context = ""
            relevant_docs = []
            
            if self.db_enabled:
                search_results = self.vector_store.search_similar_transcripts(
                    query_text=user_message,
                    top_k=3
                )
                
                if search_results:
                    self.logger.info(f"Found {len(search_results)} relevant documents")
                    relevant_docs = search_results
                    
                    # Build context from search results
                    context_parts = []
                    for idx, result in enumerate(search_results, 1):
                        transcript_text = result.get('transcript_text', '')
                        analysis = result.get('analysis_result', {})

                        if isinstance(analysis, str):
                            try:
                                analysis = json.loads(analysis)
                            except:
                                analysis = {}

                        context_parts.append(f"Document {idx}:")
                        # Increased from 300 to 2000 characters to provide more context
                        max_context_chars = 2000
                        if len(transcript_text) > max_context_chars:
                            context_parts.append(f"Transcript: {transcript_text[:max_context_chars]}...")
                        else:
                            context_parts.append(f"Transcript: {transcript_text}")

                        if analysis:
                            if 'summary' in analysis:
                                summary = analysis['summary']
                                if isinstance(summary, dict):
                                    overview = summary.get('overview', '')
                                    sentiment = summary.get('sentiment', '')
                                    context_parts.append(f"Summary: {overview}")
                                    if sentiment:
                                        context_parts.append(f"Sentiment: {sentiment}")
                            if 'requirements' in analysis:
                                reqs = analysis['requirements'][:3]  # First 3 requirements
                                context_parts.append(f"Requirements: {json.dumps(reqs)}")
                            if 'key_points' in analysis:
                                key_points = analysis['key_points'][:5]  # First 5 key points
                                context_parts.append(f"Key Points: {json.dumps(key_points)}")
                            if 'action_items' in analysis:
                                actions = analysis['action_items'][:3]  # First 3 action items
                                context_parts.append(f"Action Items: {json.dumps(actions)}")
                            if 'recommendations' in analysis:
                                recs = analysis['recommendations'][:2]  # First 2 recommendations
                                context_parts.append(f"Recommendations: {json.dumps(recs)}")

                        context_parts.append("")

                    context = "\n".join(context_parts)
            
            # Get chat history
            # Format chat history for prompt
            history_text = ""
            if self.chat_history_list:
                for msg in self.chat_history_list[-4:]:  # Last 4 messages (2 exchanges)
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user':
                        history_text += f"User: {content}\n"
                    elif role == 'assistant':
                        history_text += f"Assistant: {content}\n"
            
            # Build prompt
            prompt_text = self.chat_prompt.format(
                context=context if context else "No relevant documents found in database.",
                question=user_message,
                chat_history=history_text if history_text else "No previous conversation."
            )
            
            # Get response from LLM
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            answer = response.content

            # Save to chat history
            self.chat_history_list.append({"role": "user", "content": user_message})
            self.chat_history_list.append({"role": "assistant", "content": answer})
            
            self.logger.info("Chat response generated successfully")
            
            return {
                "success": True,
                "answer": answer,
                "relevant_documents": len(relevant_docs),
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "answer": "I apologize, but I encountered an error processing your message."
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.chat_history_list = []
        self.logger.info("Conversation memory cleared")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history.

        Returns:
            List of message dictionaries
        """
        return self.chat_history_list

