"""
Memory Retriever Tool for the AI Data Analyst Agent.
Retrieves session context from ChromaDB.
Phase 3: Tool Implementation
"""

import time
import datetime
import chromadb
from config import settings
from tools.registry import ToolResult

def get_or_create_collection(session_id: str) -> chromadb.Collection:
    """
    Retrieves or establishes a ChromaDB collection for the current session.
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection_name = f"session_{session_id}".replace("-", "_")
    return client.get_or_create_collection(name=collection_name)

def store_qa_pair(question: str, answer: str, session_id: str) -> None:
    """
    Saves a Q&A pair to persistent memory.
    """
    try:
        collection = get_or_create_collection(session_id)
        timestamp = datetime.datetime.now().isoformat()
        
        collection.add(
            documents=[f"Q: {question} A: {answer}"],
            metadatas=[{"question": question, "timestamp": timestamp}],
            ids=[f"qa_{int(time.time() * 1000)}"]
        )
    except Exception as e:
        print(f"Memory storage failed: {e}")

def run(query: str, session_state: dict) -> ToolResult:
    """
    Finds relevant past session context using vector search.
    """
    start_time = time.time()
    session_id = session_state.get("session_id", "default_session")
    
    try:
        collection = get_or_create_collection(session_id)
        
        # Check if collection has documents
        if collection.count() == 0:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=True,
                tool_name="memory_retriever",
                output="No prior context found for this session.",
                output_type="string",
                error_message=None,
                execution_time_ms=execution_time,
                code_executed=None
            )
            
        # Search for top 3 relevant results
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        documents = results.get("documents", [[]])[0]
        
        if not documents:
            formatted_output = "No statistically relevant prior context found for the current query."
        else:
            formatted_output = "PRIOR CONTEXT:\n"
            for i, doc in enumerate(documents):
                formatted_output += f"[{i+1}] {doc}\n"
        
        # Store in session
        session_state["retrieved_memory"] = formatted_output
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            tool_name="memory_retriever",
            output=formatted_output,
            output_type="string",
            error_message=None,
            execution_time_ms=execution_time,
            code_executed=None
        )
        
    except Exception as e:
        # Memory failure should not crash the agent
        execution_time = (time.time() - start_time) * 1000
        return ToolResult(
            success=True, # Still return success but with error notice
            tool_name="memory_retriever",
            output=f"Memory recovery failed temporarily: {str(e)}",
            output_type="string",
            error_message=f"Memory Error: {str(e)}",
            execution_time_ms=execution_time,
            code_executed=None
        )
