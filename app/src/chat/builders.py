from .classes import ChatSession, ChatMessage ,MessageRole

class ConversationContextBuilder:
    """Utility class for building conversation contexts."""
    
    @staticmethod
    def build_rag_context(conversation_context: str, current_question: str) -> str:
        """Build a context-aware query for RAG systems."""
        if conversation_context:
            return f"Previous conversation:\n{conversation_context}\n\nCurrent question: {current_question}"
        return current_question
    
    @staticmethod
    def build_system_prompt_with_context(base_system_prompt: str, 
                                       conversation_context: str) -> str:
        """Build a system prompt that includes conversation context."""
        if conversation_context:
            return f"{base_system_prompt}\n\nPrevious conversation context:\n{conversation_context}"
        return base_system_prompt
    
    @staticmethod
    def extract_relevant_context(session: ChatSession, 
                               max_messages: int = 5,
                               exclude_system: bool = True) -> str:
        """Extract relevant context from a session."""
        if not session.messages:
            return ""
        
        relevant_messages = []
        for msg in reversed(session.messages):
            if exclude_system and msg.role == MessageRole.SYSTEM:
                continue
            relevant_messages.append(msg)
            if len(relevant_messages) >= max_messages:
                break
        
        # Reverse to get chronological order
        relevant_messages.reverse()
        
        context_parts = []
        for msg in relevant_messages:
            role_name = "User" if msg.role == MessageRole.USER else "Assistant"
            context_parts.append(f"{role_name}: {msg.content}")
        
        return "\n".join(context_parts)
    
class ConversationContextBuilder:
    """Utility class for building conversation contexts."""
    
    @staticmethod
    def build_rag_context(conversation_context: str, current_question: str) -> str:
        """Build a context-aware query for RAG systems."""
        if conversation_context:
            return f"Previous conversation:\n{conversation_context}\n\nCurrent question: {current_question}"
        return current_question
    
    @staticmethod
    def build_system_prompt_with_context(base_system_prompt: str, 
                                       conversation_context: str) -> str:
        """Build a system prompt that includes conversation context."""
        if conversation_context:
            return f"{base_system_prompt}\n\nPrevious conversation context:\n{conversation_context}"
        return base_system_prompt
    
    @staticmethod
    def extract_relevant_context(session: ChatSession, 
                               max_messages: int = 5,
                               exclude_system: bool = True) -> str:
        """Extract relevant context from a session."""
        if not session.messages:
            return ""
        
        relevant_messages = []
        for msg in reversed(session.messages):
            if exclude_system and msg.role == MessageRole.SYSTEM:
                continue
            relevant_messages.append(msg)
            if len(relevant_messages) >= max_messages:
                break
        
        # Reverse to get chronological order
        relevant_messages.reverse()
        
        context_parts = []
        for msg in relevant_messages:
            role_name = "User" if msg.role == MessageRole.USER else "Assistant"
            context_parts.append(f"{role_name}: {msg.content}")
        
        return "\n".join(context_parts)