"""
classes.py - Chat business logic and session management
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from bson import ObjectId

# Import only the repository interfaces, not the full db module
from db import ChatSessionRepository, MongoDB


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Represents a single message in a chat session."""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create ChatMessage from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ChatSession:
    """Represents a complete chat session."""
    session_id: str
    title: Optional[str]
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create ChatSession from dictionary."""
        return cls(
            session_id=data["session_id"],
            title=data.get("title"),
            messages=[ChatMessage.from_dict(msg) for msg in data["messages"]],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {})
        )


class ChatSessionManager:
    """Manages chat sessions and their lifecycle."""
    
    def __init__(self, session_repository: ChatSessionRepository):
        self.repository = session_repository
        self.current_session_id: Optional[str] = None
    
    def start_new_session(self, title: Optional[str] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new chat session."""
        session_id = str(ObjectId())
        now = datetime.now(timezone.utc)
        
        session = ChatSession(
            session_id=session_id,
            title=title or f"Chat Session {now.strftime('%Y-%m-%d %H:%M')}",
            messages=[],
            created_at=now,
            updated_at=now,
            metadata=metadata
        )
        
        self.repository.create_session(session.to_dict())
        self.current_session_id = session_id
        return session_id
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load an existing chat session."""
        session_data = self.repository.get_session(session_id)
        if session_data:
            session_data.pop('_id', None)  # Remove MongoDB _id
            self.current_session_id = session_id
            return ChatSession.from_dict(session_data)
        return None
    
    def add_user_message(self, content: str) -> bool:
        """Add a user message to the current session."""
        if not self.current_session_id:
            self.start_new_session()
        
        message = ChatMessage(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now(timezone.utc)
        )
        
        return self.repository.add_message_to_session(
            self.current_session_id, 
            message.to_dict()
        )
    
    def add_assistant_message(self, content: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add an assistant message to the current session."""
        if not self.current_session_id:
            return False
        
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        success = self.repository.add_message_to_session(
            self.current_session_id, 
            message.to_dict()
        )
        
        # Auto-generate title if this is the first exchange
        if success:
            session = self.get_current_session()
            if session and len(session.messages) == 2 and session.title.startswith("Chat Session"):
                # Get the first user message for title generation
                first_user_msg = next(
                    (msg for msg in session.messages if msg.role == MessageRole.USER), 
                    None
                )
                if first_user_msg:
                    title = self._generate_session_title(first_user_msg.content)
                    self.update_session_title(title)
        
        return success
    
    def add_system_message(self, content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a system message to the current session."""
        if not self.current_session_id:
            return False
        
        message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        return self.repository.add_message_to_session(
            self.current_session_id, 
            message.to_dict()
        )
    
    def get_conversation_context(self, max_messages: int = 10, 
                               exclude_last: bool = False) -> str:
        """Get conversation context from recent messages."""
        session = self.get_current_session()
        if not session or not session.messages:
            return ""
        
        # Get recent messages
        messages = session.messages
        if exclude_last and messages:
            messages = messages[:-1]
        
        recent_messages = messages[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            role_prefix = {
                MessageRole.USER: "User",
                MessageRole.ASSISTANT: "Assistant", 
                MessageRole.SYSTEM: "System"
            }.get(msg.role, "Unknown")
            context_parts.append(f"{role_prefix}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current active session."""
        if self.current_session_id:
            return self.load_session(self.current_session_id)
        return None
    
    def update_session_title(self, title: str) -> bool:
        """Update the title of the current session."""
        if not self.current_session_id:
            return False
        
        return self.repository.update_session(
            self.current_session_id, 
            {"title": title, "updated_at": datetime.now(timezone.utc)}
        )
    
    def delete_current_session(self) -> bool:
        """Delete the current session."""
        if not self.current_session_id:
            return False
        
        success = self.repository.delete_session(self.current_session_id)
        if success:
            self.current_session_id = None
        return success
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        success = self.repository.delete_session(session_id)
        if success and session_id == self.current_session_id:
            self.current_session_id = None
        return success
    
    def list_sessions(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """List chat sessions."""
        return self.repository.list_sessions(limit, skip)
    
    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search chat sessions."""
        return self.repository.search_sessions(query, limit)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.repository.get_session_stats()
    
    def export_session(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Export a session to dictionary."""
        target_id = session_id or self.current_session_id
        if not target_id:
            return None
        
        session = self.load_session(target_id)
        if session:
            return session.to_dict()
        return None
    
    def import_session(self, session_data: Dict[str, Any]) -> str:
        """Import a session from dictionary data."""
        session = ChatSession.from_dict(session_data)
        # Generate new session ID to avoid conflicts
        session.session_id = str(ObjectId())
        session.created_at = datetime.now(timezone.utc)
        session.updated_at = datetime.now(timezone.utc)
        
        self.repository.create_session(session.to_dict())
        return session.session_id
    
    def _generate_session_title(self, first_message: str) -> str:
        """Generate a session title based on the first message."""
        # Simple title generation - could be enhanced with LLM
        words = first_message.split()[:6]  # First 6 words
        title = " ".join(words)
        if len(first_message.split()) > 6:
            title += "..."
        return title[:50]  # Limit length


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