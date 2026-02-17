"""Qualia API Models - Request/Response structures"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class EffortLevel(str, Enum):
    ADAPTIVE = "adaptive"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    COMPACTER = "compacter"

class Message(BaseModel):
    role: MessageRole
    content: str

class QualiaRequest(BaseModel):
    model: str = Field(default="defrateio/qualia-v1-20260221")
    max_tokens: int = Field(default=4096, le=4294967296)
    effort: EffortLevel = Field(default=EffortLevel.ADAPTIVE)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    messages: List[Message]
    mcp: bool = Field(default=True)

class MessageType(str, Enum):
    PLAN = "plan"
    THINK = "think"
    IMAGE = "image"
    TEXT = "text"
    MCP = "mcp"

class QualiaResponse(BaseModel):
    msg_type: MessageType
    text: Optional[str] = None
    content: Optional[str] = None  # Base64 encoded
    mcp_content: Optional[dict] = None
    continue_: bool = Field(alias="continue")
    stop_reason: Optional[str] = None
    stop_seq: Optional[str] = None
    model: str
    usage_billing: Optional[float] = None
    use_tier: Optional[str] = "standard"

    class Config:
        populate_by_name = True
