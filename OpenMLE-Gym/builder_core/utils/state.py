from typing import Annotated, TypedDict, Sequence, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    """State container for the agent workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task_result: Optional[dict]