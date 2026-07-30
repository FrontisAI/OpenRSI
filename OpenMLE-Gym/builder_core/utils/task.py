from typing import TypedDict
from langchain_core.messages import BaseMessage

class Task(TypedDict):
    name: str
    success: bool
    download: bool
    copy: bool
    web_info: bool
    perceive: bool
    describe: bool
    gen_prep: bool
    metric: bool
    prepare: bool
    retry: int
    prepares: list[BaseMessage]
    errors: list[str]
    
    