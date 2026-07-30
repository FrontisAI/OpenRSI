from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pathlib import Path
if __package__:
    from .utils.nodes import NodeExecutor
    from .utils.state import AgentState
    from .tools.tools import (
        configure_task_paths,
        get_csv_summary,
        list_directory_contents,
        read_txt_md,
        save,
    )
else:
    from utils.nodes import NodeExecutor
    from utils.state import AgentState
    from tools.tools import (
        configure_task_paths,
        get_csv_summary,
        list_directory_contents,
        read_txt_md,
        save,
    )
    
    
def Graph(
    batch_name,
    competition,
    base_dir=None,
    info_csv=None,
    sample_dir=None,
    delete_raw=False,
    code_execution_mode="process",
    code_timeout=600,
):
    resolved_base = Path(base_dir).resolve() if base_dir is not None else Path("./").resolve()
    task_root = resolved_base / batch_name
    configure_task_paths(
        read_root=task_root / "data" / competition["name"] / "raw",
        save_path=task_root / "forge" / competition["name"] / "fileinfo.txt",
    )
    tools = [
                list_directory_contents, 
                get_csv_summary, 
                read_txt_md, 
                save
            ]

    Nodes = NodeExecutor(
        base_dir = resolved_base,
        batch_name = batch_name,
        todo = competition,
        tools = tools,
        info_csv = Path(info_csv).resolve() if info_csv is not None else None,
        sample_dir = Path(sample_dir).resolve() if sample_dir is not None else None,
        delete_raw = delete_raw,
        code_execution_mode = code_execution_mode,
        code_timeout = code_timeout,
    )

    graph = StateGraph(AgentState)
    graph.add_node("download", Nodes.Download)          # bool 
    graph.add_node("R1", lambda state:{"messages": []})
    graph.add_node("copy", Nodes.Copy)                  # tuple[bool, str]
    graph.add_node("R2", lambda state:{"messages": []})
    graph.add_node("web_info", Nodes.Scrape)            # dict
    graph.add_node("R3", lambda state:{"messages": []})
    graph.add_node("perceive", Nodes.Perceive)
    graph.add_node("tool", ToolNode(tools=tools))
    graph.add_node("R4", lambda state:{"messages": []})
    graph.add_node("describe", Nodes.Describe)          # bool
    graph.add_node("R5", lambda state:{"messages": []})
    graph.add_node("prepare", Nodes.Prepare)            # bool
    graph.add_node("R6", lambda state:{"messages": []})
    graph.add_node("metric", Nodes.Metric)              # bool
    graph.add_node("next", Nodes.Next)                  # switching
    ## graph.add_node("R7", lambda state:{"messages": []})
    ## graph.add_node("summarize", Nodes.Summarize)

    graph.add_edge(START, "download")
    graph.add_edge("download", "R1")
    graph.add_conditional_edges(
        "R1",
        Nodes.Router1,
        {
            "failed": "next",
            "successful": "copy"
        }
    )
    graph.add_edge("copy", "R2")
    graph.add_conditional_edges(
        "R2",
        Nodes.Router2,
        {
            "failed": "next",
            "successful": "web_info"
        }
    )
    graph.add_edge("web_info", "R3")
    graph.add_conditional_edges(
        "R3",
        Nodes.Router3,
        {
            "failed": "next",
            "successful": "perceive"
        }
    )
    graph.add_edge("perceive", "R4")
    graph.add_conditional_edges(
        "R4",
        Nodes.Router4,
        {
            "continue": "tool",
            "successful": "describe"
        }
    )
    graph.add_edge("tool", "perceive")
    graph.add_edge("describe", "R5")
    graph.add_conditional_edges(
        "R5",
        Nodes.Router5,
        {
            "failed": "next",
            "successful": "prepare"
        }
    )
    graph.add_edge("prepare", "R6")
    graph.add_conditional_edges(
        "R6",
        Nodes.Router6,
        {
            "failed": "next",
            "retry": "prepare",
            "successful": "metric"
        }
    )
    graph.add_edge("metric", "next")
    ## graph.add_edge("next", "R7")
    ## graph.add_conditional_edges(
    ##     "R7",
    ##     Nodes.Router7,
    ##     {
    ##         "done": "summarize",
    ##         "todo": "download"
    ##     }
    ## )
    graph.add_edge("next", END)
    
    return graph
