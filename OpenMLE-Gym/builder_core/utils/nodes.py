import sys
import os
import re
import shutil
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple

import pandas as pd
import py7zr
import requests
import zipfile
import importlib.util
from tqdm.auto import tqdm
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from openmle_gym.common import atomic_write_text
from openmle_gym.process_runner import run_task_process

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from .task import Task
from .state import AgentState
from .struct import Structure
from .chat import OpenAILLMProvider
from .prompts import (
    gen_perceiver,
    gen_description, 
    description_info, 
    gen_prepare_script, 
    gen_metric_script, 
    metric_info,
)


# ============================================================================
# Main NodeExecutor Class
# ============================================================================

class NodeExecutor:
    """Orchestrates downloading, processing, and preparing Kaggle competition data."""
    
    # Supported compressed file extensions
    COMPRESSED_EXTENSIONS = {".zip", ".7z"}
    KAGGLE_DATA_URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/{comp_id}"
    
    def __init__(
        self,
        base_dir: str,
        batch_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        todo: Task = None,
        tools: List[Any] = None,
        info_csv: Optional[Path] = None,
        sample_dir: Optional[Path] = None,
        delete_raw: bool = False,
        code_execution_mode: str = "process",
        code_timeout: float = 600,
    ) -> None:
        """Initialize the NodeExecutor.
        
        Args:
            base_dir: Base directory for data storage
            batch_name: Name of the batch/processing session
            web_timeout: Timeout for web operations in seconds
            api_key: OpenAI API key
            base_url: Base URL for API calls
            model: Model name for LLM
            tools: List of tools for LLM
        """
        load_dotenv()
        
        self.structure = Structure(base_dir, batch_name)
        self.tree = ""
        self.info_csv = Path(info_csv) if info_csv is not None else Path("info.csv")
        self.sample_dir = Path(sample_dir) if sample_dir is not None else Path(base_dir) / "samples"
        self.delete_raw = delete_raw
        self.code_execution_mode = code_execution_mode
        self.code_timeout = code_timeout
        self.csvinfo = pd.read_csv(self.info_csv)
        
        if tools is None:
            self.tools = []
        else:
            self.tools = [
                getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                for tool in tools
            ]
        self.max_tool_call = 40

        self.llm_provider = OpenAILLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
            
        self.llm_provider_tools = OpenAILLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            tools=tools
        )
        
        self.todo = todo
            
    # ========================================================================
    # Core Data Processing Methods
    # ========================================================================
    
    def _is_compressed(self, file_path: Path) -> bool:
        """Check if a file is compressed based on its extension.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file has a compressed extension, False otherwise
        """
        return file_path.suffix.lower() in self.COMPRESSED_EXTENSIONS

    def _load_kaggle_auth(self) -> HTTPBasicAuth:
        """Load Kaggle username/key auth from env or KAGGLE_CONFIG_DIR/kaggle.json."""
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if username and key:
            return HTTPBasicAuth(username, key)

        config_paths = []
        kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if kaggle_config_dir:
            config_paths.append(Path(kaggle_config_dir).expanduser() / "kaggle.json")
        config_paths.append(Path("~/.kaggle/kaggle.json").expanduser())

        for config_path in config_paths:
            if not config_path.exists():
                continue
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Failed to read Kaggle credentials from {config_path}: {exc}") from exc
            username = data.get("username")
            key = data.get("key")
            if username and key:
                return HTTPBasicAuth(username, key)

        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY "
            "or KAGGLE_CONFIG_DIR containing kaggle.json."
        )
    
    def download(self, comp_id: str, target_dir: Path) -> Dict[str, Any]:
        """Download a Kaggle competition dataset using Kaggle CLI.
        
        Args:
            comp_id: Kaggle competition ID
            target_dir: Directory to save downloaded files
            
        Returns:
            Dictionary with success flag and error message if failed
        """
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{comp_id}.zip"
            
            print(f"Downloading competition: {comp_id}")
            
            response = requests.get(
                self.KAGGLE_DATA_URL.format(comp_id=comp_id),
                auth=self._load_kaggle_auth(),
                stream=True,
                timeout=120,
            )
            if response.status_code != 200:
                preview = response.text[:1000].replace("\n", " ")
                return {
                    "success": False,
                    "error": f"Kaggle API HTTP {response.status_code}: {preview}",
                }

            with target_file.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

            if not target_file.exists() or target_file.stat().st_size == 0:
                return {
                    "success": False,
                    "error": f"Downloaded empty file: {target_file}",
                }
            
            print(f"Download completed: {comp_id}")
            return {"success": True, "path": target_file}
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Unexpected error: {e}"
            }
    
    def extract(
        self, 
        compressed_path: Path, 
        dst: Path, 
        recursive: bool = True, 
        already_extracted: Optional[Set[Path]] = None
    ) -> None:
        """Extract compressed files with optional recursive extraction.
        
        Args:
            compressed_path: Path to compressed file
            dst: Destination directory for extraction
            recursive: Whether to recursively extract nested archives
            already_extracted: Set of already processed files to avoid cycles
            
        Raises:
            Exception: If extraction fails
        """
        if already_extracted is None:
            already_extracted = set()

        if compressed_path in already_extracted or not compressed_path.exists():
            return

        print(f"Extracting: {compressed_path.name}")
        
        try:
            destination = dst.resolve()

            def validate_member(name: str) -> None:
                member_path = (destination / name).resolve()
                try:
                    member_path.relative_to(destination)
                except ValueError as exc:
                    raise ValueError(f"Archive member escapes destination: {name}") from exc

            # Extract based on file type
            if compressed_path.suffix == ".7z":
                with py7zr.SevenZipFile(compressed_path, mode="r") as ref:
                    for name in ref.getnames():
                        validate_member(name)
                    ref.extractall(dst)
            elif compressed_path.suffix == ".zip":
                with zipfile.ZipFile(compressed_path, "r") as ref:
                    for member in ref.infolist():
                        validate_member(member.filename)
                        unix_mode = member.external_attr >> 16
                        if (unix_mode & 0o170000) == 0o120000:
                            raise ValueError(f"Archive symlink is not allowed: {member.filename}")
                    ref.extractall(dst)
            else:
                print(f"Unsupported compression format: {compressed_path.suffix}")
                return
            
            already_extracted.add(compressed_path)
            compressed_path.unlink()

            # Recursively extract nested archives
            if recursive:
                new_compressed_files = [
                    f for f in dst.iterdir() 
                    if f.is_file() and self._is_compressed(f) and f.exists()
                ]
                
                for file_path in new_compressed_files:
                    if file_path not in already_extracted:
                        self.extract(
                            file_path, 
                            dst, 
                            recursive=True, 
                            already_extracted=already_extracted
                        )

        except Exception as e:
            raise RuntimeError(f"Failed to extract {compressed_path}: {e}") from e
    
    def _generate_tree(
            self, 
            start_path: Path, 
            max_depth: int = 6, 
            current_depth: int = 0, 
            prefix: str = ''
        ) -> None:
        if current_depth > max_depth:
            return
        
        if current_depth == 0:
            self.tree = f"{start_path}/\n"
        
        if not start_path.is_dir():
            return

        try:
            items = sorted(
                list(start_path.iterdir()), 
                key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except PermissionError:
            return

        if len(items) > 30:
            ext = items[0].suffix if items[0].suffix else "items"
            self.tree += f"{prefix}`-- *There are {len(items)} {ext}*\n"
            return

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = '`-- ' if is_last else '|-- '
            
            item_type = "/" if item.is_dir() else ""
            self.tree += f"{prefix}{connector}{item.name}{item_type}\n"
            
            if item.is_dir() and current_depth < max_depth:
                new_prefix = prefix + ('    ' if is_last else '|   ')
                self._generate_tree(
                    item, 
                    max_depth, 
                    current_depth + 1, 
                    new_prefix
                )
    
    def get_directory_tree(self, start_path: Path, max_depth: int = 6) -> str:
        """Get tree representation of a directory.
        
        Args:
            start_path: Path to directory
            max_depth: Maximum depth to display
            
        Returns:
            String representation of directory tree
        """
        self.tree = ""
        self._generate_tree(start_path, max_depth)
        return self.tree
    
    # ========================================================================
    # File Loading and Processing Methods
    # ========================================================================
    
    def _load_sample_file(self, filename: str) -> str:
        """Load a sample file from the samples directory.
        
        Args:
            filename: Name of sample file
            
        Returns:
            Content of sample file or empty string if not found
        """
        sample_path = self.sample_dir / filename
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def _clean_code_output(self, message: str) -> str:
        """Clean code output from LLM (remove markdown formatting).
        
        Args:
            code: Raw code string from LLM
            
        Returns:
            Cleaned code string
            
        Raises:
            Exception: If code doesn't contain expected tags
        """
        message = message.strip()
        if "<code>" in message and "</code>" in message:
            parts = message.rsplit("<code>", 1)
            think = parts[0].strip()
            code = parts[1].split("</code>", 1)[0].strip()
            return think, code

        fenced_blocks = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```",
            message,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_blocks:
            code = fenced_blocks[-1].strip()
            think = message.rsplit("```", 2)[0].strip()
            return think, code

        raise Exception(
            "Parsing failed: The returned text does not contain "
            "complete wrapping tags <code> and </code> or a Python code fence. "
            "Please regenerate the code with strict adherence to the format."
        )

    # ========================================================================
    # Module Loading and Preparation Methods
    # ========================================================================
    
    def _load_module(self, module_name: str, file_path: Path):
        """Dynamically load a Python module from a file.
        
        Args:
            module_name: Name to assign to loaded module
            file_path: Path to Python file
            
        Returns:
            Loaded module or None if loading fails
        """
        if not file_path.exists():
            return None
        
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, 
                str(file_path)
            )
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            print(f"Error loading module {module_name}: {e}")
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None
    
    def get_prepare_function(self, comp_id: str):
        """Get the prepare function from prepare.py.
        
        Args:
            comp_id: Competition ID
            
        Returns:
            Prepare function or None if not found
        """
        prepare_path = self.structure.comp_id_dir(comp_id) / "prepare.py"
        module = self._load_module(f"prepare_{comp_id}", prepare_path)
        
        if module:
            prepare_func = getattr(module, 'prepare', None)
            if callable(prepare_func):
                return prepare_func
        
        return None
    
    def prepare_validation(self, comp_id: str) -> List[str]:
        """Check if required files and directories exist.
        
        Args:
            comp_id: Competition ID
            
        Returns:
            List of missing file names
        """
        files = [
            self.structure.public_dir(comp_id) / "description.txt",
            self.structure.public_dir(comp_id) / "train.csv",
            self.structure.public_dir(comp_id) / "test.csv",
            self.structure.public_dir(comp_id) / "sample_submission.csv",
            self.structure.private_dir(comp_id) / "test_answer.csv"
        ]

        missing = []
        for file in files:
            if not file.is_file():
                missing.append(file.name)
        return missing
    
    def prepare_single_competition(self, comp_id: str) -> Tuple[bool, Optional[str]]:
        """Prepare data for a single competition.
        
        Args:
            comp_id: Competition ID
            
        Returns:
            Tuple of (success_flag, error_message)
        """
        
        try:
            print(f"Preparing competition: {comp_id}")
            
            # Clean and recreate directories
            for directory in [
                self.structure.data_dir(comp_id), 
                self.structure.utils_dir(comp_id)
            ]:
                shutil.rmtree(directory, ignore_errors=True)
            
            self.structure.data_dir(comp_id).mkdir(parents=True, exist_ok=True)
            self.structure.public_dir(comp_id).mkdir(exist_ok=True)
            self.structure.private_dir(comp_id).mkdir(exist_ok=True)
            self.structure.utils_dir(comp_id).mkdir(exist_ok=True)
            
            prepare_path = self.structure.comp_id_dir(comp_id) / "prepare.py"
            if self.code_execution_mode == "isolated":
                outcome = run_task_process(
                    "prepare",
                    {
                        "prepare_path": str(prepare_path),
                        "raw_dir": str(self.structure.raw_dir(comp_id)),
                        "public_dir": str(self.structure.public_dir(comp_id)),
                        "private_dir": str(self.structure.private_dir(comp_id)),
                    },
                    timeout=self.code_timeout,
                    execution_mode="isolated",
                    readonly_paths=(prepare_path, self.structure.raw_dir(comp_id)),
                    writable_paths=(
                        self.structure.public_dir(comp_id),
                        self.structure.private_dir(comp_id),
                    ),
                )
                if outcome.stdout:
                    print(outcome.stdout, end="")
                if outcome.stderr:
                    print(outcome.stderr, end="", file=sys.stderr)
                if not outcome.ok:
                    raise RuntimeError(
                        outcome.error or "Isolated prepare worker returned an invalid result"
                    )
            else:
                prepare_func = self.get_prepare_function(comp_id)
                if prepare_func is None:
                    raise ValueError("No prepare function found, code inexecutable")
                prepare_func(
                    self.structure.raw_dir(comp_id),
                    self.structure.public_dir(comp_id),
                    self.structure.private_dir(comp_id)
                )
            
            # Validate preparation
            missing = self.prepare_validation(comp_id)
            if missing:
                raise ValueError(
                    f"Prepare script executed successfully, but failed to "
                    f"pass the file validation: {missing} MISSING. Check your NAMING."
                )
            
            # Copy prepare.py
            prepare_source = self.structure.comp_id_dir(comp_id) / "prepare.py"
            if prepare_source.exists():
                shutil.copy2(
                    prepare_source, 
                    self.structure.utils_dir(comp_id) / "prepare.py"
                )
            
            print(f"Successfully prepared {comp_id}")
            return True, None
                    
        except Exception as e:
            error_msg = str(e)
            return False, error_msg
    
    def prepare_implement(self, comp_id: str) -> None:
        """Wrapper for prepare_single_competition with logging.
        
        Args:
            comp_id: Competition ID
            
        Raises:
            Exception: If preparation fails
        """
        success_flag, error_msg = self.prepare_single_competition(comp_id)
        
        if success_flag:
            return
        else:
            # Clean up failed competition
            self._cleanup_failed_competition(comp_id)

            raise Exception(error_msg)
    
    def _cleanup_failed_competition(self, comp_id: str) -> None:
        """Clean up directories for a failed competition.
        
        Args:
            comp_id: Competition ID
        """
        for directory in [
            self.structure.data_dir(comp_id), 
            self.structure.utils_dir(comp_id)
        ]:
            shutil.rmtree(directory, ignore_errors=True)
        
        # Recreate empty directories
        self.structure.data_dir(comp_id).mkdir(exist_ok=True)
        self.structure.public_dir(comp_id).mkdir(exist_ok=True)
        self.structure.private_dir(comp_id).mkdir(exist_ok=True)
        self.structure.utils_dir(comp_id).mkdir(exist_ok=True)
            
    # ========================================================================
    # Agent Workflow Integrated Methods
    # ========================================================================
    
    def Download(self, state: AgentState) -> AgentState:
        """Download a single competition.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        comp_id = self.todo["name"]
        comp_dir = self.structure.comp_id_dir(comp_id)
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[PROCESSING COMPETITION: {comp_id}")
        print(f"{'='*60}")
        print("\nPhase 1: Downloading competition...")
        update = []
        
        target_dir = self.structure.DOWNLOAD_DIR / comp_id
        result = self.download(comp_id, target_dir)
        
        if result["success"]:
            self.todo["download"] = True
            update += [HumanMessage(content="[OK] Download completed")]
            print(f"[OK] Download completed for {comp_id}")
        else:
            error_msg = result.get('error', 'Unknown error')
            self.todo["errors"].append(
                f"Download dataset failed: {error_msg}"
            )
            update += [HumanMessage(content=f"[ERROR] Download failed: {error_msg}")]
            print(f"[ERROR] Download failed for {comp_id}: {error_msg}")
        
        return {"messages": update}
        # return {"messages": []}
    
    def Copy(self, state: AgentState) -> AgentState:
        """Prepare data for a single competition by copying and extracting.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        print("\nPhase 2: Copying and extracting dataset...")
        comp_id = self.todo["name"]
        update = []
        try:
            print(f"Processing competition: {comp_id}")
            
            # Copy and extract data
            source_zip = self.structure.DOWNLOAD_DIR / comp_id / f"{comp_id}.zip"
            if not source_zip.exists():
                raise Exception(f"Source zip not found: {source_zip}")
            
            # Create all necessary directories
            directories = [
                self.structure.comp_id_dir(comp_id),
                self.structure.data_comp_id_dir(comp_id),
                self.structure.raw_dir(comp_id),
                self.structure.data_dir(comp_id),
                self.structure.utils_dir(comp_id),
                self.structure.public_dir(comp_id),
                self.structure.private_dir(comp_id),
                # self.structure.evo_dir(comp_id)
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            target_zip = self.structure.raw_dir(comp_id) / f"{comp_id}.zip"
            shutil.copy2(source_zip, target_zip)
            self.extract(target_zip, self.structure.raw_dir(comp_id))

            shutil.rmtree(source_zip.parent, ignore_errors=True)
            
            # Generate tree file
            tree_info = self.get_directory_tree(
                self.structure.raw_dir(comp_id)
            )
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'w', encoding='utf-8') as file:
                file.write(tree_info)
                
            # create file info file
            file_file = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            with open(file_file, 'w', encoding='utf-8') as file:
                file.write("")
            
            self.todo["copy"] = True
            update += [HumanMessage(content="[OK] Dataset copied and extracted")]
            print(f"[OK] Dataset copied and extracted for {comp_id}")
            
        except Exception as e:
            # Create empty tree file for failed competitions
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'w', encoding='utf-8') as file:
                file.write("Error copying files.")
            
            self.todo["errors"].append(f"Copy dataset failed: {e}")
            update += [HumanMessage(content=f"[ERROR] Copy dataset failed: {e}")]
            print(f"[ERROR] Copy dataset failed for {comp_id}: {e}")
        
        return {"messages": update}

    def Scrape(self, state: AgentState) -> AgentState:
        print("\nPhase 3: Fetching web information...")
        comp_id = self.todo["name"]
        update = []
        info = {}

        print(f"Fetching information for: {comp_id}")

        try:
            match = self.csvinfo[self.csvinfo["Slug"] == comp_id]

            if not match.empty:
                overview = match["Overview"].iloc[0]
                dataset = match["DatasetDescription"].iloc[0]
                if not pd.isna(overview):
                    info["overview"] = str(overview)
                if not pd.isna(dataset):
                    info["dataset"] = str(dataset)

            if info.get("overview") is None and info.get("dataset") is None:
                print(
                    f"[ERROR] CSV information fetching failed for {comp_id}: "
                    "overview and dataset-information Info missing"
                )
                update += [HumanMessage(content=
                                "[ERROR] CSV information fetching failed: "
                                "overview and dataset-information Info missing")]
                return {"messages": update}

            info_file = self.structure.comp_id_dir(comp_id) / "webinfo.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)

            self.todo["web_info"] = True
            update += [HumanMessage(content="[OK] CSV information fetched")]
            print(f"[OK] CSV information fetched for {comp_id}")

        except Exception as e:
            update += [HumanMessage(content=f"[ERROR] CSV information fetching failed: {e}")]
            print(f"[ERROR] CSV information fetching failed for {comp_id}: {e}")

        return {"messages": update}
    
    
    def Perceive(self, state: AgentState) -> AgentState:
        """Run optional file inventory.

        ``todo["perceive"]`` means the tool loop has terminated. Inventory
        failures are recorded but deliberately degrade to the Describe stage.
        """
        print("\nPhase 4: Perceiving dataset information...")
        comp_id = self.todo["name"]
        update = []
        
        try:
            # update
            if state.get("messages") and isinstance(state["messages"][-1], ToolMessage):
                last_tool = state["messages"][-1]
                if (
                    getattr(last_tool, "name", None) == "save"
                    and last_tool.content.strip() == "Successfully saved to fileinfo.txt"
                ):
                    self.todo["perceive"] = True
                    update += [HumanMessage(content="[OK] File info digested")]
                    print(f"[OK] File info digested for {comp_id}")
                    return {"messages": update}
                
            tool_messages = [tool_info.content for tool_info in state["messages"] if isinstance(tool_info, ToolMessage)]
            if len(tool_messages) >= self.max_tool_call:
                raise Exception(f"Tool call times exceeded maximum({self.max_tool_call}), which indicates unsafe deadloop")
            
            # load tree
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'r', encoding='utf-8') as file:
                tree_info = file.read()
            
            # provide absolute path
            info_path = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            query = [
                SystemMessage(content=gen_perceiver),
                HumanMessage(content=f"Now perform inventory of this directory. FILE TREE:\n{tree_info}\n'fileinfo.txt' PATH:'{info_path}'"),
            ]
            update += query
            
            response = self.llm_provider_tools.query(
                [query[0]] +
                tool_messages + 
                [query[1]]
            )
            
            update += [response]
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                response.tool_calls = [tc for tc in response.tool_calls if tc["name"] in self.tools]
                tool_use = [tc["name"] for tc in response.tool_calls]
                print(f"USING TOOLS: {tool_use}")
                if not response.tool_calls:
                    raise RuntimeError("LLM returned no allowed tool calls")
            else:
                raise RuntimeError("LLM returned no tool calls")
            
        except Exception as e:
            update += [HumanMessage(content=f"[ERROR] File info digestion failed halfway: {e}")]
            self.todo["errors"].append(
                f"File info digestion failed: {e}"
            )
            self.todo["perceive"] = True
            print(f"[ERROR] File info digestion for {comp_id} failed halfway: {e}")
            
        return {"messages": update}
        
        
    def Describe(self, state: AgentState) -> AgentState:
        """Generate description for a single competition.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        print("\nPhase 5: Generating description...")

        comp_id = self.todo["name"]
        update = []
        
        try:
            # Load web info and tree
            web_file = self.structure.comp_id_dir(comp_id) / "webinfo.json"
            with open(web_file, 'r', encoding='utf-8') as file:
                web_info = json.load(file)
            tree_file = self.structure.comp_id_dir(comp_id) / "rawtree.txt"
            with open(tree_file, 'r', encoding='utf-8') as file:
                tree_info = file.read()
            
            # File Content Perceived
            file_file = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            with open(file_file, 'r', encoding='utf-8') as file:
                file_info = file.read()
            
            # Load sample description
            sample_description = self._load_sample_file("sample_description.txt")
            
            # Construct prompt
            # infos = description_info.format(
            #     description=web_info.get('description', ''),
            #     evaluation=web_info.get('evaluation', ''),
            #     submission=web_info.get('submission-instructions', ''),
            #     dataset_description=web_info.get('dataset-description', ''),
            #     tree_info=tree_info,
            #     file_info=file_info
            # )
            infos = description_info.format(
                description=web_info.get('overview', ''),
                dataset_description=web_info.get('dataset', ''),
                tree_info=tree_info,
                file_info=file_info
            )
            
            query = [
                SystemMessage(content=gen_description.format(sample_description=sample_description)), 
                HumanMessage(content=f"Now perform detailed summarization on these TEXTUAL MATERIALS:\n{infos}")
            ]
            update += query
            
            response = self.llm_provider.query(query)
            
            desc_file = self.structure.comp_id_dir(comp_id) / "description.txt"
            with open(desc_file, 'w', encoding='utf-8') as file:
                file.write(response.content)
            
            self.todo["describe"] = True
            update += [response, HumanMessage(content="[OK] Description generated")]
            print(f"[OK] Description generated for {comp_id}")
                
        except Exception as e:
            update += [HumanMessage(content=f"[ERROR] Description generation failed: {e}")]
            self.todo["errors"].append(
                f"Description generation failed: {e}"
            )
            print(f"[ERROR] Description generation failed for {comp_id}: {e}")

        return {"messages": update}
    
    def Prepare(self, state: AgentState) -> AgentState:
        """Generate prepare script for a single competition.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        print("\nPhase 6: Generating prepare script and Preparing...")
        comp_id = self.todo["name"]
        update = []
        
        try:
            # Check description
            desc_file = self.structure.comp_id_dir(comp_id) / "description.txt"
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()
            
            # Load prepare sample
            prepare_sample = self._load_sample_file("sample_prepare.py")
            
            # Construct prompt
            query = [
                SystemMessage(content=gen_prepare_script.format(prepare_sample=prepare_sample)), 
                HumanMessage(content=
                    f"Now THINK and WRITE the CODE based on the DESCRIPTION:\n{description}\nDO NOT FORGET to wrap the CODE part in <code> and </code>!"
                )
            ]
            update += query
            
            response = self.llm_provider.query(
                [query[0]] + self.todo["prepares"][-2:] + [query[1]]
            )
            
            update += [response, HumanMessage(content="[OK] Prepare script generated")]
            self.todo["prepares"] += [query[1]]
            self.todo["gen_prep"] = True
            print(f"[OK] Prepare script generated for {comp_id}")
            
        except Exception as e:
            update += [HumanMessage(content=f"[ERROR] Prepare script generation failed: {e}")]
            self.todo["retry"] = -1
            self.todo["errors"].append(
                f"Prepare script generation failed: {e}"
            )
            print(f"[ERROR] Prepare script generation failed for {comp_id}: {e}")
            return {"messages": update}
        
        think = "VOID"
        prepare_code = "VOID"
        
        try:
            think, prepare_code = self._clean_code_output(response.content)
            prepare_file = self.structure.comp_id_dir(comp_id) / "prepare.py"
            compile(prepare_code, str(prepare_file), "exec")
            atomic_write_text(prepare_file, prepare_code)
            
            self.prepare_implement(comp_id)
            
            self.todo["prepare"] = True
            update += [HumanMessage(content="[OK] Data preparation completed")]
            print(f"[OK] Data preparation completed for {comp_id}")
            
            # Response and Feedback
            self.todo["prepares"] += [
                AIMessage(content=think),
                AIMessage(content=prepare_code)                          
            ]
            self.todo["prepares"] += [
                HumanMessage(
                    content="Data preparation successful using the latest code script"
                )
            ]
            
        except Exception as e:
            self.todo["retry"] -= 1
            self.todo["errors"].append(f"[ERROR] Data preparation failed: {e}")
            
            # Response and Feedback
            self.todo["prepares"] += [
                AIMessage(content=think if think != "VOID" else response.content),
                AIMessage(content=prepare_code if prepare_code != "VOID" else str(e))                          
            ]
            self.todo["prepares"] += [
                HumanMessage(content=f"Data preparation failed using the latest code script: {e}")
            ]
            
            if self.todo["retry"] > -1:
                retry_msg = (
                    f"[ERROR] Data preparation failed: {e}\n"
                    f"Retrying (Remaining attempts: {self.todo['retry']})..."
                )
                update += [HumanMessage(content=retry_msg)]
                print(
                    f"[ERROR] Data preparation failed for {comp_id}: {e}\n"
                    f"Retrying (Remaining attempts: {self.todo['retry']})..."
                )
            else:
                update += [HumanMessage(content=f"[ERROR] Data preparation FAILED: {e}")]
                print(f"[ERROR] Data preparation FAILED for {comp_id}: {e}")
        
        return {"messages": update}
    
    def Metric(self, state: AgentState) -> AgentState:
        """Generate metric script for a single competition.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        print("\nPhase 7: Generating metric...")
        comp_id = self.todo["name"]
        update = []
        
        try: 
            # Check description
            desc_file = self.structure.public_dir(comp_id) / "description.txt"
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()
            
            # Check for sample submission
            submission_sample = ""
            submission_path = self.structure.public_dir(comp_id) / "sample_submission.csv"
            if submission_path.exists():
                df = pd.read_csv(submission_path, nrows=2)
                submission_sample = str(df)
                    
            # Check for answer sample
            answer_sample = ""
            answer_path = self.structure.private_dir(comp_id) / "test_answer.csv"
            if answer_path.exists():
                df = pd.read_csv(answer_path, nrows=2)
                answer_sample = str(df)
                    
            # Load metric sample
            metric_sample = self._load_sample_file("sample_metric.py")
            
            # Construct prompt
            infos = metric_info.format(
                description=description,
                submission_sample=submission_sample,
                answer_sample=answer_sample,
            )
            
            query = [
                SystemMessage(content=gen_metric_script.format(metric_sample=metric_sample)), 
                HumanMessage(content=
                    f"Now THINK and WRITE the CODE based on the INFORMATION:\n{infos}\nDO NOT FORGET to wrap the CODE part in <code> and </code>!"
                )
            ]
            update += query
            
            response = self.llm_provider.query(query)
            metric_file = self.structure.utils_dir(comp_id) / "metric.py"
            _, metric_code = self._clean_code_output(response.content)
            compile(metric_code, str(metric_file), "exec")
            atomic_write_text(metric_file, metric_code)
                
            self.todo["metric"] = True
            update += [response, HumanMessage(content="[OK] Metric script generated")]
            print(f"[OK] Metric script generated for {comp_id}")

        except Exception as e:
            update += [HumanMessage(content=f"[ERROR] Metric script generation failed: {e}")]
            self.todo["errors"].append(
                f"Metric script generation failed: {e}"
            )
            print(f"[ERROR] Metric script generation failed for {comp_id}: {e}")

        return {"messages": update}
    
    
    # ========================================================================
    # Workflow Management Methods
    # ========================================================================
    
    def Next(self, state: AgentState):
        comp_id = self.todo["name"]

        self.todo["success"] = all([
            self.todo["download"],
            self.todo["copy"],
            self.todo["web_info"],
            self.todo["describe"],
            self.todo["prepare"],
            self.todo["metric"]
        ])

        if self.delete_raw and self.todo["success"]:
            raw_placeholder = self.structure.data_comp_id_dir(comp_id) / "raw.txt"
            raw_placeholder.parent.mkdir(parents=True, exist_ok=True)
            fileinfo = self.structure.comp_id_dir(comp_id) / "fileinfo.txt"
            if fileinfo.is_file() and fileinfo.stat().st_size > 0:
                atomic_write_text(
                    raw_placeholder,
                    fileinfo.read_text(encoding="utf-8"),
                )
                shutil.rmtree(self.structure.raw_dir(comp_id), ignore_errors=True)
            else:
                self.todo["errors"].append(
                    "Raw data retained: fileinfo.txt is missing or empty"
                )

        print(
            f"\n[OK] Successful for {comp_id}"
            if self.todo["success"]
            else f"\n[ERROR] Failed for {comp_id}"
        )

        preps_file = self.structure.comp_id_dir(comp_id) / "prepares.json"
        preps = []

        for idx in range(0, len(self.todo["prepares"]), 4):
            if idx + 3 < len(self.todo["prepares"]):
                prep_entry = {
                    "description": self.todo["prepares"][idx].content,
                    "think": self.todo["prepares"][idx + 1].content,
                    "code": self.todo["prepares"][idx + 2].content,
                    "feedback": self.todo["prepares"][idx + 3].content
                }
                preps.append(prep_entry)

        with open(preps_file, "w", encoding="utf-8") as f:
            json.dump(preps, f, indent=4, ensure_ascii=False)

        return {
            "messages": [],
            "task_result": self.todo
        }

    # ========================================================================
    # Routing Methods
    # ========================================================================
    
    def Router1(self, state: AgentState) -> str:
        """Route based on download status."""
        return "successful" if self.todo["download"] else "failed"
    
    def Router2(self, state: AgentState) -> str:
        """Route based on copy status."""
        return "successful" if self.todo["copy"] else "failed"
    
    def Router3(self, state: AgentState) -> str:
        """Route based on web info status."""
        return "successful" if self.todo["web_info"] else "failed"
    
    def Router4(self, state: AgentState) -> str: # added
        """Route based on file info status."""
        return "successful" if self.todo["perceive"] else "continue"
    
    def Router5(self, state: AgentState) -> str:
        """Route based on describe status."""
        return "successful" if self.todo["describe"] else "failed"
    
    def Router6(self, state: AgentState) -> str:
        """Route based on prepare status."""
        if self.todo["prepare"]:
            return "successful"
        elif self.todo["retry"] > -1:
            return "retry"
        else:
            return "failed"
