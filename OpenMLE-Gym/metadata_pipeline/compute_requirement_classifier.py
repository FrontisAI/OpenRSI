import os

import pandas as pd
if __package__:
    from .common import merge_task_values
    from .utils.llm_client import OpenAILLMProvider
    from .utils.prompts import gen_ifgpu
else:
    from common import merge_task_values
    from utils.llm_client import OpenAILLMProvider
    from utils.prompts import gen_ifgpu
from tqdm import tqdm

llm = OpenAILLMProvider()


def find_and_extract(target_name, csv_name=None):
    target_name = target_name.name
    csv = pd.read_csv(csv_name)
    size = csv[csv["Name"] == target_name]["Final Size"].values
    if len(size) == 0:
        return "Unknown"
    if "timeout" in str(size[0]):
        return ">10GB"
    return size[0]


def _clean_code_output(code: str):
    """Clean code output from LLM (remove markdown formatting)."""
    code = code.strip()
    try:
        code = code.split("<out>")[1].split("</out>")[0].strip()
    except IndexError:
        raise IndexError("Parsing failed: The returned text does not contain complete <out>...</out> tags.")
    return code


def extract_compute_requirement_with_llm(dire: str, size: str):
    if os.path.exists(f"{dire}/data/public/description.txt"):
        with open(f"{dire}/data/public/description.txt", "r") as f:
            desc = f.read()
    else:
        return "#DESCRIPTION_NOT_FOUND"
    try:
        system_prompt = ("system", gen_ifgpu[0])
        user_prompt = ("user", gen_ifgpu[1].format(
            description=desc,
            size=size,
        ))
        ans = llm.query([system_prompt, user_prompt]).content
        ans = _clean_code_output(ans)
        return ans
    except Exception as e:
        print(f"CPU/GPU error: {e}")
        return f"#LLM_API_ERROR: {e}"


def extract_choice_with_llm(dire: str, csv_name: str):
    size = find_and_extract(dire, csv_name)
    return extract_compute_requirement_with_llm(dire, size)


def annotate_compute_requirements(all_subdirs: list = None, csv_name="overview.csv"):
    ans = [extract_choice_with_llm(subdir, csv_name) for subdir in tqdm(all_subdirs)]
    csv = pd.read_csv(csv_name)
    csv = merge_task_values(csv, all_subdirs or [], {"CPU/GPU": ans})
    csv.to_csv(csv_name, index=False)


if __name__ == "__main__":
    pass
