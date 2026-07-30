import os

import pandas as pd
from tqdm import tqdm

if __package__:
    from .common import merge_task_values
    from .utils.llm_client import OpenAILLMProvider
    from .utils.prompts import gen_catg
else:
    from common import merge_task_values
    from utils.llm_client import OpenAILLMProvider
    from utils.prompts import gen_catg

llm = OpenAILLMProvider()


def _clean_code_output(code: str):
    """Clean code output from LLM (remove markdown formatting)."""
    code = code.strip()
    try:
        code = code.split("<out>")[1].split("</out>")[0].strip()
    except IndexError:
        raise IndexError("Parsing failed: The returned text does not contain complete <out>...</out> tags.")
    return code


def extract_choice_with_llm(dire: str):
    if os.path.exists(f"{dire}/data/public/description.txt"):
        with open(f"{dire}/data/public/description.txt", "r") as f:
            desc = f.read()
    else:
        return "#DESCRIPTION_NOT_FOUND"
    try:
        system_prompt = ("system", gen_catg[0])
        user_prompt = ("user", gen_catg[1].format(description=desc))
        ans = llm.query([system_prompt, user_prompt]).content
        ans = _clean_code_output(ans)
        return ans
    except Exception as e:
        print(f"Category error: {e}")
        return f"#LLM_API_ERROR: {e}"


def annotate_task_categories(all_subdirs: list = None, csv_name="overview.csv"):
    modalities = []
    task_types = []
    ans = []
    for subdir in tqdm(all_subdirs):
        answ = extract_choice_with_llm(subdir)
        ans.append(answ)
        modalities.append(answ.split("@")[0].strip() if len(answ.split("@")) > 0 else answ)
        task_types.append(answ.split("@")[1].strip() if len(answ.split("@")) > 1 else answ)

    csv = pd.read_csv(csv_name)
    csv = merge_task_values(
        csv,
        all_subdirs or [],
        {"Modality": modalities, "Task": task_types},
    )
    csv.to_csv(csv_name, index=False)


if __name__ == "__main__":
    pass
