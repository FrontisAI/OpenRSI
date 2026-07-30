from __future__ import annotations

import pandas as pd

from tts_search.validation_split_guidance import (
    load_guidance_map,
    patch_eval_data_with_guidance,
)


def test_load_guidance_map_extracts_recommended_blocks(tmp_path):
    md = tmp_path / "guidance.md"
    md.write_text(
        """# Demo

## 1. task-a

Recommended prompt instruction:

```text
Validation split protocol: use split A.

Validation metric: ROC-AUC. Higher is better.
```

原因：
中文解释。
""",
        encoding="utf-8",
    )

    guidance = load_guidance_map(md)

    assert guidance == {
        "task-a": (
            "Validation split protocol: use split A.\n\n"
            "Validation metric: ROC-AUC. Higher is better."
        )
    }


def test_patch_eval_data_injects_before_final_output_and_skips_existing(tmp_path):
    source = tmp_path / "eval.parquet"
    output = tmp_path / "patched.parquet"
    md = tmp_path / "guidance.md"
    md.write_text(
        """# Demo

## 1. task-a

Recommended prompt instruction:

```text
Validation split protocol: use split A.

Validation metric: ROC-AUC. Higher is better.
```
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "prompt": "Task body\n\n**FINAL OUTPUT**: print score",
                "metadata": {"task_name": "task-a"},
            },
            {
                "prompt": (
                    "Task body\n\nValidation Split Guidance:\nAlready here\n\n"
                    "**FINAL OUTPUT**: print score"
                ),
                "metadata": {"task_name": "task-a"},
            },
        ]
    ).to_parquet(source, index=False)

    stats = patch_eval_data_with_guidance(
        eval_data_path=source,
        output_path=output,
        instructions_path=md,
        input_key="prompt",
        metadata_key="metadata",
    )
    patched = pd.read_parquet(output)

    assert stats.total_rows == 2
    assert stats.patched_rows == 1
    assert stats.skipped_existing_rows == 1
    assert patched.loc[0, "prompt"].count("Validation Split Guidance:") == 1
    assert (
        patched.loc[0, "prompt"].index("Validation Split Guidance:")
        < patched.loc[0, "prompt"].index("**FINAL OUTPUT**")
    )
    assert patched.loc[1, "prompt"].count("Validation Split Guidance:") == 1
