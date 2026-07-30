# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from dojo.config_dataclasses.task.base import TaskConfig
from dojo.utils.environment import get_mlebench_data_dir

mlebench_operator_prompts: dict[str, str] = {
    "draft_intro": """You are a Kaggle grandmaster attending a competition.
        In order to win this competition, you need to come up with an excellent and creative plan
        for a solution and then implement this solution in Python. We will now provide a description of the task.""",
    "improve_intro": """You are a Kaggle grandmaster attending a competition. You are provided with a previously developed
        solution below and should improve it in order to further increase the (test time) performance.
        For this you should first outline a brief plan in natural language for how the solution can be improved and
        then implement this improvement in Python based on the provided previous solution.""",
    "debug_intro": """You are a Kaggle grandmaster attending a competition.
        Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug.
        Your response should be an implementation outline in natural language, followed by a single markdown code block which implements the bugfix/solution.""",
    "analyze_intro": """You are a Kaggle grandmaster attending a competition.
        You have written code to solve this task and now need to evaluate the output of the code execution.
        You should determine if there were any bugs as well as report the empirical findings.""",
    "implementation_guideline": """The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,
        **AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE CURRENT DIRECTORY.**
        The code should be a single-file python program that is self-contained and can be executed as-is.
        No parts of the code should be skipped, don't terminate the before finishing the script.
        Your response should only contain a single code block.
        All the provided input data is stored in "./data" directory.
        **If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!
        You can also use the current directory to store any temporary files that your code needs to create.
        REMEMBER THE ./submission.csv FILE!!!!! The correct directory is important too.
        The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand.""",
    "execution_intro": """Be aware of the running time of the code, it should complete within""",
    "response_format": """Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences),
        followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric.
        There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.""",
    "solution_sketch": """This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
        Take the Memory section into consideration when proposing the design,
        don't propose the same modelling solution but keep the evaluation the same.
        The solution sketch should be 3-5 sentences.
        Propose an evaluation metric that is reasonable for this task.
        Don't suggest to do EDA.
        The data is already prepared and available in the `./data` directory. There is no need to unzip any files.""",
    "improvement_sketch": """The solution sketch should be a brief natural language description of how the previous solution can be improved.
        You should be very specific and should only propose a single actionable improvement.
        This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.
        Take the Memory section into consideration when proposing the improvement.
        The solution sketch should be 3-5 sentences.
        Don't suggest to do EDA.""",
    "bugfix_sketch": """You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.
        Don't suggest to do EDA.""",
    "analyze_func_name": """submit_review""",
    "analyze_func_desc": """Submit a review evaluating the output of the training script.""",
    "crossover_intro": """You are a Kaggle grandmaster attending a competition.
        You have two previously developed solutions that have both shown strong performance.
        Your task is to combine the best aspects of both solutions into a single, improved implementation.
        First, outline a brief plan in natural language for how to merge the strengths of both approaches while maintaining or improving performance.
        Then, implement this combined solution in Python.""",
    "crossover_sketch": """The solution sketch should be a brief natural language description (3-5 sentences) of how the best aspects of two previous solutions can be combined into a single, more effective implementation.
        Focus on merging complementary strengths while maintaining efficiency and performance.
        Ensure that the proposed approach does not introduce unnecessary complexity.
        The evaluation metric should remain consistent with the previous solutions.
        Do not suggest exploratory data analysis (EDA).""",
}


@dataclass
class MLEBenchTaskConfig(TaskConfig):
    benchmark: str = field(
        default="mlebench",
        metadata={
            "help": "Type of the task.",
        },
    )

    data_dir: str = field(
        default=SI("${task.public_dir}"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )
    cache_dir: str = field(
        default=get_mlebench_data_dir(),
        metadata={
            "help": "The directory where the task data is cached.",
            "exclude_from_hash": True,
        },
    )
    public_dir: str = field(
        default=SI("${task.cache_dir}/${task.name}/prepared/public"),
        metadata={
            "help": "The directory where the public data is stored.",
            "exclude_from_hash": True,
        },
    )
    private_dir: str = field(
        default=SI("${task.cache_dir}/${task.name}/prepared/private"),
        metadata={
            "help": "The directory where the private data is stored.",
            "exclude_from_hash": True,
        },
    )
    submission_fname: str = field(
        default="submission.csv",
        metadata={
            "help": "The name of the submission file.",
            "exclude_from_hash": True,
        },
    )
    results_output_dir: str = field(
        default=SI("${logger.output_dir}/results"),
        metadata={
            "help": "The directory where the results are stored.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
