1. Clone the mle-bench repo.
    ```
    cd src/dojo/tasks/mlebench
    git clone --depth 1 https://github.com/openai/mle-bench.git
    cd mle-bench
    git fetch --depth 1 origin d0f60ad0d3b2287469ac3c8ac9767330c928c980
    git checkout FETCH_HEAD
    ```

2. Download additional files.
    ```
    git lfs fetch --all
    git lfs pull
    ```

3. **Important:**
Replace line 29 in `src/dojo/tasks/mlebench/mle-bench/mlebench/data.py` with:
    ```
    cache = dc.Cache(size_limit=2**26)
    ```
    so that a temporary directory is created for the cache.

4. Install mle-bench.
    ```
    pip install -e .
    ```

5. Prepare tasks

    - Setup Kaggle API Token: Create a kaggle account if you don't have one and get an API token by following the [instructions](https://www.kaggle.com/docs/api) in the 'Authentication' section

    - Download and prepare the tasks you want to run (you can find all tasks in `src/dojo/tasks/mlebench/splits/all.txt`). To download the data in a specific directory, use the `--data-dir` argument (by default it will be downloaded to `shared/cache/dojo/tasks/mlebench/`):
        ```
        # Single task preparation example
        python src/dojo/tasks/mlebench/utils/prepare.py -c random-acts-of-pizza --data-dir=/path/to/data
        # Prepare all tasks
        python src/dojo/tasks/mlebench/utils/prepare.py -s all --data-dir=/path/to/data
        ```
    - Change your `MLE_BENCH_DATA_DIR` environment variable in your `.env` file to your data directory:
        ```shell
        ...
        MLE_BENCH_DATA_DIR=/<PATH_TO_TEAM_STORAGE>/shared/cache/dojo/tasks/mlebench/ # <---- Set to your data directory
        ...
        ```

6. Run your first task on mle-bench:
    ```shell
    # Run AIRA_Greedy on MLE-Bench's random-acts-of-pizza task
    python -m dojo.main_run +_exp=run_example logger.use_wandb=False task.name=random-acts-of-pizza
    ```
