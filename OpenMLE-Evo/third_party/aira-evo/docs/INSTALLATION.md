# Installation Guide

This guide provides detailed instructions for setting up the RAD framework on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Git
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Anaconda)
- Python 3.11+ (will be installed via conda)

## Step 1: Clone the Repository

```bash
git clone https://github.com/facebookresearch/aira-dojo
cd aira-dojo
```

## Step 2: Set Up the Conda Environment

The project includes an `environment.yaml` file that defines all the necessary dependencies:

```bash
conda env create -f environment.yaml
conda activate aira-dojo
```

This will create a new conda environment named `aira-dojo` with all the required packages.

## Step 3. Install aira-dojo via pip
```bash
pip install -e .
```

## Step 4: Configure Environment Variables

The project uses environment variables for configuration. A template file `.env_default` is provided:

```bash
cp .env_default .env
```

Open the `.env` file in your preferred text editor and update the values as needed. Typical settings include:

- API keys for language models
- Paths do directories for logging, data etc.

**Note**: The `.env` file is ignored by git to prevent accidentally pushing sensitive information.

## Step 5: Change LLM Client Configs**
If you are using different endpoints, you should change them accordingly in `dojo/configs/run/solver/client`
Examples:
- **Changing Azure endpoint for 4o:**

  Go to [`src/dojo/configs/run/solver/client/litellm_4o.yaml`](../src/dojo/configs/solver/client/litellm_4o.yaml) and change the `base_url` to your Azure endpoint:
  ```yaml
    ...
    base_url: https://azure-services-endpoint-here.azure-api.net #<---- Set to your Azure endpoint
    ...
  ```
- **Changing to openai endpoint for 4o:**

  Go to [`src/dojo/configs/run/solver/client/litellm_4o.yaml`](../src/dojo/configs/solver/client/litellm_4o.yaml) and change the `base_url` and `use_azure_client` to the following:
  ```yaml
    ...
    base_url: null  # litellm will use the openai endpoint by default
    use_azure_client: False
    ...
  ```
  Finally, in `.env`, set your primary key to you openai key:
  ```yaml
  PRIMARY_KEY="sk-..." # <---- Set to your OpenAI key>
  ```

Note: To run the examples in the "Example Usage" section of this read me, you must setup the following models:
- `o3`: Set the `base_url` in [`src//dojo/configs/solver/client/litellm_o3.yaml`](./src//dojo/configs/solver/client/litellm_o3.yaml) and set the `PRIMARY_KEY_O3` in `.env`.
- `gpt-4o`: Set the `base_url` in [`src//dojo/configs/solver/client/litellm_4o.yaml`](./src/dojo/configs/solver/client/litellm_4o.yaml) and set the `PRIMARY_KEY` in `.env`.

## Step 6: Build a superimage with apptainer
Follow the steps in [`docs/BUILD_SUPERIMAGE.md`](./BUILD_SUPERIMAGE.md) to build your superimage. This is necessary to run tasks that use jupyter as the interpreter.

## Step 7: Install Task-Specific Dependencies

### MLE-Bench

Follow the steps in [`src/dojo/tasks/mlebench/README.md`](../src/dojo/tasks/mlebench/README.md) to install mle-bench and run your first task.

## Step 8. Setting up wandb
Log in with the following command:
```bash
wandb login
```
It will ask you your API key, which you can get by going into "User settings" (click top right of screen) and scrolling down.

## Required Environment Variables

The `.env` file should include these important variables:

```bash
# API Keys for Language Models
PRIMARY_KEY=sk-...
# Requiered for running AIRA_GREEDY, AIDE_GREEDY, AIRA_MCTS, AIRA_EVO with default configs
PRIMARY_KEY_O3=sk-...
GEMINI_API_KEY=sk-...

# Change these paths to a suitable location on your system
# path where experiment logs will be stored
LOGGING_DIR="/<<<PATH_TO_TEAM_STORAGE>>>/shared/logs/"
# path to mlebench data (see src/dojo/tasks/mlebench/README.md for instructions)
MLE_BENCH_DATA_DIR="/<<<PATH_TO_TEAM_STORAGE>>>/shared/cache/dojo/tasks/mlebench/"
# path to superimage directory (see docs/BUILD_SUPERIMAGE.md for instructions)
SUPERIMAGE_DIR="/<<<PATH_TO_TEAM_STORAGE>>>/shared/sif/"

# Requiredf for running with slurm
# default slurm accout (used for slurm jobs)
DEFAULT_SLURM_ACCOUNT="<<<YOUR_SLURM_ACCOUNT>>>"
# default slurm partition (used for slurm jobs)
DEFAULT_SLURM_PARTITION="<<<YOUR_SLURM_PARTITION>>>"
# default slurm qos (used for slurm jobs
DEFAULT_SLURM_QOS="<<<YOUR_SLURM_QOS>>>"

```

## Development Setup

If you're developing the framework itself:

1. Install additional development dependencies:
   ```bash
   pip install -r requirements/requirements.txt
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```
## Troubleshooting

### Common Issues

1. **Conda environment creation fails**
   - Check if you have the latest conda version: `conda update -n base conda`
   - Try creating the environment with specific Python version: `conda create -n aira-dojo python=3.11`
   - Then install packages manually: `pip install -r requirements/requirements.txt`
   - Install the dojo package: `pip install -e .`

2. **Package import errors after installation**
   - Ensure the conda environment is activated: `conda activate aira-dojo`
   - Verify the package was installed correctly: `pip list | grep aira-dojo`
   - Check Python is looking in the right paths: `python -c "import sys; print(sys.path)"`

3. **Environment variable issues**
   - Confirm your `.env` file exists and has the correct format
   - Restart your terminal after creating the `.env` file
   - Check variables are loaded: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('PRIMARY_KEY'))"`

4. **LLM API errors**
   - Verify API keys are correct and have sufficient quota
   - Check network connectivity to API endpoints
   - Look for rate limiting errors in logs

5. **Task-specific errors**
   - Ensure task data files are downloaded and in the correct location
   - Check dependencies specific to that task are installed
   - Look at task-specific documentation