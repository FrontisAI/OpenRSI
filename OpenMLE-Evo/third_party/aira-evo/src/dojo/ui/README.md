# Dashboard

This UI allows users to visulalize the results of runs in aira-dojo.

## Starting the UI

```bash
streamlit run src/dojo/ui/lightweight_dashboard.py
```
## Visualizing Runs: The Basics

### Selecting a Meta Experiment
- Select a Base Directory Path (by default it's the logging directory `$LOGGING_DIR/aira-dojo` from your `.env` file) and click "Scan Base Directory".
- Select a Meta Experiment from the dropdown menu.

### Generating and Visualizing a Tree
- "⚙️ Analysis Utilities" > "Select Utility" > "Generate JSON/HTML Trees from Logs" > "Run Utility".
- Go to "🌳 Tree Visualization" Tab and select an experiment from the dropdown menu "Focus on Specific Experiment (Optional)"

### Generating and Inspecting Crash Reports
- "Analysis Utilities" > "Select Utility" > "Generate Crash/Error Reports" > "Run Utility". This will generate a report of all crashes in the selected meta experiment.
- Go to "📁 File Explorer" > "Select File to View" > click on "error_analysis_report.md"

**IMPORTANT**: If you want to genereate crash reports, you must set your `GEMINI_API_KEY` in your `.env` file

### Generating and Inspecting Tree Statistics
- "Analysis Utilities" > "Select Utility" > "Generate Tree Statistics" > "Run Utility". This will generate a report of all tree statistics in the selected meta experiment.
- Go to "📁 File Explorer" > "Select File to View" > select any file starting with "tree_stats/"

**IMPORTANT**: If you want to generate a journal report, you must set your `GEMINI_API_KEY` in your `.env` file


Note: If you don't see any of your files you generated, you may need to refresh the page.
