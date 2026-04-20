---
description: "Provides a detailed execution flow starting from run_agent.py, explaining which functions are called where in the agentic data scientist pipeline."
applyTo: "**/*.py"
---

# Execution Flow of the Agentic Data Scientist

This instruction outlines the sequence of function calls starting from the main entry point `run_agent.py`. It helps in understanding the codebase structure and dependencies when modifying or debugging the agent.

## Main Entry Point: `run_agent.py`

- `main()`: Parses command-line arguments and creates an instance of `AgenticDataScientist`.
- Calls `AgenticDataScientist.run()` with parsed arguments.

## AgenticDataScientist Class (`agentic_data_scientist.py`)

### `run()` Method (Main Orchestration)

1. **Initialization**:
   - Generates a unique `run_id` and creates an output directory.
   - Populates `RunContext` with run metadata.
   - Initializes state for replanning.

2. **Data Loading**:
   - `load_data(data_path)`: Loads the CSV dataset into a pandas DataFrame and logs its shape.

3. **Target Column Handling**:
   - If `target == "auto"`: Calls `infer_target_column(df)` from `tools.data_profiler` to infer the target column.

4. **Data Profiling**:
   - `profile_dataset(df, target)` from `tools.data_profiler`: Produces an EDA summary.
   - `dataset_fingerprint(df, target)` from `tools.data_profiler`: Creates a fingerprint for memory lookup.

5. **Memory Lookup**:
   - `self.memory.get_dataset_record(fp)`: Checks for previous runs on the same dataset.

6. **Planning**:
   - `create_plan(profile, memory_hint=prev)` from `agents.planner`: Creates an initial plan.

7. **Execution Loop** (with optional replanning):
   - `build_preprocessor(profile)` from `tools.modelling`: Builds preprocessing pipeline.
   - `select_models(profile, seed)` from `tools.modelling`: Chooses candidate models.
   - `train_models(df, target, preprocessor, candidates, seed, test_size, output_dir, verbose)` from `tools.modelling`: Trains models and saves artifacts.
   - `evaluate_best(results, output_dir)` from `tools.evaluation`: Evaluates models and selects the best.
   - `reflect(dataset_profile, evaluation, all_metrics)` from `agents.reflector`: Reflects on results.
   - Saves artifacts: `save_json()` for eda_summary.json, plan.json, metrics.json, reflection.json.
   - `write_markdown_report()` from `tools.evaluation`: Generates the report.md.
   - `self.memory.upsert_dataset_record(fp, record)`: Updates memory with run outcomes.
   - `should_replan(reflection)` from `agents.reflector`: Decides if replanning is needed.
   - If replanning: `apply_replan_strategy(plan, profile, reflection)` from `agents.reflector`: Updates plan and profile.

8. **Termination**:
   - Returns the output directory path.

## Key Modules and Their Functions

- **agents/memory.py**: `JSONMemory` class with `get_dataset_record()` and `upsert_dataset_record()`.
- **agents/planner.py**: `create_plan()`.
- **agents/reflector.py**: `reflect()`, `should_replan()`, `apply_replan_strategy()`.
- **tools/data_profiler.py**: `profile_dataset()`, `infer_target_column()`, `dataset_fingerprint()`.
- **tools/modelling.py**: `build_preprocessor()`, `select_models()`, `train_models()`.
- **tools/evaluation.py**: `evaluate_best()`, `write_markdown_report()`, `save_json()`.

Use this flow to trace dependencies and understand where changes might affect the pipeline.