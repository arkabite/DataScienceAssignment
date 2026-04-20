# # Orchestrator for an "agentic" offline data scientist pipeline.
# # Handles dataset loading, profiling, planning, training, evaluation, reflection,
# # and optional re-planning cycles. Designed primarily for classification tasks.
# import os
# import json
# import uuid
# from dataclasses import dataclass
# from datetime import datetime
# from typing import Any, Dict, List, Optional

# import pandas as pd

# # Agent components and tooling used by the orchestrator
# from agents.planner import create_plan
# from agents.reflector import reflect, should_replan, apply_replan_strategy
# from agents.memory import JSONMemory
# from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
# from tools.modelling import build_preprocessor, select_models, train_models
# from tools.evaluation import evaluate_best, write_markdown_report, save_json


# # Lightweight container for run metadata and parameters
# @dataclass
# class RunContext:
#     run_id: str
#     started_at: str
#     data_path: str
#     target: str
#     output_dir: str
#     seed: int
#     test_size: float
#     max_replans: int


# def now_iso() -> str:
#     """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
#     return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# class AgenticDataScientist:
#     """
#     Offline Agentic Data Scientist (classification-focused).

#     Responsibilities:
#     - load and profile datasets
#     - create a plan (via planner)
#     - build preprocessors and select candidate models
#     - train and evaluate models
#     - reflect on results and optionally re-plan
#     - persist artefacts and update memory
#     """

#     def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
#         # Verbose controls logging output
#         self.verbose = verbose
#         # Simple persistent memory used to remember prior runs for a dataset fingerprint
#         self.memory = JSONMemory(memory_path)

#         # Context and transient state populated when run() is executed
#         self.ctx: Optional[RunContext] = None
#         self.state: Dict[str, Any] = {}

#     def log(self, msg: str) -> None:
#         """Print a log message when verbose is enabled."""
#         if self.verbose:
#             print(f"[AgenticDataScientist] {msg}")

#     def load_data(self, path: str) -> pd.DataFrame:
#         """Load a CSV into a pandas DataFrame and log its shape."""
#         self.log(f"Loading dataset: {path}")
#         df = pd.read_csv(path)
#         self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
#         return df

#     def run(
#         self,
#         data_path: str,
#         target: str,
#         output_root: str = "outputs",
#         seed: int = 42,
#         test_size: float = 0.2,
#         max_replans: int = 1,
#     ) -> str:
#         """
#         Main orchestration entry point.

#         Parameters:
#         - data_path: path to the CSV dataset
#         - target: target column name or 'auto' to infer
#         - output_root: directory where outputs are stored (subdir will be created)
#         - seed/test_size: training reproducibility and test split
#         - max_replans: maximum number of times to re-plan and re-run

#         Returns: path to the output directory for this run
#         """
#         # Create a unique id for ther run and creates a dedicated output folder in outputs to sotre logs,rtc;
#         run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
#         output_dir = os.path.join(output_root, run_id)
#         os.makedirs(output_dir, exist_ok=True)

#         # Populate run context with parameters and metadata
#         self.ctx = RunContext(
#             run_id=run_id,
#             started_at=now_iso(),
#             data_path=data_path,
#             target=target,
#             output_dir=output_dir,
#             seed=seed,
#             test_size=test_size,
#             max_replans=max_replans,
#         )
#         # Internal state used to track replanning attempts
#         self.state = {"replan_count": 0}

#         # Load dataset into memory
#         df = self.load_data(data_path)

#         # If client requested auto target detection, infer it from data
#         if target.strip().lower() == "auto":
#             inferred = infer_target_column(df)
#             if not inferred:
#                 raise ValueError("Could not infer target column. Please provide --target <name>.")
#             # Update context with inferred target name
#             self.ctx.target = inferred
#             self.log(f"Inferred target: {inferred}")

#         # Produce a dataset profile (EDA summary) and a fingerprint used for memory
#         profile = profile_dataset(df, self.ctx.target)
#         fp = dataset_fingerprint(df, self.ctx.target)

#         # Look up previous runs for the same dataset fingerprint (memory hint)
#         prev = self.memory.get_dataset_record(fp)
#         if prev:
#             self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fp}")

#         # Create an initial plan informed by the profile and optional memory hint
#         plan = create_plan(profile, memory_hint=prev)
#         self.log(f"Plan: {plan}")

#         # Execution loop: trains and evaluates, then optionally replans and repeats
#         while True:
#             # Build preprocessing pipeline tailored to the profile
#             preprocessor = build_preprocessor(profile,plan)
#             # Choose candidate models to try based on the profile
#             candidates = select_models(profile,plan, seed=self.ctx.seed)
#             self.log(f"Candidate models: {[n for n, _ in candidates]}")

#             # Train candidate models and persist intermediate artefacts
#             results = train_models(
#                 df=df,
#                 target=self.ctx.target,
#                 preprocessor=preprocessor,
#                 candidates=candidates,
#                 seed=self.ctx.seed,
#                 test_size=self.ctx.test_size,
#                 output_dir=self.ctx.output_dir,
#                 verbose=self.verbose,
#             )

#             # Evaluate the trained models and pick the best one
#             eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir)

#             # Reflect on the evaluation in the context of the dataset profile
#             reflection = reflect(
#                 dataset_profile=profile,
#                 evaluation=eval_payload["best_metrics"],
#                 all_metrics=eval_payload["all_metrics"],
#             )

#             # Persist core run artefacts for later review
#             save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
#             save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": plan})
#             save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
#             save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

#             # Generate a human-readable markdown report summarising the run
#             write_markdown_report(
#                 out_path=os.path.join(self.ctx.output_dir, "report.md"),
#                 ctx=self.ctx,
#                 fingerprint=fp,
#                 dataset_profile=profile,
#                 plan=plan,
#                 eval_payload=eval_payload,
#                 reflection=reflection,
#             )

#             # Update the memory store with outcomes from this run
#             self.memory.upsert_dataset_record(fp, {
#                 "last_seen": now_iso(),
#                 "target": self.ctx.target,
#                 "shape": profile["shape"],
#                 "best_model": eval_payload["best_metrics"]["model"],
#                 "best_metrics": eval_payload["best_metrics"],
#             })

#             # Decide whether the agent should attempt to re-plan and re-run
#             if not should_replan(reflection):
#                 # No replan suggested — finish the run
#                 break

#             # If we've already replanned the allowed number of times, stop
#             if self.state["replan_count"] >= self.ctx.max_replans:
#                 self.log("Replan suggested, but max_replans reached. Stopping.")
#                 break

#             # Otherwise, increment replan counter and apply the replan strategy
#             self.state["replan_count"] += 1
#             self.log(f"Replanning attempt #{self.state['replan_count']}...")

#             # apply_replan_strategy returns an updated (plan, profile) pair
#             plan, profile = apply_replan_strategy(plan, profile, reflection)

#         # Final log and return the directory containing run outputs
#         self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
#         return self.ctx.output_dir

# Orchestrator for an "agentic" offline data scientist pipeline.
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Agents
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory

# Tools
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models
from tools.evaluation import evaluate_best, write_markdown_report, save_json


# -----------------------------
# RUN CONTEXT
# -----------------------------
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# -----------------------------
# MAIN AGENT
# -----------------------------
class AgenticDataScientist:

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        self.verbose = verbose
        self.memory = JSONMemory(memory_path)

        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}

    # -----------------------------
    # LOGGING
    # -----------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")

    # -----------------------------
    # DATA LOADING
    # -----------------------------
    def load_data(self, path: str) -> pd.DataFrame:
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # -----------------------------
    # PLAN EXECUTION ENGINE (NEW 🔥)
    # -----------------------------
    def _execute_plan_steps(
        self,
        df: pd.DataFrame,
        plan: List[str],
        profile: Dict[str, Any]
    ) -> pd.DataFrame:

        target = self.ctx.target

        try:
            # -----------------------------
            # DROP CONSTANT FEATURES
            # -----------------------------
            if "drop_constant_features" in plan:
                cols = profile.get("constant_features", [])
                cols = [c for c in cols if c in df.columns]
                if cols:
                    df = df.drop(columns=cols)
                    self.log(f"Dropped constant features: {cols}")

            # -----------------------------
            # HANDLE MISSING VALUES
            # -----------------------------
            if "handle_severe_missing_data" in plan or "impute_missing_values" in plan:
                num_cols = profile["feature_types"]["numeric"]
                for col in num_cols:
                    if col in df.columns and df[col].isna().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
                self.log("Applied numeric median imputation")

            # -----------------------------
            # SMOTE (IMBALANCE HANDLING)
            # -----------------------------
            if "apply_smote" in plan:
                try:
                    from imblearn.over_sampling import SMOTE

                    X = df.drop(columns=[target])
                    y = df[target]

                    sm = SMOTE(random_state=self.ctx.seed)
                    X_res, y_res = sm.fit_resample(X, y)

                    df = pd.concat([X_res, y_res], axis=1)
                    self.log("Applied SMOTE oversampling")

                except ImportError:
                    self.log("WARNING: imblearn not installed — skipping SMOTE")

            # -----------------------------
            # SIMPLE FEATURE ENGINEERING
            # -----------------------------
            if "feature_engineering" in plan:
                self.log("Feature engineering placeholder (extend here)")

            # -----------------------------
            # DIMENSIONALITY REDUCTION (placeholder)
            # -----------------------------
            if "dimensionality_reduction" in plan:
                self.log("Dimensionality reduction placeholder (not implemented)")

        except Exception as e:
            self.log(f"WARNING: Plan step execution failed: {e}")

        return df

    # -----------------------------
    # MAIN RUN LOOP
    # -----------------------------
    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 1,
    ) -> str:

        # -----------------------------
        # INIT
        # -----------------------------
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )

        self.state = {"replan_count": 0}

        # -----------------------------
        # LOAD DATA (SAFE)
        # -----------------------------
        try:
            df = self.load_data(data_path)
        except Exception as e:
            self.log(f"FATAL: Failed to load dataset — {e}")
            raise

        # -----------------------------
        # TARGET INFERENCE
        # -----------------------------
        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError("Could not infer target column.")
            self.ctx.target = inferred
            self.log(f"Inferred target: {inferred}")

        # -----------------------------
        # PROFILE + MEMORY
        # -----------------------------
        profile = profile_dataset(df, self.ctx.target)
        fp = dataset_fingerprint(df, self.ctx.target)

        prev = self.memory.get_dataset_record(fp)
        if prev:
            self.log(f"Memory hit: best_model={prev.get('best_model')}")

        plan = create_plan(profile, memory_hint=prev)
        self.log(f"Plan: {plan}")

        # -----------------------------
        # EXECUTION LOOP
        # -----------------------------
        while True:

            try:
                # 🔥 APPLY PLAN STEPS
                df_processed = self._execute_plan_steps(df.copy(), plan, profile)

                # BUILD PIPELINE
                preprocessor = build_preprocessor(profile, plan)
                candidates = select_models(profile, plan, seed=self.ctx.seed)

                self.log(f"Candidate models: {[n for n, _ in candidates]}")

                # -----------------------------
                # RETRY TRAINING
                # -----------------------------
                results = None
                for attempt in range(3):
                    try:
                        results = train_models(
                            df=df_processed,
                            target=self.ctx.target,
                            preprocessor=preprocessor,
                            candidates=candidates,
                            seed=self.ctx.seed,
                            test_size=self.ctx.test_size,
                            output_dir=self.ctx.output_dir,
                            verbose=self.verbose,
                        )
                        break
                    except Exception as e:
                        self.log(f"Training attempt {attempt+1} failed: {e}")
                        if attempt == 2:
                            raise

                # EVALUATION
                eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir)

            except Exception as e:
                self.log(f"ERROR during execution: {e}")
                break

            # -----------------------------
            # REFLECTION
            # -----------------------------
            reflection = reflect(
                dataset_profile=profile,
                evaluation=eval_payload["best_metrics"],
                all_metrics=eval_payload["all_metrics"],
            )

            # -----------------------------
            # SAVE OUTPUTS
            # -----------------------------
            save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
            save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": plan})
            save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
            save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

            write_markdown_report(
                out_path=os.path.join(self.ctx.output_dir, "report.md"),
                ctx=self.ctx,
                fingerprint=fp,
                dataset_profile=profile,
                plan=plan,
                eval_payload=eval_payload,
                reflection=reflection,
            )

            # -----------------------------
            # MEMORY UPDATE
            # -----------------------------
            self.memory.upsert_dataset_record(fp, {
                "last_seen": now_iso(),
                "target": self.ctx.target,
                "shape": profile["shape"],
                "best_model": eval_payload["best_metrics"]["model"],
                "best_metrics": eval_payload["best_metrics"],
            })

            # -----------------------------
            # REPLAN LOGIC
            # -----------------------------
            if not should_replan(reflection):
                break

            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log("Max replans reached. Stopping.")
                break

            self.state["replan_count"] += 1
            self.log(f"Replanning attempt #{self.state['replan_count']}")

            plan, profile = apply_replan_strategy(plan, profile, reflection)

        # -----------------------------
        # DONE
        # -----------------------------
        self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
        return self.ctx.output_dir