import asyncio
import secrets
import random



# titanic code
CODE_SUCCESS = r'''
from __future__ import annotations

import sys
import warnings
from pathlib import Path
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parent
# DATA_DIR = Path(os.environ.get("DATA_DIR", "/mnt/pubdatasets2/MLTasks/Selected_Dojo/titanic"))
DATA_DIR = Path(os.environ.get("DATA_DIR"))
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe enriched with engineered features."""
    engineered = df.copy()

    engineered["FamilySize"] = (
        engineered["SibSp"].fillna(0) + engineered["Parch"].fillna(0) + 1
    )
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    engineered["CabinKnown"] = engineered["Cabin"].notna().astype(int)

    titles = engineered["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna(
        "Unknown"
    )
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Jonkheer": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Capt": "Officer",
    }
    engineered["Title"] = titles.replace(title_map)

    ticket_prefix = (
        engineered["Ticket"]
        .astype(str)
        .str.replace(r"[\d\.\/]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )
    engineered["TicketPrefix"] = ticket_prefix.replace("", "NONE")

    engineered["FarePerPerson"] = engineered["Fare"] / engineered["FamilySize"]

    return engineered


def build_model(feature_frame: pd.DataFrame) -> Pipeline:
    """Create a model pipeline configured for the engineered Titanic features."""
    numeric_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "FarePerPerson",
        "Pclass",
        "IsAlone",
        "CabinKnown",
    ]
    categorical_features = ["Sex", "Embarked", "Title", "TicketPrefix"]

    onehot_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        onehot_kwargs["sparse_output"] = False
    else:
        onehot_kwargs["sparse"] = False

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**onehot_kwargs)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = GradientBoostingClassifier(random_state=RANDOM_SEED)

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def run_cross_validation(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> None:
    """Report cross-validation accuracy using stratified folds."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        model,
        features,
        target,
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )
    print(
        f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}",
        flush=True,
    )
def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    print(f"Loading data from {TRAIN_PATH}")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print(
            "Missing data files. Expected train.csv and test.csv under data/public/",
            file=sys.stderr,
        )
        return 1

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)

    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "CabinKnown",
        "Title",
        "TicketPrefix",
        "FarePerPerson",
    ]

    X_train = train_features[feature_columns]
    y_train = train_df["Survived"]
    X_test = test_features[feature_columns]

    model = build_model(X_train)

    print("Training model...", flush=True)
    run_cross_validation(model, X_train, y_train)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions.astype(int),
        }
    )

    submission_path = ROOT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved predictions to {submission_path.as_posix()}", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


'''

# client users can copy from here (共享httpx client)


import asyncio
import time
import uuid
import httpx
BASE_URL = "http://203.0.113.10:6580"  # 占位地址：替换为你的控制器地址
TEST_CONCURRENCY = 16


API_KEY = "mlsandbox-oss-3fad186210e530e6b4cc53576cd2723e"

async def get_sandbox_result(
    client: httpx.AsyncClient,         # ✅ 复用的客户端从外部传入
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "gpu",
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    # wait_timeout: int = 100,
    poll_interval: int = 10,
):
    normalized_resource = (resource_type or "cpu").lower()
    normalized_priority = priority if priority in (1, 2) else 1
    trace_id = uuid.uuid4().hex
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": normalized_priority,
        "idempotency_key": trace_id,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
            "HF_ENDPOINT": "https://hf-mirror.com",
        },
    }
    headers = {"X-API-Key": API_KEY, "X-Trace-ID": trace_id}

    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    def format_httpx_error(exc: Exception, fallback_url: str) -> str:
        parts = [f"type={type(exc).__name__}"]
        req = getattr(exc, "request", None)
        if req is not None:
            method = getattr(req, "method", "UNKNOWN")
            url = getattr(req, "url", fallback_url)
            parts.append(f"method={method}")
            parts.append(f"url={url}")
        else:
            parts.append(f"url={fallback_url}")

        message = str(exc).strip()
        if message:
            parts.append(f"message={message}")
        else:
            parts.append(f"repr={exc!r}")

        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        return "; ".join(str(p) for p in parts)

    def retry_after_seconds(resp: httpx.Response, fallback: float) -> float:
        value = resp.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, min(float(value), 30.0))
            except ValueError:
                pass
        return fallback

    # 1) 只提交，不等待。同一个逻辑 job 的 submit retry 复用同一个 idempotency_key。
    submit_connect_retries = 50
    transient_submit_statuses = {429, 500, 502, 503, 504}
    submit_resp = None
    for submit_attempt in range(1, submit_connect_retries + 1):
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",              # ✅ 使用 base_url，path 写相对地址
                json=payload,
                headers=headers,
            )
            if submit_resp.status_code in transient_submit_statuses and submit_attempt < submit_connect_retries:
                retry_delay = retry_after_seconds(
                    submit_resp,
                    min(30.0, 1.5 ** submit_attempt) + secrets.SystemRandom().uniform(0.0, 1.0),
                )
                print(
                    f"WARNING: transient submit status {submit_resp.status_code} trace_id={trace_id}; "
                    f"retry {submit_attempt}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            break
        except (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.TransportError,
        ) as e:
            error_detail = format_httpx_error(e, BASE_URL)
            if submit_attempt < submit_connect_retries:
                retry_delay = min(30.0, 1.5 ** submit_attempt) + secrets.SystemRandom().uniform(0.0, 1.0)
                print(
                    f"WARNING: transient submit error to {BASE_URL}: {error_detail}; trace_id={trace_id}; "
                    f"retry {submit_attempt}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to submit job to sandbox API at {BASE_URL}: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {
                "error": "submit_failed_after_retries",
                "detail": error_msg,
                "type": type(e).__name__,
                "trace_id": trace_id,
            }

    if submit_resp is None:
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at {BASE_URL}",
            "type": "ConnectError",
            "trace_id": trace_id,
        }
    submit_data = safe_json(submit_resp)
    print("submit_data:", submit_data)
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data, "trace_id": trace_id}

    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {"error": "no job_id returned", "data": submit_data, "trace_id": trace_id}

    # 2) 轮询直到结束或完全超时
    wait_status = ["running", "queued"]
    finished_status = ["completed", "failed", "cancelled"]
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    max_poll_error_retries = 20
    malformed_poll_retries = 0
    max_malformed_poll_retries = 3
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if r.status_code in (502, 503, 504, 429):
                if poll_error_retries < max_poll_error_retries:
                    poll_error_retries += 1
                    retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + secrets.SystemRandom().uniform(0.0, 0.5)
                    print(
                        f"WARNING: transient poll status {r.status_code} for {job_id}, "
                        f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                        flush=True,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
        except httpx.TransportError as e:
            error_detail = format_httpx_error(e, f"{BASE_URL}/api/v1/jobs/{job_id}")
            if poll_error_retries < max_poll_error_retries:
                poll_error_retries += 1
                retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + secrets.SystemRandom().uniform(0.0, 0.5)
                print(
                    f"WARNING: transient poll transport error for {job_id}: {error_detail}; "
                    f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to poll job status: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "poll_failed", "detail": error_msg, "job_id": job_id, "type": type(e).__name__}
        poll_error_retries = 0
        data = safe_json(r)
        status = data.get("status")
        print(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        if status in wait_status:
            await asyncio.sleep(poll_interval + secrets.SystemRandom().uniform(0.0, 1.0))
            continue
        if status in finished_status:
            return 200, data
        malformed_poll_retries += 1
        if malformed_poll_retries <= max_malformed_poll_retries:
            retry_delay = min(5.0, 1.0 + malformed_poll_retries) + secrets.SystemRandom().uniform(0.0, 0.5)
            print(
                f"WARNING: unexpected poll body for {job_id}: status={status}; "
                f"retry {malformed_poll_retries}/{max_malformed_poll_retries} in {retry_delay:.2f}s",
                flush=True,
            )
            await asyncio.sleep(retry_delay)
            continue
        return 502, {
            "error": "unexpected_poll_response",
            "job_id": job_id,
            "trace_id": trace_id,
            "data": data,
        }

    return 504, {"error": "wait_timeout exceeded", "job_id": job_id, "trace_id": trace_id}


TITANIC_DATA_DIR = "/mnt/pubdatasets2/MLTasks/Selected_Dojo/titanic"


async def main(task_id, resource_type: str, priority: int) -> None:
    if task_id != 1:
        raise ValueError("Only task_id=1 (Titanic) is included in this public test client.")
    code = CODE_SUCCESS
    data_dir = TITANIC_DATA_DIR

    # ✅ 这里统一创建并复用 AsyncClient
    n_concurrent = TEST_CONCURRENCY  # 可以根据需要调整并发数
    max_connections = min(max(int(1.5 * n_concurrent), 16), 512)
    max_keepalive_connections = min(max_connections, 256)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(connect=10, read=100, write=30, pool=60)
    async with httpx.AsyncClient(base_url=BASE_URL, limits=limits, timeout=timeout) as client:

        print(f"\n== Starting {n_concurrent} concurrent requests ==")

        tasks = []
        for i in range(n_concurrent):
            task = get_sandbox_result(
                client=client,            # ✅ 传入复用的 client
                code_str=code,
                data_dir=data_dir,
                resource_type=resource_type,
                priority=priority,
                job_timeout=3600
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        print(f"\n== All {n_concurrent} requests completed in {end_time - start_time:.2f} seconds ==\n")

        # 打印
        count_normal = 0
        count_unexpected = 0
        unexpected_details = {}
        result_200 = {}
        read_timeout_503_details = {}
        failed_200_details = []
        for i, result in enumerate(results):
            print(f"\n== Request {i+1}/{n_concurrent} Result Summary ==")
            if isinstance(result, Exception):
                print(f"Error: {result}")
                continue

            status_code, payload = result
            if status_code == 503:
                read_timeout_503_details[i] = payload

            if status_code == 200:
                print("job_status:", payload.get("status"))
                print("job_id:", payload.get("job_id"))
                result_data = payload.get("result") or {}
                print("run_result_status:", result_data.get("result"))
                print("score:", result_data.get("score"))
                if result_data.get("result") not in result_200:
                    result_200[result_data.get("result")] = 1
                else:
                    result_200[result_data.get("result")] += 1
                if result_data.get("result") != "success":
                    run_log = result_data.get("run_log") or ""
                    if isinstance(run_log, str):
                        first_line = next((line.strip() for line in run_log.splitlines() if line.strip()), "")
                    else:
                        first_line = str(run_log)
                    failed_200_details.append(
                        {
                            "request_index": i + 1,
                            "job_id": payload.get("job_id"),
                            "job_status": payload.get("status"),
                            "run_result_status": result_data.get("result"),
                            "score": result_data.get("score"),
                            "error_message": payload.get("error_message"),
                            "run_log_first_line": first_line[:300],
                        }
                    )
                count_normal += 1
            else:
                print("job_id:", payload.get("job_id"))
                print("run_status:", payload.get("status"))
                if payload.get("error"):
                    print("error:", payload["error"])
                count_unexpected += 1
                unexpected_details[i] = payload

        print(f"\n== Summary: {count_normal} normal completions, {count_unexpected} unexpected results out of {n_concurrent} requests ==\n")
        print("Details of unexpected results:")
        for i, payload in unexpected_details.items():
            print(f"{i}:, {payload}")
        print("\nDetails of 503 read timeout errors:")
        for i, payload in read_timeout_503_details.items():
            print(f"{i}:, {payload}")
        print("\nDetails of 200 results:")
        print(result_200)
        print("\nDetails of 200 but non-success results:")
        for item in failed_200_details:
            print(item)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run task by ID")
    p.add_argument("--task_id", "-t", type=int, choices=[1], default=1, help="Public test task ID (Titanic only)")
    p.add_argument(
        "--resource_type",
        "-r",
        choices=["gpu", "cpu"],
        default="gpu",
        help="指定任务可使用的资源类型(cpu/gpu)",
    )
    p.add_argument(
        "--priority",
        "-p",
        type=int,
        choices=[1, 2],
        default=1,
        help="任务优先级：1(高) / 2(低)，默认1",
    )
    args = p.parse_args()
    task_id = args.task_id
    asyncio.run(
        main(
            task_id=task_id,
            resource_type=args.resource_type.lower(),
            priority=args.priority,
        )
    )
