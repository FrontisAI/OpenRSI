# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2024 Edan Toledo
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/EdanToledo/Stoix/blob/main/LICENSE

import abc
import json
import logging
import os
import time
from uuid import uuid4
import zipfile
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import jsonlines
import numpy as np

# import weave
from colorama import Fore, Style
from dataclasses_json import DataClassJsonMixin
from omegaconf import DictConfig
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict

import wandb
from dojo.config_dataclasses.logger import LoggerConfig


class LogEvent(Enum):
    AGENT = "agent"
    SOLVER = "solver"
    TASK = "task"
    INTERPRETER = "interpreter"
    MISC = "misc"
    EVAL = "eval"
    LLM_CLIENT = "llm_client"


EVENT_COLOURS = {
    LogEvent.AGENT: Fore.MAGENTA,
    LogEvent.EVAL: Fore.GREEN,
    LogEvent.INTERPRETER: Fore.WHITE,
    LogEvent.SOLVER: Fore.CYAN,
    LogEvent.TASK: Fore.YELLOW,
    LogEvent.MISC: Fore.RED,
    LogEvent.LLM_CLIENT: Fore.BLUE,
}


class CollectiveLogger:
    """The main logger."""

    def __init__(self) -> None:
        self._cfg = ...

    @property
    def cfg(self):
        if self._cfg == ...:
            raise ValueError("No config set.")
        return self._cfg

    def config(self, config: DictConfig) -> None:
        if config:
            self.logger: BaseLogger = _make_multi_logger(config)
            self._cfg = config
            self.low_resource_use = False
        else:
            self.low_resource_use = True
            self._cfg = None
        return self

    def log(self, metrics: Dict, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        """Log a dictionary metrics at a given timestep.

        Args:
            metrics (Dict): dictionary of metrics to log.
            event (LogEvent): the event that the metrics are associated with.
            step (Optional[int]): the step at which these metrics are logged.
        """
        if not self.low_resource_use:
            self.logger.log_dict(metrics, event, step)

    def log_file(self, file_path: str) -> None:
        if not self.low_resource_use:
            self.logger.log_file(file_path)

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()

    def process_event_message(self, msg: object, event: Optional[LogEvent] = None) -> None:
        if not event:
            event = LogEvent.MISC
        return f"{EVENT_COLOURS[event]}{Style.BRIGHT}[{event.value.upper()}] - {msg}{Style.RESET_ALL}"

    def info(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.info(self.process_event_message(msg, event))

    def warning(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.warning(self.process_event_message(msg, event))

    def error(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.error(self.process_event_message(msg, event))

    def debug(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.debug(self.process_event_message(msg, event))

    def critical(self, msg: object, event: Optional[LogEvent] = None) -> None:
        logging.critical(self.process_event_message(msg, event))


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        """Log a single metric."""
        raise NotImplementedError

    def log_dict(self, data: Dict, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            if isinstance(value, DataClassJsonMixin):
                value = value.to_dict()
            self.log_stat(
                key,
                value,
                event,
                step,
            )

    def log_file(self, file_path: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, loggers: List[BaseLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, event, step)

    def log_dict(self, data: Dict, event: LogEvent, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_dict(data, event, step)

    def log_file(self, file_path: str) -> None:
        for logger in self.loggers:
            logger.log_file(file_path)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class WandBLogger(BaseLogger):
    """Logger for wandb.ai."""

    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        tags = list(cfg.logger.tags)
        project = cfg.logger.wandb_project_name
        entity = cfg.logger.wandb_entity

        wandb.init(entity=entity, project=project, tags=tags, config=cfg)

        self.detailed_logging = cfg.logger.detailed_logging
        self.unique_token = unique_token

    def log_stat(self, key: str, value: float, event: Union[LogEvent, str], step: Optional[int] = None) -> None:
        if isinstance(event, LogEvent):
            namespace = event.value
        else:
            namespace = event
        data_to_log = {f"{namespace}/{key}": value}
        wandb.log(data_to_log, step=step)

    def log_file(self, file_path: str) -> None:
        wandb.save(file_path)

    def stop(self) -> None:
        wandb.finish()  # type: ignore


class ConsoleLogger(BaseLogger):
    """Logger for writing to stdout."""

    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        self.logger = logging.getLogger("Console")

        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s{Style.RESET_ALL}", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, event: LogEvent, step: Optional[int] = None) -> None:
        if isinstance(event, LogEvent):
            namespace = event.value
            colour = EVENT_COLOURS[event]
        else:
            namespace = event
            colour = EVENT_COLOURS[LogEvent.MISC]

        # Replace underscores with spaces and capitalise keys.
        key = key.replace("_", " ").capitalize()

        self.logger.info(f"{colour}{Style.BRIGHT}{namespace.upper()} - {key}: {value:.3f}{Style.RESET_ALL}")

    def log_dict(self, data: Dict, event: LogEvent, step: Optional[int] = None) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")

        if isinstance(event, LogEvent):
            namespace = event.value
            colour = EVENT_COLOURS[event]
        else:
            namespace = event
            colour = EVENT_COLOURS[LogEvent.MISC]
        # Replace underscores with spaces and capitalise keys.
        keys = [k.replace("_", " ").capitalize() for k in data.keys()]
        # Round values to 3 decimal places if they are floats.
        values = []
        for v in data.values():
            if isinstance(v, int):
                values.append(v)
            elif isinstance(v, float):
                values.append(f"{float(v):.3f}")
            elif isinstance(v, str):
                values.append(f"\n'{v}'\n")
            else:
                t = type(v)
                v = str(v)
                values.append(f"\n\n{t.__name__.capitalize()}: \n'{v}' \n")

        log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values)])

        self.logger.info(f"{colour}{Style.BRIGHT}{namespace.upper()} - {log_str}{Style.RESET_ALL}")

    def log_file(self, file_path: str) -> None:
        pass


class JsonLogger(BaseLogger):
    def __init__(self, cfg: LoggerConfig, unique_token: str, flush_interval: int = 100) -> None:
        self.json_logs_path = Path(os.path.join(cfg.logger.output_dir, "json"))
        self.json_logs_path.mkdir(parents=True, exist_ok=True)

        self.buffer = defaultdict(list)  # {namespace: [log_entries]}
        self.last_flush_time = time.monotonic()
        self.flush_interval = flush_interval  # Time in seconds between file writes

    def log_dict(self, data: Dict, event: Union["LogEvent", str], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics under the same step."""
        data = flatten_dict(data, sep="/")  # Flatten nested dicts
        self.log_stat(data, event, step)

    def log_stat(self, data: Dict[str, float], event: Union["LogEvent", str], step: Optional[int] = None) -> None:
        """Logs all key-value pairs under the same step in a buffer."""
        if isinstance(event, LogEvent):
            namespace = event.value
        else:
            namespace = event

        # Create a log entry with timestamp, step, and data
        log_entry = {"timestamp": datetime.now().isoformat(), "step": step, "data": data}

        # Check if data is JSON serializable
        try:
            json.dumps(log_entry)
            self.buffer[namespace].append(log_entry)
        except (TypeError, ValueError):
            print(f"Skipping non-serializable data: {data}")
            return

        # Check if it's time to flush
        if time.monotonic() - self.last_flush_time > self.flush_interval:
            self.flush_logs()

    def flush_logs(self):
        """Writes all buffered logs to disk using jsonlines."""
        for namespace, entries in self.buffer.items():
            if not entries:
                continue

            file_path = self.json_logs_path / f"{namespace}.jsonl"

            # Append to the jsonlines file
            with jsonlines.open(file_path, mode="a") as writer:
                for entry in entries:
                    writer.write(entry)

        # Clear buffer and update last flush time
        self.buffer.clear()
        self.last_flush_time = time.monotonic()

    def log_file(self, file_path: str) -> None:
        pass

    def stop(self) -> None:
        """Ensure all logs are written before stopping."""
        self.flush_logs()


def _make_multi_logger(cfg: LoggerConfig) -> BaseLogger:
    """Creates a MultiLogger given a config"""

    loggers: List[BaseLogger] = []
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S%f") + str(uuid4())

    if cfg.logger.use_wandb:
        loggers.append(WandBLogger(cfg, unique_token))
    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg, unique_token))
    if cfg.logger.use_json:
        loggers.append(JsonLogger(cfg, unique_token))

    return MultiLogger(loggers)


def describe(x: np.ndarray) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""

    if not isinstance(x, np.ndarray):
        return x
    elif x.size <= 1:
        return np.squeeze(x)

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}


LOGGERS = dict()


def init_logger():
    LOGGERS["main"] = CollectiveLogger()


def get_logger():
    return LOGGERS["main"]


def config_logger(*args, **kwargs):
    return get_logger().config(*args, **kwargs)


init_logger()
