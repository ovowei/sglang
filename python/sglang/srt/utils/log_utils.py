from __future__ import annotations

import json
import logging
import os
import queue
import socket
import sys
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from typing import List, Optional, Union

import torch.distributed as dist
# Keep strong references so QueueListeners are not GC'd
_queue_listeners: List[QueueListener] = [] 

def create_log_targets(
    *, targets: Optional[List[str]], name_prefix: str
) -> List[logging.Logger]:
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    return [_create_log_target(t, name_prefix) for t in targets]


def _create_log_target(target: str, name_prefix: str) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(directory: str, name_prefix: str) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    pid = os.getpid()
    filename = os.path.join(directory, f"{hostname}_{rank}_{pid}.log")
    file_handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    # Use an unbounded queue so that logging never blocks the asyncio event loop.
    # A background thread (QueueListener) drains the queue and does the actual
    # disk writes, keeping the caller non-blocking.
    log_queue: queue.Queue = queue.Queue(-1)
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    _queue_listeners.append(listener)  # prevent GC

    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{hostname}_{rank}_{pid}",
        QueueHandler(log_queue),
    )

def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)
