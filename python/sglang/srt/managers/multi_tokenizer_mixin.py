from __future__ import annotations

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
"""

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import sys
import threading
from functools import partialmethod
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, Union

import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.tokenizer_communicator_mixin import _Communicator
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.environ import envs
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any):
        if ipc_name is None:
            # Some unhandled cases
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=False)
        self._mapping[ipc_name].send_pyobj(output)


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True
) -> Any:
    """Extract a field value from output by index, handling None and length checks.

    Args:
        output: The output object containing the field
        field_name: The name of the field to extract
        index: The index to access in the field list
        check_length: If True, check both field existence and length. If False, only check field existence.

    Returns:
        A list containing the field value at index, or None if not available.
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    if isinstance(field, dict):
        new_field = {}
        for k, v in field.items():
            if len(v) <= index:
                new_field[k] = None
            new_field[k] = v[index]
        return new_field

    if check_length:
        if len(field) <= index:
            return None

    return [field[index]]


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here."""
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_tokens=_extract_field_by_index(
                output, "spec_accepted_tokens", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),
            decode_ids=_extract_field_by_index(output, "decode_ids", i),
            read_offsets=_extract_field_by_index(output, "read_offsets", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            skip_special_tokens=_extract_field_by_index(
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            embeddings=_extract_field_by_index(output, "embeddings", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_tokens=_extract_field_by_index(
                output, "spec_accepted_tokens", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            output_strs=_extract_field_by_index(output, "output_strs", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is None:
                continue

            assert isinstance(
                recv_obj, BaseBatchReq
            ), "for multi-http-worker, recv_obj must be BaseBatchReq"

            # Send data using the corresponding socket
            for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):
                new_output = _handle_output_by_index(output, i)
                self.socket_mapping.send_output(ipc_name, new_output)


class MultiTokenizerRouter:
    """A router to receive requests from TokenizerWorker"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args
        context = zmq.asyncio.Context(3)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        # Start handle_loop simultaneously
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

    def _run_loop(self):
        self._loop.run_forever()

    async def router_worker_obj(self):
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()
            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        # special reqs will recv from scheduler, need to route to right worker
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        # Distribute result to each worker
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        # prevent init prefill bootstrapserver again
        disaggregation_mode = server_args.disaggregation_mode
        server_args.disaggregation_mode = "null"
        super().__init__(server_args, port_args)

        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        # For PD disaggregtion
        self.server_args.disaggregation_mode = disaggregation_mode
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.disaggregation_transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        # Communicator
        self.register_multi_tokenizer_communicator = _Communicator(
            self.send_to_scheduler, 2
        )

    def _attach_multi_http_worker_info(self, req: Union[BaseReq, BaseBatchReq]):

        if isinstance(req, BaseReq):
            req.http_worker_ipc = self.tokenizer_ipc_name
        elif isinstance(req, BaseBatchReq):
            req.http_worker_ipcs = [self.tokenizer_ipc_name] * len(req.rids)
        else:
            raise ValueError(f"Unknown req type: {type(req)}")


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID"""
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args information to share memory for multi-tokenizer"""
    # get main process ID
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    return args_shm


def _p2p_preflight_check(gpu_ids):
    """Verify every pair in gpu_ids has P2P read access enabled via NVML.

    Raises RuntimeError if pynvml is unavailable, NVML init fails, or any
    pair lacks P2P read access. Caller (the planner) should not catch
    this — the user explicitly opted into distribution and a silent
    fallback would reproduce the OOM the feature exists to fix.

    `gpu_ids` is a list of physical device id strings (as they appear in
    CUDA_VISIBLE_DEVICES). NVML uses physical indices independently of
    CUDA_VISIBLE_DEVICES, so we convert directly via int().

    Pattern follows custom_all_reduce_utils.is_full_nvlink (same module
    already uses pynvml). nvmlInit() does NOT initialize a CUDA context,
    so calling this in the parent process is safe before worker spawn.
    """
    try:
        import pynvml
    except ImportError as e:
        raise RuntimeError(
            f"[mm_worker_distribute] P2P preflight requires pynvml "
            f"(nvidia-ml-py) but it is not available: {e}"
        )
    if pynvml is None:
        raise RuntimeError(
            "[mm_worker_distribute] P2P preflight requires pynvml "
            "but the import returned None"
        )

    pynvml.nvmlInit()
    try:
        phys_ids = [int(g) for g in gpu_ids]
        handles = {p: pynvml.nvmlDeviceGetHandleByIndex(p) for p in phys_ids}
        # Use NVML_P2P_CAPS_INDEX_NVLINK (= 2). Same index used by
        # custom_all_reduce_utils.is_full_nvlink, proven to work in this
        # codebase. On the production target (H100/H200/H800/B300 + NVLink)
        # this is equivalent to "P2P read access available". The
        # NVML_P2P_CAPS_INDEX_READ constant has a type issue in some pynvml
        # builds; we hard-code the int 2 instead of relying on the symbol.
        p2p_index_nvlink = 2
        failed = []
        for a in phys_ids:
            for b in phys_ids:
                if a == b:
                    continue
                try:
                    status = pynvml.nvmlDeviceGetP2PStatus(
                        handles[a], handles[b], p2p_index_nvlink
                    )
                except pynvml.NVMLError as e:
                    failed.append(f"GPU{a}->GPU{b}: NVML error {e}")
                    continue
                if status != pynvml.NVML_P2P_STATUS_OK:
                    failed.append(f"GPU{a}->GPU{b}: status={status}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if failed:
        raise RuntimeError(
            "[mm_worker_distribute] P2P preflight FAILED. The feature "
            "requires every (worker GPU, scheduler GPU) pair to have "
            "P2P read access enabled. Failures:\n  "
            + "\n  ".join(failed)
            + "\nDisable SGLANG_MM_WORKER_GPU_DISTRIBUTE or fix the "
            "topology (NVLink) and retry."
        )


def _plan_slots_for(processes_num: int):
    """Decide whether to distribute tokenizer workers across GPUs.

    Returns a list of physical GPU ids (strings, as they appear in the
    parent's CUDA_VISIBLE_DEVICES) to round-robin across, or None if
    distribution should be disabled (no-op fall through to upstream
    behavior).

    When distribution is enabled, runs a P2P preflight against the
    selected GPU set; raises RuntimeError if any pair lacks P2P. The
    raise is intentional — the user opted in via env var, and silently
    falling back to single-GPU would reproduce the OOM the feature
    exists to fix.
    """
    if not envs.SGLANG_MM_WORKER_GPU_DISTRIBUTE.get():
        return None
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        logger.warning(
            "[mm_worker_distribute] requested but CUDA_VISIBLE_DEVICES is "
            "not set in the parent env; cannot infer the GPU set; skipping"
        )
        return None
    gpu_ids = [g.strip() for g in cvd.split(",") if g.strip()]
    if len(gpu_ids) <= 1 or processes_num <= 1:
        return None
    _p2p_preflight_check(gpu_ids)
    return gpu_ids


def monkey_patch_uvicorn_multiprocessing(timeout: float = 10):
    """Monkey patch uvicorn Multiprocess.

    Two patches:
    1. Existing: extend Process.is_alive default timeout. (Note: in current
       upstream uvicorn, keep_subprocess_alive passes timeout= explicitly,
       which overrides the partialmethod default; this patch is preserved
       as-is from prior code and is version-sensitive.)
    2. New (when SGLANG_MM_WORKER_GPU_DISTRIBUTE=1): rewrite parent
       CUDA_VISIBLE_DEVICES per worker slot before each Process spawn,
       restore after, so each worker child sees only one logical cuda:0.
       Mirrors the scheduler maybe_reindex_device_id pattern.
    """
    try:
        from uvicorn.supervisors import multiprocess as uvm
    except ImportError:
        logger.warning(
            "uvicorn.supervisors.multiprocess not found, skipping monkey patch"
        )
        return

    # 1. Existing: extend Process.is_alive default timeout
    uvm.Process.is_alive = partialmethod(uvm.Process.is_alive, timeout=timeout)

    # 2. New: GPU distribution patches
    UvicornProcess = uvm.Process

    def _spawn_pinned(multiprocess_self, idx):
        """Construct + start a uvicorn Process with CVD pinned for slot idx.

        If self._sglang_phys_gpus is None, falls through to unmodified
        behavior. Otherwise temporarily rewrites
        os.environ["CUDA_VISIBLE_DEVICES"] to the slot's physical GPU id,
        constructs and starts the Process (CPython spawn captures the env
        at .start() time), then restores the parent env.
        """
        phys_gpus = getattr(multiprocess_self, "_sglang_phys_gpus", None)
        original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        target = None
        if phys_gpus is not None:
            target = phys_gpus[idx % len(phys_gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = target
        try:
            process = UvicornProcess(
                multiprocess_self.config,
                multiprocess_self.target,
                multiprocess_self.sockets,
            )
            process.start()
        finally:
            if phys_gpus is not None:
                if original_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
        if target is not None:
            logger.info(
                f"[mm_worker_distribute] worker slot={idx} pid={process.pid} "
                f"pinned to CUDA_VISIBLE_DEVICES={target}"
            )
        return process

    def patched_init_processes(self):
        self._sglang_phys_gpus = _plan_slots_for(self.processes_num)
        if self._sglang_phys_gpus is not None:
            logger.info(
                f"[mm_worker_distribute] {self.processes_num} workers "
                f"will round-robin across {self._sglang_phys_gpus}"
            )
        for idx in range(self.processes_num):
            self.processes.append(_spawn_pinned(self, idx))

    def patched_keep_subprocess_alive(self):
        if self.should_exit.is_set():
            return
        for idx, process in enumerate(self.processes):
            if process.is_alive(timeout=self.config.timeout_worker_healthcheck):
                continue
            process.kill()
            process.join()
            if self.should_exit.is_set():
                return
            logger.info(f"Child process [{process.pid}] died")
            self.processes[idx] = _spawn_pinned(self, idx)

    def patched_restart_all(self):
        for idx, process in enumerate(self.processes):
            process.terminate()
            process.join()
            self.processes[idx] = _spawn_pinned(self, idx)

    def patched_handle_ttin(self):
        # SIGTTIN: append a new worker. Slot index = current len(processes).
        self.processes_num += 1
        idx = len(self.processes)
        self.processes.append(_spawn_pinned(self, idx))

    uvm.Multiprocess.init_processes = patched_init_processes
    uvm.Multiprocess.keep_subprocess_alive = patched_keep_subprocess_alive
    uvm.Multiprocess.restart_all = patched_restart_all
    uvm.Multiprocess.handle_ttin = patched_handle_ttin


class SenderWrapper:
    def __init__(self, port_args: PortArgs, send_to_scheduler: zmq.Socket):
        self.port_args = port_args
        self.send_to_scheduler = send_to_scheduler

    def send_pyobj(self, obj):
        if isinstance(obj, BaseReq):
            obj.http_worker_ipc = self.port_args.tokenizer_ipc_name
        self.send_to_scheduler.send_pyobj(obj)
