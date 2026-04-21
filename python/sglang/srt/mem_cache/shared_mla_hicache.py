from __future__ import annotations

import ctypes
import logging
import os
import time
from typing import Optional

import numpy as np
import torch

from sglang.srt.distributed import in_the_same_node_as
from sglang.srt.layers.dp_attention import get_attention_dp_rank
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import HostTensorAllocator
from sglang.srt.utils import check_cuda_result, is_cuda

logger = logging.getLogger(__name__)

_ENV_FLAG = "SGLANG_EXPERIMENTAL_SHARE_CUDA_MLA_HICACHE_L2"
_POSIX_SHM_PREFIX = "/sglang_cuda_mla_hicache_l2"

_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01
_MAP_POPULATE = 0x8000
_O_CREAT = 0o100
_O_EXCL = 0o200
_O_RDWR = 0o2
_S_IRUSR = 0o400
_S_IWUSR = 0o200
_MAP_FAILED = ctypes.c_void_p(-1).value

_libc = ctypes.CDLL(None, use_errno=True)
_libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_libc.shm_open.restype = ctypes.c_int
_libc.shm_unlink.argtypes = [ctypes.c_char_p]
_libc.shm_unlink.restype = ctypes.c_int
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_longlong]
_libc.ftruncate.restype = ctypes.c_int
_libc.close.argtypes = [ctypes.c_int]
_libc.close.restype = ctypes.c_int
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_longlong,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int


class _PosixSharedMemory:
    def __init__(self, name: str, create: bool, size: int):
        self.name = name if name.startswith("/") else f"/{name}"
        self.size = size
        self._fd = -1
        self._addr = None
        self._owner = create
        self._name_bytes = self.name.encode("utf-8")

        flags = _O_RDWR | (_O_CREAT | _O_EXCL if create else 0)
        fd = _libc.shm_open(self._name_bytes, flags, _S_IRUSR | _S_IWUSR)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"shm_open failed for {self.name}")
        self._fd = fd

        if create and _libc.ftruncate(fd, size) != 0:
            err = ctypes.get_errno()
            _libc.close(fd)
            raise OSError(err, f"ftruncate failed for {self.name}")

        addr = _libc.mmap(
            None,
            size,
            _PROT_READ | _PROT_WRITE,
            _MAP_SHARED | _MAP_POPULATE,
            fd,
            0,
        )
        if ctypes.c_void_p(addr).value == _MAP_FAILED:
            err = ctypes.get_errno()
            _libc.close(fd)
            raise OSError(err, f"mmap failed for {self.name}")
        self._addr = addr

        if _libc.close(fd) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"close failed for {self.name}")
        self._fd = -1

    @property
    def buf(self):
        return (ctypes.c_uint8 * self.size).from_address(self._addr)

    @property
    def addr(self) -> int:
        return ctypes.c_void_p(self._addr).value

    def close(self):
        if self._addr is not None:
            if _libc.munmap(self._addr, self.size) != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"munmap failed for {self.name}")
            self._addr = None

    def unlink(self):
        if self._owner and _libc.shm_unlink(self._name_bytes) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"shm_unlink failed for {self.name}")
        self._owner = False

    @staticmethod
    def unlink_name(name: str):
        norm_name = name if name.startswith("/") else f"/{name}"
        name_bytes = norm_name.encode("utf-8")
        if _libc.shm_unlink(name_bytes) != 0:
            err = ctypes.get_errno()
            if err != 2:
                raise OSError(err, f"shm_unlink failed for {norm_name}")

def _is_mla_pool(pool) -> bool:
    if pool is None:
        return False
    # Unwrap HybridLinearKVPool to its full_kv_pool for the MLA check.
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool
    return isinstance(pool, MLATokenToKVPool)


def should_enable_shared_cuda_mla_l2(
    kv_cache,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    draft_kv_cache=None,
) -> bool:
    def group_size(group):
        return 1 if group is None else torch.distributed.get_world_size(group=group)

    def group_same_node(group, size):
        return size <= 1 or all(in_the_same_node_as(group, source_rank=0))

    if (
        os.getenv(_ENV_FLAG, "0").lower() not in {"1", "true", "yes", "on"}
        or not is_cuda()
        or not isinstance(kv_cache, MLATokenToKVPool)
        or not isinstance(draft_kv_cache, MLATokenToKVPool)
    ):
        return False

    cp_size = group_size(attn_cp_group)
    tp_size = group_size(attn_tp_group)
    if cp_size * tp_size <= 1:
        logger.warning(
            "%s is ignored because attn_cp_size * attn_tp_size <= 1 (%s * %s).",
            _ENV_FLAG,
            cp_size,
            tp_size,
        )
        return False

    cp_same_node = group_same_node(attn_cp_group, cp_size)
    tp_same_node = group_same_node(attn_tp_group, tp_size)
    local_ok = cp_same_node and tp_same_node

    if attn_cp_group is not None and cp_size > 1:
        sync_tensor = torch.tensor(
            [int(local_ok)],
            dtype=torch.int32,
            device="cpu",
        )
        torch.distributed.all_reduce(
            sync_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=attn_cp_group,
        )
        local_ok = bool(sync_tensor.item())

    if not local_ok:
        logger.warning(
            "%s is ignored because the attention communication plane is not fully "
            "local to one node: cp_same_node=%s, tp_same_node=%s, "
            "attn_cp_size=%s, attn_tp_size=%s",
            _ENV_FLAG,
            cp_same_node,
            tp_same_node,
            cp_size,
            tp_size,
        )
        return False

    return True


def get_shared_cuda_mla_l2_name(pp_rank: int) -> str:
    run_id = os.environ.get("SGLANG_RUN_ID", "sglang")
    dp_rank = get_attention_dp_rank()
    return (
        f"{_POSIX_SHM_PREFIX}_{run_id}"
        f"_pp{pp_rank}_dp{dp_rank}"
    )


class SharedMemoryHostTensorAllocator(HostTensorAllocator):
    supports_cuda_batch_memcpy = True

    def __init__(
        self,
        open_retry_count: int = 20,
        open_retry_interval_s: float = 0.05,
    ):
        super().__init__()
        self._records = []
        self._open_retry_count = open_retry_count
        self._open_retry_interval_s = open_retry_interval_s

    def allocate(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str = "cpu",
        shared_memory_name: Optional[str] = None,
        create: bool = True,
    ) -> torch.Tensor:
        if not shared_memory_name:
            raise ValueError(
                "SharedMemoryHostTensorAllocator requires shared_memory_name."
            )

        self.dims = dims
        self.dtype = dtype
        meta_tensor = torch.empty(size=dims, dtype=dtype, device="meta")
        num_bytes = meta_tensor.nbytes

        shm = None
        if create:
            try:
                shm = _PosixSharedMemory(
                    name=shared_memory_name,
                    create=True,
                    size=num_bytes,
                )
            except OSError:
                _PosixSharedMemory.unlink_name(shared_memory_name)
                shm = _PosixSharedMemory(
                    name=shared_memory_name,
                    create=True,
                    size=num_bytes,
                )
        else:
            last_err = None
            for _ in range(self._open_retry_count):
                try:
                    shm = _PosixSharedMemory(
                        name=shared_memory_name,
                        create=False,
                        size=num_bytes,
                    )
                    break
                except OSError as e:
                    last_err = e
                    time.sleep(self._open_retry_interval_s)
            if shm is None:
                raise RuntimeError(
                    f"Failed to attach shared memory {shared_memory_name}."
                ) from last_err

        np_array = np.ctypeslib.as_array(shm.buf)
        tensor = torch.from_numpy(np_array).view(dtype).view(*dims)
        self._records.append((shm, np_array, tensor))
        return tensor
