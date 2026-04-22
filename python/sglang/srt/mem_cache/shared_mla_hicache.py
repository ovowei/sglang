from __future__ import annotations

import atexit
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
_SHARED_CUDA_MLA_L2_RANK_INFO = None

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
_PAGE_SIZE = 4096

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

try:
    _libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    _libnuma.numa_tonode_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _libnuma.numa_tonode_memory.restype = ctypes.c_int
except OSError:
    _libnuma = None


class _PosixSharedMemory:
    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        numa_node: Optional[int] = None,
    ):
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

        mmap_flags = _MAP_SHARED
        if not (create and numa_node is not None and numa_node >= 0):
            mmap_flags |= _MAP_POPULATE

        addr = _libc.mmap(
            None,
            size,
            _PROT_READ | _PROT_WRITE,
            mmap_flags,
            fd,
            0,
        )
        if ctypes.c_void_p(addr).value == _MAP_FAILED:
            err = ctypes.get_errno()
            _libc.close(fd)
            raise OSError(err, f"mmap failed for {self.name}")
        self._addr = addr
        if create and numa_node is not None and numa_node >= 0:
            _bind_and_touch_memory(self._addr, self.size, numa_node, self.name)

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


def _bind_and_touch_memory(addr: int, size: int, numa_node: int, name: str) -> None:
    if _libnuma is None:
        logger.warning(
            "libnuma is unavailable; shared L2 segment %s cannot be bound to "
            "NUMA node %s. Falling back to first-touch placement.",
            name,
            numa_node,
        )
    else:
        ret = _libnuma.numa_tonode_memory(
            ctypes.c_void_p(addr),
            ctypes.c_size_t(size),
            ctypes.c_int(numa_node),
        )
        if ret != 0:
            err = ctypes.get_errno()
            logger.warning(
                "numa_tonode_memory failed for shared L2 segment %s on NUMA "
                "node %s: errno=%s. Falling back to first-touch placement.",
                name,
                numa_node,
                err,
            )

    buf = (ctypes.c_uint8 * size).from_address(addr)
    for offset in range(0, size, _PAGE_SIZE):
        buf[offset] = 0
    if size > 0:
        buf[size - 1] = 0


def get_current_cuda_device_numa_node() -> int:
    if not is_cuda() or not torch.cuda.is_available():
        return -1

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:
        logger.exception("Failed to query current CUDA device properties.")
        return -1

    candidates = _get_cuda_pci_device_candidates(props)
    if not candidates:
        logger.warning(
            "Failed to build CUDA PCI id candidates from device properties %s; "
            "shared L2 will use one fallback replica.",
            props,
        )
        return -1

    for pci_id in candidates:
        path = f"/sys/bus/pci/devices/{pci_id}/numa_node"
        try:
            with open(path, "r", encoding="utf-8") as f:
                numa_node = int(f.read().strip())
            return numa_node if numa_node >= 0 else -1
        except OSError:
            continue
        except ValueError:
            logger.warning("Invalid NUMA node content in %s.", path)
            return -1

    logger.warning(
        "Failed to locate NUMA node for CUDA PCI id candidates %s; shared L2 will use "
        "one fallback replica.",
        candidates,
    )
    return -1


def _get_cuda_pci_device_candidates(props) -> list[str]:
    def append(candidate: str):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    candidates = []
    pci_bus_id = getattr(props, "pci_bus_id", None)

    if isinstance(pci_bus_id, str):
        pci_bus_id = pci_bus_id.strip()
        parts = pci_bus_id.split(":")
        if len(parts) == 2:
            append(pci_bus_id)
            append(f"0000:{pci_bus_id}")
        elif len(parts) == 3:
            domain, bus, dev_func = parts
            if len(domain) > 4:
                domain = domain[-4:]
            append(f"{domain}:{bus}:{dev_func}")
            append(pci_bus_id)
        else:
            append(pci_bus_id)
        return candidates

    if isinstance(pci_bus_id, int):
        domain = int(getattr(props, "pci_domain_id", 0) or 0)
        pci_device_id = getattr(props, "pci_device_id", None)
        if pci_device_id is not None:
            append(f"{domain:04x}:{pci_bus_id:02x}:{int(pci_device_id):02x}.0")

        sysfs_prefix = f"{domain:04x}:{pci_bus_id:02x}:"
        try:
            for entry in os.listdir("/sys/bus/pci/devices"):
                if entry.startswith(sysfs_prefix):
                    append(entry)
        except OSError:
            pass

    return candidates


def get_shared_cuda_mla_l2_rank_info(
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> dict:
    global _SHARED_CUDA_MLA_L2_RANK_INFO
    if _SHARED_CUDA_MLA_L2_RANK_INFO is not None:
        return _SHARED_CUDA_MLA_L2_RANK_INFO

    numa_node = get_current_cuda_device_numa_node()

    def group_rank(group):
        if group is None:
            return 0
        return torch.distributed.get_rank(group=group)

    def gather_object(obj, group):
        if group is None or torch.distributed.get_world_size(group=group) <= 1:
            return [obj]
        gathered = [None] * torch.distributed.get_world_size(group=group)
        torch.distributed.all_gather_object(gathered, obj, group=group)
        return gathered

    local = {
        "global_rank": torch.distributed.get_rank(),
        "attn_cp_rank": group_rank(attn_cp_group),
        "attn_tp_rank": group_rank(attn_tp_group),
        "numa_node": numa_node,
    }

    records_by_rank = {local["global_rank"]: local}
    for record in gather_object(local, attn_cp_group):
        records_by_rank[record["global_rank"]] = record

    for payload in gather_object(list(records_by_rank.values()), attn_tp_group):
        for record in payload:
            records_by_rank[record["global_rank"]] = record

    same_numa_records = [
        record
        for record in records_by_rank.values()
        if record["numa_node"] == numa_node
    ]
    leader = min(
        same_numa_records or [local],
        key=lambda record: (
            record["attn_cp_rank"],
            record["attn_tp_rank"],
            record["global_rank"],
        ),
    )
    _SHARED_CUDA_MLA_L2_RANK_INFO = {
        "numa_node": numa_node,
        "is_shared_l2_numa_leader": local["global_rank"] == leader["global_rank"],
        "is_shared_l2_attn_leader": (
            local["attn_cp_rank"] == 0 and local["attn_tp_rank"] == 0
        ),
    }
    return _SHARED_CUDA_MLA_L2_RANK_INFO


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


def get_shared_cuda_mla_l2_name(pp_rank: int, numa_node: Optional[int] = None) -> str:
    run_id = os.environ.get("SGLANG_RUN_ID", "sglang")
    dp_rank = get_attention_dp_rank()
    name = f"{_POSIX_SHM_PREFIX}_{run_id}_pp{pp_rank}_dp{dp_rank}"
    if numa_node is not None and numa_node >= 0:
        name += f"_numa{numa_node}"
    return name


class SharedMemoryHostTensorAllocator(HostTensorAllocator):
    supports_cuda_batch_memcpy = True

    def __init__(
        self,
        open_retry_count: int = 20,
        open_retry_interval_s: float = 0.05,
    ):
        super().__init__()
        self._records = []
        self._closed = False
        self._open_retry_count = open_retry_count
        self._open_retry_interval_s = open_retry_interval_s
        atexit.register(self.close)

    def allocate(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str = "cpu",
        shared_memory_name: Optional[str] = None,
        create: bool = True,
        numa_node: Optional[int] = None,
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
                    numa_node=numa_node,
                )
            except OSError:
                _PosixSharedMemory.unlink_name(shared_memory_name)
                shm = _PosixSharedMemory(
                    name=shared_memory_name,
                    create=True,
                    size=num_bytes,
                    numa_node=numa_node,
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for shm, _np_array, _tensor in reversed(self._records):
            try:
                shm.close()
            except Exception:
                logger.exception("Failed to close shared memory segment %s.", shm.name)
            try:
                shm.unlink()
            except Exception:
                logger.exception("Failed to unlink shared memory segment %s.", shm.name)
        self._records.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
