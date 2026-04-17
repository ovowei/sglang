from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    PoolEntry,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _build_attention_host_pool(
    pool,
    *,
    host_to_device_ratio: float,
    host_size: int,
    page_size: int,
    layout: str,
    allocator_type: Optional[str],
):
    common_kw = dict(
        host_to_device_ratio=host_to_device_ratio,
        host_size=host_size,
        page_size=page_size,
        layout=layout,
    )
    if isinstance(pool, NSATokenToKVPool):
        return MLATokenToKVPoolHost(
            pool,
            **common_kw,
            allocator_type=allocator_type,
            override_kv_cache_dim=pool.kv_cache_dim,
        )
    if isinstance(pool, MLATokenToKVPool):
        return MLATokenToKVPoolHost(pool, **common_kw, allocator_type=allocator_type)
    if isinstance(pool, MHATokenToKVPool):
        return MHATokenToKVPoolHost(pool, **common_kw, allocator_type=allocator_type)
    raise ValueError(f"Attention pool type {type(pool).__name__} not supported")


def _append_nsa_indexer_entry(
    entries: list[PoolEntry],
    *,
    name: PoolName,
    pool: NSATokenToKVPool,
    anchor_host,
    layout: str,
    allocator_type: Optional[str],
    layer_mapper,
) -> NSAIndexerPoolHost:
    indexer_host = NSAIndexerPoolHost(
        pool,
        anchor_host,
        layout,
        allocator_type=allocator_type,
    )

    entries.append(
        PoolEntry(
            name=name,
            host_pool=indexer_host,
            device_pool=pool,
            layer_mapper=layer_mapper,
            share_indices_with_anchor=True,
        )
    )
    return indexer_host


def build_radix_hybrid_stack(
    radix_cache: "HiRadixCache",
    params: "CacheInitParams",
    server_args: "ServerArgs",
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """HostPoolGroup + HybridCacheController for NSA and/or speculative draft."""
    try:
        kv = radix_cache.kv_cache
        layer_num = kv.layer_num

        # --- KV anchor host pool ---
        if isinstance(kv, NSATokenToKVPool):
            kv_host = _build_attention_host_pool(
                kv,
                host_to_device_ratio=server_args.hicache_ratio,
                host_size=server_args.hicache_size,
                page_size=radix_cache.page_size,
                layout=server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            # MHA / MLA: already created by HiRadixCache.__init__
            kv_host = radix_cache.token_to_kv_pool_host

        def layer_mapper(layer_id: int):
            if 0 <= layer_id < layer_num:
                return layer_id
            return None

        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=kv_host,
                device_pool=kv,
                layer_mapper=layer_mapper,
                is_primary_index_anchor=True,
            ),
        ]

        # --- NSA/DSA indexer sidecar ---
        if isinstance(kv, NSATokenToKVPool):
            _append_nsa_indexer_entry(
                entries,
                name=PoolName.INDEXER,
                pool=kv,
                anchor_host=kv_host,
                layout=server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
                layer_mapper=layer_mapper,
            )

        # --- Speculative draft KV sidecar ---
        transfer_layer_num = layer_num
        draft_pool = params.draft_token_to_kv_pool
        if draft_pool is not None:
            if isinstance(draft_pool, HybridLinearKVPool):
                draft_pool = draft_pool.full_kv_pool

            draft_host = _build_attention_host_pool(
                draft_pool,
                host_to_device_ratio=kv_host.size / draft_pool.size,
                host_size=0,
                page_size=radix_cache.page_size,
                layout=server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )

            draft_layer_num = draft_pool.layer_num
            transfer_layer_num = max(layer_num, draft_layer_num)

            def draft_layer_mapper(layer_id: int):
                if 0 <= layer_id < draft_layer_num:
                    return layer_id
                return None

            entries.append(
                PoolEntry(
                    name=PoolName.DRAFT,
                    host_pool=draft_host,
                    device_pool=draft_pool,
                    layer_mapper=draft_layer_mapper,
                    share_indices_with_anchor=True,
                )
            )

            radix_cache.draft_kv_pool_host = draft_host
            if isinstance(draft_pool, NSATokenToKVPool):
                draft_indexer_host = _append_nsa_indexer_entry(
                    entries,
                    name=PoolName.DRAFT_INDEXER,
                    pool=draft_pool,
                    anchor_host=draft_host,
                    layout=server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                    layer_mapper=draft_layer_mapper,
                )
                radix_cache.draft_indexer_pool_host = draft_indexer_host

        host_pool_group = HostPoolGroup(entries)
        cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            radix_cache.page_size,
            radix_cache.tp_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=radix_cache.pp_rank,
            pp_size=radix_cache.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            transfer_layer_num=transfer_layer_num,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = kv_host
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller

        if draft_pool is not None:
            draft_pool.register_layer_transfer_counter(
                cache_controller.layer_done_counter
            )

        pool_names = [e.name.value.upper() for e in entries]
        logger.info(
            "Hybrid hierarchical cache: HostPoolGroup(%s), HybridCacheController, "
            "transfer_layer_num=%s",
            " + ".join(pool_names),
            transfer_layer_num,
        )
    except Exception:
        logger.exception("build_radix_hybrid_stack failed")
        raise


def build_mamba_hybrid_stack(
    mamba_cache: "HiMambaRadixCache",
    params: "CacheInitParams",
    server_args: "ServerArgs",
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
) -> None:
    """HostPoolGroup (KV + Mamba) + HybridCacheController for hybrid SSM models."""
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_kv_pool_host = _build_attention_host_pool(
            kvcache,
            host_to_device_ratio=server_args.hicache_ratio,
            host_size=server_args.hicache_size,
            page_size=params.page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        mamba_pool_host = MambaPoolHost(
            params.req_to_token_pool.mamba_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            allocator_type=server_args.hicache_storage_backend,
            layout=server_args.hicache_mem_layout,
        )

        full_layer_ids = sorted(hybrid_kv.full_attention_layer_id_mapping.keys())
        mamba_layer_ids = sorted(params.req_to_token_pool.mamba_map.keys())
        transfer_layer_num = len(set(full_layer_ids) | set(mamba_layer_ids))
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)

        def kv_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return full_layer_mapping.get(layer_id)

        def mamba_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return mamba_layer_mapping.get(layer_id)

        host_pool_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=full_kv_pool_host,
                    device_pool=kvcache,
                    layer_mapper=kv_layer_mapper,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.MAMBA,
                    host_pool=mamba_pool_host,
                    device_pool=params.req_to_token_pool.mamba_pool,
                    layer_mapper=mamba_layer_mapper,
                    host_evict_fn=mamba_cache.evict_mamba_host,
                    device_evict_fn=mamba_cache.evict_mamba,
                ),
            ]
        )
        cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            transfer_layer_num=transfer_layer_num,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = full_kv_pool_host
        mamba_cache.mamba_pool_host = mamba_pool_host
        mamba_cache.transfer_layer_num = transfer_layer_num
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Hybrid hierarchical cache: HostPoolGroup(KV + MAMBA), HybridCacheController, "
            "transfer_layer_num=%s",
            transfer_layer_num,
        )
    except Exception:
        logger.exception("build_mamba_hybrid_stack failed")
        raise
