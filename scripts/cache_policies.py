"""Route codec-salted objects to their corresponding LMCache L2 adapter."""

from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
    register_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    StorePolicy,
    register_store_policy,
)

SALTS = {"fp8": "jevkv-fp8-v1", "tq4": "jevkv-tq4-v1"}
ADAPTER = {SALTS["fp8"]: 0, SALTS["tq4"]: 1}


class CodecStorePolicy(StorePolicy):
    def select_store_targets(self, keys, adapters):
        present = {adapter.index for adapter in adapters}
        if present != {0, 1}:
            raise RuntimeError(f"Expected FP8 then TQ4 adapters, got {present}")
        return {index: [key for key in keys if ADAPTER.get(key.cache_salt) == index]
                for index in (0, 1)}

    def select_l1_deletions(self, keys):
        return list(keys)


class CodecPrefetchPolicy(PrefetchPolicy):
    def select_load_plan(self, keys, lookup_results, adapters):
        plan = {}
        for adapter in adapters:
            bitmap = lookup_results.get(adapter.index)
            if bitmap is None:
                continue
            selected = Bitmap(len(keys))
            for index, key in enumerate(keys):
                if ADAPTER.get(key.cache_salt) == adapter.index and bitmap.test(index):
                    selected.set(index)
            if selected.popcount():
                plan[adapter.index] = selected
        return plan


register_store_policy("jevkv_codec", CodecStorePolicy)
register_prefetch_policy("jevkv_codec", CodecPrefetchPolicy)
