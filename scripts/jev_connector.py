"""LMCache MP connector that discards KV computed for salted requests."""

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector


class JevLMCacheMPConnector(LMCacheMPConnector):
    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        metadata.requests = [
            item for item in metadata.requests
            if item.direction != "STORE" or not item.cache_salt
        ]
        return metadata
