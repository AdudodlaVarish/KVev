"""LMCache server entry point with JevKV's codec policies and TQ4 serde."""

import cache_policies  # noqa: F401 - register policies before CLI starts
import turboquant_packed  # noqa: F401 - register packed KV serde
from lmcache.cli.main import main


if __name__ == "__main__":
    main()
