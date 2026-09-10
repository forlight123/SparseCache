"""Enable SparseCache's opt-in LMCache runtime instrumentation."""

import os


if os.environ.get("SPARSECACHE_LAYERWISE_PD_PATCH") == "1":
    from experiments.lossless_pd.lmcache_pd.layerwise_pd_runtime import install

    install()


if os.environ.get("SPARSECACHE_ANCHOR_PATCH") == "1":
    from experiments.lossless_pd.lmcache_pd.anchor_runtime import install

    install()

if os.environ.get("SPARSECACHE_RECEIVER_PATCH") == "1":
    from experiments.lossless_pd.lmcache_pd.receiver_runtime import install_receiver

    install_receiver()
