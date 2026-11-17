# SPDX-License-Identifier: Apache-2.0
"""vLLM SAIL: inference on T-Head PPU accelerators.

PPU is CUDA-compatible, so this plugin deliberately reuses upstream vLLM's CUDA
code paths rather than vendoring its own attention backends, model runner or
sampler. See ``docs/developer_guide/architecture.md`` for the full rationale.

Two callables are exported as vLLM entry points:

* :func:`register`              -- ``vllm.platform_plugins``
* :func:`register_out_of_tree`  -- ``vllm.general_plugins``

``register_out_of_tree`` owns the complete general-plugin lifecycle,
including model registration. :func:`register_model` remains public for
backward compatibility and focused tests, but is not a separate entry point.
"""

# Route ``vllm_sail.*`` loggers through vLLM's logging configuration before
# any submodule creates its logger; without this, vLLM only configures its own
# ``vllm.*`` namespace and our INFO records would be silently dropped. Must run
# before the first ``vllm_sail.*`` import below.
from vllm_sail.logger import configure_sail_logging

configure_sail_logging()

from vllm_sail.version import __version__, __version_tuple__

__all__ = [
    "__version__",
    "__version_tuple__",
    "collect_env",
    "register",
    "register_model",
    "register_out_of_tree",
]

# ---------------------------------------------------------------------------
# Idempotency guards.
#
# vLLM loads `vllm.general_plugins` in every process -- API server, engine core
# subprocess, and each worker -- and may call a hook more than once within one
# process. Our `@patch` decorator raises RuntimeError on double application by
# design, so the hooks below must be individually idempotent rather than relying
# on module-import caching. (vLLM-MetaX omits this guard; vllm-ascend documents
# the same subtlety in its `_ensure_global_patch`.)
# ---------------------------------------------------------------------------
_SHIMS_INSTALLED = False
_PATCHES_APPLIED = False
_REGISTRIES_APPLIED = False
_MODELS_REGISTERED = False
_MONITOR_SHIM_INSTALLED = False


def _install_compat_shims() -> None:
    """Inject stubs for symbols missing due to vLLM version skew.

    Some vLLM builds ship a ``gpu_model_runner.py`` from a newer revision while
    other modules remain at an older version.  The runner then imports names that
    do not yet exist in those older modules, causing an ``ImportError`` before
    any patch can be applied.  This function injects minimal no-op stubs so the
    import succeeds; runtime behavior is unchanged because the stubs are only
    reached when the real symbol is absent.

    Known skew points handled here:

    * ``vllm.compilation.monitor.set_cudagraph_capturing_enabled``
      – newer runner imports it; older monitor.py lacks it.
    * ``vllm.config.ec_manager_config.EncoderCacheManagerMetadata``
      – newer runner imports it; older ec_manager_config.py only has
      ``EncoderCacheManagerConfig``.
    """
    global _MONITOR_SHIM_INSTALLED
    if _MONITOR_SHIM_INSTALLED:
        return

    import contextlib
    from dataclasses import dataclass as _dataclass

    # ── vllm.compilation.monitor ─────────────────────────────────────────────
    try:
        import vllm.compilation.monitor as _monitor
        if not hasattr(_monitor, "set_cudagraph_capturing_enabled"):
            @contextlib.contextmanager
            def _noop_set_cudagraph_capturing_enabled(enabled: bool = True):  # noqa: ARG001
                yield
            _monitor.set_cudagraph_capturing_enabled = _noop_set_cudagraph_capturing_enabled
    except ImportError:
        pass

    # ── vllm.config.ec_manager_config ────────────────────────────────────────
    try:
        import vllm.config.ec_manager_config as _ec_cfg
        if not hasattr(_ec_cfg, "EncoderCacheManagerMetadata"):
            @_dataclass
            class _EncoderCacheManagerMetadataStub:  # noqa: N801
                """Minimal stub; newer gpu_model_runner.py imports this name."""
            _ec_cfg.EncoderCacheManagerMetadata = _EncoderCacheManagerMetadataStub
    except ImportError:
        pass

    _MONITOR_SHIM_INSTALLED = True


def register() -> str:
    """``vllm.platform_plugins`` hook: return the PPU platform class path.

    Called when ``vllm.platforms.current_platform`` is first resolved. This is
    the *earliest* plugin hook vLLM offers, which is why the import shims that
    must beat upstream's own imports are installed here rather than in
    :func:`register_out_of_tree`.

    Keep this function cheap and free of torch-heavy imports: it runs during
    vLLM's platform resolution, before the platform object exists.
    """
    global _SHIMS_INSTALLED
    if not _SHIMS_INSTALLED:
        # vllm.platforms.cuda unconditionally imports this extension even when
        # vLLM_TARGET_DEVICE=empty deliberately omitted it. PPU registers the
        # actual ops later from PPUPlatform.import_kernels().
        from vllm_sail.native.bootstrap import install as _install_native_shim

        _install_native_shim()

        # Must run before anything imports `vllm.vllm_flash_attn`, whose
        # __init__ raises ImportError when vLLM's own FA extensions are absent
        # (PPU ships its own flash_attn / flash_attn_3 wheels instead).
        from vllm_sail.attention.flash_attn_shim import install as _install_fa_shim

        _install_fa_shim()
        _SHIMS_INSTALLED = True

    return "vllm_sail.platform.PPUPlatform"


def register_out_of_tree() -> None:
    """Install the complete ``vllm.general_plugins`` lifecycle.

    Order matters: the compatibility check runs first so that a version
    mismatch produces a clear error instead of an obscure patch failure. Model
    registration belongs to this same hook so an exact-name plugin allowlist
    cannot enable the PPU platform while filtering out its model overrides.
    """
    global _PATCHES_APPLIED, _REGISTRIES_APPLIED

    from vllm_sail.compat import check_vllm_compatibility

    check_vllm_compatibility()
    _install_compat_shims()

    if not _PATCHES_APPLIED:
        import vllm_sail.patch

        vllm_sail.patch.install()
        _PATCHES_APPLIED = True

    if not _REGISTRIES_APPLIED:
        import vllm_sail.ops  # noqa: F401
        import vllm_sail.registry

        # Call the explicit lifecycle API. Import-only registration is unsafe:
        # tuned-config patches import a registry child while patch.install() is
        # still running, so the package may already be cached at this point.
        vllm_sail.registry.register()

        # Layer NVTX follows its config flag, independent of the outer profiling env.
        from vllm_sail.profiling.layerwise import install as _install_layerwise

        _install_layerwise()

        # NVTX instrumentation is opt-in: this installs nothing unless
        # VLLM_SAIL_NVTX_PROFILE is set and the optional deps import.
        from vllm_sail.profiling import install as _install_profiling

        _install_profiling()
        _REGISTRIES_APPLIED = True

    register_model()


def register_model() -> None:
    """``vllm.general_plugins`` hook: register PPU model implementations."""
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return

    from vllm_sail.models import register_model as _register_model

    _register_model()
    _MODELS_REGISTERED = True


def collect_env() -> None:
    """Console entry point for the PPU environment report."""
    from vllm_sail.collect_env import main

    main()
