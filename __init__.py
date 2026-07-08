"""
comfyui-animoflow — ComfyUI custom-node package root.

Symlink THIS directory into ComfyUI:

    ln -sfn /path/to/comfyui-animoflow ComfyUI/custom_nodes/comfyui-animoflow

(The pre-2026-07 layout symlinked ``nodes/`` directly; the root is the
package now so the nodes can import the shared stage library at
``animoflow_stages/`` as a sibling subpackage.)
"""

if __package__:
    # Normal path: ComfyUI (or any importer) loads this directory as a
    # package, so the relative import below resolves.
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    def _check_backend():
        # First-run guidance: the nodes are thin HTTP clients — without the
        # model containers they load fine but every generate fails. Probe in
        # a daemon thread so ComfyUI startup is never delayed, and print ONE
        # actionable banner instead of letting the user discover the missing
        # backend at their first queued prompt.
        import urllib.request

        from .nodes import config as _cfg

        endpoints = (_cfg.MDM_ENDPOINT, _cfg.PRIORMDM_ENDPOINT, _cfg.MOMASK_ENDPOINT)
        up = 0
        for url in endpoints:
            try:
                with urllib.request.urlopen(url + "/health", timeout=2):
                    up += 1
            except Exception:
                pass
        if up == 0:
            print(
                "\n[AnimoFlow] Nodes loaded — but no model containers detected.\n"
                "[AnimoFlow] Motion generation needs the Docker backend. From the\n"
                "[AnimoFlow] comfyui-animoflow directory run:  ./install.sh up\n"
                "[AnimoFlow] (first time? start with:  ./install.sh doctor)\n",
                flush=True,
            )
        else:
            print(f"[AnimoFlow] Nodes loaded — {up}/{len(endpoints)} model containers reachable.", flush=True)

    import threading as _threading

    _threading.Thread(target=_check_backend, daemon=True).start()
else:
    # Imported WITHOUT package context — pytest's rootdir Package
    # collector and ad-hoc tooling do this. There is no ComfyUI here,
    # so empty mappings are the truthful answer, not a fallback.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Frontend extensions (draw-pad widgets) served by ComfyUI from here.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
