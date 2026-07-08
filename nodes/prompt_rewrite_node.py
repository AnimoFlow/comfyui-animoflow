"""
AnimoFlow Prompt Rewrite node — thin wrapper over animoflow_stages.rewrite.

Rewrites a free-form prompt (any language) into a HumanML3D-style English
caption before it reaches a generator node. GUI-workflow only: the API
server rewrites at its own layer before compiling the DAG, so API-built
workflows never contain this node (no double rewrite). The implementation
is shared with the API server — see animoflow_stages/rewrite.py.

First non-skipped rewrite lazy-downloads ~3.2 GB from HF Hub (Qwen2.5-1.5B
+ MiniLM retriever + caption corpus). HumanML3D-style English input skips
the model entirely in ``auto`` mode. Failures raise loudly — no fallback
to the original prompt (per the no-silent-fallback policy).
"""
import json

from ..animoflow_stages.rewrite import rewrite


class AnimoFlowPromptRewriteNode:
    CATEGORY = "AnimoFlow/Text"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "rewrite_info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a person walks forward",
                    "tooltip": (
                        "Motion prompt in any language. Rewritten to a "
                        "HumanML3D-style English caption. First rewrite of a "
                        "non-English/free-form prompt downloads ~3.2 GB of "
                        "rewriter weights from HF Hub (one-time)."
                    ),
                }),
                "mode": (["auto", "force", "skip"], {
                    "default": "auto",
                    "tooltip": (
                        "auto: rewrite unless the input already looks like a "
                        "HumanML3D caption. force: always rewrite. skip: pass "
                        "through unchanged."
                    ),
                }),
            }
        }

    def run(self, prompt: str, mode: str) -> dict:
        res = rewrite(prompt, mode=mode)
        info = json.dumps(res.to_dict(), ensure_ascii=False)
        if res.skipped:
            print(f"[AnimoFlow_PromptRewrite] skipped ({mode}): {prompt!r}")
        else:
            print(f"[AnimoFlow_PromptRewrite] {prompt!r} → {res.rewritten!r} "
                  f"({res.latency_s:.2f}s)")
        return {"ui": {"text": [res.rewritten]},
                "result": (res.rewritten, info)}
