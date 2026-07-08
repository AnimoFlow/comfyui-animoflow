# Security Policy

## Reporting a vulnerability

Please report security issues privately to **guy@animoflow.ai** — do not
open a public issue. You'll get an acknowledgment within a few days.

## Scope notes

- `start.sh` binds ComfyUI and the model containers to **loopback by
  default** — ComfyUI is unauthenticated, so exposing it on a LAN is an
  explicit opt-in (`ANIMOFLOW_BIND`), at your own risk.
- Model containers speak plain HTTP on localhost and are process-isolated;
  they execute no code from prompts or uploaded files.
- Your `HF_TOKEN` (used to pull Kimodo weights) stays in your local `.env`
  and is sent only to Hugging Face.

There is no bug bounty program at this time.
