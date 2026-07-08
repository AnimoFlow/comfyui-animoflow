# Contributing to comfyui-animoflow

Bug reports, fixes, and new ideas are welcome. Two very different paths:

## Adding a model? You probably don't need a PR here

If you want your motion model available in AnimoFlow, wrap it in a container
behind the small HTTP contract and register it — your code keeps its own
repo, weights, and license, and **no CLA is needed**. Start at
[community/README.md](community/README.md) and
[community/CONTRACT.md](community/CONTRACT.md); registry entries are one
small PR against [community/models.json](community/models.json) (validated
automatically by CI).

## Contributing to the node pack itself

External contributions to this repository require a one-time Contributor
License Agreement (CLA) — it's what lets the project offer the
[commercial dual license](README.md#licensing) alongside AGPL-3.0. A bot
will ask you to sign on your first pull request; nothing to do in advance.
CLA text and the full per-repo license map: [AnimoFlow/legal](https://github.com/AnimoFlow/legal).

### Development setup

```bash
pip install pytest jsonschema
python -m pytest -q          # unit tests run without ComfyUI or a GPU
```

The full stack (model containers + ComfyUI) runs with `./install.sh up` —
see the [README](README.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Guidelines

- No silent fallbacks: a load/import/checkpoint failure must raise a loud,
  indicative error — never degrade to placeholder output.
- User-facing error messages come from the backend's classifier verbatim;
  don't invent new error strings in nodes.
- Workflow JSONs under `workflows/` carry hand-arranged layouts — don't
  regenerate them wholesale; edit the node you're changing.
- One change per PR, with a test where the change is testable headless.

Questions first? Open an issue or write to guy@animoflow.ai.
