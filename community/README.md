# AnimoFlow community model containers

This directory is how researchers plug **their own motion models** into
AnimoFlow — as Docker containers speaking a small, versioned HTTP contract —
**without their code ever entering AnimoFlow's repositories**.

| File | What it is |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | The normative HTTP contract (`contract_version: 1`): `/health`, `/generate_async`, `/progress/{job_id}`, the NPZ payload, container conventions. |
| [`model-container-template/`](model-container-template/) | A runnable skeleton: copy it, fill in `inference.py`, `docker build`. |
| [`models.json`](models.json) | The community registry — one JSON entry per published container. |
| [`models.schema.json`](models.schema.json) | JSON Schema that CI validates `models.json` against. |

## The researcher journey

1. **Template** — copy `model-container-template/` into your own repo and
   fill in `inference.py` (load weights, generate, report progress). The
   HTTP layer is already written.
2. **Build** — `docker build` a self-contained image; pick a weights
   delivery mode (baked / bind-mount / download-at-start, CONTRACT.md §3.3).
3. **Test** — run `smoke_test.sh` against your container. It checks the
   contract end-to-end, including that failures are loud and `/health` is
   truthful.
4. **PR one JSON entry** — open a pull request against this repo that adds a
   single entry to `models.json` (name, description, repo_url, image,
   license, maintainer, contract_version). That entry is the *only* thing
   that lands here.
5. **CI validates** — `.github/workflows/validate-community.yml` checks your
   entry against the schema (blocking) and pings your repo_url (advisory).
6. **Merged** — your model appears in the community list and any AnimoFlow
   user can run it by pointing an endpoint env var at your container.

## No code review, no CLA

We do not review, host, or take ownership of your model code — it never
enters this repository, so there is nothing to review and nothing to sign.
The PR you send contains one JSON object; CI, not a human gatekeeper,
decides whether it is well-formed. Listing is not endorsement, and you can
remove or update your entry with another one-line PR at any time.

## License isolation: the HTTP boundary is the license boundary

AnimoFlow follows a strict *wrap, don't fork* rule for model code: research
models keep their own repos, licenses, and roadmaps, and AnimoFlow talks to
them only over HTTP. Your container is a separate work under **your**
license (and your upstream model's license) running in its own process and
image; AnimoFlow's repos contain only the contract, the template, and your
registry entry. Nothing crosses the process boundary except JSON and a
base64 NPZ — so GPL, research-only, or proprietary model code can
interoperate with AnimoFlow without license entanglement in either
direction. Each `models.json` entry declares its own `license` field, and
users decide what they are willing to pull and run.
