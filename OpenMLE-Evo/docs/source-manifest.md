# Release manifest

The public source identity of this module is the repository commit or release tag containing this file. Internal worktree names, experiment branches, and deployment revisions are not part of the public interface.

## Curated runtime scope

- `scripts/evaluate_airaevo.py`
- Required `tts_search` runtime modules and Hydra configuration groups
- AIRA-Dojo `src/` runtime
- AIRA-Evo MLE-Bench task builder, runner, task adapter, and single-task runner
- NatureBench task builder/adapter, scoring path, operator prompts, Lite-v2 task manifest, configs, and ten visible-data analysis addenda
- Standard/multi-GPU profile tests and async scheduler tests
- NatureBench integration tests and benchmark-specific runbooks
- Public operations and validation documentation

Historical experiment launchers, unrelated search algorithms, generated task configs, private datasets, leaderboard files, result outputs, virtual environments, caches, model weights, and credentials are intentionally excluded.

## Public-release adjustments

The runtime logic is preserved while deployment-specific values are normalized:

1. data, leaderboard, submit-root, sandbox, router, model, and key values come from environment variables;
2. the final package root is `third_party/aira-evo`, not the obsolete `third_party/aira-dojo` path;
3. launchers put this release and its vendored AIRA-Evo `src/` first on `PYTHONPATH`;
4. no default sandbox key or internal network endpoint is embedded;
5. NatureBench SCM hosts, workspaces, task roots, evaluator endpoints, Docker image, and GPU pools are deployment environment variables;
6. MLE-Bench and NatureBench remain parallel adapters over one runtime instead of duplicating AIRA-Evo;
7. standard and multi-GPU modes remain explicit Hydra execution profiles.
