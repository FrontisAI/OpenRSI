# Release validation

## Static validation

- Hydra standard profile resolves to `generation` with one worker.
- Hydra multi-GPU profile resolves to `async_steady_state` with the requested worker count and router endpoint.
- NatureBench Lite-v2 composes as a ten-task SCM evaluation with both standard and async steady-state search profiles.
- NatureBench task building, score selection, evaluator registration, GPU pool, prompt, resume, and summary behavior are covered by integration tests.
- Runtime imports resolve from the current release rather than another editable worktree.
- Public runtime/config scans contain no internal data paths, sandbox IPs, or default credentials.

## GPU smoke validation

Validation was performed on 2026-07-28 against the release tree described in [`source-manifest.md`](source-manifest.md).

### Standard

- one search worker and one sandbox concurrency slot;
- model completion returned HTTP 200;
- generated Python solution executed in the GPU sandbox;
- submission format validation passed;
- final sandbox score was non-null;
- process exited with code 0 and left no evaluation process behind.

### Async multi-GPU

- `execution_mode=async_steady_state`;
- two requested and two resolved async workers;
- two model completions were issued concurrently;
- two sandbox jobs ran concurrently on distinct router workers;
- both nodes completed with `status=success`;
- out-of-order completion preserved each node's attempt and worker metadata;
- final best-node submission completed;
- process exited with code 0 and left no evaluation process behind.

## External validation boundary

The MLE-Bench smoke establishes that the entrypoint, generation, concurrency, router, sandbox, scoring, checkpoint, output, and final-submit paths work together. NatureBench static/unit validation establishes configuration and adapter behavior; a full NatureBench score additionally requires its external task packages, eval service, SCM host, and container image. Neither validation independently reproduces paper tables.
