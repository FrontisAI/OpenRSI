# OpenRSI TODO Tracking Report

During the code review, multiple `TODO` and `FIXME` comments were identified across the codebase. These represent technical debt and potential future improvements. 

We recommend converting these into GitHub Issues to track them systematically.

## High Priority TODOs (Functional & Refactoring)

### SLIME Framework (`OpenMLE-ERL/SFT/slime/`)
1. **Rollout Synchronization**:
   - `slime/ray/rollout.py:210`: `TODO: remove once health_monitor operates per-group.`
   - `slime/ray/rollout.py:839`: `TODO: remove once all consumers read from RolloutServer directly.`
   - `slime/router/router.py:114`: `TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync`

2. **Megatron Integration**:
   - `slime/backends/megatron_utils/update_weight/common.py:38`: `TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?`
   - `slime/backends/megatron_utils/update_weight/common.py:152`: `TODO shall we handle (almost) all buffers like Megatron Bridge`
   - `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py:215`: `TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.`

3. **PPO & Training**:
   - `slime/utils/ppo_utils.py:152`: `TODO: when megatron is not installed, fall back to naive implementation`
   - `slime/ray/train_actor.py:45`: `TODO: currently this doesn't work as ray has already set torch.cuda.device_count().`

## Medium Priority TODOs (Optimization & Cleanup)

1. **Data Processing**:
   - `slime/utils/data.py:155`: `TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.`
   - `slime/rollout/data_source.py:49`: `TODO may further refactor data-loading part later`

2. **Configuration & Arguments**:
   - `slime/utils/eval_config.py:9`: `TODO: This is ugly, temporarily leave this. We should unify all the config name for dataset, default, and args.`
   - `slime/backends/sglang_utils/arguments.py:7`: `TODO: use all sglang router arguments with --sglang-router prefix`

## Low Priority TODOs (Minor Improvements)

- `slime/utils/logging_utils.py:32`: `TODO further refactor, e.g. put TensorBoard init to the "init" part`
- `slime/utils/processing_utils.py:56`: `TODO: temporary solution, will write image utils for slime later`
- `slime/rollout/generate_hub/__init__.py:1`: `TODO: maybe move sglang_rollout::generate to this folder`

## Action Items
- [ ] Create GitHub Issues for the High Priority items.
- [ ] Assign owners to the Megatron integration and PPO fallback tasks.
- [ ] Schedule a refactoring sprint for the `slime/rollout/` module.
