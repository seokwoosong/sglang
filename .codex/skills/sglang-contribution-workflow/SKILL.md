---
name: sglang-contribution-workflow
description: Implement, review, validate, commit, push, or prepare pull requests for the SGLang repository using its contribution guide and repository-local conventions. Use for SGLang contribution hygiene, test selection, commit organization, fork/upstream handling, CI preparation, or review-ready PR content; do not use as a generic Git workflow outside SGLang.
---

# SGLang Contribution Workflow

Follow the current checkout's instructions before relying on remembered commands. The official contribution guide and repository templates evolve; prefer their local versions when present, and consult the [published guide](https://docs.sglang.io/docs/developer_guide/contribution_guide) when the user asks for current upstream policy.

## Route the task

- For implementation, code review, test placement, formatting, documentation, accuracy, or performance validation, read [references/code-and-testing.md](references/code-and-testing.md).
- Before creating commits, rebasing, pushing, opening or updating a PR, handling CI, or replying to review comments, read [references/git-and-pr.md](references/git-and-pr.md).
- Read both references when delivering a change from implementation through PR preparation.

## Operating rules

1. Inspect `git status`, the current branch, remotes, applicable `AGENTS.md` files, and nearby tests before changing anything. Preserve unrelated user changes.
2. Do not develop or commit directly on `main`. Use a focused branch based on the intended upstream base.
3. Implement the smallest coherent change, add focused regression coverage for a concrete behavior or invariant, and validate in proportion to output, performance, hardware, and kernel risk.
4. Treat pre-commit hooks as mutating checks: inspect their edits, rerun until clean, and rerun affected tests when formatting changes executable code.
5. Stage exact files and inspect the staged diff. Organize commits as independently reviewable units; do not mix unrelated cleanup with functional changes.
6. A request to code, commit, rebase, or prepare a PR does not by itself authorize a remote push or PR mutation. Push or create/update the PR only when explicitly requested.
7. Before any push, resolve the exact branch and remote. Use the contributor's fork remote by default, never assume `upstream` is writable, and use `--force-with-lease` rather than `--force` after an authorized history rewrite.
8. Build PR content from `.github/pull_request_template.md`. Report exact commands and observed results; do not mark checklist items complete without evidence.

When a local rule conflicts with this skill, follow the repository rule. When the requested action would rewrite shared history, trigger remote CI, or mutate a PR, state the target and obtain any authority not already provided.
