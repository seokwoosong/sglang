# SGLang Git and Pull Request Workflow

Use this reference before committing, rebasing, pushing, opening or updating a PR, triggering CI, or responding to review.

## Repository and branch safety

SGLang's contributor model normally uses:

- `origin`: the contributor's fork, used for branch pushes
- `upstream`: `sgl-project/sglang`, used to fetch the official base

Do not trust these names without checking `git remote -v`. Confirm the current branch and worktree before any history operation. Never commit directly to `main`, and do not push to the official repository unless the user explicitly names that target and has permission.

Before rebasing, inspect divergence and local changes. Rebase only when requested or when the user has authorized updating the branch history. After a rebase of an already-pushed PR branch, use:

```bash
git push --force-with-lease origin <branch>
```

Never replace it with raw `--force`. A normal new branch can use:

```bash
git push -u origin <branch>
```

Push only after explicit authorization. Recheck the exact remote and ref immediately before executing it.

## Commit preparation

1. Inspect `git status`, unstaged diff, staged diff, and recent history for the affected area.
2. Stage exact intended paths rather than sweeping unrelated work into the commit.
3. Ensure each commit has one coherent purpose and its tests. Separate functional behavior, regression tests, and unrelated cleanup when they are independently reviewable; combine a test with its fix when separating them would leave a misleading or broken intermediate commit.
4. Do not rewrite, amend, squash, or reorder user commits unless requested.
5. Verify the staged diff and run the checks appropriate to that commit before creating it.

SGLang does not enforce one universal commit-title grammar. Match recent history in the affected subsystem. Prefer a concise imperative title with a useful area or type when that improves discovery, for example:

```text
fix(mem-cache): preserve schedulable allocation capacity
perf(unified-memory): batch compaction lookups
[diffusion] Add ...
```

Do not fabricate a conventional-commit requirement. Avoid vague titles such as `fix bug`, and keep benchmark or correctness claims out of the title unless directly established.

## Push handoff

Before pushing:

- Confirm the branch is not `main`.
- Confirm `origin` points to the intended fork and the destination branch is exact.
- Ensure the worktree and commit list match what the user expects.
- Run the required formatting and tests, or clearly report what could not be run.
- Check that no secrets, large accidental artifacts, local benchmark outputs, or unrelated changes are included.
- Explain when history was rewritten and a force-with-lease push is required.

Do not create tags, releases, CI comments, or additional remote branches as part of an ordinary push.

## Pull request content

Use the current `.github/pull_request_template.md`; do not replace its sections with a generic template.

- **Motivation:** State the user-visible or internal problem, why the existing behavior is insufficient, and the affected scope. Include a minimal failure scenario when useful.
- **Modifications:** Explain behavior and contracts, not merely a list of filenames. Call out compatibility boundaries and intentionally unchanged paths.
- **Accuracy Tests:** If output may change, give exact commands, model/configuration, results, and parity or expected differences. If it cannot affect model math, explain the narrower correctness risk and the regression coverage used instead.
- **Speed Tests and Profiling:** If performance may change, give the reproducible environment, baseline/proposed revisions, workload, repetitions, statistic, and results. If not applicable, say why rather than leaving the section ambiguous.
- **Checklist:** Check only verified items. For documentation, accuracy, or speed items that are not applicable, provide a concise rationale.

Also include exact focused test and pre-commit results. Keep raw logs out of the body unless needed; summarize them and attach artifacts or commands that make the result reproducible.

Use `.github/CODEOWNERS` and `.github/MAINTAINER.md` to identify the current Codeowners, Merge Oncall, and merge process. The normal path is a completed checklist, required approvals, green required CI, and then merge by someone with permission.

## CI and review updates

Read the current contribution guide before issuing CI commands because permissions and command behavior can change. At the time this skill was created, notable behavior included:

- A `run-ci` label is needed for PR CI.
- `/tag-run-ci-label` affects future commits but does not run the current commit.
- `/tag-and-rerun-ci` labels and runs the current commit.
- PR authors can use `/rerun-failed-ci` on their own PRs.
- Selective `/rerun-test` and `/rerun-group` have stricter permissions and do not install a PR-local AOT kernel wheel.

Do not post CI-triggering comments unless explicitly requested. Respect cooldowns and avoid repeated comments; the guide documents editing an existing command comment as an alternative.

When addressing review comments:

- Analyze whether each comment is functional, contract/API, coverage, performance, or style.
- Implement independent findings in clear commits when requested.
- Reply in the reviewer's terminology: state what changed, the semantic reason, the affected paths, and the tests added or run.
- Do not say a comment is addressed on the PR until the corresponding commit is actually pushed and visible remotely. Before that, describe it as prepared locally.

## Current authority map

Prefer the current checkout, with the published guide as a freshness check:

- `docs/docs/developer_guide/contribution_guide.mdx`
- `.github/pull_request_template.md`
- `.github/MAINTAINER.md`
- `.github/CODEOWNERS`
- `test/README.md`
- `test/registered/README.md`
- `test/registered/unit/README.md`
- `docs/README.md`
- [Published contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide)
