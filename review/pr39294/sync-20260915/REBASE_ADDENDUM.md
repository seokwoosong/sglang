# Rebase addendum — latest user direction

User explicitly requests: use review branch as working baseline, combine latest PR36729 and upstream main, and rebase.

Refreshed heads: PR36729 a6eb82dc77353249fff1b08356533c2dc16ea6ec; main 832ec39cc0324cb0e7823dc8385e27a30c356bdd. Review HEAD remains 68a70f6cc8ca174dd75606723424b210cb664ed3.

Supersedes merge-only steps of SYNC_PLAN.md:
1. Preserve pre-rebase review HEAD under a local backup branch.
2. In an isolated worktree create integration base at latest PR36729; merge latest main into that base (no-ff if needed). This preserves both actual upstream histories, without replaying the other author's full merged stack.
3. Rebase ONLY our four follow-up commits (6c8bbdf610..68a70f6cc8) onto that combined base. This includes three code commits and review-document bundle. Resolve conflicts according to existing approved design; no published PR rewrite.
4. Validate as SYNC_PLAN.md, inspect range-diff for the four rebased commits, and record both baseline parents and result SHA.
5. After final review, move the clean local review/pr39294-unified-joint-allocation branch/worktree to the tested rebased result. Keep the backup and remote branch at previous SHA; no force-push requested in this turn. Future work uses review branch, not old PR branch. Preserve unrelated root worktree.

Please approve this exact bounded rebase and validation plan. Write REBASE_PLAN_REVIEW.md in this directory. Approval has not yet been received from the earlier queued request. No implementation or experiment by reviewer, only read-only review and decision artifact.
