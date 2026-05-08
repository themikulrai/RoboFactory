# Plans (workflow improvement #10)

Claude plan files that started life under `~/.claude/plans/` but live here
so revisions show up in `git log` / PR review.

The active plan files in this directory are referenced by symlinks back
in `~/.claude/plans/` so the Claude harness can still find them at the
expected path. Editing either path edits the same file (it's one inode);
`git status` from the repo root surfaces the change.

## What's tracked here

- `the-complete-openpi-robofactory-pure-lamport.md` — the master cleanup
  plan covering Phase A + B + C1 + C2 + workflow improvements. ~870 lines.

## How to migrate a future plan

When a new robofactory-relevant plan lands under `~/.claude/plans/`,
move it here and re-symlink:

```bash
bash docs/plans/migrate.sh <plan_filename>
```

This is reversible: removing the symlink and copying back to the
original path restores the pre-migration state.

## What's NOT tracked here

The 40+ unrelated plan files under `~/.claude/plans/` (other tasks,
side-projects, debug sessions). Migration is opt-in per plan to keep
this directory focused on the openpi-robofactory effort.
