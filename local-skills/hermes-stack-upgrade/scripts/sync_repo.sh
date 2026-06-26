#!/usr/bin/env bash
# Merge the latest upstream into a repo's mods branch (the agreed merge-based,
# non-force-push integration). Generic over both repos.
#
# Usage: sync_repo.sh <repo_dir> <mods_branch> <upstream_branch>
#   e.g. sync_repo.sh ~/.hermes/hermes-agent main main
#        sync_repo.sh ~/camofox hermes-mods master
#
# Exit 0 = clean merge (or already up to date).
# Exit 2 = merge conflicts left in the tree — the AGENT resolves them, then runs
#          `git -C <repo_dir> commit --no-edit` to complete the merge.
# Exit 1 = hard error (checkout/fetch failed).
set -uo pipefail

dir="${1:?repo_dir required}"
mods_branch="${2:?mods_branch required}"
upstream_branch="${3:?upstream_branch required}"

echo "── syncing $dir : merge upstream/$upstream_branch into $mods_branch"
git -C "$dir" checkout "$mods_branch" || { echo "✗ checkout $mods_branch failed"; exit 1; }

pre="$(git -C "$dir" rev-parse HEAD)"
echo "   pre-merge HEAD = $pre"

git -C "$dir" fetch upstream "$upstream_branch" || { echo "✗ fetch upstream failed"; exit 1; }

behind="$(git -C "$dir" rev-list --count "HEAD..upstream/$upstream_branch")"
echo "   $behind upstream commit(s) to integrate"
if [ "$behind" -eq 0 ]; then
  echo "✓ already up to date with upstream/$upstream_branch"
  exit 0
fi

if git -C "$dir" merge --no-edit "upstream/$upstream_branch"; then
  echo "✓ clean merge of $behind commit(s)"
  exit 0
fi

# Merge stopped with conflicts.
echo "⚠ merge conflicts — files needing resolution:"
git -C "$dir" diff --name-only --diff-filter=U | sed 's/^/   /'
echo
echo "AGENT: resolve the conflicts in the files above (preserve BOTH the upstream"
echo "changes and the local mods), then run:  git -C $dir add -A && git -C $dir commit --no-edit"
echo "To bail out instead:  git -C $dir merge --abort   (restores $pre)"
exit 2
