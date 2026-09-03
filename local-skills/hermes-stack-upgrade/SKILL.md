---
name: hermes-stack-upgrade
description: Safely upgrade the Hermes agent while preserving local mods (fork workflow), with conflict-aware merges, an end-to-end smoke gate, and one-command rollback.
version: 2.2.0
author: Koitrin KOFFI
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [Maintenance, Upgrade, Git, Fork, Browser]
    related_skills: [github-auth]
---

# Hermes Stack Upgrade

Upgrades the Hermes agent (`~/.hermes/hermes-agent`) without losing your local
modifications, using the **fork workflow**: `origin` = your GitHub fork
(carries your mods), `upstream` = the official repo. Your mods live on the
**`hermes-mods`** branch; `main` stays a clean upstream mirror. Normal
operation runs checked out on `hermes-mods`.

The browser backend (agent-browser CLI + Helium) is an npm dependency pinned
in `package.json`, not a separate coupled repo — it upgrades via the normal
`pip install -e` step below, not a second merge. **Camofox is not part of this
machine's stack** (disabled 2026-07-17, replaced by agent-browser + a
persistent Helium profile) — this skill no longer touches `~/camofox` or the
`camofox.service` unit. If Camofox is ever re-enabled, this skill would need
its coupled-repo handling re-added; until then, do not resurrect it here.

## When to Use

- The user asks to **update / upgrade Hermes**.
- After seeing "update available" — instead of bare `hermes update`, which would
  switch HEAD to the mods-free `main` and **deactivate your mods**.

Do **not** use bare `hermes update` on this machine: it checks out `main` (now a
clean upstream mirror with none of your mods) and drops your customizations from
the working tree. This skill is the supported path here.

## Why this exists (the trap)

`hermes update` checks out `main` and fast-forwards/`reset --hard`s it to
`origin/main`. Your mods live on **`hermes-mods`**, not `main`, so this switches
the working tree to the mods-free `main` and **deactivates every mod** until you
`git checkout hermes-mods` again. Bundled upstream fork-sync also **deliberately
skips** when your fork is ahead (`"Skipping upstream sync to preserve your
changes"`), so the upstream → `hermes-mods` merge is *yours* to drive. This skill
drives it and verifies the result.

## Procedure (recurring upgrade)

Scripts live in `${HERMES_SKILL_DIR}/scripts`. Run them in order. Stop and think
at any ✗ or exit code 2.

1. **Preflight** — verify fork remotes, record the pre-merge SHA for rollback:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/preflight.sh
   ```
   If it fails, the one-time setup below hasn't been done (or `origin` still
   points at the official repo). Fix that first.

2. **Dry-run the merge in a throwaway worktree FIRST.** Do not resolve
   conflicts in the live checkout. A worktree costs one command, keeps the
   running gateway on known-good code while you work, and lets you run the
   test suites against the resolution *before* the real repo ever moves:
   ```bash
   git -C ~/.hermes/hermes-agent worktree add --detach /tmp/mergedry hermes-mods
   cd /tmp/mergedry && git merge upstream/main --no-commit
   ```
   Resolve there, verify (below), then apply. When the real merge runs,
   confirm `upstream/main` is still the SHA you validated and prove the
   applied resolution is the validated one:
   ```bash
   git -C <real> write-tree   # must equal   git -C /tmp/mergedry write-tree
   ```
   If upstream moved in the meantime, re-check only the files its new commits
   touch. Clean up with `git worktree remove` when done.

   **Verify before applying**, in this order — each of these caught a real
   defect on 2026-09-03 that reading the conflict hunks did not:
   - every file compiles (`python -m py_compile`);
   - every local mod is still present (grep for a signature string per mod,
     not a diffstat — a mod can survive as dead code);
   - no orphaned references (a function upstream deleted whose callers you
     kept, or a parameter you dropped whose body still uses it);
   - the targeted test suites pass.

3. **Merge upstream into hermes-mods:**
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/sync_repo.sh "$HOME/.hermes/hermes-agent" hermes-mods main
   ```
   - **Exit 0** → clean merge, continue.
   - **Exit 2** → conflicts. **YOU (the agent) resolve them**: open each listed
     file, keep BOTH the upstream change and the local mod. Then complete the
     merge:
     ```bash
     git -C "$HOME/.hermes/hermes-agent" add -A && git -C "$HOME/.hermes/hermes-agent" commit --no-edit
     ```
     Re-read the diff against `upstream/main` to confirm the mod survived.
     If the file diverged too far to merge cleanly, prefer **re-applying** the
     mod against current upstream over forcing a messy merge.

4. **Reinstall Hermes deps** for the newly merged code (the merge may have
   changed dependencies, including the pinned `agent-browser` npm version —
   see the Pitfalls note on that). The venv is `uv`-managed (no `pip` binary
   inside it — confirmed live 2026-07-19, `python -m pip` fails with
   `No module named pip`):
   ```bash
   cd "$HOME/.hermes/hermes-agent" && uv pip install -e ".[all]" --python venv/bin/python
   ```

5. **Hand off the rest to a detached finisher, then STOP.** Everything from here
   — restart, smoke gate, deploy-or-rollback — is deterministic shell that does
   **not** need you (the agent). Crucially, the restart in step 1 of the finisher
   restarts `hermes-gateway`, **the unit you are running inside** (kanban is
   dispatched in-gateway). That unit is `KillMode=mixed`, so on restart the whole
   cgroup is SIGKILLed: if you ran these steps inline you would be killed at the
   restart and the smoke gate + deploy/rollback would **never run** (the classic
   "it restarted but did nothing after" bug). So launch them as their **own
   transient unit**, which lives in a separate cgroup and survives the restart:
   ```bash
   systemd-run --user --collect --unit="hermes-upgrade-finish-$(date +%s)" \
     bash ${HERMES_SKILL_DIR}/scripts/finish_upgrade.sh
   ```
   `finish_upgrade.sh` then runs on its own: restart services → wait until
   `hermes-gateway` is active → **smoke gate** → **green**: `push_backups.sh` +
   `deploy_self.sh` / **red**: `rollback.sh` (fork untouched) → **Telegram ping**
   (`hermes send`) with the verdict. It first self-copies to `/tmp` so
   `deploy_self.sh` replacing the live skill dir can't pull the rug from under it.

   **After launching it, you are done.** Tell the user the upgrade is finishing in
   the background and they'll get a Telegram ping with the result (🟢 deployed /
   🔴 rolled back). **Do NOT** run `restart_services.sh` yourself. The verdict also
   lands in `~/.hermes/state/stack-upgrade-last-result.txt` (full log:
   `stack-upgrade-finish.log`).

   Whether you survive the restart depends on your cgroup, so check instead of
   assuming: `cat /proc/self/cgroup`. Inside `hermes-gateway.service` (the
   normal case — kanban is dispatched in-gateway) you are SIGKILLed at the
   restart, so launching the finisher is the last thing you do. A caller
   outside it (e.g. a Claude Code terminal in `session-NN.scope`, as on
   2026-09-03) survives and can read the verdict back to the user. Launch the
   finisher as its own transient unit either way — that is what makes the
   sequence safe regardless of who called it.

## One-Time Setup (already done on this machine; documented for a fresh rebuild)

1. Install + auth GitHub CLI: `gh auth login` (HTTPS).
2. Fork the repo and rewire remotes:
   ```bash
   gh repo fork NousResearch/hermes-agent --clone=false --remote=false
   cd ~/.hermes/hermes-agent
   git remote rename origin upstream
   git remote add origin https://github.com/<you>/hermes-agent.git
   ```
3. Put your mods on the `hermes-mods` branch and push to your fork. Stay
   checked out on `hermes-mods` for normal operation.
4. Deploy this skill: `bash <canonical>/scripts/deploy_self.sh`.

## Pitfalls

- **Never `git push --force`** to the fork — it is your only off-machine
  backup. The merge workflow never needs it.
- **Push backups only after the smoke gate is green** (`push_backups.sh` is step
  6a) so the fork never points at a broken state.
- **A merge can change behaviour through a config DEFAULT, with no conflict
  anywhere.** Upstream added `browser.backend` (default `""`), which resolves
  to **Browser Use mode** whenever the browser-use CLI *or* `uvx` is runnable —
  and `uvx` is installed on this machine. An absent key is not a safe key. This
  install must carry `browser.backend: 'off'` in `~/.hermes/config.yaml` or the
  whole agent-browser + detached-Helium stack is silently bypassed (not broken:
  bypassed, which is worse to diagnose). Quote it — YAML 1.1 parses a bare
  `off` as `False`; upstream handles both, but do not rely on that.
  **Before every restart, diff upstream's `hermes_cli/config_defaults.py`
  against the merge base for new keys whose default is not the behaviour you
  have**, and verify the resolved value against the MERGED code, not by
  reading the YAML:
  ```bash
  <repo>/venv/bin/python -c "from tools.browser_use_cli import is_browser_use_cli_mode as f; print(f())"
  ```
- **`browser_tool.py` is hot upstream AND heavily locally modified** (persistent
  profile, blank-tab steer, turn-boundary/close() exemptions, browser_read/
  browser_wait removed) — expect conflicts there on nearly every upgrade;
  that's the agent-resolution step, not a failure. Read the full local diff
  against `upstream/main` before resolving so you don't accidentally re-add a
  removed tool or drop an exemption.
- **agent-browser is a LOCAL dependency now, and its version is pinned for a
  reason.** Upstream removed it from the root npm deps on 2026-08-13
  (`5f5f8d5b62`) and resolves the CLI from PATH / on demand instead; the pin in
  `package.json` (`0.35.1`, exact — as of 2026-09-03) is ours, so **both
  `package.json` hunks resolve to OURS** and the `package-lock.json` entry has
  to be carried across too. The binary MUST match the pin: the CLI's JSON
  output schema is what `browser_tool.py` parses, and 0.32.1 added a
  `lifecycle` object that broke it. Do **not** bump it during a routine merge
  without re-validating the mods against the new version.
- **Do not regenerate `package-lock.json` with `npm install`.** The repo
  `.npmrc` carries `min-release-age=14` (upstream supply-chain policy since
  `43f18cca81`), so npm refuses to fetch any release younger than two weeks —
  including the agent-browser pin, if it is recent. `npm install
  --package-lock-only` then dies with `ETARGET` and tells you nothing useful.
  Merge the lock surgically instead: take upstream's, re-add the root
  `dependencies` block and the `node_modules/agent-browser` entry from ours
  (it has no transitive deps), then prove it with
  `npm ls --package-lock-only --workspaces=false --depth=0`, which works
  offline. Nothing in the upgrade flow runs `npm install`, so the existing
  `node_modules` is untouched either way.
- **RESOLVED conflict (merged 2026-07-19, commit `a76d9afff`, 1141 upstream
  commits integrated)**: upstream `29899c2aa` ("Fix headed browser sessions
  being killed after every turn") independently fixed the same turn-boundary
  browser-kill bug our `is_persistent_browser_session` exemption fixes, at the
  exact two spots predicted below. Both conflicts resolved by OR'ing the two
  conditions together (skip `cleanup_browser` if persistent-profile OR headed)
  rather than letting one clobber the other — see `cleanup_task_resources` in
  `agent/chat_completion_helpers.py` and the local-mode branch of
  `_run_browser_command` in `tools/browser_tool.py`. `run_agent.py::Agent.close()`
  merged clean as predicted, no upstream equivalent existed there. Kept as a
  worked example: **when the next upgrade shows conflicts in these same two
  functions again, this is the pattern to reapply** — read both sides, keep
  every distinct exemption condition, OR them, don't let a `git merge` autopick
  silently drop one.
- **WORKED EXAMPLES (merged 2026-09-03, commit `0673b1ec9e`, 6888 upstream
  commits, 0.20.0 → 0.21.0).** 31 hunks / 11 files. `browser_tool.py` had ONE
  conflict despite 43 upstream commits — volume of upstream churn does not
  predict conflict count. Four resolutions that a naive "take one side" gets
  wrong, all four found by the dry-run in step 2 and none visible in the hunks:
  - `hermes_cli/web_server.py` — upstream replaced `extra_allowed_hosts` with a
    `dashboard.public_url`-derived `trusted_public_hosts`. Two traps. (a) Taking
    upstream on all 5 hunks left the auto-merged body of `_is_accepted_host`
    referencing an `extra_allowed` parameter that no longer existed →
    `NameError`. (b) Folding our host INTO `trusted_public_hosts` looks like the
    tidy fix, but that set also drives `should_require_dashboard_auth`, so a
    non-loopback entry **arms the auth gate** — on an install with neither
    `basic_auth` nor OAuth configured, that locks the dashboard. Correct
    resolution: upstream's mechanism, our key re-applied as a SEPARATE 4th
    parameter that only affects Host-header acceptance.
  - `toolsets.py` — upstream renamed `process` → `process_manage` and `cronjob`
    → `cronjob_manage`. `_LEGACY_TOOL_ALIASES` in `model_tools.py` maps old
    names **at dispatch only**, not for `_HERMES_CORE_TOOLS` membership, so
    keeping the old names would have silently deferred both tools behind
    `tool_search`. After any upstream tool rename, re-pin the canonical name.
  - `tools/delegate_tool.py` — upstream dropped `goal`/`context`/`role`/
    `background` from the static `DELEGATE_TASK_SCHEMA` (still accepted by the
    handler, just unadvertised), and the auto-merge took our `model`/
    `agent_type` out with them. Our `get_definitions()` then rewrote
    `properties["role"]["description"]` on a key that no longer existed →
    `KeyError` at boot. Re-added the two local params; the rewrites are now
    behind membership tests so the next schema churn degrades instead of
    crashing. All 8 hunks were otherwise additive on both sides — union.
  - `agent/system_prompt.py` — upstream restructured four blocks that the
    `_prompt_override` mod wraps (help_guidance became a deferred slot, memory
    split in two, the gpt/codex/grok block became a model-agnostic
    execution-discipline gate). Pattern: **take upstream's structure, re-apply
    the override on top**, never the reverse. The `openai_guidance` override
    NAME was kept deliberately even though the block widened, so the existing
    empty `~/.hermes/prompts/openai_guidance.md` keeps dropping it.
  - `tools/browser_camofox.py` — Camofox is disabled on this machine, so
    deleting the local `camofox_*` helpers looks free. It is not:
    `browser_tool.py` still imports four of them. Grep for callers before
    dropping anything that "isn't used any more".
- **A failing test after the merge is not automatically a bad resolution.**
  Triage in this order: (1) missing NEW upstream dependency — step 4 has not
  run yet, or the dep is new this release (`snowballstemmer==3.1.1` accounted
  for 45 of 48 failures on 2026-09-03); (2) a local fixture made obsolete by
  new upstream behaviour (a batch-goal minimum length now rejects one-letter
  test goals) — fix the fixture; (3) a PRE-EXISTING failure — check it against
  the pre-merge branch before blaming the merge; (4) an actual regression.
- **Never let a long test sweep straddle step 4.** `uv pip install` swaps
  packages under a running pytest (mcp 1.28 → 2.0 mid-run on 2026-09-03),
  and its results become uninterpretable. Finish the sweep, or kill it.

## Verification

- The finisher (step 5) runs async and reports the verdict on **Telegram**; it is
  also written to `~/.hermes/state/stack-upgrade-last-result.txt`, with the full
  transcript in `~/.hermes/state/stack-upgrade-finish.log`. Check it is still
  running with `systemctl --user list-units 'hermes-upgrade-finish-*'` (it
  self-removes on completion via `--collect`).
- `smoke.sh` exits 0 (both layers pass).
- `git -C ~/.hermes/hermes-agent log --oneline -1` shows your mod commit reachable
  from `hermes-mods`.
- `gh repo view <you>/hermes-agent` reflects the new push timestamp.
