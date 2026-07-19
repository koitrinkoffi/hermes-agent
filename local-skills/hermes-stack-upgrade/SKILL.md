---
name: hermes-stack-upgrade
description: Safely upgrade the Hermes agent while preserving local mods (fork workflow), with conflict-aware merges, an end-to-end smoke gate, and one-command rollback.
version: 2.0.0
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

2. **Merge upstream into hermes-mods:**
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

3. **Reinstall Hermes deps** for the newly merged code (the merge may have
   changed dependencies, including the pinned `agent-browser` npm version —
   see the Pitfalls note on that). The venv is `uv`-managed (no `pip` binary
   inside it — confirmed live 2026-07-19, `python -m pip` fails with
   `No module named pip`):
   ```bash
   cd "$HOME/.hermes/hermes-agent" && uv pip install -e ".[all]" --python venv/bin/python
   ```

4. **Restart services** so the new code is live:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/restart_services.sh
   ```

5. **Smoke gate** — must be green to proceed:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/smoke.sh
   ```

6. **On green → back up + redeploy this skill:**
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/push_backups.sh
   bash ${HERMES_SKILL_DIR}/scripts/deploy_self.sh
   ```
   **On red → roll back** (fork untouched), then investigate:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/rollback.sh
   ```

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
- **`browser_tool.py` is hot upstream AND heavily locally modified** (persistent
  profile, blank-tab steer, turn-boundary/close() exemptions, browser_read/
  browser_wait removed) — expect conflicts there on nearly every upgrade;
  that's the agent-resolution step, not a failure. Read the full local diff
  against `upstream/main` before resolving so you don't accidentally re-add a
  removed tool or drop an exemption.
- **agent-browser version is pinned for a reason** (`package.json` `^0.26.0`,
  and the resolved binary MUST match — see `project-hermes-agent-browser-migration`
  memory): 0.32.1 changes the CLI's JSON output schema (adds a `lifecycle`
  object) which breaks `browser_tool.py`'s parsing, and both 0.26.0 and 0.32.1
  share the auto-activate-newest-tab bug that the blank-tab steer mod works
  around. Do **not** bump this dependency during a routine upgrade merge
  without deliberately re-validating the local mods against the new version.
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

## Verification

- `smoke.sh` exits 0 (both layers pass).
- `git -C ~/.hermes/hermes-agent log --oneline -1` shows your mod commit reachable
  from `hermes-mods`.
- `gh repo view <you>/hermes-agent` reflects the new push timestamp.
