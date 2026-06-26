---
name: hermes-stack-upgrade
description: Safely upgrade the Hermes agent and the camofox browser server while preserving local mods (fork workflow), with conflict-aware merges, an end-to-end smoke gate, and one-command rollback.
version: 1.0.0
author: Koitrin KOFFI
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [Maintenance, Upgrade, Git, Fork, Browser, Camofox]
    related_skills: [github-auth]
---

# Hermes Stack Upgrade

Upgrades two coupled repos at once — the Hermes agent (`~/.hermes/hermes-agent`)
and the camofox browser server (`~/camofox`) — without losing your local
modifications, using the **fork workflow**: `origin` = your GitHub fork (carries
your mods), `upstream` = the official repo. Your mods live on `main` (hermes) and
`hermes-mods` (camofox).

## When to Use

- The user asks to **update / upgrade Hermes** (or "update the browser stack").
- After seeing "update available" — instead of bare `hermes update`, which would
  switch HEAD to `main` and **deactivate your mods**.

Do **not** use bare `hermes update` on this machine: it targets official `main`
and drops your customizations. This skill is the supported path here.

## Why this exists (the trap)

`hermes update` checks out `main` and fast-forwards/`reset --hard`s it to
`origin/main`. Bundled upstream fork-sync **deliberately skips** when your fork
is ahead (`"Skipping upstream sync to preserve your changes"`), so the upstream
merge is *yours* to drive. This skill drives it and verifies the result.

## Procedure (recurring upgrade)

Scripts live in `${HERMES_SKILL_DIR}/scripts`. Run them in order. Stop and think
at any ✗ or exit code 2.

1. **Preflight** — verify fork remotes, record pre-merge SHAs for rollback:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/preflight.sh
   ```
   If it fails, the one-time setup below hasn't been done (or `origin` still
   points at the official repo). Fix that first.

2. **Merge upstream into each repo.** Hermes first, then camofox:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/sync_repo.sh "$HOME/.hermes/hermes-agent" main main
   bash ${HERMES_SKILL_DIR}/scripts/sync_repo.sh "$HOME/camofox" hermes-mods master
   ```
   - **Exit 0** → clean merge, continue.
   - **Exit 2** → conflicts. **YOU (the agent) resolve them**: open each listed
     file, keep BOTH the upstream change and the local mod (the upload endpoint /
     client, the desktop-fingerprint mod). Then complete the merge:
     ```bash
     git -C <repo> add -A && git -C <repo> commit --no-edit
     ```
     Re-read the diff against `upstream/<branch>` to confirm the mod survived.
     If the file diverged too far to merge cleanly, prefer **re-applying** the
     mod against current upstream over forcing a messy merge.

3. **Reinstall Hermes deps** for the newly merged code (the merge may have
   changed dependencies):
   ```bash
   "$HOME/.hermes/hermes-agent/venv/bin/python" -m pip install -e "$HOME/.hermes/hermes-agent[all]"
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
   **On red → roll back** (forks untouched), then investigate:
   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/rollback.sh
   ```

## One-Time Setup (already done on this machine; documented for a fresh rebuild)

1. Install + auth GitHub CLI: `gh auth login` (HTTPS).
2. Fork both repos and rewire remotes:
   ```bash
   gh repo fork NousResearch/hermes-agent --clone=false --remote=false
   cd ~/.hermes/hermes-agent
   git remote rename origin upstream
   git remote add origin https://github.com/<you>/hermes-agent.git

   gh repo fork jo-inc/camofox-browser --clone=false --remote=false
   cd ~/camofox
   git remote rename origin upstream
   git remote add origin https://github.com/<you>/camofox-browser.git
   ```
3. Put your mods on the upgrade branches (`main` for hermes, `hermes-mods` for
   camofox) and push to your forks.
4. Deploy this skill: `bash <canonical>/scripts/deploy_self.sh`.

## Pitfalls

- **Never `git push --force`** to the forks — they are your only off-machine
  backup. The merge workflow never needs it.
- **Push backups only after the smoke gate is green** (`push_backups.sh` is step
  6a) so the fork never points at a broken state.
- **`browser_tool.py` / `server.js` are hot upstream** — expect conflicts there
  every few upgrades; that's the agent-resolution step, not a failure.
- **Don't run bare `hermes update`** between upgrades — it deactivates your mods.
- The live smoke needs the **running systemd camofox**; `restart_services.sh`
  must succeed (health check) before `smoke.sh`.

## Verification

- `smoke.sh` exits 0 (all three layers pass).
- `git -C ~/.hermes/hermes-agent log --oneline -1` shows your mod commit reachable
  from `main`; `git -C ~/camofox log --oneline -1` likewise on `hermes-mods`.
- `gh repo view <you>/hermes-agent` and `<you>/camofox-browser` reflect the new
  push timestamps.
