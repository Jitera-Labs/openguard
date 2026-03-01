---
name: release
description: >
  End-to-end release process for the openguard project: version bump in
  pyproject.toml, pre-release checks, git tag + push (which automatically
  triggers PyPI publish and Docker image push to GHCR via GitHub Actions),
  GitHub Release creation, and Cloudflare Pages docs deploy.
  Use this skill whenever the user says anything like "cut a release",
  "release a new version", "publish v1.2.3", "tag a release", "ship it",
  "bump the version and release", or asks about the release workflow.
---

# Release Skill

## What this skill does

A complete release of openguard involves these moving parts:

| Step | Where it runs |
|---|---|
| Version bump in `pyproject.toml` | Locally |
| `make check-release` (lint + type-check + unit tests + clean tree) | Locally |
| `git commit` + `git tag vX.Y.Z` + `git push origin vX.Y.Z` | Locally → triggers CI |
| PyPI publish (`uv build` + `uv publish`) | GitHub Actions (`.github/workflows/pypi-publish.yaml`) |
| Docker image pushed to `ghcr.io` | GitHub Actions (`.github/workflows/docker-publish.yaml`) |
| GitHub Release created with auto-generated notes | Locally via `gh release create` |
| Cloudflare Pages docs deploy | Locally via `make cf-deploy` |

The GitHub Actions workflows fire automatically once the tag is pushed — you don't need to invoke them manually.

---

## Step 1 — Determine the new version

Read the current version from `pyproject.toml`:

```bash
grep '^version' pyproject.toml
```

Default behaviour: **patch bump** (e.g. `0.1.1` → `0.1.2`).

If the user specified `minor`, `major`, or an explicit version string (like `1.0.0`), use that instead.

Semver rules:
- `patch`: increment the third number, reset nothing
- `minor`: increment the second number, reset patch to 0
- `major`: increment the first number, reset minor and patch to 0

Explicitly state both existing and new versions to the user, so that the user has a chance to stop the workflow.

---

## Step 2 — Run pre-release checks

```bash
make check-release
```

This verifies:
1. The working tree is clean (no uncommitted changes)
2. `ruff` linting + formatting
3. `mypy` type checking
4. Unit tests pass (`uv run pytest`)

If any check fails, stop and surface the error. Do **not** proceed to the version bump until everything passes.

---

## Step 3 — Bump version in `pyproject.toml`

Update the `version = "..."` line in `pyproject.toml` to the new version. Use the file editing tool — never `sed` or shell redirection.

---

## Step 4 — Commit, tag, and push

```bash
git add pyproject.toml
git commit -m "chore: release v<NEW_VERSION>"
git tag v<NEW_VERSION>
git push origin main
git push origin v<NEW_VERSION>
```

Pushing the tag fires both GitHub Actions workflows:
- **PyPI**: builds the package and publishes to PyPI using OIDC (no token needed locally)
- **Docker**: builds the `base` image target and pushes to `ghcr.io/<repo>` with tags for the version, `main`, and the commit SHA

Wait for the user to confirm the tag was pushed before continuing (CI runs in GitHub, not locally).

---

## Step 5 — Create the GitHub Release

```bash
gh release create v<NEW_VERSION> \
  --title "v<NEW_VERSION>" \
  --generate-notes
```

`--generate-notes` fills the body with a changelog based on merged PRs and commits since the previous tag. The user can edit the release on GitHub afterwards if they want to polish it.

If `gh` is not authenticated, remind the user to run `gh auth login` first.

---

## Step 6 — Deploy docs to Cloudflare Pages

```bash
make cf-deploy
```

This builds the Astro docs site (`public/`) and deploys it via `wrangler pages deploy`. If `wrangler` is not authenticated, the user will need to run `wrangler login` first.

---

## Reporting back

Once all steps are done, report:

- New version
- PyPI link: `https://pypi.org/project/openguard/<NEW_VERSION>/`
- GHCR image: `ghcr.io/<repo-owner>/openguard:v<NEW_VERSION>`
- GitHub Release URL (from `gh release view v<NEW_VERSION> --json url`)
- Note that GitHub Actions are running in the background (PyPI + Docker)

---

## Rollback notes

If something goes wrong after tagging:

- To delete the local tag: `git tag -d v<NEW_VERSION>`
- To delete the remote tag: `git push origin --delete v<NEW_VERSION>`
- Revert the version bump commit: `git revert HEAD`

PyPI releases cannot be deleted, but a new patch release can supersede a broken one.
