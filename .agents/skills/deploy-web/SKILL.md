---
name: deploy-web
description: Deploy the OpenGuard website for this repository: the Astro landing page, docs, and blog in `public/`, built via the repo's existing scripts and published to the `openguard` Cloudflare Pages project with `make cf-deploy`. Use when asked to deploy the landing page, docs, blog, or the whole site for this repo.
---

# Deploy Web

Use this skill when the user wants the OpenGuard website deployed from this repository.

Trigger phrases include:

- deploy the landing page
- deploy the docs
- deploy the blog
- publish the website
- ship the frontend
- push this site live

This repo has a single established deployment path. The website is an Astro + Starlight app in `public/`, its docs content is generated during the build, and the published target is the `openguard` Cloudflare Pages project.

## Repository facts

- Site source lives in `public/`
- The landing page, docs, and blog are deployed together as one static site
- `public/package.json` runs `uv run python scripts/generate_docs.py` in `prebuild`
- `wrangler.toml` sets `pages_build_output_dir = "public/dist"`
- The repo deploy target is `make cf-deploy`
- `make cf-deploy` publishes to Cloudflare Pages project `openguard`

## Default approach

Use the repo's existing flow exactly:

1. Confirm the user wants the website deployed from the current workspace state.
2. Verify Cloudflare auth with `make cf-whoami`.
3. Run `make cf-deploy` from the repo root.
4. Capture the Cloudflare Pages deployment URL.
5. Report the URL and any warnings that did not block publish.

Do not invent an alternate deploy path unless `make cf-deploy` is broken.

## What `make cf-deploy` does

- Runs `docs-build`
- Builds the Astro site from `public/`
- Triggers `prebuild`, which regenerates docs via `scripts/generate_docs.py`
- Produces static output in `public/dist`
- Runs `wrangler pages deploy public/dist --project-name openguard --branch main --commit-dirty=true`

## Pre-flight checks

Before deploying, verify:

- `wrangler` is installed
- `wrangler whoami` succeeds
- `wrangler.toml` still points to `public/dist`
- The repo root is the working directory before running `make cf-deploy`

If a prerequisite is missing, stop and tell the user exactly what is missing.

## Execution

Primary command:

```bash
make cf-deploy
```

Auth check:

```bash
wrangler whoami
```

Only fall back to the raw Wrangler command if the Make target is unavailable and the repo configuration still clearly supports it:

```bash
wrangler pages deploy public/dist --project-name openguard --branch main --commit-dirty=true
```

## Completion criteria

The task is complete only when all of the following are true:

- `make cf-deploy` or the equivalent Wrangler fallback exited successfully
- Cloudflare Pages returned a deployment URL
- You report that URL back to the user
- You mention any non-blocking warnings from the build or deploy output

## Reporting

Keep the final report short and concrete:

- State that the OpenGuard website was deployed
- Mention that the repo deploy path was `make cf-deploy`
- Include the returned Cloudflare Pages URL
- Include any warnings worth follow-up

## Example prompts

- Deploy the landing page
- Publish the docs site
- Deploy the blog
- Ship the OpenGuard site
- Push this marketing site live and tell me the URL

## Avoid

- Do not rewrite deployment around another platform
- Do not bypass `make cf-deploy` unless it is actually unavailable or broken
- Do not skip `wrangler whoami` when authentication is in doubt
- Do not claim success without the Cloudflare Pages deployment URL
- Do not hide warnings that may matter for the next deploy