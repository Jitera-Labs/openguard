---
name: test-frontend
description: Test and visually inspect the OpenGuard frontend (marketing site + docs). Use when asked to "test the frontend", "check the docs site", "review the UI", "open the site", "verify the landing page", or anything involving the Astro/Starlight site running on port 4400.
---

# Test Frontend

The OpenGuard frontend is an Astro + Starlight site: a marketing landing page plus a documentation section. It runs in Docker on port 4400.

Uses the `agent-browser` skill for all browser interactions. Load it if not already loaded.

## Launch

```bash
# Start everything (backend on :23294 + frontend on :4400)
make dev

# Or frontend-only container
docker compose up ui
```

## Testing Workflow

Use `agent-browser` to navigate, snapshot, and visually verify each page.

### Landing page

```bash
agent-browser open http://localhost:4400 && agent-browser wait --load networkidle && agent-browser screenshot
agent-browser snapshot -i  # Check hero, nav links, feature list are present
```

### Docs pages

Navigate each route and snapshot to verify content, sidebar, and code blocks render:

```bash
# Docs index — sidebar should be visible
agent-browser open http://localhost:4400/docs/ && agent-browser wait --load networkidle && agent-browser screenshot

# Getting Started
agent-browser open http://localhost:4400/docs/getting-started && agent-browser wait --load networkidle && agent-browser screenshot

# Configuration
agent-browser open http://localhost:4400/docs/configuration && agent-browser wait --load networkidle && agent-browser screenshot

# Guard pages
for path in guards/content_filter guards/keyword_filter guards/llm_input_inspection guards/max_tokens guards/pii_filter; do
  agent-browser open "http://localhost:4400/docs/$path" && agent-browser wait --load networkidle && agent-browser screenshot
done

# 404
agent-browser open http://localhost:4400/does-not-exist && agent-browser wait --load networkidle && agent-browser screenshot
```

### Responsive check

```bash
agent-browser set device "iPhone 14" && agent-browser open http://localhost:4400 && agent-browser wait --load networkidle && agent-browser screenshot
agent-browser set viewport 1920 1080 && agent-browser open http://localhost:4400 && agent-browser wait --load networkidle && agent-browser screenshot
```

## What to Look For

- No broken layouts or unstyled content
- Sidebar renders and nav links are interactive (`snapshot -i` shows clickable refs)
- Code blocks have syntax highlighting
- Fonts load (Geist Pixel from CDN — may be slow on first load)
- Dark theme applies (Starlight black theme)
- 404 page shows custom layout, not a blank page

## Source Locations

- Pages: `public/src/pages/` (landing = `index.astro`, 404 = `404.astro`)
- Docs content: `public/src/content/docs/docs/`
- Styles: `public/src/styles/custom.css`
- Config: `public/astro.config.mjs`

## Hot Reload

The `ui` container bind-mounts the repo, so edits to `public/src/` are reflected immediately — no rebuild needed.
