# Landing Page Improvement Proposals

## Current State Summary

The page is a single monolithic Astro file with a retro CRT aesthetic — pixel fonts (Geist Pixel family), scanlines, chromatic aberration SVG filter, vignette overlay, flicker animation, and two interactive `<canvas>` animations (traffic flow + modular defense stack). The visual identity is strong and distinctive. The problems are structural, functional, and UX-related.

---

## 1. CRITICAL: Accessibility Is Non-Existent

- **Zero semantic structure.** No `<header>`, `<nav>`, `<main>`, `<article>` — just raw `<section>` and `<div>`. Screen readers get nothing useful.
- **Canvas animations have no text alternatives.** Two major interactive diagrams are completely invisible to assistive tech. The `.labels` div exists but is `display: none`.
- **CRT flicker animation** (`body::after` with 0.15s infinite flicker) is a **seizure/migraine trigger**. No `prefers-reduced-motion` check anywhere on the page.
- **Color contrast issues.** Body text at `opacity: 0.8` on `#0a0a0a` yields `#c4c4c4` — that's ~8.8:1, fine. But footer links at `opacity: 0.5-0.7` and status labels at `#666` on near-black fail WCAG AA (ratio ~3.9:1).
- **Toggle has no keyboard semantics.** The guard toggle is a `<div>` with a click handler — no `role="switch"`, no `aria-checked`, no keyboard support.
- **No skip-to-content link.**

**Proposal:** Add `prefers-reduced-motion` media query to disable ALL animations (flicker, scanlines, canvas). Add proper ARIA roles to the toggle. Make the hidden `.labels` div into an accessible fallback for the canvas. Add semantic landmarks.

---

## 2. HIGH: Performance — 1561 Lines of Inline Everything

- **Three perpetual `requestAnimationFrame` loops** running simultaneously (background grid, flow canvas, stack canvas). On low-end devices this will cook the CPU, especially mobile.
- **Six web fonts loaded from CDN**, five of which are pixel variants used only for the title cycling effect. That's ~6 network requests on the critical path despite `font-display: swap`.
- **All CSS is inline in `<style>`** (~580 lines). No extraction, no caching across navigations.
- **All JS is inline in `<script>`** (~750 lines). No code splitting, no lazy loading of canvas animations that are below the fold.
- **Canvas animations never pause** when off-screen. The background grid renders even when scrolled past the hero. The stack animation renders even before it's visible.

**Proposal:** Use `IntersectionObserver` to pause/start canvas animations when sections enter/exit viewport. Consider reducing to 2 pixel font variants (the cycling effect is novel but 5 fonts for a gimmick is expensive). Extract JS into a separate file for caching.

---

## 3. HIGH: Hero Section Lacks Clarity and Punch

- The tagline *"The security gateway for LLMs. Inspect, redact, and govern AI traffic in real-time."* is decent but generic. It doesn't differentiate from competitors.
- **No social proof** — no GitHub stars count, no "trusted by X" badges, no logos.
- **Single CTA** ("Get Started") with no secondary action (e.g., "View on GitHub", "See the Demo").
- The `<code>` blocks for `docker run` and `uvx openguard` are buried in section 4, far below the fold. These are the **primary conversion actions** for a dev tool — they should be near the hero or immediately after it.
- The "Get Started" button has no hover state feedback beyond a subtle white shift. No focus ring visible.

**Proposal:** Move the install commands (`docker run` / `uvx openguard`) directly under the hero tagline. Add a secondary CTA linking to GitHub. Add a GitHub stars badge. Strengthen the tagline to emphasize the zero-code-change value prop (e.g., *"Drop-in security proxy for LLM apps. Zero code changes. Full control."*).

---

## 4. MEDIUM: Interactive Demos Are Unclear Without Context

- **Traffic Control canvas:** The eye/blink mechanic is creative, but the meaning of red vs. green diamonds, "BLOCKED" vs "BREACH" events, and the toggle behavior aren't explained. A first-time visitor sees colorful dots bouncing around.
- **Modular Defense canvas:** Guard block names (Max Tokens, PII Filter, etc.) are rendered on canvas at 12px monospace — hard to read, especially on mobile. The lane concept isn't explained. Packets appear and disappear with no legend.
- Neither animation has any explanatory text overlay or caption beyond the section heading.

**Proposal:** Add a brief legend/key below each canvas (e.g., "Green = clean request, Red = malicious, Purple = breach"). Or better: overlay persistent labels directly on the canvas with larger font. For the stack animation, add a short sentence explaining what's being visualized: *"Each column is a request path. Guards activate as traffic passes through."*

---

## 5. MEDIUM: Mobile Experience Is Rough

- `html { font-size: 24px }` is the base — that's 50% larger than standard. All `rem` values are relative to this, making everything oversized on mobile.
- The flow canvas is forced to `220px` on mobile (`@media max-width: 600px`) — too cramped for the 4-node architecture diagram.
- There's only one `@media` breakpoint. No intermediate tablet handling.
- The sub-footer uses `font-size: 5vw` which becomes microscopic on small screens and enormous on ultrawide.
- The giant footer text (`font-size: calc(100vw / 6)`) will be ~60px on a 360px phone — fine, but on a 2560px monitor it's 426px, creating a massive scroll block.

**Proposal:** Reconsider the 24px base font-size — use 16px with explicit large sizes where needed. Add at least one more breakpoint (768px for tablets). Cap the footer text with a `max()` or `clamp()`. Test the canvas animations at 320px width.

---

## 6. MEDIUM: Information Architecture — Missing Sections

The page jumps from "here's what it is" to "here's an abstract animation" to "run this command." Missing:

- **Feature list / value props.** What guards exist? What protocols are supported? (The README answers this but the landing page doesn't.)
- **How it works in 3 steps.** The architecture animation is cool but doesn't replace a simple "1. Set your API key → 2. Point your client at OpenGuard → 3. Define your rules" flow.
- **Code example.** Show a before/after of a client config change — this is the product's killer feature (zero code changes).
- **Comparison or differentiation.** Why OpenGuard over rolling your own middleware?

**Proposal:** Add a structured features section between the hero and the architecture demo. A simple 3-column grid: "PII Filtering", "Content Moderation", "Token Limits" — each with a one-liner and an icon. Add a "How it works" 3-step flow above the architecture animation.

---

## 7. LOW: Code Blocks Are Click-to-Copy with No Feedback

Both `<code>` elements have `user-select: all` and a click handler calling `navigator.clipboard.writeText()`, but there's **no visual confirmation** that the copy succeeded. No tooltip, no icon change, no "Copied!" flash.

**Proposal:** Add a brief "Copied!" toast or text flash on successful clipboard write. Add a small copy icon to signal the affordance.

---

## 8. LOW: Font Cycling Effect Creates Visual Noise

The `startFontToggle()` function wraps every character of the title and footer in individual `<span>` elements, each with a `setInterval` at 1000ms cycling through 5 pixel fonts. This creates:
- Dozens of DOM elements for a short string
- Dozens of concurrent `setInterval` timers
- Constant layout thrash as font metrics change per character
- Distracting movement that competes with the hero message

**Proposal:** Either slow the cycle significantly (3-5s) and stagger the letters with a wave pattern, or limit it to a single font swap on the whole word (not per-letter). Consider making it trigger once on scroll-into-view rather than running infinitely.

---

## Priority Summary

| # | Severity | Proposal |
|---|----------|----------|
| 1 | CRITICAL | Accessibility: reduced-motion, ARIA, semantics, contrast |
| 2 | HIGH | Performance: pause off-screen canvases, reduce font loads |
| 3 | HIGH | Hero: move install commands up, add secondary CTA, social proof |
| 4 | MEDIUM | Canvas legends/explanations for both animations |
| 5 | MEDIUM | Mobile responsiveness: base font, breakpoints, caps |
| 6 | MEDIUM | Add features section, how-it-works steps, code example |
| 7 | LOW | Click-to-copy feedback |
| 8 | LOW | Tame the per-character font cycling |
