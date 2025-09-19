# Repository Guidelines

## Project Structure & Module Organization
- `index.html`: Single-file Snake game (HTML/CSS/JS), no build system.
- `fractal-think.md`: 分形思考流程说明（Core4 与运行原则）。
- `thinkon.md`: Fractal Thinkon 规范（最小递归框架）。
- `README.md`: 项目摘要。若需图片/附件，请新建 `assets/` 目录存放。

## Build, Test, and Development Commands
- Preview locally (no build):
  - `open index.html` (macOS) or open in any browser.
  - Or serve: `python3 -m http.server 8000` → visit `http://localhost:8000/`.
- Dev workflow: edit `index.html` or Markdown, refresh the page; use DevTools Console for errors.

## Coding Style & Naming Conventions
- HTML/CSS/JS: 2-space indent; prefer `const/let` over `var`; keep functions small and pure; avoid frameworks for now.
- Text/UI language: Chinese for UI/docs; use English technical terms适度穿插。
- Markdown: use `#`, `##` headings; favor lists over long paragraphs; wrap lines ~80–100 chars.
- Filenames: kebab-case for new docs (e.g., `design-notes.md`); place assets under `assets/`.

## Testing Guidelines
- Manual checks for the game:
  - Space toggles start/pause; `R` resets; arrow keys move; no 180° reversal.
  - Score increments; speed increases every 5 foods; overlay messages correct.
  - Collision with wall/self triggers game over; no Console errors.
  - Canvas remains focusable (`tabindex="0"`) and playable via keyboard.
- Accessibility: keep `aria-label` on `canvas`; maintain sufficient contrast in styles.

## Commit & Pull Request Guidelines
- Commits: short, imperative, optional scope.
  - Examples: `docs: clarify Core4 metric examples`; `game: fix collision timing`.
- PRs: clear description, motivation, before/after screenshots or GIF for UI changes; link related issues; keep changes focused; include manual test notes (commands used, browser/version).

## Security & Configuration Tips (Optional)
- No external dependencies or trackers. If adding libraries, vendor under `third_party/` and document license in the PR.
