# Compass

A personal daily operating dashboard: the morning sequence (prayer, reading),
today's checklist with automatic carryover for anything left undone, and a
Monday–Sunday view of the week.

God first, family second, business third.

## Files

- `public/compass.html` — the whole app (markup, styles, script). It is written
  as a page fragment so the same file can be published as a claude.ai Artifact.
- `page.js` — wraps the fragment in a full HTML document.
- `build.js` — writes the static site to `dist/` for GitHub Pages.
- `server.js` — zero-dependency Node server that wraps the page and stores data
  in a JSON file.
- `.github/workflows/pages.yml` — deploys `dist/` to GitHub Pages on every push
  to `main`.

## Storage

The page picks the first storage that works:

1. **claude.ai Artifact database** when opened as the published Artifact
   (synced across devices, readable only by the owner).
2. **A secret GitHub Gist**, once you tap "Sync across devices" and paste a
   GitHub token with only the `gist` scope. The token stays in that browser.
   This is how the GitHub Pages copy syncs between devices.
3. **The bundled server's API** (`GET /api/days`, `PUT /api/days/:date`) when
   run with `npm start`. Data lives in `data/compass.json` (or `$COMPASS_DATA`).
4. **This browser's localStorage** as a last resort, for example when the file
   is opened directly.

The server has no authentication. If you host it on a public URL, put it
behind something that restricts access.

## Host on GitHub Pages

1. Merge this branch into `main`.
2. In the repo on GitHub: **Settings → Pages → Build and deployment → Source:
   GitHub Actions**. (Pages on a private repo needs a paid GitHub plan; on a
   free plan the repo must be public.)
3. The workflow deploys on every push to `main`. The site appears at
   `https://<user>.github.io/<repo>/`.
4. Open it on each device, tap **Sync across devices**, and paste a token from
   <https://github.com/settings/tokens/new?scopes=gist&description=Compass>.

The page carries a `noindex` tag, but anyone with the URL can open it. Your
checklist data lives only in the secret gist.

## Run locally

```sh
npm start            # http://localhost:3000
PORT=8080 npm start
```

## Carryover

Opening the app on a new day never writes anything. The page shows a virtual
"today" built from the most recent saved day: unfinished custom tasks carry
forward (keeping their original `carriedFrom`), and each unchecked fixed item
becomes a task tagged with that day. The first tap on anything saves today's
entry for real.

A fixed item that is already riding along from an earlier day isn't added a
second time, so a missed item shows once with its oldest date.
