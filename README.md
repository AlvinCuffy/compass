# Compass

A personal daily operating dashboard: the morning sequence (prayer, reading),
today's checklist with automatic carryover for anything left undone, and a
Monday–Sunday view of the week.

God first, family second, business third.

## Files

- `public/compass.html` — the whole app (markup, styles, script). It is written
  as a page fragment so the same file can be published as a claude.ai Artifact.
- `server.js` — zero-dependency Node server that wraps the page and stores data
  in a JSON file.

## Storage

The page picks the first storage that works:

1. **claude.ai Artifact database** when opened as the published Artifact
   (synced across devices, readable only by the owner).
2. **The bundled server's API** (`GET /api/days`, `PUT /api/days/:date`) when
   run with `npm start`. Data lives in `data/compass.json` (or `$COMPASS_DATA`).
3. **This browser's localStorage** as a last resort, for example when the file
   is opened directly.

The server has no authentication. If you host it on a public URL, put it
behind something that restricts access.

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
