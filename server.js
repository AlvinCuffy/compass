// Compass: serves the single page and a tiny JSON API backed by a file.
//   GET  /api/days          -> { days: { "YYYY-MM-DD": entry } }
//   PUT  /api/days/:date    -> saves one day's entry
// No dependencies. Data lives in $COMPASS_DATA (default ./data/compass.json).
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT) || 3000;
const DATA = process.env.COMPASS_DATA || path.join(__dirname, "data", "compass.json");
const { renderPage } = require("./page");

function readData() {
  try {
    return JSON.parse(fs.readFileSync(DATA, "utf8"));
  } catch {
    return { days: {} };
  }
}

function writeData(data) {
  fs.mkdirSync(path.dirname(DATA), { recursive: true });
  const tmp = DATA + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
  fs.renameSync(tmp, DATA);
}

function send(res, status, body, type = "application/json") {
  res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  res.end(typeof body === "string" ? body : JSON.stringify(body));
}

function validEntry(e) {
  return (
    e && typeof e === "object" &&
    Array.isArray(e.morning) && Array.isArray(e.nightly) && Array.isArray(e.tasks) &&
    e.tasks.every((t) => t && typeof t.id === "string" && typeof t.text === "string")
  );
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");

  if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
    return send(res, 200, renderPage(), "text/html; charset=utf-8");
  }

  if (req.method === "GET" && url.pathname === "/api/days") {
    return send(res, 200, readData());
  }

  const m = url.pathname.match(/^\/api\/days\/(\d{4}-\d{2}-\d{2})$/);
  if (req.method === "PUT" && m) {
    let body = "";
    req.on("data", (c) => {
      body += c;
      if (body.length > 256 * 1024) req.destroy();
    });
    req.on("end", () => {
      let entry;
      try {
        entry = JSON.parse(body);
      } catch {
        return send(res, 400, { error: "Body must be JSON" });
      }
      if (!validEntry(entry)) return send(res, 400, { error: "Entry needs morning, nightly and tasks arrays" });
      const data = readData();
      data.days[m[1]] = { ...entry, date: m[1] };
      writeData(data);
      send(res, 200, { ok: true });
    });
    return;
  }

  send(res, 404, { error: "Not found" });
});

server.listen(PORT, () => console.log(`Compass on http://localhost:${PORT}`));
