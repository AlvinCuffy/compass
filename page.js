// Wraps public/compass.html (a body fragment, also published as a claude.ai
// Artifact) in a full HTML document. Used by server.js and build.js.
const fs = require("fs");
const path = require("path");

const HEAD =
  '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
  '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">' +
  '<meta name="robots" content="noindex,nofollow">' +
  '<meta name="apple-mobile-web-app-capable" content="yes">' +
  '<meta name="mobile-web-app-capable" content="yes">' +
  '<meta name="apple-mobile-web-app-title" content="Compass">' +
  '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' +
  '<meta name="theme-color" content="#070B1C">' +
  "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}" +
  "body{margin:0}[hidden]{display:none!important}</style></head><body>";

function renderPage() {
  return HEAD + fs.readFileSync(path.join(__dirname, "public", "compass.html"), "utf8") + "</body></html>";
}

module.exports = { renderPage };
