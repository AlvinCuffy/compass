// Builds the static site for GitHub Pages into dist/.
const fs = require("fs");
const path = require("path");
const { renderPage } = require("./page");

const out = path.join(__dirname, "dist");
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out);
fs.writeFileSync(path.join(out, "index.html"), renderPage());
fs.writeFileSync(path.join(out, ".nojekyll"), "");
console.log("Built dist/index.html");
