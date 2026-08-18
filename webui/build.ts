// Emits jongbench/webui_page.html: one self-contained page (JS, CSS and the
// pai.svg sprite inlined) served as-is by jongbench/webui.py.

import { watch } from "node:fs";

const ROOT = import.meta.dir;
const OUT_PATH = `${ROOT}/../jongbench/webui_page.html`;
const SPRITE_PATH = `${ROOT}/../assets/pai.svg`;
// Dela Gothic One subset to ascii + the handful of kanji the UI renders,
// and DSEG7 Classic subset to digits for the table's LED score readouts.
const FONT_PATH = `${ROOT}/../assets/dela.woff2`;
const LED_FONT_PATH = `${ROOT}/../assets/dseg7.woff2`;

async function build(): Promise<void> {
  const result = await Bun.build({
    entrypoints: [`${ROOT}/src/main.tsx`],
    minify: true,
    target: "browser",
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error("bundle failed");
  }
  let js = "";
  let css = "";
  for (const artifact of result.outputs) {
    if (artifact.path.endsWith(".css")) css += await artifact.text();
    else js += await artifact.text();
  }
  const font = Buffer.from(await Bun.file(FONT_PATH).arrayBuffer()).toString("base64");
  css = css.replace("__DELA_WOFF2__", font);
  const led = Buffer.from(await Bun.file(LED_FONT_PATH).arrayBuffer()).toString("base64");
  css = css.replace("__DSEG7_WOFF2__", led);
  // The sprite's #tile body rect has no fill (renders black); make it
  // transparent so the CSS tile face shows through.
  const sprite = (await Bun.file(SPRITE_PATH).text()).replace(
    '<rect x="0" y="0" width="320" height="446" rx="30" ry="30" />',
    '<rect x="0" y="0" width="320" height="446" rx="30" ry="30" fill="none" />',
  );
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jongbench</title>
<style>${css}</style>
</head>
<body>
<div id="sprite" hidden>${sprite}</div>
<div id="app"></div>
<script>${js}</script>
</body>
</html>
`;
  if (process.argv.includes("--check")) {
    const output = Bun.file(OUT_PATH);
    if (!(await output.exists()) || (await output.text()) !== html) {
      throw new Error(
        "jongbench/webui_page.html is missing or stale; run `bun run build` and commit it",
      );
    }
    console.log(`verified ${OUT_PATH} (${(html.length / 1024).toFixed(0)} KiB)`);
    return;
  }
  await Bun.write(OUT_PATH, html);
  console.log(`built ${OUT_PATH} (${(html.length / 1024).toFixed(0)} KiB)`);
}

if (process.argv.includes("--check") && process.argv.includes("--watch")) {
  throw new Error("--check and --watch cannot be used together");
}

await build();

if (process.argv.includes("--watch")) {
  let pending: ReturnType<typeof setTimeout> | null = null;
  watch(`${ROOT}/src`, { recursive: true }, () => {
    if (pending) clearTimeout(pending);
    pending = setTimeout(() => build().catch(console.error), 100);
  });
  console.log("watching webui/src ...");
}
