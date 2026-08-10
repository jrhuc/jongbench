import type { ComponentChildren } from "preact";

export function Shell({ names, children }: { names?: string[]; children: ComponentChildren }) {
  return (
    <>
      <header class="app-header">
        <span class="brand">jongbench</span>
        {names && <span class="header-seats">{names.join(" · ")}</span>}
      </header>
      {children}
    </>
  );
}
