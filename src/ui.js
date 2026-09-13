// Node-safe UI atoms shared by the widget modules (viewer.js re-exports
// them; the fsdp module imports them directly so its model stays importable
// under node, where viewer.js's HTMLElement classes cannot load).

// binary byte formatter (the param-bytes lens and the bytes tally)
export const fmtBytes = (b) =>
  !b ? '0'   // empty PP stages: a true zero, not a rounded one
    : b >= 2 ** 40 ? (b / 2 ** 40).toFixed(2) + ' TiB'
      : b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + ' GiB'
        : b >= 2 ** 20 ? (b / 2 ** 20).toFixed(1) + ' MiB' : (b / 1024).toFixed(1) + ' KiB';

// the local-knob control-strip styles (steppers, grouped rows) — shared by
// the layer's mini-head, <dsv3-pp-schedule>'s pipeline group, <dsv3-fsdp>
export const knobCss = (s) => `
${s} .pargrp { display: inline-flex; flex-direction: column; gap: 2px;
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; padding: 3px 8px 5px; align-self: stretch; }
${s} .pargrp.center { justify-content: center; }
${s} .parlab { font: italic 10px system-ui; color: var(--c-898781); }
${s} .parrow { display: flex; align-items: center; gap: 5px; min-height: 20px; }
${s} .stp { display: inline-flex; align-items: stretch; }
${s} .stp button { font: 12px ui-monospace, monospace; width: 20px; padding: 0 0 1px; border: 1px solid var(--c-c3c2b7); background: var(--c-ffffff); color: var(--c-52514e); cursor: pointer; }
@media (hover: hover) { ${s} .stp button:hover:not(:disabled) { background: var(--c-f3f2ee); } }
${s} .stp button:disabled { color: var(--c-dedcd3); cursor: default; }
${s} .stp button:first-child { border-radius: 4px 0 0 4px; }
${s} .stp button:last-child { border-radius: 0 4px 4px 0; }
${s} .stp button + button { border-left: none; }
${s} .stp button.on { background: var(--c-f3f2ee); color: var(--c-0b0b0b); font-weight: 600; cursor: default; }
${s} .stp button { width: auto; min-width: 20px; padding: 0 5px 1px; }
${s} .stp select.v { font: 11px ui-monospace, monospace; min-width: 4ch; padding: 2px 5px;
  border: 1px solid var(--c-c3c2b7); border-left: none; border-right: none; border-radius: 0;
  background: var(--c-ffffff); appearance: none; -webkit-appearance: none; text-align: center;
  text-align-last: center; cursor: pointer; }
${s} select { font: 12px system-ui; padding: 2px 6px; border: 1px solid var(--c-c3c2b7); border-radius: 4px; background: var(--c-ffffff); }
`;
