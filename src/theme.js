// Night mode. Single source for every color on the site: the LIGHT values
// are the canonical literals the renderers were written in; DARK maps each
// to its night counterpart (OKLCH lightness-flip, hand-tuned: accents lift
// instead of darken, card/paper elevation inverts, heat ramps reverse so
// "hotter" stays the brighter end). Two consumption paths:
//   · CSS strings say var(--c-xxxxxx) — themeCss() emits the variable
//     blocks; stamp.mjs splices them into study.css/index.html for pre-JS
//     paint (with the head anti-flash snippet), and initTheme() injects
//     them for pages without study.css.
//   · SVG/canvas emissions call C('#xxxxxx') AT RENDER TIME — fills stay
//     concrete hexes (the fit chart lerps colors and tests read the
//     attributes), so a theme flip re-renders widgets (the 'dsv3-theme'
//     event, handled in viewer.js).
// Theme choice: prefers-color-scheme, overridden by the ◐ toggle
// (localStorage 'dsv3-theme').
export const DARK = {
  // neutrals (warm greys) — elevation inverts: cards sit ABOVE the paper
  '#f9f9f7': '#161614', '#fcfcfb': '#191917', '#ffffff': '#232320', '#fff': '#232320',
  '#f7f6f1': '#262521', '#f3f2ee': '#2a2923', '#f0efe9': '#282720', '#eeede7': '#2a2822',
  '#e9e8e2': '#2e2c26', '#e1e0d9': '#3d3b34', '#dedcd3': '#3b3931', '#dcdad2': '#3d3b33',
  '#d8d6cb': '#403d33', '#d5d4cb': '#423f35', '#d5d4cc': '#423f35', '#c3c2b7': '#615e51',
  '#aba89f': '#6a675e', '#a8a69e': '#6c6960', '#8f8d86': '#7b786f', '#898781': '#97948b',
  '#52514e': '#b0ada4', '#1c1c1a': '#d6d3ca', '#0b0b0b': '#e8e5dc',
  // byte components (weights blue · grads orange · optim green) — lifted
  '#2a78d6': '#5c9ae6', '#eb6834': '#f0824f', '#1baf7a': '#2bbd88',
  // blue family (weights tints, edit cells, blue text)
  '#bcd8f3': '#2b4767', '#dcebfa': '#223950', '#eef4fc': '#1d2b3c',
  '#0b3d75': '#a5c8f0', '#134a8e': '#8fb8ea',
  // save-amber family (stash accent + text ambers + tint backgrounds)
  '#eda100': '#eaa620', '#b05f00': '#d18d3c', '#a86e00': '#c08b2e', '#8a5f00': '#b28438',
  '#875600': '#ae7f3c', '#7a5200': '#c08e46', '#6f4712': '#b98e5c', '#5c3d00': '#a87f4b',
  '#4a3000': '#997347', '#3a2500': '#8c6a45', '#8c5a19': '#bb8649',
  '#fff8ea': '#2a2416', '#fdeab5': '#3d3213', '#fff3d1': '#322a10', '#f8f2e6': '#2b271d',
  '#fdefe8': '#2d221b',
  // pp-fold gold ramp (hot cells; cap = most intense = brightest on dark)
  '#f6cd74': '#6e5522', '#d19023': '#a3762a', '#d69432': '#a1752c',
  '#c98800': '#a5761f', '#eab04a': '#8a6c2b',
  // heat ramp (attention/act intensity): light ramp runs pale→near-black,
  // dark ramp runs dim→bright — monotone flip keeps "hotter = more"
  '#fbd4c0': '#4a2c1e', '#f9ded0': '#40291d', '#f3c8b3': '#573424', '#f3ac8b': '#6f4128',
  '#ecb298': '#67402c', '#e58a63': '#935134', '#d16b42': '#b95e37', '#c74e1d': '#d4632f',
  '#a63c12': '#e0714a', '#88300c': '#eb8666', '#7a2f12': '#f0916f', '#5c2410': '#f6a888',
  '#471b09': '#fcbda0', '#361406': '#ffd0b8',
  // precision family (e4m3 pink · e5m6 purple · fp32 brick) + trace cats
  '#d6408b': '#e0559b', '#7b2fa8': '#b675e8', '#8a3324': '#d97862',
  '#e87ba4': '#d76d97', '#008300': '#43a83c', '#d03b3b': '#e25555',
  '#4a3aa7': '#948ee8', '#4636a3': '#8f88e4', '#6b5bd2': '#9a8df0',
  // odd tints (a2a violet bg, alert bg, green bg) + status greens
  '#f3f1fb': '#272438', '#fdf1f1': '#322323', '#f0faf4': '#1e2a23', '#1a7a43': '#3dae6e',
};

let dark = false;
export const isDark = () => dark;
// render-time color resolve: identity in light, DARK counterpart in dark
export const C = (h) => (dark && DARK[h]) || h;

export const themeCss = () => {
  const six = Object.entries(DARK).filter(([k]) => k.length === 7);
  const decl = (mode) => six.map(([k, v]) => `--c-${k.slice(1)}: ${mode === 'dark' ? v : k};`).join(' ');
  return `:root { color-scheme: light; ${decl('light')} --spot: rgba(252, 252, 251, 0.78); }\n` +
    `:root.dark { color-scheme: dark; ${decl('dark')} --spot: rgba(14, 14, 13, 0.82); }`;
};

// the head anti-flash snippet pages inline BEFORE their stylesheets — must
// stay in lockstep with initTheme()'s choice logic (stamp.mjs splices it)
export const FLASH_SNIPPET = "document.documentElement.classList.toggle('dark', " +
  "(localStorage.getItem('dsv3-theme') ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark');";

const apply = (d) => {
  dark = d;
  document.documentElement.classList.toggle('dark', d);
  dispatchEvent(new Event('dsv3-theme'));   // widgets re-render their C() colors
};
// explicit choice (the ◐, tests, the dark pixel goldens): persists + applies
export const setTheme = (d) => { localStorage.setItem('dsv3-theme', d ? 'dark' : 'light'); apply(d); };

export function initTheme() {
  const saved = () => localStorage.getItem('dsv3-theme');
  const mq = matchMedia('(prefers-color-scheme: dark)');
  const compute = () => (saved() ?? (mq.matches ? 'dark' : 'light')) === 'dark';
  dark = compute();
  document.documentElement.classList.toggle('dark', dark);
  mq.addEventListener('change', () => { if (!saved()) apply(compute()); });
  // variable blocks for pages that don't load study.css, + the ◐ toggle
  const st = document.createElement('style');
  st.textContent = themeCss() + `
.themeb { position: fixed; right: 12px; bottom: 12px; z-index: 900; font: 15px/1 system-ui;
  width: 34px; height: 34px; border: 1px solid var(--c-c3c2b7); border-radius: 999px;
  background: var(--c-ffffff); color: var(--c-52514e); cursor: pointer;
  box-shadow: 0 1px 5px rgba(11, 11, 11, 0.12); }
@media (hover: hover) { .themeb:hover { color: var(--c-0b0b0b); } }`;
  document.head.append(st);
  const b = document.createElement('button');
  b.className = 'themeb'; b.type = 'button'; b.textContent = '◐';
  b.title = 'light / dark';
  b.onclick = () => setTheme(!dark);
  const mount = () => document.body.append(b);
  document.body ? mount() : addEventListener('DOMContentLoaded', mount, { once: true });
}
