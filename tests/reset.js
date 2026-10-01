// @page studies/03-roofline.html
// floating reset (reset.js): an untouched page has no hash (widgets write only
// departures from their defaults), the dot tracks a non-empty hash, and the
// button clears the hash and reloads to the defaults. The reload re-runs this
// scenario, so sessionStorage sequences the two phases.
const btn = T.el('.resetb');
if (!sessionStorage.getItem('resettest')) {
  await T.tick(300);
  T.check('untouched page: no hash at rest', location.hash === '', location.hash);
  T.check('untouched page: no dot, button inert', !btn.classList.contains('mod') && btn.disabled, '');
  T.check('fixed in the top-right corner', getComputedStyle(btn).position === 'fixed'
    && innerWidth - btn.getBoundingClientRect().right === 12 && btn.getBoundingClientRect().top === 12, '');
  T.click('#mesh-ep button[data-v="ep"]'); await T.tick(50);
  T.check('a knob off its default writes the hash', location.hash.includes('mesh-ep'), location.hash);
  T.check('… and lights the dot', btn.classList.contains('mod') && !btn.disabled, '');
  T.click('#mesh-ep button[data-v="efsdp"]'); await T.tick(50);
  T.check('back to the default: hash clean, dot off', location.hash === '' && !btn.classList.contains('mod'), location.hash);
  T.click('#mesh-ep button[data-v="ep"]');
  T.click('#fsdpsched-fit button[data-v="2"]');
  T.el('#epsim [data-knob="step"] button').click(); await T.tick(50);
  T.check('three widgets modified', ['mesh-ep', 'fsdpsched-fit', 'epsim'].every(k => location.hash.includes(k)), location.hash);
  sessionStorage.setItem('resettest', JSON.stringify(T._out));   // phase 1's results survive the reload
  btn.click();   // reloads
} else {
  T._out.push(...JSON.parse(sessionStorage.getItem('resettest')));
  sessionStorage.removeItem('resettest');
  await T.tick(300);
  T.check('after reset: hash cleared', location.hash === '', location.hash);
  T.check('after reset: dot off', !btn.classList.contains('mod'), '');
  T.check('after reset: widgets at defaults', T.el('#mesh-ep button.on')?.dataset.v === 'efsdp'
    && T.el('#fsdpsched-fit button.on')?.dataset.v === '8', '');
  T.done();
}
