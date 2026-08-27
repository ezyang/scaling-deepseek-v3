// @page studies/02-hopper-memory.html
const l = document.getElementById('local-diagram');
l.pp = 1; l.stage = 0; l.cumulative = true; l.render(); l.changed(false); await T.tick(300);
const plan = l.closest('.anat-grid').querySelector('dsv3-anatomy-plan');
const rects = [...plan.querySelectorAll('g[data-op="embed"] rect')].slice(1);
const kinds = rects.map(r => r.getAttribute('fill') === 'none' ? 'hollow' : 'fill');
T.log('embed strips PP1 cum', JSON.stringify(kinds.reduce((m,k)=>(m[k]=(m[k]??0)+1,m),{})));
T.check('embedding mostly filled (unit stays fixed)', kinds.filter(k => k === 'fill').length >= 10, kinds.join(','));
T.done();
