// Run with node scripts/test_report_axes.cjs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const template = fs.readFileSync(path.join(__dirname, '../crates/autoeq-report-wasm/shell/template.html'), 'utf8');
const source = template.slice(template.indexOf('function axisBounds('), template.indexOf('function figureForRender('));
const axisBounds = vm.runInNewContext(source + '; axisBounds');
const fig = {
    x: {scale: 'log', min: null, max: null},
    y: {min: 0, max: null},
    series: [{visible: true, x: [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000], y: [0.3, 0.8]}],
};
const [lo, hi] = axisBounds(fig, 'x');
assert(lo > 50 && lo < 63);
assert(hi > 16000 && hi < 20000);
assert(Math.abs(63 / lo - hi / 16000) < 1e-12);
assert.equal(axisBounds(fig, 'y')[0], 0);
fig.x.min = 20; fig.x.max = 20000;
assert(Math.abs(axisBounds(fig, 'x')[0] - 20) < 1e-10);
fig.x.min = 0; fig.x.max = null;
assert(axisBounds(fig, 'x')[0] > 50);
fig.series[0].x = [1000];
assert(axisBounds(fig, 'x')[0] > 0);
fig.series[0].x = [-1, 0];
assert(axisBounds(fig, 'x')[0] > 0);
console.log('Report axis bounds regression checks passed');

// Exercise section-local selectors without a browser or a rendering engine.
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.dataset = {}; this.attrs = {}; }
    appendChild(child) { this.children.push(child); return child; }
    setAttribute(key, value) { this.attrs[key] = value; }
    addEventListener(name, fn) { this.events[name] = fn; }
    get classList() { return {
        add: name => { this.className = (this.className || '') + ' ' + name; },
        toggle: (name, on) => { this.className = (this.className || '').split(' ').filter(c => c && c !== name).concat(on ? [name] : []).join(' '); },
    }; }
}
const localSource = template.slice(template.indexOf('function buildSectionGroup('), template.indexOf('function buildDom('));
const buildGroup = vm.runInNewContext(localSource + '; buildSectionGroup', {
    document: { createElement: tag => new Element(tag) }, renderVisible: () => {},
    appendSection: (host, section) => { host.appendChild({section}); },
});
const first = new Element('div'), second = new Element('div');
const sections = [{kind:'html', tab:'L'}, {kind:'html', tab:'R'}];
buildGroup(first, sections); buildGroup(second, sections);
first.children[0].children[1].events.click();
assert(!first.children[1].className.includes('active'));
assert(first.children[2].className.includes('active'));
assert(second.children[1].className.includes('active'));
assert(!second.children[2].className.includes('active'));
assert.equal(first.children[0].children[1].attrs['aria-pressed'], 'true');
console.log('Independent speaker selectors passed');

// Rotation is the same interaction for both rasterizers: only d3rs camera state changes.
const panSource = template.slice(template.indexOf('function wireGraphPan('), template.indexOf('function smoothedSection('));
let frame;
const pan = vm.runInNewContext(panSource + '; wireGraphPan', {
    structuredClone, ensureViewport: () => ({x:[20,20000],y:[0,500]}),
    requestAnimationFrame: fn => {frame = fn; return 1;}, renderOne: () => {},
});
const handlers = {};
const graph = {panEnabled:true, section:{kind:'grid',grid:{surface:true}},
    el:{addEventListener:(name,fn)=>{handlers[name]=fn;},setPointerCapture(){},
        getBoundingClientRect:()=>({width:1000,height:500}),classList:{add(){},remove(){}}}};
pan(graph);
handlers.pointerdown({clientX:100,clientY:100,pointerId:1});
handlers.pointermove({clientX:200,clientY:150});
assert.deepEqual(Array.from(graph.section.grid.rotation), [81,6]);
assert.equal(typeof frame, 'function'); frame();
handlers.pointerup();
console.log('Shared surface rotation controls passed');
