// Run with node scripts/test_report_zoom.cjs.
// Sankey/bar cards have no data viewport, so the +/− Zoom toolbar applies a
// geometric magnification (crisp re-render at a larger canvas size) instead
// of the figure/grid viewport zoom. The Move toggle is hidden there because
// drag-pan only works on viewport kinds.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const template = fs.readFileSync(path.join(__dirname, '../crates/autoeq-report-wasm/shell/template.html'), 'utf8');
const slice = (from, to) => template.slice(template.indexOf(from), template.indexOf(to));

// --- Pure zoom math (no DOM): figure viewport zoom still works, sankey zoom clamps.
const math = slice('function isVisible(', 'function wireGraphPan(');
const sandbox = {structuredClone, renderOne: () => {}};
vm.createContext(sandbox);
vm.runInContext(math, sandbox);
const figure = {
    kind: 'figure',
    figure: {
        x: {scale: 'log', min: null, max: null},
        y: {min: null, max: null},
        series: [{visible: true, x: [20, 20000], y: [50, 90]}],
    },
};
const fgraph = {section: figure, viewport: null};
sandbox.fgraph = fgraph;
vm.runInContext('zoomGraph(fgraph, 1 / 1.5)', sandbox);
assert(fgraph.viewport !== null);
assert(fgraph.viewport.x[0] > 20 && fgraph.viewport.x[1] < 20000);

const sgraph = {section: {kind: 'sankey'}, viewport: null, geomZoom: 1};
sandbox.sgraph = sgraph;
vm.runInContext('geomZoom(sgraph, 1.25)', sandbox);
assert.equal(sgraph.geomZoom, 1.25);
vm.runInContext('geomZoom(sgraph, 100)', sandbox);
assert.equal(sgraph.geomZoom, 4);
vm.runInContext('geomZoom(sgraph, 0.001)', sandbox);
assert.equal(sgraph.geomZoom, 0.5);
// Zooming back out of a 1.25 step lands exactly on fit.
sgraph.geomZoom = 1;
vm.runInContext('geomZoom(sgraph, 1.25); geomZoom(sgraph, 1 / 1.25)', sandbox);
assert.equal(sgraph.geomZoom, 1);
console.log('Zoom math checks passed');

// --- Toolbar wiring with a stub DOM: sankey zoom buttons drive geomZoom,
// --- Move is hidden; figure keeps Move and viewport zoom.
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.style = {}; this.dataset = {}; this.attrs = {}; }
    appendChild(child) { this.children.push(child); return child; }
    append(...kids) { this.children.push(...kids); }
    setAttribute(key, value) { this.attrs[key] = value; }
    addEventListener(name, fn) { this.events[name] = fn; }
    get classList() { return {
        add: name => { this.className = (this.className || '') + ' ' + name; },
        toggle: (name, on) => { this.className = (this.className || '').split(' ').filter(c => c && c !== name).concat(on ? [name] : []).join(' '); },
    }; }
}
const wiring = slice('function appendSection(', 'function setFreqPreset(')
    + slice('function isVisible(', 'function wireGraphPan(');
const canvases = [];
const ui = {
    document: {createElement: tag => new Element(tag)},
    structuredClone, canvases,
    wireGraphPan: () => {}, renderOne: () => {},
    smoothedSection: s => s, setFreqPreset: () => {}, setDbSpan: () => {},
    syncSurfaceControls: () => {},
};
vm.createContext(ui);
vm.runInContext(wiring, ui);
const runAppend = (sec) => {
    const host = new Element('div');
    ui.host = host; ui.sec = sec;
    vm.runInContext('appendSection(host, sec, 0)', ui);
    return host;
};
const sankeyHost = runAppend({kind: 'sankey', chart: {title: 'flow'}});
const sankeyGraph = canvases[canvases.length - 1];
const toolbar = sankeyHost.children[0].children[0];
const [zoomOut, zoomIn, reset, move] = toolbar.children;
assert.equal(move.style.display, 'none');
zoomIn.events.click();
assert.equal(sankeyGraph.geomZoom, 1.25);
zoomIn.events.click();
assert(sankeyGraph.geomZoom > 1.5);
zoomOut.events.click();
assert(sankeyGraph.geomZoom < 1.6);
reset.events.click();
assert.equal(sankeyGraph.geomZoom, 1);

const figHost = runAppend({kind: 'figure', figure: structuredClone(figure.figure)});
const figGraph = canvases[canvases.length - 1];
const figToolbar = figHost.children[0].children[0];
assert(figToolbar.children[3].style.display !== 'none');
figToolbar.children[1].events.click();
assert(figGraph.viewport !== null);
console.log('Toolbar wiring checks passed');

// --- renderOne magnifies crisply: bitmap follows the zoomed CSS size.
const renderSrc = slice('function sectionHeight(', 'function appendSection(')
    + slice('function isVisible(', 'function renderVisible(');
let rendered = null;
const win = {devicePixelRatio: 1};
const rsandbox = {
    structuredClone, window: win,
    wasm2d: {render_section: (id, json, w, h) => { rendered = {id, w, h}; return 0; }},
};
vm.createContext(rsandbox);
vm.runInContext(renderSrc, rsandbox);
const canvas = () => ({
    clientWidth: 760, style: {}, dataset: {},
    parentElement: {clientWidth: 784, style: {}},
    getClientRects: () => [1], closest: () => null,
});
const render = (g) => { rsandbox.g = g; vm.runInContext('renderOne(g)', rsandbox); };
const zg = {id: 'plot-9', section: {kind: 'sankey'}, el: canvas(), geomZoom: 1};
render(zg);
assert.equal(rendered.w, 760);
assert.equal(rendered.h, 380);
assert.equal(zg.el.style.width, undefined);
const zg2 = {id: 'plot-9', section: {kind: 'sankey'}, el: canvas(), geomZoom: 2};
render(zg2);
assert.equal(rendered.w, 1520);
assert.equal(rendered.h, 760);
assert.equal(zg2.el.style.width, '1520px');
assert.equal(zg2.el.style.height, '760px');
assert.equal(zg2.el.parentElement.style.overflow, 'auto');
// Back to fit clears the explicit size so the card fills its container again.
zg2.geomZoom = 1;
render(zg2);
assert.equal(zg2.el.style.width, '');
console.log('Geometric render checks passed');
