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
