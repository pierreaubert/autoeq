// Run with node scripts/test_report_gpui.cjs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const template = fs.readFileSync(path.join(__dirname, '../crates/autoeq-report-wasm/shell/template.html'), 'utf8');
const source = template.slice(template.indexOf('async function bootGpu()'), template.lastIndexOf('\nif (payload) {'));
assert(!template.includes('id="view-gpui"'));
assert(!template.includes('id="report-gpui"'));
assert(template.includes('bootGpu().then('));

async function check(mode) {
    const status = {}, window = {}, events = {};
    let lost, submissions = 0, copies = 0, redraws = 0, uploaded;
    const texture = () => ({createView: () => ({}), destroy() {}});
    const device = {
        limits: {maxTextureDimension2D: 8192}, destroy() {},
        lost: new Promise(resolve => {lost = resolve;}),
        addEventListener: (name, fn) => {events[name] = fn;},
        createShaderModule: () => ({}),
        createRenderPipelineAsync: async () => { if (mode === 'pipeline-error') throw Error('shader failed'); return {}; },
        createTexture: texture, createBuffer: () => ({destroy() {}}),
        queue: {writeBuffer: (_buffer, _offset, data) => {uploaded = [...data];}, submit: () => {submissions++;}},
        createCommandEncoder: () => ({finish: () => ({}), beginRenderPass: () => ({
            setPipeline() {}, setVertexBuffer() {}, draw(count) {assert.equal(count, 3);}, end() {},
        })}),
    };
    const canvas = {width: 1, height: 1, getContext: () => ({configure() {}, unconfigure() {}, getCurrentTexture: texture})};
    const gpu = {requestAdapter: async () => mode === 'no-adapter' ? null : ({requestDevice: async () => {
        if (mode === 'device-error') throw Error('device failed'); return device;
    }}), getPreferredCanvasFormat: () => 'bgra8unorm'};
    const context = vm.createContext({window, navigator: mode === 'no-api' ? {} : {gpu},
        URLSearchParams, location: {search: mode === 'forced' ? '?renderer=canvas' : ''},
        document: {getElementById: id => {assert.equal(id, 'renderer-status'); return status;}, createElement: () => canvas},
        GPUTextureUsage: {RENDER_ATTACHMENT: 1}, GPUBufferUsage: {VERTEX: 1, COPY_DST: 2},
        console: {warn() {}}, requestAnimationFrame: fn => fn(), renderVisible: () => {redraws++;},
    });
    await vm.runInContext(source + '; bootGpu()', context);
    if (mode !== 'available') {
        assert.equal(status.textContent, 'Renderer: Canvas fallback');
        assert.equal(window.__reportTriangles, undefined);
        return;
    }
    assert.equal(status.textContent, 'Renderer: WebGPU');
    const ctx = {canvas: {width: 200, height: 100, dataset: {}},
        getTransform: () => ({a:2, b:0, c:0, d:2, e:0, f:0}),
        save() {}, resetTransform() {}, restore() {}, drawImage() {copies++;}};
    const vertices = new Float32Array([0,0,1,0,0,1, 100,0,1,0,0,1, 0,50,1,0,0,1]);
    assert.equal(window.__reportTriangles(ctx, vertices), true);
    assert.equal(submissions, 1); assert.equal(copies, 1);
    assert.equal(ctx.canvas.dataset.renderer, 'webgpu');
    assert.deepEqual([uploaded[0],uploaded[1],uploaded[6],uploaded[7],uploaded[12],uploaded[13]], [-1,1,1,1,-1,-1]);
    lost({reason:'unknown', message:'GPU disconnected'});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(window.__reportTriangles, undefined);
    assert.equal(status.textContent, 'Renderer: Canvas fallback');
    assert.equal(redraws, 1);
}
Promise.all(['available','no-api','no-adapter','device-error','pipeline-error','forced'].map(check))
    .then(() => console.log('Automatic WebGPU selection, shared compositor, and fallback checks passed'))
    .catch(error => {console.error(error); process.exitCode = 1;});
