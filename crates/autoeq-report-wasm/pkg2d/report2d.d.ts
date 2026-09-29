/* tslint:disable */
/* eslint-disable */

/**
 * Last internal error captured by [`toggle_series`] (empty when none).
 */
export function last_error(): string;

/**
 * Legend hit rectangles for a rendered canvas as JSON
 * (`[{"series":i,"x":..,"y":..,"w":..,"h":..}]`, `"[]"` when unknown).
 */
export function legend_json(canvas_id: string): string;

/**
 * Render one payload section into a canvas.
 *
 * Returns 0 on success, 1 when the canvas is missing, 2 on bad JSON.
 */
export function render_section(canvas_id: string, section_json: string, w: number, h: number, dpr: number): number;

/**
 * Payload schema discriminator (lets the shell refuse stale documents).
 */
export function schema_version(): string;

/**
 * Flip one series' visibility and re-render. Returns 1 when re-rendered,
 * 0 when the canvas or series is unknown, -99 on an internal panic
 * (see [`last_error`]).
 */
export function toggle_series(canvas_id: string, idx: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly last_error: () => [number, number];
    readonly legend_json: (a: number, b: number) => [number, number];
    readonly render_section: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly schema_version: () => [number, number];
    readonly toggle_series: (a: number, b: number, c: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
