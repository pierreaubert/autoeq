"""Recording wizard application (gpui_toolkit declarations + handlers).

One sidebar section per workflow step; each section carries a stepper
marking the step position. Sidebar navigation moves between steps (the v1
session protocol has no subtree replace, so there are no synthetic
Next/Back buttons); action buttons drive the backend and live status /
progress / result nodes update through ``set`` patches.

Host-side control edits flow back into the model through select actions
and commit actions with ``payload.value`` (demo_app pattern).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from ._toolkit import ensure_toolkit  # noqa: E402

ensure_toolkit()

from gpui_toolkit import (  # noqa: E402
    App,
    Event,
    SessionContext,
    data as tkdata,
    px,
    section,
    ui,
)
from gpui_toolkit import audio as tk_audio  # noqa: E402
from gpui_toolkit.miniapp import MiniAppConfig  # noqa: E402

from .backend import CaptureBackend, TakeRequest  # noqa: E402
from .model import (  # noqa: E402
    STEPS,
    TAKE_DONE,
    TAKE_FAILED,
    TAKE_PENDING,
    TAKE_RECORDING,
    TakeState,
    WizardModel,
)

_APP_WIDTH = 1240.0
_APP_HEIGHT = 840.0
_CARD_WIDTH = 900.0

SIGNAL_OPTIONS = (
    "sweep",
    "tone",
    "two-tone",
    "white-noise",
    "pink-noise",
    "m-noise",
    "mls",
    "dirac",
)


def _stepper(index: int) -> ui.Node:
    return ui.stepper(
        id=f"wizard-stepper-{STEPS[index][0]}",
        steps=[label for _, label in STEPS],
        active=index,
    )


def _status_row(status_id: str, status: str) -> ui.Node:
    return ui.hstack(
        [ui.metric("Status", status, id=status_id)],
        gap=12.0,
    )


def _config_section(model: WizardModel, backend_name: str) -> ui.Node:
    outputs = [d.name for d in model.output_devices]
    inputs = [d.name for d in model.input_devices]
    output_labels = [d.label() for d in model.output_devices]
    input_labels = [d.label() for d in model.input_devices]
    return ui.vstack(
        [
            _stepper(0),
            ui.section_header(
                "Devices & setup",
                "Playback / recording devices, microphone calibration, "
                "output directory, and sweep quality.",
            ),
            ui.hstack(
                [
                    ui.badge(
                        f"backend: {backend_name}"
                        + (" (demo devices)" if model.demo_mode else ""),
                        tone="neutral",
                    ),
                ],
                gap=8.0,
            ),
            _status_row("config-status", model.status),
            ui.accordion(
                id="wizard-config",
                items=[
                    ("playback", "Playback device", [
                        ui.select(
                            id="wizard-output-device",
                            label="Output device",
                            value=model.output_name or "",
                            options=list(zip(outputs, output_labels))
                            or [("", "no devices")],
                            action="wizard_output_device",
                        ),
                    ]),
                    ("recording", "Recording device", [
                        ui.select(
                            id="wizard-input-device",
                            label="Input device",
                            value=model.input_name or "",
                            options=list(zip(inputs, input_labels))
                            or [("", "no devices")],
                            action="wizard_input_device",
                        ),
                    ]),
                    ("calibration", "Microphone calibration", [
                        ui.path_input(
                            id="wizard-mic-cal",
                            label="Calibration file",
                            value=model.mic_calibration_path or "",
                            mode="open_file",
                            filters=[("Calibration", ["txt", "csv", "cal"])],
                            commit_action="wizard_mic_cal",
                        ),
                    ]),
                    ("output", "Output directory", [
                        ui.path_input(
                            id="wizard-output-dir",
                            label="Takes directory",
                            value=model.output_dir or "",
                            mode="directory",
                            commit_action="wizard_output_dir",
                        ),
                    ]),
                    ("advanced", "Advanced sweep quality", [
                        ui.number_input(
                            id="wizard-sweeps",
                            label="Sweeps per channel",
                            value=model.signal.num_sweeps,
                            minimum=1, maximum=8, step=1,
                            commit_action="wizard_sweeps",
                        ),
                        ui.number_input(
                            id="wizard-pre-silence",
                            label="Pre-silence (s)",
                            value=model.signal.pre_silence_s,
                            unit="s", minimum=0.0, maximum=10.0,
                            commit_action="wizard_pre_silence",
                        ),
                        ui.number_input(
                            id="wizard-bass-octave",
                            label="Bass octave duration (s/oct)",
                            value=model.signal.bass_octave_s,
                            unit="s", minimum=0.5, maximum=10.0,
                            commit_action="wizard_bass_octave",
                        ),
                    ]),
                ],
                expanded=["playback", "recording"],
                multiple=True,
            ),
        ],
        gap=16.0,
    )


def _spl_section(model: WizardModel) -> ui.Node:
    result = (model.spl_result or {}).get("message", "-")
    return ui.vstack(
        [
            _stepper(1),
            ui.section_header(
                "SPL calibration",
                "Reference tone for absolute SPL anchoring.",
            ),
            _status_row("spl-status", model.status),
            ui.hstack(
                [
                    ui.metric("Result", result, id="spl-result"),
                    ui.button("Run SPL calibration", id="spl-start",
                              action="spl_start"),
                ],
                gap=12.0,
            ),
            tk_audio.level_meter(
                id="spl-meter", levels=(0.0,), peaks=(0.0,),
                channel_names=["Mic"],
            ),
            ui.text(
                "Live metering arrives with the host audio loop; this "
                "meter shows last-known levels (none yet).",
                tone="secondary",
            ),
        ],
        gap=16.0,
    )


def _takes_table(takes: list[TakeState]) -> ui.Node:
    rows = [
        [take.label, str(take.out_channel), str(take.in_channel),
         take.state, take.message]
        for take in takes
    ]
    return ui.table(
        ["Take", "Out ch", "In ch", "State", "Message"], rows,
        id="capture-takes",
    )


def _capture_section(model: WizardModel) -> ui.Node:
    signal = model.signal
    return ui.vstack(
        [
            _stepper(2),
            ui.section_header(
                "Capture",
                "Signal stimulus per channel. Sweeps are self-timed: the "
                "duration knob applies to non-sweep signals.",
            ),
            _status_row("capture-status", model.status),
            ui.card(
                [
                    ui.heading("Signal", level=3),
                    ui.select(
                        id="wizard-signal-type",
                        label="Signal type",
                        value=signal.signal_type,
                        options=[(name, name) for name in SIGNAL_OPTIONS],
                        action="wizard_signal_type",
                    ),
                    ui.hstack(
                        [
                            ui.number_input(
                                id="wizard-duration", label="Duration (s)",
                                value=signal.duration_s, unit="s",
                                minimum=0.1, maximum=60.0,
                                commit_action="wizard_duration",
                            ),
                            ui.number_input(
                                id="wizard-level", label="Level (dBFS)",
                                value=signal.level_db, unit="dB",
                                minimum=-60.0, maximum=0.0,
                                commit_action="wizard_level",
                            ),
                        ],
                        gap=12.0,
                    ),
                    ui.hstack(
                        [
                            ui.number_input(
                                id="wizard-start-hz", label="Start (Hz)",
                                value=signal.start_hz, unit="Hz",
                                minimum=1.0, maximum=24000.0,
                                commit_action="wizard_start_hz",
                            ),
                            ui.number_input(
                                id="wizard-end-hz", label="End (Hz)",
                                value=signal.end_hz, unit="Hz",
                                minimum=1.0, maximum=96000.0,
                                commit_action="wizard_end_hz",
                            ),
                            ui.number_input(
                                id="wizard-rate", label="Sample rate (Hz)",
                                value=signal.sample_rate, unit="Hz",
                                minimum=8000, maximum=192000, step=1,
                                commit_action="wizard_rate",
                            ),
                        ],
                        gap=12.0,
                    ),
                ],
                title="Signal configuration",
                width=_CARD_WIDTH,
            ),
            _takes_table(model.takes) if model.takes else ui.empty_state(
                "No takes",
                description="Pass --channels L,R,... to plan takes.",
            ),
            ui.hstack(
                [
                    ui.button("Record all pending", id="capture-start",
                              action="capture_start"),
                    ui.button("Cancel", id="capture-cancel",
                              action="capture_cancel"),
                ],
                gap=12.0,
            ),
            ui.progress(model.overall_progress, label="Overall progress",
                        id="capture-progress"),
        ],
        gap=16.0,
    )


def _results_table(results: list[dict]) -> ui.Node:
    rows = [
        [str(item.get("label", "?")),
         "ok" if item.get("ok") else "failed",
         str(item.get("message", ""))]
        for item in results
    ]
    return ui.table(["Channel", "State", "Message"], rows)


def _probe_section(model: WizardModel) -> ui.Node:
    return ui.vstack(
        [
            _stepper(3),
            ui.section_header(
                "Probe",
                "Tone-burst arrival-time probe per channel.",
            ),
            _status_row("probe-status", model.status),
            ui.button("Run probe", id="probe-start", action="probe_start"),
            _results_table(model.probe_results)
            if model.probe_results else ui.empty_state(
                "No probe results yet",
                description="Run the probe to fill this table.",
            ),
        ],
        gap=16.0,
    )


def _bass_section(model: WizardModel) -> ui.Node:
    return ui.vstack(
        [
            _stepper(4),
            ui.section_header(
                "Bass anchor",
                "Low-frequency tone burst for the first-bin phase anchor.",
            ),
            _status_row("bass-status", model.status),
            ui.button("Run bass anchor", id="bass-start",
                      action="bass_start"),
            _results_table(model.bass_results)
            if model.bass_results else ui.empty_state(
                "No bass-anchor results yet",
                description="Run the bass anchor to fill this table.",
            ),
        ],
        gap=16.0,
    )


def _eval_chart(
    label: str, curve: dict, resources: list[tkdata.Dataset]
) -> ui.Node | None:
    freq = curve.get("freq") or []
    spl = curve.get("spl") or []
    points = [
        (f, s) for f, s in zip(freq, spl)
        if isinstance(f, (int, float)) and isinstance(s, (int, float))
        and f > 0.0
    ]
    if not points:
        return None
    ds_id = f"ds-eval-{label}".replace(" ", "_")
    dataset = tkdata.Dataset.from_mapping(
        {
            "frequency": [f for f, _ in points],
            "level": [s for _, s in points],
            "series": ["Measured"] * len(points),
            "color": ["#38bdf8"] * len(points),
        },
        id=ds_id,
    )
    resources.append(dataset)
    return (
        px.line(f"chart-eval-{label}".replace(" ", "_")).data(dataset)
        .x("frequency").y("level").series("series").color("color")
        .title(f"{label} — measured")
        .x_log().x_label("Frequency (Hz)").y_label("SPL (dB)")
        .x_range(20.0, 20_000.0)
        .legend_position(px.LegendPosition.BOTTOM)
    )


def _evaluating_section(
    model: WizardModel,
    eval_curves: dict[str, dict],
    resources: list[tkdata.Dataset],
) -> ui.Node:
    children: list[ui.Node] = [
        _stepper(5),
        ui.section_header(
            "Evaluating",
            "Review measured takes (pass --curves to preload take CSVs).",
        ),
    ]
    charts = [
        chart for label, curve in sorted(eval_curves.items())
        if (chart := _eval_chart(label, curve, resources)) is not None
    ]
    if charts:
        children.extend(
            ui.card([chart], width=_CARD_WIDTH) for chart in charts
        )
    else:
        children.append(ui.empty_state(
            "No curves loaded",
            description="Capture takes, then reopen with --curves, "
            "or pick take CSVs below.",
        ))
    rows = [
        [t.label, t.state, t.wav_path or "-", t.csv_path or "-"]
        for t in model.takes
    ]
    if rows:
        children.append(ui.table(
            ["Take", "State", "WAV", "CSV"], rows, id="eval-takes",
        ))
    return ui.vstack(children, gap=16.0)


def _saving_section(model: WizardModel) -> ui.Node:
    return ui.vstack(
        [
            _stepper(6),
            ui.section_header(
                "Saving",
                "Session name and room metadata (full bundle assembly "
                "stays in the sotf frontends).",
            ),
            _status_row("save-status", model.status),
            ui.card(
                [
                    ui.text_input(
                        id="wizard-save-name", label="Session name",
                        value=model.save_name,
                        commit_action="wizard_save_name",
                    ),
                    ui.hstack(
                        [
                            ui.number_input(
                                id="wizard-room-w", label="Width (m)",
                                value=model.room_width_m, unit="m",
                                minimum=0.0, commit_action="wizard_room_w",
                            ),
                            ui.number_input(
                                id="wizard-room-d", label="Depth (m)",
                                value=model.room_depth_m, unit="m",
                                minimum=0.0, commit_action="wizard_room_d",
                            ),
                            ui.number_input(
                                id="wizard-room-h", label="Height (m)",
                                value=model.room_height_m, unit="m",
                                minimum=0.0, commit_action="wizard_room_h",
                            ),
                            ui.select(
                                id="wizard-room-unit", label="Unit",
                                value=model.room_unit,
                                options=[("metric", "Metric (m)"),
                                         ("imperial", "Imperial (ft)")],
                                action="wizard_room_unit",
                            ),
                        ],
                        gap=12.0,
                    ),
                    ui.text_input(
                        id="wizard-description", label="Setup description",
                        value=model.setup_description,
                        commit_action="wizard_description",
                    ),
                    ui.button("Save session JSON", id="wizard-save",
                              action="wizard_save"),
                ],
                title="Session metadata",
                width=_CARD_WIDTH,
            ),
        ],
        gap=16.0,
    )


def _set(node_id: str, prop: str, value: object) -> dict:
    return {"op": "set", "id": node_id, "property": prop, "value": value}


@dataclass
class WizardApp(App):
    """Recording wizard with backend-driving action handlers."""

    model: WizardModel = field(default_factory=WizardModel)
    backend: CaptureBackend | None = None
    eval_curves: dict[str, dict] = field(default_factory=dict)

    # -- generic control plumbing --------------------------------------
    def _patch_status(self, context: SessionContext, event_id: str,
                      node_id: str, message: str) -> None:
        self.model.status = message
        context.patch([_set(node_id, "value", message)],
                      request_id=event_id)

    def _select_value(self, event: Event) -> str | None:
        value = (event.payload or {}).get("value")
        return None if value is None else str(value)

    def _commit_float(self, event: Event) -> float | None:
        value = (event.payload or {}).get("value")
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result

    def _on_select(self, event: Event, context: SessionContext,
                   choices: list[str], apply, status_id: str) -> None:
        value = self._select_value(event)
        if value not in choices:
            context.error(event.id, "invalid_choice",
                          f"invalid choice: {value!r}")
            return
        apply(value)
        context.acknowledge(event)
        self._patch_status(context, event.id, status_id,
                           f"selected {value}")

    def _on_commit_float(self, event: Event, context: SessionContext,
                         apply, status_id: str, label: str,
                         minimum: float | None = None,
                         maximum: float | None = None) -> None:
        value = self._commit_float(event)
        if value is None:
            context.error(event.id, "invalid_number",
                          f"invalid number for {label}")
            return
        if minimum is not None and value < minimum:
            context.error(event.id, "out_of_range",
                          f"{label} below minimum {minimum}")
            return
        if maximum is not None and value > maximum:
            context.error(event.id, "out_of_range",
                          f"{label} above maximum {maximum}")
            return
        apply(value)
        context.acknowledge(event)
        self._patch_status(context, event.id, status_id,
                           f"{label} = {value:g}")

    def _on_commit_text(self, event: Event, context: SessionContext,
                        apply, status_id: str, label: str) -> None:
        value = self._select_value(event)
        if value is None:
            context.error(event.id, "invalid_text",
                          f"invalid text for {label}")
            return
        apply(value)
        context.acknowledge(event)
        self._patch_status(context, event.id, status_id,
                           f"{label} set")

    # -- capture --------------------------------------------------------
    def _take_request(self, take: TakeState, signal: str,
                      freq: float | None, duration: float,
                      name: str) -> TakeRequest:
        model = self.model
        device = None
        if model.input_name and model.input_name == model.output_name:
            # The CLI addresses a single device; distinct input/output
            # pairs need an aggregate device (CLI limitation).
            device = model.input_name
        return TakeRequest(
            signal=signal,
            duration_s=duration,
            sample_rate=model.signal.sample_rate,
            out_channels=(take.out_channel,),
            in_channels=(take.in_channel,),
            name=name,
            output_dir=model.output_dir,
            device=device,
            start_hz=model.signal.start_hz,
            end_hz=model.signal.end_hz,
            amplitude=model.signal.amplitude(),
            freq=freq,
            mic_calibration=model.mic_calibration_path,
        )

    def _run_one_take(self, take: TakeState, signal: str,
                      freq: float | None, duration: float,
                      prefix: str) -> None:
        assert self.backend is not None
        take.state = TAKE_RECORDING
        try:
            result = self.backend.capture(self._take_request(
                take, signal, freq, duration,
                f"{self.model.save_name}-{prefix}-{take.label}",
            ))
        except Exception as error:  # backend failure is a take failure
            take.state = TAKE_FAILED
            take.message = str(error)
            return
        take.progress = 1.0
        take.message = result.message
        if result.ok:
            take.state = TAKE_DONE
            take.wav_path = result.wav_paths[0] if result.wav_paths else None
            take.csv_path = result.csv_paths[0] if result.csv_paths else None
            take.summary = result.summaries[0] if result.summaries else None
        else:
            take.state = TAKE_FAILED

    def _on_capture_start(self, event: Event,
                          context: SessionContext) -> None:
        model = self.model
        if self.backend is None:
            context.error(event.id, "no_backend", "no capture backend")
            return
        if model.busy:
            context.acknowledge(event)
            self._patch_status(context, event.id, "capture-status",
                               "capture already running")
            return
        pending = model.pending_takes
        if not pending:
            context.acknowledge(event)
            self._patch_status(context, event.id, "capture-status",
                               "no pending takes")
            return
        context.acknowledge(event)
        model.busy = True
        model.cancel_requested = False
        total = len(pending)
        for index, take in enumerate(pending):
            if model.cancel_requested:
                take.message = "cancelled"
                continue
            self._patch_status(
                context, event.id, "capture-status",
                f"take {index + 1}/{total}: {take.label}")
            self._run_one_take(take, model.signal.signal_type, None,
                               model.signal.duration_s, "take")
            context.patch(
                [_set("capture-progress", "value",
                      model.overall_progress)],
                request_id=event.id,
            )
        model.busy = False
        done = len(model.done_takes)
        cancelled = model.cancel_requested
        self._patch_status(
            context, event.id, "capture-status",
            f"{done}/{len(model.takes)} takes done"
            + (" (cancelled)" if cancelled else ""))

    def _on_capture_cancel(self, event: Event,
                           context: SessionContext) -> None:
        self.model.cancel_requested = True
        context.acknowledge(event)
        self._patch_status(context, event.id, "capture-status",
                           "cancel requested (applies between takes)")

    # -- spl / probe / bass ---------------------------------------------
    def _on_spl_start(self, event: Event, context: SessionContext) -> None:
        model = self.model
        if self.backend is None or model.busy:
            context.acknowledge(event)
            self._patch_status(context, event.id, "spl-status",
                               "busy or no backend")
            return
        context.acknowledge(event)
        model.busy = True
        try:
            probe = TakeState(key="spl", label="spl", channel="spl",
                              out_channel=0, in_channel=0)
            self._run_one_take(probe, "tone", 1000.0, 2.0, "spl")
            model.spl_result = {
                "ok": probe.state == TAKE_DONE,
                "message": probe.message,
                "wav": probe.wav_path,
                "csv": probe.csv_path,
            }
            context.patch(
                [_set("spl-result", "value", probe.message)],
                request_id=event.id,
            )
            self._patch_status(context, event.id, "spl-status",
                               "SPL calibration done"
                               if probe.state == TAKE_DONE
                               else f"SPL failed: {probe.message}")
        finally:
            model.busy = False

    def _on_step_runner(self, event: Event, context: SessionContext,
                        kind: str, signal: str, freq: float,
                        duration: float, store: list[dict],
                        status_id: str) -> None:
        model = self.model
        if self.backend is None or model.busy:
            context.acknowledge(event)
            self._patch_status(context, event.id, status_id,
                               "busy or no backend")
            return
        if not model.takes:
            context.acknowledge(event)
            self._patch_status(context, event.id, status_id, "no takes")
            return
        context.acknowledge(event)
        model.busy = True
        try:
            for take in model.takes:
                if model.cancel_requested:
                    break
                worker = TakeState(key=f"{kind}-{take.key}",
                                   label=take.label, channel=take.channel,
                                   out_channel=take.out_channel,
                                   in_channel=take.in_channel)
                self._run_one_take(worker, signal, freq, duration, kind)
                store.append({
                    "label": take.label,
                    "ok": worker.state == TAKE_DONE,
                    "message": worker.message,
                    "wav": worker.wav_path,
                    "csv": worker.csv_path,
                })
                self._patch_status(context, event.id, status_id,
                                   f"{kind}: {len(store)}/{len(model.takes)}")
        finally:
            model.busy = False

    def _on_save(self, event: Event, context: SessionContext) -> None:
        model = self.model
        if self.backend is None:
            context.error(event.id, "no_backend", "no capture backend")
            return
        if not model.metadata_is_valid():
            context.acknowledge(event)
            self._patch_status(context, event.id, "save-status",
                               "room dimensions must be all zero or all "
                               "positive")
            return
        context.acknowledge(event)
        import os

        directory = model.output_dir or "."
        path = os.path.join(directory, f"{model.save_name}.json")
        try:
            written = self.backend.save_session(path, model.session_payload())
        except Exception as error:
            self._patch_status(context, event.id, "save-status",
                               f"save failed: {error}")
            return
        self._patch_status(context, event.id, "save-status",
                           f"saved {written}")

    # -- dispatcher -------------------------------------------------------
    def on_action(self, event: Event, context: SessionContext) -> None:
        action = event.action or ""
        model = self.model
        outputs = [d.name for d in model.output_devices]
        inputs = [d.name for d in model.input_devices]
        if action == "capture_start":
            self._on_capture_start(event, context)
        elif action == "capture_cancel":
            self._on_capture_cancel(event, context)
        elif action == "spl_start":
            self._on_spl_start(event, context)
        elif action == "probe_start":
            self._on_step_runner(event, context, "probe", "tone", 2000.0,
                                 0.5, model.probe_results, "probe-status")
        elif action == "bass_start":
            self._on_step_runner(event, context, "bass", "tone", 50.0, 1.0,
                                 model.bass_results, "bass-status")
        elif action == "wizard_save":
            self._on_save(event, context)
        elif action == "wizard_output_device":
            self._on_select(event, context, outputs,
                            lambda v: setattr(model, "output_name", v),
                            "config-status")
        elif action == "wizard_input_device":
            self._on_select(event, context, inputs,
                            lambda v: setattr(model, "input_name", v),
                            "config-status")
        elif action == "wizard_signal_type":
            self._on_select(event, context, list(SIGNAL_OPTIONS),
                            lambda v: setattr(model.signal, "signal_type", v),
                            "capture-status")
        elif action == "wizard_room_unit":
            self._on_select(event, context, ["metric", "imperial"],
                            lambda v: setattr(model, "room_unit", v),
                            "save-status")
        elif action == "wizard_duration":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "duration_s", v), "capture-status",
                "duration", 0.1, 60.0)
        elif action == "wizard_level":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "level_db", v), "capture-status",
                "level", -60.0, 0.0)
        elif action == "wizard_start_hz":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "start_hz", v), "capture-status",
                "start_hz", 1.0, 24000.0)
        elif action == "wizard_end_hz":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "end_hz", v), "capture-status",
                "end_hz", 1.0, 96000.0)
        elif action == "wizard_rate":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "sample_rate", int(v)), "capture-status",
                "sample_rate", 8000, 192000)
        elif action == "wizard_sweeps":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "num_sweeps", int(v)), "config-status",
                "sweeps", 1, 8)
        elif action == "wizard_pre_silence":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "pre_silence_s", v), "config-status",
                "pre_silence", 0.0, 10.0)
        elif action == "wizard_bass_octave":
            self._on_commit_float(event, context, lambda v: setattr(
                model.signal, "bass_octave_s", v), "config-status",
                "bass_octave", 0.5, 10.0)
        elif action == "wizard_room_w":
            self._on_commit_float(event, context, lambda v: setattr(
                model, "room_width_m", v), "save-status", "width",
                0.0, None)
        elif action == "wizard_room_d":
            self._on_commit_float(event, context, lambda v: setattr(
                model, "room_depth_m", v), "save-status", "depth",
                0.0, None)
        elif action == "wizard_room_h":
            self._on_commit_float(event, context, lambda v: setattr(
                model, "room_height_m", v), "save-status", "height",
                0.0, None)
        elif action == "wizard_save_name":
            self._on_commit_text(event, context, lambda v: setattr(
                model, "save_name", v), "save-status", "save_name")
        elif action == "wizard_description":
            self._on_commit_text(event, context, lambda v: setattr(
                model, "setup_description", v), "save-status",
                "description")
        elif action == "wizard_mic_cal":
            self._on_commit_text(event, context, lambda v: setattr(
                model, "mic_calibration_path", v or None), "config-status",
                "mic_calibration")
        elif action == "wizard_output_dir":
            self._on_commit_text(event, context, lambda v: setattr(
                model, "output_dir", v or None), "config-status",
                "output_dir")
        else:
            # The host also delivers synthesized events the app never
            # declared (dropdown open/focus, shell chrome). They carry no
            # app action, so acknowledge and ignore them instead of
            # surfacing a user-facing error; log to stderr for debugging.
            print(f"recording-gui: ignoring unhandled action {action!r} "
                  f"(node {event.node_id}, event {event.event})",
                  file=sys.stderr)
            context.acknowledge(event)

    def run(self) -> None:
        # The native host otherwise prefers its own repository venv. A
        # console-script installation lives in this interpreter's
        # site-packages, so ensure the supervised child uses the same one.
        os.environ.setdefault("GPUI_PYTHON", sys.executable)
        super().run()


def build_app(
    model: WizardModel,
    backend: CaptureBackend,
    eval_curves: dict[str, dict] | None = None,
) -> WizardApp:
    """Build the recording wizard application."""
    resources: list[tkdata.Dataset] = []
    curves = eval_curves or {}
    app = WizardApp(
        title="Recording wizard",
        sidebar_title="Recording",
        sidebar_subtitle="Python declarations, Rust renderers",
        width=_APP_WIDTH,
        height=_APP_HEIGHT,
        sections=[
            section("config", "Devices & setup",
                    _config_section(model, backend.name)),
            section("spl", "SPL calibration", _spl_section(model)),
            section("capture", "Capture", _capture_section(model)),
            section("probe", "Probe", _probe_section(model)),
            section("bass", "Bass anchor", _bass_section(model)),
            section("evaluating", "Evaluating",
                    _evaluating_section(model, curves, resources)),
            section("saving", "Saving", _saving_section(model)),
        ],
        model=model,
        backend=backend,
        eval_curves=dict(curves),
        miniapp=MiniAppConfig(
            title="Recording wizard",
            app_name="Recording wizard",
            width=_APP_WIDTH,
            height=_APP_HEIGHT,
            with_theme=True,
            initial_theme="dark",
        ),
    )
    app.resources = tuple(resources)
    return app
