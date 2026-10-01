"""Display saved plugin order and explicit route ownership; never infer an EQ plan."""

import json
from html import escape

from wasm_report import html_section, sankey_chart


def _plugin_name(plugin):
    kind = str(plugin.get("plugin_type", "unknown"))
    return {"eq": "EQ", "gain": "Gain", "delay": "Delay",
            "crossover": "Crossover", "limiter": "Limiter",
            "convolution": "FIR"}.get(kind, kind)


class _Diagram:
    def __init__(self, title, tab):
        self.title, self.tab = title, tab
        self.nodes, self.links, self.rows = [], [], []

    def node(self, label, detail=None):
        index = len(self.nodes)
        # IDs distinguish identical plugins and keep the plotted labels short.
        self.nodes.append(f"{index + 1}: {label}")
        self.rows.append((index + 1, label, detail or {}))
        return index

    def edge(self, source, target):
        self.links.append((source, target, 1.0, "rgba(74,144,217,0.25)"))

    def chain(self, source, plugins):
        for plugin in plugins:
            node = self.node(_plugin_name(plugin), plugin)
            self.edge(source, node)
            source = node
        return source

    def sections(self):
        rows = "".join(
            f"<tr><td>{number}</td><td>{escape(label)}</td><td><details>"
            f"<summary>{escape(str(detail.get('parameters', {}).get('label') or label))}</summary>"
            f"<pre>{escape(json.dumps(detail, indent=2, ensure_ascii=False))}</pre>"
            "</details></td></tr>"
            for number, label, detail in self.rows
        )
        figure = sankey_chart(self.title, self.nodes, self.links, tab=self.tab)
        figure["chart"]["node_boxes"] = True
        # The shared shell permits horizontal scrolling instead of squeezing
        # many serial plugin stages into overlapping labels on a narrow screen.
        depth = [0] * len(self.nodes)
        for source, target, *_ in self.links:
            depth[target] = max(depth[target], depth[source] + 1)
        figure["min_width"] = max(1000, (max(depth, default=0) + 1) * 190)
        # Leave room for parallel branches in the complete-system view.
        layers = [depth.count(level) for level in set(depth)]
        figure["min_height"] = max(320, max(layers, default=1) * 70 + 100)
        return [figure, html_section(
            '<table><thead><tr><th>Node</th><th>Stage</th><th>Saved parameters</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>', tab=self.tab)]


def signal_flow_sections(data):
    """Return all-channel and per-output diagrams using the saved chains and routing metadata.

    Routed ownership follows roomeq-engine/physical_routing.rs: input plugins,
    route gain/crossover/delay, destination sum, output plugins. Redundant
    route-owned plugins are not executed a second time. Unknown ownership is
    reported instead of silently guessing a processing order.
    """
    sections = [html_section(
        '<p>Read left to right. Each numbered node is a saved processing stage; '
        'expand its row below for parameters. Branches join at Σ before output '
        'processing. Link widths show connections, not signal level. '
        'Select All channels for the complete graph, or an output for its inputs.</p>')]
    channels = data.get("channels") or {}
    bass = (data.get("metadata") or {}).get("bass_management") or {}
    routing = bass.get("routing_graph") or {}
    globals_ = data.get("global_plugins") or []
    if not routing.get("routes"):
        for plugin in globals_:
            params = plugin.get("parameters") or {}
            metadata = params.get("metadata") or {}
            if metadata.get("routes"):
                routing = metadata
                break
    routes = routing.get("routes") or []
    if routes and any(not ((p.get("parameters") or {}).get("metadata") or {}).get("routes")
                      for p in globals_):
        return [html_section('<p>Signal flow unavailable: additional global processing '
                             'has no explicit position in the saved routing graph.</p>')]
    if not channels:
        return [html_section("<p>DSP signal flow unavailable: no saved channel chains.</p>")]
    if not routes:
        def add_channel(diagram, name, channel):
            if channel.get("drivers") or globals_:
                raise ValueError("this saved graph has driver or global processing "
                                 "without explicit routing. No connection order was guessed")
            start = diagram.node(f"Input {name}")
            end = diagram.chain(start, channel.get("plugins") or [])
            diagram.edge(end, diagram.node(f"Output {name}"))

        views = [("All channels", list(channels))] + [(name, [name]) for name in channels]
        for tab, outputs in views:
            try:
                diagram = _Diagram(f"Saved DSP: {tab}", tab)
                for name in outputs:
                    add_channel(diagram, name, channels[name])
                sections.extend(diagram.sections())
            except ValueError as error:
                sections.append(html_section(
                    f'<p>Signal flow unavailable: {escape(str(error))}.</p>', tab=tab))
        return sections

    shared_name = routing.get("physical_sub_output") or bass.get("physical_sub_output")
    if not shared_name:
        # Older files omit the parent field but retain one explicit driver
        # container. Resolve only a unique container, never a guessed channel.
        parents = [name for name, chain in channels.items() if chain.get("drivers")]
        if len(parents) == 1:
            shared_name = parents[0]
    shared = channels.get(shared_name) or {}
    drivers = {d["name"]: d for d in shared.get("drivers") or []}

    def plugins(channel, stage):
        selected = []
        for plugin in channel.get("plugins") or []:
            params = plugin.get("parameters") or {}
            owner = params.get("room_eq_stage")
            if owner not in {"pre_route", "post_route", "route_owned"}:
                raise ValueError("a plugin has no explicit pre-route/post-route ownership")
            if owner == stage:
                selected.append(plugin)
        return selected

    def add_destination(diagram, destination, inputs):
        incoming = [r for r in routes if r["destination"] == destination]
        tails = []
        for route in incoming:
            source = route["source_channel"]
            if source not in channels and source.upper() != "LFE":
                raise ValueError(f"input chain {source} is missing")
            if source not in inputs:
                start = diagram.node(f"Input {source}")
                inputs[source] = diagram.chain(start, plugins(channels.get(source) or {}, "pre_route"))
            transfer = [{"plugin_type": "gain", "parameters": {
                "gain_db": route.get("gain_db", 0),
                "invert": route.get("polarity_inverted", False),
                "label": f"Route {source} → {destination}"}}]
            high = route.get("high_pass_hz")
            low = route.get("low_pass_hz")
            if high is not None or low is not None:
                transfer.append({"plugin_type": "crossover", "parameters": {
                    "type": route.get("crossover_type"), "frequency": high if high is not None else low,
                    "output": "high" if high is not None else "low"}})
            if route.get("delay_ms", 0):
                transfer.append({"plugin_type": "delay", "parameters": {"delay_ms": route["delay_ms"]}})
            tails.append(diagram.chain(inputs[source], transfer))
        junction = diagram.node(f"Σ {destination}" if len(tails) > 1 else f"Bus {destination}")
        for tail in tails:
            diagram.edge(tail, junction)
        if destination in drivers:
            output_plugins = plugins(shared, "post_route")
            for plugin in plugins(drivers[destination], "post_route"):
                kind, params = plugin.get("plugin_type"), plugin.get("parameters") or {}
                if kind == "gain" and not params.get("room_eq_correction_gain"):
                    continue
                if kind == "delay" and not params.get("room_eq_correction_delay"):
                    continue
                output_plugins.append(plugin)
        else:
            if destination not in channels:
                raise ValueError("the physical output chain is missing")
            output_plugins = plugins(channels[destination], "post_route")
        end = diagram.chain(junction, output_plugins)
        diagram.edge(end, diagram.node(f"Output {destination}"))

    destinations = list(dict.fromkeys(r["destination"] for r in routes))
    views = [("All channels", destinations)] + [(name, [name]) for name in destinations]
    for tab, outputs in views:
        try:
            diagram = _Diagram(f"Saved DSP: {tab}", tab)
            inputs = {}
            for destination in outputs:
                add_destination(diagram, destination, inputs)
            sections.extend(diagram.sections())
        except (KeyError, ValueError, TypeError) as error:
            sections.append(html_section(
                f'<p>Signal flow unavailable: {escape(str(error))}. '
                'Regenerate this legacy output with explicit routing ownership.</p>', tab=tab))
    return sections
