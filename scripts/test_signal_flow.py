import unittest

from scripts.src.signal_flow import signal_flow_sections


def plugin(kind, **params):
    return {"plugin_type": kind, "parameters": params}


class SignalFlowTests(unittest.TestCase):
    def charts(self, data):
        return [s["chart"] for s in signal_flow_sections(data)
                if s["kind"] == "sankey" and s["tab"] != "All channels"]

    def test_all_channels_serial_overview_and_output_tabs(self):
        data = {"channels": {name: {"plugins": [plugin("eq")]}
                             for name in ["L", "R"]}}
        figures = [s for s in signal_flow_sections(data) if s["kind"] == "sankey"]
        self.assertEqual([s["tab"] for s in figures], ["All channels", "L", "R"])
        names = [n.split(": ", 1)[1] for n in figures[0]["chart"]["nodes"]]
        self.assertEqual(names, ["Input L", "EQ", "Output L", "Input R", "EQ", "Output R"])

    def test_all_channels_routed_fanout_reuses_input_processing(self):
        data = {"channels": {
            "L": {"plugins": [plugin("eq", room_eq_stage="pre_route")]},
            "R": {"plugins": []}, "Sub": {"plugins": []}},
            "metadata": {"bass_management": {"routing_graph": {"routes": [
                {"source_channel": "L", "destination": "L"},
                {"source_channel": "R", "destination": "R"},
                {"source_channel": "L", "destination": "Sub"},
                {"source_channel": "R", "destination": "Sub"}]}}}}
        figures = [s for s in signal_flow_sections(data) if s["kind"] == "sankey"]
        self.assertEqual([s["tab"] for s in figures], ["All channels", "L", "R", "Sub"])
        chart = figures[0]["chart"]
        names = [n.split(": ", 1)[1] for n in chart["nodes"]]
        self.assertEqual(names.count("Input L"), 1)
        self.assertEqual(names.count("Input R"), 1)
        self.assertEqual(names.count("EQ"), 1)
        self.assertEqual({n for n in names if n.startswith("Output ")},
                         {"Output L", "Output R", "Output Sub"})
        eq = names.index("EQ")
        self.assertEqual(sum(l["source"] == eq for l in chart["links"]), 2)
        junction = names.index("Σ Sub")
        self.assertEqual(sum(l["target"] == junction for l in chart["links"]), 2)
        self.assertGreaterEqual(figures[0]["min_height"], 320)

    def test_overview_height_grows_for_many_channels(self):
        data = {"channels": {f"Ch{i}": {"plugins": []} for i in range(8)}}
        overview = next(s for s in signal_flow_sections(data) if s["kind"] == "sankey")
        self.assertEqual(overview["min_height"], 660)

    def test_serial_order_and_duplicate_plugins(self):
        data = {"channels": {"L": {"plugins": [plugin("gain", gain_db=-2),
                    plugin("delay", delay_ms=1), plugin("eq"), plugin("eq"), plugin("limiter")]}}}
        chart, = self.charts(data)
        self.assertEqual([n.split(": ", 1)[1] for n in chart["nodes"]],
                         ["Input L", "Gain", "Delay", "EQ", "EQ", "Limiter", "Output L"])
        self.assertEqual([(l["source"], l["target"]) for l in chart["links"]],
                         list(zip(range(6), range(1, 7))))

    def test_sum_then_output_limiter_and_route_owned_not_duplicated(self):
        data = {"channels": {
            "L": {"plugins": [plugin("gain", room_eq_stage="pre_route", gain_db=-1)]},
            "R": {"plugins": [plugin("delay", room_eq_stage="pre_route", delay_ms=2)]},
            "Sub1": {"plugins": [plugin("crossover", room_eq_stage="route_owned"),
                                  plugin("eq", room_eq_stage="post_route"),
                                  plugin("limiter", room_eq_stage="post_route")]},
        }, "metadata": {"bass_management": {"routing_graph": {"routes": [
            {"source_channel": s, "destination": "Sub1", "gain_db": -3,
             "low_pass_hz": 80, "crossover_type": "LR24", "delay_ms": 1}
            for s in ["L", "R"]]}}}}
        chart, = self.charts(data)
        names = chart["nodes"]
        self.assertEqual(sum(n.endswith(": Crossover") for n in names), 2)
        junction = next(i for i, n in enumerate(names) if "Σ Sub1" in n)
        self.assertEqual(sum(l["target"] == junction for l in chart["links"]), 2)
        self.assertEqual([n.split(": ", 1)[1] for n in names[junction + 1:]],
                         ["EQ", "Limiter", "Output Sub1"])

    def test_missing_ownership_is_not_guessed(self):
        data = {"channels": {"L": {"plugins": [plugin("eq")]}},
                "metadata": {"bass_management": {"routing_graph": {
                    "routes": [{"source_channel": "L", "destination": "L"}]}}}}
        self.assertFalse(self.charts(data))
        self.assertIn("ownership", str(signal_flow_sections(data)))

    def test_parameter_html_is_escaped(self):
        data = {"channels": {"L": {"plugins": [plugin("eq", label="<script>bad</script>")]}}}
        sections = signal_flow_sections(data)
        self.assertIn("&lt;script&gt;", sections[-1]["html"])
        self.assertNotIn("<script>", sections[-1]["html"])

    def test_driver_eq_after_shared_eq_without_repeating_route_gain(self):
        data = {"channels": {
            "L": {"plugins": []},
            "Sub1": {"plugins": [plugin("eq", room_eq_stage="post_route")],
                     "drivers": [{"name": "Sub2", "plugins": [
                         plugin("gain", room_eq_stage="post_route", gain_db=-2),
                         plugin("eq", room_eq_stage="post_route"),
                         plugin("limiter", room_eq_stage="post_route")]}]},
        }, "metadata": {"bass_management": {"routing_graph": {"routes": [
            {"source_channel": "L", "destination": "Sub2", "gain_db": -2}]}}}}
        chart, = self.charts(data)
        names = [n.split(": ", 1)[1] for n in chart["nodes"]]
        self.assertEqual(names, ["Input L", "Gain", "Bus Sub2", "EQ", "EQ", "Limiter", "Output Sub2"])
        self.assertTrue(chart["node_boxes"])


if __name__ == "__main__":
    unittest.main()
