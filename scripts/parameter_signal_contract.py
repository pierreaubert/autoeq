"""Check matrix signal-axis evidence; not a usefulness or backend-render gate."""


def verify_signal_axes(row):
    axes = row["requested_axes"]
    expected_rate = [44100, 48000, 96000][axes["sample_rate"]]
    assert row["sample_rate_hz"] == expected_rate, "requested/runtime rate mismatch"
    measurements = row["effective_config"]["measurements"]
    assert measurements, "missing measurement evidence"
    for measurement in measurements.values():
        assert measurement["has_phase"] == (axes["phase"] == 1), "phase axis mismatch"
    rates = row["delivered_biquad_sample_rates_hz"]
    system = row["effective_config"].get("system")
    roles = system["speakers"] if system else {name: name for name in measurements}
    sub_outputs = system.get("subwoofers", {}).get("outputs", []) if system else []
    expected_measurements = set(roles.values()) | {output["speaker"] for output in sub_outputs}
    expected_channels = set(roles) | {output["id"] for output in sub_outputs}
    assert expected_measurements == set(measurements), "missing role measurement evidence"
    assert set(rates) == expected_channels, "missing delivered channel rate evidence"
    count = 0
    for channel_rates in rates.values():
        for rate in channel_rates:
            assert rate == expected_rate, "delivered biquad design rate mismatch"
            count += 1
    # An empty/reverted chain cannot prove a filter's actual design rate.
    return "biquad_design_rate_verified" if count else "no_delivered_biquad"
