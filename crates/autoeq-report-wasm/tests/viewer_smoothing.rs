use autoeq_report_wasm::{schema::Figure, viewer_smoothing::ViewerSmoothing};
use serde_json::json;

#[test]
fn smoothing_matches_the_previous_browser_curves() {
    let original: Figure = serde_json::from_value(json!({
        "x": { "scale": "log" },
        "y": {},
        "series": [
            {
                "name": "Response",
                "x": [90, 97, 98, 100, 103, 120, 135, 990, 1020],
                "y": [-20, -8, 0, -4, null, -11, -16, -8, 3]
            },
            { "name": "Marker", "x": [100, 100], "y": [-20, 0] }
        ]
    }))
    .unwrap();

    // The nine-point fixture was evaluated with the previous browser viewer.
    // Index four is a gap, so each row has eight numeric samples.
    let expected = [
        [-20.0, -8.0, 0.0, -4.0, -11.0, -16.0, -8.0, 3.0],
        [-20.0, -8.0, 0.0, -4.0, -11.0, -16.0, -8.0, 3.0],
        [-20.0, -4.0, -4.0, -4.0, -11.0, -16.0, -8.0, 3.0],
        [-20.0, -4.0, -4.0, -2.0, -11.0, -16.0, -8.0, 3.0],
        [-20.0, -4.0, -4.0, -4.0, -11.0, -16.0, -2.5, -2.5],
        [-8.0, -8.0, -8.0, -8.0, -11.0, -16.0, -2.5, -2.5],
        [-8.0, -8.0, -8.0, -8.0, -13.5, -13.5, -2.5, -2.5],
        [
            -8.6,
            -59.0 / 6.0,
            -59.0 / 6.0,
            -59.0 / 6.0,
            -59.0 / 6.0,
            -7.8,
            -2.5,
            -2.5,
        ],
        [-20.0, -8.0, 0.0, -4.0, -11.0, -16.0, -2.5, -2.5],
        [
            -3.220268863679,
            -3.220268863679,
            -3.220268863679,
            -3.220268863679,
            -11.0,
            -16.0,
            1.057236085174,
            1.057236085174,
        ],
        [-8.0, -8.0, -8.0, -8.0, -13.5, -13.5, -2.5, -2.5],
    ];
    let mut displayed = original.clone();
    displayed.x.min = Some(95.0);
    displayed.series[0].visible = false;

    for ((mode, id, _), expected) in ViewerSmoothing::OPTIONS.into_iter().zip(expected) {
        mode.apply(&original, &mut displayed);
        let response = &displayed.series[0];
        assert_eq!(response.y[4], None, "{id}");
        assert_eq!(response.y.iter().flatten().count(), expected.len(), "{id}");
        for (actual, expected) in response.y.iter().flatten().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-10,
                "{id}: {actual} != {expected}"
            );
        }
        assert_eq!(displayed.series[1].y, original.series[1].y);
        assert_eq!(displayed.x.min, Some(95.0));
        assert!(!response.visible);
    }

    ViewerSmoothing::None.apply(&original, &mut displayed);
    assert_eq!(displayed.series[0].y, original.series[0].y);
}
