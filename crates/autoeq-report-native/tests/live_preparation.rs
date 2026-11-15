use autoeq_report_native::prepare_live_report;
use serde_json::json;

#[test]
fn live_preparation_preserves_embedded_target_and_result_provenance() {
    let target = json!({"freq":[40.0, 1000.0], "spl":[3.0, 0.0]});
    let mut graph = json!({
        "version":"fixture-version",
        "channels":{},
        "metadata":{"timestamp":"fixture-time", "target_curve":target},
        "future_graph_field":{"retained":true}
    });
    let prepared = prepare_live_report(&graph, None).unwrap();
    assert_eq!(prepared.graph, graph);
    assert_eq!(
        prepared.provenance,
        Some(json!({
            "versions":["fixture-version"], "timestamps":["fixture-time"]
        }))
    );
    assert_eq!(
        prepared.target_snapshot,
        json!({
            "schema_version":1,
            "state":"available",
            "resolution":"embedded_result",
            "target":target
        })
    );

    graph["metadata"]
        .as_object_mut()
        .unwrap()
        .remove("target_curve");
    let unavailable = prepare_live_report(&graph, None).unwrap();
    assert_eq!(
        unavailable.target_snapshot,
        json!({
            "schema_version":1, "state":"not_configured", "target":null
        })
    );
    assert_eq!(unavailable.graph, graph);
}
