//! Deterministic constrained pairwise parameter matrix for pull-request QA.

use std::collections::BTreeSet;

pub const MAX_PR_ROWS: usize = 24;
const DIMENSION_CARDINALITIES: [usize; 9] = [3, 3, 3, 3, 3, 3, 2, 3, 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ParameterRow {
    pub topology: u8,
    pub mode: u8,
    pub sample_rate: u8,
    pub filter_count: u8,
    pub grid_size: u8,
    pub measurement_shape: u8,
    pub phase: u8,
    pub crossover: u8,
    pub fir_duration: u8,
}

impl ParameterRow {
    const WIDTH: usize = 9;

    fn value(self, dimension: usize) -> u8 {
        [
            self.topology,
            self.mode,
            self.sample_rate,
            self.filter_count,
            self.grid_size,
            self.measurement_shape,
            self.phase,
            self.crossover,
            self.fir_duration,
        ][dimension]
    }

    fn from_values(values: [u8; Self::WIDTH]) -> Self {
        Self {
            topology: values[0],
            mode: values[1],
            sample_rate: values[2],
            filter_count: values[3],
            grid_size: values[4],
            measurement_shape: values[5],
            phase: values[6],
            crossover: values[7],
            fir_duration: values[8],
        }
    }

    #[cfg(test)]
    fn values(self) -> [u8; Self::WIDTH] {
        [
            self.topology,
            self.mode,
            self.sample_rate,
            self.filter_count,
            self.grid_size,
            self.measurement_shape,
            self.phase,
            self.crossover,
            self.fir_duration,
        ]
    }
}

fn applicable(row: ParameterRow) -> bool {
    // FIR duration is irrelevant to IIR, but retaining a canonical value
    // makes the matrix rectangular and keeps pair accounting mechanical.
    // Automatic crossover is only meaningful for redirected/sub topologies.
    if row.crossover == 1 && row.topology == 0 {
        return false;
    }
    true
}

fn pair_key(a: usize, av: u8, b: usize, bv: u8) -> (usize, u8, usize, u8) {
    (a, av, b, bv)
}

fn applicable_candidates() -> Vec<ParameterRow> {
    let candidate_count = DIMENSION_CARDINALITIES.iter().product();
    (0..candidate_count)
        .filter_map(|encoded| {
            let mut remaining = encoded;
            let mut values = [0_u8; ParameterRow::WIDTH];
            for (index, value) in values.iter_mut().enumerate() {
                let cardinality = DIMENSION_CARDINALITIES[index];
                *value = (remaining % cardinality) as u8;
                remaining /= cardinality;
            }
            let row = ParameterRow::from_values(values);
            applicable(row).then_some(row)
        })
        .collect()
}

/// Generate at most 24 rows with a fixed-seed greedy covering algorithm.
pub fn generate_pr_matrix() -> Vec<ParameterRow> {
    let mut candidates = applicable_candidates();

    let mut uncovered = BTreeSet::new();
    for a in 0..ParameterRow::WIDTH {
        for b in (a + 1)..ParameterRow::WIDTH {
            for candidate in &candidates {
                uncovered.insert(pair_key(a, candidate.value(a), b, candidate.value(b)));
            }
        }
    }

    let mut rows = Vec::new();
    while !uncovered.is_empty() && rows.len() < MAX_PR_ROWS {
        let selected = candidates
            .iter()
            .max_by_key(|candidate| {
                (0..ParameterRow::WIDTH)
                    .flat_map(|a| ((a + 1)..ParameterRow::WIDTH).map(move |b| (a, b)))
                    .filter(|(a, b)| {
                        uncovered.contains(&pair_key(
                            *a,
                            candidate.value(*a),
                            *b,
                            candidate.value(*b),
                        ))
                    })
                    .count()
            })
            .copied()
            .expect("parameter matrix has candidates");
        for a in 0..ParameterRow::WIDTH {
            for b in (a + 1)..ParameterRow::WIDTH {
                uncovered.remove(&pair_key(a, selected.value(a), b, selected.value(b)));
            }
        }
        rows.push(selected);
        candidates.retain(|candidate| *candidate != selected);
    }
    assert!(
        uncovered.is_empty(),
        "pairwise matrix exceeds {MAX_PR_ROWS} rows"
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_is_bounded_and_covers_all_valid_pairs() {
        let rows = generate_pr_matrix();
        assert!(!rows.is_empty());
        assert!(rows.len() <= MAX_PR_ROWS);
        let candidates = applicable_candidates();
        for a in 0..ParameterRow::WIDTH {
            for b in (a + 1)..ParameterRow::WIDTH {
                for candidate in &candidates {
                    assert!(
                        rows.iter().any(|row| row.value(a) == candidate.value(a)
                            && row.value(b) == candidate.value(b)),
                        "uncovered valid pair"
                    );
                }
            }
        }
    }

    #[test]
    fn checked_in_pr_manifest_exactly_matches_generated_rows_and_domains() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../qa/registry/parameter-matrix-pr.json"
        ))
        .expect("checked-in pairwise manifest should parse");
        let dimensions = manifest["dimensions"]
            .as_object()
            .expect("manifest dimensions should be an object");
        assert_eq!(
            dimensions["phase"],
            serde_json::json!(["absent", "coherent"]),
            "phase values must map to the supported no-phase and coherent cases"
        );
        assert_eq!(
            dimensions["crossover"],
            serde_json::json!(["fixed_lr24", "automatic_lr24", "fixed_lr48"]),
            "crossover values must map to distinct runtime policies"
        );
        let dimension_order = [
            "topology",
            "mode",
            "sample_rate",
            "filter_count",
            "grid_size",
            "measurement_shape",
            "phase",
            "crossover",
            "fir_duration_ms",
        ];
        for (index, name) in dimension_order.iter().enumerate() {
            let values = dimensions[*name]
                .as_array()
                .unwrap_or_else(|| panic!("manifest dimension {name} should be an array"));
            assert_eq!(
                values.len(),
                DIMENSION_CARDINALITIES[index],
                "manifest dimension {name} disagrees with the executable axis"
            );
            let unique: BTreeSet<_> = values
                .iter()
                .map(|value| serde_json::to_string(value).unwrap())
                .collect();
            assert_eq!(
                unique.len(),
                values.len(),
                "duplicate semantic value in {name}"
            );
        }

        let checked_rows: Vec<[u8; ParameterRow::WIDTH]> =
            serde_json::from_value(manifest["rows"].clone())
                .expect("manifest rows should be fixed-width integer vectors");
        let generated_rows: Vec<_> = generate_pr_matrix()
            .into_iter()
            .map(ParameterRow::values)
            .collect();
        assert_eq!(checked_rows, generated_rows);
    }
}
