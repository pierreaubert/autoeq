//! Typed parser for Equalizer APO filter output used by integration tests.

#[derive(Debug, Clone, PartialEq)]
pub struct ApoFilter {
    pub index: usize,
    pub kind: String,
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
}

pub fn parse_apo_filters(content: &str) -> Result<Vec<ApoFilter>, String> {
    content
        .lines()
        .filter(|line| line.trim_start().starts_with("Filter"))
        .map(parse_filter_line)
        .collect()
}

fn parse_filter_line(line: &str) -> Result<ApoFilter, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let value_after = |label: &str| -> Result<&str, String> {
        let index = fields
            .iter()
            .position(|field| *field == label)
            .ok_or_else(|| format!("missing {label} in APO filter line: {line}"))?;
        fields
            .get(index + 1)
            .copied()
            .ok_or_else(|| format!("missing value after {label} in APO filter line: {line}"))
    };

    if fields.get(2) != Some(&"ON") {
        return Err(format!("filter is not enabled in APO line: {line}"));
    }

    Ok(ApoFilter {
        index: fields
            .get(1)
            .ok_or_else(|| format!("missing filter index: {line}"))?
            .trim_end_matches(':')
            .parse()
            .map_err(|error| format!("invalid filter index in '{line}': {error}"))?,
        kind: fields
            .get(3)
            .ok_or_else(|| format!("missing filter kind: {line}"))?
            .to_string(),
        freq_hz: value_after("Fc")?
            .parse()
            .map_err(|error| format!("invalid frequency in '{line}': {error}"))?,
        gain_db: value_after("Gain")?
            .parse()
            .map_err(|error| format!("invalid gain in '{line}': {error}"))?,
        q: value_after("Q")?
            .parse()
            .map_err(|error| format!("invalid Q in '{line}': {error}"))?,
    })
}
