//! Strictly verifies the supported Equalizer APO text emitted for product profiles.

// Rust guideline compliant 2026-02-21

use autoeq::iir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_PASS};
use autoeq::workflow::{
    PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER, PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB,
    PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE,
};

const MAX_VERIFIED_TEXT_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct ParsedFilter {
    index: usize,
    kind: String,
    frequency_hz: f64,
    gain_db: Option<f64>,
    q: Option<f64>,
    slope_db_per_octave: Option<u8>,
}

#[derive(Debug)]
pub(super) struct VerifiedApoFilterFields {
    pub(super) kind: String,
    pub(super) frequency_hz: f64,
    pub(super) gain_db: Option<f64>,
    pub(super) q: Option<f64>,
    pub(super) slope_db_per_octave: Option<u8>,
    pub(super) frequency_convention: Option<&'static str>,
}

#[derive(Debug)]
pub(super) struct VerifiedApoText {
    #[cfg(test)]
    pub(super) filters: Vec<Biquad>,
    pub(super) emitted_filters: Vec<VerifiedApoFilterFields>,
    pub(super) preamp_db: f64,
    pub(super) sample_rate_hz: f64,
    pub(super) max_transfer_delta_db: f64,
    pub(super) max_shelf_scaled_coefficient_delta: f64,
    pub(super) max_shelf_source_transfer_delta_db: f64,
}

/// Verifies emitted APO text against the caller-approved serialized filters.
///
/// The parser accepts only the formatter subset checked for profiled output. It
/// does not claim to be an Equalizer APO parser or to verify installation,
/// device selection, channel routing, or runtime behavior. Routing is inherited
/// from the surrounding Equalizer APO configuration. Shelf output uses only
/// LSC/HSC at 12 dB/octave and is checked against equations from the frozen
/// Equalizer APO source revision. This is a local source-derived verification,
/// not a check with the consumer parser or runtime.
pub(super) fn verify_emitted_apo_text(
    contents: &[u8],
    sample_rate_hz: f64,
    expected_filters: &[Biquad],
    expected_preamp_db: f64,
    frequencies_hz: &[f64],
) -> Result<VerifiedApoText, String> {
    if contents.len() > MAX_VERIFIED_TEXT_BYTES {
        return Err("profiled APO text exceeds the 1 MiB verification limit".into());
    }
    if !contents.ends_with(b"\n") || contents.contains(&b'\r') {
        return Err("profiled APO text must use the emitted LF line format".into());
    }
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err("profiled APO verification sample rate must be finite and positive".into());
    }
    if !expected_preamp_db.is_finite() || expected_preamp_db > 0.0 {
        return Err("profiled APO preamp must be finite and non-positive".into());
    }
    if frequencies_hz.is_empty() {
        return Err("profiled APO transfer comparison requires frequencies".into());
    }
    for &frequency_hz in frequencies_hz {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || frequency_hz >= sample_rate_hz / 2.0
        {
            return Err("profiled APO transfer comparison has an invalid frequency".into());
        }
    }

    let text = std::str::from_utf8(contents)
        .map_err(|_| "profiled APO text must be valid UTF-8".to_string())?;
    let mut parsed_preamp = None;
    let mut parsed_filters = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.trim() != line {
            return Err(format!(
                "profiled APO line {} has non-canonical surrounding whitespace",
                line_index + 1
            ));
        }
        if line.starts_with("Preamp:") {
            if parsed_preamp.is_some() || !parsed_filters.is_empty() {
                return Err("profiled APO text must have one preamp before all filters".into());
            }
            parsed_preamp = Some(parse_preamp(line)?);
            continue;
        }
        if line.starts_with("Filter ") {
            if parsed_preamp.is_none() {
                return Err("profiled APO filter appears before its explicit preamp".into());
            }
            let filter = parse_filter(line)?;
            parsed_filters.push(filter);
            continue;
        }
        return Err(format!(
            "profiled APO line {} contains an unsupported command or routing directive",
            line_index + 1
        ));
    }

    let parsed_preamp = parsed_preamp.ok_or_else(|| {
        "profiled APO text must contain one explicit preamp before its filters".to_string()
    })?;
    if parsed_preamp > 0.0 || parsed_preamp != expected_preamp_db {
        return Err("emitted APO preamp does not match the approved profile value".into());
    }
    if parsed_filters.len() != expected_filters.len() {
        return Err("emitted APO filter count does not match the approved profile".into());
    }

    let mut expected = expected_filters.to_vec();
    expected.sort_by(|left, right| left.freq.total_cmp(&right.freq));
    let mut realized_filters = Vec::with_capacity(parsed_filters.len());
    let mut emitted_filters = Vec::with_capacity(parsed_filters.len());
    let mut max_shelf_scaled_coefficient_delta = 0.0_f64;
    let mut max_shelf_source_transfer_delta_db = 0.0_f64;
    let mut previous_frequency_hz = 0.0;
    for (index, (parsed, approved)) in parsed_filters.iter().zip(&expected).enumerate() {
        if parsed.index != index + 1 {
            return Err("emitted APO filter numbers must be contiguous and start at one".into());
        }
        if parsed.frequency_hz < previous_frequency_hz {
            return Err("emitted APO filters are not ordered by ascending frequency".into());
        }
        previous_frequency_hz = parsed.frequency_hz;

        let (expected_kind, expected_has_gain, expected_has_q, expected_slope) =
            emitted_kind(approved)?;
        if parsed.kind != expected_kind
            || parsed.frequency_hz != approved.freq
            || parsed.gain_db.is_some() != expected_has_gain
            || parsed.q.is_some() != expected_has_q
            || parsed.slope_db_per_octave != expected_slope
        {
            return Err(format!(
                "emitted APO filter {} does not match the approved profile fields",
                index + 1
            ));
        }
        if !approved.srate.is_finite()
            || (approved.srate - sample_rate_hz).abs()
                > approved.srate.abs().max(sample_rate_hz.abs()) * 1e-12
            || !approved.freq.is_finite()
            || approved.freq <= 0.0
            || approved.freq >= sample_rate_hz / 2.0
            || !approved.q.is_finite()
            || approved.q <= 0.0
            || !approved.db_gain.is_finite()
        {
            return Err(format!(
                "approved APO filter {} is invalid at the verification sample rate",
                index + 1
            ));
        }

        let realized_gain_db = match parsed.gain_db {
            Some(value) if value == approved.db_gain => value,
            None if approved.db_gain == 0.0 => 0.0,
            _ => {
                return Err(format!(
                    "emitted APO filter {} gain does not match the approved profile",
                    index + 1
                ));
            }
        };
        let realized_q = match parsed.q {
            Some(value) if value == approved.q => value,
            None if is_shelf(approved.filter_type) => approved.q,
            None if approved.q == DEFAULT_Q_HIGH_LOW_PASS => approved.q,
            _ => {
                return Err(format!(
                    "emitted APO filter {} Q does not match the approved profile",
                    index + 1
                ));
            }
        };
        let realized_filter = Biquad::new(
            approved.filter_type,
            parsed.frequency_hz,
            sample_rate_hz,
            realized_q,
            realized_gain_db,
        );
        if expected_slope.is_some() {
            let check = verify_source_shelf_semantics(
                expected_kind,
                parsed.frequency_hz,
                realized_gain_db,
                sample_rate_hz,
                &realized_filter,
                frequencies_hz,
            )?;
            max_shelf_scaled_coefficient_delta =
                max_shelf_scaled_coefficient_delta.max(check.scaled_coefficient_delta);
            max_shelf_source_transfer_delta_db =
                max_shelf_source_transfer_delta_db.max(check.max_transfer_delta_db);
        }
        emitted_filters.push(VerifiedApoFilterFields {
            kind: parsed.kind.clone(),
            frequency_hz: parsed.frequency_hz,
            gain_db: parsed.gain_db,
            q: parsed.q,
            slope_db_per_octave: parsed.slope_db_per_octave,
            frequency_convention: expected_slope.map(|_| "center_frequency_fc"),
        });
        realized_filters.push(realized_filter);
    }

    let max_transfer_delta_db = max_transfer_delta_db(
        frequencies_hz,
        &realized_filters,
        &expected,
        parsed_preamp,
        expected_preamp_db,
    )?;
    if max_transfer_delta_db != 0.0 {
        return Err(format!(
            "emitted APO transfer differs from the approved profile by {max_transfer_delta_db} dB"
        ));
    }

    Ok(VerifiedApoText {
        #[cfg(test)]
        filters: realized_filters,
        emitted_filters,
        preamp_db: parsed_preamp,
        sample_rate_hz,
        max_transfer_delta_db,
        max_shelf_scaled_coefficient_delta,
        max_shelf_source_transfer_delta_db,
    })
}

fn parse_preamp(line: &str) -> Result<f64, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 3
        || fields[0] != "Preamp:"
        || fields[2] != "dB"
        || fields[1].starts_with('+')
    {
        return Err("profiled APO preamp line has unsupported syntax".into());
    }
    let value = parse_fixed_decimal(fields[1], 1, false, "preamp")?;
    if value > 0.0 {
        return Err("profiled APO text only verifies non-positive preamp values".into());
    }
    Ok(value)
}

fn parse_filter(line: &str) -> Result<ParsedFilter, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 4 || fields[0] != "Filter" || fields[2] != "ON" {
        return Err("profiled APO filter line has unsupported syntax".into());
    }
    let index = fields[1]
        .strip_suffix(':')
        .ok_or_else(|| "profiled APO filter number is missing its colon".to_string())?
        .parse::<usize>()
        .map_err(|_| "profiled APO filter number is invalid".to_string())?;
    let kind = fields[3].to_owned();
    let (frequency_hz, gain_db, q, slope_db_per_octave) = match kind.as_str() {
        "LS" | "HS" => {
            return Err(format!(
                "profiled APO shelf output requires {kind}C with an explicit 12 dB slope and center frequency"
            ));
        }
        "LSC" | "HSC" => {
            if fields.len() != 12
                || fields[4] != "12"
                || fields[5] != "dB"
                || fields[6] != "Fc"
                || fields[8] != "Hz"
                || fields[9] != "Gain"
                || fields[11] != "dB"
            {
                return Err(format!(
                    "profiled APO {kind} shelf requires the exact '12 dB Fc <integer> Hz Gain <signed-value> dB' form"
                ));
            }
            (
                parse_frequency(fields[7])?,
                Some(parse_fixed_decimal(fields[10], 2, true, "filter gain")?),
                None,
                Some(PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE),
            )
        }
        "PK" => {
            if fields.len() != 12
                || fields[4] != "Fc"
                || fields[6] != "Hz"
                || fields[7] != "Gain"
                || fields[9] != "dB"
                || fields[10] != "Q"
            {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (
                parse_frequency(fields[5])?,
                Some(parse_fixed_decimal(fields[8], 2, true, "filter gain")?),
                Some(parse_fixed_decimal(fields[11], 2, false, "filter Q")?),
                None,
            )
        }
        "AP" | "LPQ" | "HPQ" => {
            if fields.len() != 9 || fields[4] != "Fc" || fields[6] != "Hz" || fields[7] != "Q" {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (
                parse_frequency(fields[5])?,
                None,
                Some(parse_fixed_decimal(fields[8], 2, false, "filter Q")?),
                None,
            )
        }
        "LP" | "HP" => {
            if fields.len() != 7 || fields[4] != "Fc" || fields[6] != "Hz" {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (parse_frequency(fields[5])?, None, None, None)
        }
        "BP" | "NO" => Err(format!(
            "profiled APO {kind} filters are refused because the emitted gain fields are not verified"
        ))?,
        "LSO" | "HSO" | "PKM" => Err(format!(
            "profiled APO {kind} filters are not in the verified emitted-text subset"
        ))?,
        _ => {
            return Err(format!(
                "profiled APO filter type '{kind}' is not in the verified emitted-text subset"
            ));
        }
    };
    if q.is_some_and(|value| !value.is_finite() || value <= 0.0)
        || gain_db.is_some_and(|value| !value.is_finite())
    {
        return Err("profiled APO filter parameters must be finite and Q positive".into());
    }
    Ok(ParsedFilter {
        index,
        kind,
        frequency_hz,
        gain_db,
        q,
        slope_db_per_octave,
    })
}

fn parse_frequency(text: &str) -> Result<f64, String> {
    if !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("profiled APO center frequency must be an integer number of Hz".into());
    }
    let frequency_hz = text
        .parse::<f64>()
        .map_err(|_| "profiled APO center frequency is invalid".to_string())?;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err("profiled APO center frequency must be finite and positive".into());
    }
    Ok(frequency_hz)
}

fn parse_fixed_decimal(
    text: &str,
    decimal_places: usize,
    require_sign: bool,
    label: &str,
) -> Result<f64, String> {
    let unsigned = text
        .strip_prefix('-')
        .or_else(|| text.strip_prefix('+'))
        .unwrap_or(text);
    if (require_sign && unsigned == text)
        || unsigned.matches('.').count() != 1
        || !unsigned.split_once('.').is_some_and(|(integer, fraction)| {
            !integer.is_empty()
                && integer.bytes().all(|byte| byte.is_ascii_digit())
                && fraction.len() == decimal_places
                && fraction.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return Err(format!(
            "profiled APO {label} must have exactly {decimal_places} decimal places"
        ));
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("profiled APO {label} is invalid"))?;
    if !value.is_finite() {
        return Err(format!("profiled APO {label} must be finite"));
    }
    Ok(value)
}

fn emitted_kind(filter: &Biquad) -> Result<(&'static str, bool, bool, Option<u8>), String> {
    let (kind, has_gain, has_q, slope_db_per_octave) = match filter.filter_type {
        BiquadFilterType::Peak => ("PK", true, true, None),
        BiquadFilterType::Lowpass => {
            if (filter.q - DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                ("LP", false, false, None)
            } else {
                ("LPQ", false, true, None)
            }
        }
        BiquadFilterType::Highpass => {
            if (filter.q - DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                ("HP", false, false, None)
            } else {
                ("HPQ", false, true, None)
            }
        }
        BiquadFilterType::HighpassVariableQ => ("HPQ", false, true, None),
        BiquadFilterType::Lowshelf => (
            "LSC",
            true,
            false,
            Some(PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE),
        ),
        BiquadFilterType::Highshelf => (
            "HSC",
            true,
            false,
            Some(PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE),
        ),
        BiquadFilterType::AllPass => ("AP", false, true, None),
        BiquadFilterType::Bandpass
        | BiquadFilterType::Notch
        | BiquadFilterType::LowshelfOrf
        | BiquadFilterType::HighshelfOrf
        | BiquadFilterType::PeakMatched => {
            return Err(format!(
                "APO filter type '{}' is outside the verified profiled subset",
                filter.filter_type.short_name()
            ));
        }
    };
    if !has_gain && filter.db_gain != 0.0 {
        return Err(format!(
            "APO {kind} output does not encode the approved filter gain"
        ));
    }
    Ok((kind, has_gain, has_q, slope_db_per_octave))
}

fn is_shelf(filter_type: BiquadFilterType) -> bool {
    matches!(
        filter_type,
        BiquadFilterType::Lowshelf | BiquadFilterType::Highshelf
    )
}

#[derive(Debug, Clone, Copy)]
struct SourceShelfCheck {
    scaled_coefficient_delta: f64,
    max_transfer_delta_db: f64,
}

fn verify_source_shelf_semantics(
    kind: &str,
    frequency_hz: f64,
    gain_db: f64,
    sample_rate_hz: f64,
    core_filter: &Biquad,
    comparison_frequencies_hz: &[f64],
) -> Result<SourceShelfCheck, String> {
    let source = source_shelf_coefficients(kind, frequency_hz, gain_db, sample_rate_hz)?;
    let core_coefficients = core_coefficients(core_filter);
    if core_coefficients.iter().any(|value| !value.is_finite()) {
        return Err("profiled APO core shelf coefficients are non-finite".into());
    }
    let scale = source
        .iter()
        .chain(core_coefficients.iter())
        .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
    if !scale.is_finite() || scale <= 0.0 {
        return Err("profiled APO shelf coefficient scale is invalid".into());
    }
    let max_coefficient_delta = source
        .iter()
        .zip(core_coefficients)
        .map(|(source, core)| (source - core).abs())
        .fold(0.0_f64, f64::max);
    let scaled_coefficient_delta = max_coefficient_delta / scale;
    let coefficient_limit =
        f64::EPSILON * f64::from(PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER);
    if !scaled_coefficient_delta.is_finite() || scaled_coefficient_delta > coefficient_limit {
        return Err(format!(
            "profiled APO {kind} shelf coefficients exceed the source-derived bound: scaled delta {scaled_coefficient_delta:e}, limit {coefficient_limit:e}"
        ));
    }

    let mut max_transfer_delta_db = 0.0_f64;
    for &frequency in comparison_frequencies_hz {
        let source_magnitude = response_magnitude(&source, frequency, sample_rate_hz)?;
        let core_magnitude = response_magnitude(&core_coefficients, frequency, sample_rate_hz)?;
        let delta_db = (20.0 * core_magnitude.log10() - 20.0 * source_magnitude.log10()).abs();
        if !delta_db.is_finite() {
            return Err("profiled APO source shelf response difference is non-finite".into());
        }
        max_transfer_delta_db = max_transfer_delta_db.max(delta_db);
    }
    if max_transfer_delta_db > PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB {
        return Err(format!(
            "profiled APO {kind} shelf sampled source transfer delta {max_transfer_delta_db:e} dB exceeds {:.1e} dB",
            PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB
        ));
    }
    Ok(SourceShelfCheck {
        scaled_coefficient_delta,
        max_transfer_delta_db,
    })
}

/// Reconstructs the 12 dB/octave LSC/HSC equations in Equalizer APO source
/// revision bbfcc3e5024cbb9d61ba75fc88d78605cc4c9687. Its factory divides the
/// explicit 12 dB slope by 12, giving S=1, and the `C` token keeps Fc as the
/// center frequency. These are source-derived equations, not a consumer parse.
fn source_shelf_coefficients(
    kind: &str,
    frequency_hz: f64,
    gain_db: f64,
    sample_rate_hz: f64,
) -> Result<[f64; 5], String> {
    if !frequency_hz.is_finite()
        || frequency_hz <= 0.0
        || frequency_hz >= sample_rate_hz / 2.0
        || !gain_db.is_finite()
        || !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
    {
        return Err("profiled APO shelf source-equation inputs are invalid".into());
    }
    let amplitude = 10.0_f64.powf(gain_db / 40.0);
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return Err("profiled APO shelf gain is outside source-equation range".into());
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz / sample_rate_hz;
    let cosine = omega.cos();
    let sine = omega.sin();
    let slope = f64::from(PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE);
    let shelf_s = slope / 12.0;
    let alpha = (sine / 2.0) * ((amplitude + 1.0 / amplitude) * (1.0 / shelf_s - 1.0) + 2.0).sqrt();
    let beta = 2.0 * amplitude.sqrt() * alpha;
    let (b0, b1, b2, a0, a1, a2) = match kind {
        "LSC" => (
            amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine + beta),
            2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine - beta),
            (amplitude + 1.0) + (amplitude - 1.0) * cosine + beta,
            -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            (amplitude + 1.0) + (amplitude - 1.0) * cosine - beta,
        ),
        "HSC" => (
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta),
            -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta),
            (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta,
        ),
        _ => {
            return Err(format!(
                "unsupported source-derived APO shelf type '{kind}'"
            ));
        }
    };
    let raw = [b0, b1, b2, a0, a1, a2];
    if raw.iter().any(|value| !value.is_finite()) || a0 <= f64::MIN_POSITIVE {
        return Err(
            "profiled APO shelf source coefficients are non-finite or ill-conditioned".into(),
        );
    }
    let normalized = [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err("profiled APO normalized shelf coefficients are non-finite".into());
    }
    Ok(normalized)
}

fn core_coefficients(filter: &Biquad) -> [f64; 5] {
    let coefficients = filter.coefficients();
    [
        coefficients.b0,
        coefficients.b1,
        coefficients.b2,
        coefficients.a1,
        coefficients.a2,
    ]
}

fn response_magnitude(
    coefficients: &[f64; 5],
    frequency_hz: f64,
    sample_rate_hz: f64,
) -> Result<f64, String> {
    let omega = 2.0 * std::f64::consts::PI * frequency_hz / sample_rate_hz;
    let (sin1, cos1) = omega.sin_cos();
    let (sin2, cos2) = (2.0 * omega).sin_cos();
    let [b0, b1, b2, a1, a2] = *coefficients;
    let numerator = (b0 + b1 * cos1 + b2 * cos2).hypot(-b1 * sin1 - b2 * sin2);
    let denominator = (1.0 + a1 * cos1 + a2 * cos2).hypot(-a1 * sin1 - a2 * sin2);
    if !numerator.is_finite()
        || numerator <= 0.0
        || !denominator.is_finite()
        || denominator <= f64::MIN_POSITIVE
    {
        return Err("profiled APO source shelf response is non-finite or ill-conditioned".into());
    }
    let magnitude = numerator / denominator;
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return Err("profiled APO source shelf response magnitude is invalid".into());
    }
    Ok(magnitude)
}

fn max_transfer_delta_db(
    frequencies_hz: &[f64],
    emitted: &[Biquad],
    expected: &[Biquad],
    emitted_preamp_db: f64,
    expected_preamp_db: f64,
) -> Result<f64, String> {
    let mut maximum = 0.0_f64;
    for &frequency_hz in frequencies_hz {
        let emitted_db = emitted_preamp_db
            + emitted
                .iter()
                .map(|filter| filter.log_result(frequency_hz))
                .sum::<f64>();
        let expected_db = expected_preamp_db
            + expected
                .iter()
                .map(|filter| filter.log_result(frequency_hz))
                .sum::<f64>();
        if !emitted_db.is_finite() || !expected_db.is_finite() {
            return Err("profiled APO transfer comparison is non-finite".into());
        }
        let delta_db = (emitted_db - expected_db).abs();
        if !delta_db.is_finite() {
            return Err("profiled APO transfer difference is non-finite".into());
        }
        maximum = maximum.max(delta_db);
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peak_filter() -> Biquad {
        Biquad::new(BiquadFilterType::Peak, 1_000.0, 48_000.0, 1.0, 6.0)
    }

    fn verify_fixture(text: &[u8]) -> Result<VerifiedApoText, String> {
        verify_emitted_apo_text(
            text,
            48_000.0,
            &[peak_filter()],
            -3.0,
            &[100.0, 1_000.0, 10_000.0],
        )
    }

    #[test]
    fn independent_known_value_fixture_checks_full_delivered_transfer() {
        // This fixture is handwritten; it does not use peq_format_apo.
        let verified = verify_fixture(
            b"# independent known-value preset\nPreamp: -3.0 dB\n\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
        )
        .unwrap();
        let center_db = verified.preamp_db + verified.filters[0].log_result(1_000.0);
        assert!((center_db - 3.0).abs() < 1e-12);
        assert_eq!(verified.sample_rate_hz, 48_000.0);
        assert_eq!(verified.max_transfer_delta_db, 0.0);
    }

    #[test]
    fn legacy_ls_and_hs_text_is_refused_in_profiled_output() {
        for (kind, filter_type) in [
            ("LS", BiquadFilterType::Lowshelf),
            ("HS", BiquadFilterType::Highshelf),
        ] {
            let shelf = Biquad::new(filter_type, 100.0, 48_000.0, 0.71, 3.0);
            let text = format!(
                "# independent shelf fixture\nPreamp: -3.0 dB\nFilter 1: ON {kind} Fc 100 Hz Gain +3.00 dB Q 0.71\n"
            );
            let error = verify_emitted_apo_text(
                text.as_bytes(),
                48_000.0,
                &[shelf],
                -3.0,
                &[50.0, 100.0, 1_000.0],
            )
            .expect_err("profiled path must not equate core and consumer shelf semantics");
            assert!(
                error.contains("requires") && error.contains("12 dB"),
                "{error}"
            );
        }
    }

    #[test]
    fn lsc_and_hsc_require_the_exact_center_slope_form() {
        let shelf = Biquad::new(BiquadFilterType::Lowshelf, 100.0, 48_000.0, 0.71, 3.0);
        let malformed = [
            "Filter 1: ON LSC 6 dB Fc 100 Hz Gain +3.00 dB",
            "Filter 1: ON LSC Fc 100 Hz 12 dB Gain +3.00 dB",
            "Filter 1: ON LSC 12 dB Fc 100 Hz Gain +3.00 dB Q 0.71",
            "Filter 1: ON LSC 12 dB Fc 100.0 Hz Gain +3.00 dB",
        ];
        for filter_line in malformed {
            let text = format!("Preamp: 0.0 dB\n{filter_line}\n");
            assert!(
                verify_emitted_apo_text(
                    text.as_bytes(),
                    48_000.0,
                    std::slice::from_ref(&shelf),
                    0.0,
                    &[50.0, 100.0, 1_000.0],
                )
                .is_err(),
                "malformed shelf line should be refused: {filter_line}"
            );
        }
        let high_shelf = Biquad::new(BiquadFilterType::Highshelf, 100.0, 48_000.0, 1.4, -3.0);
        assert!(
            verify_emitted_apo_text(
                b"Preamp: 0.0 dB\nFilter 1: ON LSC 12 dB Fc 100 Hz Gain -3.00 dB\n",
                48_000.0,
                &[high_shelf],
                0.0,
                &[50.0, 100.0, 1_000.0],
            )
            .is_err(),
            "low/high shelf token mismatches must fail"
        );
    }

    #[test]
    fn source_formula_coefficients_match_independent_official_reference_vectors() {
        // Fixed values were calculated independently from the Equalizer APO
        // bbfcc3e source equations. This test does not call the production
        // source-equation reconstruction below.
        let cases: [(BiquadFilterType, f64, f64, f64, [f64; 5]); 4] = [
            (
                BiquadFilterType::Lowshelf,
                6.0,
                1_000.0,
                48_000.0,
                [
                    1.0325624832475901,
                    -1.8388568718996405,
                    0.828_747_684_312_469_8,
                    -1.8444568671609198,
                    0.855_710_172_298_780_8,
                ],
            ),
            (
                BiquadFilterType::Highshelf,
                -6.0,
                10_000.0,
                44_100.0,
                [
                    0.687_392_156_331_420_6,
                    0.021_044_882_595_103_83,
                    0.11805174933139004,
                    -0.3692965111837429,
                    0.19578529944165726,
                ],
            ),
            (
                BiquadFilterType::Lowshelf,
                24.0,
                1.0,
                44_100.0,
                [
                    1.000_150_532_885_394,
                    -1.9998989772826405,
                    0.999_748_525_206_413_9,
                    -1.9998990151378668,
                    0.999_899_020_236_580_6,
                ],
            ),
            (
                BiquadFilterType::Highshelf,
                -24.0,
                22_049.0,
                44_100.0,
                [
                    0.999_849_489_771_345,
                    1.999_598_009_879_812,
                    0.999_748_525_206_413_2,
                    1.9995979720302828,
                    0.999_598_052_827_287_6,
                ],
            ),
        ];
        for (filter_type, gain, frequency, sample_rate, reference) in cases {
            let actual = Biquad::new(filter_type, frequency, sample_rate, 1.37, gain);
            let actual = core_coefficients(&actual);
            let scale = reference
                .iter()
                .fold(1.0_f64, |maximum, value| maximum.max((*value).abs()));
            for (actual, expected) in actual.iter().zip(reference) {
                assert!(
                    (actual - expected).abs()
                        <= f64::EPSILON
                            * f64::from(PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER)
                            * scale,
                    "{filter_type:?} coefficient differs from frozen source reference: {actual:.17e} vs {expected:.17e}"
                );
            }
        }

        let low: Biquad<f64> = Biquad::new(BiquadFilterType::Lowshelf, 1_000.0, 48_000.0, 1.0, 6.0);
        let low_reference = [
            1.0325624832475901,
            -1.8388568718996405,
            0.828_747_684_312_469_8,
            -1.8444568671609198,
            0.855_710_172_298_780_8,
        ];
        let high: Biquad<f64> =
            Biquad::new(BiquadFilterType::Highshelf, 10_000.0, 44_100.0, 1.0, -6.0);
        let high_reference = [
            0.687_392_156_331_420_6,
            0.021_044_882_595_103_83,
            0.11805174933139004,
            -0.3692965111837429,
            0.19578529944165726,
        ];
        for (filter, coefficients, frequency, sample_rate) in [
            (&low, low_reference, 1_200.0, 48_000.0),
            (&high, high_reference, 7_300.0, 44_100.0),
        ] {
            let expected = reference_response(&coefficients, frequency, sample_rate);
            let actual = filter.complex_response(frequency);
            let phase_delta = (actual.im.atan2(actual.re) - expected.1.atan2(expected.0)).abs();
            assert!((actual.re - expected.0).abs() < 1.0e-14);
            assert!((actual.im - expected.1).abs() < 1.0e-14);
            assert!(
                phase_delta < 1.0e-14,
                "complex phase delta was {phase_delta:e}"
            );
        }
    }

    #[test]
    fn source_shelf_check_covers_rates_gain_and_frequency_limits() {
        for sample_rate in [44_100.0_f64, 48_000.0, 96_000.0, 192_000.0] {
            let nyquist = sample_rate / 2.0;
            let frequencies = [20.0, (sample_rate * 0.45).round()];
            for frequency in frequencies {
                if frequency <= 0.0 || frequency >= nyquist {
                    continue;
                }
                for gain in [-24.0, -6.0, 0.0, 6.0, 24.0] {
                    for (kind, filter_type) in [
                        ("LSC", BiquadFilterType::Lowshelf),
                        ("HSC", BiquadFilterType::Highshelf),
                    ] {
                        let filter = Biquad::new(filter_type, frequency, sample_rate, 0.73, gain);
                        let line = format!(
                            "Preamp: 0.0 dB\nFilter 1: ON {kind} 12 dB Fc {frequency:.0} Hz Gain {gain:+.2} dB\n"
                        );
                        let verified = verify_emitted_apo_text(
                            line.as_bytes(),
                            sample_rate,
                            &[filter],
                            0.0,
                            &[1.0, 20.0, frequency.min(nyquist - 0.5), nyquist - 1.0],
                        )
                        .unwrap_or_else(|error| {
                            panic!("{kind} gain={gain} Fc={frequency} Fs={sample_rate}: {error}")
                        });
                        assert_eq!(verified.max_transfer_delta_db, 0.0);
                        assert!(verified.max_shelf_scaled_coefficient_delta <= 16.0 * f64::EPSILON);
                        assert!(verified.max_shelf_source_transfer_delta_db <= 1.0e-10);
                        assert_eq!(verified.emitted_filters[0].kind, kind);
                        assert_eq!(verified.emitted_filters[0].q, None);
                        assert_eq!(verified.emitted_filters[0].slope_db_per_octave, Some(12));
                        assert_eq!(
                            verified.emitted_filters[0].frequency_convention,
                            Some("center_frequency_fc")
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn low_frequency_center_remains_within_the_declared_source_bounds() {
        let filter = Biquad::new(BiquadFilterType::Lowshelf, 1.0, 44_100.0, 0.73, 24.0);
        let text = b"Preamp: 0.0 dB\nFilter 1: ON LSC 12 dB Fc 1 Hz Gain +24.00 dB\n";
        let verified = verify_emitted_apo_text(text, 44_100.0, &[filter], 0.0, &[0.5, 1.0, 20.0])
            .expect("low-frequency shelf should pass when it satisfies both numeric bounds");
        assert_eq!(verified.max_transfer_delta_db, 0.0);
        assert!(verified.max_shelf_scaled_coefficient_delta <= 16.0 * f64::EPSILON);
        assert!(verified.max_shelf_source_transfer_delta_db <= 1.0e-10);
    }

    #[test]
    fn ill_conditioned_near_nyquist_roundtrip_is_refused_without_relaxing_bounds() {
        let filter = Biquad::new(BiquadFilterType::Highshelf, 22_049.0, 44_100.0, 0.73, -6.0);
        let text = b"Preamp: 0.0 dB\nFilter 1: ON HSC 12 dB Fc 22049 Hz Gain -6.00 dB\n";
        let error = verify_emitted_apo_text(text, 44_100.0, &[filter], 0.0, &[1.0, 20.0, 22_049.0])
            .expect_err("non-finite or out-of-bound responses must fail closed");
        assert!(
            error.contains("transfer comparison is non-finite"),
            "the reported limitation must remain tied to the failed comparison: {error}"
        );
    }

    #[test]
    fn non_representable_shelf_source_equations_fail_closed() {
        assert!(source_shelf_coefficients("LSC", 100.0, 100_000.0, 48_000.0).is_err());
        let invalid = Biquad::new(BiquadFilterType::Lowshelf, 100.0, 48_000.0, 1.0, 100_000.0);
        let text = b"Preamp: 0.0 dB\nFilter 1: ON LSC 12 dB Fc 100 Hz Gain +100000.00 dB\n";
        assert!(
            verify_emitted_apo_text(text, 48_000.0, &[invalid], 0.0, &[50.0, 100.0, 1_000.0])
                .is_err()
        );
    }

    fn reference_response(
        coefficients: &[f64; 5],
        frequency_hz: f64,
        sample_rate_hz: f64,
    ) -> (f64, f64) {
        let omega = std::f64::consts::TAU * frequency_hz / sample_rate_hz;
        let (sin1, cos1) = omega.sin_cos();
        let (sin2, cos2) = (2.0 * omega).sin_cos();
        let [b0, b1, b2, a1, a2] = *coefficients;
        let numerator_re = b0 + b1 * cos1 + b2 * cos2;
        let numerator_im = -b1 * sin1 - b2 * sin2;
        let denominator_re = 1.0 + a1 * cos1 + a2 * cos2;
        let denominator_im = -a1 * sin1 - a2 * sin2;
        let denominator_power = denominator_re * denominator_re + denominator_im * denominator_im;
        (
            (numerator_re * denominator_re + numerator_im * denominator_im) / denominator_power,
            (numerator_im * denominator_re - numerator_re * denominator_im) / denominator_power,
        )
    }

    #[test]
    fn hpq_and_lpq_fixtures_accept_explicit_q_fields() {
        // These handwritten lines cover the variable-Q commands separately
        // from the legacy HP/LP forms that omit Q at the default value.
        let highpass = Biquad::new(
            BiquadFilterType::HighpassVariableQ,
            125.0,
            48_000.0,
            0.70,
            0.0,
        );
        let lowpass = Biquad::new(BiquadFilterType::Lowpass, 12_000.0, 48_000.0, 0.80, 0.0);
        let verified = verify_emitted_apo_text(
            b"# independent explicit-Q fixture\nPreamp: 0.0 dB\nFilter 1: ON HPQ Fc 125 Hz Q 0.70\nFilter 2: ON LPQ Fc 12000 Hz Q 0.80\n",
            48_000.0,
            &[highpass, lowpass],
            0.0,
            &[50.0, 125.0, 1_000.0, 12_000.0, 20_000.0],
        )
        .unwrap();

        assert_eq!(verified.filters.len(), 2);
        assert_eq!(
            verified.filters[0].filter_type,
            BiquadFilterType::HighpassVariableQ
        );
        assert_eq!(verified.filters[0].q, 0.70);
        assert_eq!(verified.filters[1].filter_type, BiquadFilterType::Lowpass);
        assert_eq!(verified.filters[1].q, 0.80);
        assert_eq!(verified.max_transfer_delta_db, 0.0);
    }

    #[test]
    fn hp_and_lp_omitted_q_use_the_approved_default_q() {
        // Equalizer APO's HP/LP spellings omit Q. The approved model is valid
        // for that spelling only at the writer's established default Q.
        let highpass = Biquad::new(
            BiquadFilterType::Highpass,
            80.0,
            48_000.0,
            DEFAULT_Q_HIGH_LOW_PASS,
            0.0,
        );
        let lowpass = Biquad::new(
            BiquadFilterType::Lowpass,
            16_000.0,
            48_000.0,
            DEFAULT_Q_HIGH_LOW_PASS,
            0.0,
        );
        let verified = verify_emitted_apo_text(
            b"# independent default-Q fixture\nPreamp: 0.0 dB\nFilter 1: ON HP Fc 80 Hz\nFilter 2: ON LP Fc 16000 Hz\n",
            48_000.0,
            &[highpass, lowpass],
            0.0,
            &[40.0, 80.0, 1_000.0, 16_000.0, 20_000.0],
        )
        .unwrap();

        assert_eq!(verified.filters.len(), 2);
        assert_eq!(verified.filters[0].q, DEFAULT_Q_HIGH_LOW_PASS);
        assert_eq!(verified.filters[1].q, DEFAULT_Q_HIGH_LOW_PASS);
        assert_eq!(verified.max_transfer_delta_db, 0.0);
        assert!(
            (verified.filters[0].log_result(80.0) + 3.0103).abs() < 0.02,
            "default-Q high-pass should have the known second-order cutoff response"
        );
    }

    #[test]
    fn unsupported_commands_and_filter_semantics_fail_closed() {
        for text in [
            b"Preamp: -3.0 dB\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\nChannel: L\n".as_slice(),
            b"Preamp: -3.0 dB\nInclude: other.txt\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: ON PK Fc 1000 Hz Gain +5.99 dB Q 1.00\n",
            b"Preamp: -2.9 dB\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 2: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: OFF PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: ON NO Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: ON BP Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: ON LSO Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
            b"Preamp: -3.0 dB\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.0\n",
        ] {
            assert!(verify_fixture(text).is_err(), "accepted {text:?}");
        }
    }

    #[test]
    fn duplicate_late_or_positive_preamp_is_refused() {
        for text in [
            b"Preamp: -3.0 dB\nPreamp: -3.0 dB\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n"
                .as_slice(),
            b"Filter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\nPreamp: -3.0 dB\n",
            b"Preamp: +1.0 dB\nFilter 1: ON PK Fc 1000 Hz Gain +6.00 dB Q 1.00\n",
        ] {
            assert!(verify_fixture(text).is_err(), "accepted {text:?}");
        }
    }
}
