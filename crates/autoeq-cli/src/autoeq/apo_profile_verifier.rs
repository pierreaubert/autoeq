//! Strictly verifies the supported Equalizer APO text emitted for product profiles.

// Rust guideline compliant 2026-02-21

use autoeq::iir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_PASS};

const MAX_VERIFIED_TEXT_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct ParsedFilter {
    index: usize,
    kind: String,
    frequency_hz: f64,
    gain_db: Option<f64>,
    q: Option<f64>,
}

#[derive(Debug)]
pub(super) struct VerifiedApoText {
    pub(super) filters: Vec<Biquad>,
    pub(super) preamp_db: f64,
    pub(super) sample_rate_hz: f64,
    pub(super) max_transfer_delta_db: f64,
}

/// Verifies emitted APO text against the caller-approved serialized filters.
///
/// The parser accepts only the formatter subset checked for profiled output. It
/// does not claim to be an Equalizer APO parser or to verify installation,
/// device selection, channel routing, or runtime behavior. Routing is inherited
/// from the surrounding Equalizer APO configuration. LS/HS shelves are
/// refused because this core's fixed-slope shelf transfer does not match the
/// consumer's Q and corner-frequency semantics.
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
    let mut previous_frequency_hz = 0.0;
    for (index, (parsed, approved)) in parsed_filters.iter().zip(&expected).enumerate() {
        if parsed.index != index + 1 {
            return Err("emitted APO filter numbers must be contiguous and start at one".into());
        }
        if parsed.frequency_hz < previous_frequency_hz {
            return Err("emitted APO filters are not ordered by ascending frequency".into());
        }
        previous_frequency_hz = parsed.frequency_hz;

        let (expected_kind, expected_has_gain, expected_has_q) = emitted_kind(approved)?;
        if parsed.kind != expected_kind
            || parsed.frequency_hz != approved.freq
            || parsed.gain_db.is_some() != expected_has_gain
            || parsed.q.is_some() != expected_has_q
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
            None if approved.q == DEFAULT_Q_HIGH_LOW_PASS => approved.q,
            _ => {
                return Err(format!(
                    "emitted APO filter {} Q does not match the approved profile",
                    index + 1
                ));
            }
        };
        realized_filters.push(Biquad::new(
            approved.filter_type,
            parsed.frequency_hz,
            sample_rate_hz,
            realized_q,
            realized_gain_db,
        ));
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
        filters: realized_filters,
        preamp_db: parsed_preamp,
        sample_rate_hz,
        max_transfer_delta_db,
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
    if fields.len() < 7 || fields[0] != "Filter" || fields[2] != "ON" || fields[4] != "Fc" {
        return Err("profiled APO filter line has unsupported syntax".into());
    }
    let index = fields[1]
        .strip_suffix(':')
        .ok_or_else(|| "profiled APO filter number is missing its colon".to_string())?
        .parse::<usize>()
        .map_err(|_| "profiled APO filter number is invalid".to_string())?;
    let kind = fields[3].to_owned();
    let frequency_text = fields[5];
    if fields[6] != "Hz" || !frequency_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("profiled APO center frequency must be an integer number of Hz".into());
    }
    let frequency_hz = frequency_text
        .parse::<f64>()
        .map_err(|_| "profiled APO center frequency is invalid".to_string())?;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err("profiled APO center frequency must be finite and positive".into());
    }

    let (gain_db, q) = match kind.as_str() {
        "LS" | "HS" => Err(format!(
            "profiled APO refuses {kind} shelves because core fixed-slope shelves do not match consumer Q and corner-frequency semantics"
        ))?,
        "PK" => {
            if fields.len() != 12 || fields[7] != "Gain" || fields[9] != "dB" || fields[10] != "Q" {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (
                Some(parse_fixed_decimal(fields[8], 2, true, "filter gain")?),
                Some(parse_fixed_decimal(fields[11], 2, false, "filter Q")?),
            )
        }
        "AP" | "LPQ" | "HPQ" => {
            if fields.len() != 9 || fields[7] != "Q" {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (
                None,
                Some(parse_fixed_decimal(fields[8], 2, false, "filter Q")?),
            )
        }
        "LP" | "HP" => {
            if fields.len() != 7 {
                return Err(format!("profiled APO {kind} filter has unsupported fields"));
            }
            (None, None)
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
    })
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

fn emitted_kind(filter: &Biquad) -> Result<(&'static str, bool, bool), String> {
    let (kind, has_gain, has_q) = match filter.filter_type {
        BiquadFilterType::Peak => ("PK", true, true),
        BiquadFilterType::Lowpass => {
            if (filter.q - DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                ("LP", false, false)
            } else {
                ("LPQ", false, true)
            }
        }
        BiquadFilterType::Highpass => {
            if (filter.q - DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                ("HP", false, false)
            } else {
                ("HPQ", false, true)
            }
        }
        BiquadFilterType::HighpassVariableQ => ("HPQ", false, true),
        BiquadFilterType::Lowshelf | BiquadFilterType::Highshelf => {
            return Err(format!(
                "profiled APO refuses {} shelves because core fixed-slope shelves do not match consumer Q and corner-frequency semantics",
                filter.filter_type.short_name()
            ));
        }
        BiquadFilterType::AllPass => ("AP", false, true),
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
    Ok((kind, has_gain, has_q))
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
    fn shelf_filter_text_is_refused_even_when_its_fields_match_core_output() {
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
            assert!(error.contains("fixed-slope shelves"), "{error}");
        }
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
