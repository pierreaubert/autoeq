pub(super) fn parse_nonnegative_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("invalid float: {s}"))?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err("value must be non-negative (>= 0)".to_string())
    }
}

pub(super) fn parse_finite_f64(s: &str) -> Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| format!("invalid float: {s}"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("value must be finite".to_string())
    }
}

pub(super) fn parse_recombination_probability(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("invalid float: {s}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err("recombination probability must be between 0.0 and 1.0".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nonnegative_f64_accepts_zero_and_positive() {
        assert_eq!(parse_nonnegative_f64("0").unwrap(), 0.0);
        assert_eq!(parse_nonnegative_f64("0.0").unwrap(), 0.0);
        assert_eq!(parse_nonnegative_f64("3.5").unwrap(), 3.5);
        assert_eq!(parse_nonnegative_f64("1e3").unwrap(), 1000.0);
    }

    #[test]
    fn parse_nonnegative_f64_rejects_negative_and_invalid() {
        assert!(parse_nonnegative_f64("-1").is_err());
        assert!(parse_nonnegative_f64("-0.001").is_err());
        assert!(parse_nonnegative_f64("not-a-number").is_err());
        assert!(parse_nonnegative_f64("").is_err());
    }

    #[test]
    fn parse_finite_f64_accepts_signed_values_and_rejects_non_finite_values() {
        assert_eq!(parse_finite_f64("-12").unwrap(), -12.0);
        assert_eq!(parse_finite_f64("0").unwrap(), 0.0);
        assert_eq!(parse_finite_f64("3.5").unwrap(), 3.5);
        for invalid in ["NaN", "inf", "-inf", "not-a-number", ""] {
            assert!(parse_finite_f64(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn parse_recombination_probability_accepts_valid_range() {
        assert_eq!(parse_recombination_probability("0").unwrap(), 0.0);
        assert_eq!(parse_recombination_probability("1").unwrap(), 1.0);
        assert_eq!(parse_recombination_probability("0.5").unwrap(), 0.5);
        assert_eq!(parse_recombination_probability("0.0").unwrap(), 0.0);
        assert_eq!(parse_recombination_probability("1.0").unwrap(), 1.0);
    }

    #[test]
    fn parse_recombination_probability_rejects_out_of_range_and_invalid() {
        assert!(parse_recombination_probability("-0.1").is_err());
        assert!(parse_recombination_probability("1.0001").is_err());
        assert!(parse_recombination_probability("2").is_err());
        assert!(parse_recombination_probability("foo").is_err());
        assert!(parse_recombination_probability("").is_err());
    }
}
