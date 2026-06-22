use super::misc::processing_mode_name;
use super::scenario_kind::ScenarioKind;
use super::scenario_kind::required_scenarios;
use autoeq::roomeq::{RoomConfig, SpeakerConfig};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct CoverageCounters {
    pub(super) scenario_counts: BTreeMap<ScenarioKind, usize>,
    pub(super) mode_counts: BTreeMap<&'static str, usize>,
    pub(super) feature_counts: BTreeMap<&'static str, usize>,
}

impl CoverageCounters {
    pub(super) fn record(&mut self, scenario: ScenarioKind, config: &RoomConfig) {
        *self.scenario_counts.entry(scenario).or_insert(0) += 1;
        *self
            .mode_counts
            .entry(processing_mode_name(&config.optimizer.processing_mode))
            .or_insert(0) += 1;

        if config.optimizer.multi_measurement.is_some() {
            *self.feature_counts.entry("multi_measurement").or_insert(0) += 1;
        }
        if config.optimizer.multi_seat.is_some() {
            *self.feature_counts.entry("multi_seat").or_insert(0) += 1;
        }
        if config.optimizer.channel_matching.is_some() {
            *self.feature_counts.entry("channel_matching").or_insert(0) += 1;
        }
        if config.optimizer.group_delay.is_some() {
            *self.feature_counts.entry("group_delay").or_insert(0) += 1;
        }
        if config
            .speakers
            .values()
            .any(|speaker| matches!(speaker, SpeakerConfig::MultiSub(group) if group.allpass_optimization))
        {
            *self.feature_counts.entry("multi_sub_allpass").or_insert(0) += 1;
        }
    }

    pub(super) fn print(&self, skip_kautz_modal: bool) {
        let required = required_scenarios(skip_kautz_modal);
        println!("\nCoverage buckets:");
        for scenario in &required {
            println!(
                "  {:<38} {}",
                scenario.name(),
                self.scenario_counts.get(scenario).copied().unwrap_or(0)
            );
        }
        println!(
            "  {:<38} {}",
            ScenarioKind::RandomMixed.name(),
            self.scenario_counts
                .get(&ScenarioKind::RandomMixed)
                .copied()
                .unwrap_or(0)
        );

        println!("\nProcessing modes:");
        for (mode, count) in &self.mode_counts {
            println!("  {:<38} {}", mode, count);
        }

        println!("\nFeature flags:");
        for (feature, count) in &self.feature_counts {
            println!("  {:<38} {}", feature, count);
        }
    }

    pub(super) fn missing_required(
        &self,
        num_tests: usize,
        skip_kautz_modal: bool,
    ) -> Vec<&'static str> {
        let required = required_scenarios(skip_kautz_modal);
        if num_tests < required.len() {
            return Vec::new();
        }

        required
            .iter()
            .filter(|scenario| self.scenario_counts.get(scenario).copied().unwrap_or(0) == 0)
            .map(|scenario| scenario.name())
            .collect()
    }
}
