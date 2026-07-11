use crate::MeasurementSource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Declared acoustic role of a measured driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerDriverRole {
    Subwoofer,
    Woofer,
    Midrange,
    Tweeter,
    FullRange,
    Other,
}

/// Frequency region in which per-driver linearization is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DriverCrossoverBand {
    pub min_hz: f64,
    pub max_hz: f64,
}

/// One separately measured driver in an explicit speaker topology.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SpeakerDriver {
    /// Stable identifier used by output DSP chains and parallel-group membership.
    pub id: String,
    pub role: SpeakerDriverRole,
    pub measurement: MeasurementSource,
    /// Optional explicit linearization band. Crossover-derived defaults are used when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossover_band: Option<DriverCrossoverBand>,
}

/// Drivers that radiate in parallel within one acoustic crossover band.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParallelDriverGroup {
    pub id: String,
    pub driver_ids: Vec<String>,
}

/// Explicit multi-driver speaker topology.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SpeakerTopology {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    /// Drivers in low-to-high acoustic-band order. Members of a parallel group
    /// may appear next to each other in any order within their shared band.
    pub drivers: Vec<SpeakerDriver>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parallel_groups: Vec<ParallelDriverGroup>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossover: Option<String>,
}

impl SpeakerTopology {
    pub fn resolve_paths(&mut self, base_dir: &std::path::Path) {
        for driver in &mut self.drivers {
            driver.measurement.resolve_paths(base_dir);
        }
    }

    /// Validate identifiers, bands, and parallel membership at the config boundary.
    pub fn validate(&self) -> Result<(), String> {
        if self.drivers.is_empty() {
            return Err("speaker topology requires at least one driver".to_string());
        }

        let mut driver_ids = HashSet::new();
        for driver in &self.drivers {
            if driver.id.trim().is_empty() {
                return Err("speaker topology driver IDs must not be empty".to_string());
            }
            if !driver_ids.insert(driver.id.as_str()) {
                return Err(format!(
                    "duplicate speaker topology driver ID '{}'",
                    driver.id
                ));
            }
            if let Some(band) = driver.crossover_band
                && (!band.min_hz.is_finite()
                    || !band.max_hz.is_finite()
                    || band.min_hz <= 0.0
                    || band.max_hz <= band.min_hz)
            {
                return Err(format!(
                    "driver '{}' has invalid crossover band [{}, {}] Hz",
                    driver.id, band.min_hz, band.max_hz
                ));
            }
        }

        let mut group_ids = HashSet::new();
        let mut grouped_drivers = HashSet::new();
        for group in &self.parallel_groups {
            if group.id.trim().is_empty() {
                return Err("parallel-group IDs must not be empty".to_string());
            }
            if !group_ids.insert(group.id.as_str()) {
                return Err(format!("duplicate parallel-group ID '{}'", group.id));
            }
            if group.driver_ids.len() < 2 {
                return Err(format!(
                    "parallel group '{}' requires at least two drivers",
                    group.id
                ));
            }
            if group.driver_ids.len() > 4 {
                return Err(format!(
                    "parallel group '{}' has {} drivers; at most four are supported",
                    group.id,
                    group.driver_ids.len()
                ));
            }
            let mut members = HashSet::new();
            let mut member_indices = Vec::with_capacity(group.driver_ids.len());
            let mut role = None;
            for driver_id in &group.driver_ids {
                if !driver_ids.contains(driver_id.as_str()) {
                    return Err(format!(
                        "parallel group '{}' references unknown driver '{}'",
                        group.id, driver_id
                    ));
                }
                if !members.insert(driver_id.as_str()) {
                    return Err(format!(
                        "parallel group '{}' repeats driver '{}'",
                        group.id, driver_id
                    ));
                }
                if !grouped_drivers.insert(driver_id.as_str()) {
                    return Err(format!(
                        "driver '{}' belongs to more than one parallel group",
                        driver_id
                    ));
                }
                let Some(driver_index) = self
                    .drivers
                    .iter()
                    .position(|driver| driver.id == *driver_id)
                else {
                    return Err(format!(
                        "parallel group '{}' references unknown driver '{}'",
                        group.id, driver_id
                    ));
                };
                member_indices.push(driver_index);
                let driver_role = self.drivers[driver_index].role;
                if role.is_some_and(|role| role != driver_role) {
                    return Err(format!(
                        "parallel group '{}' mixes incompatible driver roles",
                        group.id
                    ));
                }
                role = Some(driver_role);
            }
            member_indices.sort_unstable();
            if member_indices
                .windows(2)
                .any(|indices| indices[1] != indices[0] + 1)
            {
                return Err(format!(
                    "parallel group '{}' members must be contiguous in driver order",
                    group.id
                ));
            }
        }
        Ok(())
    }

    /// Driver indices grouped into ordered acoustic crossover bands.
    pub fn acoustic_bands(&self) -> Result<Vec<Vec<usize>>, String> {
        self.validate()?;
        let mut group_for_driver = vec![None; self.drivers.len()];
        for (group_index, group) in self.parallel_groups.iter().enumerate() {
            for driver_id in &group.driver_ids {
                let Some(driver_index) = self
                    .drivers
                    .iter()
                    .position(|driver| driver.id == *driver_id)
                else {
                    return Err(format!(
                        "parallel group '{}' references unknown driver '{}'",
                        group.id, driver_id
                    ));
                };
                group_for_driver[driver_index] = Some(group_index);
            }
        }

        let mut emitted_groups = HashSet::new();
        let mut bands = Vec::new();
        for (driver_index, group_index) in group_for_driver.iter().enumerate() {
            match group_index {
                None => bands.push(vec![driver_index]),
                Some(group_index) if emitted_groups.insert(*group_index) => {
                    bands.push(
                        group_for_driver
                            .iter()
                            .enumerate()
                            .filter_map(|(index, candidate)| {
                                (*candidate == Some(*group_index)).then_some(index)
                            })
                            .collect(),
                    );
                }
                Some(_) => {}
            }
        }
        Ok(bands)
    }
}
