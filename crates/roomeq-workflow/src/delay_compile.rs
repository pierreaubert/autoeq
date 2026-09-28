//! Graph-domain compilation passes over replay-mirrored branches.
//!
//! Optimizers express alignment as relative advances that may be negative,
//! but a real-time delay block cannot produce future samples. The causal
//! pass compiles every delay element in the serialized graph (route delays,
//! channel delay plugins, driver delay plugins) to nonnegative values while
//! preserving all inter-branch timing relationships: every branch total
//! rises by the same common offset, which the report serializes. A second
//! pass bounds electrical filter boost per branch for audit.
//!
//! The algorithm is additive-only and idempotent. Per branch, negative
//! elements are zeroed once each (shared elements affect every sharing
//! branch identically) and the branch deficit is recorded. The common
//! offset is the maximum deficit; each branch receives its remainder on
//! its route delay, or on a channel delay plugin for non-routed branches.
//! A graph without negatives compiles to a zero offset with no changes.
//!
//! Branch membership mirrors final-seat replay exactly: routed branches
//! cover the input chain `pre_route` delays, the route delay, the post
//! chain `post_route` delays, and the matched driver delays; non-routed
//! branches cover whole channel and driver chains. Delays outside the
//! replayed stages are still zeroed for playback causality, with an
//! advisory recording the replay divergence.

use roomeq_model::{BassManagementRoute, ChannelDspChain};
use std::collections::{HashMap, HashSet};

/// Magnitudes below this are snapped to zero silently.
///
/// Covers float dust from earlier normalization without churning the graph
/// or the report. Well below any acoustic or sample-grid relevance.
const SNAP_EPSILON_MS: f64 = 1e-9;

/// Label marking delay padding appended by causal compilation.
///
/// The padding carries the branch's share of the common latency offset; it
/// is derived from optimized relative timing, not a structural alignment.
/// Baseline restoration strips marked padding so refused graphs assert no
/// compile-derived timing the evidence never supported.
pub const COMPILE_PADDING_LABEL: &str = "delay_compile_common_latency";

/// Outcome of one delay-compilation pass.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayCompilationReport {
    /// Uniform offset added to every branch total, in ms.
    ///
    /// Zero when the graph was already causal. Serialize alongside the
    /// graph: realizing the compiled delays requires this much common
    /// latency beyond the previously most advanced path.
    pub common_latency_ms: f64,
    /// Branches compiled (routed plus standalone).
    pub branches: usize,
    /// True when the causality net zeroed a delay outside replayed stages.
    ///
    /// Pre-compile replay verdicts (including the splice recheck) evaluated
    /// different acoustics than the shipped graph realizes at that stage,
    /// so the pipeline must fail closed rather than transfer the verdict.
    pub replay_diverged: bool,
    /// Human-readable record of shifts, skips, and divergences.
    pub advisories: Vec<String>,
}

/// Location of one adjustable delay element.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DelayLocation {
    /// `routes[index].delay_ms`.
    Route(usize),
    /// Channel plugin: (channel, plugin index).
    ChannelPlugin(String, usize),
    /// Driver plugin: (channel, driver index, plugin index).
    DriverPlugin(String, usize, usize),
}

/// Compile graph delays to nonnegative values, preserving branch totals.
///
/// See the module documentation for the algorithm. Returns the compilation
/// report; `common_latency_ms` is the serialized common offset.
pub fn compile_graph_delays_causal(
    channels: &mut HashMap<String, ChannelDspChain>,
    mut routes: Option<&mut Vec<BassManagementRoute>>,
) -> DelayCompilationReport {
    let mut advisories = Vec::new();
    // Branch membership in deterministic order: routes in graph order,
    // then standalone channels and drivers by sorted name.
    let (branches, _) = assemble_branches(
        channels,
        routes.as_deref().map(Vec::as_slice),
        &mut advisories,
    );
    // Element values keyed by location, read before any mutation.
    let mut values: HashMap<DelayLocation, f64> = HashMap::new();
    for branch in &branches {
        for location in branch.locations() {
            if values.contains_key(&location) {
                continue;
            }
            let value = match &location {
                DelayLocation::Route(index) => routes
                    .as_deref()
                    .and_then(|routes| routes.get(*index))
                    .map(|route| route.delay_ms),
                _ => read_plugin_delay(channels, &location),
            };
            match value {
                Some(value) => {
                    values.insert(location.clone(), value);
                }
                None => advisories.push(format!(
                    "delay_compile: skipped malformed delay element {location:?}"
                )),
            }
        }
    }
    // Per-branch deficits; shared elements count toward every sharing branch.
    let mut deficits = vec![0.0; branches.len()];
    for (index, branch) in branches.iter().enumerate() {
        deficits[index] = branch
            .locations()
            .iter()
            .filter_map(|location| values.get(location))
            .filter(|value| **value < -SNAP_EPSILON_MS)
            .map(|value| -*value)
            .sum();
    }
    let common = deficits
        .iter()
        .fold(0.0_f64, |max, deficit| max.max(*deficit));
    // Zero negatives once each (idempotent), snap dust silently.
    let mut zeroed: HashSet<DelayLocation> = HashSet::new();
    for branch in &branches {
        for location in branch.locations() {
            if !zeroed.insert(location.clone()) {
                continue;
            }
            let Some(value) = values.get(&location).copied() else {
                continue;
            };
            // Zero deficits; negative dust snaps along silently.
            if value < 0.0 {
                write_delay(channels, routes.as_deref_mut(), &location, 0.0);
            }
        }
    }
    // Residual per branch: routes carry it for routed branches, channel
    // plugins (or one appended padding plugin) for standalone branches.
    for (index, branch) in branches.iter().enumerate() {
        let residual = common - deficits[index];
        if residual < SNAP_EPSILON_MS {
            continue;
        }
        match &branch.kind {
            BranchKind::Routed { route_index } => {
                let routes = routes.as_deref_mut().expect("routed branch needs routes");
                routes[*route_index].delay_ms += residual;
            }
            BranchKind::Standalone { channel } => {
                pad_standalone_branch(channels, channel, residual, &mut advisories);
            }
        }
    }
    // Causality net: zero negatives outside replayed stages for playback.
    let mut replay_diverged = false;
    for (name, chain) in channels.iter_mut() {
        for (plugin_index, plugin) in chain.plugins.iter_mut().enumerate() {
            zero_unreplayed_plugin_delay(
                plugin,
                &format!("channel '{name}' plugin {plugin_index}"),
                &mut advisories,
                &mut replay_diverged,
            );
        }
        if let Some(drivers) = chain.drivers.as_mut() {
            for (driver_index, driver) in drivers.iter_mut().enumerate() {
                for (plugin_index, plugin) in driver.plugins.iter_mut().enumerate() {
                    zero_unreplayed_plugin_delay(
                        plugin,
                        &format!("channel '{name}' driver {driver_index} plugin {plugin_index}"),
                        &mut advisories,
                        &mut replay_diverged,
                    );
                }
            }
        }
    }
    if common > 0.0 {
        advisories.push(format!(
            "delay_compile: shifted {} branches by {common:.6} ms common latency",
            branches.len()
        ));
    }
    DelayCompilationReport {
        common_latency_ms: common,
        branches: branches.len(),
        replay_diverged,
        advisories,
    }
}

/// One acoustic branch: an ordered set of delay elements sharing a total.
#[derive(Debug, Clone)]
struct Branch {
    kind: BranchKind,
    route: Option<usize>,
    channel_plugins: Vec<DelayLocation>,
    driver_plugins: Vec<DelayLocation>,
}

#[derive(Debug, Clone)]
enum BranchKind {
    Routed { route_index: usize },
    Standalone { channel: String },
}

impl Branch {
    fn with_route(mut self, route_index: usize) -> Self {
        self.route = Some(route_index);
        self.kind = BranchKind::Routed { route_index };
        self
    }

    fn locations(&self) -> Vec<DelayLocation> {
        let mut locations = Vec::new();
        if let Some(route) = self.route {
            locations.push(DelayLocation::Route(route));
        }
        locations.extend(self.channel_plugins.iter().cloned());
        locations.extend(self.driver_plugins.iter().cloned());
        locations
    }
}

/// Upper bound on electrical filter boost across graph branches.
///
/// Sums positive IIR-element gains (EQ bands, gain trims, route gains)
/// per replay-mirrored branch and returns the maximum. Branches with
/// convolution content are excluded since sidecar taps are unevaluated
/// here; returns `None` when no branch is fully evaluable. Cascade
/// overlap inflates the bound above the realized peak, which is the safe
/// direction for a backstop: the acceptance gate enforces this bound and
/// fails closed when it is unevaluable.
pub(crate) fn electrical_boost_bound_db(
    channels: &HashMap<String, ChannelDspChain>,
    routes: Option<&[BassManagementRoute]>,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut consider = |bound: Option<f64>| {
        if let Some(bound) = bound {
            best = Some(best.map_or(bound, |current: f64| current.max(bound)));
        }
    };
    match routes {
        Some(routes) => {
            for route in routes {
                consider(
                    resolve_branch_endpoints(channels, route).and_then(|endpoints| {
                        branch_gain_bound(channels, &endpoints, route.gain_db)
                    }),
                );
            }
            standalone_channel_names(channels, Some(routes))
                .into_iter()
                .for_each(|name| consider(standalone_gain_bound(channels, name)));
        }
        None => standalone_channel_names(channels, None)
            .into_iter()
            .for_each(|name| consider(standalone_gain_bound(channels, name))),
    }
    best
}

/// Positive-gain sum for one routed branch; `None` when unevaluable.
fn branch_gain_bound(
    channels: &HashMap<String, ChannelDspChain>,
    endpoints: &BranchEndpoints,
    route_gain_db: f64,
) -> Option<f64> {
    if !route_gain_db.is_finite() {
        return None;
    }
    let mut sum = route_gain_db.max(0.0);
    if let Some(name) = endpoints.input.as_ref()
        && let Some(input) = channels.get(name.as_str())
    {
        sum += chain_gain_sum(&input.plugins)?;
    }
    let post = channels.get(endpoints.post.as_str())?;
    sum += chain_gain_sum(&post.plugins)?;
    if let Some(driver) = endpoints.driver {
        sum += chain_gain_sum(&post.drivers.as_ref()?.get(driver)?.plugins)?;
    }
    Some(sum)
}

/// Gain bound for one standalone channel (per driver when present).
fn standalone_gain_bound(channels: &HashMap<String, ChannelDspChain>, name: &str) -> Option<f64> {
    let chain = channels.get(name)?;
    let shared = chain_gain_sum(&chain.plugins)?;
    match chain.drivers.as_ref() {
        Some(drivers) if !drivers.is_empty() => drivers
            .iter()
            .filter_map(|driver| chain_gain_sum(&driver.plugins).map(|bound| shared + bound))
            .fold(None, |max, bound| {
                Some(max.map_or(bound, |current: f64| current.max(bound)))
            }),
        // No drivers (or an empty driver list): the channel is the branch.
        _ => Some(shared),
    }
}

/// Positive IIR-element gains in one plugin list.
///
/// Returns `None` for convolution content, malformed gain, or non-finite
/// gain. Cascade overlap inflates the sum above the realized peak.
fn chain_gain_sum(plugins: &[roomeq_model::PluginConfigWrapper]) -> Option<f64> {
    let mut sum = 0.0;
    for plugin in plugins {
        match plugin.plugin_type.as_str() {
            "gain" => {
                let gain = plugin.parameters.get("gain_db")?.as_f64()?;
                if !gain.is_finite() {
                    return None;
                }
                sum += gain.max(0.0);
            }
            "eq" => {
                for filter in plugin.parameters.get("filters")?.as_array()? {
                    let gain = filter.get("db_gain")?.as_f64()?;
                    if !gain.is_finite() {
                        return None;
                    }
                    sum += gain.max(0.0);
                }
            }
            "convolution" => return None,
            _ => {}
        }
    }
    Some(sum)
}

/// Resolved branch endpoints shared by delay and gain passes.
struct BranchEndpoints {
    input: Option<String>,
    post: String,
    driver: Option<usize>,
}

/// Resolve one route to its branch endpoints, mirroring replay.
///
/// Post chain is the destination channel, else the channel owning the
/// destination driver. Returns `None` for unresolvable routes.
fn resolve_branch_endpoints(
    channels: &HashMap<String, ChannelDspChain>,
    route: &BassManagementRoute,
) -> Option<BranchEndpoints> {
    let (post, driver) = if channels.contains_key(&route.destination) {
        (route.destination.clone(), None)
    } else {
        channels.iter().find_map(|(name, chain)| {
            chain
                .drivers
                .as_ref()?
                .iter()
                .position(|driver| driver.name == route.destination)
                .map(|driver| (name.clone(), Some(driver)))
        })?
    };
    let input = channels
        .contains_key(&route.source_channel)
        .then(|| route.source_channel.clone());
    Some(BranchEndpoints {
        input,
        post,
        driver,
    })
}

/// Channels covered through routes (endpoints or driver owners).
fn fed_channels<'a>(
    channels: &'a HashMap<String, ChannelDspChain>,
    routes: &'a [BassManagementRoute],
) -> HashSet<&'a str> {
    let mut fed = HashSet::new();
    for route in routes {
        fed.insert(route.source_channel.as_str());
        fed.insert(route.destination.as_str());
        if let Some(owner) = driver_owner(channels, &route.destination) {
            fed.insert(owner.as_str());
        }
    }
    fed
}

/// Sorted names of channels no route feeds, for determinism.
fn standalone_channel_names<'a>(
    channels: &'a HashMap<String, ChannelDspChain>,
    routes: Option<&[BassManagementRoute]>,
) -> Vec<&'a String> {
    let fed = routes.map(|routes| fed_channels(channels, routes));
    let mut names: Vec<&String> = channels
        .keys()
        .filter(|name| fed.as_ref().is_none_or(|fed| !fed.contains(name.as_str())))
        .collect();
    names.sort();
    names
}

/// Assemble replay-mirrored branches with their route gains.
///
/// Returns branches with parallel route gains (0.0 for standalone
/// branches). Unresolvable routes are skipped with an advisory.
fn assemble_branches(
    channels: &HashMap<String, ChannelDspChain>,
    routes: Option<&[BassManagementRoute]>,
    advisories: &mut Vec<String>,
) -> (Vec<Branch>, Vec<f64>) {
    let mut branches = Vec::new();
    let mut route_gains = Vec::new();
    if let Some(routes) = routes {
        for (index, route) in routes.iter().enumerate() {
            match routed_branch(channels, route) {
                Some(branch) => {
                    branches.push(branch.with_route(index));
                    route_gains.push(route.gain_db);
                }
                None => advisories.push(format!(
                    "delay_compile: skipped unresolvable route '{}'->'{}'",
                    route.source_channel, route.destination
                )),
            }
        }
    }
    for branch in standalone_branches(channels, routes) {
        branches.push(branch);
        route_gains.push(0.0);
    }
    (branches, route_gains)
}

/// Build the branch for one route, mirroring replay membership.
fn routed_branch(
    channels: &HashMap<String, ChannelDspChain>,
    route: &BassManagementRoute,
) -> Option<Branch> {
    let endpoints = resolve_branch_endpoints(channels, route)?;
    let mut channel_plugins = Vec::new();
    // Input chain pre_route delays (shared across the input's routes).
    if let Some(name) = endpoints.input.as_ref()
        && let Some(input) = channels.get(name.as_str())
    {
        channel_plugins.extend(
            staged_delay_plugins(&route.source_channel, &input.plugins, "pre_route")
                .into_iter()
                .map(|plugin_index| {
                    DelayLocation::ChannelPlugin(route.source_channel.clone(), plugin_index)
                }),
        );
    }
    // Post chain post_route delays (shared across routes into the channel).
    if let Some(post) = channels.get(endpoints.post.as_str()) {
        channel_plugins.extend(
            staged_delay_plugins(&endpoints.post, &post.plugins, "post_route")
                .into_iter()
                .map(|plugin_index| {
                    DelayLocation::ChannelPlugin(endpoints.post.clone(), plugin_index)
                }),
        );
    }
    let mut driver_plugins = Vec::new();
    if let Some(driver) = endpoints.driver {
        driver_plugins.extend(
            all_delay_plugins(
                &channels
                    .get(endpoints.post.as_str())?
                    .drivers
                    .as_ref()?
                    .get(driver)?
                    .plugins,
            )
            .into_iter()
            .map(|plugin_index| {
                DelayLocation::DriverPlugin(endpoints.post.clone(), driver, plugin_index)
            }),
        );
    }
    Some(Branch {
        kind: BranchKind::Standalone {
            channel: endpoints.post,
        },
        route: None,
        channel_plugins,
        driver_plugins,
    })
}

/// Branches for channels no route feeds, by sorted name for determinism.
fn standalone_branches(
    channels: &HashMap<String, ChannelDspChain>,
    routes: Option<&[BassManagementRoute]>,
) -> Vec<Branch> {
    let mut branches = Vec::new();
    for name in standalone_channel_names(channels, routes) {
        let chain = &channels[name.as_str()];
        match chain.drivers.as_ref() {
            Some(drivers) => {
                for (driver_index, driver) in drivers.iter().enumerate() {
                    let mut channel_plugins: Vec<DelayLocation> = all_delay_plugins(&chain.plugins)
                        .into_iter()
                        .map(|plugin_index| {
                            DelayLocation::ChannelPlugin(name.clone(), plugin_index)
                        })
                        .collect();
                    // Deterministic element order: channel plugins first.
                    channel_plugins.sort_by_key(element_key);
                    let mut driver_plugins: Vec<DelayLocation> = all_delay_plugins(&driver.plugins)
                        .into_iter()
                        .map(|plugin_index| {
                            DelayLocation::DriverPlugin(name.clone(), driver_index, plugin_index)
                        })
                        .collect();
                    driver_plugins.sort_by_key(element_key);
                    branches.push(Branch {
                        kind: BranchKind::Standalone {
                            channel: name.clone(),
                        },
                        route: None,
                        channel_plugins,
                        driver_plugins,
                    });
                }
            }
            None => {
                let mut channel_plugins: Vec<DelayLocation> = all_delay_plugins(&chain.plugins)
                    .into_iter()
                    .map(|plugin_index| DelayLocation::ChannelPlugin(name.clone(), plugin_index))
                    .collect();
                channel_plugins.sort_by_key(element_key);
                branches.push(Branch {
                    kind: BranchKind::Standalone {
                        channel: name.clone(),
                    },
                    route: None,
                    channel_plugins,
                    driver_plugins: Vec::new(),
                });
            }
        }
    }
    branches
}

/// Owning channel of a named driver, if any.
fn driver_owner<'a>(
    channels: &'a HashMap<String, ChannelDspChain>,
    driver: &str,
) -> Option<&'a String> {
    channels.iter().find_map(|(name, chain)| {
        chain
            .drivers
            .as_ref()?
            .iter()
            .any(|candidate| candidate.name == driver)
            .then_some(name)
    })
}

/// Stable ordering key for plugin delay elements.
fn element_key(location: &DelayLocation) -> (u8, String, usize, usize) {
    match location {
        DelayLocation::Route(index) => (0, String::new(), *index, 0),
        DelayLocation::ChannelPlugin(channel, plugin) => (1, channel.clone(), *plugin, 0),
        DelayLocation::DriverPlugin(channel, driver, plugin) => {
            (2, channel.clone(), *driver, *plugin)
        }
    }
}

/// Indices of delay plugins carrying one replayed stage.
fn staged_delay_plugins(
    _channel: &str,
    plugins: &[roomeq_model::PluginConfigWrapper],
    stage: &str,
) -> Vec<usize> {
    plugins
        .iter()
        .enumerate()
        .filter(|(_, plugin)| {
            plugin.plugin_type == "delay"
                && plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(|v| v.as_str())
                    == Some(stage)
        })
        .map(|(index, _)| index)
        .collect()
}

/// Indices of all delay plugins in a chain or driver.
fn all_delay_plugins(plugins: &[roomeq_model::PluginConfigWrapper]) -> Vec<usize> {
    plugins
        .iter()
        .enumerate()
        .filter(|(_, plugin)| plugin.plugin_type == "delay")
        .map(|(index, _)| index)
        .collect()
}

/// Read one delay element; `None` for malformed or non-finite values.
fn read_plugin_delay(
    channels: &HashMap<String, ChannelDspChain>,
    location: &DelayLocation,
) -> Option<f64> {
    let plugins = match location {
        DelayLocation::Route(_) => return None,
        DelayLocation::ChannelPlugin(channel, _) => &channels.get(channel.as_str())?.plugins,
        DelayLocation::DriverPlugin(channel, driver, _) => {
            &channels
                .get(channel.as_str())?
                .drivers
                .as_ref()?
                .get(*driver)?
                .plugins
        }
    };
    let index = match location {
        DelayLocation::ChannelPlugin(_, index) | DelayLocation::DriverPlugin(_, _, index) => *index,
        DelayLocation::Route(_) => return None,
    };
    let value = plugins.get(index)?.parameters.get("delay_ms")?.as_f64()?;
    value.is_finite().then_some(value)
}

/// Write one delay element, replacing its millisecond value.
fn write_delay(
    channels: &mut HashMap<String, ChannelDspChain>,
    routes: Option<&mut Vec<BassManagementRoute>>,
    location: &DelayLocation,
    value: f64,
) {
    match location {
        DelayLocation::Route(index) => {
            if let Some(routes) = routes
                && let Some(route) = routes.get_mut(*index)
            {
                route.delay_ms = value;
            }
        }
        DelayLocation::ChannelPlugin(channel, plugin_index) => {
            if let Some(chain) = channels.get_mut(channel.as_str())
                && let Some(plugin) = chain.plugins.get_mut(*plugin_index)
                && let Some(parameters) = plugin.parameters.as_object_mut()
            {
                parameters.insert("delay_ms".to_string(), serde_json::json!(value));
            }
        }
        DelayLocation::DriverPlugin(channel, driver_index, plugin_index) => {
            if let Some(chain) = channels.get_mut(channel.as_str())
                && let Some(drivers) = chain.drivers.as_mut()
                && let Some(driver) = drivers.get_mut(*driver_index)
                && let Some(plugin) = driver.plugins.get_mut(*plugin_index)
                && let Some(parameters) = plugin.parameters.as_object_mut()
            {
                parameters.insert("delay_ms".to_string(), serde_json::json!(value));
            }
        }
    }
}

/// Add a residual to a standalone branch without a route.
fn pad_standalone_branch(
    channels: &mut HashMap<String, ChannelDspChain>,
    channel: &str,
    residual_ms: f64,
    advisories: &mut Vec<String>,
) {
    let Some(chain) = channels.get_mut(channel) else {
        advisories.push(format!(
            "delay_compile: cannot pad missing channel '{channel}'"
        ));
        return;
    };
    if let Some(plugin) = chain
        .plugins
        .iter_mut()
        .find(|plugin| plugin.plugin_type == "delay")
    {
        let current = plugin
            .parameters
            .get("delay_ms")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if let Some(parameters) = plugin.parameters.as_object_mut() {
            parameters.insert(
                "delay_ms".to_string(),
                serde_json::json!(current + residual_ms),
            );
        }
        return;
    }
    // No delay plugin exists: append marked padding, mirroring non-routed
    // time alignment (replay applies whole chains there). The label lets
    // baseline restoration strip compile-derived timing from refused graphs
    // without touching structural delays.
    let mut padding = roomeq_engine::output::create_delay_plugin(residual_ms);
    if let Some(parameters) = padding.parameters.as_object_mut() {
        parameters.insert(
            "label".to_string(),
            serde_json::json!(COMPILE_PADDING_LABEL),
        );
    }
    chain.plugins.push(padding);
    advisories.push(format!(
        "delay_compile: appended {residual_ms:.6} ms padding to channel '{channel}'"
    ));
}

/// Zero a delay outside the replayed stages, recording the divergence.
fn zero_unreplayed_plugin_delay(
    plugin: &mut roomeq_model::PluginConfigWrapper,
    context: &str,
    advisories: &mut Vec<String>,
    replay_diverged: &mut bool,
) {
    if plugin.plugin_type != "delay" {
        return;
    }
    let Some(value) = plugin.parameters.get("delay_ms").and_then(|v| v.as_f64()) else {
        return;
    };
    if !value.is_finite() || value >= -SNAP_EPSILON_MS {
        return;
    }
    if let Some(parameters) = plugin.parameters.as_object_mut() {
        parameters.insert("delay_ms".to_string(), serde_json::json!(0.0));
    }
    advisories.push(format!(
        "delay_compile: zeroed {value:.6} ms outside replayed stages ({context}); replay diverges from playback here"
    ));
    *replay_diverged = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delay_plugin(delay_ms: f64) -> roomeq_model::PluginConfigWrapper {
        roomeq_engine::output::create_delay_plugin(delay_ms)
    }

    fn staged_delay_plugin(delay_ms: f64, stage: &str) -> roomeq_model::PluginConfigWrapper {
        roomeq_engine::topology::mark_plugin_stage(delay_plugin(delay_ms), stage)
    }

    fn chain_with_drivers(
        plugin_delays: &[(f64, &str)],
        drivers: &[(&str, f64)],
    ) -> ChannelDspChain {
        ChannelDspChain {
            channel: String::new(),
            plugins: plugin_delays
                .iter()
                .map(|(delay, stage)| staged_delay_plugin(*delay, stage))
                .collect(),
            drivers: Some(
                drivers
                    .iter()
                    .enumerate()
                    .map(|(index, (name, delay))| roomeq_model::DriverDspChain {
                        name: name.to_string(),
                        index,
                        plugins: vec![delay_plugin(*delay)],
                        initial_curve: None,
                        measured_band_hz: None,
                    })
                    .collect(),
            ),
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
            joint_sub: None,
            early_late_curves: None,
            early_reflections: None,
            waterfall: None,
            resonance_decays: None,
            wavelet: None,
            t60_octaves: None,
        }
    }

    fn route(source: &str, destination: &str, delay_ms: f64) -> BassManagementRoute {
        BassManagementRoute {
            group_id: None,
            source_channel: source.to_string(),
            source_index: 0,
            destination: destination.to_string(),
            destination_index: 0,
            pre_chain_channel: None,
            post_chain_channel: None,
            route_kind: "test".to_string(),
            crossover_type: "LR24".to_string(),
            high_pass_hz: None,
            low_pass_hz: None,
            gain_db: 0.0,
            gain_linear: 1.0,
            matrix_gain: 1.0,
            delay_ms,
            polarity_inverted: false,
        }
    }

    fn all_delays(
        channels: &HashMap<String, ChannelDspChain>,
        routes: &[BassManagementRoute],
    ) -> Vec<f64> {
        let mut delays: Vec<f64> = routes.iter().map(|route| route.delay_ms).collect();
        let mut names: Vec<&String> = channels.keys().collect();
        names.sort();
        for name in names {
            let chain = &channels[name.as_str()];
            delays.extend(chain.plugins.iter().filter_map(|plugin| {
                (plugin.plugin_type == "delay")
                    .then(|| plugin.parameters.get("delay_ms")?.as_f64())
                    .flatten()
            }));
            if let Some(drivers) = chain.drivers.as_ref() {
                for driver in drivers {
                    delays.extend(driver.plugins.iter().filter_map(|plugin| {
                        (plugin.plugin_type == "delay")
                            .then(|| plugin.parameters.get("delay_ms")?.as_f64())
                            .flatten()
                    }));
                }
            }
        }
        delays
    }

    #[test]
    fn sigberg3_advances_compile_causal_with_common_offset() {
        // Study F4 values: L sub -7.644 ms, R sub -0.701 ms, R main -17.051 ms.
        let mut channels = HashMap::from([
            ("L".to_string(), chain_with_drivers(&[], &[])),
            ("R".to_string(), chain_with_drivers(&[], &[])),
            (
                "Sub".to_string(),
                chain_with_drivers(&[], &[("L-sub", -7.644), ("R-sub", -0.701)]),
            ),
        ]);
        // R main advance on a second R driver.
        channels.get_mut("R").unwrap().drivers = Some(vec![roomeq_model::DriverDspChain {
            name: "R-main".to_string(),
            index: 0,
            plugins: vec![delay_plugin(-17.051)],
            initial_curve: None,
            measured_band_hz: None,
        }]);
        let mut routes = vec![
            route("L", "L-sub", 0.0),
            route("R", "R-sub", 0.0),
            route("R", "R-main", 0.0),
        ];
        let before: Vec<f64> = [-7.644, -0.701, -17.051]
            .into_iter()
            .chain(routes.iter().map(|route| route.delay_ms))
            .collect();
        let _ = before;
        let report = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        assert!((report.common_latency_ms - 17.051).abs() < 1e-9);
        assert_eq!(report.branches, 3);
        assert!(!report.replay_diverged);
        for delay in all_delays(&channels, &routes) {
            assert!(delay >= 0.0, "negative delay remains: {delay}");
        }
        // Uniform shift: every branch total rose by exactly the offset.
        let totals: Vec<f64> = [0.0 - 7.644, 0.0 - 0.701, 0.0 - 17.051]
            .into_iter()
            .map(|total| total + report.common_latency_ms)
            .collect();
        assert!((totals[0] - 9.407).abs() < 1e-9);
        assert!((totals[1] - 16.35).abs() < 1e-9);
        assert!((totals[2] - 0.0).abs() < 1e-9);
        // Residuals landed on routes: drivers read zero.
        assert!((routes[0].delay_ms - 9.407).abs() < 1e-9);
        assert!((routes[1].delay_ms - 16.35).abs() < 1e-9);
        assert!((routes[2].delay_ms - 0.0).abs() < 1e-9);
    }

    #[test]
    fn already_causal_graph_is_untouched() {
        let mut channels = HashMap::from([(
            "L".to_string(),
            chain_with_drivers(&[(2.5, "post_route")], &[("L-sub", 1.0)]),
        )]);
        let mut routes = vec![route("L", "L-sub", 3.0)];
        let report = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        assert_eq!(report.common_latency_ms, 0.0);
        assert!(report.advisories.is_empty());
        assert!(!report.replay_diverged);
        assert_eq!(routes[0].delay_ms, 3.0);
    }

    #[test]
    fn unreplayed_negative_delay_flags_replay_divergence() {
        // Routed input whose delay plugin carries no pre_route stage mark:
        // outside branch membership, so the causality net zeroes it and the
        // pre-compile splice verdict no longer transfers to playback.
        let mut input = chain_with_drivers(&[], &[]);
        input.plugins.push(delay_plugin(-2.0));
        let mut channels = HashMap::from([
            ("L".to_string(), input),
            (
                "Sub".to_string(),
                chain_with_drivers(&[], &[("L-sub", 1.0)]),
            ),
        ]);
        let mut routes = vec![route("L", "L-sub", 0.0)];
        let report = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        assert!(report.replay_diverged);
        assert!(
            report
                .advisories
                .iter()
                .any(|advisory| advisory.contains("replay diverges from playback"))
        );
        assert_eq!(
            channels["L"].plugins[0].parameters["delay_ms"],
            serde_json::json!(0.0)
        );
    }

    #[test]
    fn shared_channel_negative_compiles_uniformly() {
        // One negative channel plugin shared by two routes into Sub.
        let mut channels = HashMap::from([
            ("L".to_string(), chain_with_drivers(&[], &[])),
            ("R".to_string(), chain_with_drivers(&[], &[])),
            (
                "Sub".to_string(),
                chain_with_drivers(&[(-4.0, "post_route")], &[("L-sub", 1.0), ("R-sub", 2.0)]),
            ),
        ]);
        let mut routes = vec![route("L", "L-sub", 0.0), route("R", "R-sub", 0.0)];
        let report = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        // Branch deficits: L: 4.0, R: 4.0 -> offset 4.0, residuals zero.
        assert!((report.common_latency_ms - 4.0).abs() < 1e-9);
        assert_eq!(routes[0].delay_ms, 0.0);
        assert_eq!(routes[1].delay_ms, 0.0);
        for delay in all_delays(&channels, &routes) {
            assert!(delay >= 0.0, "negative delay remains: {delay}");
        }
    }

    #[test]
    fn nonrouted_branch_pads_without_route() {
        // L defines the offset; R needs residual padding without any route.
        let mut channels = HashMap::from([
            (
                "L".to_string(),
                chain_with_drivers(&[], &[("L-woofer", -5.0)]),
            ),
            (
                "R".to_string(),
                chain_with_drivers(&[], &[("R-woofer", -1.0)]),
            ),
        ]);
        let report = compile_graph_delays_causal(&mut channels, None);
        assert!((report.common_latency_ms - 5.0).abs() < 1e-9);
        assert_eq!(report.branches, 2);
        for delay in all_delays(&channels, &[]) {
            assert!(delay >= 0.0, "negative delay remains: {delay}");
        }
        // R's residual (5 - 1) lands on an appended channel plugin.
        let padding = channels["R"]
            .plugins
            .iter()
            .filter_map(|plugin| plugin.parameters.get("delay_ms")?.as_f64())
            .sum::<f64>();
        assert!((padding - 4.0).abs() < 1e-9);
        // The appended padding is marked so baseline restoration can strip
        // compile-derived timing from refused graphs.
        assert_eq!(channels["R"].plugins.len(), 1);
        assert_eq!(
            channels["R"].plugins[0]
                .parameters
                .get("label")
                .and_then(serde_json::Value::as_str),
            Some(COMPILE_PADDING_LABEL)
        );
        // L needed no residual: zeroing defined the whole shift.
        assert!(channels["L"].plugins.is_empty());
    }

    #[test]
    fn compilation_is_idempotent() {
        let mut channels = HashMap::from([(
            "Sub".to_string(),
            chain_with_drivers(&[], &[("L-sub", -3.25)]),
        )]);
        let mut routes = vec![route("L", "L-sub", 1.0)];
        // Input chain L is absent: branch still resolves (no input delays).
        let first = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        assert!(first.common_latency_ms > 0.0);
        let snapshot = all_delays(&channels, &routes);
        let second = compile_graph_delays_causal(&mut channels, Some(&mut routes));
        assert_eq!(second.common_latency_ms, 0.0);
        assert!(second.advisories.is_empty());
        assert_eq!(all_delays(&channels, &routes), snapshot);
    }

    fn gain_plugin(gain_db: f64) -> roomeq_model::PluginConfigWrapper {
        roomeq_model::PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: serde_json::json!({"gain_db": gain_db}),
        }
    }

    fn eq_plugin(gains_db: &[f64]) -> roomeq_model::PluginConfigWrapper {
        roomeq_model::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({
                "filters": gains_db
                    .iter()
                    .map(|gain| serde_json::json!({"db_gain": gain}))
                    .collect::<Vec<_>>(),
            }),
        }
    }

    fn convolution_plugin() -> roomeq_model::PluginConfigWrapper {
        roomeq_model::PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: serde_json::json!({"ir_file": "sidecar.wav"}),
        }
    }

    fn chain_with_gains(
        plugins: Vec<roomeq_model::PluginConfigWrapper>,
        drivers: &[(&str, Vec<roomeq_model::PluginConfigWrapper>)],
    ) -> ChannelDspChain {
        let mut chain = chain_with_drivers(&[], &[]);
        chain.plugins = plugins;
        chain.drivers = Some(
            drivers
                .iter()
                .enumerate()
                .map(|(index, (name, plugins))| roomeq_model::DriverDspChain {
                    name: name.to_string(),
                    index,
                    plugins: plugins.clone(),
                    initial_curve: None,
                    measured_band_hz: None,
                })
                .collect(),
        );
        chain
    }

    #[test]
    fn electrical_bound_sums_positive_branch_gains() {
        // Route +2, channel eq(+3,-2) + gain(+1.5), driver eq(+1) = 7.5.
        let channels = HashMap::from([
            ("L".to_string(), chain_with_gains(vec![], &[])),
            (
                "Sub".to_string(),
                chain_with_gains(
                    vec![eq_plugin(&[3.0, -2.0]), gain_plugin(1.5)],
                    &[("L-sub", vec![eq_plugin(&[1.0])])],
                ),
            ),
        ]);
        let routes = vec![route("L", "L-sub", 0.0)];
        let mut routes = routes;
        routes[0].gain_db = 2.0;
        let bound = electrical_boost_bound_db(&channels, Some(&routes)).unwrap();
        assert!((bound - 7.5).abs() < 1e-9, "bound was {bound}");
    }

    #[test]
    fn electrical_bound_excludes_convolution_branches() {
        let channels = HashMap::from([
            (
                "L".to_string(),
                chain_with_gains(vec![convolution_plugin()], &[]),
            ),
            (
                "R".to_string(),
                chain_with_gains(vec![gain_plugin(4.0)], &[]),
            ),
        ]);
        // L unevaluable (taps), R bounds 4.0.
        let bound = electrical_boost_bound_db(&channels, None).unwrap();
        assert!((bound - 4.0).abs() < 1e-9, "bound was {bound}");
        // All-convolution graphs report None rather than zero.
        let fir_only = HashMap::from([(
            "L".to_string(),
            chain_with_gains(vec![convolution_plugin()], &[]),
        )]);
        assert_eq!(electrical_boost_bound_db(&fir_only, None), None);
    }

    #[test]
    fn electrical_bound_takes_branch_maximum() {
        let channels = HashMap::from([
            (
                "L".to_string(),
                chain_with_gains(vec![gain_plugin(2.0)], &[]),
            ),
            (
                "R".to_string(),
                chain_with_gains(vec![gain_plugin(6.0)], &[]),
            ),
        ]);
        let bound = electrical_boost_bound_db(&channels, None).unwrap();
        assert!((bound - 6.0).abs() < 1e-9, "bound was {bound}");
    }
}
