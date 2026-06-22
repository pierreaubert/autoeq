use super::super::*;

#[derive(Debug, Clone)]
pub(in super::super) struct RoleChannelMatchingGroup {
    pub(in super::super) role_key: &'static str,
    pub(in super::super) curves: HashMap<String, crate::Curve>,
}

#[derive(Debug, Clone, Copy)]
pub(in super::super) struct RoleChannelMatchingProfile {
    pub(in super::super) rms_threshold_db: f64,
    pub(in super::super) correction:
        crate::roomeq::spectral_align::ChannelMatchingCorrectionProfile,
}

pub(in super::super) fn channel_matching_role_key(channel_name: &str) -> Option<&'static str> {
    crate::roomeq::home_cinema::matching_group_key(channel_name)
}
