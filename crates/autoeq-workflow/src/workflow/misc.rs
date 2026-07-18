use crate::Cea2034Data;
use crate::Curve;
use crate::read;
use ndarray::Array1;
use std::collections::HashMap;

/// Interpolate all curves in Cea2034Data to a standard frequency grid
/// Note: Does NOT normalize - preserves original dB levels for proper visualization
pub(super) fn interpolate_cea2034_data(
    spin_data: &Cea2034Data,
    standard_freq: &Array1<f64>,
) -> Cea2034Data {
    let interpolate = |curve: &Curve| read::interpolate_response(standard_freq, curve);

    let on_axis = interpolate(&spin_data.on_axis);
    let listening_window = interpolate(&spin_data.listening_window);
    let early_reflections = interpolate(&spin_data.early_reflections);
    let sound_power = interpolate(&spin_data.sound_power);
    let estimated_in_room = interpolate(&spin_data.estimated_in_room);
    let er_di = interpolate(&spin_data.er_di);
    let sp_di = interpolate(&spin_data.sp_di);

    // Build interpolated curves HashMap
    let mut curves = HashMap::new();
    curves.insert("On Axis".to_string(), on_axis.clone());
    curves.insert("Listening Window".to_string(), listening_window.clone());
    curves.insert("Early Reflections".to_string(), early_reflections.clone());
    curves.insert("Sound Power".to_string(), sound_power.clone());
    curves.insert(
        "Estimated In-Room Response".to_string(),
        estimated_in_room.clone(),
    );

    Cea2034Data {
        on_axis,
        listening_window,
        early_reflections,
        sound_power,
        estimated_in_room,
        er_di,
        sp_di,
        curves,
    }
}
