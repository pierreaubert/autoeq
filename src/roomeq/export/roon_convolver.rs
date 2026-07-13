use super::super::types::DspChainOutput;
use super::collect::collect_all_plugins;
use super::extract::extract_convolution_paths;
use anyhow::Context;
use std::collections::HashSet;
use std::io::{Cursor, Write};
use std::path::{Component, Path};
use zip::write::FileOptions;

#[derive(Debug)]
pub(super) struct RoonConvolutionArchive {
    pub(super) sample_rate: u32,
    pub(super) channel_count: usize,
    pub(super) channel_mask_hex: String,
    pub(super) config_name: String,
}

struct ArchiveChannel {
    name: String,
    mask: u32,
    wav_bytes: Option<Vec<u8>>,
}

pub(super) fn package_roon_convolution_archive(
    output: &DspChainOutput,
    source_dir: &Path,
    archive_path: &Path,
    sample_rate: f64,
) -> anyhow::Result<Option<RoonConvolutionArchive>> {
    if !sample_rate.is_finite()
        || sample_rate <= 0.0
        || sample_rate.fract() != 0.0
        || sample_rate > u32::MAX as f64
    {
        anyhow::bail!("Roon convolution export requires an integral positive sample rate");
    }
    let sample_rate = sample_rate as u32;

    let mut channels = Vec::new();
    let mut has_convolution = false;
    let mut masks = HashSet::new();
    for (channel_name, chain) in &output.channels {
        let (mask, canonical_name) = roon_wave_channel(channel_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Roon convolution export does not know the WAVE channel mapping for '{channel_name}'"
            )
        })?;
        if !masks.insert(mask) {
            anyhow::bail!(
                "Roon convolution export has duplicate WAVE channel mapping for '{channel_name}'"
            );
        }
        let plugins: Vec<_> = collect_all_plugins(chain).into_iter().cloned().collect();
        let paths = extract_convolution_paths(&plugins);
        if paths.len() > 1 {
            anyhow::bail!(
                "Roon convolution export supports one impulse per channel; '{channel_name}' has {}",
                paths.len()
            );
        }
        has_convolution |= !paths.is_empty();
        let wav_bytes = paths
            .first()
            .map(|path| read_and_validate_wav(path, source_dir, sample_rate))
            .transpose()?;
        channels.push(ArchiveChannel {
            name: canonical_name.to_string(),
            mask,
            wav_bytes,
        });
    }
    if !has_convolution {
        return Ok(None);
    }

    channels.sort_by_key(|channel| channel.mask);
    let channel_mask = channels
        .iter()
        .fold(0_u32, |mask, channel| mask | channel.mask);
    let supplied_lengths: Vec<_> = channels
        .iter()
        .filter_map(|channel| {
            channel
                .wav_bytes
                .as_ref()
                .map(|bytes| wav_frame_count(bytes))
        })
        .collect::<anyhow::Result<_>>()?;
    let ir_length = supplied_lengths[0];
    if supplied_lengths.iter().any(|length| *length != ir_length) {
        anyhow::bail!("Roon convolution export requires equal impulse-response lengths");
    }

    let config_name = format!("room_eq_{sample_rate}_{}ch.cfg", channels.len());
    let mut config = format!(
        "{sample_rate} {} {} {:X}\n",
        channels.len(),
        channels.len(),
        channel_mask
    );
    config.push_str(
        &std::iter::repeat_n("0", channels.len() * 2)
            .collect::<Vec<_>>()
            .join(" "),
    );
    config.push('\n');
    for (index, channel) in channels.iter().enumerate() {
        let wav_name = archive_wav_name(index, &channel.name);
        config.push_str(&format!("{wav_name}\n0\n{index}.0\n{index}.0\n"));
    }

    let options = FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .last_modified_time(zip::DateTime::default())
        .unix_permissions(0o644);
    let cursor = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    zip.start_file(&config_name, options)?;
    zip.write_all(config.as_bytes())?;
    for (index, channel) in channels.iter().enumerate() {
        zip.start_file(archive_wav_name(index, &channel.name), options)?;
        if let Some(bytes) = &channel.wav_bytes {
            zip.write_all(bytes)?;
        } else {
            zip.write_all(&identity_wav(sample_rate, ir_length))?;
        }
    }
    let bytes = zip.finish()?.into_inner();
    std::fs::write(archive_path, bytes).with_context(|| {
        format!(
            "failed to write Roon convolution archive '{}'",
            archive_path.display()
        )
    })?;

    Ok(Some(RoonConvolutionArchive {
        sample_rate,
        channel_count: channels.len(),
        channel_mask_hex: format!("{channel_mask:X}"),
        config_name,
    }))
}

fn roon_wave_channel(name: &str) -> Option<(u32, &'static str)> {
    match name {
        "left" => Some((0x0001, "L")),
        "right" => Some((0x0002, "R")),
        "center" => Some((0x0004, "C")),
        "lfe" | "sub" | "subwoofer" => Some((0x0008, "LFE")),
        "back_left" => Some((0x0010, "BL")),
        "back_right" => Some((0x0020, "BR")),
        "surround_left" => Some((0x0200, "SL")),
        "surround_right" => Some((0x0400, "SR")),
        _ => None,
    }
}

fn archive_wav_name(index: usize, name: &str) -> String {
    format!("filters/{index:02}_{name}.wav")
}

fn read_and_validate_wav(
    ir_file: &str,
    source_dir: &Path,
    sample_rate: u32,
) -> anyhow::Result<Vec<u8>> {
    let relative = Path::new(ir_file);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!("Roon convolution impulse path '{ir_file}' must be a safe relative path");
    }
    if relative
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        != Some("wav".to_string())
    {
        anyhow::bail!("Roon convolution impulse '{ir_file}' must be a WAV file");
    }
    let root = source_dir.canonicalize().with_context(|| {
        format!(
            "failed to resolve convolution source directory '{}'",
            source_dir.display()
        )
    })?;
    let path = root
        .join(relative)
        .canonicalize()
        .with_context(|| format!("Roon convolution impulse '{ir_file}' was not found"))?;
    if !path.starts_with(&root) {
        anyhow::bail!("Roon convolution impulse '{ir_file}' escapes its source directory");
    }
    let bytes = std::fs::read(&path)?;
    let cursor = Cursor::new(&bytes);
    let mut reader = hound::WavReader::new(cursor)
        .with_context(|| format!("Roon convolution impulse '{ir_file}' is not a valid WAV"))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("Roon convolution impulse '{ir_file}' must be mono");
    }
    if spec.sample_rate != sample_rate {
        anyhow::bail!(
            "Roon convolution impulse '{ir_file}' has sample rate {}, expected {sample_rate}",
            spec.sample_rate
        );
    }
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                sample.with_context(|| format!("malformed samples in '{ir_file}'"))?;
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>() {
                sample.with_context(|| format!("malformed samples in '{ir_file}'"))?;
            }
        }
    }
    Ok(bytes)
}

fn wav_frame_count(bytes: &[u8]) -> anyhow::Result<u32> {
    let reader = hound::WavReader::new(Cursor::new(bytes))?;
    Ok(reader.duration())
}

fn identity_wav(sample_rate: u32, frames: u32) -> Vec<u8> {
    let data_size = frames * 4;
    let mut wav = Vec::with_capacity(44 + data_size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_size).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&3_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    wav.extend_from_slice(&4_u16.to_le_bytes());
    wav.extend_from_slice(&32_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&1.0_f32.to_le_bytes());
    wav.resize(44 + data_size as usize, 0);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_channel_masks_follow_canonical_order() {
        assert_eq!(roon_wave_channel("left"), Some((0x1, "L")));
        assert_eq!(roon_wave_channel("back_left"), Some((0x10, "BL")));
        assert_eq!(roon_wave_channel("surround_left"), Some((0x200, "SL")));
    }

    #[test]
    fn identity_wav_is_mono_and_has_requested_shape() {
        let wav = identity_wav(48_000, 128);
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 48_000);
        assert_eq!(reader.duration(), 128);
    }
}
