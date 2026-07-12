# RoomEQ acoustic corpus provenance

The corpus is stored in the repository so every quality decision can be
reproduced offline. A scenario must use an opaque identifier and must not add a
person's name, postal address, room photograph, device serial number, or
embedded location metadata.

## Sources and redistribution

| Scenario family | Source | Rights classification | Privacy review |
|---|---|---|---|
| fem_* | Deterministic finite-element fixtures under ../generate/fem/ | Generated project test data; covered by the repository license | Contains numeric curves/configuration only; no personal data |
| measured_stereo_8361a | Contributor-supplied stereo room capture under ../measured/2.0_8361a/ | LicenseRef-SOTF-Project-Test-Data | Opaque room ID; CSV/WAV and channel names only |
| measured_stereo_d3v | Contributor-supplied stereo room capture under ../measured/2.0_d3v/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; CSV/WAV and channel names only |
| measured_stereo_t7v | Contributor-supplied stereo room capture under ../measured/2.0_t7v/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; CSV/WAV and channel names only |

LicenseRef-SOTF-Project-Test-Data means the files are retained and exercised
as part of this repository's test suite. It is not a grant to extract and
redistribute the recordings as a separate dataset. A maintainer must confirm
the contributor's rights before adding another real capture.

## Intake checklist

Before committing a real measurement:

1. Confirm the contributor created the recording or has permission to share it.
2. Strip names, addresses, geolocation, free-form notes, and device serials.
3. Use opaque scenario and directory identifiers.
4. Record the source family, rights classification, and privacy result here.
5. Add at least one held-out position where the capture contains multiple seats.
6. Run the PR corpus twice and confirm byte-identical JSON output.

The unused ../measured/5_1_kef/*.mdat capture is not in the acoustic corpus.
The privacy-safe converter can recover its seven channel labels and numeric
SPL/phase curves without exporting embedded note bodies, but contributor rights
remain unverified and the file contains only one listening position. It cannot
serve as held-out or multi-seat evidence until a maintainer completes the
rights review and obtains additional positions.
