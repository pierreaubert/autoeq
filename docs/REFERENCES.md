<!-- markdownlint-disable-file MD013 -->

# References

Foundational papers and standards behind the algorithms in `autoeq`. Citations are restricted to references actually invoked in the `src/` source comments — every entry below maps to at least one file/symbol in the crate.

## Source-file index

| Source file | References used |
|---|---|
| `cea2034.rs`, `loss/speaker.rs`, `loss/types.rs` | [CEA/CTA-2034](#ceacta-2034-spinorama-standard), [Olive preference score](#olive-speaker-preference-score--harman-target) |
| `loss/headphone.rs`, `loss/slope.rs` | [Olive/Welti/McMullin headphone preference](#olivewelti-headphone-preference-model) |
| `loss/bass_boost.rs` | [Harman in-room target (Olive / Toole)](#harman-in-room-target--bass-boost) |
| `loss/phase_aware.rs` | [Phase-coherent EQ / phase audibility](#phase-coherent-eq--phase-audibility) |
| `loss/enhanced_weights.rs` | [ERB scale](#erb--cochlear-bandwidth), [perceptual EQ weighting](#perceptual-eq-weighting) |
| `loss/epa/bark.rs`, `loudness.rs`, `sharpness.rs`, `roughness.rs`, `score.rs` | [Zwicker / Fastl psychoacoustics](#zwicker--fastl-psychoacoustic-metrics-bark-loudness-sharpness-roughness), [ISO 226](#iso-226-equal-loudness-contours), [DIN 45692](#din-45692-sharpness), [Osgood semantic differentials (EPA)](#osgood-semantic-differential-evaluationpotencyactivity) |
| `loss/epa/cdt.rs` | [Cubic distortion tones](#cubic-distortion-tones-cdt) |
| `roomeq/spatial_robustness.rs`, `roomeq/mixed_phase.rs` | [Brännmark & Sternad — robust room correction + EP2104374B1](#brännmark--sternad--robust-room-correction) |
| `roomeq/impulse_analysis.rs` | [Laborie, Bruno & Montoya — decomposed correction](#laborie-bruno--montoya--decomposed-room-correction) |
| `roomeq/reflection_cancel.rs` | [Johnston — first-reflection cancellation](#johnston--first-reflection-cancellation) |
| `roomeq/group_processing.rs`, `roomeq/multiseat.rs`, `roomeq/multisub.rs`, `roomeq/types/config.rs`, `roomeq/workflows/bass_management.rs` | [Multi-Sub Optimizer (MSO)](#multi-sub-optimizer-mso) |
| `roomeq/multiseat.rs`, `roomeq/types/config.rs` | [Acoustic multi-channel equalization robustness](#acoustic-multi-channel-equalization-robustness) |
| `roomeq/eq.rs`, `roomeq/speaker_eq.rs` | [Schroeder frequency](#schroeder-frequency--mode-density-cutoff) |
| `optim/de.rs`, `optim/init_sobol.rs` | [Sobol quasi-random sequences](#sobol-quasi-random-sequences) |
| `optim/cobyla.rs`, `optim/isres.rs`, `optim/de.rs` | See [`math-optimisation/REFERENCES.md`](../math-audio/math-optimisation/REFERENCES.md) for COBYLA, ISRES, DE/JADE, Levenberg-Marquardt |
| `optim/pareto.rs` | [Multi-objective EQ — NSGA-II](#multi-objective-genetic-algorithms-for-eq) |
| `fir.rs`, `roomeq/fir.rs` | See [`math-iir-fir/REFERENCES.md`](../math-audio/math-iir-fir/REFERENCES.md) for Kirkeby + pre-ringing |

## CEA/CTA-2034 spinorama standard

`cea2034.rs` computes the on-axis, listening-window, sound-power, and PIR curves and the NBD / LFX / SM PIR sub-metrics that feed the preference score.

```bibtex
@techreport{cta2034,
  author      = {{Consumer Technology Association}},
  title       = {{ANSI/CTA-2034-A — Standard Method of Measurement for In-Home Loudspeakers}},
  institution = {CTA},
  year        = {2015},
  note        = {Originally CEA-2034. Defines the spinorama capture and curve-derivation method.}
}
```

## Olive speaker preference score / Harman target

`cea2034::compute_metrics`, `loss/speaker.rs::speaker_preference_score`, and the `score`-mode CLI presets implement the Olive preference model.

```bibtex
@inproceedings{olive2004multiple,
  author    = {Olive, Sean E.},
  title     = {A Multiple Regression Model for Predicting Loudspeaker Preference Using Objective Measurements: Part {II} — Development of the Model},
  booktitle = {Audio Engineering Society Convention 117},
  year      = {2004},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=12847}
}

@inproceedings{olive2004part1,
  author    = {Olive, Sean E.},
  title     = {A Multiple Regression Model for Predicting Loudspeaker Preference Using Objective Measurements: Part {I} — Listening Test Results},
  booktitle = {Audio Engineering Society Convention 116},
  year      = {2004},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=12794}
}

@book{toole2017sound,
  author    = {Toole, Floyd E.},
  title     = {Sound Reproduction: The Acoustics and Psychoacoustics of Loudspeakers and Rooms},
  edition   = {3},
  publisher = {Routledge},
  year      = {2017},
  isbn      = {978-1138921368}
}
```

## Olive/Welti headphone preference model

`loss/headphone.rs` and `loss/slope.rs` apply the equation `pref = 114.49 - 12.62·SD - 15.52·AS` over 50 Hz–10 kHz.

```bibtex
@inproceedings{olive2013headphone,
  author    = {Olive, Sean E. and Welti, Todd and McMullin, Elisabeth},
  title     = {A Statistical Model that Predicts Listeners' Preference Ratings of Around-Ear and On-Ear Headphones},
  booktitle = {Audio Engineering Society Convention 135},
  year      = {2013},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=16980}
}

@inproceedings{olive2018harman,
  author    = {Olive, Sean E. and Welti, Todd and Khonsaripour, Omid},
  title     = {A Statistical Model That Predicts Listeners' Preference Ratings of In-Ear Headphones: Part 1 — Listening Test Results and Acoustic Measurements},
  booktitle = {Audio Engineering Society Convention 144},
  year      = {2018},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=19436}
}
```

## Harman in-room target / bass boost

`loss/bass_boost.rs::BassBoostCurve::Harman` cites these for the in-room target shape.

```bibtex
@inproceedings{olive2013target,
  author    = {Olive, Sean E. and Welti, Todd and McMullin, Elisabeth},
  title     = {The Influence of Listeners' Experience, Age, and Culture on Headphone Sound Quality Preferences},
  booktitle = {Audio Engineering Society Convention 137},
  year      = {2014},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=17467}
}

@article{toole2015room,
  author  = {Toole, Floyd E.},
  title   = {The Measurement and Calibration of Sound Reproducing Systems},
  journal = {Journal of the Audio Engineering Society},
  volume  = {63},
  number  = {7/8},
  pages   = {512--541},
  year    = {2015},
  doi     = {10.17743/jaes.2015.0064}
}
```

## Phase-coherent EQ / phase audibility

`loss/phase_aware.rs` — phase-aware speaker EQ.

```bibtex
@inproceedings{klein2017phase,
  author    = {Klein, Joachim and Werner, Stephan and Brandenburg, Karlheinz},
  title     = {Phase-Coherent Equalization of Loudspeakers},
  booktitle = {Proceedings of the 142nd AES Convention},
  year      = {2017}
}

@inproceedings{zacharov1998phase,
  author    = {Zacharov, Nick and Bech, S{\o}ren and Meares, David},
  title     = {The Use of Trained Listeners in Multichannel Sound Evaluation: Effect of Phase on Loudspeaker Sound Quality},
  booktitle = {Audio Engineering Society Convention 105},
  year      = {1998}
}
```

## ERB / cochlear bandwidth

`loss/enhanced_weights.rs::erb_for_freq` uses Glasberg & Moore's ERB formula `24.7·(1 + 4.37·f/1000)`.

```bibtex
@article{glasberg1990derivation,
  author  = {Glasberg, Brian R. and Moore, Brian C. J.},
  title   = {Derivation of auditory filter shapes from notched-noise data},
  journal = {Hearing Research},
  volume  = {47},
  number  = {1-2},
  pages   = {103--138},
  year    = {1990},
  doi     = {10.1016/0378-5955(90)90170-T}
}
```

## Perceptual EQ weighting

`loss/enhanced_weights.rs` — frequency-dependent weighting blending ERB + band penalties.

```bibtex
@inproceedings{kulkarni2008perceptual,
  author    = {Kulkarni, Abhijit and Hartmann, William M.},
  title     = {Perceptually-Motivated Audio Equalization},
  booktitle = {Audio Engineering Society Convention 124},
  year      = {2008}
}

@book{zolzer2011dafx,
  editor    = {Z{\"o}lzer, Udo},
  title     = {{DAFX}: Digital Audio Effects},
  edition   = {2},
  publisher = {Wiley},
  year      = {2011},
  isbn      = {978-0470665992},
  note      = {Chapters on equalization and frequency-domain weighting.}
}
```

## Zwicker / Fastl psychoacoustic metrics (Bark, loudness, sharpness, roughness)

`loss/epa/bark.rs::hz_to_bark`, `loudness.rs::specific_loudness`, `sharpness.rs::sharpness`, and `roughness.rs` use the Zwicker / Fastl model.

```bibtex
@book{zwicker2007psychoacoustics,
  author    = {Zwicker, Eberhard and Fastl, Hugo},
  title     = {Psychoacoustics: Facts and Models},
  edition   = {3},
  publisher = {Springer},
  year      = {2007},
  isbn      = {978-3540231592},
  doi       = {10.1007/978-3-540-68888-4}
}

@article{zwicker1980loudness,
  author  = {Zwicker, Eberhard},
  title   = {{Ein Verfahren zur Berechnung der Lautst{\"a}rke}},
  journal = {Acustica},
  volume  = {10},
  pages   = {304--308},
  year    = {1960},
  note    = {Foundational power-law specific loudness formulation.}
}
```

## ISO 226 equal-loudness contours

`loss/epa/loudness.rs::THRESHOLD_IN_QUIET` and `loss/epa/score.rs` (phon → SPL conversion) approximate ISO 226.

```bibtex
@techreport{iso226,
  author      = {{International Organization for Standardization}},
  title       = {{ISO 226:2003 — Acoustics — Normal equal-loudness-level contours}},
  institution = {ISO},
  year        = {2003}
}
```

## DIN 45692 (sharpness)

`loss/epa/sharpness.rs::SHARPNESS_WEIGHT` and the `S = 0.11·∑ N'(z)·g(z)·z / N_total` formulation come from DIN 45692.

```bibtex
@techreport{din45692,
  author      = {{Deutsches Institut f{\"u}r Normung}},
  title       = {{DIN 45692:2009-08 — Measurement technique for the simulation of the auditory sensation of sharpness}},
  institution = {DIN},
  year        = {2009}
}

@article{aures1985sensory,
  author  = {Aures, Wolfgang},
  title   = {{Berechnungsverfahren f{\"u}r den sensorischen Wohlklang beliebiger Schallsignale}},
  journal = {Acustica},
  volume  = {59},
  pages   = {130--141},
  year    = {1985},
  note    = {Sensory pleasantness model that includes sharpness, roughness, tonality and loudness.}
}
```

## Osgood semantic differential (Evaluation/Potency/Activity)

`loss/epa/score.rs::compute_epa` maps Zwicker metrics onto the three EPA dimensions of Osgood's semantic differential.

```bibtex
@book{osgood1957measurement,
  author    = {Osgood, Charles E. and Suci, George J. and Tannenbaum, Percy H.},
  title     = {The Measurement of Meaning},
  publisher = {University of Illinois Press},
  year      = {1957},
  isbn      = {978-0252745393}
}

@article{vonbismarck1974sharpness,
  author  = {von Bismarck, Gottfried},
  title   = {Sharpness as an attribute of the timbre of steady sounds},
  journal = {Acustica},
  volume  = {30},
  pages   = {159--172},
  year    = {1974},
  note    = {Early mapping of acoustic descriptors onto Osgood's EPA dimensions.}
}
```

## Cubic distortion tones (CDT)

`loss/epa/cdt.rs::cdt_level` uses the `L_cdt ≈ 2·L1 - L2 - 63 dB` approximation.

```bibtex
@article{goldstein1967auditory,
  author  = {Goldstein, Julius L.},
  title   = {Auditory nonlinearity},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {41},
  number  = {3},
  pages   = {676--689},
  year    = {1967},
  doi     = {10.1121/1.1910396}
}

@article{pressnitzer2000cdt,
  author  = {Pressnitzer, Daniel and Patterson, Roy D.},
  title   = {Distortion products and the perceived pitch of harmonic complex tones},
  journal = {Physiological and Psychophysical Bases of Auditory Function},
  pages   = {97--104},
  year    = {2001},
  note    = {Levels of the cubic difference tone $2f_1 - f_2$ generated by the cochlea.}
}
```

## Brännmark & Sternad — robust room correction

`roomeq/spatial_robustness.rs` and `roomeq/mixed_phase.rs` cite the AES paper.

```bibtex
@inproceedings{brannmark2008robust,
  author    = {Br{\"a}nnmark, Lars-Johan and Ahlen, Anders},
  title     = {Spatially Robust Audio Compensation Based on {SIMO} Feedforward Equalization},
  booktitle = {Audio Engineering Society Convention 124},
  year      = {2008},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=14529}
}
```

## Laborie, Bruno & Montoya — decomposed room correction

`roomeq/impulse_analysis.rs` — splits room response into modal / early-reflection / steady-state regions for differentiated correction.

```bibtex
@inproceedings{laborie2003decomposed,
  author    = {Laborie, Arnaud and Bruno, R{\'e}mi and Montoya, S{\'e}bastien},
  title     = {A New Comprehensive Approach of Surround Sound Recording},
  booktitle = {Audio Engineering Society Convention 114},
  year      = {2003},
  url       = {https://www.aes.org/e-lib/browse.cfm?elib=12565}
}
```

## Johnston — first-reflection cancellation

`roomeq/reflection_cancel.rs::cancel_first_reflection` applies `y[n] = x[n] - g·LP(x[n - d])` below ~500 Hz.

```bibtex
@inproceedings{johnston2008reflection,
  author    = {Johnston, James D.},
  title     = {Loudspeaker / Room Equalization in the Time and Frequency Domains},
  booktitle = {Audio Engineering Society Convention 125},
  year      = {2008},
  note      = {The "Johnston (AES)" reference cited in reflection\_cancel.rs}
}
```

## Multi-Sub Optimizer (MSO)

`roomeq/multiseat.rs`, `roomeq/multisub.rs`, `roomeq/group_processing.rs`, and `roomeq/workflows/bass_management.rs` implement MSO-style multi-subwoofer gain, delay, polarity, all-pass, and per-sub PEQ optimization across listening positions.

```bibtex
@manual{carter2026mso,
  author       = {Carter, Andy},
  title        = {{Multi-Sub Optimizer Help}},
  organization = {DIY Audio Engineering},
  year         = {2026},
  url          = {https://www.andyc.diy-audio-engineering.org/mso/html/},
  note         = {Documentation for Multi-Sub Optimizer, including multi-seat subwoofer optimization, SPL/headroom trade-offs, and DSP export.}
}
```

## Acoustic multi-channel equalization robustness

`roomeq/multiseat.rs::MultiSeatStrategy::ModalBasis` and the complex modal-basis SFM path use multi-channel room equalization concepts where robustness depends on the conditioning of the room-response inverse problem.

```bibtex
@article{kodrasi2018multichannel,
  author  = {Kodrasi, Ina and Doclo, Simon},
  title   = {Improving the conditioning of the optimization criterion in acoustic multi-channel equalization using shorter reshaping filters},
  journal = {{EURASIP} Journal on Advances in Signal Processing},
  volume  = {2018},
  number  = {11},
  year    = {2018},
  doi     = {10.1186/s13634-018-0532-1},
  url     = {https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-018-0532-1}
}
```

## Schroeder frequency / mode-density cutoff

`roomeq/eq.rs::estimate_schroeder_frequency` uses `f_S ≈ 2000·√(RT60 / V)`.

```bibtex
@article{schroeder1996sound,
  author  = {Schroeder, Manfred R.},
  title   = {The "Schroeder Frequency" Revisited},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {99},
  number  = {5},
  pages   = {3240--3241},
  year    = {1996},
  doi     = {10.1121/1.414868}
}

@article{schroeder1962frequency,
  author  = {Schroeder, Manfred R.},
  title   = {Frequency-correlation functions of frequency responses in rooms},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {34},
  number  = {12},
  pages   = {1819--1823},
  year    = {1962},
  doi     = {10.1121/1.1909136}
}
```

## Sobol quasi-random sequences

`optim/de.rs::init_sobol` initializes the DE population with a Sobol sequence for better space-filling than uniform random.

```bibtex
@article{sobol1967distribution,
  author  = {Sobol', Ilya M.},
  title   = {On the distribution of points in a cube and the approximate evaluation of integrals},
  journal = {USSR Computational Mathematics and Mathematical Physics},
  volume  = {7},
  number  = {4},
  pages   = {86--112},
  year    = {1967},
  doi     = {10.1016/0041-5553(67)90144-9}
}

@article{joe2008sobol,
  author  = {Joe, Stephen and Kuo, Frances Y.},
  title   = {Constructing Sobol sequences with better two-dimensional projections},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {30},
  number  = {5},
  pages   = {2635--2654},
  year    = {2008},
  doi     = {10.1137/070709359}
}
```

## Multi-objective genetic algorithms for EQ

`optim/pareto.rs` enumerates a Pareto front over filter count vs. flatness/score.

```bibtex
@article{deb2002nsga2,
  author  = {Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and Meyarivan, T.},
  title   = {A fast and elitist multiobjective genetic algorithm: {NSGA-II}},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {6},
  number  = {2},
  pages   = {182--197},
  year    = {2002},
  doi     = {10.1109/4235.996017}
}

@inproceedings{ramos2006multiobjective,
  author    = {Ramos, Germ{\'a}n and L{\'o}pez, Jos{\'e} J.},
  title     = {Multiobjective Genetic Algorithm Optimization of Linkwitz-Riley Crossovers Using Group Delay and Magnitude Response Criteria},
  booktitle = {Audio Engineering Society Convention 121},
  year      = {2006}
}
```

## Brooks-Park et al. — room compensation with supporting source

Candidate reference for `roomeq/` spatial room compensation. The paper proposes modifying the direct-to-reverberant ratio in a frequency-selective manner using a delayed secondary supporting source, with perceptual evaluation against a commercial room-compensation algorithm.

```bibtex
@article{Brooks_Park_2026,
  title={Room compensation for loudspeaker reproduction using a supporting source},
  volume={159},
  ISSN={1520-8524},
  url={http://dx.doi.org/10.1121/10.0043238},
  DOI={10.1121/10.0043238},
  number={4},
  journal={The Journal of the Acoustical Society of America},
  publisher={Acoustical Society of America (ASA)},
  author={Brooks-Park, James and Bech, S{\o}ren and {\O}stergaard, Jan and van de Par, Steven},
  year={2026},
  month=Apr,
  pages={3006--3017}
}
```

## Marcos-Macias et al. — DDSP adaptive room equalization

Replaces classical Filtered-X LMS with a modular Differentiable Digital Signal Processing (DDSP) framework for closed-loop adaptive room EQ. The important part for `autoeq` is that automatic differentiation and frequency-domain optimization let the system optimize parametric IIR/biquad EQ nodes directly while real music is playing, rather than relying on fixed linear correction filters.

```bibtex
@article{marcosmacias2026ddsp,
  title   = {A {DDSP} Framework for Adaptive Room Equalization},
  author  = {Marcos-Macias, Fernando and Daza-Llin, Mar{\'i}a P. and Camara, Miguel and Blanco, Jos{\'e} L.},
  journal = {arXiv preprint arXiv:2606.22563},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.22563}
}
```

## Brooks-Park et al. — supporting source room compensation (arXiv)

Extends room correction from spectral fixes into the spatial domain by adding a delayed secondary supporting source. The important part is that the direct-to-reverberant ratio can be reshaped frequency-selectively, addressing both frequency-response inaccuracies and distance-perception shifts caused by reflective boundaries.

```bibtex
@article{brookspark2026supporting,
  title   = {Room Compensation for Loudspeaker Reproduction Using a Supporting Source},
  author  = {Brooks-Park, James and van de Par, Steven and M{\o}ller, Martin B.},
  journal = {arXiv preprint arXiv:2604.12439},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.12439}
}
```

## Erdem et al. — flow matching sound field estimation

Applies generative Flow Matching models to reconstruct absolute Acoustic Transfer Function magnitudes from a handful of microphone positions. The important part for room EQ is that sparse measurements can be turned into a continuous spatial sound-field estimate, which is useful for multi-point correction and automated subwoofer placement.

```bibtex
@article{erdem2026flowmatching,
  title   = {Sound Field Magnitude Estimation via Flow Matching Guided by Sparse Measurements},
  author  = {Erdem, Emir and others},
  journal = {arXiv preprint arXiv:2605.10398},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.10398}
}
```

## Fuglsig et al. — online sound speed estimation

Builds a real-time single-channel estimator that tracks sound-speed variations while normal music is playing. The important part for multi-speaker correction is that temperature-induced sound-speed changes shift the phase alignment of static Room Impulse Responses; tracking them online keeps the correction filters stable without re-measuring the room.

```bibtex
@article{fuglsig2026soundspeed,
  title   = {Online Single-Channel Audio-Based Sound Speed Estimation for Robust Multi-Channel Audio Control},
  author  = {Fuglsig, Andreas J. and others},
  journal = {arXiv preprint arXiv:2602.16416},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.16416}
}
```

## Koyama et al. — past, present and future of spatial audio

Foundational survey outlining the shift from static one-dimensional room correction filters to high-order multi-speaker wavefield synthesis and object-based spatial rendering. The important part is the mathematical framing of modern sound-field reproduction and the persistent problem of user-to-loudspeaker distance variation.

```bibtex
@article{koyama2025spatialaudio,
  title   = {Past, Present, and Future of Spatial Audio and Room Acoustics},
  author  = {Koyama, Shoichi and De Sena, Enzo and Samarasinghe, Prasanga and Thomas, M. R. P. and Antonacci, Fabio},
  journal = {arXiv preprint arXiv:2503.12948},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.12948}
}
```

## Sprunck et al. — reversing the shoebox image source method

Presents an algorithm that back-calculates exact room boundaries, dimensions, and orientations directly from measured multi-channel impulse responses. The important part is that knowing the physical room geometry lets a correction system extrapolate its filter profile across unmeasured regions instead of being trapped in a single sweet spot.

```bibtex
@article{sprunck2025shoebox,
  title   = {Fully Reversing the Shoebox Image Source Method: From Impulse Responses to Room Parameters},
  author  = {Sprunck, Tom and Deleforge, Antoine and Privat, Yannick and Foy, C{\'e}dric},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2025},
  url     = {https://arxiv.org/abs/2405.03385}
}
```

## Seidel et al. — neural Kalman filters for echo cancellation

Targets acoustic echo cancellation but is directly relevant to room EQ: it combines deep neural networks with Frequency-Domain Kalman Filters for fast, real-time acoustic state estimation. The important part is the ability to adapt to nonlinear loudspeaker behavior and sudden room changes such as people moving.

```bibtex
@article{seidel2025neuralkalman,
  title   = {Neural {Kalman} Filters for Acoustic Echo Cancellation},
  author  = {Seidel, Eric and Enzner, Gerald and Mowlaee, Pejman and Fingscheidt, Tim},
  journal = {IEEE Signal Processing Magazine},
  year    = {2025},
  url     = {https://arxiv.org/abs/2501.16367}
}
```

## Brooks-Park et al. — RIR prototyping by receiver distance

Avoids the single-point-measurement problem by computing an optimized weighted prototype response from multiple distributed microphones. The important part is that spatial weights are set dynamically from the estimated physical distance of each microphone to the main listening position, so the resulting EQ does not ruin the response elsewhere in the room.

```bibtex
@inproceedings{brookspark2024prototype,
  title     = {Room Impulse Response Prototyping Using Receiver Distance Estimations for High Quality Room Equalisation Algorithms},
  author    = {Brooks-Park, James and Van De Par, Steven and {\O}stergaard, Jan and Bech, S{\o}ren and M{\o}ller, Martin B.},
  booktitle = {International Workshop on Acoustic Signal Enhancement (IWAENC)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2409.10131}
}
```

## Kitić & Daniel — blind Ambisonic RIR identification

Performs blind room correction in the Ambisonic (spherical harmonic) domain, separating early reflections from late reverberation using only unknown far-field source signals. The important part is that calibration can happen passively during normal playback without explicit sweep or pink-noise signals.

```bibtex
@article{kitic2023blind,
  title   = {Blind Identification of Ambisonic Reduced Room Impulse Response},
  author  = {Kiti{\'c}, Sr{\dj}an and Daniel, J{\'e}r{\^o}me},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2023},
  url     = {https://arxiv.org/abs/2305.03558}
}
```

## Ick et al. — blind room parameter estimation via phase features

Uses Convolutional Neural Networks to extract macro room metrics such as broadband RT60 and overall room volume from single-channel noisy audio. The important part is that these parameters can drive automated room EQ decisions, for example how aggressively to apply low-frequency decay compensation.

```bibtex
@inproceedings{ick2023blind,
  title     = {Blind Acoustic Room Parameter Estimation Using Phase Features},
  author    = {Ick, Constantin and Mehrabi, Armin and Jin, Wissam},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.07449}
}
```

## See also

Optimization-algorithm citations (DE, JADE/L-SHADE, COBYLA, ISRES, Levenberg-Marquardt) live in:

- [`crates/math-audio/math-optimisation/REFERENCES.md`](../math-audio/math-optimisation/REFERENCES.md)

Filter-design citations (RBJ cookbook, Butterworth, Linkwitz-Riley, Orfanidis, Vicanek, Zavalishin TPT/SVF, Kirkeby, warped/Kautz, filtfilt) live in:

- [`crates/math-audio/math-iir-fir/REFERENCES.md`](../math-audio/math-iir-fir/REFERENCES.md)
