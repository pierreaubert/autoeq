# --------------------------------------------------------- -*- just -*-
alias qa-autoeq := qa-autoeq-all

# Bounded AutoEQ smoke (refactor-20261025 Phase 1): one flat and one
# score case over checked-in curve data, same --qa thresholds as the
# full recipes below, bounded --maxeval. Uses `--algo autoeq:cobyla`
# because DE cannot report convergence below its ~1.5M-evaluation
# generation floor, which would make a bounded DE smoke fail its own
# convergence gate. This is a smoke lane only; `qa-autoeq-all` keeps
# the full DE budgets and verdicts.
[group('qa-autoeq')]
qa-autoeq-quick:
	#!/usr/bin/env bash
	set -euo pipefail
	target_dir="$({{cargo}} metadata --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
	{{cargo}} build -p autoeq-cli --locked --release --bin autoeq
	curve='./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic DT1990 Pro Headphone Frequency Response Measurement.csv'
	"$target_dir/release/autoeq" -n 5 \
	--curve "$curve" \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-flat --max-db 6 --max-q 6 --maxeval 20000 --algo autoeq:cobyla --seed 42 --min-freq=20 --max-freq 10000 --peq-model pk \
	--qa 0.5
	"$target_dir/release/autoeq" -n 5 \
	--curve "$curve" \
	--target ./data_tests/targets/harman-over-ear-2018.csv --loss headphone-score \
	--maxeval 20000 --algo autoeq:cobyla --seed 42 --qa 3.0

[group('qa-autoeq')]
qa-autoeq-all: prod-autoeq \
	qa-ascilab-6b \
	qa-jbl-m2-flat \
	qa-jbl-m2-score \
	qa-beyerdynamic-dt1990pro \
	qa-edifierw830nb

[group('qa-autoeq')]
qa-ascilab-6b:
	./target/release/autoeq --speaker="AsciLab F6B" --version asr --measurement CEA2034 \
	--algo autoeq:de --loss speaker-score -n 7 --min-freq=30 --max-q=6 \
	--maxeval 100000 --seed 4 --qa 0.5

[group('qa-autoeq')]
qa-jbl-m2-flat:
	./target/release/autoeq --speaker="JBL M2" --version eac --measurement CEA2034 \
	--algo autoeq:de --loss speaker-flat -n 7 --min-freq=20 --max-q=6 --peq-model hp-pk \
	--maxeval 100000 --qa 0.5

[group('qa-autoeq')]
qa-jbl-m2-score:
	./target/release/autoeq --speaker="JBL M2" --version eac --measurement CEA2034 \
	--algo autoeq:de --loss speaker-score -n 7 --min-freq=20 --max-q=6 --peq-model hp-pk \
	--maxeval 100000 --qa 0.5

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro: qa-beyerdynamic-dt1990pro-flat qa-beyerdynamic-dt1990pro-score	qa-beyerdynamic-dt1990pro-score2

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-score:
	./target/release/autoeq -n 5 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv --loss headphone-score  \
	--maxeval 100000 --qa 3.0

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-score2:
	./target/release/autoeq -n 7 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-score	--max-db 6 --max-q 6 --algo autoeq:de --seed 42 --maxeval 100000 --min-freq=20 --max-freq 10000 --peq-model hp-pk-lp --min-q 0.6 --min-db 0.25 \
	--qa 1.5

# Strict QA requires solver convergence; MH backends report budget stops only.
[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-flat:
	./target/release/autoeq -n 5 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-flat  --max-db 6 --max-q 6 --maxeval 100000 --algo autoeq:de --seed 42 --min-freq=20 --max-freq 10000 --peq-model pk \
	--qa 0.5

[group('qa-autoeq')]
qa-edifierw830nb: qa-edifierw830nb-autoeqde qa-edifierw830nb-autoeqde-pop100 qa-edifierw830nb-autoeqde-pop80

[group('qa-autoeq')]
qa-edifierw830nb-autoeqde:
	./target/release/autoeq -n 9 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.08 \
	--algo autoeq:de --population 70 --maxeval 80000 --seed 42 \
	--qa 6.0

[group('qa-autoeq')]
qa-edifierw830nb-autoeqde-pop100:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.0000001 --algo autoeq:de --seed 42 --population 100 --maxeval 80000 \
	--qa 2.5
[group('qa-autoeq')]
qa-edifierw830nb-autoeqde-pop80:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.000000001 --algo autoeq:de --seed 42 --population 80 --maxeval 80000 \
	--qa 2.5
