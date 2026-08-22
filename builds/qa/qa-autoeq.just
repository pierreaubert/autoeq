# --------------------------------------------------------- -*- just -*-
alias qa-autoeq := qa-autoeq-all

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
	--loss headphone-score	--max-db 6 --max-q 6 --algo mh:rga --maxeval 100000 --min-freq=20 --max-freq 10000 --peq-model hp-pk-lp --min-q 0.6 --min-db 0.25 \
	--qa 1.5

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-flat:
	./target/release/autoeq -n 5 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-flat  --max-db 6 --max-q 6 --maxeval 100000 --algo mh:pso --min-freq=20 --max-freq 10000 --peq-model pk \
	--qa 0.5

[group('qa-autoeq')]
qa-edifierw830nb: qa-edifierw830nb-autoeqde qa-edifierw830nb-mhrga qa-edifierw830nb-mhfirefly

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
qa-edifierw830nb-mhrga:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.0000001 --algo mh:rga --population 100 --maxeval 80000 \
	--qa 2.5
[group('qa-autoeq')]
qa-edifierw830nb-mhfirefly:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.000000001 --algo mh:rga --population 80 --maxeval 80000 \
	--qa 2.5
