IN=./data_tests/roomeq/measured/5.1.4_genelec
OUT=./data_generated/roomeq/measured/5.1.4_genelec

cargo build --release --features cli --bin roomeq

rm -fr ${OUT} && mkdir -p ${OUT}

./target/release/roomeq \
    --config  ${IN}/recordings-5.1.4.json \
    --override-config ${IN}/optimiser-iir.json \
    --output ${OUT}/dsp-iir.json

./venv/bin/python3 ./scripts/display-roomeq.py \
    ${OUT}/dsp-iir.json \
    --output ${OUT}/dsp-iir.html

./target/release/roomeq \
    --config  ${IN}/recordings-5.1.4.json \
    --override-config ${IN}/optimiser-fir.json \
    --output ${OUT}/dsp-fir.json

./venv/bin/python3 ./scripts/display-roomeq.py \
    ${OUT}/dsp-fir.json \
    --output ${OUT}/dsp-fir.html

./target/release/roomeq \
    --config  ${IN}/recordings-5.1.4.json \
    --override-config ${IN}/optimiser-mixed.json \
    --output ${OUT}/dsp-mixed.json

./venv/bin/python3 ./scripts/display-roomeq.py \
    ${OUT}/dsp-mixed.json \
    --output ${OUT}/dsp-mixed.html

./venv/bin/python3 ./scripts/display-roomeq.py --compare \
    ${OUT}/dsp-iir.json \
    ${OUT}/dsp-fir.json \
    ${OUT}/dsp-mixed.json \
    --output ${OUT}/comparison.html
