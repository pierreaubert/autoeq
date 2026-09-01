IN=./data_tests/roomeq/measured
OUT=./data_generated/roomeq/measured
LOG=warn

cargo build --release --features cli --bin roomeq


for scenario in 2.0_8361a 2.0_d3v 2.0_fidelia 2.0_t7v 5_0.genelec 5.1_kef 5.1.4_genelec; do
  SIN=$IN/$scenario;
  SOUT=$OUT/$scenario;
  rm -fr ${SOUT} && mkdir -p ${SOUT};
  for mode in iir fir mixed mixed-phase; do
    RUST_LOG=$LOG ./target/release/roomeq \
       --config  ${SIN}/recordings.json \
       --override-config ${SIN}/optimiser-$mode.json \
       --output ${SOUT}/dsp-$mode.json;

    ./venv/bin/python3 ./scripts/display-roomeq.py \
        ${SOUT}/dsp-$mode.json \
        --output ${SOUT}/dsp-$mode.html;
  done
  ./venv/bin/python3 ./scripts/display-roomeq.py \
        --compare ${SOUT}/dsp-iir.json ${SOUT}/dsp-fir.json ${SOUT}/dsp-mixed.json ${SOUT}/dsp-mixed-phase.json \
	--output ${SOUT}/compare.html;
done
