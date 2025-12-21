<!-- markdownlint-disable-file MD013 -->

# AutoEQ: a cli for computing EQ for headphones and speakers.

The software helps you to get better sound from your speakers or your headsets. If you prefer a classical UI, please use [SotF](https://github.com/pierreaubert/sotf).

## Install

### Cargo

Install [rustup](https://rustup.rs/) first.

If you already have cargo / rustup, you can jump to:

```shell
cargo install just
export AUTOEQ_DIR=`pwd`
just
```

Select the correct install just command for your platform:
```shell
just install-...
```

You can build or test with a simple:
```shell
just build
just test
just qa
```

and you are set up. See this [README](autoeq/README.md) for instructions on how to use it.

## Toolkit

### autoeq-cea2034

A implementation of CEA2034 aka [Spinorama](https://spinorama.org): a set of metrics and curves that describe a loudspeaker performance.

Status: mature.

### autoeq

A [CLI](autoeq/README.md) to optimise the response of your headset or headphone.

Status: good up to very good depending what you optimise for.

### autoeq-roomsim

A room simulator to help you to understand the response of your speakers on your room. [Available online](https://roomsim.spinorama.org) try it out!

Status: getting good.

### autoeq-env

A small set of functions and constants used by the other crates but you are unlikely to be interested.

