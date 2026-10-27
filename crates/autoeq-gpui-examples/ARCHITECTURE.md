# Local GPUI demo ownership

`autoeq-gpui-examples` is excluded from the backend workspace and has a separate
workspace and lockfile. It owns the `d3rs-spinorama` and `px-spinorama` demo
binaries. Explicit `--manifest-path` commands or their Justfile recipes opt into
this dependency set.

The demos consume AutoEQ core and measurement owners directly. Measurements
explicitly enable `api`; the native GPUI plotting and UI dependencies belong
here rather than in the default backend library set. This separation keeps
local demo dependency resolution out of ordinary backend installs and QA lanes.

The D3RS demo retains its interpolation and rendering tests beside the demo
source. These excluded-workspace tests need their own explicit invocation;
`--workspace` at the AutoEQ root does not include them.

See the [build and test commands](README.md) and
[workspace architecture](../../docs/ARCHITECTURE.md).
