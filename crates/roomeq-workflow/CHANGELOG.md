# Changelog

## Unreleased

- Validate file-backed acoustic resources before declaring a production
  configuration ready and preserve sidecar names across repeated exports.
- Inherited the workspace policy forbidding unsafe Rust code.
- Documented crate ownership and verification expectations.

## 0.4.51

- Established `roomeq-workflow` as the production composition boundary over RoomEQ resources, engine, and exporters.
