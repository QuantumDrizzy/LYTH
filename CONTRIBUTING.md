# Contributing to LYTH

Bug reports, machine files and kernels are all welcome. The rules are the project's own, applied to
everyone including its author.

## Before you send code

```bash
cargo test --workspace            # tests that need a device or the Unibit emulator print where they looked and skip
cargo run -p lyth -- check examples/saxpy.lyth --machine fixtures/machine/sm_120.json
```

## The rules

- **A claim needs a measurement.** A new performance or traffic claim comes with the measurement that
  made it (`ncu` output, the command that produced it) and the baseline it is against.
- **Machine files are measured, not copied.** `tools/peak_probe.py --machine-out`. A datasheet peak in a
  machine file makes every ratio printed against it wrong.
- **Predictions before measurements.** A change that is meant to move a number states, in its ADR under
  `docs/`, what it expects *before* the run. A prediction that fails stays in the record; it is not
  edited into a pass.
- **Known limits are written down.** `[KNOWN_LIMIT]` in code, docs and tests, so a limitation is never
  quietly widened.
- **Refusals over guesses.** When the compiler cannot know something, it refuses and says why. A missing
  number is never turned into a zero.

## Licence

Contributions are dual licensed under MIT OR Apache-2.0, as the project is; see `README.md`.
