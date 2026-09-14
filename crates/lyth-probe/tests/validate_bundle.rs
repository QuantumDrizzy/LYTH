use lyth_probe::{
    validate, Baseline, Bundle, CacheState, ClockState, KnownLimit, LimitStatus, SCHEMA_ID,
};
use serde_json::json;

#[test]
fn template_fails_validate_until_filled() {
    let b = Bundle::template("REPLACE_ME");
    assert!(validate(&b).is_err());
}

#[test]
fn complete_unverified_bundle_passes() {
    let b = Bundle {
        schema: SCHEMA_ID.into(),
        claim: "k_integrate median step ≤ 12 µs on sm_120 at N=166700".into(),
        value: json!(10.76),
        unit: "us/step".into(),
        baseline: Baseline {
            name: "plain launch (no CUDA graph)".into(),
            value: json!(52.0),
            unit: "us/step".into(),
        },
        n_reps: 12,
        arch: "sm_120".into(),
        compile_flags: vec!["-O3".into(), "-arch=sm_120".into()],
        clock_state: ClockState::Unknown,
        cache_state: CacheState::Unknown,
        known_limits: vec![
            KnownLimit {
                id: "clock-unmeasured".into(),
                text: "boost not locked".into(),
                status: LimitStatus::Open,
            },
            KnownLimit {
                id: "cache-unmeasured".into(),
                text: "L2 state not flushed".into(),
                status: LimitStatus::Open,
            },
        ],
        verified: false,
        device: Some("RTX 5060 Ti".into()),
        source_repo: Some("rse-hpc-lab/08".into()),
        notes: vec![],
    };
    assert!(validate(&b).is_ok());
}

#[test]
fn verified_with_open_limit_fails() {
    let mut b = Bundle {
        schema: SCHEMA_ID.into(),
        claim: "example".into(),
        value: json!(1.0),
        unit: "x".into(),
        baseline: Baseline {
            name: "ref".into(),
            value: json!(2.0),
            unit: "x".into(),
        },
        n_reps: 3,
        arch: "sm_120".into(),
        compile_flags: vec![],
        clock_state: ClockState::Measured {
            detail: "locked application clocks via nvidia-smi".into(),
        },
        cache_state: CacheState::Flushed {
            detail: "l2 flush before each rep".into(),
        },
        known_limits: vec![KnownLimit {
            id: "no-nsight-bytes".into(),
            text: "dram__bytes.sum not collected".into(),
            status: LimitStatus::Open,
        }],
        verified: true,
        device: None,
        source_repo: None,
        notes: vec![],
    };
    let err = validate(&b).unwrap_err();
    assert!(err.iter().any(|v| v.0.contains("verified=true")));

    b.known_limits[0].status = LimitStatus::Closed;
    assert!(validate(&b).is_ok());
}

#[test]
fn unknown_clock_without_limit_fails_with_next_step() {
    let b = Bundle {
        schema: SCHEMA_ID.into(),
        claim: "example".into(),
        value: json!(1.0),
        unit: "x".into(),
        baseline: Baseline {
            name: "ref".into(),
            value: json!(2.0),
            unit: "x".into(),
        },
        n_reps: 3,
        arch: "sm_120".into(),
        compile_flags: vec![],
        clock_state: ClockState::Unknown,
        cache_state: CacheState::Flushed {
            detail: "ok".into(),
        },
        known_limits: vec![],
        verified: false,
        device: None,
        source_repo: None,
        notes: vec![],
    };
    let err = validate(&b).unwrap_err();
    let msg = err
        .iter()
        .map(|v| v.0.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(msg.contains("options:"));
    assert!(msg.contains("clock"));
}

#[test]
fn no_baseline_fails() {
    let b = Bundle {
        schema: SCHEMA_ID.into(),
        claim: "example".into(),
        value: json!(1.0),
        unit: "x".into(),
        baseline: Baseline {
            name: "REPLACE_ME".into(),
            value: json!(null),
            unit: "REPLACE_ME".into(),
        },
        n_reps: 1,
        arch: "sm_120".into(),
        compile_flags: vec![],
        clock_state: ClockState::Measured {
            detail: "locked".into(),
        },
        cache_state: CacheState::Warm {
            detail: "steady".into(),
        },
        known_limits: vec![],
        verified: false,
        device: None,
        source_repo: None,
        notes: vec![],
    };
    let err = validate(&b).unwrap_err();
    assert!(err.iter().any(|v| v.0.contains("baseline")));
}
