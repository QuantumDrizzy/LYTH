//! Content-addressing for future `lyth probe anchor` — hash only, no chain.

use sha2::{Digest, Sha256};

/// SHA-256 of canonical JSON (sorted object keys via serde_json Value round-trip).
pub fn content_hash(value: &serde_json::Value) -> String {
    let canonical = canonicalize(value);
    let bytes = serde_json::to_vec(&canonical).expect("json serialize");
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

/// Local hex without pulling the `hex` crate — keep deps thin.
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        const T: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.as_ref().len() * 2);
        for &b in bytes.as_ref() {
            out.push(T[(b >> 4) as usize] as char);
            out.push(T[(b & 0xf) as usize] as char);
        }
        out
    }
}

fn canonicalize(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonicalize(&map[k]));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn key_order_does_not_change_hash() {
        let a = json!({"b": 1, "a": 2});
        let b = json!({"a": 2, "b": 1});
        assert_eq!(content_hash(&a), content_hash(&b));
    }
}
