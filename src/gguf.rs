//! Read just enough of a GGUF header to budget any model, not only the ones in the catalog.
//!
//! GGUF starts with key/value metadata (architecture, layer count, attention heads, context length),
//! then the tensor index, then the weights. We stream the front of the file and stop as soon as the
//! tokenizer section begins, so for a remote file this costs a few MB at most, not the whole download.

use std::collections::HashMap;
use std::io::Read;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meta {
    pub arch: String,
    pub name: Option<String>,
    pub layers: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding: Option<u32>,
    pub heads: Option<u32>,
    pub kv_heads: Option<f64>,
    pub key_length: Option<u32>,
    pub experts: Option<u32>,
    pub experts_used: Option<u32>,
}

impl Meta {
    pub fn head_dim(&self) -> Option<u32> {
        self.key_length
            .or_else(|| Some(self.embedding? / self.heads?.max(1)))
    }

    /// fp16 KV cache bytes per token of context.
    pub fn kv_bytes_per_token(&self) -> Option<f64> {
        let kvh = self.kv_heads.or(self.heads.map(|h| h as f64))?;
        Some(2.0 * self.layers? as f64 * kvh * self.head_dim()? as f64 * 2.0)
    }

    /// Fraction of weights read per token. MoE models only touch the routed experts.
    pub fn active_fraction(&self) -> f64 {
        match (self.experts, self.experts_used) {
            (Some(n), Some(k)) if n > 1 && k < n => (k as f64 / n as f64 + 0.05).min(1.0),
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
enum Val {
    Int(i64),
    Float(f64),
    Str(String),
    IntArrayMean(f64),
    Skip,
}

struct R<T: Read> {
    inner: T,
    read: u64,
    cap: u64,
}

impl<T: Read> R<T> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        if self.read + n as u64 > self.cap {
            return Err("GGUF header larger than expected".into());
        }
        let mut b = vec![0u8; n];
        self.inner
            .read_exact(&mut b)
            .map_err(|e| format!("GGUF read: {e}"))?;
        self.read += n as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            return Err("GGUF string too long".into());
        }
        Ok(String::from_utf8_lossy(&self.bytes(n as usize)?).into_owned())
    }
    fn skip(&mut self, n: u64) -> Result<(), String> {
        if self.read + n > self.cap {
            return Err("GGUF header larger than expected".into());
        }
        let copied = std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())
            .map_err(|e| e.to_string())?;
        if copied != n {
            return Err("GGUF truncated".into());
        }
        self.read += n;
        Ok(())
    }

    fn scalar(&mut self, ty: u32) -> Result<Val, String> {
        Ok(match ty {
            0 => Val::Int(self.bytes(1)?[0] as i64),
            1 => Val::Int(self.bytes(1)?[0] as i8 as i64),
            2 => Val::Int(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as i64),
            3 => Val::Int(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as i64),
            4 => Val::Int(self.u32()? as i64),
            5 => Val::Int(self.u32()? as i32 as i64),
            6 => Val::Float(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()) as f64),
            7 => Val::Int(self.bytes(1)?[0] as i64),
            8 => Val::Str(self.string()?),
            10 => Val::Int(self.u64()? as i64),
            11 => Val::Int(self.u64()? as i64),
            12 => Val::Float(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
            t => return Err(format!("unknown GGUF type {t}")),
        })
    }

    fn value(&mut self, ty: u32) -> Result<Val, String> {
        if ty != 9 {
            return self.scalar(ty);
        }
        let ety = self.u32()?;
        let n = self.u64()?;
        let fixed = match ety {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        // Small int arrays (per-layer head counts) are worth reading; everything else is skipped.
        if matches!(ety, 0..=5 | 10 | 11) && n <= 4096 {
            let mut sum = 0f64;
            for _ in 0..n {
                if let Val::Int(v) = self.scalar(ety)? {
                    sum += v as f64;
                }
            }
            return Ok(Val::IntArrayMean(if n > 0 { sum / n as f64 } else { 0.0 }));
        }
        match fixed {
            Some(sz) => self.skip(n * sz)?,
            None => {
                for _ in 0..n {
                    self.value(ety)?;
                }
            }
        }
        Ok(Val::Skip)
    }
}

/// Parse metadata from the start of a GGUF stream. Reads at most `cap` bytes.
pub fn read_meta(reader: impl Read, cap: u64) -> Result<Meta, String> {
    let mut r = R {
        inner: reader,
        read: 0,
        cap,
    };
    if r.bytes(4)? != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(format!("GGUF v{version} is too old"));
    }
    let _tensors = r.u64()?;
    let kv_count = r.u64()?;
    let mut kv: HashMap<String, Val> = HashMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        // The tokenizer vocab is the bulk of the header and we need nothing from it or after it.
        if key.starts_with("tokenizer.") && kv.keys().any(|k| k.ends_with(".block_count")) {
            break;
        }
        let ty = r.u32()?;
        let v = r.value(ty)?;
        if !matches!(v, Val::Skip) {
            kv.insert(key, v);
        }
    }
    let arch = match kv.get("general.architecture") {
        Some(Val::Str(s)) => s.clone(),
        _ => return Err("GGUF has no general.architecture".into()),
    };
    let int = |k: &str| match kv.get(&format!("{arch}.{k}")) {
        Some(Val::Int(v)) => Some(*v as u32),
        Some(Val::IntArrayMean(m)) => Some(m.round() as u32),
        Some(Val::Float(f)) => Some(*f as u32),
        _ => None,
    };
    let kv_heads = match kv.get(&format!("{arch}.attention.head_count_kv")) {
        Some(Val::Int(v)) => Some(*v as f64),
        Some(Val::IntArrayMean(m)) => Some(*m),
        _ => None,
    };
    Ok(Meta {
        name: match kv.get("general.name") {
            Some(Val::Str(s)) => Some(s.clone()),
            _ => None,
        },
        layers: int("block_count"),
        context_length: int("context_length"),
        embedding: int("embedding_length"),
        heads: int("attention.head_count"),
        kv_heads,
        key_length: int("attention.key_length"),
        experts: int("expert_count"),
        experts_used: int("expert_used_count"),
        arch,
    })
}

pub fn read_meta_file(path: &std::path::Path) -> Result<Meta, String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    read_meta(std::io::BufReader::new(f), 256 << 20)
}

/// Stream the header of a remote GGUF. Only the bytes we need cross the wire.
pub fn read_meta_url(url: &str) -> Result<Meta, String> {
    let resp = crate::net::hf_get(&crate::net::agent_for(url), url)
        .set("Range", "bytes=0-67108863")
        .call()
        .map_err(crate::net::err_text)?;
    read_meta(std::io::BufReader::new(resp.into_reader()), 64 << 20)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn s(b: &mut Vec<u8>, x: &str) {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    }

    /// A tiny but valid GGUF header shaped like Llama 3.1 8B.
    pub fn fake_llama() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(9u64.to_le_bytes());
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "llama");
        s(&mut b, "general.name");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "Fake 8B");
        for (k, v) in [
            ("llama.block_count", 32u32),
            ("llama.context_length", 131072),
            ("llama.embedding_length", 4096),
            ("llama.attention.head_count", 32),
            ("llama.attention.head_count_kv", 8),
        ] {
            s(&mut b, k);
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        // A float array we must skip.
        s(&mut b, "llama.rope.freqs");
        b.extend(9u32.to_le_bytes());
        b.extend(6u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend([0u8; 12]);
        // A string array (vocab) that should stop the parse.
        s(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        s(&mut b, "a");
        s(&mut b, "b");
        b
    }

    #[test]
    fn parses_llama_shape() {
        let m = read_meta(&fake_llama()[..], 1 << 20).unwrap();
        assert_eq!(m.arch, "llama");
        assert_eq!(m.name.as_deref(), Some("Fake 8B"));
        assert_eq!(m.layers, Some(32));
        assert_eq!(m.head_dim(), Some(128));
        // Same 128 KiB per token as the catalog maths for Llama 3.1 8B.
        assert_eq!(m.kv_bytes_per_token(), Some(131072.0));
        assert_eq!(m.active_fraction(), 1.0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(read_meta(&b"PK\x03\x04garbage"[..], 1 << 20).is_err());
    }

    #[test]
    fn stops_at_cap() {
        assert!(read_meta(&fake_llama()[..], 40).is_err());
    }
}
