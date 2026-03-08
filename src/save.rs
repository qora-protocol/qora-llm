//! Save/load DecodeWeights to a compact binary format.
//!
//! File format (.qora):
//!   Header: magic "QOR3" + version(u32) + format(u8: 0=F16, 1=Q4)
//!   Metadata: num_layers, vocab, hidden, rope_half_dim, no_rope_layers
//!   Per-layer: 7 weight matrices + 2 norm vectors + attention config
//!   Global: embedding weight + final norm + RoPE tables
//!
//! All multi-byte values are little-endian. Vectors are length-prefixed.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;

use crate::gemv::DecodeWeights;

const MAGIC: &[u8; 4] = b"QOR3";
const VERSION: u32 = 1;

// ============================================================
// Low-level I/O helpers
// ============================================================

fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_u8(w: &mut impl Write, val: u8) -> io::Result<()> {
    w.write_all(&[val])
}

fn write_f16_vec(w: &mut impl Write, data: &[f16]) -> io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_bool_vec(w: &mut impl Write, data: &[bool]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
    w.write_all(&bytes)
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_bytes(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_f16_vec(r: &mut impl Read) -> io::Result<Vec<f16>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        let ptr = bytes.as_ptr() as *const f16;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    Ok(data)
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        let ptr = bytes.as_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    Ok(data)
}

fn read_bool_vec(r: &mut impl Read) -> io::Result<Vec<bool>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|&b| b != 0).collect())
}

// ============================================================
// Public save/load API
// ============================================================

/// Save DecodeWeights to a .qora binary file.
pub fn save_model(weights: &DecodeWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);

    // Header
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u8(&mut w, weights.format_id())?;

    // Metadata
    write_u32(&mut w, weights.num_layers() as u32)?;
    write_u32(&mut w, weights.vocab() as u32)?;
    write_u32(&mut w, weights.hidden() as u32)?;
    write_u32(&mut w, weights.rope_half_dim() as u32)?;
    write_bool_vec(&mut w, weights.no_rope_layers_ref())?;

    // Per-layer weights
    for i in 0..weights.num_layers() {
        weights.write_layer(&mut w, i)?;
    }

    // Global: embedding
    weights.write_embed(&mut w)?;

    // Final norm gamma
    write_f16_vec(&mut w, weights.final_norm_ref())?;

    // RoPE tables
    write_f32_vec(&mut w, weights.rope_cos_ref())?;
    write_f32_vec(&mut w, weights.rope_sin_ref())?;

    w.flush()?;
    Ok(())
}

/// Load DecodeWeights from a .qora binary file.
pub fn load_model(path: &Path) -> io::Result<DecodeWeights> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic bytes"));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported version {version}, expected {VERSION}"),
        ));
    }
    let format_id = read_u8(&mut r)?;

    // Metadata
    let num_layers = read_u32(&mut r)? as usize;
    let vocab = read_u32(&mut r)? as usize;
    let hidden = read_u32(&mut r)? as usize;
    let rope_half_dim = read_u32(&mut r)? as usize;
    let no_rope_layers = read_bool_vec(&mut r)?;

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 {
            eprintln!("  Loading layer {i}/{num_layers}...");
        }
        layers.push(read_layer_weights(&mut r, format_id)?);
    }

    // Embedding
    let embed = read_weight_data(&mut r, format_id)?;

    // Final norm
    let final_norm_gamma = read_f16_vec(&mut r)?;

    // RoPE
    let rope_cos = read_f32_vec(&mut r)?;
    let rope_sin = read_f32_vec(&mut r)?;

    let format_name = match format_id {
        0 => "F16",
        1 => "Q4",
        _ => "unknown",
    };
    eprintln!("  Loaded {format_name} model: {num_layers} layers, vocab={vocab}, hidden={hidden}");

    Ok(DecodeWeights::from_parts(
        layers, embed, vocab, hidden,
        final_norm_gamma, rope_cos, rope_sin, rope_half_dim,
        no_rope_layers, format_id,
    ))
}

// ============================================================
// Per-weight serialization helpers
// ============================================================

/// Read a single weight (F16 or Q4) from the stream.
fn read_weight_data(r: &mut impl Read, format_id: u8) -> io::Result<crate::gemv::WeightData> {
    let k = read_u64(r)? as usize;
    let n = read_u64(r)? as usize;

    match format_id {
        0 => {
            // F16
            let data = read_f16_vec(r)?;
            Ok(crate::gemv::WeightData::F16 { data, k, n })
        }
        1 => {
            // Q4
            let packed = read_bytes(r)?;
            let scales = read_f16_vec(r)?;
            Ok(crate::gemv::WeightData::Q4 { packed, scales, k, n })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown format id {format_id}"),
        )),
    }
}

/// Read a full layer's weights.
fn read_layer_weights(r: &mut impl Read, format_id: u8) -> io::Result<crate::gemv::LayerWeightData> {
    let q_proj = read_weight_data(r, format_id)?;
    let k_proj = read_weight_data(r, format_id)?;
    let v_proj = read_weight_data(r, format_id)?;
    let o_proj = read_weight_data(r, format_id)?;
    let gate_proj = read_weight_data(r, format_id)?;
    let up_proj = read_weight_data(r, format_id)?;
    let down_proj = read_weight_data(r, format_id)?;
    let input_norm_gamma = read_f16_vec(r)?;
    let post_attn_norm_gamma = read_f16_vec(r)?;
    let num_heads = read_u32(r)? as usize;
    let num_kv_heads = read_u32(r)? as usize;
    let head_dim = read_u32(r)? as usize;
    let num_kv_groups = read_u32(r)? as usize;

    Ok(crate::gemv::LayerWeightData {
        q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj,
        input_norm_gamma, post_attn_norm_gamma,
        num_heads, num_kv_heads, head_dim, num_kv_groups,
    })
}
