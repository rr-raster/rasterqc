/// RasterQC command‑line interface
///
/// A minimal, fast CLI for computing raster quality and distribution metrics
/// over raw `f32` pixel data. The tool reads **little‑endian f32 values from
/// stdin**, making it easy to integrate into scientific imaging pipelines,
/// shell scripts, and batch processing workflows.
///
/// Supported commands:
///
/// * **stats** — compute min, max, mean, std, median  
/// * **histogram** — fixed‑width histogram with configurable bin count  
/// * **percentile** — compute a percentile using the distribution module  
/// * **snr** — signal‑to‑noise ratio (mean / std)  
/// * **uniformity** — 1 − (σ / μ)  
/// * **dr** — dynamic range (linear or dB)  
///
/// All commands accept binary f32 input via stdin. Many commands support
/// optional JSON output for machine‑readable integration.
///
/// Internally, the CLI delegates computation to the `stats`, `quality`,
/// and `distribution` modules, each of which is optimized for performance
///  using parallel processing and efficient algorithms.
///
/// Example usage on raw f32 data array:
///
/// ```bash
/// cat image.raw | rasterqc stats --json
/// cat image.raw | rasterqc histogram --bins 512
/// cat image.raw | rasterqc percentile --p 95
/// cat image.raw | rasterqc dr --db
/// ```

use clap::{Parser, Subcommand};
use rasterqc::{stats, quality, distribution};
use serde_json::json;
use std::io::{self, Read};

#[derive(Parser)]
#[command(name = "rasterqc", version, about = "Raster quality metrics CLI - reads f32 data from stdin")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute basic statistics (min, max, mean, std, median)
    Stats {
        #[arg(long)]
        json: bool,
    },

    /// Compute histogram
    Histogram {
        #[arg(long, default_value_t = 256)]
        bins: usize,
        #[arg(long)]
        json: bool,
    },

    /// Compute percentile
    Percentile {
        #[arg(long)]
        p: f32,
    },

    /// Compute signal-to-noise ratio
    Snr,

    /// Compute uniformity metric
    Uniformity,

    /// Compute dynamic range (linear or dB)
    Dr {
        #[arg(long)]
        db: bool,
    },
}

/// Read f32 binary data from stdin
fn read_stdin_data() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;

    // Convert bytes to f32 values (little-endian)
    if buffer.len() % 4 != 0 {
        return Err("Input size must be a multiple of 4 bytes".into());
    }

    let data: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(data)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Stats { json } => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let (min, max, mean, std, median) = stats::compute_stats(&data).expect("Failed to compute stats");

            if json {
                let out = json!({
                    "min": min,
                    "max": max,
                    "mean": mean,
                    "std": std,
                    "median": median,
                });
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                println!("Min: {}", min);
                println!("Max: {}", max);
                println!("Mean: {}", mean);
                println!("Std: {}", std);
                println!("Median: {}", median);
            }
        }

        Command::Histogram { bins, json } => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let hist = stats::compute_histogram(&data, bins).expect("Failed to compute histogram");

            if json {
                println!("{}", serde_json::to_string_pretty(&hist).unwrap());
            } else {
                println!("Histogram ({} bins):", bins);
                for (i, v) in hist.iter().enumerate() {
                    println!("{:4}: {}", i, v);
                }
            }
        }

        Command::Percentile { p } => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let val = distribution::percentile(&data, p).expect("Failed to compute percentile");
            println!("Percentile {}: {}", p, val);
        }

        Command::Snr => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let snr = quality::snr(&data).expect("Failed to compute SNR");
            println!("SNR: {}", snr);
        }

        Command::Uniformity => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let u = quality::uniformity(&data).expect("Failed to compute uniformity");
            println!("Uniformity: {}", u);
        }

        Command::Dr { db } => {
            let data = read_stdin_data().expect("Failed to read data from stdin");
            let dr = quality::dynamic_range(&data).expect("Failed to compute dynamic range");

            if db {
                let dbv = 20.0 * dr.log10();
                println!("Dynamic Range (dB): {}", dbv);
            } else {
                println!("Dynamic Range: {}", dr);
            }
        }
    }
}
