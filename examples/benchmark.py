import numpy as np
import time
import rasterqc  # your Rust extension


def numpy_stats(arr):
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def numpy_histogram(arr, bins):
    hist, _ = np.histogram(arr, bins=bins)
    return hist


def numpy_entropy(counts):
    # Convert counts to probabilities
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probabilities = counts / total

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def numpy_masked_stats(arr, mask):
    m = arr[mask]
    return {
        "min": float(np.min(m)),
        "max": float(np.max(m)),
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
        "median": float(np.median(m)),
    }


def numpy_percentile(arr, p):
    return float(np.percentile(arr, p))


def numpy_skewness(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return float(np.mean(((arr - mean) / std) ** 3))


def numpy_kurtosis(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return float(np.mean(((arr - mean) / std) ** 4) - 3)


def benchmark(label, func, *args):
    t0 = time.perf_counter()
    result = func(*args)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def main():
    arr = np.random.rand(2048, 2048).astype(np.float32)
    mask = arr > 0.5
    bins = 64

    print("=== Benchmark: Rust vs NumPy ===")

    # --- Basic Stats ---
    rust_stats, rust_t = benchmark("rust_stats", rasterqc.stats_fast, arr)
    numpy_stats_res, np_t = benchmark("numpy_stats", numpy_stats, arr)

    print("\nBasic Stats:")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Histogram ---
    rust_hist, rust_t = benchmark("rust_hist", rasterqc.histogram, arr, bins)
    numpy_hist_res, np_t = benchmark("numpy_hist", numpy_histogram, arr, bins)

    print("\nHistogram:")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Entropy ---
    rust_entropy, rust_t = benchmark("rust_entropy", rasterqc.entropy, rust_hist)
    numpy_entropy_res, np_t = benchmark("numpy_entropy", numpy_entropy, numpy_hist_res)

    print("\nEntropy:")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Masked Stats ---
    # rust_mstats, rust_t = benchmark(
    #    "rust_masked_stats", rasterqc.masked_stats, arr, mask
    # )
    # numpy_mstats_res, np_t = benchmark(
    #    "numpy_masked_stats", numpy_masked_stats, arr, mask
    # )

    # print("\nMasked Stats:")
    # print(f"Rust: {rust_t * 1000:.3f} ms")
    # print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Percentile ---
    rust_p95, rust_t = benchmark("rust_percentile", rasterqc.percentile, arr, 95)
    numpy_p95, np_t = benchmark("numpy_percentile", numpy_percentile, arr, 95)

    print("\nPercentile (95):")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Skewness ---
    rust_skew, rust_t = benchmark("rust_skewness", rasterqc.skewness, arr)
    numpy_skew_res, np_t = benchmark("numpy_skewness", numpy_skewness, arr)

    print("\nSkewness:")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")

    # --- Kurtosis ---
    rust_kurt, rust_t = benchmark("rust_kurtosis", rasterqc.kurtosis, arr)
    numpy_kurt_res, np_t = benchmark("numpy_kurtosis", numpy_kurtosis, arr)

    print("\nKurtosis:")
    print(f"Rust: {rust_t * 1000:.3f} ms")
    print(f"NumPy: {np_t * 1000:.3f} ms")


if __name__ == "__main__":
    main()
