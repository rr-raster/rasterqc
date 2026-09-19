import numpy as np
import rasterqc


def main():
    # Generate example raster
    arr = np.random.rand(512, 512).astype(np.float64)
    arr = arr.ravel()

    print("=== Basic Stats ===")
    stats = rasterqc.stats(arr)
    print(
        f"min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, std={stats.get('std')}, median={stats.get('median')}"
    )

    print("\n=== Histogram ===")
    hist = rasterqc.histogram(arr, bins=64)
    print(f"histogram (64 bins): {hist[:10]} ...")

    print("\n=== Distribution Stats ===")
    p95 = rasterqc.percentile(arr, 95)
    q50 = rasterqc.quantile(arr, 0.5)
    # iqr = rasterqc.iqr(arr)
    skew = rasterqc.skewness(arr)
    kurt = rasterqc.kurtosis(arr)

    print(f"95th percentile: {p95}")
    print(f"median (quantile 0.5): {q50}")
    # print(f"IQR: {iqr}")
    print(f"skewness: {skew}")
    print(f"kurtosis: {kurt}")

    print("\n=== Masked Stats ===")
    mask = arr > 0.5


# m_min, m_max, m_mean, m_std, m_median = rasterqc.masked_stats(arr, mask)
# print(f"masked mean={m_mean}, masked std={m_std}")

# print("\n=== Masked Histogram ===")
# masked_hist = rasterqc.masked_histogram(arr, mask, bins=64)
# print(f"masked histogram (64 bins): {masked_hist[:10]} ...")


if __name__ == "__main__":
    main()
