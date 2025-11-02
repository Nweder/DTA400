import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# ---------- Hjälpfunktioner ----------

def ensure_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def detect_ground_level(z, dataset_name, bins=100):
    """Histogram-baserad marknivå."""
    hist, edges = np.histogram(z, bins=bins)
    i = np.argmax(hist)
    ground = 0.5 * (edges[i] + edges[i+1])

    plt.figure()
    plt.hist(z, bins=bins)
    plt.axvline(ground, linestyle="--")
    plt.title(f"Ground histogram – {dataset_name}")
    plt.xlabel("Z"); plt.ylabel("Freq")
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_ground_hist.png", dpi=300)
    plt.close()
    return ground

def run_dbscan(X2d, eps, min_samples):
    """Kör DBSCAN på X,Y."""
    db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X2d)
    labels = db.labels_
    n_points = labels.size
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n_points if n_points else 0.0

    uniq = set(labels)
    n_clusters = len(uniq) - (1 if -1 in uniq else 0)

    largest = 0
    if n_clusters > 0:
        for lab in uniq:
            if lab == -1:
                continue
            sz = int(np.sum(labels == lab))
            if sz > largest:
                largest = sz
    return n_clusters, noise_ratio, largest, labels

def plot_clusters(X2d, labels, dataset_name, eps, min_samples):
    """Snabb 2D-plot för en konfiguration."""
    plt.figure(figsize=(6,6))
    plt.scatter(X2d[:,0], X2d[:,1], c=labels, s=1, cmap="tab20")
    plt.title(f"{dataset_name}: eps={eps:.3f}, min_samples={int(min_samples)}")
    plt.xlabel("X"); plt.ylabel("Y")
    fn = f"plots/{dataset_name}_clusters_eps{eps:.3f}_m{int(min_samples)}.png".replace('.', '_')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()

def kdist_elbow(X2d, dataset_name, k=4):
    """Elbow-kurva för eps-gissning."""
    nn = NearestNeighbors(n_neighbors=k).fit(X2d)
    dists, _ = nn.kneighbors(X2d)
    d = np.sort(dists[:, k-1])
    plt.figure()
    plt.plot(d)
    plt.title(f"k-distance (k={k}) – {dataset_name}")
    plt.xlabel("Sorted points"); plt.ylabel("k-distance")
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_kdist_k{k}.png", dpi=300)
    plt.close()

def ecdf(y):
    y = np.sort(y)
    n = y.size
    x = np.arange(1, n+1)/n
    return y, x

def spearman_rankcorr(x, y):
    """Spearman approx via rangordning."""
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    c = np.corrcoef(rx, ry)[0,1]
    return float(c)

# ---------- Grid-svep (din ursprungliga del, lätt putsad) ----------

def grid_sweep(XY, eps_values, min_samples_values, dataset_name, max_plots=6):
    rows = []
    plot_count = 0
    for m in min_samples_values:
        for eps in eps_values:
            n_clusters, noise_ratio, largest, labels = run_dbscan(XY, eps, m)
            rows.append([eps, m, n_clusters, noise_ratio, largest])

            if plot_count < max_plots:
                plot_clusters(XY, labels, dataset_name, eps, m)
                plot_count += 1

            print(f"[GRID {dataset_name}] eps={eps:.3f} m={int(m)} -> clusters={n_clusters} noise={noise_ratio:.3f} largest={largest}")

    rows = np.array(rows, dtype=float)
    out_csv = f"results/{dataset_name}_grid.csv"
    np.savetxt(out_csv,
               rows,
               delimiter=",",
               header="eps,min_samples,n_clusters,noise_ratio,largest_cluster",
               comments="")
    print(f"[GRID {dataset_name}] Saved: {out_csv}")

    # Plotta vs eps för minsta min_samples (översikt)
    m0 = int(np.min(min_samples_values))
    mask = rows[:,1] == m0
    r = rows[mask]
    idx = np.argsort(r[:,0]); r = r[idx]

    plt.figure(); plt.plot(r[:,0], r[:,2], marker='o')
    plt.xlabel("eps"); plt.ylabel("number of clusters")
    plt.title(f"{dataset_name}: clusters vs eps (min_samples={m0})")
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_clusters_vs_eps_m{m0}.png", dpi=300)
    plt.close()

    plt.figure(); plt.plot(r[:,0], r[:,3], marker='o')
    plt.xlabel("eps"); plt.ylabel("noise ratio")
    plt.title(f"{dataset_name}: noise vs eps (min_samples={m0})")
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_noise_vs_eps_m{m0}.png", dpi=300)
    plt.close()

    plt.figure(); plt.plot(r[:,0], r[:,4], marker='o')
    plt.xlabel("eps"); plt.ylabel("largest cluster size")
    plt.title(f"{dataset_name}: largest cluster vs eps (min_samples={m0})")
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_largest_vs_eps_m{m0}.png", dpi=300)
    plt.close()

# ---------- Monte Carlo ----------

def monte_carlo(XY,
                dataset_name,
                N=2000,
                eps_prior=("uniform", 0.05, 0.25),
                min_s_prior=("discrete_uniform", 3, 7),
                subsample_frac=0.8,
                seed=42):
    """
    Kör Monte Carlo över parametrar + bootstrap/subsampling av punkter.
    Sparar:
      - results/{name}_mc_samples.csv   (råa samples och utfall)
      - results/{name}_mc_summary.json  (sammanfattning)
      - plots/{name}_mc_*.png           (figurer)
    """
    rng = np.random.default_rng(seed)

    def draw_eps(size):
        if eps_prior[0] == "uniform":
            _, a, b = eps_prior
            return rng.uniform(a, b, size=size)
        elif eps_prior[0] == "triangular":
            _, left, mode, right = eps_prior
            return rng.triangular(left, mode, right, size=size)
        else:
            raise ValueError("Okänd eps_prior")

    def draw_min_samples(size):
        kind = min_s_prior[0]
        if kind == "discrete_uniform":
            _, a, b = min_s_prior
            return rng.integers(a, b+1, size=size)
        elif kind == "list":
            _, vals = min_s_prior
            idx = rng.integers(0, len(vals), size=size)
            return np.array([vals[i] for i in idx], dtype=int)
        else:
            raise ValueError("Okänd min_s_prior")

    def subsample(X, frac):
        n = X.shape[0]
        k = max(2, int(frac*n))
        # bootstrap: ersättning=True
        idx = rng.choice(n, size=k, replace=True)
        return X[idx]

    eps_s = draw_eps(N)
    ms_s  = draw_min_samples(N)

    # Kör simulering
    n_cl_list, noise_list, largest_list = [], [], []
    for i in range(N):
        XY_i = subsample(XY, subsample_frac)
        eps_i = float(eps_s[i])
        ms_i  = int(ms_s[i])
        n_cl, noise, largest, _ = run_dbscan(XY_i, eps_i, ms_i)
        n_cl_list.append(n_cl)
        noise_list.append(noise)
        largest_list.append(largest)

        if (i+1) % max(1, N//10) == 0:
            print(f"[MC {dataset_name}] {i+1}/{N}")

    # Till DataFrame
    df = pd.DataFrame({
        "eps": eps_s,
        "min_samples": ms_s,
        "n_clusters": n_cl_list,
        "noise_ratio": noise_list,
        "largest_cluster": largest_list
    })

    # Sammanfattning
    def pct(x, p): return float(np.percentile(x, p))
    def ci_mean(x):
        se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        m  = float(np.mean(x))
        return (m - 1.96*se, m + 1.96*se)

    ncl = df["n_clusters"].to_numpy()
    noi = df["noise_ratio"].to_numpy()
    lar = df["largest_cluster"].to_numpy()

    summary = {
        "N": int(N),
        "subsample_frac": float(subsample_frac),
        "seed": int(seed),
        "eps_prior": eps_prior,
        "min_s_prior": min_s_prior,

        "n_clusters": {
            "mean": float(np.mean(ncl)),
            "std": float(np.std(ncl, ddof=1)),
            "p5": pct(ncl,5), "p50": pct(ncl,50), "p95": pct(ncl,95),
            "ci95_mean": ci_mean(ncl)
        },
        "noise_ratio": {
            "mean": float(np.mean(noi)),
            "std": float(np.std(noi, ddof=1)),
            "p5": pct(noi,5), "p50": pct(noi,50), "p95": pct(noi,95),
            "ci95_mean": ci_mean(noi)
        },
        "largest_cluster": {
            "mean": float(np.mean(lar)),
            "std": float(np.std(lar, ddof=1)),
            "p5": pct(lar,5), "p50": pct(lar,50), "p95": pct(lar,95),
            "ci95_mean": ci_mean(lar)
        },
        # Exempel riskmått: sannolikhet att få ≥5 kluster
        "risk_many_clusters_p_ge_5": float(np.mean(ncl >= 5))
    }

    # Känslighet (Spearman)
    sens = {
        "spearman_eps_to_n_clusters": spearman_rankcorr(df["eps"], df["n_clusters"]),
        "spearman_min_samples_to_n_clusters": spearman_rankcorr(df["min_samples"], df["n_clusters"]),
        "spearman_eps_to_noise": spearman_rankcorr(df["eps"], df["noise_ratio"]),
        "spearman_min_samples_to_noise": spearman_rankcorr(df["min_samples"], df["noise_ratio"]),
        "spearman_eps_to_largest": spearman_rankcorr(df["eps"], df["largest_cluster"]),
        "spearman_min_samples_to_largest": spearman_rankcorr(df["min_samples"], df["largest_cluster"]),
    }
    summary["sensitivity_spearman"] = sens

    # --- Plots (MC) ---
    # Histogram och ECDF för n_clusters
    plt.figure(); plt.hist(ncl, bins=40, density=True); plt.title(f"{dataset_name} MC: n_clusters histogram")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_hist_nclusters.png", dpi=300); plt.close()

    y,x = ecdf(ncl); plt.figure(); plt.plot(y, x); plt.title(f"{dataset_name} MC: n_clusters ECDF")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_ecdf_nclusters.png", dpi=300); plt.close()

    # Histogram noise
    plt.figure(); plt.hist(noi, bins=40, density=True); plt.title(f"{dataset_name} MC: noise_ratio histogram")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_hist_noise.png", dpi=300); plt.close()

    # Histogram largest
    plt.figure(); plt.hist(lar, bins=40, density=False); plt.title(f"{dataset_name} MC: largest_cluster histogram")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_hist_largest.png", dpi=300); plt.close()

    # Konvergens för P95(n_clusters)
    steps = np.linspace(100, N, 40, dtype=int)
    p95_run = [np.percentile(ncl[:k], 95) for k in steps]
    plt.figure(); plt.plot(steps, p95_run)
    plt.title(f"{dataset_name} MC: convergence P95(n_clusters)")
    plt.xlabel("iterations"); plt.ylabel("P95")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_convergence_p95_nclusters.png", dpi=300); plt.close()

    # Tornado-liknande bar av |Spearman|
    keys = list(sens.keys())
    vals = [abs(sens[k]) for k in keys]
    idx = np.argsort(vals)[::-1]
    keys_sorted = [keys[i] for i in idx]
    vals_sorted = [vals[i] for i in idx]
    plt.figure(figsize=(7,4)); plt.barh(range(len(vals_sorted)), vals_sorted)
    plt.yticks(range(len(vals_sorted)), keys_sorted)
    plt.title(f"{dataset_name} MC: sensitivity (|Spearman|)")
    plt.tight_layout(); plt.savefig(f"plots/{dataset_name}_mc_sensitivity.png", dpi=300); plt.close()

    # Spara resultat
    samples_csv = f"results/{dataset_name}_mc_samples.csv"
    df.to_csv(samples_csv, index=False)
    summary_json = f"results/{dataset_name}_mc_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[MC {dataset_name}] Saved: {samples_csv}")
    print(f"[MC {dataset_name}] Saved: {summary_json}")
    return df, summary

# ---------- Huvudflöde ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python simulation_mc.py <dataset.npy> [N_mc] [eps_start eps_stop eps_step] [min_s_min min_s_max] [subsample_frac] [seed]")
        sys.exit(1)

    ensure_dirs()
    dataset_path = sys.argv[1]
    name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Standardparametrar
    N_mc = int(sys.argv[2]) if len(sys.argv) >= 3 else 2000
    if len(sys.argv) >= 6:
        eps_start, eps_stop, eps_step = map(float, sys.argv[3:6])
    else:
        eps_start, eps_stop, eps_step = 0.05, 0.25, 0.025
    if len(sys.argv) >= 8:
        ms_min, ms_max = map(int, sys.argv[6:8])
    else:
        ms_min, ms_max = 3, 7
    subsample_frac = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.8
    seed = int(sys.argv[9]) if len(sys.argv) >= 10 else 42

    # Ladda data
    pcd = np.load(dataset_path)  # (N,3+)
    assert pcd.shape[1] >= 3, "Dataset måste ha minst 3 kolumner (X,Y,Z)."
    print(f"[{name}] points: {pcd.shape[0]}")

    # Markfiltrering
    ground = detect_ground_level(pcd[:,2], name)
    above = pcd[pcd[:,2] > ground]
    XY = above[:, :2]
    print(f"[{name}] ground={ground:.3f}, remaining above-ground: {XY.shape[0]}")

    # Elbow-kurva
    kdist_elbow(XY, name, k=4)

    # Grid-svep
    eps_values = np.arange(eps_start, eps_stop + 1e-12, eps_step)
    min_samples_values = list(range(ms_min, ms_max + 1))
    grid_sweep(XY, eps_values, min_samples_values, name, max_plots=6)

    # Monte Carlo
    eps_prior = ("uniform", eps_start, eps_stop)           # byt till ("triangular", left, mode, right) om du vill
    min_s_prior = ("discrete_uniform", ms_min, ms_max)     # eller ("list", [3,4,5,6,7])
    monte_carlo(XY,
                dataset_name=name,
                N=N_mc,
                eps_prior=eps_prior,
                min_s_prior=min_s_prior,
                subsample_frac=subsample_frac,
                seed=seed)

    print(f"[{name}] Done. Se 'plots/' och 'results/'.")

if __name__ == "__main__":
    main()
