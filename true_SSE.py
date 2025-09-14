import pickle
import argparse
import matplotlib.pyplot as plt
from trajdistance import traj2trajIED

def classical_sse_from_res(res):
    sse, n = 0.0, 0
    cluster_dict = res[0][2]
    for idx in cluster_dict:
        center = cluster_dict[idx][1]
        for tr in cluster_dict[idx][3]:
            d = traj2trajIED(center.points, tr.points)
            if d != 1e10:  # skip no-overlap sentinel
                sse += d*d
                n += 1
    return sse, n, (sse/n if n else float('inf'))

def plot_elbow(k_values, classical_means, quantum_means, out_png):
    plt.figure(figsize=(10, 6))
    if classical_means:
        plt.plot(k_values, classical_means, 'bo-', label='Classical')
    if quantum_means:
        plt.plot(k_values, quantum_means, 'ro-', label='Quantum')
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Mean SSE')
    plt.title('Elbow Plot: SSE vs Number of Clusters')
    plt.grid(True)
    plt.legend()
    plt.xticks(k_values)
    plt.savefig(out_png)
    plt.close()
    print(f"Elbow plot saved to {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Generate elbow plot for SSE across different k values")
    parser.add_argument("--k-values", "-k", nargs='+', type=int, required=True,
                      help="List of k values to analyze")
    parser.add_argument("--amount", "-a", type=int, default=200,
                      help="Amount value used in filenames")
    parser.add_argument("--classical", "-c", action="store_true",
                      help="Include classical results")
    parser.add_argument("--quantum", "-q", action="store_true",
                      help="Include quantum results")
    parser.add_argument("--out", default="elbow_plot.png",
                      help="Output PNG file for the elbow plot")
    args = parser.parse_args()

    k_values = sorted(args.k_values)
    classical_means = []
    quantum_means = []

    for k in k_values:
        print(f"\nAnalyzing k={k}...")
        
        if args.classical:
            try:
                cl_res = pickle.load(open(f"out/classical_k{k}_a{args.amount}.pkl", "rb"))
                _, _, cl_mean = classical_sse_from_res(cl_res)
                classical_means.append(cl_mean)
                print(f"Classical mean SSE for k={k}: {cl_mean:.6f}")
            except FileNotFoundError:
                print(f"Warning: Classical results file not found for k={k}")
                classical_means = []  # Clear to avoid partial plotting
        
        if args.quantum:
            try:
                qm_res = pickle.load(open(f"out/quantum_k{k}_a{args.amount}.pkl", "rb"))
                _, _, qm_mean = classical_sse_from_res(qm_res)
                quantum_means.append(qm_mean)
                print(f"Quantum mean SSE for k={k}: {qm_mean:.6f}")
            except FileNotFoundError:
                print(f"Warning: Quantum results file not found for k={k}")
                quantum_means = []  # Clear to avoid partial plotting

    if not (classical_means or quantum_means):
        print("Error: No data found for plotting")
        return

    plot_elbow(k_values, classical_means, quantum_means, args.out)

if __name__ == "__main__":
    main()