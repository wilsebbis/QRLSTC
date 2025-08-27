import pickle
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

# Example:
cl_res = pickle.load(open("out/classical_k3_a200.pkl","rb"))
qm_res = pickle.load(open("out/quantum_k3_a200.pkl","rb"))

print("Classical SSE, N, mean:", classical_sse_from_res(cl_res))
print("Quantum   SSE, N, mean:", classical_sse_from_res(qm_res))