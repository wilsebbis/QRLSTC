import pickle
import numpy as np
from datetime import datetime
from point import Point
from traj import Traj
from trajdistance import traj_mdl_comp
import random
import argparse
import time

def processtrajs(trajs):
    print("Starting to process trajectories...")
    start_time = time.time()
    trajslist = []
    total = len(trajs)
    step = max(1, total // 10)  # Update every 10%

    for i in range(len(trajs)):
        if i % step == 0 or i == total - 1:
            print(f"Processing {int((i / total) * 100)}% done...")

        temptraj = []
        for j in range(len(trajs[i])):
            if trajs[i][j][1] >= 39.4 and trajs[i][j][1] <= 41.6 and trajs[i][j][0] >= 115.4 and trajs[i][j][0] <= 117.5:
                temptraj.append(trajs[i][j])
        if len(temptraj) != 0:
            trajslist.append(temptraj)

    end_time = time.time()
    print(f"→ Trajectories processed in {end_time - start_time:.2f} seconds.\n")
    return trajslist

def processlength(trajs, max_length, min_length):
    print("Processing trajectory lengths...")
    start_time = time.time()
    trajdata = []
    total = len(trajs)
    step = max(1, total // 10)  # Update every 10%

    for i in range(len(trajs)):
        if i % step == 0 or i == total - 1:
            print(f"Processing {int((i / total) * 100)}% done...")

        length = len(trajs[i])
        if length > max_length:
            temp_traj = []
            length_list = [i for i in range(length)]
            random_sample = random.sample(length_list, max_length)
            sorted_sample = sorted(random_sample)
            for idx in sorted_sample:
                temp_traj.append(trajs[i][idx])
            trajdata.append(temp_traj)
        elif min_length <= length <= max_length:
            trajdata.append(trajs[i])

    end_time = time.time()
    print(f"→ Trajectory length processing completed in {end_time - start_time:.2f} seconds.\n")
    return trajdata

def split_traj(traj, max_length, min_length):
    print("Splitting trajectories...")
    start_time = time.time()
    sub_trajs = []
    start = 0
    while start < len(traj):
        end = start + max_length
        if end > len(traj):
            end = len(traj)
        if end - start + 1 >= min_length:
            sub_traj = traj[start:end]
            sub_trajs.append(sub_traj)
        start = end

    end_time = time.time()
    print(f"→ Splitting done in {end_time - start_time:.2f} seconds.\n")
    return sub_trajs

def normloctrajs(trajs):
    print("Normalizing trajectory locations...")
    start_time = time.time()
    norm_trajs, lons, lats = [], [], []
    for i in range(len(trajs)):
        for j in range(len(trajs[i])):
            lons.append(trajs[i][j][0])
            lats.append(trajs[i][j][1])
    mean_lon, mean_lat = np.mean(lons), np.mean(lats)
    std_lon, std_lat = np.std(lons), np.std(lats)

    for i in range(len(trajs)):
        tmp_traj = []
        for j in range(len(trajs[i])):
            norm_lon = (trajs[i][j][0] - mean_lon) / std_lon
            norm_lat = (trajs[i][j][1] - mean_lat) / std_lat
            tmp_traj.append([norm_lon, norm_lat, trajs[i][j][2]])
        norm_trajs.append(np.array(tmp_traj))

    end_time = time.time()
    print(f"→ Location normalization completed in {end_time - start_time:.2f} seconds.\n")
    return norm_trajs

def normtimetrajs(trajs):
    print("Normalizing trajectory time...")
    start_time = time.time()
    norm_trajs, ts = [], []
    for i in range(len(trajs)):
        for j in range(len(trajs[i])):
            ts.append(trajs[i][j][2])
    mean_t = np.mean(ts)
    std_t = np.std(ts)

    for i in range(len(trajs)):
        tmp_traj = []
        for j in range(len(trajs[i])):
            norm_t = (trajs[i][j][2] - mean_t) / std_t
            tmp_traj.append([trajs[i][j][0], trajs[i][j][1], norm_t])
        norm_trajs.append(np.array(tmp_traj))

    end_time = time.time()
    print(f"→ Time normalization completed in {end_time - start_time:.2f} seconds.\n")
    return norm_trajs

def convert2traj(trajdata):
    print("Converting trajectory data to Traj objects...")
    start_time = time.time()
    trajlists = []
    total = len(trajdata)
    step = max(1, total // 10)  # Update every 10%

    for i in range(len(trajdata)):
        if i % step == 0 or i == total - 1:
            print(f"Converting {int((i / total) * 100)}% done...")

        traj_points = []
        for j in range(len(trajdata[i])):
            p = Point(trajdata[i][j][0], trajdata[i][j][1], trajdata[i][j][2])
            traj_points.append(p)
        ts, te = traj_points[0].t, traj_points[-1].t
        size = len(traj_points)
        traj = Traj(traj_points, size, ts, te, i)
        trajlists.append(traj)

    end_time = time.time()
    print(f"→ Trajectory conversion completed in {end_time - start_time:.2f} seconds.\n")
    return trajlists

def simplify(points, traj_id):
    simp_points = []
    start_index = 0
    length = 1
    simp_points.append(points[start_index])
    while start_index + length < len(points):
        curr_index = start_index + length
        cost_simp = traj_mdl_comp(points, start_index, curr_index, 'simp')
        cost_origin = traj_mdl_comp(points, start_index, curr_index, 'orign')
        if cost_simp > cost_origin:
            p = points[curr_index]
            simp_points.append(p)
            start_index = curr_index
            length = 1
        else:
            length += 1
    if not simp_points[-1].equal(points[-1]):
        simp_points.append(points[-1])
    ts = simp_points[0].t
    te = simp_points[-1].t
    size = len(simp_points)
    simp_traj = Traj(simp_points, size, ts, te, traj_id)
    return simp_traj

def getsimptrajs(trajs):
    print("Simplifying trajectories...")
    start_time = time.time()
    simptrajs = []
    total = len(trajs)
    step = max(1, total // 10)  # Update every 10%

    for i in range(len(trajs)):
        if i % step == 0 or i == total - 1:
            print(f"Simplifying {int((i / total) * 100)}% done...")

        simp_traj = simplify(trajs[i].points, trajs[i].traj_id)
        simptrajs.append(simp_traj)

    end_time = time.time()
    print(f"→ Simplification completed in {end_time - start_time:.2f} seconds.\n")
    return simptrajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess trajectories")
    parser.add_argument("-trajfile", default='data/Tdrive', help="Input trajectory file")
    parser.add_argument("-maxlen", type=int, default=500, help="Maximum length of trajectories")
    parser.add_argument("-minlen", type=int, default=10, help="Minimum length of trajectories")
    parser.add_argument("-simpledtrajfile", default='data/Tdrive_norm_traj_QRLSTC', help="Output simplified trajectory file")

    args = parser.parse_args()

    print("Starting QRLSTC Preprocessing...")
    start_time = time.time()

    trajs = pickle.load(open(args.trajfile, 'rb'))
    trajslist = processtrajs(trajs)
    trajs = processlength(trajslist, args.maxlen, args.minlen)
    norm_trajs = normtimetrajs(trajs)
    trajlists = convert2traj(norm_trajs)
    simpletrajs = getsimptrajs(trajlists)

    # Save the simplified trajectories to the specified output file
    pickle.dump(simpletrajs, open(args.simpledtrajfile, 'wb'), protocol=2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"QRLSTC Preprocessing finished in {elapsed_time:.2f} seconds.")
