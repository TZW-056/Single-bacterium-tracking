import numpy as np
import os
import time
import pickle
from collections import Counter, defaultdict
import cv2
from matplotlib.animation import FuncAnimation
from utils import sort_rows, ProgressBar
import pandas as pd
import matplotlib.pyplot as plt


path_video = '../data/Test-R2.avi'

# the prefix of the file to be saved
fname = os.path.basename(path_video).split('.')[0]

 # Those in the range of lo to hi are considered bacteria(Parameters in Step 2)
lo, hi = 30, 90


########################################################################
# step1: obtain each frame of the video
########################################################################

print("obtain each frame of the video")

videoCapture = cv2.VideoCapture(path_video)
frames = []
while True:
    success, frame = videoCapture.read()
    if success:
        frames.append(frame)
    else:
        break

height, width, _ = frames[0].shape



k3c = np.array([[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]], dtype=bool)
k7c = np.array([[0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]], dtype=bool)

def pre_process(frame, lo=230, ker=np.ones((3, 3), np.uint8)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad = np.uint8(255) - cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, ker)
    edges = cv2.threshold(grad, lo, 255, cv2.THRESH_BINARY)[1]
    return gray, grad, edges


def label_region(j, i, ker, cid, labels, cond):
    """
    height, width: global variables
    ker.shape is odd number
    Return: origin (means previous label)
    """
    stack = [(j, i)]
    origin = {labels[j, i]}
    labels[j, i] = cid
    while stack:
        j, i = stack.pop()
        m, n = ker.shape
        y0, x0 = j - (m >> 1), i - (n >> 1)
        for kj in range(m):
            for ki in range(n):
                if not ker[kj, ki]:
                    continue
                jj = kj + y0
                ii = ki + x0
                if ((0 <= jj < height)
                    and (0 <= ii < width)
                    and labels[jj, ii] in cond):
                    origin.add(labels[jj, ii])
                    labels[jj, ii] = cid
                    stack.append((jj, ii))
    return origin


def find_contour_by_xy(xy):
    xy[:, 1] *= -1
    xy = xy[sort_rows(xy, (1, 2))]
    x, y = xy[:, 0], xy[:, 1]
    if x[0] == x[-1]:
        return np.array([xy[0], xy[-1]])
    outline = []
    outline.append(xy[-1])

    while outline[-1][0] != x[0]:
        idx = np.nonzero(x < outline[-1][0])[0] # <
        ks = (xy[idx, 1] - outline[-1][1]) / (xy[idx, 0] - outline[-1][0])
        idx2 = idx[ks == min(ks)][0]
        outline.append(xy[idx2])

    if outline[-1][1] != y[0]:
        outline.append(xy[0])

    while outline[-1][0] != x[-1]:
        idx = np.nonzero(x > outline[-1][0])[0] # >
        ks = (xy[idx, 1] - outline[-1][1]) / (xy[idx, 0] - outline[-1][0])
        idx2 = idx[ks == min(ks)][-1]
        outline.append(xy[idx2])

    outline = np.array(outline)
    outline[:, 1] *= -1
    return outline


def find_contour_by_x_y(px, py):
    return find_contour_by_xy(np.vstack([px, py]).T)

# Find the outline of a bunch of pixels
def find_contour_by_label(label, labels):
    py, px = np.nonzero(labels==label)
    return find_contour_by_x_y(px, py)


def find_center(outline):
    M = cv2.moments(outline)
    return M['m10']/M['m00'], M['m01']/M['m00']


########################################################################
# step2: Process each frame to obtain the details
########################################################################

print("step2: Process each frame to obtain the details")

# TODO: Verify that frames are almost superimposed on each other
details = []
tic = time.time()
areas = []
for ii, frame in enumerate(frames):
    gray, grad, edges = pre_process(frame)

    labels = np.zeros_like(edges, dtype=int)
    labels[edges == 0] = -1 # -1 means undetermined
    cond = {-1}
    cid = 0
    for j in range(height):
        for i in range(width):
            if labels[j, i] in cond:
                cid += 1
                label_region(j, i, k3c, cid, labels, cond)

    isep = cid + 1
    area_arr = np.zeros((isep, 2))

    small_region = set()
    for l in np.arange(1, isep):
        outline = find_contour_by_label(l, labels)
        if outline.size:
            area_arr[l, 0] = cv2.contourArea(outline)
            if area_arr[l, 0] < 15:
                small_region.add(l)

    same = {}
    for j in range(height):
        for i in range(width):
            if labels[j, i] in small_region:
                cid += 1
                origin = label_region(j, i, k7c, cid, labels, small_region)
                for pre in origin:
                    area_arr[pre, 1] = 1
                if len(origin) == 1:
                    same[cid] = area_arr[list(origin)[0], 0]
                    
    area_arr_2 = np.zeros((cid + 1, 2))
    area_arr_2[:isep] = area_arr
    for l in range(isep, cid + 1):
        if l in same:
            area_arr_2[l, 0] = same[l]
        else:
            outline = find_contour_by_label(l, labels)
            if outline.size:
                area_arr_2[l, 0] = cv2.contourArea(outline)

    details.append([edges, labels, area_arr_2])
    tic = time.time()

########################################################################
# step3: Generate a preliminary trajectory
########################################################################
'''
    First: 
        Select bacteria of moderate size, all overlapping bacteria in the same position around the frame  
    Second:
        Comparison similarity: the most overlapping points, 
        the most similar size to establish the corresponding relationship  
    Third:
        After finding the corresponding, how to cut?  (Do not cut, when disappear processing)  
'''
print("step3: Generate a preliminary trajectory")

tracks = [] # Frame of number.x
centers = []
speeds = []
last = {}
dt = 2

bar = ProgressBar(len(details))
for i, (edges, labels, area) in enumerate(details):
    buf = [] 
    for l in range(1, len(area)):
        buf_l = []
        if area[l, 1] == 0 and (lo <= area[l, 0] <= hi):
            pos = labels == l
            py, px = np.nonzero(pos)
            center = np.array(find_center(find_contour_by_x_y(px, py)))
            if i == 0:
                tracks.append([(i, l)])
                centers.append([center])
                speeds.append([0])
                last[i, l] = len(tracks) - 1
            else:
                for ii in range(i - 1, i - 11, -1):
                    for k, v in Counter(details[ii][1][pos]).items():
                        iseq = last.get((ii, k), -1)
                        if k > 0 and iseq >= 0:
                            if i - tracks[iseq][0][0] >= dt:
                                i3 = len(tracks[iseq]) - 1
                                while i - tracks[iseq][i3][0] < dt:
                                    i3 -= 1
                                spd = (np.sqrt(np.sum(
                                    (center - centers[iseq][i3]) ** 2))
                                       / (i - tracks[iseq][i3][0]))
                            else:
                                spd = height # Arbitrarily selected larger value
                            buf_l.append((v, spd-speeds[iseq][-1], i, l, iseq, center))
                    if buf_l:
                        break
                buf_l.append((0, height, i, l, -1, center))
                buf.extend(buf_l)
                
    used_l = set()
    used_i = set()
    for v, spd, i, l, iseq, center in sorted(
            buf, key=lambda x: (int(abs(x[1]) / 0.1), 3 * abs(x[1]) - x[0])):
        if l in used_l or iseq in used_i:
            continue
        if iseq < 0:
            tracks.append([(i, l)])
            centers.append([center])
            speeds.append([0])
            last[i, l] = len(tracks) - 1
        else:
            assert iseq == last.pop(tracks[iseq][-1])
            tracks[iseq].append((i, l))
            centers[iseq].append(center)
            df = tracks[iseq][-1][0] - tracks[iseq][0][0]
            if df >= dt:
                speeds[iseq].append(speeds[iseq][-1] + spd / (df - dt + 1))
            else:
                speeds[iseq].append(0)
            last[i, l] = iseq
        used_l.add(l)
        if iseq >= 0: 
            used_i.add(iseq)
            
    bar.goto(i + 1)

########################################################################
# step4: Merge the similar frames
########################################################################
print("step4: Merge the similar frames")

# similar frames: nearer frame and no overlap,less difference of speed and position
comb = {}
for i1, seq1 in enumerate(tracks):
    for i2, seq2 in enumerate(tracks[i1:], i1):
        if seq1[-1][0] < seq2[0][0]:
            dd = np.sqrt(np.sum((centers[i1][-1] - centers[i2][0]) ** 2))
            df = seq2[0][0] - seq1[-1][0]
            dv = abs(speeds[i1][-1] - speeds[i2][-1])

buf2 = sorted(buf, key=lambda x: (int(abs(x[1]) / 0.1), 3 * abs(x[1]) - x[0]))

outputs = []
for i, seq in enumerate(tracks):
    outputs.append([xy.tolist() + [seq[j][0]]
                    for j, xy in enumerate(centers[i])])

########################################################################
# step5: Save data
########################################################################
print("step5: Save data to the folder of ./outputs")

df = pd.DataFrame(outputs)
df.to_excel('../outputs/{}_yxt.xlsx'.format(fname))

print("Finished")
