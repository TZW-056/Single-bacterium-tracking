# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot(Data,pixel_size,frames):
    for traj in Data:
        if len(traj) >= 100:
            traj = np.array(traj) * pixel_size / 1000
            plt.plot(traj[:frames, 0], traj[:frames, 1])
    
    plt.axis('scaled')
    plt.ylabel('y(${\mu}m$)', fontsize=17)
    plt.xlabel('x(${\mu}m$)', fontsize=17)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize = 12)
    plt.rcParams['figure.dpi'] = 600
    plt.show()
    
if __name__ == '__main__':
    f = open('../outputs/Test-B02_yxt.pkl','rb')
    Data = pickle.load(f)
    pixel_size = 135.1 # unit: nm
    dt = 1/19.86  # 19.86 frames/s
    frames = 30
    
    plot(Data,pixel_size,frames)


