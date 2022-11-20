# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot(Data,pixel_size,frames):
    for traj in Data:
        if len(traj) >= frames:
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
    data = pd.read_excel('../outputs/Test-R2_yxt.xlsx',index_col=0)
    temp_Data = np.array(data).tolist()
    
    Data = []
    for traj in temp_Data:
        str_traj = [i for i in traj if not pd.isnull(i)]
        list_traj = [json.loads(i) for i in str_traj ]
        Data.append(list_traj)
    
    pixel_size = 135.1 # unit: nm
    dt = 1/19.86  # 19.86 frames/s
    frames = 30  # the num of frames recording the trajectory
    
    plot(Data,pixel_size,frames)


