import pickle as pkl
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "/data3/rlbench_demos/pick_up_coke_1k_match/converted/demo_1.pkl"

with open(path, 'rb') as dbfile:
    db = pkl.load(dbfile)

# visualize the frames
for i in range(5):
    frame = db['rgb_frames'][10, i]
    plt.imshow(frame)
    # add title
    plt.title(f'Cam {i}')
    plt.show()


# I want frames 0, 2, 3
