import pickle as pkl
import easydict
import numpy as np

syn_path = "/data3/rlbench_demos/sim_demos_domain_rand/converted/demo_50.pkl"
# syn_path = "/data3/rlbench_demos/sim_demos/converted/demo_1.pkl"
# real_path = "/data3/raw_robot_data/coke_demos/demonstration_1/demo_1.pkl"

# load both files
with open(syn_path, 'rb') as f:
    syn_db = pkl.load(f)
# with open(real_path, 'rb') as f:
#     real_db = pkl.load(f)

# visualize image observations
import cv2
import matplotlib.pyplot as plt
breakpoint()
syn_img = syn_db['rgb_frames'][0, 0]
flipped_img = syn_img[..., ::-1]
# convert to rgb
# real_img = real_db['rgb_frames'][0, 0]

fig, ax = plt.subplots(1, 2)
ax[0].imshow(syn_img)
ax[0].set_title("Synthetic")
ax[1].imshow(flipped_img)
ax[1].set_title("flipped")
plt.show()
# for k in syn_db.keys():
#     if k in ['controller_type', 'controller_cfg']:
#         continue
#     print(k)
#     if isinstance(syn_db[k], list):
#         print("syn:", syn_db[k][0].dtype)
#     elif syn_db[k] is None:
#         print("syn: None")
#     else:
#         print("syn:", syn_db[k].dtype)
#     print("real", real_db[k].dtype)
    # print()
breakpoint()
