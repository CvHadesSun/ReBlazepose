import numpy as np
import os
import glob
from tqdm import tqdm
dir = './'

npy_files = glob.glob(os.path.join(dir,'*.npy'))

# for i in tqdm(range(len(npy_files)),desc="tranform..."):
# for i in range(len(npy_files)):

# print(npy_files)
file_name = npy_files[-3].replace('.npy','.txt')
fp = open(file_name,'w')
data = np.load(npy_files[-3])
print(file_name)
# data.squeeze()
if len(data.shape)>2:
    data = data.reshape(-1)
# print(data)
np.savetxt(fp,data,fmt="%f",delimiter=" ")







