import numpy as np
import os
import glob
from tqdm import tqdm
dir = './'

npy_files = glob.glob(os.path.join(dir,'*.npy'))

for i in tqdm(range(len(npy_files)),desc="tranform..."):
    file_name = npy_files[i].replace('.npy','.txt')
    fp = open(file_name,'wb')
    data = np.load(npy_files[i])
    data.squeeze()
    if len(data.shape)>2:
        data = data.reshape(-1)
    np.savetxt(fp,data)







