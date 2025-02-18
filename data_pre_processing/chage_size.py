
import numpy as np

#Change file size
original_array = np.load('/media/ubuntu/1276A91876A8FD9B/zy/WSI_path/TCGA-HZ-7926-01.svs/TCGA-HZ-7926-01.svs_0_0.npy')
new_shape = (20000, 20000)
resized_array = np.resize(original_array, new_shape)
np.save('/media/ubuntu/1276A91876A8FD9B/zy/WSI_path/TCGA-HZ-7926-01.svs/TCGA-HZ-7926-01.svs_0_0.npy', resized_array)