import numpy as np
import pandas as pd
from dataloader import gen_dataset
from MVIT import mvit
from losses import concordance_index


image_size = 224
model = mvit.vit_b16(
    image_size=image_size,
    classes=1,
    activation=None,
    include_top=True,
    pretrained=True,
    pretrained_top=False)

model.load_weights('best_model1.h5')

clini_df = pd.read_csv('valid（1）.csv')
name_list = clini_df.id.tolist()

file_dir = '/media/ubuntu/1276A91876A8FD9B/zy'
valid_img_list, valid_os_list, valid_os_event_list, \
valid_dfs_list, valid_dfs_event_list = gen_dataset(file_dir, name_list, clini_df)
valid_img = np.array(valid_img_list)

valid_score = -model.predict(valid_img)[:, 0]

valid_y=np.stack([valid_os_list,valid_os_event_list,valid_dfs_list,valid_dfs_event_list], axis=1).astype('float64')

os_ci_valid= concordance_index(valid_y[:,0:2],valid_score)
dfs_ci_valid= concordance_index(valid_y[:,2:4],valid_score)
print(f'valid os cindex:{os_ci_valid}')
print(f'valid dfs cindex:{dfs_ci_valid}')

pd.DataFrame({'id': name_list, 'score': -valid_score}).to_csv('valid_score.csv')
