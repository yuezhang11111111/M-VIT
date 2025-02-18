import cv2
import numpy as np
from PIL import Image
import skimage
import skimage.morphology
import os

def remove_small_hole(mask, h_size=10):
    """remove the small hole

    Args:
        mask (_type_): a binary mask, can be 0-1 or 0-255
        h_size (int, optional): min_size of the hole

    Returns:
        mask
    """
    value = np.unique(mask)
    if len(value) > 2:
        return None
    pre_mask_rever = mask <= 0
    pre_mask_rever = skimage.morphology.remove_small_objects(pre_mask_rever,min_size=h_size)
    mask[pre_mask_rever <= 0] = np.max(mask)
    return mask

n=0
input_folder="/media/ubuntu/1276A91876A8FD9B/zy/ori"
svs_files = [f for f in os.listdir(input_folder) if f.endswith('.svs')]

WSI_files='/media/ubuntu/1276A91876A8FD9B/zy/WSI_path'
for svs_file in svs_files:
    svs_file = os.path.splitext(svs_file)[0]
    result_path = os.path.join(WSI_files,svs_file + '.svs', svs_file + ".svs_1_1.npy")
    X = np.load(result_path)



    gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    # Remove noise using a Gaussian filter
    gray = cv2.GaussianBlur(gray, (35, 35), 0)
    # Otsu thresholding and mask generation
    ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_size=10
    # 删除小的组织区域
    thresh_otsu = remove_small_hole(thresh_otsu, min_size * 4)
    thresh_otsu = 255 - thresh_otsu
    # 删除小的空白区域
    thresh_otsu = remove_small_hole(thresh_otsu, min_size)

    #闭运算：先膨胀后腐蚀  开运算：先腐蚀后膨胀
    kernel = np.ones((100, 100), np.uint8)#巻积核大小
    thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    output_folder='/media/ubuntu/1276A91876A8FD9B/zy/mask'
    np.save(os.path.join(output_folder,svs_file+'-01.svsmask.npy'), thresh_otsu)
    n=n+1
    print(n,svs_file)




