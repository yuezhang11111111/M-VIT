import os
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator

def svs_to_path(input_folder,output_folder):

    n=0
    if not os.path.exists(output_folder ):
        os.makedirs(output_folder )

    svs_files=[f for f in os.listdir(input_folder) if f.endswith('.ndpi')]

    for svs_file in svs_files:

        svs_path=os.path.join(input_folder,svs_file)
        slide = openslide.open_slide(svs_path)

        n=n+1
        print(n)

        highth = 20000
        width = 20000
        data_gen = DeepZoomGenerator(slide, tile_size=highth, overlap=0, limit_bounds=False) 
        print(data_gen.tile_count)
        print(data_gen.level_count)
        m=data_gen.level_count-1
        print(m)
        folder_name = svs_file
        result_path = os.path.join(output_folder, folder_name)

        if not os.path.exists(result_path):
            os.mkdir(os.path.join(output_folder,folder_name))
        else:
            print(f"Folder {folder_name} exists. Exiting loop.")
            continue


        [w, h] = slide.dimensions
        print(w, h)
        num_w = int(np.floor(w / width)) + 1
        num_h = int(np.floor(h / highth)) + 1
        for i in range(num_w):
            if i==6:
                break
            for j in range(num_h):

                img = np.array(data_gen.get_tile(m, (i, j))) 
                np.save(os.path.join(result_path, svs_file + '_' + str(i) + '_' + str(j)), img)  



        slide.close()

    print("Conversion completed!")


input_folder="/media/ubuntu/1276A91876A8FD9B/zy/test"
output_folder="/media/ubuntu/1276A91876A8FD9B/zy/panwsi_path"

svs_to_path(input_folder,output_folder)
