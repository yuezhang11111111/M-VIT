import os
i=0
j=0
n=8
m=5
count=0
input_folder = "/media/ubuntu/1276A91876A8FD9B/zy/down"
svs_files=[f for f in os.listdir(input_folder) if f.endswith('.svs')]
for svs_file in svs_files:
    count=count+1
    print(count)
# Define the folder path and file name to delete
    folder_path = '/media/ubuntu/1276A91876A8FD9B/zy/WSI_path'
    file_to_delete =os.path.join(folder_path,svs_file)
# Constructing the full file path
    for i in range(n):
      for j in range(m):
         if i<2 and j<2:
           continue

         file_path = os.path.join(file_to_delete,svs_file +'_' + str(i) + '_' + str(j) +'.npy')
# Check if a file exists
         if os.path.isfile(file_path):
           os.remove(file_path)
           print(f"The file {file_to_delete} has been deleted.")
         else:
           print(f"The file {file_to_delete} does not exist.")


