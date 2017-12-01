import pydicom
import matplotlib.pylab as plt
import os
def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of file path
    '''
    dataset_path = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            dataset_path.append(os.path.join(root,file))
    return dataset_path

path = 'C:\\Users\\wu1114\\Documents\\Data_Science_Bowl\\train\\train\\8\\study\\sax_54\\'
files = get_files(path)
for file in files:
    print(file)

ds = []
for i in range(len(files)):
    temp = pydicom.read_file(files[i])
    ds.append(temp)
    plt.subplot(1,30,i+1)
    plt.axis('off')
    plt.imshow(temp)
plt.show()

#
# print(ds.PatientName)
# print(ds.PixelData)
# print(ds.pixel_array)
