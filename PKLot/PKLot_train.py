from PKLot import *

BATCH_SIZE = 50
data_folder = 'F:\datasets\PKLot\PKLotSegmented'
tfrecord_name = 'PKLot_segmented'
tfrecord_path = 'F:\python\ML\PKLot'

image_list, label_list = get_file(data_folder)
convert_to_tfrecord(image_list,label_list,tfrecord_path,tfrecord_name)

