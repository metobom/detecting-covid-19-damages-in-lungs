
from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20,'data/train','image','label',data_gen_args,save_to_dir = "data/train/aug")

num_batch = 3
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break

image_arr,mask_arr = geneTrainNpy("data/train/aug/","data/train/aug/")

