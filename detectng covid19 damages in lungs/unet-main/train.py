from model import *
from data import *


data_gen_args = dict()
#first param ('2') is batch number.
myGene = trainGenerator(2,'data/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('lungs_for_corona.hdf5', monitor='loss',verbose=1, save_best_only=True)
#specify your step and epoch 
model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])

'''
testGene = testGenerator("data/test")
model = unet()
model.load_weights('lungs_for_corona.hdf5')
results = model.predict_generator(testGene,68,verbose=1)# put your test image number to '68'
saveResult("data/test",results)
'''
