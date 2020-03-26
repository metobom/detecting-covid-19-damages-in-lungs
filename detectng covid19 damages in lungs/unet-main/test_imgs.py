from model import *
from data import *

testGene = testGenerator("data/test")
model = unet()
model.load_weights("lungs_for_corona.hdf5")
results = model.predict_generator(testGene,17,verbose=1)#put number of test img to '20'
saveResult("data/test",results)



