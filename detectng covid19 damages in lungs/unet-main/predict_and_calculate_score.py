from model import *
from data import *
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt

model = unet()

def findPixelArea(outs):
    img = outs.reshape(256,256)
    cntr=0
    for i in range(256):
        for j in range(256):
            if img[i,j]>0.2:
                cntr+=1
    return cntr

def calculatePercent(num, total):
    return (num/total)*100
    


#CONSOLIDATION
testGeneConsolidation = testGenerator("test_inputs/consolidations")
model.load_weights("corona.hdf5")
consolidation_outs = model.predict_generator(testGeneConsolidation,2,verbose=1)
#saveResult("data/test_outs/consolidations",consolidation_outs)

cnsl_holder = []
for i in consolidation_outs:
    cnsl_pix = findPixelArea(i)
    #print('consolidation pixel: {}'.format(cnsl_pix))
    cnsl_holder.append(cnsl_pix)

#LUNGS
testGeneLungs = testGenerator("test_inputs/lungs")
model.load_weights("lungs_for_corona.hdf5")
lung_outs = model.predict_generator(testGeneLungs,2,verbose=1)
#saveResult("data/test_outs/consolidations",lung_outs)

lung_holder = []
for j in lung_outs:
    lung_pix = findPixelArea(j)
    #print('lung pixel: {}'.format(lung_pix))
    lung_holder.append(lung_pix)

print(cnsl_holder)
print(lung_holder)

for a in range(len(lung_holder)):
    print('%{}'.format(calculatePercent(cnsl_holder[a], lung_holder[a])))
















    


