# detecting-covid-19-damages-in-lungs
To find damage in lungs caused by Covid-19 I trained UNet (by zhixuhao) for lungs and consolidations seperately. Consolidations are fusion of infections which can be observed in lung CTs. Since UNet returns white detection and black background, I counted white pixels for consolidations and lungs and calculated % of consolidations for lung.

# Explanation
corona.hdf5 weights of consolidation detection and lungs_for_corona is weights for lung detection. train.py, model.py, main.py, test_imgs, data.py and dataPrepare.py files are about UNet. predict_and_calculate_score is the one file that calculates % of consolidations. You can run predict_and_calculate_score to test images inside of test_inputs folder.

NOTE: Weights are trained for 2000 steps and 2 epoch. There are errors due to poor annotation, less training and stuff. 

Read this to have some more idea about Covid-19's effects on lungs: https://pubs.rsna.org/doi/pdf/10.1148/radiol.2020200370

zhixuhao's repo for UNet: https://github.com/zhixuhao/unet
