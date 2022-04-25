import scipy.io as sio

from mask import cs_mask


mask = cs_mask([120, 256, 256], acc=8.0, sample_n=8)
mask = mask.transpose(1, 2, 0)


sio.savemat('Patient25_Part1_Stage3_mFFE_heating_mask.mat', {'mask': mask})




