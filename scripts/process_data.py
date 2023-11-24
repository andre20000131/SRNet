import os
import shutil
path = '/data/FFLQ_data'
num = 0
for i in os.listdir(path):
    if int(i[:5])<5000:
        print(i)
        print(num)
        num+=1
        shutil.move('/data/FFLQ_data/'+i,'/data/5000_LR')