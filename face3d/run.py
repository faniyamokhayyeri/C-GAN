import os

#os.system('python ./3d-generator/face3dmm.py')

for n in range(1,31):
    if n is not 6:
       os.system('python ./3d-generator/render.py ./3d-objs/' + str(n)+ '/ 200')
