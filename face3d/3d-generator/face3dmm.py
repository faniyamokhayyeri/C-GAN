import os
import shutil
import time

if not os.path.isdir("./landmarks/"):
    os.mkdir("./landmarks/")
if not os.path.isdir("./3d-objs/"):
    os.mkdir("./3d-objs/")

input_dirs = os.listdir('./inputs')
for dir in input_dirs:
    if os.path.isdir('./inputs/'+dir):
        files = os.listdir('./inputs/'+dir+'/')
        for file in files:
            if os.path.isfile('./inputs/'+dir+'/'+file):
                print("Calculating landmarks of the image:"+ "./inputs/" + dir + '/' + file)
                if not os.path.isdir("./landmarks/" + dir):
                    os.mkdir("./landmarks/" + dir )
                os.system("python ./3d-generator/generator/landmark_detector.py --shape-predictor ./3d-generator/generator/shape_predictor_68_face_landmarks.dat \
                        --image ./inputs/" + dir + '/' + file +\
                        " -o " + "./landmarks/" + dir )
                print("Fitting a 3d morfable model to the image:"+ "./inputs/" + dir + '/' + file)
                if not os.path.isdir("./3d-objs/" + dir):
                    os.mkdir("./3d-objs/" + dir )
                os.system("./3d-generator/generator/eos/bin/fit-model-simple -m ./3d-generator/generator/eos/share/sfm_shape_3448.bin -p ./3d-generator/generator/eos/share/ibug_to_sfm.txt " 
                        + "-i ./inputs/" + dir + '/' + file 
                        + " -l ./landmarks/" + dir + '/' + file[:-4] + ".pts")
                print("Scaling ...")
                os.system("python ./3d-generator/scale.py")
                print("Calculating mesh normals")
                #print("meshlabserver -i out.obj -o " + file[:-4] + ".obj " + "-mo vt vn fc -s ./3d-generator/generator/filter.mlx")
                os.system("meshlabserver -i out.obj -o " + file[:-4] + ".obj " + "-s ./3d-generator/generator/filter.mlx -om vt vn fc")
                shutil.move("./" + file[:-4] + ".obj", "./3d-objs/"+dir+"/"+ file[:-4] + ".obj")
                os.remove("./" + file[:-4] + ".obj.mtl")
                shutil.move("./out.isomap.png", "./3d-objs/"+dir+"/"+ file[:-4] + ".isomap.png")
            
                with open("./3d-objs/"+dir+"/"+ file[:-4] + ".obj.mtl" , 'w') as f:
                    f.write("newmtl material_0\n")
                    f.write("Ns 500\n")
                    f.write("Ka 1.000000 1.000000 1.000000\n")
                    f.write("Kd 1.000000 1.000000 1.000000\n")
                    f.write("Ks 1.000000 1.000000 1.000000\n")
                    f.write("Ni 1.000000\n")
                    f.write("d 1.000000\n")
                    f.write("illum 5\n")
                    f.write("map_Kd " + file[:-4] + ".isomap.png\n")
                print('--------------------------')

os.remove('out.obj')
os.remove('out.mtl')