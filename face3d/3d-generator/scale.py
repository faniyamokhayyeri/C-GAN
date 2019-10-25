import re
import os
import sys
import math
from zipfile import ZipFile
from shutil import rmtree, copyfile, copy2, move
import numpy as np
import random
from time import sleep

FILE_PATH = './out.obj'
NEW_FILE_PATH = './out.obj'
scale = 0.02

new_file_lines=[]
with open(FILE_PATH, 'r') as file:
    for line_terminated in file:
        line = line_terminated.rstrip('\n')
        temp = [x for x in line.split()]
        if temp[0] == 'v':
            v=([float(temp[1]), float(temp[2]), float(temp[3])])
            new_v = np.array(v)*scale
            new_file_lines.append('v '+str(new_v[0])+' '+ str(new_v[1])+' '+ str(new_v[2])+'\n')
        else:
            new_file_lines.append(line_terminated)

with open(NEW_FILE_PATH , 'w') as f:
    for line in new_file_lines:
        f.write("%s" % line)
