import os
import shutil

dirclist = os.listdir('./output/')
print(dirclist)
filecount = 0
for dirc in dirclist:
    if os.path.isdir('./output/' + dirc):
        filelist = os.listdir('./output/' + dirc)
        print(filelist)
        for f in filelist:
            filecount += 1
            if os.path.isfile('./output/' + dirc + '/' + f):
                shutil.move('./output/' + dirc + '/' + f, './output/' + str(filecount) + '_' + dirc + '.png')
        shutil.rmtree('./output/' + dirc)
