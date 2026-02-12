from glob import glob
import sys
import shutil

thisfilename = sys.argv[0].split('/')[-1]
dirlist = [dir[2:] for dir in glob('./*') if dir[2:] != thisfilename]
for index, patientname in enumerate(dirlist):
    newdirname = '../id_selectedpoints/Patient{:02d}'.format(index + 1)
    shutil.copytree(patientname, newdirname)
