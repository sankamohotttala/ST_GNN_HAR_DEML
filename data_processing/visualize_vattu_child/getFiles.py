#get all the files named in the text file; not used at the moment
import os
from shutil import copy as cp
import random 
import copy

file_path=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\Processed skeleton files\ntu 120\10 classes\visualization\VisualizeFileText\visualize_files.txt'
destFolder=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\Processed skeleton files\ntu 120\10 classes\visualization\test_files'
#NTU 60
sourceFolder=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\Processed skeleton files\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons'
#NTU 120
#sourceFolder=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\Processed skeleton files\ntu 120\nturgbd_skeletons_s018_to_s032'

with open(file_path,'r') as g:
    all_files=[line.strip() for line in g.readlines()]

for file in all_files:
    if not os.path.exists(destFolder):
        os.makedirs(destFolder)
    cp(sourceFolder+'/'+file, destFolder)
