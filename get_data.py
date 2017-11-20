# Convert HD5 files to CSV

import hdf5_getters
import os
import csv

#h5 = hdf5_getters.open_h5_file_read("/mnt/c/Users/Aumit/Desktop/final-proj/MillionSongSubset/data/A/A/A/TRAAADZ128F9348C2E.h5")
#title = hdf5_getters.get_artist_name(h5)
#h5.close()
#print title

# Provide the starting root dir
indir = '/mnt/c/Users/Aumit/Desktop/final-proj/MillionSongSubset/data/B/'
with open("output.csv", 'wb') as csvfile:
    csvfile.write("artist_name, title")
    csvfile.write("\n")
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            #log = open(os.path.join(root, f),'r')
            #print os.path.join(root, f)
            h5 = hdf5_getters.open_h5_file_read(os.path.join(root, f))
            artist_name = hdf5_getters.get_artist_name(h5)
            title = hdf5_getters.get_title(h5)
            h5.close()
            csvfile.write(artist_name + "," + title)
            csvfile.write("\n")
        # print f
            print
            print
