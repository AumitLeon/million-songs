# Convert HD5 files to CSV

import hdf5_getters
import os
import csv
import string

# Debugging
#h5 = hdf5_getters.open_h5_file_read("/mnt/c/Users/Aumit/Desktop/final-proj/MillionSongSubset/data/A/A/A/TRAAADZ128F9348C2E.h5")
#title = hdf5_getters.get_artist_name(h5)
#h5.close()
#hello
# HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#print title

# Provide the starting root dir 
indir = '/mnt/c/Users/Aumit/Desktop/final-proj/MillionSongSubset/data/B/'

# Open the CSV file we will write to
with open("output.csv", 'wb') as csvfile:
    # Column headers
    # Currently only 4 features being extracted
    # We can add as many as we want, just seperate with commas
    csvfile.write("artist_name, title, artist_location, release")
    csvfile.write("\n")

    # Recursively visit each sub-dir till we reach the h5 files
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            #log = open(os.path.join(root, f),'r')
            #print os.path.join(root, f)

            # Use the hd5 wrappers to open the h5 file
            h5 = hdf5_getters.open_h5_file_read(os.path.join(root, f))

            # EXTRACTING FEATURES 
            # See hd5_getter.py for the various features that we can extract from each h5 file

            # Get the artist name
            artist_name = hdf5_getters.get_artist_name(h5)
            artist = artist_name.translate(None, string.punctuation)
            

            # Get the title of the song
            title_song = hdf5_getters.get_title(h5)
            title = title_song.translate(None, string.punctuation)

            # Get artist location
            artist_location = hdf5_getters.get_artist_location(h5)
            artist_loc = artist_location.translate(None, string.punctuation)

            # Get release
            release_song = hdf5_getters.get_release(h5)
            release = release_song.translate(None, string.punctuation)

            # Close the h5 file
            h5.close()

            # Write to the CSV file
            csvfile.write(artist + "," + title + "," + artist_loc + "," + release)
            csvfile.write("\n")

            # Print the current song and arists: 
            print title + " by " + artist_name
            # Move on to the next h5 file
            print
            print
