## Million Songs Data Set Analysis! 

This project depends on the Million Songs Dataset available at: https://labrosa.ee.columbia.edu/millionsong/

Initial experiments are done with the smaller, experimental subset provided at: https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset

Contributers: Aumit Leon, Mariana Echeverria

### Experiments and Results
To view our experiments and results, check out our wiki: https://github.com/AumitLeon/million-songs/wiki

### Directory Overview
Once you download the dataset, you'll notice that the file structure is as follows: 
```
    MillionSongSubset/
        AdditionalFiles/
            ...
        data/
            A/
                A/
                ...
                Z/
            B/
                A/
                ...
                I/
```
The data directory has subdirectories that act like volumes-- if you go deep enough you'll find the H5 files that correspond to each song. 

### Converting the data to a usable format
The data is given to us in HD5 format (https://support.hdfgroup.org/HDF5/whatishdf5.html).

HD5 files are binary files, so they are not very useful to us as they are given. In order to extract data from the h5 files, use `get_data.py`. 

The million song dataset provides python wrappers within `hd5_getters.py` that can be used to recursively loop through each subdirectory and h5 file to extract certain features of the data. 

`get_data.py` will visit every subdirectory (starting from the path you give `indir`), and will create a CSV of the data extracted from each h5 file. You don't need to put this script any place special, just be sure to provide it a proper path for `indir`. The `output.csv` file will be created in the same directory as this python script, so be sure not to commit that CSV file to Git :) 

As far as I can tell, each h5 file corresponds to one song. THat might not be true of every h5 file, maybe there is a way we can verify this?

