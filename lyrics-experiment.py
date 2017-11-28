import numpy as np 
import csv



def read_txt(filename):
    """ 
    Reads txt file and returns tuple with
    list of top 5,000 words and 
    the index : frequency for each track
    """
    content = [] # list with word index : word count for each track
    string = '%'
    find = False 
    words = [] 
    track_id = [] # list with track ID's from the MSD
    mxm_tid = [] # track ID's from musiXmatch
    str_data = []

    read_file = open(filename, "r")
    
    for line in read_file:
        if find:
            # obtaining data 
            list_line = line.strip().split(',')
            track_id.append(list_line[0])
            mxm_tid.append(list_line[1])
            data = list_line[2:]
            dictionary = {}

            for el in data:
                index = el.find(':')
                key = el[:index]
                val = el[index+1:]
                dictionary[int(key)] = int(val)
            content.append(dictionary)

        else:
            # obtaining line with 5,000 words 
            if line.startswith(string):
                line = line[1:]
                words = [word.strip() for word in line.split(',')]
                find = True
    read_file.close() 
    

    return (words, content)


def write_data(filename):
   

    with open(fieldname, 'w') as csvfile:
        fieldnames = [i for i in range(1,5001)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})




if __name__ == "__main__":
    #(track_id, words, mxm_tid, data) = extract_data("mxm_dataset_test.csv")
    #print(data)
    #print(track_id)
    #print(mxm_tid)
    #print(words)

    (words, content) = read_txt('mxm_dataset_train.txt')
    print(len(content))
    print(content[0])
    