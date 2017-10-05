def path(path = None):
    """
    Customized data extracting function for *.csv.

    id[index] = [AMB_TEMP, CH4, ..., WIND_SPEED, WS_HR]

    Args:
        path:
            A string of data file path.
   
    Returns:
        data_sets:
            A list of data read from `path`.

    Raises:
        ValueError:
            If `path` is not set.
        FileNotFoundError:
            If file `path` not found.
    """
    
    # raise error if path not set
    if path is None:
        raise ValueError('Arguemnt `path`  is not set.')

    # open file
    data_file = open(path,'r')
        
    # data container
    data_sets = []

    # read line until EOF
    while True:    
        # temporary data container for formating
        temp = []

        # format each line, 18 is a round
        for id_x in range(0, 18):
        
            # read one line
            line = data_file.readline()
            

            # strip '\n' and split by ','
            parts = line.rstrip().split(',')

            # discard first three term and temporarily store it
            temp.append(parts[2:])
            
        # reach end of file
        if line == "":
            break

        # format each id, make a single data set by hour 
        for hour in range(0, 9):
            single_data = []

            for data_type in range(0, 18):
                if temp[data_type][hour] == 'NR':
                    single_data.append(0)
                else:
                    single_data.append(float(temp[data_type][hour]))

            # add single data set to data container
            data_sets.append(single_data)

    # close file
    data_file.close()

    # answer data container
    ans_sets = []

    # get answer set for each data set
    for data in data_sets:
        ans_sets.append(data[9])
        del data[9]

    return data_sets, ans_sets
