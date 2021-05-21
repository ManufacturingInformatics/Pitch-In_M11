import configparser
import json
import numpy as np
import errno
import os

# create parser class
config = configparser.ConfigParser()
# read in file, reads in as an ordered dictionary
config.read('mat_conf.ini')

## convert file to dictionary of values
mat_dict = {}
for key in config:
    # default contains all the default values
    if key != "DEFAULT":
        # for each key, create a empty dictionary
        mat_dict[key] = {}
        # iterate through each value and convert to actual values
        for k in config[key]:
            # use json library to read values and convert them to numbers
            mat_dict[key][k] = json.loads(config.get(key,k))

def buildMaterialDict():
    """ Build the module material dictionary by reading in the config file

        Reads in config file mat_conf.ini and uses the field and section names
        to update the module dictionary variable mat_dict. The values are converted
        to lists using the json library.

        This function can also be used to rebuild the dictionary.

        This function returns nothing
    """
    import configparser
    import json
    # create parser object
    config = configparser.ConfigParser()
    # if config file does not exist
    if os.path.isfile('mat_conf.ini'):
        raise FileNotFoundError(errno.ENOENT, 'LBAMMaterials : Cannot find material config file!','mat_conf.ini')
    # attempt to interpret config
    if 'mat_conf.ini' in config.read('mat_conf.ini'):
        raise ValueError('LBAMMaterials : Failed to read mat_conf.ini')
    # clear local dictionary
    mat_dict.clear()
    # for each section found in config
    for key in config:
        # if the name is not DEFAULT
        # default contains all the default values
        if key != "DEFAULT":
            # for each key, create a empty dictionary
            mat_dict[key] = {}
            # iterate through each value and convert to actual values
            for k in config[key]:
                # use json library to read values and convert them to numbers
                mat_dict[key][k] = json.loads(config.get(key,k))

# iterate through dictionary to show structure
def printDict(d,leading='...'):
    # if the object is a dictionary
    if str(type(d))=="<class 'dict'>":
        # iterate through keys
        for key in d.keys():
            print('..',leading,key)
            printDict(d[key],leading=leading+leading)
            
# print structure
printDict(mat_dict)

# iterate through possible fields searching for values
fields = ['k','k_t','ev','ev_t','c','c_t','tm','e','p','p_t','d','d_t']
for key in config:
    if key != "DEFAULT":
        print(key)
        for f in fields:
            # read in value, if it doesn't exist return Noone
            val = config[key].get(f)
            # if a value was returned
            if val != None:
                # print field and value
                print(f,json.loads(val))

# read in values and construct matricies

starter_res = 100
res_scale = 2

fields = ['k','ev','c','p','d']
# search for value
for f in fields:
    vals = mat_dict["SS-316L"].get(f)
    # if there's a value
    if vals != None:
        mats = []
        # get associated temperature values
        temp = mat_dict["SS-316L"].get(f+'_t')
        # for each of the values in the list 
        for cv,v in enumerate(vals):
            # construct values matrix and add to list
            mats.append(np.full(starter_res*(temp[cv][1]-temp[cv][0]),v,dtype='float'))
        # concatenate the matricies together
        full_mat = np.concatenate(mats,axis=0)
        # construct temperature matrix
        temp_mat = np.linspace(temp[0][0], # min temperature value
                               temp[-1][1],# max temperature value
                               num=full_mat.shape[0]) # ensure it's the same shape as the values

        print(f," : ",temp_mat.shape)

def genKData(material="SS-316L"):
    """ Generate thermal conductivity matricies by reading data from global dictionary

        material : Material to search for

        Searches the dictionary for the thermal conductivity values and associated
        temperature ranges. It then converts the values into two matricies of equal length.

        Returns the thermal conductivity values and associated temperature range
        
    """
    vals = mat_dict[material].get('k')
    # if there's a value
    if vals != None:
        mats = []
        # get associated temperature values
        temp = mat_dict["SS-316L"].get(f+'_t')
        # for each of the values in the list 
        for cv,v in enumerate(vals):
            # construct values matrix and add to list
            mats.append(np.full(starter_res*(temp[cv][1]-temp[cv][0]),v,dtype='float'))
        # concatenate the matricies together
        full_mat = np.concatenate(mats,axis=0)
        # construct temperature matrix
        temp_mat = np.linspace(temp[0][0], # min temperature value
                               temp[-1][1],# max temperature value
                               num=full_mat.shape[0]) # ensure it's the same shape as the values
        return full_mat,temp_mat
    return None

def getMaterialData(material="SS-316L"):
    """ Get the material data from the module config file for specified material

        material : Requested material

        Access the module dictionary and retrieve the data stored in the config file.
        Constructs continous matricies of values for each material parameter. It then
        packs the values as tuples of the values and associated temperature range.

        The resolution is set in params under variable "starter_res".

        Order of returned matricies:
            - Thermal conductivity
            - Volume Expansion factor
            - Specific Heat Capacity
            - Density
            - Thermal Diffusivity

        Added at the end of the list is the following single values:
            - Melting point
            - Emissivity

        If there isn't a value for the specific parameter, a blank list is added i.e. ([],[])

        If the material is not support, a blank list is returned.
        
        Returns a list of tuples containing a matrix of parameter values and the associated temperature range.
    """
    fields = ['k','ev','c','p','d']
    data = []
    if material in mat_dict.keys():
        # search for value
        for f in fields:
            vals = mat_dict[material].get(f)
            # if there's a value
            if vals != None:
                mats = []
                # get associated temperature values
                temp = mat_dict[material].get(f+'_t')
                # for each of the values in the list 
                for cv,v in enumerate(vals):
                    # construct values matrix and add to list
                    mats.append(np.full(starter_res*(temp[cv][1]-temp[cv][0]),v,dtype='float'))
                # concatenate the matricies together
                full_mat = np.concatenate(mats,axis=0)
                # construct temperature matrix
                temp_mat = np.linspace(temp[0][0], # min temperature value
                                       temp[-1][1],# max temperature value
                                       num=full_mat.shape[0]) # ensure it's the same shape as the values

                # add as tuples to list
                data.append((full_mat,temp_mat))
            else:
                data.append(([],[]))
        # get the melting point and emissivity
        data.append(mat_dict[material].get('tm'))
        data.append(mat_dict[material].get('e'))
        # return result
        return data
    else:
        print("Unsupported material")
        return []
            
mat_data = getMaterialData()
