#!/usr/bin/env python
from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import numpy as np

# parameters for generating data and using splines
params={"starter_res" : 200,    # number of data points for a 0-20 K temperature range, used to set resolution for large data sets
        "T0" : 296.15,          # room temperature in Kelvin
        "T_test_max" : 1773.15, # temperature range over which new data was predicted across, Kelvin
        "min_points" : 2,       # min number of reference points
        "past_max_temp" : 1000} # how far past the max reference temperature predictions should go, Kelvin

# dictionary of material data
mat_dict = {}

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

# iterate through dictionary to show structure
def printDict(d,leading='...'):
    """ Print the dictionary object d
    
        d : Dictionary object
        leading : Characters to print before each entry to the screen
        
        Iterates through the given dictionary object and prints the field 
        names and values. If the object is a nested dictionary, then it 
        will continue to iterate.
        
        If the given object is not a dict, then nothing will happen.
    """
    # if the object is a dictionary
    if str(type(d))=="<class 'dict'>":
        # iterate through keys
        for key in d.keys():
            print(leading+key)
            printDict(d[key],leading=leading+leading)

def buildMaterialDict():
    """ Build the module material dictionary by reading in the config file

        Reads in config file mat_conf.ini and uses the field and section names
        to update the module dictionary variable mat_dict. The values are converted
        to lists using the json library.
        
        Materials must have at least one value in the following fields
            - k : Thermal conductivity reference values
            - k_t : Thermal conductivity reference temperature ranges
            - ev : Volume expansion factor reference values
            - ev_t : Volume expansion factor temperature ranges
            - c : Specific heat capacity reference values
            - c_t : Specific heat capacity temperature ranges 
            - p : Density reference values
            - p_t : Density temperature ranges

        If the material does not have the required fields, it is not included in the dictionary.
        
        This function can also be used to rebuild the dictionary.

        This function returns nothing
    """
    import configparser
    import json
    import os
    import errno
    # config is saved in the same directory as this file
    # generate absolute path to the config file
    config_path = os.path.join(os.path.dirname(__file__),'mat_conf.ini')
    # create parser object
    config = configparser.ConfigParser()
    # if config file does not exist
    if not os.path.isfile(config_path):
        raise FileNotFoundError(errno.ENOENT, 'LBAMMaterials : Cannot find material config file!','mat_conf.ini')
    # attempt to interpret config, if failed raise value error bc something is wrong with the file
    if not 'mat_conf.ini' in config.read(config_path)[0]:
        raise ValueError('LBAMMaterials : Failed to read mat_conf.ini')
    # clear local dictionary
    mat_dict.clear()
    # required fields
    req_fields = ['k','k_t','ev','ev_t','c','c_t','p','p_t','tm','e']
    # for each material found in config
    for mat in config:
        # if the name is not DEFAULT
        # default contains all the default values
        if mat != "DEFAULT":
            # for each material, create a empty dictionary
            mat_dict[mat] = {}
            # iterate through each value and convert to actual values
            for k in config[mat]:
                # use json library to read values and convert them to numbers
                mat_dict[mat][k] = json.loads(config.get(mat,k))
            # check that the material dictionary entry contains the required fields
            # if not, delete the material from internal dictionary
            if not all(field in mat_dict[mat].keys() for field in req_fields):
                print("Material {0} does not have the requisite fields! Removing from mat_dict!".format(mat))
                del mat_dict[mat]

# build the material dictionary on import
buildMaterialDict()

def getMaterialData(material="SS-316L"):
    """ Get the material data from the module config file for specified material

        material : Requested material

        Accesses the module dictionary mat_dict and retrieve the data stored in the config file.
        Constructs continous matricies of values for each material parameter. It then
        packs the values as tuples of the values and associated temperature range.

        The resolution of the generated matricies is set in params under variable "starter_res".
        i.e. a temperature range of 0-100 K with a resolution of 100 will generate a matrix of 
        (100-0) * 100 = 10000 values

        Order of returned matricies:
            - Thermal conductivity
            - Volume Expansion factor
            - Specific Heat Capacity
            - Density
            - Thermal Diffusivity

        If there isn't a value for the specific parameter, a blank list is added i.e. ([],[])

        If the material is not support, a blank list is returned.
        
        Returns a list of tuples containing a matrix of parameter values and the associated temperature range.
    """
    fields = ['k','ev','c','p','d','tm','e']
    data = []
    if material in mat_dict.keys():
        # search for value
        for f in fields:
            #print("Searching for {0} and temp matrix {1}".format(f,f+'_t'))
            vals = mat_dict[material].get(f)
            # if there's a value
            if vals != None:
                # if the target field is not melting point or emissivity
                # build matricies
                if not f in ['tm','e']:
                    #print("Found ",f)
                    mats = []
                    # get associated temperature values
                    temp = mat_dict[material].get(f+'_t')
                    # for each of the values in the list 
                    for cv,v in enumerate(vals):
                        # construct values matrix and add to list
                        mats.append(np.full(int(params["starter_res"]*(temp[cv][1]-temp[cv][0])/20),v,dtype='float'))
                        #print("mats length ",len(mats))
                    # concatenate the matricies together
                    full_mat = np.concatenate(mats,axis=0)
                    # construct temperature matrix
                    temp_mat = np.linspace(temp[0][0], # min temperature value
                                           temp[-1][1],# max temperature value
                                           num=full_mat.shape[0]) # ensure it's the same shape as the values
                    #print("Temperature matrix shape: ",temp_mat.shape)
                    # add as tuples to list
                    data.append((full_mat,temp_mat))
                # if the target field is melting point or emissivity, then append the values
                else:
                    data.append(mat_dict[material].get(f))
            # if there is no value, append blank values
            # current optional fields are value, temperature pairs
            else:
                data.append(([],[]))
        # return completed matrix
        return data
    else:
        return []

# functions for converting temperature between celcius and kelvin
def CelciusToK(C):
    """ Function for converting Celcius to Kelvin

        C : Temperature in Celcius

        Returns temperature in Kelvin
    """
    return float(C)+273.15

def KelvinToC(K):
    """ Function for converting Kelvin to Celcius

        K : Temperature in Kelvin

        Returns temperature in Celcius
    """
    return float(K)-273.15

def Kp(Ks,T,Tm,scale=0.01):
    """ Thermal conductivity of powder from solid material
        
        Ks : Thermal condutivity of solid material, numpy array
        T  : Temperature data to genereate Ks, K
        Tm : Melting point of material, K
        scale : Value to scale thermal conductivity at temperatures above
                melting point Tm

        Taken to be fraction of solid thermal conductivity when
        temperature is less than the melting point, Tm.

        Source : 3-Dimensional heat transfer modeling for laser 
        powder-bed fusion additive manufacturing with volumetric 
        heat sources based on varied thermal conductivity and 
        absorptivity (2019), Z. Zhang et. al.

        Returns thermal conductivity for powder as approximated
    """

    from numpy import where,asarray,copy
    # make a copy of solid thermal conductivity matrix
    Kp = np.copy(Ks)
    # for all values where the temperature is less than the melting point
    # scale the thermal conductivity values
    Kp[where(T<Tm)] *= scale
    # return modified matrix
    return Kp
    
def genDensityData(Ev,p_ref,ref_temp_range,max_temp=1273.15):
    """ Generate density data based of thermal volumetric expansion data for specified material
        
        Ev : spline for thermal volumetric expansion
        p_ref : Reference density data
        ref_temp : Reference temperature data, Kelvin
        max_temp : Max temperature to predict up to, Kelvin

        Uses a combination of the spline for the volumetric density and some known data points to construct
        density data.

        Returns the predicted density data data and the temperature values they are valid for.
        If an unsupported material is specified, then an error message is printed to screen and None is returned
    """
    starter_res = params["starter_res"]
    if len(p_ref==0):
        return None,None
    ## calculate density data for dT =[100 - 1000] C using Ev spline and prev range value as starter point
    p_range_2 = []
    # calculate temperature range for reference data
    temp_range = max_temp-ref_temp[1]
    # for the diffeence between the end of the reference temperature range and max_temp
    # calculate density using the volumetric expansion spline and the reference value
    # density is based on change in temperature
    for dT in np.linspace(0.0,temp_range,int(starter_res*temp_range)):
        p_range_2.append(p_ref[0]/(1+Ev(dT)*dT))
    # convert to array
    p_range_2 = np.array(p_range_2)
    # combine data ranges together
    p_data = np.concatenate([d_ref,p_range_2],axis=0)
    # create temperature data
    p_temp_range = np.linspace(ref_temp_range[1],max_temp,p_data.shape[0])
    return p_data, p_temp_range

def genThermalDiff(K,p,C,T):
    """ Generate thermal diffusivity data for solid material using previous splines

        K : thermal conductivity spline for solid material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix

        Returns thermal diffusivity data for the temperature range
    """
    return K(T)/(p(T)*C(T))

def genThermalDiff_powder(K,p,C,T,Tm):
    """ Generate thermal diffusivity data for powder using previous splines

        K : thermal conductivity spline for powder material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix
        Tm : metling temperature of the solid material

        returns thermal diffusivity data for the temperature range
    """
    # generate thermal conductivity data
    # then modifiy it according to approximation function
    return Kp(K(T),T,Tm)/(p(T)*C(T))

def buildMaterialData(material="SS-316L"):
    """ Function to create and build the material matricies and parameters used in predicting temperature and power density

        material : Specified material. Case sensitive. Default: SS-316L
                    Currently supported materials:
                        - Stainless Steel : SS-316L

        If an unsupported material is specified, then None is returned.

        Returns emissivity, univariate spline function for thermal conductivity and interpolated spline for thermal diffusivity
    """
    if material in mat_dict.keys():
        # get material data and parse to outputs
        [(K_data,K_temp),(Ev_data,Ev_temp),(C_data,C_temp),(p_data,p_temp),(D_data,D_temp),Tm,e] = getMaterialData()
        
        # get room temperature
        T0 = params["T0"]
        # temperature data to predict thermal diffusivity across, [0 - T_test_max] K
        T_test_data = np.linspace(0,params["T_test_max"],int(params["starter_res"]*params["T_test_max"]/20))
        
        ## Generate spline objects
        # thermal conductivity
        # fit spline to solid data
        K_ispline = InterpolatedUnivariateSpline(K_temp,K_data)
        # convert to thermal conductivty for powder
        Kp_data = Kp(K_data,K_temp,Tm)
        # fit spline to powder data
        Kp_spline = UnivariateSpline(K_temp,Kp_data)
        
        # Specific heat capacity
        C_ispline = InterpolatedUnivariateSpline(C_temp,C_data)
        
        # volume expansion factor
        Ev_ispline = InterpolatedUnivariateSpline(Ev_temp,Ev_data)
        
        ## density
        # if there's only one data point, use the volumetric expansion data to predict up to
        # a target temperature
        if len(p_data)<params["min_points"]:
            # set target max temperature as past_max_temp degrees K past the max temperature
            p_data,p_temp_range = genDensityData(Ev_ispline,p_data,p_temp,max_temp=p_temp.max()+params["past_max_temp"])
        # fit spline to data
        p_ispline = InterpolatedUnivariateSpline(p_temp,p_data)
        
        # thermal diffusivity of the solid material
        Ds_data = genThermalDiff(K_ispline,p_ispline,C_ispline,T_test_data)
        Ds_ispline = InterpolatedUnivariateSpline(T_test_data,Ds_data)
        
        return e,Kp_spline,Ds_ispline
    else:
        return None
