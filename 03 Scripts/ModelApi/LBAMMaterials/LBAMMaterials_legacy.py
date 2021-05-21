from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import numpy as np

# parameters for generating
params={"starter_res" : 200,
        "res_scale" : 2}

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

def genKdata(material="SS-316L"):
    """ Construct matrix of thermal conductivity values from raw values and the temperature values it is
        valid for.

        material : Specified material. Case sensitive. Default: SS-316L
                   Currently supported:
                        - Stainless steel 316L : SS-316L

        Returns the thermal conductivity vector and the associated temperature values it is valid for.
        If an unsupported material is returned, a message is printed and None is returned
    """
    starter_res = params["starter_res"]
    if material=="SS-316L":
        # temp range [0-20] avg
        K_range_1 = np.full(starter_res,16.3)
        # temp range [20-100]
        K_range_2 = np.full(int(starter_res*(80/20)),(16.3+14.6)/2)
        # temp range [100-500]
        K_range_3 = np.full(int(starter_res*(400/20)),21.5)
        # temp data, complete
        K_data = np.concatenate([K_range_1,K_range_2,K_range_3],axis=0)
        # temperature range for plotting data
        K_temp_range = np.linspace(0.0,CelciusToK(500.0),K_data.shape[0])
        return K_data,K_temp_range
    else:
        print(material," : Unsupported material")
        return None
        

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

## specific heat capacity
def genCdata(material="SS-316L"):
    """ Generates specific heat capacity data for the specified material

        material : Specified material. Case sensitive. Default: SS-316L
                   Currently supported materials:
                       - Stainless steel 316L : SS-316L

        Returns the specific heat capacity data constructed for the specified material
        and the temperature range these values are valid across.
        If an unsupported material is supplied, then a message is printed and None is returned.
    """
    starter_res = params["starter_res"]
    if material=="SS-316L":
        # temp range [0-20]
        C_range_1 = np.full(starter_res,450.0)
        # temp range [20-93]
        C_range_2 = np.full(int(starter_res*(73/20)),(485.0+500.0)/2.0)
        # temp range [93-100]
        C_range_3 = np.full(int(starter_res*(7/20)),500.0)
        # specific heat capacity, complete
        C_data = np.concatenate([C_range_1,C_range_2,C_range_3],axis=0)
        # temperature data for plotting data
        C_temp_range = np.linspace(0,CelciusToK(100),C_data.shape[0])
        return C_data,C_temp_range
    else:
        print(material," : Unsupported material")
        return None

# volume expansion factor, mm^3/mm^3.C
# same as m^3/m^3.K
# as both C and K are absolute units of measurement, a change is C is the same in K
# converting between mm to m cancels each other out
def genEvData(material="SS-316L"):
    """ Generate data for volume expansion factor based off linear expansion factors for
        the specified material

        material : Specified material. Case sensitive. Default: SS-316L
                    Currently supported materials:
                        - Stainless steel 316L : SS-316L

        Using linear expansion factor data, volume expansion data is generated assuming uniform
        cubic expansion.

        Returns the volume expansion factor data for the specified material and the temperature
        range these values are valid across.
        If an unsupported material is supplied, then a message is printed and None is returned.
    """
    starter_res = params["starter_res"]
    if material=="SS-316L":
        # for T = [20-100]
        Ev_range_1 = np.full(int(starter_res*(80/20)),3.0*(16.6+18.2+19.4)/3.0)
        # for T = [100-500]
        Ev_range_2 = np.full(int(starter_res*(400/20)),3.0*(18.2+19.4)/2.0)
        # for T = [500-1000]
        Ev_range_3 = np.full(int(starter_res*(500/20)),3.0*19.4)
        # combine data together
        Ev_data = np.concatenate([Ev_range_1,Ev_range_2,Ev_range_3],axis=0)
        # temperature data for temperature range so it can be plotted
        Ev_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),Ev_data.shape[0])
        # return data and temp data
        return Ev_data,Ev_temp_range
    else:
        print(material," : Unsupported material")
        return None
    
def genDensityData(Ev,material="SS-316L"):
    """ Generate density data based of thermal volumetric expansion data for specified material
        
        Ev : spline for thermal volumetric expansion
        material : Specified material. Case sensitive. Default: SS-316L
                    Currently supported materials:
                        - Stainless steel 316L : SS-316L

        Uses a combination of the spline for the volumetric density and some known data points to construct
        density data.

        Returns the predicted density data data and the temperature values they are valid for.
        If an unsupported material is specified, then an error message is printed to screen and None is returned
    """
    starter_res = params["starter_res"]
    if material=="SS-316L":
        # for dT = [20 - 100] C
        p_range_1 = np.full(int(starter_res*(80/20)),(7900.0+8000.0+8027.0)/3.0)
        ## calculate density data for dT =[100 - 1000] C using Ev spline and prev range value as starter point
        p_range_2 = []
        for dT in np.linspace(0.0,900.0,int(starter_res*(20/900))):
            p_range_2.append(p_range_1[0]/(1+Ev(dT)*dT))
        # convert to array
        p_range_2 = np.array(p_range_2)
        # combine data ranges together
        p_data = np.concatenate([p_range_1,p_range_2],axis=0)
        # create temperature data
        p_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),p_data.shape[0])
        return p_data, p_temp_range
    else:
        print(material," : Unsupported material!")
        return None

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
    if(material=="SS-316L"):
        # starter temperature, room temperature
        T0 = CelciusToK(23.0)

        # melting temperature of 316L
        Tm = 1390 # C
        Tm_K = CelciusToK(Tm) # K

        ### ALL TEMPS IN C
        #temperature resolution for [0-20] C or [0 - 293.15] K 
        starter_res = params["starter_res"]
        
        K_data,K_temp_range = genKdata()
        # temperature range to test fitting on
        T_test_data = np.linspace(0,CelciusToK(1500),params["res_scale"]*(1500/20))

        ### interpolate data
        ## thermal conductivity
        # returns a function that can be used later
        # s parameter is the min square distance between interpolation and data
        K_spline = UnivariateSpline(K_temp_range,K_data)
        K_ispline = InterpolatedUnivariateSpline(K_temp_range,K_data)
        
        Kp_data = Kp(K_data,K_temp_range,Tm_K)
        Kp_spline = UnivariateSpline(K_temp_range,Kp_data)
        Kp_ispline = InterpolatedUnivariateSpline(K_temp_range,Kp_data)

        ## specific heat capacity
        C_data,C_temp_range = genCdata(material)

        C_spline = UnivariateSpline(C_temp_range,C_data)
        C_ispline = InterpolatedUnivariateSpline(C_temp_range,C_data)

        ## volumetric expansion
        Ev_data, Ev_temp_range = genEvData(material)

        Ev_spline = UnivariateSpline(Ev_temp_range,Ev_data)
        Ev_ispline = InterpolatedUnivariateSpline(Ev_temp_range,Ev_data)

        ## Density
        p_data,p_temp_range = genDensityData(Ev_ispline)

        p_spline = UnivariateSpline(p_temp_range,p_data)
        p_ispline = InterpolatedUnivariateSpline(p_temp_range,p_data)

        # thermal diffusivity of the solid material
        Ds_data = genThermalDiff(K_ispline,p_ispline,C_ispline,T_test_data)

        Ds_spline = UnivariateSpline(T_test_data,Ds_data)
        Ds_ispline = InterpolatedUnivariateSpline(T_test_data,Ds_data)

        # thermal diffusivity using thermal conductivity scaling approximation
        Dp_data = genThermalDiff_powder(Kp_ispline,p_ispline,C_ispline,T_test_data,Tm)

        Dp_spline = UnivariateSpline(T_test_data,Dp_data)
        Dp_ispline = InterpolatedUnivariateSpline(T_test_data,Dp_data)
        # tests have shown that the model best performs with the univariate spline for
        # thermal conductivity powder version and the univariate spline of thermal diffusivity
        # solid version
        return 0.53,Kp_spline,Ds_ispline
    else:
        return None
