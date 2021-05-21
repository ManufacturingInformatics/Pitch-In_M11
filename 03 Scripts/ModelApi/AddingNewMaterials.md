# Adding Support for New Materials

Information about the materials is stored under a config files called [mat_conf](../LBAMMaterials/maat_conf.ini). It is a collection of reference points for certain material parameters and the temperature ranges they are valid across. The file is designed this way as the data contained in datasheets is often in this format so this config file provides a natural way to translating that data to a form the program can understand.

## File format

Below is an exampple showing how the data for Stainless Steel 316L has been added to the file.

```INI
[SS-316L]
; thermal conductivity values
K = [16.3,15.45,21.5]
; temperature ranges for each thermal conductivity value
K_T = [[0,20],[20,100],[100,500]]
; specific heat capacity values
C = [450.0,492.5,500.0]
; temperature ranges for each thermal specific heat capacity value
C_T = [[0,20],[20,93],[93,100]]
; volume expansion data, m^3/m^3.K
Ev = [54.2,56.4,58.2]
; temperature ranges for each thermal conductivity value
Ev_T = [[20,100],[100,500],[500,1000]]
; density values, rho,
p = [7975.666]
; temperature ranges for each density value, Kelvin
p_T = [[20,100]]
; melting temperature, Kelvin
Tm = 1663.15
; emissivity
e = 0.53
```

The material name is defined in the square brackets at the top. In this case, Stainless Steel 316L is defined as SS-316L. This name how the material data is referred to in the library.

The fields under each material, or sections as they are formally known, are defined as follows:
- K, K_T : Thermal conductivity reference values and the temperature ranges they are each valid across.
- C, C_T : Specific heat capacity reference values and the temperature ranges they are each valid across.
- Ev, Ev_T : Volume expansion factor reference values and the temperature ranges they are each valid across.
- p, p_T : Density reference values and the temperature ranges they are each valid across.
- Tm : Melting point of the material in Kelvin
- e : Emissivity ofo the material

The file format also has support for thermal diffusivity reference values under the field names D and D_T respectively.

Using SS-316L as an example, the first thermal conductivity value 16.3 is valid across the temperature range 0-20 degrees Celcius.

When LBAMMaterials is imported, it reads mat_conf.ini and checks if each material has the required fields. If it has all the fields, then it is included in a dictionary called mat_dict that is used by all the functions as the set of supported materials. If a material is found not to have the required fields, a message is printed to the screen informing the user that an invalid material was found. If the user attempts to use an unsupported material, an error message is printed and the required data isn't returned meaning it cannot be used in the modelling functions.

## Requirements

- Each material added to the config file must have the fields 'K','K_T','Ev','Ev_t','C','C_T','p','p_T','Tm','e'.

**While there currently is no requirement for the minimum number of reference points, more the merrier**

- The temperature ranges **must** be non overlapping. If they overlap then any attempts to fit a spline to this data, used in the modelling functions, will fail.

## Note about density

In the case of Stainless Steel there is only one reference point for density so of course it cannot be used for the entire temperature range. When the data for stainless steel is requested, the program recognises that there is only one data point and uses the volume expansion data to predict density over a wider temperature range. The maximum temperature is predicts up to is set at the maximum of the reference temperature range plus the value of the parameter past_max_temp.

This behaviour occurs if less than the minimum required data points, by default 2, is given. This is applied to all valid mateials. The minimum number of data points is set by the min_points parameter in params.

## Note about thermal diffusivity

Thermal diffusivity reference points are an optional field in the config file as it is often difficult to get reference values for the material. As such, thermal diffusivity is predicted using the thermal conductivity, specific heat capacity and density data. The temperature range it is predicted across is set in the paramter T_test_max.

**At present, the thermal diffusivity data points ARE NOT included in the spline data set. This is based on the experience thus far of not finding/obtaining reference data. Thermal diffusivity data comes purely from the predictions made using other data***
