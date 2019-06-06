# cmb_units
A quick guide to intensity and temperature conversion for the black body radiation and in particular the cosmic microwave background (CMB). Check out the `tutorial.ipynb` for a demonstration of the formulas and the code.

The module `lib/unit_conversion.py` has a convenient unit conversion function called `convert_units_of` that takes in a map/pixel with a certain unit and returns a map/pixel in another unit. 


Here's an example of how it can be used to convert a temperature fluctuation `dT` around the mean value of `T_0` to intensity fluctuations `dI_nu` at three frequency channels of 100, 143, and 217 GHz:

```
from lib.unit_conversion import convert_units_of

T_0 = 2.7255 # [K]
dT = 1E-5 # [K]

frequencies = [100,143,217] # [GHz]

dI = convert_units_of(dT, from_units="T", to_units="I",
                          at_nu=frequencies,
                          with_map_avg=T_0,
                          is_differential=True)

```
Depending on the input map and the given frequencies, the output `dI` will have the shape: 
(#frequency channels, #pixels). 


The valid keywords for `from_unit` and `to_unit` are:

| Variable [units] | `keyword` | 
| --- | --- |
| Brightness Temperature [K] | `"T", "T_b", "T_cmb", "T_CMB", "K", "K_CMB"` | 
| Specific Intensity [MJy/sr] | `"I", "I_nu", "MJy/sr"` | 
| Rayleigh_Jeans Temperature [K_RJ] | `"T_RJ", "T_rj", "s_nu", "K_RJ", "K_rj"`| 
        
 
