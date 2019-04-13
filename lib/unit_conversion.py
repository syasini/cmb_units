"""
A module for converting Intensity, thermodynamic and brightness temperature of black body radiation
"""

__author__ = "Siavash Yasini"
__email__ = "yasini@usc.edu"


import numpy as np
import pdb
import matplotlib.pyplot as plt

##################################################
#                   constants
##################################################

h_over_k = 0.047992447  # :[K*s]
h_over_c2 = 7.3724972E-4  # : [MJy/sr/GHz^3]
k3_over_c2h2 = 6.66954097  # :[MJy/sr]
c2_over_k = 65.096595  # :[K*GHz^2/MJy/sr]

##################################################
#               frequency functions
##################################################

# =================================
#         Absolute Measurement
# =================================

# ---------------
# T or T_b to I_nu
# ---------------

def black_body(nu, T, RJ=False):
    """
    BB spectrum [MJy/sr] with a temperature T [K] at frequency nu [GHz]

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :param RJ: if True, input is brightness temperature
    :return: I_nu [MJy/sr]
    """

    if RJ:
        return 2/c2_over_k * nu**2 *T

    else:
        x = h_over_k * nu / T  # nu: [GHz] ; T: [K]
        g = x ** 3 / np.expm1(x)
        return 2 * k3_over_c2h2 * T ** 3 * g


# -----------------
# I_nu to T and T_b
# -----------------

def bright_temp(nu, I_nu, RJ=False):
    """
    Thermodynamic temperature T [K] at frequency nu [GHz] for the observed intensity [MJy/sr]
    if RJ==True, return brightness temperature

    :param nu: observed frequency [GHz]
    :param I_nu: specific intensity [MJy/sr]
    :param RJ: if True, output is brightness temperature
    :return: T [K]
    """

    if RJ:
        return I_nu * c2_over_k / 2 / nu ** 2

    else:
        return h_over_k * nu / np.log(1 + 2 * h_over_c2 * nu ** 3 / I_nu)


# ---------
# T to T_b
# ---------

def T_thermo2bright(nu, T):
    """
    brightness temperature T_b [K] at frequency nu [GHz] for a BB with temperature T

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :return: T_b [K]
    """

    x = h_over_k * nu / T  # nu [GHz] ; t [#K]
    g = x / np.expm1(x)
    return g * T


# ---------
# T_b to T
# ---------

def T_bright2thermo(nu, T_b):
    """
    thermodynamic temperature T [K] at frequency nu [GHz] for the observed brightness temperature T_b

    :param nu: observed frequency [GHz]
    :param T_b: brightness temperature [K]
    :return: T [K]
    """

    x_b = h_over_k * nu / T_b  # nu [GHz] ; t [#K]
    g = x_b / np.log1p(x_b)
    return g * T_b


# =================================
#     Differential Measurement
# =================================

# NOTE: The following functions only calculate the frequency dependence of the transformation
#       Inorder to calculate the actual differential observable, the frequency functions (FF)
#       have to be multiplied by dX/X where X is the input value
#       e.g.: dI_nu = diff_black_body(nu, T0)* dT/T0


# -------------------
# dT or dT_b to dI_nu
# -------------------

def diff_black_body(nu, T, RJ=False):
    """
    differential BB spectrum [MJy/sr] with a temperature T [K] at frequency nu [GHz]
    ***multiply by dT/T to get dI_nu***

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :param RJ: if True, input is brightness temperature
    :return: dI_nu*T/dT [MJy/sr]
    """

    if RJ:
        return 2/c2_over_k * nu**2 *T

    else:
        x = h_over_k * nu / T  # nu: [GHz] ; T: [K]
        f = x ** 4 * np.exp(x) / np.expm1(x) ** 2

        return 2 * k3_over_c2h2 * T ** 3 * f


# --------------------
# dI_nu to dT and dT_b
# --------------------

def diff_bright_temp(nu, I_nu, RJ=False):
    """
    differential thermodynamic temperature dT [K] at frequency nu [GHz] for the observed
    intensity [MJy/sr]
    if RJ==True, return differential brightness temperature
    ***multiply by dI_nu/I_nu0 to get dT or dT_b***

    :param nu: observed frequency [GHz]
    :param I_nu: specific intensity [MJy/sr]
    :param RJ: if True, output is brightness temperature
    :return: dT*I_nu0/dI_nu  [K]
    """

    if RJ:
        return I_nu * c2_over_k / 2 / nu ** 2

    else:
        X = 2 * h_over_c2 * nu ** 3 / I_nu

        return 2 * h_over_k * h_over_c2 * nu ** 4 / I_nu / (1 + X) / np.log(1 + X) ** 2


# ----------
# dT to dT_b
# ----------

def dT_thermo2bright(nu, T):
    """the frequency function for converting thermodynamic temperature fluctuation
    to the frequency dependent brightness temperature fluctuation.
    ***Multiply by dT/T to get dT_b***

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :return: dT_b*T/dT  [K]
    """

    x = h_over_k * nu / T  # nu [GHz] ; t [#K]

    g = x ** 2 * np.exp(x) / np.expm1(x) ** 2

    return g * T


# ----------
# dTb to dT
# ----------

def dT_bright2thermo(nu, T_b):
    """the frequency function for converting thermodynamic temperature fluctuation
    to the frequency dependent brightness temperature fluctuation.
    ***Multiply by dT/T to get dT_b***

    :param nu: observed frequency [GHz]
    :param T_b: brightness temperature [K]
    :return: dT*T_b/dT_b  [K]
    """

    x_b = h_over_k * nu / T_b  # nu [GHz] ; t [#K]

    g = x_b / (1+x_b) / np.log1p(x_b)

    return g * T_b


##################################################
#              Conversion Wrappers
##################################################

# =================================
#        Absolute Measurement
# =================================

# --------
# T & I_nu
# --------

def _convert_T_2_I(nu, T, verbose=True):
    if verbose:
        print(converting_message("T","I"))
    I_nu = black_body(nu, T ,RJ=False)
    return I_nu


def _convert_I_2_T(nu, I_nu, verbose=True):
    if verbose:
        print(converting_message("I","T"))
    T = bright_temp(nu, I_nu, RJ=False)
    return T

# ---------
# Tb & I_nu
# ---------

def _convert_Tb_2_I(nu, T_b, verbose=True):
    if verbose:
        print(converting_message("Tb","I"))
    I_nu = black_body(nu, T_b, RJ=True)
    return I_nu


def _convert_I_2_Tb(nu, I_nu, verbose=True):
    if verbose:
        print(converting_message("I","Tb"))
    T_b = bright_temp(nu, I_nu, RJ=True)
    return T_b


# --------
# T & T_b
# --------

def _convert_T_2_Tb(nu, T, verbose=True):
    if verbose:
        print(converting_message("T","Tb"))
    T_b = T_thermo2bright(nu, T)
    return T_b


def _convert_Tb_2_T(nu, T_b, verbose=True):
    if verbose:
        print(converting_message("Tb","T"))
    T = T_bright2thermo(nu, T_b)
    return T

# =================================
#       Differential Measurement
# =================================

# ----------
# dT & dI_nu
# ----------

def _convert_dT_2_dI(nu, dT, T0, verbose=True):
    if verbose:
        print(converting_message("dT","dI"))
    dT_ovr_T0 = dT / T0
    dI_nu = diff_black_body(nu, T0, RJ=False) * dT_ovr_T0
    return dI_nu


def _convert_dI_2_dT(nu, dI_nu, I_nu0, verbose=True):
    if verbose:
        print(converting_message("dI","dT"))
    dI_ovr_I0 = dI_nu / I_nu0
    dT = diff_bright_temp(nu, I_nu0, RJ=False) * dI_ovr_I0
    return dT

# -----------
# dTb & dI_nu
# -----------

# NOTE: These two conversions don't necessarily need T_b0 or I_nu0
def _convert_dTb_2_dI(nu, dT_b, T_b0=1, verbose=True):
    if verbose:
        print(converting_message("dTb","dI"))
    dTb_ovr_Tb0 = dT_b / T_b0
    dI_nu = diff_black_body(nu, T_b0, RJ=True) * dTb_ovr_Tb0
    return dI_nu


def _convert_dI_2_dTb(nu, dI_nu, I_nu0=1, verbose=True):
    if verbose:
        print(converting_message("dI","dTb"))
    dI_ovr_I0 = dI_nu / I_nu0
    dT_b = diff_bright_temp(nu, dI_nu, RJ=True) * dI_ovr_I0
    return dT_b

# --------
# dTb & dT
# --------

def _convert_dT_2_dTb(nu, dT, T0, verbose=True):
    if verbose:
        print(converting_message("dT","dTb"))
    dT_ovr_T0 = dT / T0
    dT_b = dT_thermo2bright(nu, T0) * dT_ovr_T0
    return dT_b


def _convert_dTb_2_dT(nu, dT_b, T_b0, verbose=True):
    if verbose:
        print(converting_message("dTb","dT"))
    dTb_ovr_Tb0 = dT_b / T_b0
    dT = dT_bright2thermo(nu, T_b0) * dTb_ovr_Tb0
    return dT


# =================================
#  conversion wrapper dictionaries
# =================================

def converting_message(unit1_str, unit2_str):

    comments = {"T" : "T: thermodynamic (T)emperature",
                "I" : "I: specific (I)ntensity",
                "Tb": "Tb: (b)rightness (T)emperature",
                "dT" : "dT: (d)ifferential thermodynamic (T)emperature",
                "dI" : "dI: (d)ifferential specific (I)ntensity",
                "dTb": "dTb: (d)ifferential (b)rightness (T)emperature"
                }

    units = {"T" : "K",
             "I" : "MJy/sr",
             "Tb": "K_RJ",
             "dT" : "K",
             "dI" : "MJy/sr",
             "dTb": "K_RJ",
              }

    return ("\nconverting {} [{}] to {} [{}]\n".format(unit1_str,
                                                       units[unit1_str],
                                                       unit2_str,
                                                       units[unit2_str])
             + comments[unit1_str] + "\n"
             + comments[unit2_str] + "\n")


def _convert_abs_unit_of(map_, from_, to_, nu_,verbose):
    """call the absolyte conversion function"""

    conversion_dict = {"T_2_I" : _convert_T_2_I,
                       "T_2_Tb": _convert_T_2_Tb,

                       "I_2_T" : _convert_I_2_T,
                       "I_2_Tb": _convert_I_2_Tb,

                       "Tb_2_T": _convert_Tb_2_T,
                       "Tb_2_I": _convert_T_2_I,
                       }

    return conversion_dict.get("{}_2_{}".format(from_, to_))(nu_, map_, verbose)


def _convert_diff_unit_of(map_, from_, to_, nu_, map_avg, verbose):
    """call the differential conversion function"""

    conversion_dict = {"dT_2_dI" : _convert_dT_2_dI,
                       "dT_2_dTb": _convert_dT_2_dTb,

                       "dI_2_dT" : _convert_dI_2_dT,
                       "dI_2_dTb": _convert_dI_2_dTb,

                       "dTb_2_dT": _convert_dTb_2_dT,
                       "dTb_2_dI": _convert_dT_2_dI,
                       }
    return conversion_dict.get("d{}_2_d{}".format(from_, to_))(nu_, map_, map_avg, verbose)


def lookup(unit_str):
    """look up the input keyword in the dictionary and return the standard synonym"""

    unit_dict = {"T": ["T", "T_cmb", "T_CMB", "K", "K_CMB"],
                 "Tb": ["Tb", "T_RJ", "T_nu", "K_RJ"],
                 "I": ["I", "I_nu", "MJy/sr"]
                 }

    unit_synonym = [key for key, value in unit_dict.items() if unit_str in value][0]

    return unit_synonym

# =================================
#      main conversion wrapper
# =================================

def convert_units_of(input_map,
                     from_units="T",
                     to_units="I",
                     at_nu=217,
                     with_map_avg=None,
                     is_differential=True,
                     verbose=True):
    """
    Parameters
    ----------
    input_map:
        input map or pixel

    from_units:
        the units of input map. can be one of the keywords below:

        "T" or: "T_cmb", "T_CMB", "K", "K_CMB";
        "Tb" or: "T_RJ", "T_nu", "K_RJ";
        "I" or: "I_nu", "MJy/sr"

    to_units:
        the units of the output map (can be "T", "T_b" or "I"); see above for
        equivalent keywords

    at_nu:
        the observed frequency of the input map in **GHz** (can be an array for pixels)
        the input can be a scalar (single frequency)
        or a 1d list/np.ndarray (a range of frequencies)
        The default value is 217 GHz

    with_map_avg:
        the average of the input map e.g. T_0 for a dT map or I_nu0 for dI_nu.
        This is necessary for differential maps because the frequency function of the transformation
        depends on the average. If using spherical harmonic monopole, make sure you divide by 2pi;
        average=monopole/(2*np.sqrt(np.pi)

    is_differential:
        if True, the unit conversion is done for differential measurement.
        e.g. dT to dI_nu etc.

    verbose:
        if True, print out the conversion type

    Returns
    -------
        the original map/pixel in the units provided in "to_units" at frequency "at_nu"

    """

    # look up from_ and to_ from the dictionary and return the standard units
    from_ = lookup(from_units)
    to_ = lookup(to_units)

    # check the dimensions of the frequency array
    # if the input is a map (len(map) = npix) and more than one frequency is needed (n_freq),
    # the shape of the frequency input should match (n_freq,npix)

    if np.isscalar(input_map):
        npix = 1
    elif isinstance(input_map,(list,np.ndarray)):
        npix = len(input_map)
        input_map = np.array(input_map)
    else:
        raise TypeError("input_map must be either a scalar, list, or np.ndarray")

    #the input frequency must be either a scalar or a 1d array
    assert np.ndim(at_nu)<2

    if np.isscalar(at_nu):
        n_freq = 1
    elif isinstance(at_nu, (list, np.ndarray)):
        n_freq = len(at_nu)
    else:
        raise TypeError("at_nu (frequency) must be either a scalar, list, or np.ndarray")

    # if the input is a map, promote the frequency array to a 2d matrix with the shape(n_freq, npix)
    if (npix != 1) and (n_freq != 1):
        at_nu = np.tensordot(at_nu, np.ones_like(input_map),axes=0)

    if from_ == to_:
        print("returning the original input.\n")
        return input_map

    # check if the conversion if for differential or absolute measurements
    if is_differential:
        return _convert_diff_unit_of(input_map, from_, to_, at_nu, with_map_avg, verbose)
    else:
        return _convert_abs_unit_of(input_map, from_, to_, at_nu, verbose)


if __name__ == "__main__":

    print("\nrunning some test conversions\n{:=<30}".format(""))

    # set the thermodynamic temperature
    T_0 = 2.7255 # [K]
    dT = 1E-5 # [K]

    #set the frequency range
    nu_arr = np.linspace(1,1000,1000)

    # convert T and dT to I, dI, Tb & dTb
    I_nu = convert_units_of(T_0, from_units="T", to_units="I",
                            at_nu=nu_arr,
                            with_map_avg=None,
                            is_differential=False)

    dI_nu = convert_units_of(dT, from_units="T", to_units="I",
                             at_nu=nu_arr,
                             with_map_avg=T_0,
                             is_differential=True)

    T_b = convert_units_of(T_0, from_units="T", to_units="Tb",
                           at_nu=nu_arr,
                           with_map_avg=None,
                           is_differential=False)

    dT_b = convert_units_of(dT, from_units="T", to_units="Tb",
                             at_nu=nu_arr,
                             with_map_avg=T_0,
                             is_differential=True)

    #plot the results
    fig, axis = plt.subplots(2,2,figsize=(8,8),dpi=100)

    axis[0,0].plot(nu_arr,I_nu,label="I_nu [MJy/sr]")
    axis[0,1].plot(nu_arr,dI_nu,label="dI_nu [MJy/sr]")
    axis[1,0].semilogx(nu_arr,T_b,label="T_b [K_RJ]")
    axis[1,1].semilogx(nu_arr,dT_b,label="dT_b [K_RJ]")

    fig.suptitle("T = {}, dT = {}".format(T_0,dT),y =1)

    for ax in axis.flatten():
        ax.legend()
        ax.set_xlabel("frequency [GHz]")

    plt.show()



