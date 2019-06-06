"""
A library for converting Intensity, brightness and Rayleigh_jeans temperature of black body
radiation
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
#           conversion wrapper
##################################################

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
        must be either a scalar or a list or np.ndarray of pixel values

    from_units:
        the units of input map. can be one of the keywords below:

        "T" or: "T_b", "T_cmb", "T_CMB", "K", "K_CMB";
        "T_RJ" or: "T_rj", "s_nu", "K_RJ", "K_rj";
        "I" or: "I_nu", "MJy/sr"

    to_units:
        the units of the output map (can be "T", "T_RJ" or "I"); see above for
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
    from_units = lookup(from_units)
    to_units = lookup(to_units)

    # check the dimensions of the frequency array
    # if the input is a map (len(map) = npix) and more than one frequency is needed (n_freq),
    # the shape of the frequency input should match (n_freq,npix)

    if np.isscalar(input_map):
        npix = 1

    #FIXME: include multifrequency input maps
    elif isinstance(input_map,(list,np.ndarray)):
        npix = len(input_map)
        input_map = np.array(input_map)
    else:
        raise TypeError("input_map must be either a scalar, list, or np.ndarray")

    #the input frequency must be either a scalar or a 1d array
    assert np.ndim(at_nu)<2, "The dimension of input frequency must be smaller than 2 (only " \
                             "scalar or vector)"

    if np.isscalar(at_nu):
        n_freq = 1
    elif isinstance(at_nu, (list, np.ndarray)):
        n_freq = len(at_nu)
    else:
        raise TypeError("at_nu (frequency) must be either a scalar, list, or np.ndarray")

    # if the input is a map, promote the frequency array to a 2d matrix with the shape(n_freq, npix)
    if (npix != 1) and (n_freq != 1):
        at_nu = np.tensordot(at_nu, np.ones_like(input_map),axes=0)

    if from_units == to_units:
        print("returning the original input.\n")
        return input_map

    # check if the conversion is for differential or absolute measurements
    if is_differential:
        return _convert_diff_unit_of(input_map, from_units, to_units, at_nu, with_map_avg, verbose)
    else:
        return _convert_abs_unit_of(input_map, from_units, to_units, at_nu, verbose)


##################################################
#               frequency functions
##################################################

# =================================
#         Absolute Measurement
# =================================

# -----------------
# T or T_RJ to I_nu
# -----------------

def black_body(nu, T, RJ=False):
    """
    BB spectrum [MJy/sr] with a temperature T [K] at frequency nu [GHz]

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :param RJ: if True, input is RJ temperature
    :return: I_nu [MJy/sr]
    """

    if RJ:
        return 2/c2_over_k * nu**2 *T

    else:
        x = h_over_k * nu / T  # nu: [GHz] ; T: [K]
        g = x ** 3 / np.expm1(x)
        return 2 * k3_over_c2h2 * T ** 3 * g


# ------------------
# I_nu to T and T_RJ
# ------------------

def bright_temp(nu, I_nu, RJ=False):
    """
    Thermodynamic temperature T [K] at frequency nu [GHz] for the observed intensity [MJy/sr]
    if RJ==True, return RJ temperature

    :param nu: observed frequency [GHz]
    :param I_nu: specific intensity [MJy/sr]
    :param RJ: if True, output is RJ temperature
    :return: T [K]
    """

    if RJ:
        return I_nu * c2_over_k / 2 / nu ** 2

    else:
        return h_over_k * nu / np.log(1 + 2 * h_over_c2 * nu ** 3 / I_nu)


# ---------
# T to T_RJ
# ---------

def T_bright2RJ(nu, T):
    """
    RJ temperature T_RJ [K] at frequency nu [GHz] for a BB with temperature T

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :return: T_RJ [K]
    """

    x = h_over_k * nu / T  # nu [GHz] ; t [#K]
    g = x / np.expm1(x)
    return g * T


# ---------
# T_RJ to T
# ---------

def T_RJ2bright(nu, T_RJ):
    """
    thermodynamic temperature T [K] at frequency nu [GHz] for the observed brightness temperature T_b

    :param nu: observed frequency [GHz]
    :param T_RJ: RJ temperature [K]
    :return: T [K]
    """

    x_RJ = h_over_k * nu / T_RJ  # nu [GHz] ; t [#K]
    g = x_RJ / np.log1p(x_RJ)
    return g * T_RJ


# =================================
#     Differential Measurement
# =================================

# NOTE: The following functions only calculate the frequency dependence of the transformation
#       Inorder to calculate the actual differential observable, the frequency functions (FF)
#       have to be multiplied by dX/X where X is the input value
#       e.g.: dI_nu = diff_black_body(nu, T0)* dT/T0


# --------------------
# dT or dT_RJ to dI_nu
# --------------------

def diff_black_body(nu, T, RJ=False):
    """
    differential BB spectrum [MJy/sr] with a temperature T [K] at frequency nu [GHz]
    ***multiply by dT/T to get dI_nu***

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :param RJ: if True, input is RJ temperature
    :return: dI_nu*T/dT [MJy/sr]
    """

    if RJ:
        return 2/c2_over_k * nu**2 *T

    else:
        x = h_over_k * nu / T  # nu: [GHz] ; T: [K]
        f = x ** 4 * np.exp(x) / np.expm1(x) ** 2

        return 2 * k3_over_c2h2 * T ** 3 * f


# ---------------------
# dI_nu to dT and dT_RJ
# ---------------------

def diff_bright_temp(nu, I_nu, RJ=False):
    """
    differential thermodynamic temperature dT [K] at frequency nu [GHz] for the observed
    intensity [MJy/sr]
    if RJ==True, return differential RJ temperature
    ***multiply by dI_nu/I_nu0 to get dT or dT_RJ***

    :param nu: observed frequency [GHz]
    :param I_nu: specific intensity [MJy/sr]
    :param RJ: if True, output is RJ temperature
    :return: dT*I_nu0/dI_nu  [K]
    """

    if RJ:
        return I_nu * c2_over_k / 2 / nu ** 2

    else:
        X = 2 * h_over_c2 * nu ** 3 / I_nu

        return 2 * h_over_k * h_over_c2 * nu ** 4 / I_nu / (1 + X) / np.log(1 + X) ** 2


# -----------
# dT to dT_RJ
# -----------

def dT_bright2RJ(nu, T):
    """the frequency function for converting thermodynamic temperature fluctuation
    to the frequency dependent RJ temperature fluctuation.
    ***Multiply by dT/T to get dT_RJ***

    :param nu: observed frequency [GHz]
    :param T: thermodynamic temperature [K]
    :return: dT_RJ*T/dT  [K]
    """

    x = h_over_k * nu / T  # nu [GHz] ; t [#K]

    g = x ** 2 * np.exp(x) / np.expm1(x) ** 2

    return g * T


# -----------
# dT_RJ to dT
# -----------

def dT_RJ2bright(nu, T_RJ):
    """the frequency function for converting thermodynamic temperature fluctuation
    to the frequency dependent RJ temperature fluctuation.
    ***Multiply by dT/T to get dT_RJ***

    :param nu: observed frequency [GHz]
    :param T_b: brightness temperature [K]
    :return: dT*T_RJ/dT_RJ  [K]
    """

    x_RJ = h_over_k * nu / T_RJ  # nu [GHz] ; t [#K]

    g = x_RJ / (1+x_RJ) / np.log1p(x_RJ)

    return g * T_RJ


##################################################
#         T, T_RJ, & I Conversion Wrappers
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

# -----------
# T_RJ & I_nu
# -----------

def _convert_T_RJ_2_I(nu, T_RJ, verbose=True):
    if verbose:
        print(converting_message("T_RJ","I"))
    I_nu = black_body(nu, T_RJ, RJ=True)
    return I_nu


def _convert_I_2_T_RJ(nu, I_nu, verbose=True):
    if verbose:
        print(converting_message("I","T_RJ"))
    T_RJ = bright_temp(nu, I_nu, RJ=True)
    return T_RJ


# --------
# T & T_RJ
# --------

def _convert_T_2_T_RJ(nu, T, verbose=True):
    if verbose:
        print(converting_message("T","T_RJ"))
    T_RJ = T_bright2RJ(nu, T)
    return T_RJ


def _convert_T_RJ_2_T(nu, T_RJ, verbose=True):
    if verbose:
        print(converting_message("T_RJ","T"))
    T = T_RJ2bright(nu, T_RJ)
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

# -------------
# dT_RJ & dI_nu
# -------------

# NOTE: Due to their linear nature, these two conversions don't need T_b0 or I_nu0
def _convert_dT_RJ_2_dI(nu, dT_RJ, T_RJ0=1, verbose=True):
    if verbose:
        print(converting_message("dT_RJ","dI"))
    dT_RJ_ovr_T_RJ0 = dT_RJ / T_RJ0
    dI_nu = diff_black_body(nu, T_RJ0, RJ=True) * dT_RJ_ovr_T_RJ0
    return dI_nu


def _convert_dI_2_dT_RJ(nu, dI_nu, I_nu0=1, verbose=True):
    if verbose:
        print(converting_message("dI","dT_RJ"))
    dI_ovr_I0 = dI_nu / I_nu0
    dT_RJ = diff_bright_temp(nu, I_nu0, RJ=True) * dI_ovr_I0
    return dT_RJ

# ----------
# dT_RJ & dT
# ----------

def _convert_dT_2_dT_RJ(nu, dT, T0, verbose=True):
    if verbose:
        print(converting_message("dT","dT_RJ"))
    dT_ovr_T0 = dT / T0
    dT_RJ = dT_bright2RJ(nu, T0) * dT_ovr_T0
    return dT_RJ


def _convert_dT_RJ_2_dT(nu, dT_RJ, T_RJ0, verbose=True):
    if verbose:
        print(converting_message("dT_RJ","dT"))
    dT_RJ_ovr_T_RJ0 = dT_RJ / T_RJ0
    dT = dT_RJ2bright(nu, T_RJ0) * dT_RJ_ovr_T_RJ0
    return dT


# =================================
#  conversion wrapper dictionaries
# =================================

def converting_message(unit1_str, unit2_str):

    comments = {"T" : "T: brightness (T)emperature",
                "I" : "I: specific (I)ntensity",
                "T_RJ": "T_RJ: (R)ayleigh-(J)eans (T)emperature",
                "dT" : "dT: (d)ifferential brightness (T)emperature",
                "dI" : "dI: (d)ifferential specific (I)ntensity",
                "dT_RJ": "dT_RJ: (d)ifferential (R)ayleigh-(J)eans (T)emperature"
                }

    units = {"T" : "K",
             "I" : "MJy/sr",
             "T_RJ": "K_RJ",
             "dT" : "K",
             "dI" : "MJy/sr",
             "dT_RJ": "K_RJ",
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
                       "T_2_T_RJ": _convert_T_2_T_RJ,

                       "I_2_T" : _convert_I_2_T,
                       "I_2_TRJ": _convert_I_2_T_RJ,

                       "T_RJ_2_T": _convert_T_RJ_2_T,
                       "T_RJ_2_I": _convert_T_2_I,
                       }

    return conversion_dict.get("{}_2_{}".format(from_, to_))(nu_, map_, verbose)


def _convert_diff_unit_of(map_, from_, to_, nu_, map_avg, verbose):
    """call the differential conversion function"""

    conversion_dict = {"dT_2_dI" : _convert_dT_2_dI,
                       "dT_2_dT_RJ": _convert_dT_2_dT_RJ,

                       "dI_2_dT" : _convert_dI_2_dT,
                       "dI_2_dT_RJ": _convert_dI_2_dT_RJ,

                       "dT_RJ_2_dT": _convert_dT_RJ_2_dT,
                       "dT_RJ_2_dI": _convert_dT_2_dI,
                       }
    return conversion_dict.get("d{}_2_d{}".format(from_, to_))(nu_, map_, map_avg, verbose)


def lookup(unit_str):
    """look up the input keyword in the dictionary and return the standard synonym"""

    unit_dict = {"T": ["T", "T_b", "T_cmb", "T_CMB", "K", "K_CMB"],
                 "T_RJ": ["T_rj", "T_RJ", "s_nu", "K_RJ", "K_rj"],
                 "I": ["I", "I_nu", "MJy/sr"]
                 }

    try:
        unit_synonym = [key for key, value in unit_dict.items() if unit_str in value][0]

    except IndexError:
        import pprint
        pprint.pprint("Use a valid keyword from:")
        pprint.pprint(unit_dict)
        raise

    return unit_synonym


if __name__ == "__main__":

    print("\nrunning some test conversions\n{:=<30}".format(""))

    # set the thermodynamic temperature
    T_0 = 2.7255 # [K]
    dT = 1E-5 # [K]

    #set the frequency range
    nu_arr = np.linspace(1,1000,1000)

    # convert T and dT to I, dI, T_RJ & dTRJ
    I_nu = convert_units_of(T_0, from_units="T", to_units="I",
                            at_nu=nu_arr,
                            with_map_avg=None,
                            is_differential=False)

    dI_nu = convert_units_of(dT, from_units="T", to_units="I",
                             at_nu=nu_arr,
                             with_map_avg=T_0,
                             is_differential=True)

    T_b = convert_units_of(T_0, from_units="T", to_units="T_RJ",
                           at_nu=nu_arr,
                           with_map_avg=None,
                           is_differential=False)

    dT_b = convert_units_of(dT, from_units="T", to_units="T_RJ",
                             at_nu=nu_arr,
                             with_map_avg=T_0,
                             is_differential=True)

    #plot the results
    fig, axis = plt.subplots(2,2,figsize=(8,8),dpi=100)

    axis[0,0].plot(nu_arr,I_nu,label="I_nu [MJy/sr]")
    axis[0,1].plot(nu_arr,dI_nu,label="dI_nu [MJy/sr]")
    axis[1,0].semilogx(nu_arr,T_b,label="T_RJ [K_RJ]")
    axis[1,1].semilogx(nu_arr,dT_b,label="dT_RJ [K_RJ]")

    fig.suptitle("T = {}, dT = {}".format(T_0,dT),y =1)

    for ax in axis.flatten():
        ax.legend()
        ax.set_xlabel("frequency [GHz]")

    plt.show()



