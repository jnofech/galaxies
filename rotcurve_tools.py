import numpy as np
import math
import emcee

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy.table import Table
from spectral_cube import SpectralCube
from galaxies.galaxies import Galaxy
from astropy.coordinates import SkyCoord, Angle, FK5

from scipy import ndimage, misc, interpolate, optimize
from scipy.interpolate import BSpline, make_lsq_spline, splrep, LSQUnivariateSpline
from scipy.optimize import least_squares as lsq
from pandas import DataFrame, read_csv
import pandas as pd
import statsmodels.formula.api as smf
import os
import sys

# Import my own code
import galaxytools as tools
import diskfit_input_generator as dig

# Import Erik's code
from rotcurves import MCurc             # Performs MC sampling to improve URC fits and find errors

class silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def rotcurve(gal,data_mode='',mapmode='mom1',\
             rcpath='/media/jnofech/BigData/PHANGS/OtherData/derived/Rotation_curves/'):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    data_mode(='') : str
        '7m'            - uses 7m data.
        '12m' (default) - 12m data.
        'hybrid'        - combines 7m and 12m (OUTDATED).
        'phangs'        - Uses the PHANGS team's
                            12m+7m rotcurves that were on
                            the servers since before I
                            started working with DiskFit
                            (OUTDATED).
    mapmode(='mom1') : str
        'mom1'     - Reads DiskFit output based on mom1 map 
                     of specified `data_mode`.
        'peakvels' - Reads DiskFit output based on vpeak map 
                     of specified `data_mode` (OUTDATED).
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s.
    R_e : np.ndarray
        1D array of original rotcurve radii, in pc.
    vrot_e : np.ndarray
        1D array of original rotcurve errors, in pc.
    '''

    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = tools.galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Generates folder name and RC mode (which are used in RC filenames).
    if mapmode in ['peakvels','vpeak']:
        mapmode = 'vpeak'
    if mapmode not in ['vpeak','mom1']:
        raise ValueError('rotcurve : Invalid mapmode! Should be mom1 or peakvels.')
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'  
    elif data_mode=='':
        print('rotcurve : No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'
    elif data_mode=='phangs':
        print('rotcurve : WARNING: Using old PHANGS 12m+7m rotcurves. Very outdated!')
    else:
        raise ValueError('rotcurve : Invalid data_mode! Should be 7m, 12m, or hybrid.')
    rcmode = mapmode+'_'+data_mode       # RC is generated from <mapmode> data at <data_mode> resolution.
    folder='diskfit_auto_'+rcmode+'/'

    d = (gal.distance).to(u.parsec)      # Distance to galaxy, from Mpc to pc
    
    # Rotation Curves
    if data_mode.lower()=='phangs':
        fname = rcpath+name.lower()+"_co21_12m+7m+tp_RC.txt"
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    else:
        fname   = rcpath+folder+name.lower()+"_co21_"+rcmode+"_RC_procedural.txt"
        if os.path.isfile(fname):
            R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
        elif data_mode=='12m+7m' and os.path.isfile(fname_b):
            # Backup purposes only! Maybe replace these in the future?
            print('rotcurve : BACKUP - Reading "'+fname_b+'" instead! (Old manual 12m fits)')
            R, vrot, vrot_e = np.loadtxt(fname_b,skiprows=True,unpack=True)
        else:
            raise ValueError('rotcurve : File not found: '+fname)
        
    # R = Radius from center of galaxy, in arcsec.
    # vrot = Rotational velocity, in km/s.
    # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but both are 
    #     floats!
    
    # Units & conversions
    R = R*u.arcsec
    vrot = vrot*u.km/u.s
    R = R.to(u.rad)            # Radius, in radians.
    R = (R*d).value            # Radius, in pc, but treated as unitless.
    R_e = np.copy(R)           # Radius, corresponding to error bars.

    # Check if rotcurve is valid
    if np.isnan(np.sum(vrot)):
        print('ERROR: Rotcurve failed to generate!')
        return [np.nan]*4

    # BSpline interpolation of vrot(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot = interpolate.BSpline(t,c,k, extrapolate=False)    # Cubic interpolation of vrot(R).
                                                            # 'vrot' is now a function, not an array.
    # Creating "higher-resolution" rotation curve
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    
    return R, vrot, R_e, vrot_e
    
def rotcurve_smooth(R,vrot,R_e,vrot_e=None,smooth='spline',knots=3,returnparams=False):
    '''
    Takes a provided rotation curve and
    smooths it by simply fitting the DiskFit
    rotcurve to the specified model.
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s.
    R_e=None : np.ndarray
        1D array of original rotcurve radii, in pc.
    vrot_e=None : np.ndarray
        1D array of original rotcurve errors, in pc.
    smooth='spline' : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of `knots`)
        'brandt' (an analytical model)
        'universal' (Persic & Salucci 1995) (used in thesis)
    knots=3 : int
        Number of internal knots in BSpline of
        vrot, if smooth=='spline'.
    returnparams=False : bool
        Return RC fit parameters based on the
        model, on top of `R` and `vrot`?
        smooth='spline' : none           (0)
        smooth='brandt' : vmax,rmax,n    (3)
        smooth='universal' : vmax,rmax,A (3)
        smooth='simple' : vflat,rflat    (2)
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        SMOOTHED function for the interpolated
        rotation curve, in km/s.
    And more, if returnparams==True!
    '''
    # Check if rotcurve is valid
    if np.isnan(np.sum(R)):
        return [np.nan]*2
    
    # Check if any error values are exactly zero
    if vrot_e is None:
        print('WARNING (rotcurve_tools.rotcurve_smooth) : \
        vrot_e not defined, so all values are assumed to be zero.')
        vrot_e = np.zeros(R_e.size)
    if vrot_e[vrot_e==0].size>0:
        print('WARNING (rotcurve_tools.rotcurve_smooth) : \
        vrot_e has at least one value that is exactly zero. Will change to 1e-9.')
        vrot_e[vrot_e==0] = 1e-9
        
    # SMOOTHING:
    if smooth==None or smooth.lower()=='none':
        print( "WARNING: Smoothing disabled!")
    elif smooth.lower()=='spline':
        # BSpline of vrot(R)
        vrot = bspline(R_e,vrot(R_e),vrot_e,knots=knots,lowclamp=False)
    elif smooth.lower()=='brandt':
        def brandt(R, p0, p1, p2):
            '''
            Fit Eq. 1 Faber & Gallagher 79.
            This is taken right out of Erik Rosolowsky's code.
            '''
            p = np.array([p0, p1, p2])
            x = R / p[1]
            vmodel = p[0] * (x / (1 / 3 + 2 / 3 * x**p[2]))**(1 / (2 * p[2]))
            return(vmodel)
        def gof_brandt(p, radius=None, vrot=None, verr=None):
            vmodel = brandt(radius, *p)
            gof = np.abs((vmodel - vrot) / (verr))
            return(gof)
        p = np.zeros(3)
        p[0] = np.median(vrot(R_e))
        p[1] = np.max(R_e/1000.) / 2
        p[2] = 1
        output = optimize.least_squares(gof_brandt, p, loss='soft_l1',
                                        kwargs={'radius': R_e/1000.,
                                                'vrot': vrot(R_e),
                                                'verr': vrot_e},
                                        max_nfev=1e6,
                                        bounds=((0.01,0,0),(np.inf,np.inf,np.inf)))
        n_brandt = output.x[2]
        rmax_brandt = output.x[1]*1000.
        vmax_brandt = output.x[0]
        vrot_b = brandt(R,vmax_brandt,rmax_brandt,n_brandt)   # Array.
        # BSpline interpolation of vrot_b(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_b,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=False)  # Now it's a function.
        if returnparams==True:
            return R,vrot,vmax_brandt,rmax_brandt,n_brandt
        else:
            return R,vrot
    elif smooth.lower()=='universal':
        def urc(R, p0, p1, p2):
            '''
            Fit Eq. 14 from Persic & Salucci 1995.
            '''
            p = np.array([p0, p1, p2])
            x = R / p[1]
            vmodel = p[0] * ((0.72 + 0.44 * np.log(p[2]))
                             * (1.95 * x**1.22)/(x**2 + 0.78**2)
                             + 1.6 * np.exp(-0.4 * p[2])
                             * (x**2/(x**2 + 1.5**2 * p[2]**2)))
#             print(p[0],p[1],p[2])
            return(vmodel)
        def gof_urc(p, radius=None, vrot=None, verr=None):
            vmodel = urc(radius, *p)
            gof = np.abs((vmodel - vrot) / (verr))
            return(gof)
        p = np.zeros(3)
        p[0] = np.median(vrot(R_e))
        p[1] = np.max(R_e/1000.) / 2
        p[2] = 1
        output = optimize.least_squares(gof_urc, p, loss='soft_l1',
                                        kwargs={'radius': R_e/1000.,
                                                'vrot': vrot(R_e),
                                                'verr': vrot_e},
                                        max_nfev=1e6,
                                        bounds=((0.01,0,0),(np.inf,np.inf,np.inf)))
        A_urc = output.x[2]
        rmax_urc = output.x[1]*1000
        vmax_urc = output.x[0]
        vrot_u = urc(R,vmax_urc,rmax_urc,A_urc)   # Array.
        # BSpline interpolation of vrot_u(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_u,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=False)  # Now it's a function.
        if returnparams==True:
            return R,vrot,vmax_urc,rmax_urc,A_urc
        else:
            return R,vrot
    elif smooth.lower() in ['simple','simp','exponential','expo','exp']:
        def simple(R, p0, p1):
            '''
            Fit Eq. 8 from Leroy et al. 2013.
            '''
            p = np.array([p0, p1])
            x = R / p[1]
            vmodel = p[0]*(1.0-np.exp(-x))
#             print(p0,p1)
            return(vmodel)
        def gof_simple(p, radius=None, vrot=None, verr=None):
            vmodel = simple(radius, *p)
            gof = np.abs((vmodel - vrot) / (verr))
            return(gof)
        p = np.zeros(2)
        p[0] = np.median(vrot(R_e))
        p[1] = np.max(R_e/1000) / 2
        output = optimize.least_squares(gof_simple, p, loss='soft_l1',
                                        kwargs={'radius': R_e/1000,
                                                'vrot': vrot(R_e),
                                                'verr': vrot_e},
                                        max_nfev=1e6)
        rflat_simple = output.x[1]*1000
        vflat_simple = output.x[0]
        vrot_s = simple(R,vflat_simple,rflat_simple)   # Array.
        # BSpline interpolation of vrot_u(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_s,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=False)  # Now it's a function.
        if returnparams==True:
            return R,vrot,vflat_simple,rflat_simple
        else:
            return R,vrot
    else:
        raise ValueError('Invalid smoothing mode.')
    
    return R, vrot

def epicycles(R,vrot):
    '''
    Returns the epicyclic frequency from a
    given rotation curve.
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated
        rotation curve, in km/s.
        Smoothing is recommended!
        
        
    Returns:
    --------
    k : scipy.interpolate.interp1d
        Function for the interpolated epicyclic
        frequency.
    '''
    # Epicyclic Frequency
    dVdR = np.gradient(vrot(R),R)
    k2 =  2.*(vrot(R)**2 / R**2 + vrot(R)/R*dVdR)
    k = interpolate.interp1d(R,np.sqrt(k2))
    
    return k
    
def rotmap(gal,header=None,position_angle=None,inclination=None,data_mode='',mapmode='mom1',mode='best'):
    '''
    Returns 'model' V_los map from smoothed rotation curve,
    with corresponding 'radius map'.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    header=None : astropy.io.fits.header.Header
        Header for the galaxy.
        Will read header automatically if not provided.
    position_angle=None : astropy.units.Quantity
        Override for KINEMATIC position
        angle, in degrees. The Galaxy
        object's photometric PA value
        will be used if not specified.
    inclination=None : astropy.units.Quantity
        Override for inclination, in degrees.
    data_mode(='') : str
        '7m'            - uses 7m data.
        '12m' (default) - 12m data.
        'hybrid'        - combines 7m and 12m.
        'phangs'        - Uses the PHANGS team's
                            12m+7m rotcurves,
                            provided on server.
    mapmode(='mom1') : str
        'mom1' - uses mom1 map of specified
                 data_mode.
        'peakvels' - uses peakvels map of 
                     specified data_mode.
    mode(='best') : str
        - "smooth(ed)"  : uses smoothed rotcurve.
        - "(LSQ)urc" : uses final, lsq-improved rotcurve.
        - "(MC)urc" : uses final, MC-improved rotcurve (different method).
        - "best"    : uses MC, LSQ, or smoothed URC (in order of descending accuracy).
        
    Returns:
    --------
    vobs : np.ndarray
        Map of observed velocity, in km/s.
    rad : np.ndarray
        Map of radii in disk plane, up to
        extent of the rotcurve; in pc.
    Dec, RA : np.ndarray
        2D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    
    # Do not include in galaxies.py!
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('rc.rotmap() : WARNING: Galaxy not provided. Generating a new one automatically (NOT RECOMMENDED)!')
        gal = tools.galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    # data_mode, mapmode
    if mapmode in ['peakvels','vpeak']:
        mapmode = 'vpeak'
    if mapmode not in ['vpeak','mom1']:
        raise ValueError('rc.rotmap() : Invalid mapmode! Should be mom1 or peakvels.')
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'  
    elif data_mode=='':
        print('rc.rotmap() : No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'
    elif data_mode=='phangs':
        print('rc.rotmap() : WARNING: Using old PHANGS 12m+7m rotcurves. Likely outdated!')
    else:
        raise ValueError('rotcurve : Invalid data_mode! Should be 7m, 12m, or hybrid.')
    
    if header is None:
        print('rc.rotmap(): WARNING: Header was not provided. Automatically reading header instead.')
        header = tools.hdr_get(gal,data_mode,dim=2)
        
    vsys = gal.vsys
    
    if not inclination:
        I = gal.inclination
    else:
        I = inclination
    if not position_angle:
        PA = (gal.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
    else:
        PA = position_angle
        
    RA_cen = gal.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
    Dec_cen = gal.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc

    # vrot Interpolation
    R_1d, vrot = RC(gal,data_mode,mode=mode)

    # Generating displayable grids
    X,Y = gal.radius(header=header, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
    #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
    #               to the line of nodes.

    rad = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    rad = ( (rad.value<R_1d.max()) * (rad.value>R_1d.min())).astype(int) * rad  
    rad[ rad==0 ] = np.nan                         # Grid of radius, with values outside interpolation range removed.

    vobs = (vsys.value + vrot(rad)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, rad

def bspline(X,Y,dY=None,knots=8,k=3,lowclamp=False, highclamp=False):
    '''
    Returns a weighted BSpline interpolation function
    of a provided 1D curve.
    With fewer knots, this will provide a
    smooth curve that ignores local wiggles.
    
    Parameters:
    -----------
    X,Y : np.ndarray
        1D arrays for the curve being interpolated.
    dY=None : np.ndarray
        Uncertainties in Y(X).
    knots=8 : int
        Number of INTERNAL knots, i.e. the number
        of breakpoints that are being considered
        when generating the BSpline.
    k=3 : int
        Degree of the BSpline. Recommended to leave
        at 3.
    lowclamp=False : bool
        Enables or disables clamping at the lowest
        X-value.
    highclamp=False : bool
        Enables or disables clamping at the highest
        X-value.
        
    Returns:
    --------
    spl : scipy.interpolate._bsplines.BSpline
        Interpolation function that works over X's
        domain.
    '''
    
    # Creating the knots
    t_int = np.linspace(X.min(),X.max(),knots)  # Internal knots, incl. beginning and end points of domain.

    t_begin = np.linspace(X.min(),X.min(),k)
    t_end   = np.linspace(X.max(),X.max(),k)
    t = np.r_[t_begin,t_int,t_end]              # The entire knot vector.
    
    # Generating the spline
    w = np.zeros(X.shape)+1                     # Weights.
    if dY is not None and dY.shape==X.shape:
        w = 1./dY
    if lowclamp==True:
        w[0]=X.max()*1000000                    # Setting a high weight for the X.min() term.
    if highclamp==True:
        w[-1]=X.max()*1000000                   # Setting a high weight for the X.max() term.

    spl = make_lsq_spline(X, Y, t, k, w)
    
    return spl

def oort(R,vrot,oort=''):
    '''
    Returns the local shear parameter (i.e. the
    Oort A constant) for a galaxy with a provided
    rotation curve, based on Equation 4 in Martin
    & Kennicutt (2001).
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated
        rotation curve, in km/s.
        Smoothing is recommended!
    oort : str
        'A' - Returns Oort A.
        'B' - Returns Oort B.
                        
    Returns:
    --------
    A : scipy.interpolate._bsplines.BSpline
        Oort A "constant", as a function of 
        radius R, in km/s/kpc.
    '''    
    # Oort A versus radius.
    Omega = vrot(R) / R     # Angular velocity.
    dOmegadR = np.gradient(Omega,R)
    A_arr = (-1./2. * R*dOmegadR )*(u.kpc.to(u.pc)) # From km/s/pc to km/s/kpc.
    B_arr = A_arr - Omega*(u.kpc.to(u.pc))          # From km/s/pc to km/s/kpc.
    
    A = bspline(R[np.isfinite(A_arr*B_arr)],A_arr[np.isfinite(A_arr*B_arr)],knots=999)
    B = bspline(R[np.isfinite(A_arr*B_arr)],B_arr[np.isfinite(A_arr*B_arr)],knots=999)
    
    if oort=='':
        print('rotcurve_tools.oort():  WARNING: No \'oort\' mode selected! Returning Oort A by default.')
        oort='A'
    if oort.lower()=='a':
        return A
    elif oort.lower()=='b':
        return B
    else:
        raise ValueError('rotcurve_tools.oort():  Invalid \'oort\' mode!')

def beta(R,vrot_s):
    '''
    Returns beta parameter (the index 
        you would get if the rotation
        curve were a power function of
        radius, e.g. vrot ~ R**(beta)).
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of galaxy radii, in pc.
    vrot_s : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s. Ideally smoothed.
        
    Returns:
    --------
    beta : scipy.interpolate._bsplines.BSpline
        Beta parameter, as a function of 
        radius R.
    '''
    # Calculating beta
    dVdR = np.gradient(vrot_s(R),R)   # derivative of rotation curve;
    beta = R/vrot_s(R)*dVdR
    # Interpolating
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,beta,s=0,k=K)
    beta = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of beta
    return beta

def linewidth_iso(gal,beam=None,smooth='universal',knots=3,data_mode='7m',mapmode='mom1',mode='best'):
    '''
    Returns the effective LoS velocity dispersion
    due to the galaxy's rotation, sigma_gal, for
    the isotropic case.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    beam=None : float
        Beam width, in deg. Used to approximate
        size of clouds.
        Will be read from header automatically if not
        specified.
    smooth='spline' : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (uses specified # of `knots`)
        'brandt' (the analytical model)
        'universal' (DEFAULT) (Persic & Salucci 1995) (used in thesis)
    knots=3 : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve, which
        is used in calculation of epicyclic
        frequency (and, therefore, sigma_gal).
    data_mode(='') : str
        '7m'            - uses 7m data.
        '12m' (default) - 12m data.
        'hybrid'        - combines 7m and 12m (OUTDATED).
        'phangs'        - Uses the PHANGS team's
                            12m+7m rotcurves that were on
                            the servers since before I
                            started working with DiskFit
                            (OUTDATED).
    mapmode(='mom1') : str
        'mom1'     - Reads DiskFit output based on mom1 map 
                     of specified `data_mode`.
        'peakvels' - Reads DiskFit output based on vpeak map 
                     of specified `data_mode` (OUTDATED).
    mode='best' : str
        - "smooth(ed)"/"URC"  : grabs URC-smoothed rotcurve. (Simplified method.)
        - "LSQurc" : grabs final, lsq-optimized rotcurve with beam smearing correction.
        - "MCurc"  : grabs final, MC-sampled rotcurve with beam smearing correction.
        - "best"    : grabs LSQ, MC, or smoothed URC (in order of descending priority)
                      based on availability.

    Returns:
    --------
    sigma_gal : scipy.interpolate._bsplines.BSpline
        Interpolation function for sigma_gal that
        works over radius R.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = tools.galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Beam width
    if beam==None:
        print('rc.linewidth_iso(): WARNING: Beam size was not provided. Reading from header instead.')
        hdr = tools.hdr_get(gal,data_mode)
        beam = hdr['BMAJ']
    beam = beam*u.deg.to(u.rad)                 # Beam size, in radians
    d = (gal.distance).to(u.pc)
    Rc = beam*d / u.rad                         # Beam size, in parsecs (approximation for cloud size)

    # Find epicyclic frequency from rotcurve
    R,vrot_best = RC(gal,data_mode,mode=mode)    
    k = epicycles(R,vrot_best)
    
    # Calculate sigma_gal = kappa*Rc
    sigma_gal = k(R)*Rc

    # Removing nans and infs
    # (Shouldn't be anything significant-- just a "nan" at R=0.)
    index = np.arange(sigma_gal.size)
    R_clean = np.delete(R, index[np.isnan(sigma_gal)==True])
    sigma_gal_clean = np.delete(sigma_gal, index[np.isnan(sigma_gal)==True])
    sigma_gal = bspline(R_clean,sigma_gal_clean,knots=20)
    
    return sigma_gal
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~    MC UNIVERSAL ROTCURVE SAMPLING   ~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#def MCurc_sample(gal,data_mode,mapmode='mom1',smooth='universal'):
#    '''
#    Runs Erik's MCurc code on the mom1 map
#    and DiskFit-fitted parameters of a
#    galaxy.
#    
#    Parameters:
#    -----------
#    gal : Galaxy
#    data_mode(='') : str
#        '7m'            - uses 7m data.
#        '12m'           - 12m data.
#        'hybrid' (not implemented?)  - bofa
#    mapmode : str
#        I cannot guarantee what happens
#        if you don't choose 'mom1'.
#    smooth : str
#        Totally optional and totally doesn't
#        have to be 'universal'.
#    '''
#    if mapmode!='mom1':
#        print('rc.MCurc_sample : WARNING: Did not select \'mom1\' as mapmode. Might get wonky.')
#    rcmode = mapmode+'_'+data_mode       # RC is generated from <mapmode> data at <data_mode> resolution.
#    diskfit_folder='diskfit_auto_'+rcmode+'/'
#    if smooth!='universal':
#        raise ValueError('You did not use \'universal\' as the \'smooth\' mode. \nWTF YOU DOING YA DINGUS ITS LITERALLY THERE IN THE FUNCTION NAME')
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#        print('rc.MCurc_sample() - Creating new galaxy object for '+name)
#        gal = tools.galaxy(name)
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#    
#    print(name+' starting...')

#    # Check if data and error BOTH have 7m+tp data, or otherwise.
#    mom1_name, emom1_name = dig.filename_get(name,data_mode,mapmode,force_same_res=False)
#    # ^ force_same_res=False causes filename_get() to leave emom1_name as 'None' if 7m+tp mom1 exists
#    #    but 7m+tp emom1 does not. It basically checks whether error maps are available for the best
#    #    mom1 map, which DiskFit was run on. 
#    if emom1_name is not 'None':       # If data&error have matching maps, i.e. everything's good:
#        mom1path = tools.mom1_get(gal,data_mode,return_mode='path')
#        emom1path = tools.emom1_get(gal,data_mode,return_mode='path')
#        mom1 = MCurc.read_moment(mom1path)
#        emom1 = MCurc.read_moment(emom1path)
#    else:
#        print(name+' has no error map! Cannot continue.')
#        return
#        # Skip this galaxy entirely! Can't do diddly squat with no error map.    
#    
#    ## This block shows how to run the code.
#    gal2 = tools.galaxy(name.upper(),diskfit_output=True)
#    Distance = gal2.distance.to(u.Mpc)
#    Center = gal2.center_position
#    Vsys = gal2.vsys
#    PA = (gal2.position_angle.to(u.rad)).value
#    Inc = (gal2.inclination.to(u.rad)).value

#    # Get smoothed rotcurve!
#    R, vrot, R_e, vrot_e = gal.rotcurve(data_mode,mapmode)
#    R,vrot_s,vmax,rmax,A = rotcurve_smooth(R,vrot,R_e,vrot_e,smooth,returnparams=True)
#    rmax = rmax/1000.   # Convert from pc to kpc, as MCurc.lp_urc's "R" is in kpc.

#    p = np.array([vmax, rmax, A, PA, Inc])      # p[0],p[1],p[2] are URC fit parameters!
#    lp = MCurc.lp_urc(p, mom1, emom1, Center, Distance, Vsys, Inc)

#    nwalkers = 30
#    ndim = 4

#    p0 = np.zeros((nwalkers, ndim))
#    p0[:, 0] = p[0] + np.random.randn(nwalkers)
#    p0[:, 1] = p[1] + 0.01 * np.random.randn(nwalkers)
#    p0[:, 2] = p[2] + 0.01 * np.random.randn(nwalkers)
#    p0[:, 3] = p[3] + 0.01 * np.random.randn(nwalkers)
#    # p0[:, 4] = p[4] + 0.01 * np.random.randn(nwalkers)

#    arglist = [mom1, emom1, Center, Distance, Vsys, Inc]
#    sampler = emcee.EnsembleSampler(nwalkers,
#                                   ndim,
#                                   MCurc.lp_urc,
#                                   args=arglist, threads=6)
#    pos, prob, state = sampler.run_mcmc(p0, 200)
#    sampler.reset()
#    pos, prob, state = sampler.run_mcmc(pos, 500)
#    sampler.reset()
#    
#    vmax_mc = np.median(pos[:,0])
#    rmax_mc = np.median(pos[:,1])
#    A_mc    = np.median(pos[:,2])
#    PA_mc   = np.median(pos[:,3])
#    print('... '+name+' finished!')
#    
#    np.savez('MCurc_save/'+name.upper()+'_'+data_mode+'_'+smooth+'.npz',\
#             egg1=[vmax_mc,rmax_mc,A_mc,PA_mc],\
#             egg2=[vmax,rmax,A],\
#             egg3=pos,egg4=prob,egg5=state[1])
#    return

def urc(R, p0, p1, p2):
    '''
    Fit Eq. 14 from Persic & Salucci 1995.
    Note that "R" and "p1" must be in the same units.
    '''
    p = np.array([p0, p1, p2])
    x = R / p[1]
    vmodel = p[0] * ((0.72 + 0.44 * np.log(p[2]))
                     * (1.95 * x**1.22)/(x**2 + 0.78**2)
                     + 1.6 * np.exp(-0.4 * p[2])
                     * (x**2/(x**2 + 1.5**2 * p[2]**2)))
    return(vmodel)
    
def RC(gal,data_mode,mapmode='mom1',smooth='universal',mode='diskfit',returnparams=False,\
       knots=3,returnknots=False,\
       path='/media/jnofech/BigData/galaxies/'):
    '''
    Returns rotcurve specified by
    `mapmode` and `smooth`, in various
    stages of smoothing.
    (Only mapmode='mom1' and smooth='universal'
    are supported for now!)
    
    Parameters:
    -----------
    gal : Galaxy
    data_mode(='') : str
        '7m'            - uses 7m data.
        '12m'           - 12m data.
        'hybrid' (not implemented)
    mapmode : str
        Only 'mom1' supported
    smooth : str
        Only 'universal' supported
    mode='diskfit' : str
        - "diskfit"           : grabs raw DiskFit rotcurve, with errors.
        - "spline"            : grabs spline-smoothed rotcurve (requires `knots`).
        - "smooth(ed)"/"URC"  : grabs URC-smoothed rotcurve. (Simplified method.)
        - "LSQurc" : grabs final, lsq-optimized rotcurve with beam smearing correction.
        - "MCurc"  : grabs final, MC-sampled rotcurve with beam smearing correction.
        - "best"    : grabs LSQ, MC, or smoothed URC (in order of descending priority)
                      based on availability.
    returnparams=False : bool
        - False : Returns rotcurve.
        - True  : Returns rotcurve with URC parameters.
    knots=-1 : int
        Number of knots if mode=='spline'.
        `knots==-1` will use custom number of knots
        if available, or 3 knots otherwise.
    returnknots=False : bool
        - False : Returns rotcurve.
        - True  : Returns rotcurve with number of knots (in case it's a custom value).
        
    Returns:
    --------
    mode='diskfit':
        R,vrot,R_e,vrot_e
          ^BSpline
    mode='spline':
        R,vrot_spl
          ^BSpline
    mode='smoothed':
        R,vrot_s(,vmax,rmax,A)
          ^BSpline     ^pc
    mode='LSQurc'
        R,vrot_lsq(,vmax,rmax,A)
          ^BSpline      ^pc
    mode='MCurc'
        R,vrot_mc(,vmax,rmax,A)
          ^BSpline      ^pc
    '''
    if mapmode!='mom1':
        print('rc.RC() : WARNING: Did not select \'mom1\' as mapmode. Might get wonky.')
    rcmode = mapmode+'_'+data_mode       # RC is generated from <mapmode> data at <data_mode> resolution.
    diskfit_folder='diskfit_auto_'+rcmode+'/'
    if smooth!='universal':
        raise ValueError('You did not use \'universal\' as the \'smooth\' mode. \nWTF YOU DOING YA DINGUS ITS LITERALLY THERE IN THE FUNCTION NAME')
    
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('rc.RC() - Creating new galaxy object for '+name)
        gal = tools.galaxy(name)
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    R,vrot,R_e,vrot_e = gal.rotcurve(data_mode,mapmode)
    if mode=='diskfit':
        return R,vrot,R_e,vrot_e
    
    if mode.lower() in ['spl','spline']:
        if knots==-1:
            if name.upper()=='NGC1566':
                knots=12
            elif name.upper()=='NGC3521':
                knots=9
            elif name.upper()=='NGC3621':
                knots=7
                print('aye')
            else:
                print('rc.RC() - WARNING: No custom knots. Spline fit will be crappy')
                knots=3
        R,vrot_spl = rotcurve_smooth(R,vrot,R_e,vrot_e,smooth='spline',knots=knots)
        if returnknots==True:
            return R,vrot_spl,knots
        else:
            return R,vrot_spl
    
    R,vrot_s,vmax,rmax,A = rotcurve_smooth(R,vrot,R_e,vrot_e,smooth=smooth,returnparams=True)
    if mode.lower() in ['smooth','smoothed','urc','universal']:
        if returnparams==False:
            return R,vrot_s
        else:
            return R,vrot_s,vmax,rmax,A
        
    def read_rotcurve_npz(savefile):
        if os.path.isfile(path+'MCurc_save/'+savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz'):
            MCurc_data = np.load(path+'MCurc_save/'+savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz')
            params_MC     = MCurc_data['egg1'] # Can be LSQ, not necessarily just MC!
            params_smooth = MCurc_data['egg2']
            if savefile=='MC_':
                pos   = MCurc_data['egg3']
                prob  = MCurc_data['egg4']
                state = MCurc_data['egg5']
            else:
                pos, prob, state = [None]*3
        else:
            if mode.lower() in ['mc', 'mcurc']:
                print('rc.RC(): WARNING - MCurc_save/'+savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz does not exist!')
            params_MC, params_smooth, pos,prob,state = [None]*5
        return params_MC, params_smooth, pos,prob,state
    
    # Read saved MC-URC or LSQ-URC results
    mode_best = 'lsq'   # LSQ is given priority over MC.
                        # (!) Replace 'lsq' with 'mc' if you want that to change!
    if mode.lower() in ['mc', 'mcurc','best']:
        params_best, params_smooth, pos,prob,state = read_rotcurve_npz('MC_')
    if mode.lower() in ['lsq', 'lsqurc', 'ls', 'lsurc'] or (mode.lower() in ['best'] and (params_best is None or mode_best.lower() in ['lsq', 'lsqurc', 'ls', 'lsurc'])):
        if mode.lower() in ['best'] and mode_best.lower() in ['mc', 'mcurc']:
            print('         Attempting to find LSQurc instead.')
        params_best, params_smooth, pos,prob,state = read_rotcurve_npz('LSQ_')
        if params_best is not None and mode.lower() in ['best'] and mode_best.lower() in ['mc', 'mcurc']:
            print('         ... LSQ successful!')
    if mode.lower() not in ['lsq', 'lsqurc', 'ls', 'lsurc','mc', 'mcurc','best']:
        raise ValueError('"'+mode+'" is not a valid "mode"! See header for details.')
    
    # If MC/LSQ/both (depending on 'mode') failed:
    if params_best is None and mode.lower()=='best':
        print('         Both MCurc and LSQurc failed; reading smoothed rotcurve instead!')
        if returnparams==False:
            return R,vrot_s
        else:
            return R,vrot_s,vmax,rmax,A
    elif params_best is None and mode.lower()!='best':
        if returnparams==False:
            return R,None
        else:
            return R,None,None,None,None
        
    # If MC/LSQ/either (depending on 'mode') was successful:
    if [vmax,rmax/1000.,A]!=list(params_smooth):
        print('Current URC params (r in pc): '+str([vmax,rmax,A]))
        print('Saved URC params (r in kpc):  '+str(params_smooth))
        print('rc.RC() - WARNING: Current URC parameters do not match the ones saved in MCurc_save/'+name.upper()+'_'+data_mode+'_'+smooth+'.npz!')
        print('                   The saved file is outdated and needs to be re-generated.')

    # Get finalized rotcurve
    vmax_f,rmax_f,A_f,PA_f = params_best[0],params_best[1]*1000.,params_best[2],params_best[3]  # Convert kpc to pc
    
    # BSpline interpolation of vrot_f(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,urc(R,vmax_f,rmax_f,A_f),s=0,k=K)
    vrot_f = interpolate.BSpline(t,c,k, extrapolate=False)  # Final smoothed rotcurve, after MC improvements.

    if returnparams==False:
        return R,vrot_f
    else:
        return R,vrot_f,vmax_f,rmax_f,A_f


# Modified version of Erik's MCurc.galfit
def galfit(gal, data_mode='7m', MCMC=False,\
           path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
           path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
           masking='broad',\
           PA_over = None):
    '''
    Fits a URC model to the mom1, emom1 data that was used in DiskFit.
    (Requires DiskFit output.)
    
    Parameters:
    -----------
    gal : Galaxy
    data_mode='7m' : str
        '7m' or '12m'
    MCMC=False : bool
        Toggles whether to attempt MCMC sampling after LSQ fit
        is complete; however, leaving this off provided better
        results for the thesis.
    masking='broad' : bool
        'broad' or 'strict', matching the setting used in DiskFit.
        We used 'broad' in thesis for its better coverage.
    PA_over=None : Quantity
        Custom "initial guess" for the position angle, replacing
        the DiskFit output value.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('rc.galfit() - Creating new galaxy object for '+name)
        gal = tools.galaxy(name)
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    # Check if mom1 and emom1 BOTH have 7m+tp data.
    mapmode = 'mom1'     # As opposed to vpeak or something.
    smooth = 'universal' # As opposed to brandt or something.
    print('rc.galfit() : mapmode is assumed to be \'mom1\' and smooth is assumed to be \'universal\'.')
    mom1_name, emom1_name = dig.filename_get(name,data_mode,mapmode,force_same_res=False,masking=masking)
    # ^ The `force_same_res=False` causes filename_get() to leave emom1_name as 'None' if 7m+tp mom1 exists
    #    exists but 7m+tp emom1 does not. It basically checks whether error maps are available for the best
    #    mom1 map, which DiskFit was run on. 
    if emom1_name is not 'None':       # If data&error have matching maps, i.e. everything's good:
        mom1path = tools.mom1_get(gal,data_mode,return_mode='path',path7m=path7m,path12m=path12m,masking=masking)
        emom1path = tools.emom1_get(gal,data_mode,return_mode='path',path7m=path7m,path12m=path12m,masking=masking)
        mom1 = MCurc.read_moment(mom1path)
        emom1 = MCurc.read_moment(emom1path)
    else:
        print(name+' has 7m+tp data, but no 7m+tp error map! Attempting to use 7m data instead.')
        mom1_name, emom1_name = dig.filename_get(name,data_mode,mapmode,force_same_res=True,masking=masking)
        if emom1_name is not None:
            mom1path = (data_mode=='7m')*path7m+(data_mode=='12m')*path12m+mom1_name
            emom1path = (data_mode=='7m')*path7m+(data_mode=='12m')*path12m+emom1_name
            mom1 = MCurc.read_moment(mom1path)
            emom1 = MCurc.read_moment(emom1path)
        else:
            raise ValueError(name+' has no errormap without "tp" either, apparently. Run failed!')
            return
            # Skip this galaxy entirely! Can't do diddly squat without an error map.  
        
    # Get parameters from DiskFit output!
    with silence():
        gal2 = tools.galaxy(name.upper(),diskfit_output=True)
        Distance = gal2.distance.to(u.Mpc)
        Center = gal2.center_position
        Vsys = gal2.vsys
        PA = (gal2.position_angle.to(u.rad)).value
        Inc = (gal2.inclination.to(u.rad)).value
    if PA_over is not None:
        if isinstance(PA_over,u.Quantity):
            PA_over = (PA_over.to(u.rad)).value
        elif isinstance(PA_over,float) or isinstance(PA_over,int):
            PA_over = ((PA_over*u.deg).to(u.rad)).value
            print(PA_over)
        print('OVERWRITING PA: '+str(PA*u.rad.to(u.deg))+' -> '+str(PA_over*u.rad.to(u.deg)))
        PA = PA_over

    # Make sure PA is between 0 and 2pi
    while PA<0:
        PA = PA+2*np.pi
        print('NOTE: Had to increase PA by 2pi since it was less than 0')
    while PA>2*np.pi:
        PA = PA-2*np.pi
        print('NOTE: Had to decrease PA by 2pi since it was more than 2pi')

    # Get smoothed rotcurve! (This is only for checking whether the rotcurve has changed when
    #   the '.npz' savez file gets loaded in other functions)
    R, vrot, R_e, vrot_e = gal2.rotcurve(data_mode,mapmode)
    R,vrot_s,vmax,rmax,A = rotcurve_smooth(R,vrot,R_e,vrot_e,smooth,returnparams=True)
    rmax = rmax/1000.        # Convert from pc to kpc, as MCurc.py's "R" is in kpc.

    p = np.array([100, 2, 2])

    # Step 1 -- fit with no convolution and fixed PA
    pout  = lsq(MCurc.urcloss, p, args=[mom1, emom1, Center,
                                Distance, Vsys, Inc, PA],
                ftol=1e-4, kwargs={'convolve':False},
                loss='soft_l1', bounds=(np.array([50, 0.1, 0.1]),
                                        np.array([400, 10.0, 5])))

    # Step 2 -- fit with no convolution and float PA
    p = np.array([*pout.x, PA])
    pout2 = lsq(MCurc.urcloss, p, args=[mom1, emom1, Center,
                                Distance, Vsys, Inc, PA],
                ftol=1e-4, kwargs={'convolve':False},
                loss='soft_l1', bounds=(np.array([50, 0.1, 0.1, 0]),
                                        np.array([400, 10.0, 5, 2 * np.pi])))

    # Step 3 -- fit with convolution
    pout3 = lsq(MCurc.urcloss, pout2.x, args=[mom1, emom1, Center,
                                        Distance, Vsys, Inc, PA],
                ftol=1e-4, kwargs={'convolve': True},
                loss='soft_l1', bounds=(np.array([50, 0.1, 0.1, 0]),
                                        np.array([400, 10.0, 5, 2 * np.pi])))

    # pout3.x is an array of [vmax,rmax(kpc),A,PA(radians)].
    vmax_lsq,rmax_lsq,A_lsq,PA_lsq = pout3.x[0],pout3.x[1],pout3.x[2],pout3.x[3]
    
    MCurc.rcfig(pout2.x, mom1, emom1, 
          Center, Distance,
          Vsys, Inc, PA, p2=pout3.x, filename='MCurc_save/rcfit/'+name+'_rcfit.png')

    # Finally, run MCMC around the regression solution
    if MCMC:
        print('MC sampling beginning.')
        nwalkers = 20
        ndim = 4

        p = pout3.x

        p0 = np.zeros((nwalkers, ndim))
        p0[:, 0] = p[0] + np.random.randn(nwalkers)
        p0[:, 1] = p[1] + 0.01 * np.random.randn(nwalkers)
        p0[:, 2] = p[2] + 0.01 * np.random.randn(nwalkers)
        p0[:, 3] = p[3] + 0.01 * np.random.randn(nwalkers)

        arglist = [mom1, emom1, Center, Distance, Vsys, Inc, PA]
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        MCurc.lp_urc,
                                        args=arglist)

        # The step numbers should be adjusted... (originally 20, 20 for fast running, or 200,500 as Erik's values)
        pos, prob, state = sampler.run_mcmc(p0, 200)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, 500)
        
        vmax_mc = np.median(pos[:,0])
        rmax_mc = np.median(pos[:,1])
        A_mc    = np.median(pos[:,2])
        PA_mc   = np.median(pos[:,3])
        print('... '+name+' finished!')
    
        # Save the fitted URC parameters!
        np.savez('MCurc_save/MC_'+name.upper()+'_'+data_mode+'_'+smooth+'.npz',\
                 egg1=[vmax_mc,rmax_mc,A_mc,PA_mc],\
                 egg2=[vmax,rmax,A],\
                 egg3=pos,egg4=prob,egg5=state[1])
        return(sampler)
    else:
        print('... '+name+' finished!')
        
        np.savez('MCurc_save/LSQ_'+name.upper()+'_'+data_mode+'_'+smooth+'.npz',\
                 egg1=[vmax_lsq,rmax_lsq,A_lsq,PA_lsq],\
                 egg2=[vmax,rmax,A])
        return(None)
    
        # When saving to .npz, note that the 'simple' URC fits (from simply fitting the
        #   rotcurve to the model, without using the mom1 data) are saved at the same time.
        # This is so that RC() can detect if the DiskFit rotation curve has been modified,
        #   as the URC fit will no longer match what's saved in this .npz file.
