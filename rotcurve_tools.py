import numpy as np
import math

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from galaxies.galaxies import Galaxy

from scipy import ndimage, misc, interpolate, optimize
from scipy.interpolate import BSpline, make_lsq_spline
from pandas import DataFrame, read_csv
import pandas as pd

# Import my own code
import galaxytools as tools


def rotcurve(gal,data_mode='',mapmode='mom1',\
#              rcpath='/mnt/bigdata/PHANGS/OtherData/derived/Rotation_curves/'):
             rcpath='/media/jnofech/BigData/PHANGS/OtherData/derived/Rotation_curves/'):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
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
    
    # Do not include in galaxies.py!
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
        print('rotcurve : WARNING: Using old PHANGS 12m+7m rotcurves. Likely outdated!')
    else:
        raise ValueError('rotcurve : Invalid data_mode! Should be 7m, 12m, or hybrid.')
    rcmode = mapmode+'_'+data_mode       # RC is generated from <mapmode> data at <data_mode> resolution.
    folder='diskfit_auto_'+rcmode+'/'
    

    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc
    
    # Rotation Curves
    if data_mode.lower()=='phangs':
        fname = rcpath+name.lower()+"_co21_12m+7m+tp_RC.txt"
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    else:
        fname = rcpath+folder+name.lower()+"_co21_"+rcmode+"_RC_procedural.txt"
        fname_b = rcpath+'diskfit_manual_12m/'+name.lower()+"_co21_12m+7m_RC.txt"   # Backup data for 12m.
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
    
    # Adding a (0,0) data point to rotation curve
#     if R[0]!=0:
#         R = np.roll(np.concatenate((R,[0]),0),1)
#         vrot = np.roll(np.concatenate((vrot,[0]),0),1)

    # Check if rotcurve is valid
    if np.isnan(np.sum(vrot)):
        print('ERROR: Rotcurve failed to generate!')
        return [np.nan]*4

    # BSpline interpolation of vrot(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of vrot(R).
                                                            # 'vrot' is now a function, not an array.
    # Creating "higher-resolution" rotation curve
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    
    
    return R, vrot, R_e, vrot_e
    
def rotcurve_smooth(R,vrot,R_e,vrot_e=None,smooth='spline',knots=8,returnparams=False):
    '''
    Takes a provided rotation curve
    and smooths it based on one of
    several models.
    
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
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (an analytical model)
        'universal' (Persic & Salucci 1995)
    knots=8 : int
        Number of internal knots in BSpline of
        vrot, if smooth=='spline'.
    returnparams=False : bool
        Returns RC fit parameters based on the
        model.
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
        vrot = bspline(R,vrot(R),knots=knots,lowclamp=False)
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
    
def rotmap(gal,header,position_angle=None,inclination=None,data_mode='',mapmode='mom1'):
    '''
    Returns "observed velocity" map, and "radius
    map". (The latter is just to make sure that the
    code is working properly.)
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    header : astropy.io.fits.header.Header
        Header for the galaxy.
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
        gal = tools.galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    # data_mode, mapmode
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
        print('rotcurve : WARNING: Using old PHANGS 12m+7m rotcurves. Likely outdated!')
    else:
        raise ValueError('rotcurve : Invalid data_mode! Should be 7m, 12m, or hybrid.')
    
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
    R_1d, vrot,R_e,vrot_e = rotcurve(name,data_mode=data_mode,mapmode=mapmode)  # Creates "vrot" interpolation function, and 1D array of R.


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

    skycoord = gal.skycoord_grid(header=header)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(rad)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, rad, Dec, RA

def bspline(X,Y,knots=8,k=3,lowclamp=False, highclamp=False):
    '''
    Returns a BSpline interpolation function
    of a provided 1D curve.
    With fewer knots, this will provide a
    smooth curve that ignores local wiggles.
    
    Parameters:
    -----------
    X,Y : np.ndarray
        1D arrays for the curve being interpolated.
    knots : int
        Number of INTERNAL knots, i.e. the number
        of breakpoints that are being considered
        when generating the BSpline.
    k : int
        Degree of the BSpline. Recommended to leave
        at 3.
    lowclamp : bool
        Enables or disables clamping at the lowest
        X-value.
    highclamp : bool
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
    if lowclamp==True:
        w[0]=X.max()*1000000                    # Setting a high weight for the X.min() term.
    if highclamp==True:
        w[-1]=X.max()*1000000                   # Setting a high weight for the X.max() term.
    spl = make_lsq_spline(X, Y, t, k,w)
    
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

def linewidth_iso(gal,beam=None,smooth='spline',knots=8,data_mode='',mapmode='mom1'):
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
        Beam width, in deg.
        Will be found automatically if not
        specified. (NOT RECOMMENDED!)
    smooth='spline' : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (the analytical model)
        'universal' (Persic & Salucci 1995)
    knots=8 : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve, which
        is used in calculation of epicyclic
        frequency (and, therefore, sigma_gal).
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
        print('rc.linewidth_iso(): WARNING: Beam size found automatically. This is NOT recommended!')
        hdr = tools.hdr_get(gal)
        beam = hdr['BMAJ']
    beam = beam*u.deg.to(u.rad)                 # Beam size, in radians
    d = (gal.distance).to(u.pc)
    Rc = beam*d / u.rad                         # Beam size, in parsecs
    
    # Use "interp" to generate R, vrot (smoothed), k (epicyclic frequency).
    R, vrot, R_e, vrot_e = gal.rotcurve(data_mode=data_mode,mapmode=mapmode)
    R, vrot_s            = rotcurve_smooth(R,vrot,R_e,vrot_e,smooth=smooth,knots=knots)
    k = epicycles(R,vrot_s)
    
    # Calculate sigma_gal = kappa*Rc
    sigma_gal = k(R)*Rc

    # Removing nans and infs
    # (Shouldn't be anything significant-- just a "nan" at R=0.)
    index = np.arange(sigma_gal.size)
    R_clean = np.delete(R, index[np.isnan(sigma_gal)==True])
    sigma_gal_clean = np.delete(sigma_gal, index[np.isnan(sigma_gal)==True])
    sigma_gal = bspline(R_clean,sigma_gal_clean,knots=20)


    # Cubic Interpolation of sigma_gal
    #K=3     # Order of the BSpline
    #t,c,k = interpolate.splrep(R,sigma_gal,s=0,k=K)
    #sigma_gal_spline = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of sigma_gal(R).
    
    return sigma_gal
