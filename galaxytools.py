import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy.table import Table
from astropy.convolution import convolve_fft, Gaussian2DKernel
from spectral_cube import SpectralCube, Projection
from radio_beam import Beam

from scipy import interpolate

from galaxies.galaxies import Galaxy
import rotcurve_tools as rc

import copy
import os

def mom0_get(gal,data_mode='',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'

    # Get the mom0 file. In K km/s.
    I_mom0=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = path+name+'_'+data_mode+'+tp_co21_mom0.fits'    # 7m+tp mom0. Ideal.
        filename_7m   = path+name+'_'+data_mode+   '_co21_mom0.fits'    # 7m mom0. Less reliable.
        if os.path.isfile(filename_7mtp):
            I_mom0 = fits.getdata(filename_7mtp)
        elif os.path.isfile(filename_7m):
            print('No 7m+tp mom0 found. Using 7m mom0 instead.')
            I_mom0 = fits.getdata(filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = path+name+'_co21_'+data_mode+'+tp_mom0.fits'
        if os.path.isfile(filename):
            I_mom0 = fits.getdata(filename)
    else:
        print('WARNING: Invalid data_mode-- No mom0 was found!')
        I_mom0 = None
        return I_mom0
    if I_mom0 is None:
        print('WARNING: No mom0 was found!')
    return I_mom0

def mom1_get(gal,data_mode='',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m' 

    # Get the mom1 file. In K km/s.
    I_mom1=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = path+name+'_'+data_mode+'+tp_co21_mom1.fits'    # 7m+tp mom1. Ideal.
        filename_7m   = path+name+'_'+data_mode+   '_co21_mom1.fits'    # 7m mom1. Less reliable.
        if os.path.isfile(filename_7mtp):
            I_mom1 = fits.getdata(filename_7mtp)
        elif os.path.isfile(filename_7m):
            print('No 7m+tp mom1 found. Using 7m mom1 instead.')
            I_mom1 = fits.getdata(filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = path+name+'_co21_'+data_mode+'+tp_mom1.fits'
        if os.path.isfile(filename):
            I_mom1 = fits.getdata(filename)
    else:
        print('WARNING: Invalid data_mode-- No mom1 was found!')
        I_mom1 = None
        return I_mom1
    if I_mom1 is None:
        print('WARNING: No mom1 was found!')
    return I_mom1

def tpeak_get(gal,data_mode='',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m' 

    # Get the tpeak file. In K km/s.
    I_tpeak=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = path+name+'_'+data_mode+'+tp_co21_tpeak.fits'    # 7m+tp tpeak. Ideal.
        filename_7m   = path+name+'_'+data_mode+   '_co21_tpeak.fits'    # 7m tpeak. Less reliable.
        if os.path.isfile(filename_7mtp):
            I_tpeak = fits.getdata(filename_7mtp)
        elif os.path.isfile(filename_7m):
            print('No 7m+tp tpeak found. Using 7m tpeak instead.')
            I_tpeak = fits.getdata(filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = path+name+'_co21_'+data_mode+'+tp_tpeak.fits'
        if os.path.isfile(filename):
            I_tpeak = fits.getdata(filename)
    else:
        print('WARNING: Invalid data_mode-- No tpeak was found!')
        I_tpeak = None
        return I_tpeak
    if I_tpeak is None:
        print('WARNING: No tpeak was found!')
    return I_tpeak

def hdr_get(gal,data_mode='',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'
    
    hdr = None
    hdr_found = False
    
    if data_mode=='7m':
        path = path7m
        for filename in [\
        path+name+'_'+data_mode+   '_co21_mom0.fits',\
        path+name+'_'+data_mode+   '_co21_mom1.fits',\
        path+name+'_'+data_mode+   '_co21_tpeak.fits',\
        path+name+'_'+data_mode+'+tp_co21_mom0.fits',\
        path+name+'_'+data_mode+'+tp_co21_mom1.fits',\
        path+name+'_'+data_mode+'+tp_co21_tpeak.fits']:
            if os.path.isfile(filename):
                hdr = fits.getheader(filename)
                hdr_found = True
    if data_mode=='12m+7m':
        path = path12m
        for filename in [\
        path+name+'_co21_'+data_mode+'+tp_mom0.fits',\
        path+name+'_co21_'+data_mode+'+tp_mom1.fits',\
        path+name+'_co21_'+data_mode+'+tp_tpeak.fits']:
            if os.path.isfile(filename):
                hdr = fits.getheader(filename)
                hdr_found = True
    if hdr_found == False:
        print('WARNING: No header was found!')
        hdr = None
    return hdr

def sfr_get(gal,hdr=None,conbeam=None,path='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    # Be sure to change 'gauss15' to 'gauss7p5' when better data becomes available!
#     filename = path+name+'_sfr_fuvw4_gauss15.fits'        # Galex FUV band + WISE Band 4 (old version)
    filename = path+name+'_sfr_nuvw3_gauss15.fits'        # Galex NUV band + WISE Band 3
    if os.path.isfile(filename):
        sfr_map = Projection.from_hdu(fits.open(filename))
    else:
        print('WARNING: No SFR map was found!')
        sfr_map = None
        return sfr_map
    
    if hdr!=None:
        sfr = sfr_map.reproject(hdr) # Msun/yr/kpc^2. See header.
                                     # https://www.aanda.org/articles/aa/pdf/2015/06/aa23518-14.pdf
    else:
        sfr = sfr_map

    if conbeam!=None:
        sfr = convolve_2D(gal,hdr,sfr,conbeam)  # Convolved SFR map.
    return sfr
            
def cube_get(gal,data_mode,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'

    # Spectral Cube
    cube=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = path+name+'_'+data_mode+'+tp_co21_pbcorr_round_k.fits'    # 7m+tp cube. Ideal.
        filename_7m   = path+name+'_'+data_mode+   '_co21_pbcorr_round_k.fits'    # 7m cube. Less reliable.
        if os.path.isfile(filename_7mtp):
            cube = SpectralCube.read(filename_7mtp)
        elif os.path.isfile(filename_7m):
            print('No 7m+tp cube found. Using 7m cube instead.')
            cube = SpectralCube.read(filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = path+name+'_co21_'+data_mode+'+tp_flat_round_k.fits'
        if os.path.isfile(filename):
            cube = SpectralCube.read(filename)
    else:
        print('WARNING: Invalid data_mode-- No cube was found!')
        cube = None
    if cube is None:
        print('WARNING: No cube was found!')
    return cube
    
def info(gal,conbeam=None,data_mode=''):
    '''
    Returns basic info from galaxies.
    Astropy units are NOT attached to outputs.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    conbeam=None : u.quantity.Quantity
        Width of the beam in pc or ",
        if you want the output to be
        convolved.
    data_mode='12m' or '7m' : str
        Chooses either 12m data or 7m
        data.
        
    Returns:
    --------
    hdr : fits.header.Header
        Header for the galaxy.
    beam : float
        Beam width, in deg.
    I_mom0 : np.ndarray
        0th moment, in K km/s.
    I_mom1 : np.ndarray
        Velocity, in km/s.
    I_tpeak : np.ndarray
        Peak temperature, in K.
    cube : SpectralCube
        Spectral cube for the galaxy.
    sfr : np.ndarray
        2D map of the SFR, in Msun/kpc^2/yr.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
        
    if data_mode == '7m':
        data_mode = '7m'
#         conbeam=None
#         print( 'WARNING: SFR maps come in 12m sizes only.') #(!!!) What about for all the new 15" maps?
#         print( 'WARNING: Convolution forcibly disabled.')
    elif data_mode in ['12m','12m+7m']:                     #(???) Do separate 12m, 12m+7m data exist?
        data_mode = '12m+7m'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'
    
    I_mom0 = mom0_get(gal,data_mode)
    I_mom1 = mom1_get(gal,data_mode)
    I_tpeak = tpeak_get(gal,data_mode)
    hdr = hdr_get(gal,data_mode)
    beam = hdr['BMAJ']                                                    # In degrees.
    
    # Fix the headers so WCS doesn't think that they're 3D!
    hdrcopy = copy.deepcopy(hdr)
    for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
        del hdrcopy[kw]
    for i in ['1','2','3']:
        for j in ['1', '2', '3']:
            if data_mode=='7m':
                del hdrcopy['PC'+i+'_'+j]
            else:
                del hdrcopy['PC0'+i+'_0'+j]
    hdr = hdrcopy
    
    sfr = sfr_get(gal,hdr)                      # Not convolved yet, despite that being an option.
    cube = cube_get(gal,data_mode)
        
        
    # CONVOLUTION, if enabled:
    if conbeam!=None:
        hdr,I_mom0, I_tpeak, cube = cube_convolved(gal,conbeam,data_mode) # CONVOLVED moments, with their cube.
#         sfr = sfr.reproject(hdr)                                  # It was reprojected already.
        sfr = convolve_2D(gal,hdr,sfr,conbeam)  # Convolved SFR map.
    else:
        sfr = sfr.value

    return hdr,beam,I_mom0,I_mom1,I_tpeak,cube,sfr
    
def beta_and_depletion(R,rad,Sigma,sfr,vrot_s):
    '''
    Returns depletion time, in years,
        and beta parameter (the index 
        you would get if the rotation 
        curve were a power function of 
        radius, e.g. vrot ~ R**(beta).
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of galaxy radii, in pc.
    rad : np.ndarray
        2D map of galaxy radii, in pc.
    Sigma : np.ndarray
        Map for surface density.
    sfr : np.ndarray
        2D map of the SFR, in Msun/kpc^2/yr.
    vrot_s : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s. Ideally smoothed.
        
    Returns:
    --------
    beta : np.ndarray
        2D map of beta parameter.
    depletion : np.ndarray
        2D map of depletion time, in yr.
    '''
    # Calculating depletion time
    # Sigma is in Msun / pc^2.
    # SFR is in Msun / kpc^2 / yr.
    depletion = Sigma/(u.pc.to(u.kpc))**2/sfr
    
    
    # Calculating beta
    dVdR = np.gradient(vrot_s(R),R)   # derivative of rotation curve;
    # Interpolating a 2D Array
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,dVdR,s=0,k=K)
    dVdR = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of dVdR
    beta = rad.value/vrot_s(rad) * dVdR(rad)
    depletion = Sigma/(u.pc.to(u.kpc))**2/sfr
    
    return beta, depletion

def beta_and_depletion_clean(beta,depletion,rad=None,stride=1):
    '''
    Makes beta and depletion time more easily
        presentable, by removing NaN values,
        converting to 1D arrays, and skipping
        numerous points to avoid oversampling.
    
    Parameters:
    -----------
    beta : np.ndarray
        2D map of beta parameter.
    depletion : np.ndarray
        2D map of depletion time, in yr.
    rad : np.ndarray
        2D map of galaxy radii, in pc.
    stride : int
        Numer of points to be skipped over.
    
    Returns:
    --------
    beta : np.ndarray
        1D array of beta parameter, with nans
        removed and points skipped.
    depletion : np.ndarray
        1D array of depletion time, with nans
        removed and points skipped.
    rad1D : np.ndarray
        1D array of radius, corresponding to
        beta and depletion
    '''
    # Making them 1D!
    beta = beta.reshape(beta.size)
    depletion = depletion.reshape(beta.size)
    if rad!=None:
        rad1D = rad.reshape(beta.size)
    
    # Cleaning the Rad/Depletion/Beta arrays!
    index = np.arange(beta.size)
    index = index[ np.isfinite(beta*np.log10(depletion)) ]
    beta = beta[index][::stride]
    depletion = depletion[index][::stride]   # No more NaNs or infs!
    if rad!=None:
        rad1D = rad1D[index][::stride]
    
    # Ordering the Rad/Depletion/Beta arrays!
    import operator
    if rad!=None:
        L = sorted(zip(np.ravel(rad1D.value),np.ravel(beta),np.ravel(depletion)), key=operator.itemgetter(0))
        rad1D,beta,depletion = np.array(list(zip(*L))[0])*u.pc, np.array(list(zip(*L))[1]),\
                               np.array(list(zip(*L))[2])
    else:
        L = sorted(zip(np.ravel(beta),np.ravel(depletion)), key=operator.itemgetter(0))
        beta,depletion = np.array(list(zip(*L))[0]), np.array(list(zip(*L))[1])
    
    # Returning everything!
    if rad!=None:
        return beta, depletion, rad1D
    else:
        return beta,depletion

def sigmas(gal,hdr=None,beam=None,I_mom0=None,I_tpeak=None,alpha=6.7,mode='',sigmode=''):
    '''
    Returns things like 'sigma' (line width, in km/s)
    or 'Sigma' (surface density) for a galaxy. The
    header, beam, and moment maps must be provided.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    hdr=None : astropy.io.fits.header.Header
        Header for the galaxy.
        Will be found automatically if not
        specified.
    beam=None : float
        Beam width, in deg.
        Will be found automatically if not
        specified.
    I_mom0=None : np.ndarray
        0th moment, in K km/s.
        Will be found automatically if not
        specified.
    I_tpeak=None : np.ndarray
        Peak temperature, in K.
        Will be found automatically if not
        specified.
    alpha=6.7 : float
        CO(2-1) to H2 conversion factor,
        in (Msun pc^-2) / (K km s^-1).
    mode='' : str
        'PHANGS'     - Uses PHANGS rotcurve.
                       (DEFAULT)
        'diskfit12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'diskfit7m'  - Uses fitted rotcurve from
                        7m data.
        This determines the min and max
        values of the output 'rad' array.
    sigmode='' : str
        'sigma' - returns linewidth.
        'Sigma' - returns H2 surface density.

    Returns:
    --------
    rad : np.ndarray
        Radius array.
    (s/S)igma : np.ndarray
        Maps for line width and H2 surface 
        density, respectively.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Header
    if hdr==None:
        print('galaxytools.sigmas(): WARNING: Header found automatically. Check that it\'s correct!')
        hdr = hdr_get(gal)
    # Beam width
    if beam==None:
        beam = hdr['BMAJ']
    if I_mom0 is None:
        print('galaxytools.sigmas(): I_mom0 found automatically.')
        I_mom0 = mom0_get(gal)
    if I_tpeak is None:
        print('galaxytools.sigmas(): I_tpeak found automatically.')
        I_tpeak = tpeak_get(gal)
    
    if mode=='':
        print('WARNING: No \'mode\' selected for galaxytools.sigmas()!\n        Will determine min and max \'rad\' values using PHANGS rotcurve.')
        mode='PHANGS'
    x, rad, x, x = gal.rotmap(header=hdr,mode=mode)
    d = gal.distance
    d = d.to(u.pc)                                          # Converts d from Mpc to pc.

    # Pixel sizes
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))*u.deg # The size of each pixel, in degrees. 
                                                                         # Ignore that third dimension; that's 
                                                                         # pixel size for the speed.
    pixsizes = pixsizes_deg[0].to(u.rad)                    # Pixel size, in radians.
    pcperpixel =  pixsizes.value*d                          # Number of parsecs per pixel.
    pcperdeg = pcperpixel / pixsizes_deg[0]

    # Beam
    beam = beam * pcperdeg                                  # Beam size, in pc

    # Line width, Surface density
    #alpha = 6.7  # (???) Units: (Msun pc^-2) / (K km s^-1)
    sigma = I_mom0 / (np.sqrt(2*np.pi) * I_tpeak)
    Sigma = alpha*I_mom0   # (???) Units: Msun pc^-2
    
    if sigmode=='sigma':
        return rad, sigma
    elif sigmode=='Sigma':
        return rad, Sigma
    else:
        print( "SELECT A MODE.")

def cube_convolved(gal,conbeam,data_mode='',\
                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
                  path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    '''
    Extracts the mom0 and tpeak maps from
        a convolved data cube.
    If pre-convolved mom0/tpeak/cube data
        already exists on the PHANGs Drive,
        then they will be used instead.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    conbeam : float
        Convolution beam width, in pc 
        OR arcsec. Must specify units!
    data_mode='12m' or '7m' : str
        Chooses either 12m data or 7m
        data.
        
    Returns:
    --------
    hdrc : fits.header.Header
        Header for the galaxy's convolved
        moment maps.
    I_mom0c : np.ndarray
        0th moment, in K km/s.
    I_tpeakc : np.ndarray
        Peak temperature, in K.
    cubec : SpectralCube
        Spectral cube for the galaxy,
        convolved to the resolution indicated
        by "conbeam".
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    resolutions = np.array([60,80,100,120,500,750,1000])*u.pc   # Available pre-convolved resolutions,
                                                                #    in PHANGS-ALMA-v1p0
    # Units for convolution beamwidth:
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        if conbeam not in resolutions:
            conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
        else:                            # Setting conbeam_filename to use int, for pre-convolved maps
            conbeam_filename = str(int(conbeam.to(u.pc).value))+'pc'
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
    else:
        raise ValueError("'conbeam' must have units of pc or arcsec.")

    # Read cube
    if data_mode=='7m':
        path = path7m
        filename = path+'cube_convolved/'+name.lower()+'_7m_co21_pbcorr_round_k_'\
                                                       +conbeam_filename+'.fits'
        # May be '7m' or '7m+tp', but we'll just name them all as '7m' for simplicity.
        if os.path.isfile(filename):
            cubec = SpectralCube.read(filename)
            cubec.allow_huge_operations=True
        else:
            raise ValueError(filename+' does not exist.')
        I_mom0c = cubec.moment0().to(u.K*u.km/u.s)
        I_tpeakc = cubec.max(axis=0).to(u.K)
        hdrc = I_mom0c.header
    elif data_mode in ['12m','12m+7m']:
        path = path12m
        if conbeam not in resolutions:
            filename = path+'cube_convolved/'+name.lower()+'_co21_12m+7m+tp_flat_round_k_'\
                                                           +conbeam_filename+'.fits'
            if os.path.isfile(filename):
                cubec = SpectralCube.read(filename)
                cubec.allow_huge_operations=True
            else:
                raise ValueError(filename+' does not exist.')
            I_mom0c = cubec.moment0().to(u.K*u.km/u.s)
            I_tpeakc = cubec.max(axis=0).to(u.K)
            hdrc = I_mom0c.header
        else:    # If pre-convolved 3D data (mom0, tpeak, cube) exist:
            I_mom0c  = fits.getdata('phangsdata/'+name.lower()+'_co21_12m+7m+tp_mom0_'+conbeam_filename+'.fits')*u.K*u.km/u.s
            I_tpeakc = fits.getdata('phangsdata/'+name.lower()+'_co21_12m+7m+tp_tpeak_'+conbeam_filename+'.fits')*u.K
            filename = 'phangsdata/'+name.lower()+'_co21_12m+7m+tp_flat_round_k_'+conbeam_filename+'.fits'
            if os.path.isfile(filename):
                cubec = SpectralCube.read(filename)
                cubec.allow_huge_operations=True
            else:
                raise ValueError(filename+' does not exist.')
            print( "IMPORTANT NOTE: This uses pre-convolved .fits files from Drive.")
            I_mom0c_DUMMY = cubec.moment0().to(u.K*u.km/u.s)
            hdrc = I_mom0c_DUMMY.header
    else:
        print('ERROR: No data_mode selected in galaxytools.convolve_cube()!')
        
    return hdrc,I_mom0c.value, I_tpeakc.value, cubec

def convolve_2D(gal,hdr,map2d,conbeam):
    '''
    Returns 2D map (e.g. SFR), convolved 
    to a beam width "conbeam".
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    hdr : fits.header.Header
        Header for the galaxy. Does
        NOT need to be convolved!
        Only needed for pixel sizes.
    map2d : np.ndarray
        The map (e.g. SFR) that needs to 
        be convolved.
    conbeam : float
        Convolution beam width, in pc 
        OR arcsec. Must specify units!
        The actual width of the Gaussian
        is conbeam/np.sqrt(8.*np.log(2)).

    Returns:
    --------
    map2d_convolved : np.ndarray
        The same map, convolved.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        conbeam_width = conbeam.to(u.pc)                         # Beam width, in pc.
        conbeam_angle = conbeam / gal.distance.to(u.pc) * u.rad  # Beam width, in radians.
        conbeam_angle = conbeam_angle.to(u.deg) / np.sqrt(8.*np.log(2)) # ..., in degrees, now as an
                                                                        #   actual Gaussian stdev.
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_angle = conbeam.to(u.deg) / np.sqrt(8.*np.log(2))# Beam width, in degrees, now as an
                                                                 #          actual Gaussian stdev.
    else:
        raise ValueError("'conbeam' must have units of pc or arcsec.")
    
    
    # Convert beam width into pixels, then feed this into a Gaussian-generating function.
    
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))[0]*u.deg # The size of each pixel, in deg.
    conbeam_pixwidth = conbeam_angle / pixsizes_deg  # Beam width, in pixels.
#     print( "Pixel width of beam: "+str(conbeam_pixwidth)+" pixels.")
    
    gauss = Gaussian2DKernel(conbeam_pixwidth)
    map2d_convolved = convolve_fft(map2d,gauss,normalize_kernel=True)
    return map2d_convolved

def convolve_cube(gal,cube,conbeam,data_mode='',\
                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
                  path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    '''
    Convolves a cube over a given beam, and
    then generates and returns the moment
    maps. The convolved cube is saved as well.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    cube : SpectralCube
        Spectral cube for the galaxy.
    conbeam : float
        Beam width, in pc OR arcsec.
        Must specify units!
    data_mode='12m' or '7m' : str
        Chooses where to save the output
        file, based on the selected data.
            
    Returns:
    --------
    cubec : SpectralCube
        Spectral cube for the galaxy,
        convolved to the resolution indicated
        by "conbeam".
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
        
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        conbeam_width = conbeam.to(u.pc)                     # Beam width in pc.
        conbeam_angle = conbeam / gal.distance.to(u.pc) * u.rad
        conbeam_angle = conbeam_angle.to(u.arcsec)
        conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_angle = conbeam.to(u.arcsec)                 # Beam width in arcsec.
        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
    else:
        raise ValueError("'beam' must have units of pc or arcsec.")
    
    bm = Beam(major=conbeam_angle,minor=conbeam_angle)    # Actual "beam" object, used for convolving cubes
    print(bm)
    
    # Convolve the cube!
    cube = cube.convolve_to(bm)
    
    # Never convolve the cube again!
    if data_mode=='7m':
        path = path7m
        filename = path+'cube_convolved/'+name.lower()+'_7m_co21_pbcorr_round_k_'\
                                                       +conbeam_filename+'.fits'
        # May be '7m' or '7m+tp', but we'll just name them all as '7m' for simplicity.
    elif data_mode in ['12m','12m+7m']:
        path = path12m
        filename = path+'cube_convolved/'+name.lower()+'_co21_12m+7m+tp_flat_round_k_'\
                                                       +conbeam_filename+'.fits'
    print('Saving convolved cube to... '+filename)
    if os.path.isfile(filename):
        os.remove(filename)
        print(filename+" has been overwritten.")
    cube.write(filename)
    
    return cube

def gaussian(beam_pixwidth):
#    ____  ____   _____  ____  _      ______ _______ ______ 
#   / __ \|  _ \ / ____|/ __ \| |    |  ____|__   __|  ____|
#  | |  | | |_) | (___ | |  | | |    | |__     | |  | |__   
#  | |  | |  _ < \___ \| |  | | |    |  __|    | |  |  __|  
#  | |__| | |_) |____) | |__| | |____| |____   | |  | |____ 
#   \____/|____/|_____/ \____/|______|______|  |_|  |______|
#
# astropy's Gaussian2DKernel does the same job, except better and with more options.

    '''
    Returns a square 2D Gaussian centered on
    x=y=0, for a galaxy "d" pc away.
    
    Parameters:
    -----------
    beam : float
        Desired width of gaussian, in pc.
    d : float
        Distance to galaxy, in pc.
        
    Returns:
    --------
    gauss : np.ndarray
        2D Gaussian with width "beam".    
    '''
    axis = np.linspace(-4*beam_pixwidth,4*beam_pixwidth,int(beam_pixwidth*8))
    x, y = np.meshgrid(axis,axis)
    d = np.sqrt(x*x+y*y)
    
    sigma, mu = beam_pixwidth, 0.0
    g = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) / (sigma*np.sqrt(2.*np.pi)))
    return g
