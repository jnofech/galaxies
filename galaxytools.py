import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy.table import Table
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.coordinates import SkyCoord, Angle, FK5
from spectral_cube import SpectralCube, Projection
from radio_beam import Beam

from scipy import interpolate
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.stats import binned_statistic

from galaxies.galaxies import Galaxy
import rotcurve_tools as rc

import copy
import os

def galaxy(name):
    '''
    Creates Galaxy object.
    Features kinematic PA and a quick
    bandaid fix for missing 'vsys' values!
    '''
    gal = Galaxy(name.upper())
    gal.position_angle = PA_get(gal)
    gal.inclination    = incl_get(gal)
    
    if gal.vsys is None or np.isnan(gal.vsys):
        I_mom1a = mom1_get(gal,data_mode='12m',verbose=False)
        I_mom1b = mom1_get(gal,data_mode='7m',verbose=False)
        if I_mom1a is not None:
            gal.vsys = np.nanmean(I_mom1a)*u.km/u.s
        elif I_mom1b is not None:
            gal.vsys = np.nanmean(I_mom1b)*u.km/u.s
        else:
            print('WARNING: mom1 maps (7m, 12m+7m) missing. Galaxy object has no vsys.')
            gal.vsys = np.nan
    
    # Custom central coordinates, in case provided values are very incorrect
    if gal.name.lower()=='ic5332':
        # Central coords look more reasonable, but unsure of effect on rotcurve..
        gal.center_position = SkyCoord(353.603, gal.center_position.dec.value,\
                                       unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc1385':
        # Noticeable improvement on RC!
        gal.center_position = SkyCoord(54.371, -24.502,\
                                       unit=(u.deg, u.deg), frame='fk5')
#    if gal.name.lower()=='ngc1559':
#        # Central coords look more reasonable, and mom1 data is now in two tight "trends"
#        # rather than being everywhere. RC got worse, though.
#        gal.center_position = SkyCoord(64.405, -62.7855,\
#                                       unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc2775':
        # Central coords look more reasonable, and mom1 data is now focused into one "trend"
        # (with much more scatter) rather than being in two tighter trends. RC looks better.
        gal.center_position = SkyCoord(137.582, 7.037970066070557,\
                                       unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc4207':
        # Central coords look more reasonable, and mom1 data is focused more into one of
        # the two "trends". RC looks better, but the error bars are much, much worse.
        gal.center_position = SkyCoord(183.879, 9.584930419921875,\
                                   unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc4254':
        # Central coords look MUCH more reasonable.
        # Much of the mom1 data is still below zero for some reason. Improved, but nowhere near perfect.
        gal.center_position = SkyCoord(184.718, gal.center_position.dec.value,\
                                       unit=(u.deg, u.deg), frame='fk5')
#    if gal.name.lower()=='ngc4694':
#        # No effect whatsoever. Mom1 too low-res to tell if the coords are more reasonable, anyways.
#        gal.center_position = SkyCoord(192.064, 10.9838,\
#                                   unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc4731':
        # Central coords look somewhat more reasonable. RC is horribly jagged and dips into negatives,
        # but is still a huge improvement.
        gal.center_position = SkyCoord(192.750, -6.39,\
                                   unit=(u.deg, u.deg), frame='fk5')
#    if gal.name.lower()=='ngc4781':
#        # It got worse. Discard.
#        gal.center_position = SkyCoord(193.596, -10.5366,\
#                                       unit=(u.deg, u.deg), frame='fk5')
    if gal.name.lower()=='ngc5068':
        # Central seems better if you compare mom1 image to other NED images, but unsure if the
        # new coords are more reasonable. Mom1 data went from 2 trends to 1 trend+more scatter.
        # RC is somewhat jaggier, but still has very small errors.
        gal.center_position = SkyCoord(199.7033, -21.045,\
                                       unit=(u.deg, u.deg), frame='fk5')
    return gal
    
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
        filename_7mtp = name+'_'+data_mode+'+tp_co21_mom0.fits'    # 7m+tp mom0. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_mom0.fits'    # 7m mom0. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            I_mom0 = fits.getdata(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            print('No 7m+tp mom0 found. Using 7m mom0 instead.')
            I_mom0 = fits.getdata(path+filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = name+'_co21_'+data_mode+'+tp_mom0.fits'
        if os.path.isfile(path+filename):
            I_mom0 = fits.getdata(path+filename)
    else:
        print('WARNING: Invalid data_mode-- No mom0 was found!')
        I_mom0 = None
        return I_mom0
    if I_mom0 is None:
        print('WARNING: No mom0 was found!')
    return I_mom0

def mom1_get(gal,data_mode='',return_best=False, verbose=True,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    '''
    data_mode = '7m','12m','12m+7m'
    '''
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
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'  
    elif data_mode=='':
        if verbose==True:
            print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m' 

    # Get the mom1 file. In K km/s.
    I_mom1     = None
    I_mom1_7m  = None
    I_mom1_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_name = '7m'
        path = path7m
        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_mom1.fits'    # 7m+tp mom1. Ideal.
        filename_7m   = name+'_'+data_mode_name+   '_co21_mom1.fits'    # 7m mom1. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            I_mom1_7m = fits.open(path+filename_7mtp,mode='update')
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            if verbose==True:
                print('No 7m+tp mom1 found. Using 7m mom1 instead.')
            I_mom1_7m = fits.open(path+filename_7m,mode='update')
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m+tp'
        else:
            best_mom1_7m = 'None'
        best_mom1 = best_mom1_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m
        filename_12mtp = name+'_co21_'+data_mode_name+'+tp_mom1.fits'  # (?) Will all the new maps have '+tp'?
        best_mom1_12m = '12m+7m+tp'                                    # (?) ^
        best_mom1 = best_mom1_12m
        if os.path.isfile(path+filename_12mtp):
            I_mom1_12m = fits.open(path+filename_12mtp,mode='update')
            I_mom1 = I_mom1_12m[0].data
    if data_mode=='hybrid':
        # Fix both of their headers!
        for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
            del I_mom1_12m[0].header[kw]
            del I_mom1_7m[0].header[kw]
        for i in ['1','2','3']:
            for j in ['1', '2', '3']:
                del I_mom1_7m[0].header['PC'+i+'_'+j]
                del I_mom1_12m[0].header['PC0'+i+'_0'+j]
        # Reproject the 7m map to the 12m's dimensions!
        # Conveniently, the interpolation is also done for us.
        I_mom1_7m_modify = Projection.from_hdu(I_mom1_7m)
        I_mom1_7m = I_mom1_7m_modify.reproject(I_mom1_12m[0].header)
        # Convert to simple np arrays!
        I_mom1_12m, I_mom1_7m = I_mom1_12m[0].data, I_mom1_7m.value
        # COMBINE!
        I_mom1_mask = (np.isfinite(I_mom1_12m) + np.isfinite(I_mom1_7m)).astype('float')
        I_mom1_mask[I_mom1_mask == 0.0] = np.nan    # np.nan where _neither_ I_mom1_12m nor I_mom1_7m have data.
        I_mom1_hybrid = np.nan_to_num(I_mom1_12m) + np.isnan(I_mom1_12m)*np.nan_to_num(I_mom1_7m) + I_mom1_mask
        I_mom1 = I_mom1_hybrid
        best_mom1 = best_mom1_7m+' & '+best_mom1_12m
    if data_mode not in ['7m','12m+7m','hybrid']:
        print('WARNING: Invalid data_mode-- No mom1 was found!')
        I_mom1 = None
        return I_mom1
    
    if I_mom1 is None:
        if verbose==True:
            print('WARNING: No mom1 was found!')
    
    if return_best==True:
        return I_mom1, best_mom1
    else:
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
        filename_7mtp = name+'_'+data_mode+'+tp_co21_tpeak.fits'    # 7m+tp tpeak. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_tpeak.fits'    # 7m tpeak. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            I_tpeak = fits.getdata(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            print('No 7m+tp tpeak found. Using 7m tpeak instead.')
            I_tpeak = fits.getdata(path+filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = name+'_co21_'+data_mode+'+tp_tpeak.fits'
        if os.path.isfile(path+filename):
            I_tpeak = fits.getdata(path+filename)
    else:
        print('WARNING: Invalid data_mode-- No tpeak was found!')
        I_tpeak = None
        return I_tpeak
    if I_tpeak is None:
        print('WARNING: No tpeak was found!')
    return I_tpeak

def peakvels_get(gal,data_mode='',cube=None,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    # Returns a map of averaged ABSOLUTE VALUE of noise values.
    # This 'noisypercent' refers to the fraction of the 'v'-layers (0th axis in cube; around the 0th "sheet")
    #     that are assumed to be noise.
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

    # Get the cube, and spectral axis.
    if cube is None:
        cube = cube_get(gal,data_mode,path7m,path12m)
    spec = cube.spectral_axis

    # Find indices peaks from this.
    data = cube.unmasked_data[:]
    peak_indices = np.argmax(data,axis=0)   # Indices of temperature peaks in spectral axis.
    
    # Generate the 2D map of peak velocity!
    peakvels = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2]) * spec.unit
    for j in range(0,cube.shape[1]):
        for i in range(0,cube.shape[2]):
            peakvels[j,i] = spec[peak_indices[j,i]]
    peakvels[np.isnan(np.max(data,axis=0))] = np.nan  # Any pixel with >1 'np.nan' in its spectral axis is ignored.
    peakvels = peakvels.to(u.km/u.s)
    
    return peakvels.value

def noisemean_get(gal,data_mode='',cube=None,noisypercent=0.3,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    # Returns a map of averaged ABSOLUTE VALUE of noise values.
    # This 'noisypercent' refers to the fraction of the 'v'-layers (0th axis in cube; around the 0th "sheet")
    #     that are assumed to be noise.
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

    # Get the cube.
    if cube is None:
        cube = cube_get(gal,data_mode,path7m,path12m)
    
    # Consider parts of the cube that are presumed to have no signal, only noise.
    sheets = int(cube.shape[0] * noisypercent)    # Number of 2D "layers" that we're getting noise from.
    index_low = int(0+sheets/2)
    index_high = int(cube.shape[0]-sheets/2)

    # Find the mean of abs(noise) through all these "sheets".
    noise_low  = np.nanmean(np.abs(cube.unmasked_data[0:index_low].value),axis=0)    # Mean |noise|, 'below' signal.
    noise_high = np.nanmean(np.abs(cube.unmasked_data[index_high:-1].value),axis=0)  # Mean |noise|, 'above' signal.
    I_noise = np.nanmean([noise_low,noise_high],axis=0)
    
    return I_noise

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
        name+'_'+data_mode+   '_co21_mom0.fits',\
        name+'_'+data_mode+   '_co21_mom1.fits',\
        name+'_'+data_mode+   '_co21_tpeak.fits',\
        name+'_'+data_mode+'+tp_co21_mom0.fits',\
        name+'_'+data_mode+'+tp_co21_mom1.fits',\
        name+'_'+data_mode+'+tp_co21_tpeak.fits']:
            if os.path.isfile(path+filename):
                hdr = fits.getheader(path+filename)
                hdr_found = True
    if data_mode=='12m+7m':
        path = path12m
        for filename in [\
        name+'_co21_'+data_mode+'+tp_mom0.fits',\
        name+'_co21_'+data_mode+'+tp_mom1.fits',\
        name+'_co21_'+data_mode+'+tp_tpeak.fits']:
            if os.path.isfile(path+filename):
                hdr = fits.getheader(path+filename)
                hdr_found = True
    if hdr_found == False:
        print('WARNING: No header was found!')
        hdr = None
    return hdr

def sfr_get(gal,hdr=None,conbeam=None,res='7p5',band_uv='nuv',band_ir='w3',autocorrect=False,\
            path='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    '''
    Recommended: NUV band + WISE Band 3.
    
    The 'res' can be '7p5' or '15',
        i.e. 7.5" SFR data or 15" data.
    The band_uv can be 'fuv', 'nuv', or None.
    The band_ir can be 'w3' or 'w4', or None.
        For 7.5" (i.e. best) data, 'nuv'+'w3'
        are recommended.
    The 7.5" is better, but keep in mind
        that some cubes have beam widths
        too high to be convolved to such
        a high resolution. For these cases,
        stick with the 15" SFR maps.
    autocorrect=False : bool
        Toggles whether to revert to 15"
        data if 7.5" is missing.
        Recommended to DISABLE if you need
        cubes to be convolved to same res.
        as SFR maps.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
        
    # Convert the UV and IR maps to Msun/yr/kpc**2!
    if band_uv!=None:
        if band_uv.lower() in ['fuv','nuv']:
            uv_to_sfr = 1.04e-1
        else:
            raise ValueError('(galaxytools.sfr_get())  Invalid \'band_uv\'. Must be \'fuv\' or \'nuv\'!')
    if band_ir!=None:
        if band_ir.lower()=='w3':
            ir_to_sfr = 3.77e-3
        elif band_ir.lower()=='w4':
            ir_to_sfr = 3.24e-3
        else:
            raise ValueError('(galaxytools.sfr_get())  Invalid \'band_ir\'. Must be \'w3\' or \'w4\'!')
    
    # Get the map for each band!
    map_uv      = band_get(gal,hdr,band_uv,res,sfr_toggle=False)    # Galex NUV band.
    map_ir      = band_get(gal,hdr,band_ir,res,sfr_toggle=False)     # WISE3 band.
    
    # Actually generate the SFR map!
    if map_uv is not None and map_ir is not None:
        sfr     = (map_uv*uv_to_sfr + map_ir*ir_to_sfr).value   # Sum of SFR contributions, in Msun/yr/kpc**2.
    elif map_uv is None and band_uv==None and map_ir is not None and band_ir!=None:
        # If UV is intentionally missing:
#         print('(galaxytools.sfr_get())  WARNING: Only considering IR ('+band_ir+') component.')
        sfr     = (map_ir*ir_to_sfr).value             # SFR from just IR contribution.
    elif map_ir is None and band_ir==None and map_uv is not None and band_uv!=None:
        # If IR is intentionally missing:
#         print('(galaxytools.sfr_get())  WARNING: Only considering UV ('+band_uv+') component.')
        sfr     = (map_uv*uv_to_sfr).value             # SFR from just UV contribution.
    else:
        print('(galaxytools.sfr_get())  WARNING: No '+str(res)+'" '+str(band_uv)\
              +'+'+str(band_ir)+' SFR map was found!')
        sfr = None
    
    # Autocorrect with 15" version?
    if sfr is None and res=='7p5' and autocorrect==True:
        print('(galaxytools.sfr_get())  WARNING: Unable to get 7.5" SFR map! Reverting to 15" instead.')
        sfr = sfr_get(gal,hdr,conbeam,band_uv,band_ir,'15',False,path)
        return sfr

    if sfr is not None and conbeam!=None:
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
        filename_7mtp = name+'_'+data_mode+'+tp_co21_pbcorr_round_k.fits'    # 7m+tp cube. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_pbcorr_round_k.fits'    # 7m cube. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            cube = SpectralCube.read(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            print('No 7m+tp cube found. Using 7m cube instead.')
            cube = SpectralCube.read(path+filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename = name+'_co21_'+data_mode+'+tp_flat_round_k.fits'
        if os.path.isfile(path+filename):
            cube = SpectralCube.read(path+filename)
    else:
        print('WARNING: Invalid data_mode-- No cube was found!')
        cube = None
    if cube is None:
        print('WARNING: No cube was found!')
    return cube

def band_get(gal,hdr=None,band='',res='15',sfr_toggle=False,path='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    '''
    Returns the 'band' map, used to
    generate SFR map.
    
    Parameters:
    -----------
    gal : Galaxy or str
        Galaxy.
    hdr : fits.header.Header
        Header for the galaxy.
    band : str
        FUV = Galex FUV band (???-??? nm)
        NUV = Galex NUV band (150-220 nm)
        W3 = WISE Band 3 (12 µm data)
        W4 = WISE Band 4 (22 µm data)
    res='7p5' : str
        Resolution, in arcsecs.
        '7p5' - 7.5" resolution.
        '15'  - 15" resolution.
    sfr_toggle=False : bool
        Toggles whether to read the
        SFR contribution (Msun/yr/kpc^2)
        directly. Disabled by default
        since these files aren't always
        included.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    if band is None:
#         print('(galaxytools.band_get()) WARNING: No band selected! Returning None.')
        return None
    
    filename = path+name+'_'+(sfr_toggle*'sfr_')+band+'_gauss'+res+'.fits'
    if os.path.isfile(filename):
        map2d = Projection.from_hdu(fits.open(filename))        # Not necessarily in Msun/yr/kpc**2. Careful!
#         print(filename)
    else:
        print('(galaxytools.band_get()) WARNING: No map was found, for '+(sfr_toggle*'sfr_')+band+'_gauss'+res+'.fits')
        map2d = None
        return map2d
    
    if hdr!=None:
        map2d_final = map2d.reproject(hdr)
    else:
        map2d_final = map2d
    return map2d_final
    
def PA_get(gal):
    '''
    Gets the kinematic PA for the
    indicated galaxy. Used when
    photometric PA != kinematic PA.    
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        
    Returns:
    --------
    PA : Quantity (float*u.deg)
        Position angle, in degrees.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
#        gal = Galaxy(name.upper())
        gal = galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Get PA!
    PA = gal.position_angle / u.deg * u.deg
    
    
    # OVERRIDES
    # a) PA for galaxies that are missing PA values. Done by eyeballing!
    if gal.name.lower()=='ic5332':
        PA = 50.*u.deg
    if gal.name.lower()=='ngc3059':
        PA = 350.*u.deg
    if gal.name.lower()=='ngc3239':
        PA = None             # Not a disk galaxy.
    if gal.name.lower()=='ngc3596':
        PA = 85.*u.deg
    if gal.name.lower()=='ngc4303':
        PA = 320.*u.deg       # Set to 0 in galaxies.py, for some reason.
    if gal.name.lower()=='ngc4571':
        PA = 210.*u.deg
    if gal.name.lower()=='ngc4941':
        PA = 190.*u.deg
        
    # b) PA is off by 180 degrees.
    galaxies_180=['IC5273','NGC1300','NGC1365','NGC1385','NGC1511','NGC1512','NGC1559',\
                  'NGC1637','NGC1792','NGC2090','NGC2283','NGC2566','NGC2835','NGC3511','NGC4298',\
                  'NGC4535','NGC4731','NGC4781','NGC4826','NGC5042','NGC5134','NGC5530']
    if name.upper() in galaxies_180:
        if PA.value>180.:
            PA = PA-180.*u.deg
        else:
            PA = PA+180.*u.deg
            
    # c) PA for galaxies whose photometric PAs are just WAY off from kinematic PAs. Done by eyeballing!
    if gal.name.lower()=='ngc1433':
        PA = 190.*u.deg
    if gal.name.lower()=='ngc3507':
        PA = 50.*u.deg
    if gal.name.lower()=='ngc4540':
        PA = 10.*u.deg
    if gal.name.lower()=='ngc5128':
        PA = 310.*u.deg
    if gal.name.lower()=='ngc5330':
        PA = 315.*u.deg
    if gal.name.lower()=='ngc1317':
        PA = 215.*u.deg
    if gal.name.lower()=='ngc1566':
        PA = 205.*u.deg
    if gal.name.lower()=='ngc5068':
        PA = 335.*u.deg
    if gal.name.lower()=='ngc5643':
        PA = 315.*u.deg
    if gal.name.lower()=='ngc1385':
        PA = 170.*u.deg
    if gal.name.lower()=='ngc0685':
        PA = 100.*u.deg
    
    return PA
    
def incl_get(gal):
    '''
    Gets the inclination PA for the
    indicated galaxy, if the provided
    one happens to look a bit off.
    (Basically, just NGC1512 for now.)
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        
    Returns:
    --------
    incl : Quantity (float*u.deg)
        Inclination, in degrees.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Get incl!
    incl = gal.inclination / u.deg * u.deg
    
    
    # OVERRIDES
    # Done by eyeballing!
    if gal.name.lower()=='ngc1512':
        print('galaxytools.incl_get():  Overwrote inclination with an eyeballed value. May not be accurate!')
        incl = 45.*u.deg
    return incl
    
def info(gal,conbeam=None,data_mode='',sfr_band_uv='nuv',sfr_band_ir='w3',sfr_autocorrect=False):
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
    sfr_band_uv='nuv' : str
        Selects 'fuv' or 'nuv' as UV
        band for SFR.
    sfr_band_ir='w3' : str
        Selects 'w3' or 'w4' as IR
        band for SFR.
        NOTE: nuv+w3 recommended.
    sfr_autocorrect=False : bool
        Attempts to get a 15" SFR
        map if the 7.5" one fails.
        
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
    beam = hdr['BMAJ']                       # In degrees.
    beam_arcsec = (beam*u.deg).to(u.arcsec)  # In arcsec. We want this to be LOWER than the SFR map's 7.5"
                                             #    beamwidth (i.e. higher resolution), but this often fails
                                             #    and we need to use 15" SFR maps instead.
    
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
    
    # Choose appropriate resolution for SFR map, changing 'conbeam' to match it if necessary.
    res='7p5'
    if beam_arcsec > 7.5*u.arcsec and conbeam is not None:
        print('(galaxytools.info())     WARNING: Beam is '+str(beam_arcsec)+', and we want to convolve.')
        print('                                  This will use a 15" SFR map instead!')
        res='15'
    
    # Get SFR at this resolution.
    sfr = sfr_get(gal,hdr,res=res,band_uv=sfr_band_uv,band_ir=sfr_band_ir,autocorrect=sfr_autocorrect) 
    #     Not convolved yet, despite that being an option.
    
    if res=='7p5' and sfr is None and sfr_autocorrect==True:  # If 7.5" isn't found and we want to try lower res:
        print('(galaxytools.info())              Will attempt a 15" SFR map instead!')
        res='15'
        sfr = sfr_get(gal,hdr,res=res,band_uv=sfr_band_uv,band_ir=sfr_band_ir) # Try again with lower resolution.
    if res=='15' and sfr is not None:  # If a 15" SFR map was successful:
        if conbeam==7.5*u.arcsec:
            print('(galaxytools.info())     NOTE:    The 15" SFR map was successful! Changing conbeam from 7.5" to 15".')
            conbeam=15.*u.arcsec
    cube = cube_get(gal,data_mode)
        
        
    # CONVOLUTION, if enabled:
    if conbeam!=None:
        hdr,I_mom0, I_tpeak, cube = cube_convolved(gal,conbeam,data_mode) # CONVOLVED moments, with their cube.
        if sfr is not None:
            sfr = convolve_2D(gal,hdr,sfr,conbeam)  # Convolved SFR map.

    return hdr,beam,I_mom0,I_mom1,I_tpeak,cube,sfr
    
def depletion(Sigma=None,sfr=None):
    '''
    Returns 2D map of depletion time, 
        in years.
    
    Parameters:
    -----------
    Sigma : np.ndarray
        Map for surface density.
    sfr : np.ndarray
        2D map of the SFR, in Msun/kpc^2/yr.
        
    Returns:
    --------
    depletion : np.ndarray
        2D map of depletion time, in yr.
    '''    
    if Sigma is None or sfr is None:
        raise ValueError('galaxytools.depletion:  SFR or Sigma not specified.')
    # Calculating depletion time
    # Sigma is in Msun / pc^2.
    # SFR is in Msun / kpc^2 / yr.
    depletion = Sigma/(u.pc.to(u.kpc))**2/sfr
    return depletion

def rad_function2D(rad,radfunct,R=None):
    '''
    Converts a function of radius R
        into a 2D map corresponding
        to radius map "rad".
        
    Parameters:
    -----------
    rad : np.ndarray
        2D map of galaxy radii, in pc.
    radfunct : BSpline OR np.ndarray
        Function of radius that will
        be converted to 2D.
    R=None : np.ndarray
        1D array of galaxy radii, in pc.
        Only needed if radfunct is an
        array.
    '''
    if isinstance(radfunct,np.ndarray):
        print('galaxytools.rad_function2D:  Warning: Provided f(R) is an array. Will be converted to BSpline!')
        if R is None:
            raise ValueError('                             BUT: \'R\' must not be None, first of all.')
        # Convert it to a BSpline!
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,radfunct,s=0,k=K)
        radfunct = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of beta
    
    # Interpolating a 2D Array
    radfunct = radfunct(rad)
    return radfunct
        
def map2D_to_1D(rad,maps,stride=1):
    '''
    Make 2D maps (beta2D, depletion, etc)
        presentable, by removing NaN values,
        converting to 1D arrays, and skipping
        numerous points to avoid oversampling.
        Also sorts in order of ascending radius.
    NOTE: Take np.log10 of maps BEFORE cleaning!
    
    Parameters:
    -----------
    rad : np.ndarray
        2D map of galaxy radii, in pc.
    maps : list
        List of 2D maps that you want organized.
    stride=1 : int
        Numer of points to be stepped over,
        per... step.
    
    Returns:
    --------
    rad1D : np.ndarray
        1D array of radius, in ascending order
    maps1D : list
        List of 1D arrays, with values
        corresponding to rad1D
    '''
    # Making them 1D!
    rad1D = np.ravel(rad)
    for i in range(0,len(maps)):
        maps[i] = np.ravel(maps[i])
    
    # Cleaning the maps!
    index = np.arange(rad.size)
    index = index[ np.isfinite(rad1D*np.sum(maps,axis=0))]
    rad1D = rad1D[index][::stride]
    for i in range(0,len(maps)):
        maps[i] = maps[i][index][::stride]
    
    # Ordering the maps!
    import operator
    L = sorted(zip(rad1D.value,*maps), key=operator.itemgetter(0))
    rad1D,maps = np.array(list(zip(*L))[0])*u.pc, np.array(list(zip(*L))[1:])
    
    # Returning everything!
    return rad1D,maps

def sigmas(gal,hdr=None,I_mom0=None,I_tpeak=None,alpha=6.7,mode='',sigmode=''):
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
    # Moments
    if I_mom0 is None:
        print('galaxytools.sigmas(): WARNING: I_mom0 found automatically.')
        I_mom0 = mom0_get(gal)
    if I_tpeak is None:
        print('galaxytools.sigmas(): WARNING: I_tpeak found automatically.')
        I_tpeak = tpeak_get(gal)
    
    if mode=='':
        print('WARNING: No \'mode\' selected for galaxytools.sigmas()!\n        Will determine min and max \'rad\' values using PHANGS rotcurve.')
        mode='PHANGS'
    
    # (!!!) Ensure beforehand that the PA is kinematic, not photometric!
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
        
def means(array1D,bins=15):
    '''
    Finds the mean values of
    some 1D array, using bins
    of equal number.
    '''
    means = [np.nan]*bins
    index = np.arange(array1D.size)
    means, index_edges, binnumber = binned_statistic(index,array1D,statistic='mean',bins=bins)
    
    return means

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
        gal = galaxy(name.upper())
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
        gal = galaxy(name.upper())
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
