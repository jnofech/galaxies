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
import diskfit_input_generator as dig

import copy
import os
import csv

def galaxy(name,customPA=True,custominc=True,customcoords='phil',\
           diskfit_output=False,data_mode='7m',mapmode='mom1'):
    '''
    Creates Galaxy object.
    Features kinematic PA and a quick
    bandaid fix for missing 'vsys' values!
    
    Parameters:
    -----------
    customPA=True : bool
        Can also be a string, 'LSQ' or 'MC', to
        grab fitted PA from the LSQ-URC or MC-URC.
        Requires data_mode='7m', mapmode='mom1'.
    custominc=True : bool
    customcoords='phil' : bool OR str
        customcoords='p','phil','philipp':
        enables Philipp's custom central coords
    diskfit_output=False : bool
        Reads fitted PA, inc, vsys, coords from
        DiskFit, "overriding" the custom params
        above
    ^ If diskfit_output==True:
        data_mode : str
            '7m' : uses 7m data
            '12m' : uses 12m data
            'hybrid' : uses a combination of both
        mapmode : str
            'mom1'  : uses mom1 data
            'vpeak' : uses peakvels map
        
    '''
    gal = Galaxy(name.upper())
    if isinstance(customPA,str):
        # Check if data_mode and mapmode are good:
        if (data_mode=='7m' and mapmode=='mom1')==False:
            raise ValueError('tools.galaxy() : Cannot use "customPA='+customPA+'" unless data_mode=\'7m\', mapmode=\'mom1\'.')
        if customPA.lower() in ['lsq', 'lsqurc', 'ls', 'lsurc']:
            MCurc_savefile='LSQ_'
        elif customPA.lower() in ['mc', 'urc', 'mcurc']:
            MCurc_savefile='MC_'
        else:
            raise ValueError('"'+customPA+'" is not a valid "customPA"! See header for details.')
        smooth = 'universal'
        print('tools.galaxy(customPA='+customPA+') : smooth=universal is assumed.')
        if os.path.isfile('MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz'):
            MCurc_data = np.load('MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz')
            params_MC     = MCurc_data['egg1']
            PA_MC = (params_MC[3]*u.rad).to(u.deg)
            # ^ This will overwrite any other PA at the end of this function.
        else:
            print('rc.MC(): WARNING - MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz does not exist!')
            customPA = True
    if customPA==True:
        gal.position_angle  = PA_get(gal)
        
    if custominc==True:
        gal.inclination     = incl_get(gal)
    if customcoords==True or customcoords=='True' or customcoords=='true':
        gal.center_position = coords_get(gal)
    elif isinstance(customcoords,str):
        if customcoords.lower() in ['p','phil','philipp']:
            gal.center_position = coords_philipp_get(gal)
        else:
            print('tools.galaxy() : WARNING: `customcoords` invalid! Disabling custom coordinates. ')        
    
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
    
    if diskfit_output==True:
        if data_mode == '7m+tp':
            print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
            data_mode = '7m'
        elif data_mode in ['12m','12m+7m']:
            data_mode = '12m+7m'  
        elif data_mode in ['12m+tp','12m+7m+tp']:
            print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
            data_mode = '12m+7m'
        elif data_mode.lower() in ['both','hybrid']:
            data_mode = 'hybrid'

        if mapmode in ['mom1']:
            mapmode='mom1'
        elif mapmode in ['peakvels','vpeak']:
            mapmode='peakvels'
        else:
            print('No mapmode set. Defaulted to "mom1".')
            data_mode = 'mom1' 
        rcmode = mapmode+'_'+data_mode       # RC is generated from <mapmode> data at <data_mode> resolution.
        diskfit_folder='diskfit_auto_'+rcmode+'/'
        xcen_out,ycen_out,PA_out,eps_out,incl_out,vsys_out,bar_PA_out = dig.read_all_outputs(gal,\
                                                                    'params',diskfit_folder,True)
        RAcen, Deccen = pixels_to_wcs(gal,data_mode,xcen_out,ycen_out)
        # Save output values into galaxy:
        gal.center_position = SkyCoord(RAcen, Deccen,\
                                       frame='fk5',unit=(u.degree, u.degree))
        gal.position_angle = PA_out
        gal.inclination = incl_out
        gal.vsys = vsys_out
    
    if isinstance(customPA,str):
        gal.position_angle = PA_MC
    return gal
    
def mom0_get(gal,data_mode='',\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/broad_maps/'):
    '''
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to mom0 file
        'hdu': returns fits.open(path)
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'

    # Get the mom0 file. In K km/s.
    I_mom0=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = name+'_'+data_mode+'+tp_co21_mom0.fits'    # 7m+tp mom0. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_mom0.fits'    # 7m mom0. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_mom0_hdu = fits.open(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            print('No 7m+tp mom0 found. Using 7m mom0 instead.')
            I_mom0_hdu = fits.open(path+filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename_12mtp = name+'_'+data_mode+'+tp_co21_broad_mom0.fits'    # 12m+tp mom0. Ideal.
        filename_12m   = name+'_'+data_mode+   '_co21_broad_mom0.fits'    # 12m mom0. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_mom0_hdu = fits.open(path+filename_12mtp)
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            print('No 12m+7m+tp mom0 found. Using 12m+7m mom0 instead.')
            I_mom0_hdu = fits.open(path+filename_12m)
    else:
        print('WARNING: Invalid data_mode-- No mom0 was found!')
        I_mom0 = None
        return I_mom0
    if I_mom0_hdu is None:
        print('WARNING: No mom0 was found!')
        finalpath=None
        return I_mom0_hdu
    
    if return_mode=='data':
        return I_mom0_hdu[0].data
    elif return_mode=='path':
        return finalpath
    elif return_mode in ['hdu','hdul']:
        return I_mom0_hdu
    else:
        print('tools.mom0_get() : Invalid "return_mode"! Must be "data", "path", or "hdu".')

def mom1_get(gal,data_mode='',return_best=False, verbose=True,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/broad_maps/',\
             folder_hybrid='jnofech_mom1_hybrid/'):
    '''
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to mom1 file
        'hdu': returns fits.open(path)
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'
        
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
            finalpath = path+filename_7mtp
            I_mom1_7m = fits.open(path+filename_7mtp,mode='update')
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m+tp'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            if verbose==True:
                print('No 7m+tp mom1 found. Using 7m mom1 instead.')
            I_mom1_7m = fits.open(path+filename_7m,mode='update')
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m'
        else:
            best_mom1_7m = 'None'
        I_mom1_hdu = I_mom1_7m
        best_mom1 = best_mom1_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m
        filename_12mtp = name+'_'+data_mode_name+'+tp_co21_broad_mom1.fits'    # 7m+tp mom1. Ideal.
        filename_12m   = name+'_'+data_mode_name+   '_co21_broad_mom1.fits'    # 7m mom1. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_mom1_12m = fits.open(path+filename_12mtp,mode='update')
            I_mom1 = I_mom1_12m[0].data
            best_mom1_12m='12m+7m+tp'      # Keeps track of whether 12m or 12m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            if verbose==True:
                print('No 12m+7m+tp mom1 found. Using 12m+7m mom1 instead.')
            I_mom1_12m = fits.open(path+filename_12m,mode='update')
            I_mom1 = I_mom1_12m[0].data
            best_mom1_12m='12m+7m'
        else:
            finalpath = None
            best_mom1_12m = 'None'
        I_mom1_hdu = I_mom1_12m
        best_mom1 = best_mom1_12m
    if data_mode=='hybrid':
        # Fix both of their headers!
        for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
            del I_mom1_7m[0].header[kw]
            del I_mom1_12m[0].header[kw]
        for i in ['1','2','3']:
            for j in ['1', '2', '3']:
                del I_mom1_7m[0].header['PC'+i+'_'+j]
                del I_mom1_12m[0].header['PC'+i+'_'+j]
        hdr = I_mom1_12m[0].header
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
        best_mom1 = 'hybrid_'+best_mom1_7m+'&'+best_mom1_12m
        
        # SAVE!
        hdr['BUNIT'] = 'km  s-1 '  # Write this instead of 'KM/S  '.
        # Save header and data into a .fits file, if specified!
        hdu      = [fits.PrimaryHDU(I_mom1_hybrid,header=hdr),'Dummy list entry, so that I_mom1=hdu[0].data.']
        filename = name+'_co21_'+best_mom1+'_mom1.fits'
        path = path12m+folder_hybrid
        finalpath = path+filename
        if os.path.isfile(path+filename)==False:
            print(path+filename)
            hdu[0].writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
        I_mom1_hdu = hdu

    if data_mode not in ['7m','12m+7m','hybrid']:
        print('WARNING: Invalid data_mode-- No mom1 was found!')
        I_mom1 = None
        return I_mom1
    if I_mom1 is None:
        if verbose==True:
            print('WARNING: No mom1 was found!')
        return I_mom1
    if return_best==True:
        return I_mom1, best_mom1
    else:
        if return_mode=='data':
            return I_mom1
        elif return_mode=='path':
            return finalpath
        elif return_mode in ['hdu','hdul']:
            return I_mom1_hdu
        else:
            print('tools.mom1_get() : Invalid "return_mode"! Must be "data", "path", or "hdu".')
    
    
def emom1_get(gal,data_mode='',return_best=False, verbose=True,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/broad_maps/',\
             folder_hybrid='jnofech_mom1_hybrid/'):
    '''
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to emom1 file
        'hdu': returns fits.open(path)
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'
        
    # Get the emom1 file. In K km/s.
    I_emom1     = None
    I_emom1_7m  = None
    I_emom1_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_name = '7m'
        path = path7m
        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_emom1.fits'    # 7m+tp emom1. Ideal.
        filename_7m   = name+'_'+data_mode_name+   '_co21_emom1.fits'    # 7m emom1. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_emom1_7m = fits.open(path+filename_7mtp,mode='update')
            I_emom1 = I_emom1_7m[0].data
            best_emom1_7m='7m+tp'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            if verbose==True:
                print('No 7m+tp emom1 found. Using 7m emom1 instead.')
            I_emom1_7m = fits.open(path+filename_7m,mode='update')
            I_emom1 = I_emom1_7m[0].data
            best_emom1_7m='7m'
        else:
            best_emom1_7m = 'None'
        I_emom1_hdu = I_emom1_7m
        best_emom1 = best_emom1_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m
        filename_12mtp = name+'_'+data_mode_name+'+tp_co21_broad_emom1.fits'    # 7m+tp emom1. Ideal.
        filename_12m   = name+'_'+data_mode_name+   '_co21_broad_emom1.fits'    # 7m emom1. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_emom1_12m = fits.open(path+filename_12mtp,mode='update')
            I_emom1 = I_emom1_12m[0].data
            best_emom1_12m='12m+7m+tp'      # Keeps track of whether 12m or 12m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            if verbose==True:
                print('No 12m+7m+tp emom1 found. Using 12m+7m emom1 instead.')
            I_emom1_12m = fits.open(path+filename_12m,mode='update')
            I_emom1 = I_emom1_12m[0].data
            best_emom1_12m='12m+7m'
        else:
            finalpath = None
            best_emom1_12m = 'None'
        I_emom1_hdu = I_emom1_12m
        best_emom1 = best_emom1_12m
    if data_mode=='hybrid':
        # Fix both of their headers!
        for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
            del I_emom1_7m[0].header[kw]
            del I_emom1_12m[0].header[kw]
        for i in ['1','2','3']:
            for j in ['1', '2', '3']:
                del I_emom1_7m[0].header['PC'+i+'_'+j]
                del I_emom1_12m[0].header['PC'+i+'_'+j]
        hdr = I_emom1_12m[0].header
        # Reproject the 7m map to the 12m's dimensions!
        # Conveniently, the interpolation is also done for us.
        I_emom1_7m_modify = Projection.from_hdu(I_emom1_7m)
        I_emom1_7m = I_emom1_7m_modify.reproject(I_emom1_12m[0].header)
        # Convert to simple np arrays!
        I_emom1_12m, I_emom1_7m = I_emom1_12m[0].data, I_emom1_7m.value
        # COMBINE!
        I_emom1_mask = (np.isfinite(I_emom1_12m) + np.isfinite(I_emom1_7m)).astype('float')
        I_emom1_mask[I_emom1_mask == 0.0] = np.nan    # np.nan where _neither_ I_emom1_12m nor I_emom1_7m have data.
        I_emom1_hybrid = np.nan_to_num(I_emom1_12m) + np.isnan(I_emom1_12m)*np.nan_to_num(I_emom1_7m) + I_emom1_mask
        I_emom1 = I_emom1_hybrid
        best_emom1 = 'hybrid_'+best_emom1_7m+'&'+best_emom1_12m
        
        # SAVE!
        hdr['BUNIT'] = 'km  s-1 '  # Write this instead of 'KM/S  '.
        # Save header and data into a .fits file, if specified!
        hdu      = [fits.PrimaryHDU(I_emom1_hybrid,header=hdr),'Dummy list entry, so that I_emom1=hdu[0].data.']
        filename = name+'_co21_'+best_emom1+'_emom1.fits'
        path = path12m+folder_hybrid
        finalpath = path+filename
        if os.path.isfile(path+filename)==False:
            print(path+filename)
            hdu[0].writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
        I_emom1_hdu = hdu

    if data_mode not in ['7m','12m+7m','hybrid']:
        print('WARNING: Invalid data_mode-- No emom1 was found!')
        I_emom1 = None
        return I_emom1
    if I_emom1 is None:
        if verbose==True:
            print('WARNING: No emom1 was found!')
        return I_emom1
    if return_best==True:
        return I_emom1, best_emom1
    else:
        if return_mode=='data':
            return I_emom1
        elif return_mode=='path':
            return finalpath
        elif return_mode in ['hdu','hdul']:
            return I_emom1_hdu
        else:
            print('tools.emom1_get() : Invalid "return_mode"! Must be "data", "path", or "hdu".')

def tpeak_get(gal,data_mode='',\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/broad_maps/'):
    '''
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to tpeak file
        'hdu': returns fits.open(path)
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
            finalpath = path+filename_7mtp
            I_tpeak_hdu = fits.open(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            print('No 7m+tp tpeak found. Using 7m tpeak instead.')
            I_tpeak_hdu = fits.open(path+filename_7m)
    elif data_mode=='12m+7m':
        path = path12m
        filename_12mtp = name+'_'+data_mode+'+tp_co21_broad_tpeak.fits'    # 12m+tp tpeak. Ideal.
        filename_12m   = name+'_'+data_mode+   '_co21_broad_tpeak.fits'    # 12m tpeak. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_tpeak_hdu = fits.open(path+filename_12mtp)
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            print('No 12m+7m+tp tpeak found. Using 12m+7m tpeak instead.')
            I_tpeak_hdu = fits.open(path+filename_12m)
        else:
            print('tools.tpeak_get(): tpeak maps missing. Calculating tpeak directly from cube.')
            cube = (cube_get(gal,data_mode).unmasked_data[:]).to(u.K).value
            finalpath = None
            I_tpeak = cube.max(axis=0)
            hdr = hdr_get(gal,data_mode,dim=2)
            hdr['BUNIT'] = 'K'
            I_tpeak_hdu = [fits.PrimaryHDU(I_tpeak,header=hdr),'Dummy list entry, so that I_tpeak=I_tpeak_hdu[0].data.']
    else:
        print('WARNING: Invalid data_mode-- No tpeak was found!')
        I_tpeak = None
        return I_tpeak
    if I_tpeak_hdu is None:
        print('WARNING: No tpeak was found!')
        return I_tpeak_hdu

    if return_mode=='data':
        return I_tpeak_hdu[0].data
    elif return_mode=='path':
        return finalpath
    elif return_mode in ['hdu','hdul']:
        return I_tpeak_hdu
    else:
        print('tools.tpeak_get() : Invalid "return_mode"! Must be "data", "path", or "hdu".')

def noise_get(gal,data_mode='',cube=None,noisypercent=0.15,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/broad_maps/'):
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
    data = cube.unmasked_data[:]
    
    # Consider parts of the cube that are presumed to have no signal, only noise.
    sheets = int(cube.shape[0] * noisypercent)           # Number of 2D "layers" that we're getting noise from.
    data_slice = np.roll(data,int(sheets/2),axis=0)[:sheets]  # A slice of data containing only noise (ideally).

    # Find the stdev of many noise "sheets".
    I_noise = np.std(data_slice,axis=0).value
    
    return I_noise

def hdr_get(gal,data_mode='',dim=3,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    
    hdr = None
    hdr_found = False
    
    cube = cube_get(gal,data_mode,False,path7m,path12m)
    if cube is not None:
        if dim in [3,'3d','3D']:
            hdr = cube.header
        elif dim in [2,'2d','2D']:
            hdr = cube[0].header
        else:
            raise ValueError ('hdr_get() : Specify number of dimensions!')
        hdr_found = True
    if hdr_found == False:
        print('WARNING: No header was found!')
        hdr = None
    return hdr
            
def cube_get(gal,data_mode,return_best=False,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/'):
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    # (!!!) COPY HERE
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    # (!!!) END COPY HERE

    # Spectral Cube
    cube=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = name+'_'+data_mode+'+tp_co21_pbcorr_round_k.fits'    # 7m+tp cube. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_pbcorr_round_k.fits'    # 7m cube. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            cube = SpectralCube.read(path+filename_7mtp)
            best_cube = '7m+tp'
        elif os.path.isfile(path+filename_7m):
            print('No 7m+tp cube found. Using 7m cube instead.')
            cube = SpectralCube.read(path+filename_7m)
            best_cube = '7m'
        else:
            best_cube = 'None'
    elif data_mode=='12m+7m':
        path = path12m
        filename_12mtp = name+'_'+data_mode+'+tp_co21_pbcorr_round_k.fits'
        filename_12m   = name+'_'+data_mode+    '_co21_pbcorr_round_k.fits'
        if os.path.isfile(path+filename_12mtp):
            cube = SpectralCube.read(path+filename_12mtp)
            best_cube = '12m+7m+tp'
        elif os.path.isfile(path+filename_12m):
            print('No 12m+tp cube found. Using 12m cube instead.')
            cube = SpectralCube.read(path+filename_12m)
            best_cube = '12m+7m'
        else:
            best_cube = 'None'
    else:
        print('WARNING: Invalid data_mode-- No cube was found!')
        cube = None
        best_cube = 'None'
    if cube is None:
        print('WARNING: No cube was found!')
        
    if return_best==True:
        return cube, best_cube
    else:
        return cube

def mask_get(gal,data_mode,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
    raise ValueError('tools.mask_get() : Has not been touched! Needs PHANGS-ALMA-LP/delivery/cubes support!')
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

    # Spectral Cube Mask
    mask=None
    if data_mode=='7m':
        path = path7m
        filename_7mtp = name+'_'+data_mode+'+tp_co21_pbcorr_round_k_mask.fits'    # 7m+tp mask. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_pbcorr_round_k_mask.fits'    # 7m mask. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            mask = SpectralCube.read(path+filename_7mtp)
        elif os.path.isfile(path+filename_7m):
            print('No 7m+tp mask found. Using 7m mask instead.')
            mask = SpectralCube.read(path+filename_7m)
        else:
            print('WARNING: \''+filename_7mtp+'\', \''+filename_7m+'\' not found!')
    elif data_mode=='12m+7m':
        path = path12m
        filename = name+'_co21_'+data_mode+'+tp_mask.fits'
        if os.path.isfile(path+filename):
            mask = SpectralCube.read(path+filename)
        else:
            print('WARNING: \''+filename+'\' not found!')
    else:
        print('WARNING: Invalid data_mode-- No mask was found!')
        
    if mask is None:
        print('WARNING: No mask was found!')
    return mask

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

def peakvels_get(gal,data_mode='',cube=None,mask=None,quadfit=True,write=True,best_cube=None,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
             path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
             folder_vpeak='jnofech_peakvels/'):
    '''
    Returns a 2D map of peak velocities.
    Can use the quadratic-fit method described
    in Teague & Foreman-Mackey 2018
    (https://arxiv.org/abs/1809.10295), 
    which improves accuracy for cubes of low
    spectral resolution.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    data_mode='12m' or '7m' : str
        Chooses where to find the output
        file, based on the selected data.
        A 'hybrid' mode also exists, which
        combines 12m and 7m data into a
        single map.
    cube(=None) : SpectralCube
        Spectral cube for the galaxy.
    mask(=None) : SpectralCube OR Quantity 
                  OR np.ndarray
        3D boolean array of cube's resolution,
        defining where the data is masked.
    quadfit(=True) : bool
        Enables the quadratic fit for finding 
        peak velocity. This means the resulting
        "peak velocity" will be far more 
        reliable at low spectral resolutions, 
        but comes at the cost of slower run 
        times.
    write(=True) : bool
        If no peakvels map is found, this
        toggles whether to write the output
        peakvels to a .fits file, in the
        `path(7/12)m_mask/jnofech_peakvels` 
        path.
    best_cube(=None) : str
        Best image quality-- e.g. '7m', 
        '7m+12m', etc., for the data cube.
        Only necessary when write=True.
        
    Returns:
    --------
    peakvels : np.ndarray
        2D map of peak velocities, in
        km/s.
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
    # Stick with 12m cubes when 'hybrid' is selected!
#     data_mode_peakvels = data_mode
#     if data_mode=='hybrid':
#         data_mode = '12m+7m'
        

    # Read from file, if possible!
    peakvels     = None
    peakvels_7m  = None
    peakvels_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_temp = '7m'
        # Get filename.
        if best_cube is None:
            cube_discard, best_cube  = cube_get(gal,data_mode_temp,return_best=True)
        if (best_cube is None) or (best_cube=='None'):
            raise ValueError('Best data mode not specified! This is needed to find the peakvels .fits file.')
        else:
            filename  = name+'_'+best_cube+'_co21_peakvels.fits'
        # Read or write.
        path=path7m_mask+folder_vpeak
        if os.path.isfile(path+filename)==True:
            peakvels_7m = fits.open(path+filename)
            peakvels    = peakvels_7m[0].data
        else:
            print(path+filename)
            if write==True:
                print('\''+filename+'\' does not exist. Generating and saving new peakvels map!')
            else:
                print('\''+filename+'\' does not exist. Generating new peakvels map!')
            # Define cube and mask!
            cube = cube_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
            mask = mask_get(gal,data_mode_temp,path7m=path7m_mask,path12m=path12m_mask)
            if (mask is not None) and (mask.size!=cube.size):
                print('WARNING: Mask has different dimensions from cube!')
                mask = None
            peakvels = peakvels_gen(gal,data_mode_temp,cube,mask,quadfit,write,best_cube)
            peakvels_7m = fits.open(path+filename)   # A new file was just created!
        best_cube_7m = best_cube
        best_cube    = None                             # Cleanup, for 'hybrid' mode.
    if data_mode in ['12m','12m+7m','hybrid']:
        data_mode_temp = '12m+7m'
        # Get filename.
        if best_cube is None:
            cube_discard, best_cube  = cube_get(gal,data_mode_temp,return_best=True)
        if (best_cube is None) or (best_cube=='None'):
            raise ValueError('Best data mode not specified! This is needed to find the peakvels .fits file.')
        else:
            filename  = name+'_co21_'+best_cube+'_peakvels.fits'
#             filename  = name+'_'+best_cube+'_co21_peakvels.fits'
        # Read or write.
        path=path12m_mask+folder_vpeak
        if os.path.isfile(path+filename)==True:
            peakvels_12m = fits.open(path+filename)
            peakvels     = peakvels_12m[0].data
        else:
            if write==True:
                print('\''+filename+'\' does not exist. Generating and saving new peakvels map!')
            else:
                print('\''+filename+'\' does not exist. Generating new peakvels map!')
            # Define cube and mask!
            cube = cube_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
            mask = mask_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
            if mask.size!=cube.size:
                print('WARNING: Mask has different dimensions from cube!')
                mask = None
            peakvels = peakvels_gen(gal,data_mode_temp,cube,mask,quadfit,write,best_cube)
            peakvels_12m = fits.open(path+filename)   # A new file was just created!
        best_cube_12m = best_cube
        best_cube     = None                             # Cleanup, for 'hybrid' mode.
        
    # Combining the two, if hybrid mode is enabled!
    if data_mode in ['hybrid']:
        hdr = peakvels_12m[0].header
        # Reproject the 7m map to the 12m's dimensions!
        # Conveniently, the interpolation is also done for us.
        peakvels_7m_modify = Projection.from_hdu(peakvels_7m)
        peakvels_7m = peakvels_7m_modify.reproject(peakvels_12m[0].header)
        # Convert to simple np arrays!
        peakvels_12m, peakvels_7m = peakvels_12m[0].data, peakvels_7m.value
        # COMBINE!
        peakvels_mask = (np.isfinite(peakvels_12m) + np.isfinite(peakvels_7m)).astype('float')
        peakvels_mask[peakvels_mask == 0.0] = np.nan    # np.nan where _neither_ 12m nor 7m have data.
        peakvels_hybrid = np.nan_to_num(peakvels_12m) + np.isnan(peakvels_12m)*np.nan_to_num(peakvels_7m) \
                                                      + peakvels_mask
        peakvels = peakvels_hybrid
        # SAVE!
        best_cube = 'hybrid_'+best_cube_7m+'&'+best_cube_12m
        hdr['BUNIT'] = 'km  s-1 '  # Write this instead of 'KM/S  '.
        # Save header and data into a .fits file, if specified!
        hdu      = fits.PrimaryHDU(peakvels_hybrid,header=hdr)
        filename = name+'_co21_'+best_cube+'_peakvels.fits'
        path = path12m_mask+folder_vpeak
        if os.path.isfile(path+filename)==False:
            hdu.writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
    if data_mode not in ['7m','12m+7m','hybrid']:
        print('WARNING: Invalid data_mode-- No mom1 was found!')
        peakvels = None
        return peakvels

    if peakvels is None:
        if verbose==True:
            print('WARNING: No peakvels was found!')

    return peakvels

def peakvels_gen(gal,data_mode='',cube=None,mask=None,quadfit=True,write=False,best_cube=None,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
             path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
             folder_vpeak='jnofech_peakvels/'):
    '''
    Returns a 2D map of peak velocities.
    Can use the quadratic-fit method described
    in Teague & Foreman-Mackey 2018
    (https://arxiv.org/abs/1809.10295), 
    which improves accuracy for cubes of low
    spectral resolution.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    data_mode='12m' or '7m' : str
        Chooses where to save the output
        file, based on the selected data.
    cube(=None) : SpectralCube
        Spectral cube for the galaxy.
    mask(=None) : SpectralCube OR Quantity 
                  OR np.ndarray
        3D boolean array of cube's resolution,
        defining where the data is masked.
    quadfit(=True) : bool
        Enables the quadratic fit for finding 
        peak velocity. This means the resulting
        "peak velocity" will be far more 
        reliable at low spectral resolutions, 
        but comes at the cost of slower run 
        times.
    write(=False) : bool
        Toggles whether to write the output
        peakvels to a .fits file, in the
        `path(7/12)m_mask/jnofech_peakvels` 
        path.
    best_cube(=None) : str
        Best image quality-- e.g. '7m', 
        '7m+12m', etc., for the data cube.
        Only necessary when write=True.
        
    Returns:
    --------
    peakvels : np.ndarray
        2D map of peak velocities, in
        km/s.
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
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m'

    if data_mode in ['7m', '12m+7m']:
        # Get the cube, and spectral axis.
        if cube is None:
            cube = cube_get(gal,data_mode,path7m,path12m)
        spec = cube.spectral_axis
        # Mask the data!
        if mask is None:
            print('WARNING: Mask not defined. Using unmasked data; will apply spatial mask later.')
            mask_provided = False
            mask = np.ones(cube.size).reshape(cube.shape)
            mask = BooleanArrayMask(mask=(mask==True), wcs=cube.wcs)
        elif (isinstance(mask,u.quantity.Quantity) or isinstance(mask,np.ndarray)):
            mask_provided = True
            mask = BooleanArrayMask(mask=(mask==True), wcs=cube.wcs)
        elif isinstance(mask,SpectralCube):
            mask_provided = True
            mask = BooleanArrayMask(mask=(mask.unmasked_data[:]==True), wcs=cube.wcs)
        else:
            mask_provided = True #(?)
            print('WARNING: Mask is not a Quantity, SpectralCube, or array. The code\'s probably about to crash.')    
        cube_m = cube.with_mask(mask)
        data = cube_m.filled_data[:].value
        # Find peak indices from this.
        data_copy = np.copy(data)
        data_copy[np.isnan(data)] = -np.inf  # This prevents 'np.argmax()' from choking on the 'np.nan' values.
        x0 = np.argmax(data_copy,axis=0)     # Indices of temperature peaks in spectral axis.
        if quadfit==True:    
            # Note: This I0 \equiv 'tpeak' will be identical to np.nanmax(data,axis=0).
            I0 = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Peak intensity, in K.
            Ip = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Intensity at (x0+1), in K.
            Im = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Intensity at (x0-1), in K.
            # Peak indices, plus or minus 1. These are the two points on a spectrum immediately around the peak.
            xp = x0+1
            xm = x0-1
            # If peak is at either end of the spectrum, then we can't really go higher or lower...
            # ... but that's almost guaranteed to be noise anyways, so we can just use a bogus value.
            xp[x0==(data.shape[0]-1)] = data.shape[0]-1
            xm[x0==0] = 0
            # Find the intensities, in K, at these indices!
            for j in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):   
                    I0[j,i] = data[x0[j,i],j,i]
                    Ip[j,i] = data[xp[j,i],j,i]
                    Im[j,i] = data[xm[j,i],j,i]
            # Find the quadratic-fitted peak indices!
            a0 = I0
            a1 = 0.5*(Ip-Im)
            a2 = 0.5*(Ip+Im-2*I0)
            xmax = x0 - (a1/(2.*a2))   # Quad-fitted indices of peak velocity! They're not integers anymore.
            # Generate 2D map of peak velocity, using these improved peak indices!
            spec_interp = interpolate.interp1d(np.arange(spec.size),spec, fill_value='extrapolate')  # Units gone.
            peakvels = spec_interp(xmax)
        else:
            # Generate 2D map of peak velocity, using default peak indices!
            spec_interp = interpolate.interp1d(np.arange(spec.size),spec, fill_value='extrapolate')  # Units gone.
            peakvels = spec_interp(x0)
    elif data_mode in ['hybrid','both']:
        raise ValueError("peakvels_gen() - 'data_mode=hybrid' should appear in peakvels_get(), not here!")
    else:
        raise ValueError("peakvels_gen() - invalid \'data_mode\'. Should be '7m', '12m', '12m+7m', or 'hybrid'.")

    # Adding units back
    peakvels = peakvels*spec.unit
    peakvels = peakvels.to(u.km/u.s)
    
    # Masking with spatial mask, if no mask was provided
    if mask_provided==False:
        print('WARNING: No cube mask provided! Using spatial mask instead.')
        I_mom1 = mom1_get(gal,data_mode,False,False,path7m,path12m)
        peakvels[np.isnan(I_mom1)] = np.nan
    
    # Give it a header!
    hdr = cube[0].header      # Take the header of a 2D slice from the cube.
    hdr['BUNIT'] = 'km  s-1 ' # Write this instead of 'KM/S  '. The 'Projection.from_hdu' will accept these units!
    
    # Save header and data into a .fits file, if specified!
    if write==True:
        hdu      = fits.PrimaryHDU(peakvels.value,header=hdr)
        if best_cube is None:
            print('WARNING: Best data mode not specified! Will assume "'+data_mode.lower()+'".')
            best_cube = data_mode.lower()
        
        if data_mode in ['7m']:
            filename = name+'_'+best_cube+'_co21_peakvels.fits'
            path=path7m_mask+folder_vpeak
        elif data_mode in ['12m+7m','hybrid']:
            filename = name+'_co21_'+best_cube+'_peakvels.fits'
            path=path12m_mask+folder_vpeak
        
#         print(path+filename)
        if os.path.isfile(path+filename)==False:
            hdu.writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.')    

    return peakvels.value
    
def wcs_to_pixels(gal,data_mode,RAcen,Deccen):
    '''
    Converts WCS central RA/Dec coordinates,
    in degrees (or u.Quantity form),
    into pixel coordinates.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    hdr = hdr_get(gal,data_mode)
    
    # Generate xcen, ycen as numbers.
    skycoord = gal.skycoord_grid(header=hdr)
    RA  = skycoord.ra.value     # 2D map of RA.
    Dec = skycoord.dec.value    # 2D map of Dec.
    
    if isinstance(RAcen,u.Quantity):
        RAcen = RAcen.to(u.deg).value
        Deccen = Deccen.to(u.deg).value
        
    xcen = RA.shape[1] * (RA.max() - RAcen) / (RA.max() - RA.min())
    ycen = RA.shape[0] * (Deccen - Dec.min()) / (Dec.max() - Dec.min())
    
    return xcen, ycen
    
def pixels_to_wcs(gal,data_mode,xcen,ycen):
    '''
    Converts central pixel coordinates
    into WCS central RA/Dec coordinates,
    in u.degrees.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    hdr = hdr_get(gal,data_mode)
    
    # Generate xcen, ycen as numbers.
    skycoord = gal.skycoord_grid(header=hdr)
    RA  = skycoord.ra.value     # 2D map of RA.
    Dec = skycoord.dec.value    # 2D map of Dec.
    
    
    RA_cen  = (RA.max()  - xcen*(RA.max()  - RA.min())/RA.shape[1])*u.deg
    Dec_cen = (Dec.min() + ycen*(Dec.max() - Dec.min())/RA.shape[0])*u.deg
    
    return RA_cen, Dec_cen
    
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
                  'NGC1637','NGC1792','NGC2090','NGC2283','NGC2835','NGC3511','NGC4298',\
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
    if gal.name.lower()=='ngc2566':
        PA = 310.*u.deg
    
    return PA
    
def incl_get(gal):
    '''
    Gets the inclination for the
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
    if gal.name.lower()=='ngc1672':
        print('galaxytools.incl_get(): Using FAKE inclination value, for DiskFit! REMOVE LATER?')
        incl = 50.*u.deg
    if gal.name.lower()=='ngc3059':
        print('galaxytools.incl_get(): Using FAKE inclination value, for DiskFit! REMOVE LATER?')
        incl = 20.*u.deg
    return incl

def coords_get(gal):
    '''
    Gets the central RA and Dec
    for the indicated galaxy, if 
    the provided ones look a bit off.
    NOTE: You'll probably need to
    save this into the headers as well,
    along with corresponding pixel
    coordinates, if you're working with
    a function that grabs central
    coords in some way!
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        
    Returns:
    --------
    RA : Quantity (float*u.deg)
        Right ascension, in degrees.
    Dec : Quantity (float*u.deg)
        Declination, in degrees.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Get coords!
    RA_cen = gal.center_position.ra.value
    Dec_cen = gal.center_position.dec.value
    
    # OVERRIDES
    # Done by eyeballing mom0 maps, mom1 maps, and optical images!
    # Custom central coordinates, in case provided values are very incorrect.
    if gal.name.lower()=='ic5332':
        # Central coords look more reasonable, but unsure of effect on rotcurve.
        RA_cen,Dec_cen = 353.61275414100163, Dec_cen
    if gal.name.lower()=='ngc1385':
        # Noticeable improvement on RC!
        RA_cen,Dec_cen = 54.371, -24.502
#     if gal.name.lower()=='ngc1559':
#         # Discard; made RC worse!
#         RA_cen,Dec_cen = 64.39710994945374,-62.78098504774272
    if gal.name.lower()=='ngc1637':
        # RC improved!
        RA_cen,Dec_cen = 70.36711177761812,-2.8589855372620074
    if gal.name.lower()=='ngc1792':
        # RC slightly worse than default, but makes more physical sense.
        RA_cen,Dec_cen = 76.30633574756656,-37.97824645945883
    if gal.name.lower()=='ngc1809':
        # mom1-vs-R makes less sense, but RC fit improved?? Unsure where center should be, exactly.
        RA_cen,Dec_cen = 75.51005604304831,-69.56348536585699
    if gal.name.lower()=='ngc2090':
        # Better RC, better mom1 fit!
        RA_cen,Dec_cen = 86.75809108234091,-34.250416010642695
    if gal.name.lower()=='ngc2283':
        # Better RC, better mom1 fit!
        RA_cen,Dec_cen = 101.469735789141,-18.210409767930265
#     if gal.name.lower()=='ngc2775':
#         # Central coords look more reasonable, and mom1 data is now focused into one "trend"
#         # (with much more scatter) rather than being in two tighter trends. RC looks better.
#         RA_cen,Dec_cen = 137.582, 7.037970066070557
    if gal.name.lower()=='ngc3511':
        # Better RC fit, better mom1 fit!
        RA_cen,Dec_cen = 165.8484352408927,-23.08680662885883
    if gal.name.lower()=='ngc4207':
        # Central coords look more reasonable, and mom1 data is focused more into one of
        # the two "trends". RC looks better, but the error bars are much, much worse.
        RA_cen,Dec_cen = 183.8765752788386, 9.584930419921875
    if gal.name.lower()=='ngc4293':
        # Slightly jaggier RC.
        RA_cen,Dec_cen = 185.30302908075126,18.3829153738402
    if gal.name.lower()=='ngc4254':
        # Central coords look MUCH more reasonable.
        # Much of the mom1 data is still below zero for some reason. Improved, but nowhere near perfect.
        RA_cen,Dec_cen = 184.7074087094364, gal.center_position.dec.value
    if gal.name.lower()=='ngc4424':
        # RC even more wild, not that it was reliable to begin with.
        RA_cen,Dec_cen = 186.7959698271931,9.421769823689191
    if gal.name.lower()=='ngc4457':
        # Much better RC and mom1 fit!
        RA_cen,Dec_cen = 187.24653811871107,3.570680697523369
    if gal.name.lower()=='ngc4569':
        # No improvement, despite it being a "clean"-looking galaxy. No idea why mom1 fit is still split.
        RA_cen,Dec_cen = 189.20679998931678,13.163193855398312
#     if gal.name.lower()=='ngc4571':
#         # Better mom1 fit, although DF still hates it for some reason.
#         RA_cen,Dec_cen = 189.2340004103119,14.219281570910422
    if gal.name.lower()=='ngc4654':
        # Improved RC and mom1 fit!
        RA_cen,Dec_cen = 190.98570142483396,13.12708187068814
    if gal.name.lower()=='ngc4694':
        # No effect.
        RA_cen,Dec_cen = 192.06253044717437,10.984342977376253
    if gal.name.lower()=='ngc4731':
        # Central coords look somewhat more reasonable. RC is horribly jagged and dips into negatives,
        # but is still a huge improvement.
        RA_cen,Dec_cen = 192.750, -6.39
    if gal.name.lower()=='ngc4781':
        # Slightly _worse_ RC and mom1 fit, although central coords make more physical sense.
        RA_cen,Dec_cen = 193.59747846280533,-10.535709989241946
    if gal.name.lower()=='ngc4826':
        # No noticeable improvement.
        RA_cen,Dec_cen = 194.1812342003185,21.68321257606272   # New (should be the same, but makes RC worse?!)
#         RA_cen,Dec_cen = 194.1847900357396,21.68321257606272  # Old
    if gal.name.lower()=='ngc4951':
        # Big improvement on RC and mom1 fit!
        RA_cen,Dec_cen = 196.28196323269697,-6.493484077561956
    if gal.name.lower()=='ngc5042':
        # Improved RC and mom1 fit!
        RA_cen,Dec_cen = 198.8787057293563,-23.98319382130314
    if gal.name.lower()=='ngc5068':
        # Central seems better if you compare mom1 image to other NED images, but unsure if the
        # new coords are more reasonable. Mom1 data went from 2 trends to 1 trend+more scatter.
        # RC is somewhat jaggier, but still has very small errors.
        RA_cen,Dec_cen = 199.73321317663428, -21.045
    if gal.name.lower()=='ngc5530':
        # Improved RC and mom1 fit!
        RA_cen,Dec_cen = 214.61009028871453,-43.386535568706506
    if gal.name.lower()=='ngc5643':
        # Slightly improved RC and mom1 fit!
        RA_cen,Dec_cen = 218.169014845381,-44.17440210179905
    if gal.name.lower()=='ngc6744':
        # Improved RC and mom1 fit!
        RA_cen,Dec_cen = 287.4432981513924,-63.857521145661536
    if gal.name.lower()=='ngc7496':
        RA_cen,Dec_cen = 347.4464952115924,-43.42790798487375
    
    # Turn into RA, Dec!
    gal.center_position = SkyCoord(RA_cen,Dec_cen,unit=(u.deg,u.deg), frame='fk5')
    return gal.center_position
    
def coords_philipp_get(gal):
    '''
    Gets the central RA and Dec
    for the indicated galaxy, if 
    the provided ones look a bit off.
    NOTE: You'll probably need to
    save this into the headers as well,
    along with corresponding pixel
    coordinates, if you're working with
    a function that grabs central
    coords in some way!
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        
    Returns:
    --------
    RA : Quantity (float*u.deg)
        Right ascension, in degrees.
    Dec : Quantity (float*u.deg)
        Declination, in degrees.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # OVERRIDES
    # Philipp's custom values!
    if gal.name.lower()=='ic1954':
        RA_cen,Dec_cen = 52.879709,-51.904862
    elif gal.name.lower()=='ic5273':
        RA_cen,Dec_cen = 344.86118,-37.702839
    elif gal.name.lower()=='ic5332':
        RA_cen,Dec_cen = 353.61443,-36.10106
    elif gal.name.lower()=='ngc0628':
        RA_cen,Dec_cen = 24.173854,15.783643
    elif gal.name.lower()=='ngc0685':
        RA_cen,Dec_cen = 26.928451,-52.761977
    elif gal.name.lower()=='ngc1087':
        RA_cen,Dec_cen = 41.604919,-0.4987175
    elif gal.name.lower()=='ngc1097':
        RA_cen,Dec_cen = 41.578957,-30.274675
    elif gal.name.lower()=='ngc1300':
        RA_cen,Dec_cen = 49.920813,-19.411114
    elif gal.name.lower()=='ngc1317':
        RA_cen,Dec_cen = 50.68454,-37.10379
    elif gal.name.lower()=='ngc1365':
        RA_cen,Dec_cen = 53.401519,-36.140403
    elif gal.name.lower()=='ngc1385':
        RA_cen,Dec_cen = 54.369015,-24.501161
    elif gal.name.lower()=='ngc1433':
        RA_cen,Dec_cen = 55.506195,-47.221943
    elif gal.name.lower()=='ngc1511':
        RA_cen,Dec_cen = 59.902459,-67.633924
    elif gal.name.lower()=='ngc1512':
        RA_cen,Dec_cen = 60.975574,-43.348724
    elif gal.name.lower()=='ngc1546':
        RA_cen,Dec_cen = 63.651219,-56.060898
    elif gal.name.lower()=='ngc1559':
        RA_cen,Dec_cen = 64.40238,-62.783411
    elif gal.name.lower()=='ngc1566':
        RA_cen,Dec_cen = 65.00159,-54.938012
    elif gal.name.lower()=='ngc1637':
        RA_cen,Dec_cen = 70.367425,-2.8579358
    elif gal.name.lower()=='ngc1672':
        RA_cen,Dec_cen = 71.42704,-59.247259
    elif gal.name.lower()=='ngc1792':
        RA_cen,Dec_cen = 76.30969,-37.980559
    elif gal.name.lower()=='ngc1809':
        RA_cen,Dec_cen = 75.520658,-69.567942
    elif gal.name.lower()=='ngc2090':
        RA_cen,Dec_cen = 86.757874,-34.250599
    elif gal.name.lower()=='ngc2283':
        RA_cen,Dec_cen = 101.46997,-18.2108
    elif gal.name.lower()=='ngc2566':
        RA_cen,Dec_cen = 124.69003,-25.499519
    elif gal.name.lower()=='ngc2775':
        RA_cen,Dec_cen = 137.58396,7.0380658
    elif gal.name.lower()=='ngc2835':
        RA_cen,Dec_cen = 139.47044,-22.354679
    elif gal.name.lower()=='ngc2903':
        RA_cen,Dec_cen = 143.04212,21.500841
    elif gal.name.lower()=='ngc2997':
        RA_cen,Dec_cen = 146.41164,-31.19109
    elif gal.name.lower()=='ngc3059':
        RA_cen,Dec_cen = 147.534,-73.922194
    elif gal.name.lower()=='ngc3137':
        RA_cen,Dec_cen = 152.28116,-29.064301
    elif gal.name.lower()=='ngc3239':
        RA_cen,Dec_cen = 156.27756,17.16119
    elif gal.name.lower()=='ngc3351':
        RA_cen,Dec_cen = 160.99064,11.70367
    elif gal.name.lower()=='ngc3507':
        RA_cen,Dec_cen = 165.85573,18.13552
    elif gal.name.lower()=='ngc3511':
        RA_cen,Dec_cen = 165.84921,-23.086713
    elif gal.name.lower()=='ngc3521':
        RA_cen,Dec_cen = 166.4524,-0.035948747
    elif gal.name.lower()=='ngc3596':
        RA_cen,Dec_cen = 168.77581,14.787066
    elif gal.name.lower()=='ngc3621':
        RA_cen,Dec_cen = 169.56792,-32.812599
    elif gal.name.lower()=='ngc3626':
        RA_cen,Dec_cen = 170.01588,18.356845
    elif gal.name.lower()=='ngc3627':
        RA_cen,Dec_cen = 170.06252,12.9915
    elif gal.name.lower()=='ngc4207':
        RA_cen,Dec_cen = 183.87681,9.5849279
    elif gal.name.lower()=='ngc4254':
        RA_cen,Dec_cen = 184.7068,14.416412
    elif gal.name.lower()=='ngc4293':
        RA_cen,Dec_cen = 185.30346,18.382575
    elif gal.name.lower()=='ngc4298':
        RA_cen,Dec_cen = 185.38651,14.60611
#     elif gal.name.lower()=='ngc4303':
#         RA_cen,Dec_cen = 185.54241,4.4722827    # New (Bad!)
    elif gal.name.lower()=='ngc4303':
        RA_cen,Dec_cen = 185.47888,4.4737438   # Newer (good!)
#    elif gal.name.lower()=='ngc4303':
#        RA_cen,Dec_cen = 185.47896,4.47359    # Old (good?)
    elif gal.name.lower()=='ngc4321':
        RA_cen,Dec_cen = 185.72886,15.822304
    elif gal.name.lower()=='ngc4424':
        RA_cen,Dec_cen = 186.79821,9.4206371
    elif gal.name.lower()=='ngc4457':
        RA_cen,Dec_cen = 187.24593,3.5706196
    elif gal.name.lower()=='ngc4535':
        RA_cen,Dec_cen = 188.58459,8.1979729
    elif gal.name.lower()=='ngc4536':
        RA_cen,Dec_cen = 188.61278,2.1882429
    elif gal.name.lower()=='ngc4540':
        RA_cen,Dec_cen = 188.71193,15.551724
    elif gal.name.lower()=='ngc4548':
        RA_cen,Dec_cen = 188.86024,14.496331
    elif gal.name.lower()=='ngc4569':
        RA_cen,Dec_cen = 189.20759,13.162875
    elif gal.name.lower()=='ngc4571':
        RA_cen,Dec_cen = 189.23492,14.217327
    elif gal.name.lower()=='ngc4579':
        RA_cen,Dec_cen = 189.43138,11.818217
    elif gal.name.lower()=='ngc4654':
        RA_cen,Dec_cen = 190.98575,13.126715
    elif gal.name.lower()=='ngc4689':
        RA_cen,Dec_cen = 191.9399,13.762724
    elif gal.name.lower()=='ngc4694':
        RA_cen,Dec_cen = 192.0627,10.983726
    elif gal.name.lower()=='ngc4731':
        RA_cen,Dec_cen = 192.75504,-6.3928396
    elif gal.name.lower()=='ngc4781':
        RA_cen,Dec_cen = 193.59916,-10.537116
    elif gal.name.lower()=='ngc4826':
        RA_cen,Dec_cen = 194.18184,21.683083
    elif gal.name.lower()=='ngc4941':
        RA_cen,Dec_cen = 196.05461,-5.5515362
    elif gal.name.lower()=='ngc4951':
        RA_cen,Dec_cen = 196.28213,-6.4938237
    elif gal.name.lower()=='ngc5042':
        RA_cen,Dec_cen = 198.8792,-23.983883
    elif gal.name.lower()=='ngc5068':
        RA_cen,Dec_cen = 199.72808,-21.038744
    elif gal.name.lower()=='ngc5128':
        RA_cen,Dec_cen = 201.35869,-43.016082
    elif gal.name.lower()=='ngc5134':
        RA_cen,Dec_cen = 201.32726,-21.134195
    elif gal.name.lower()=='ngc5248':
        RA_cen,Dec_cen = 204.38336,8.8851946
    elif gal.name.lower()=='ngc5530':
        RA_cen,Dec_cen = 214.6138,-43.38826
    elif gal.name.lower()=='ngc5643':
        RA_cen,Dec_cen = 218.16991,-44.17461
    elif gal.name.lower()=='ngc6300':
        RA_cen,Dec_cen = 259.2478,-62.820549
    elif gal.name.lower()=='ngc6744':
        RA_cen,Dec_cen = 287.44208,-63.85754
    elif gal.name.lower()=='ngc7456':
        RA_cen,Dec_cen = 345.54306,-39.569412
    elif gal.name.lower()=='ngc7496':
        RA_cen,Dec_cen = 347.44703,-43.427849

    # Turn into RA, Dec!
    gal.center_position = SkyCoord(RA_cen,Dec_cen,unit=(u.deg,u.deg), frame='fk5')
    return gal.center_position
    
def logmass_get(gal=None,path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table'):
    '''
    Returns log10(*Stellar* mass / Msun)
    for the specified galaxy, OR
    for every galaxy in 
    galaxies_list if a galaxy is not
    specified.
    
    Parameters:
    -----------
    gal=None : int or Galaxy
        Galaxy object.
    
    Returns:
    --------
    logmstar : float or array
        All galaxy masses corresponding
        to `galaxy_list`, OR the
        mass of the single galaxy specified.
    '''
    table = fits.open(path+fname+'.fits')[1].data

    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    elif gal is None:
        print('tools.get_logmstar() : No galaxy selected! Returning entire array of galaxy masses.')
        logmasses = np.zeros(len(galaxies_list))
        for i in range(0,len(galaxies_list)):
            name = galaxies_list[i]
            logmasses[i] = table.field('LOGMSTAR')[list(table.field('NAME')).index(name)]
        return logmasses
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    logmass = table.field('LOGMSTAR')[list(table.field('NAME')).index(name.upper())]
    return logmass

def TFvelocity_get(gal):
    '''
    Returns predicted Tully-
    Fisher velocity (not-projected)
    for the specified galaxy.
    
    Parameters:
    -----------
    gal : int or Galaxy
        Galaxy object.
    
    Returns:
    --------
    TFv : Quantity
        Approximate rotational velocity
        of galaxy.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    if name.lower()=='ic1954':
        TFv = 129.3896852*u.km/u.s
    if name.lower()=='ic5273':
        TFv = 126.183459*u.km/u.s
    if name.lower()=='ic5332':
        TFv = 122.6781319*u.km/u.s
    if name.lower()=='ngc0628':
        TFv = 166.1088534*u.km/u.s
    if name.lower()=='ngc0685':
        TFv = 131.2281377*u.km/u.s
    if name.lower()=='ngc1087':
        TFv = 129.6414964*u.km/u.s
    if name.lower()=='ngc1097':
        TFv = 211.1742643*u.km/u.s
    if name.lower()=='ngc1300':
        TFv = 211.7199579*u.km/u.s
    if name.lower()=='ngc1317':
        TFv = 182.5707277*u.km/u.s
    if name.lower()=='ngc1365':
        TFv = 223.2490649*u.km/u.s
    if name.lower()=='ngc1385':
        TFv = 157.4249651*u.km/u.s
    if name.lower()=='ngc1433':
        TFv = 198.1045651*u.km/u.s
    if name.lower()=='ngc1511':
        TFv = 140.4395746*u.km/u.s
    if name.lower()=='ngc1512':
        TFv = 187.8490435*u.km/u.s
    if name.lower()=='ngc1546':
        TFv = 163.1179861*u.km/u.s
    if name.lower()=='ngc1559':
        TFv = 135.9387902*u.km/u.s
    if name.lower()=='ngc1566':
        TFv = 205.4559423*u.km/u.s
    if name.lower()=='ngc1637':
        TFv = 127.155985*u.km/u.s
    if name.lower()=='ngc1672':
        TFv = 168.2787683*u.km/u.s
    if name.lower()=='ngc1792':
        TFv = 168.0101957*u.km/u.s
    if name.lower()=='ngc1809':
        TFv = 140.5480859*u.km/u.s
    if name.lower()=='ngc2090':
        TFv = 144.4697148*u.km/u.s
    if name.lower()=='ngc2283':
        TFv = 139.2615752*u.km/u.s
    if name.lower()=='ngc2566':
        TFv = 193.6596487*u.km/u.s
    if name.lower()=='ngc2775':
        TFv = 206.0931337*u.km/u.s
    if name.lower()=='ngc2835':
        TFv = 133.6302972*u.km/u.s
    if name.lower()=='ngc2903':
        TFv = 178.6525867*u.km/u.s
    if name.lower()=='ngc2997':
        TFv = 185.2073733*u.km/u.s
    if name.lower()=='ngc3059':
        TFv = 171.4108286*u.km/u.s
    if name.lower()=='ngc3137':
        TFv = 130.6415472*u.km/u.s
    if name.lower()=='ngc3239':
        TFv = 131.6274418*u.km/u.s
    if name.lower()=='ngc3351':
        TFv = 168.9887324*u.km/u.s
    if name.lower()=='ngc3507':
        TFv = 175.4601676*u.km/u.s
    if name.lower()=='ngc3511':
        TFv = 124.2545764*u.km/u.s
    if name.lower()=='ngc3521':
        TFv = 207.0739332*u.km/u.s
    if name.lower()=='ngc3596':
        TFv = 110.4244053*u.km/u.s
    if name.lower()=='ngc3621':
        TFv = 182.3324516*u.km/u.s
    if name.lower()=='ngc3626':
        TFv = 173.2528181*u.km/u.s
    if name.lower()=='ngc3627':
        TFv = 188.9890834*u.km/u.s
    if name.lower()=='ngc4207':
        TFv = 120.7242452*u.km/u.s
    if name.lower()=='ngc4254':
        TFv = 184.2006293*u.km/u.s
    if name.lower()=='ngc4293':
        TFv = 180.4140181*u.km/u.s
    if name.lower()=='ngc4298':
        TFv = 144.8225231*u.km/u.s
    if name.lower()=='ngc4303':
        TFv = 198.5801286*u.km/u.s
    if name.lower()=='ngc4321':
        TFv = 200.0481604*u.km/u.s
    if name.lower()=='ngc4424':
        TFv = 92.35864869*u.km/u.s
    if name.lower()=='ngc4457':
        TFv = 170.2458161*u.km/u.s
    if name.lower()=='ngc4535':
        TFv = 184.7751102*u.km/u.s
    if name.lower()=='ngc4536':
        TFv = 165.5247113*u.km/u.s
    if name.lower()=='ngc4540':
        TFv = 125.7450402*u.km/u.s
    if name.lower()=='ngc4548':
        TFv = 194.1613802*u.km/u.s
    if name.lower()=='ngc4569':
        TFv = 214.7993831*u.km/u.s
    if name.lower()=='ngc4571':
        TFv = 143.2840114*u.km/u.s
    if name.lower()=='ngc4579':
        TFv = 220.2774839*u.km/u.s
    if name.lower()=='ngc4654':
        TFv = 163.3954767*u.km/u.s
    if name.lower()=='ngc4689':
        TFv = 157.0624416*u.km/u.s
    if name.lower()=='ngc4694':
        TFv = 130.9081556*u.km/u.s
    if name.lower()=='ngc4731':
        TFv = 113.1246446*u.km/u.s
    if name.lower()=='ngc4781':
        TFv = 134.678573*u.km/u.s
    if name.lower()=='ngc4826':
        TFv = 154.4822491*u.km/u.s
    if name.lower()=='ngc4941':
        TFv = 146.7452837*u.km/u.s
    if name.lower()=='ngc4951':
        TFv = 115.4472648*u.km/u.s
    if name.lower()=='ngc5042':
        TFv = 115.9642888*u.km/u.s
    if name.lower()=='ngc5068':
        TFv = 107.0286549*u.km/u.s
    if name.lower()=='ngc5128':
        TFv = 223.8369277*u.km/u.s
    if name.lower()=='ngc5134':
        TFv = 164.5652803*u.km/u.s
    if name.lower()=='ngc5248':
        TFv = 157.5482845*u.km/u.s
    if name.lower()=='ngc5530':
        TFv = 149.1771141*u.km/u.s
    if name.lower()=='ngc5643':
        TFv = 170.2398255*u.km/u.s
    if name.lower()=='ngc6300':
        TFv = 184.7512441*u.km/u.s
    if name.lower()=='ngc6744':
        TFv = 237.2941833*u.km/u.s
    if name.lower()=='ngc7456':
        TFv = 90.10775682*u.km/u.s
    if name.lower()=='ngc7496':
        TFv = 144.3432589*u.km/u.s
    
    return TFv

def bar_info_get(gal,data_mode,radii='arcsec',customPA=True,check_has_bar=False,\
                 folder='drive_tables/',fname='TABLE_Environmental_masks - Parameters'):
    '''
    Gets the bar information for a specified
    galaxy. Table must be a .csv file!
    
    Parameters:
    -----------
    gal : Galaxy
        Galaxy.
    data_mode : str
        Data mode (7m or 12m) of
        galaxy, for converting
        radius units.
    radii : str
        Unit that the bar radius
        is returned in.
        'arcsec' (Default)
        'pc', 'parsec'
        'pix', 'pixel', 'pixels'  <- Dependent on data_mode
    check_has_bar(=False) : bool
        Checks whether the galaxy has
        a bar, returning 'Yes',
        'No', or 'Uncertain' (not in list).
        Replaces other outputs.
        
    Returns:
    --------
    bar_PA : Quantity
        PA of bar, in degrees.
    bar_R : Quantity
        Radius of bar, in parsec.
    
    Will return 'np.nan,np.nan' if
    galaxy does not have a bar, or if
    galaxy is not in the list.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")

    hdr = hdr_get(gal,data_mode)

    with open(folder+fname+'.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        table = [[r] for r in reader]
    for i in range(0,len(table)):
        table[i] = table[i][0]          # Cleans up the table into a proper 2D list.

    bar_index = table[0].index('-------------------------- bar ---------------------------')
    # If galaxy is in table, find ratio+PA+R!
    if name.upper() in np.array(table)[:,0]:
        row = np.where(np.array(table)[:,0] == name.upper())[0][0]
        bar_axisratio, bar_PA, bar_R = table[row][bar_index:(bar_index+3)]
        # Abort if there's no bar.
        if '-999' in [bar_axisratio,bar_PA,bar_R]:
            print('tools.bar_info_get() : '+name.upper()+' does not have a bar!')
            if check_has_bar==False:
                return np.nan, np.nan
            else:
                return 'No'
        bar_axisratio = float(bar_axisratio)   # Axis ratio == a/b == 1/cos(i)
#         bar_incl = (np.arccos(1./bar_axisratio)*u.rad).to(u.deg)     # (!!!) Delete this? Bars don't have inclinations, ya donut
        bar_PA = float(bar_PA)*u.deg           # PA (deg)
        bar_R = float(bar_R)*u.arcsec          # R (arcsec)
        # Custom PA values!
#        if customPA==True:
            # a) Bar PA is off by 180 degrees.
#            galaxies_180=['IC5273','NGC1300','NGC1365','NGC1385','NGC1511','NGC1512','NGC1559',\
#                          'NGC4535','NGC4731','NGC4781','NGC4826','NGC5042','NGC5134','NGC5530']
#            raise ValueError('tools.bar_info_get() : Custom bar PAs not decided!!')
#            if name.upper() in galaxies_180:
#            if True:
#                print('(!!!) NOTE: Flipping bar PA by 180deg has NO EFFECT for m=2 fits! Uses photometric, not kinematic.')
#                if bar_PA.value>180.:
#                    bar_PA = bar_PA-180.*u.deg
#                else:
#                    bar_PA = bar_PA+180.*u.deg
#            # b) Bar PA for galaxies whose photometric PAs are just WAY off from kinematic PAs. (Hopefully none!)
#            if gal.name.lower()=='ngcXXXX':
#                bar_PA = 999.*u.deg
    else:
        print('tools.bar_info_get() : '+name.upper()+' not in '+fname+'.csv!')
        if check_has_bar==False:
            return np.nan, np.nan
        else:
            return 'Uncertain'
    
    # Convert R to desired units!
    if radii=='arcsec':
        bar_R = bar_R            # Bar radius, in ".
    else:
        pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))[0]*u.deg # Pixel width, in deg.
        pixsizes_arcsec = pixsizes_deg.to(u.arcsec)                             # Pixel width, in arcsec.
        pixsizes_rad = pixsizes_deg.to(u.rad)                                   # Pixel width, in radians.
        bar_R_pix = bar_R / pixsizes_arcsec                                     # Bar radius, in pixels.
        if radii in ['parsec','pc']:
            pcperpixel = pixsizes_rad.value*gal.distance.to(u.pc)    # Pixel width, in pc.
            bar_R = bar_R_pix * pcperpixel        # Bar radius, in pc.
        elif radii in ['kpc']:
            kpcperpixel = pixsizes_rad.value*gal.distance.to(u.kpc)    # Pixel width, in kpc.
            bar_R = bar_R_pix * kpcperpixel       # Bar radius, in kpc.
        elif radii in ['pix','pixel','pixels']:
            bar_R = bar_R_pix                     # Bar radius, in pixels.
            
    if check_has_bar==False:
        return bar_PA, bar_R
    else:
        return 'Yes'
    
    # Convert R to desired units!
    if radii=='arcsec':
        bar_R = bar_R            # Bar radius, in ".
    else:
        pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))[0]*u.deg # Pixel width, in deg.
        pixsizes_arcsec = pixsizes_deg.to(u.arcsec)                             # Pixel width, in arcsec.
        pixsizes_rad = pixsizes_deg.to(u.rad)                                   # Pixel width, in radians.
        bar_R_pix = bar_R / pixsizes_arcsec                                     # Bar radius, in pixels.
        if radii in ['parsec','pc']:
            pcperpixel = pixsizes_rad.value*gal.distance.to(u.pc)    # Pixel width, in pc.
            bar_R = bar_R_pix * pcperpixel        # Bar radius, in pc.
        elif radii in ['kpc']:
            kpcperpixel = pixsizes_rad.value*gal.distance.to(u.kpc)    # Pixel width, in kpc.
            bar_R = bar_R_pix * kpcperpixel       # Bar radius, in kpc.
        elif radii in ['pix','pixel','pixels']:
            bar_R = bar_R_pix                     # Bar radius, in pixels.

    return bar_PA, bar_R
   
def info(gal,conbeam=None,data_mode='',sfr_band_uv='nuv',sfr_band_ir='w3',hasmask=False,sfr_autocorrect=False):
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
    hasmask : bool
        Determines whether a cube
        mask is available.
        
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
    (if hasmask==True) vpeak : np.ndarray
        Peak velocity, in km/s.
    I_tpeak : np.ndarray
        Peak temperature, in K.
    cube : SpectralCube
        Spectral cube for the galaxy.
    (if hasmask==True) mask : SpectralCube
        Mask for 'cube'.
    sfr : np.ndarray
        2D map of the SFR, in Msun/kpc^2/yr.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
        
    if data_mode == '7m+tp':
        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    
    I_mom0 = mom0_get(gal,data_mode)
    I_mom1 = mom1_get(gal,data_mode)
    I_tpeak = tpeak_get(gal,data_mode)
    hdr = hdr_get(gal,data_mode,dim=2)
    beam = hdr['BMAJ']                       # In degrees.
    beam_arcsec = (beam*u.deg).to(u.arcsec)  # In arcsec. We want this to be LOWER than the SFR map's 7.5"
                                             #    beamwidth (i.e. higher resolution), but this often fails
                                             #    and we need to use 15" SFR maps instead.
    
    # Choose appropriate resolution for SFR map, changing 'conbeam' to match it if necessary.
    res='7p5'
    if beam_arcsec > 7.5*u.arcsec and conbeam is not None:
        print('(galaxytools.info())     WARNING: Beam is '+str(beam_arcsec)+', and we want to convolve.')
        print('                                  This will use a 15" SFR map instead of 7.5"!')
        res='15'
        if conbeam==7.5*u.arcsec:
            print('(galaxytools.info())              We\'ll also use a 15" conbeam.')
            conbeam = 15.*u.arcsec
    
    # Get SFR at this resolution.
    sfr = sfr_get(gal,hdr,res=res,band_uv=sfr_band_uv,band_ir=sfr_band_ir,autocorrect=sfr_autocorrect) 
    #     Not convolved yet, despite that being an option.
    
    if res=='7p5' and sfr is None and sfr_autocorrect==True:  # If 7.5" isn't found and we want to try lower res:
        print('(galaxytools.info())     WARNING: 7.5" SFR map not found.\n\
                                  Will attempt a 15" SFR map instead!')
        res='15'
        sfr = sfr_get(gal,hdr,res=res,band_uv=sfr_band_uv,band_ir=sfr_band_ir) # Try again with lower resolution.
    if res=='15' and sfr is not None:  # If a 15" SFR map was successful:
        if conbeam==7.5*u.arcsec:
            print('(galaxytools.info())     NOTE:    The 15" SFR map was successful! Changing conbeam from 7.5" to 15".')
            conbeam=15.*u.arcsec
    # Get cube+mask!
    cube,bestcube = cube_get(gal,data_mode,return_best=True)
    if hasmask==True:
        mask          = mask_get(gal,data_mode)
        # Get peakvels!
        peakvels = peakvels_get(gal,data_mode,cube,mask,True,False,bestcube)
    
    
    # CONVOLUTION, if enabled:
    if conbeam!=None:
        hdr,I_mom0, I_tpeak, cube = cube_convolved(gal,conbeam,data_mode) # CONVOLVED moments, with their cube.
        if sfr is not None:
            sfr = convolve_2D(gal,hdr,sfr,conbeam)  # Convolved SFR map.
    
    if hasmask==True:
        return hdr,beam,I_mom0,I_mom1,peakvels,I_tpeak,cube,mask,sfr
    else:
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
    maps_notempty = []
    for i in range(0,len(maps)):
        if np.sum(~np.isnan(maps[i]))>0:    # If the map isn't empty:
            maps_notempty.append(maps[i])
    index = np.arange(rad.size)
    index = index[ np.isfinite(rad1D*np.sum(maps_notempty,axis=0))]
    rad1D = rad1D[index][::stride]
    for i in range(0,len(maps)):
        maps[i] = maps[i][index][::stride]
    
    # Ordering the maps!
    import operator
    L = sorted(zip(rad1D.value,*maps), key=operator.itemgetter(0))
    rad1D,maps = np.array(list(zip(*L))[0])*u.pc, np.array(list(zip(*L))[1:])
    
    # Returning everything!
    return rad1D,maps

def sigmas(gal,hdr=None,I_mom0=None,I_tpeak=None,alpha=6.7,data_mode='',mapmode='mom1',sigmode=''):
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
        print('New Galaxy object created for '+name+'!')
        gal = tools.galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Header
    if hdr==None:
        print('galaxytools.sigmas(): WARNING: Header found automatically. Check that it\'s correct!')
        hdr = hdr_get(gal,data_mode=data_mode,dim=2)
    # Moments
    if I_mom0 is None:
        print('galaxytools.sigmas(): WARNING: I_mom0 found automatically.')
        I_mom0 = mom0_get(gal,data_mode=data_mode)
    if I_tpeak is None:
        print('galaxytools.sigmas(): WARNING: I_tpeak found automatically.')
        I_tpeak = tpeak_get(gal,data_mode=data_mode)
    
    # (!!!) Ensure beforehand that the PA is kinematic, not photometric!
    x, rad, x, x = gal.rotmap(header=hdr,data_mode=data_mode,mapmode=mapmode)
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
