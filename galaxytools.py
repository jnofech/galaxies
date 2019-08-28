import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.wcs import WCS
from astropy.table import Table
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.coordinates import SkyCoord, Angle, FK5
from spectral_cube import SpectralCube, Projection, BooleanArrayMask
from radio_beam import Beam

from scipy import interpolate
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.stats import binned_statistic

# Import Jiayi's code for Alpha
from AlmaTools import XCO

from galaxies.galaxies import Galaxy
import rotcurve_tools as rc
import diskfit_input_generator as dig

import copy
import os
import csv

# Import silencer
import os, sys
class silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def galaxy(name,\
           customPA=True,custominc='phil',customcoords='phil',customvsys='phil',\
           customdistance=True,\
           diskfit_output=False,data_mode='7m',mapmode='mom1',\
           path='/media/jnofech/BigData/galaxies/drive_tables/',fname='philipp_Final_orientations_April16.txt'):
    '''
    Creates Galaxy object.
    Features kinematic PA and a quick
    bandaid fix for missing 'vsys' values!
    
    Parameters:
    -----------
    customPA=True : bool
        Can also be a string, 'LSQ' or 'MC', to
        grab fitted PA from the LSQ-URC or MC-URC.
        Requires mapmode='mom1'.
    custominc='phil' : bool OR str
    customcoords='phil' : bool OR str
        customcoords='p','phil','philipp':
        enables Philipp's custom central coords
    customdistance=True : bool
        Uses PHANGS updated galaxy distances
        from PHANGS v3p3 release.
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
    
    # Custom PA
    if isinstance(customPA,str):
        # Check if data_mode and mapmode are good:
        if (mapmode=='mom1')==False:
            raise ValueError('tools.galaxy() : Cannot use "customPA='+customPA+'" unless mapmode=\'mom1\'.')
        if customPA.lower() in ['lsq', 'lsqurc', 'ls', 'lsurc']:
            MCurc_savefile='LSQ_'
        elif customPA.lower() in ['mc', 'urc', 'mcurc']:
            MCurc_savefile='MC_'
        else:
            raise ValueError('"'+customPA+'" is not a valid "customPA"! See header for details.')
        smooth = 'universal'
        print('tools.galaxy(customPA='+customPA+') : smooth=universal is assumed.')
        if os.path.isfile('/media/jnofech/BigData/galaxies/MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz'):
            MCurc_data = np.load('/media/jnofech/BigData/galaxies/MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz')
            params_MC     = MCurc_data['egg1']
            PA_MC = (params_MC[3]*u.rad).to(u.deg)
            # ^ This will overwrite any other PA at the end of this function.
        else:
            print('rc.MC(): WARNING - MCurc_save/'+MCurc_savefile+name.upper()+'_'+data_mode+'_'+smooth+'.npz does not exist!')
            customPA = True
    if customPA==True:
        gal.position_angle  = PA_get(gal)
    
    # Custom inc
    if custominc==True or custominc=='True' or custominc=='true':
        gal.inclination     = incl_get(gal)
    elif custominc in ['p','phil','philipp']:
        gal.inclination     = incl_philipp_get(gal,path,fname)
    
    # Custom coords
    if customcoords==True or customcoords=='True' or customcoords=='true':
        gal.center_position = coords_get(gal)
    elif isinstance(customcoords,str):
        if customcoords.lower() in ['p','phil','philipp']:
            gal.center_position = coords_philipp_get(gal,path,fname)
        else:
            print('tools.galaxy() : WARNING: `customcoords` invalid! Disabling custom coordinates. ')        
    
    # Custom vsys
    if custominc in ['p','phil','philipp','true','True',True]:
        gal.vsys     = vsys_philipp_get(gal,path,fname)
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
            
    # Custom distance
    if customdistance==True:
        dist = distance_get(gal)
        if np.isnan(dist.value):
            print('tools.galaxy() : Custom distance failed, as '+gal.name+' is not in the table!')
        else:
            gal.distance = dist
            
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
    
def mom0_get(gal,data_mode='',conbeam=None,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             masking='strict'):
    '''
    Parameters:
    -----------
    gal : Galaxy
    data_mode : str
    conbeam(=None) : Quantity (pc or arcsec)
        Desired resolution of map.
        Will attempt to grab preset resolutions.
        Will convolve from cube+mask if a preset
        is not available (!!!not implemented).
    return_mode(='data'):
        'data': returns data
        'path': returns path to mom0 file
        'hdul': returns HDU list
                (i.e. fits.open(path))
        'hdu': returns HDU
               (i.e. fits.open(path)[0])
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.mom0_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.mom0_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'    
    if masking.lower() in ['broad','strict']:
        path_mask = masking.lower()+'_maps/'
    elif masking.lower() in ['']:
        path_mask = ''
    else:
        raise ValueError('\''+masking+'\' is not a valid \'masking\' setting!')
        
    # Conbeam - convert to angle
    conbeam_filename = ''
    if str(conbeam).replace('.0 pc','pc') in ['500pc','750pc','1000pc','1250pc']:
        conbeam_filename = '_'+str(conbeam).replace('.0 pc','pc')
        conbeam = int(str(conbeam).replace('.0 pc','pc').replace('pc',''))*u.pc
    elif type(conbeam) is str:
        raise ValueError('tools.mom0_get() : \'conbeam\' must be in [\'500pc\',\'750pc\',\'1000pc\',\'1250pc\'] as a string.')
    if conbeam is not None:
        if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
            conbeam_width = conbeam.to(u.pc)                         # Beam width, in pc.
#            conbeam_angle = conbeam_width / gal.distance.to(u.pc) * u.rad  # Beam width, in radians.
#            conbeam_angle = conbeam_angle.to(u.arcsec)               # Beam width, in arcsec.
        elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            conbeam_angle = conbeam.to(u.arcsec)
        else:
            raise ValueError("'conbeam' must have units of pc or arcsec.")

    # Get the mom0 file. In K km/s.
    I_mom0=None
    I_mom0_hdul=None
    if data_mode=='7m':
        path = path7m+path_mask
        filename_7mtp = name+'_'+data_mode+'+tp_co21_'+masking+'_mom0'+conbeam_filename+'.fits'    # 7m+tp mom0. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_'+masking+'_mom0'+conbeam_filename+'.fits'    # 7m mom0. Less reliable.
#        filename_7mtp = name+'_'+data_mode+'+tp_co21_mom0.fits'    # 7m+tp mom0. Ideal.
#        filename_7m   = name+'_'+data_mode+   '_co21_mom0.fits'    # 7m mom0. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_mom0_hdul = copy.deepcopy(fits.open(path+filename_7mtp))
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            print('No 7m+tp mom0 found. Using 7m mom0 instead.')
            I_mom0_hdul = copy.deepcopy(fits.open(path+filename_7m))
        else:
            print(path+filename_7mtp)
    elif data_mode=='12m+7m':
        path = path12m+path_mask
        filename_12mtp = name+'_'+data_mode+'+tp_co21_'+masking+'_mom0'+conbeam_filename+'.fits'    # 12m+tp mom0. Ideal.
        filename_12m   = name+'_'+data_mode+   '_co21_'+masking+'_mom0'+conbeam_filename+'.fits'    # 12m mom0. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_mom0_hdul = copy.deepcopy(fits.open(path+filename_12mtp))
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            print('No 12m+7m+tp mom0 found. Using 12m+7m mom0 instead.')
            I_mom0_hdul = copy.deepcopy(fits.open(path+filename_12m))
    else:
        raise ValueError('tools.mom0_get() : Invalid data_mode-- No mom0 was found!')
    if conbeam is not None and conbeam_filename=='':
        # Convolve cube manually
        raise ValueError('tools.mom0_get() : Conbeam of '+str(conbeam)+' requested, but convolution not implemented.')
    if I_mom0_hdul is None:
        print('WARNING: No mom0 was found!')
        finalpath=None
        return I_mom0_hdul

    # Clean the header!
    I_mom0_hdul[0].data    # <-- This is necessary for some reason. The HDUL won't have any data otherwise.
    I_mom0_hdul[0].header = hdr_clean(I_mom0_hdul[0].header)
    
    if return_mode=='data':
        return I_mom0_hdul[0].data
    elif return_mode=='path':
        return finalpath
    elif return_mode.lower() in ['hdul','hdulist','hdu_list']:
        return I_mom0_hdul
    elif return_mode.lower() in ['hdu']:
        return I_mom0_hdul[0]
    else:
        print('tools.mom0_get() : Invalid "return_mode"! Must be "data", "path", or "hdu(l)".')

def mom1_get(gal,data_mode='',return_best=False, verbose=True,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             masking='strict',\
             folder_hybrid='jnofech_mom1_hybrid/'):
    '''
    return_best:
        False: returns mom1 (see return_mode for format)
        True:  returns mom1 (see return_mode for format), best_emom1 (str)
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to mom1 file
        'hdul': returns HDU list
                (i.e. fits.open(path))
        'hdu': returns HDU
               (i.e. fits.open(path)[0])
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.mom1_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.mom1_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'
    if data_mode not in ['7m','12m+7m','hybrid']:
        raise ValueError('tools.mom1_get() : \''+data_mode+'\' is not a valid data_mode!')
    if masking.lower() in ['broad','strict']:
        path_mask = masking.lower()+'_maps/'
    elif masking.lower() in ['']:
        path_mask = ''
    else:
        raise ValueError('\''+masking+'\' is not a valid \'masking\' setting!')
     
    # Get the mom1 file. In K km/s.
    I_mom1     = None
    I_mom1_7m  = None
    I_mom1_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_name = '7m'
        path = path7m+path_mask
        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_mom1.fits'    # 7m+tp mom1. Ideal.
        filename_7m   = name+'_'+data_mode_name+   '_co21_'+masking+'_mom1.fits'    # 7m mom1. Less reliable.
#        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_mom1.fits'    # 7m+tp mom1. Ideal.
#        filename_7m   = name+'_'+data_mode_name+   '_co21_mom1.fits'    # 7m mom1. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_mom1_7m = copy.deepcopy(fits.open(path+filename_7mtp,mode='update'))
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m+tp'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            if verbose==True:
                print('No 7m+tp mom1 found. Using 7m mom1 instead.')
            I_mom1_7m = copy.deepcopy(fits.open(path+filename_7m,mode='update'))
            I_mom1 = I_mom1_7m[0].data
            best_mom1_7m='7m'
        else:
            best_mom1_7m = 'None'
        I_mom1_hdul = I_mom1_7m
        best_mom1 = best_mom1_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m+path_mask
        filename_12mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_mom1.fits'    # 7m+tp mom1. Ideal.
        filename_12m   = name+'_'+data_mode_name+   '_co21_'+masking+'_mom1.fits'    # 7m mom1. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_mom1_12m = copy.deepcopy(fits.open(path+filename_12mtp,mode='update'))
            I_mom1 = I_mom1_12m[0].data
            best_mom1_12m='12m+7m+tp'      # Keeps track of whether 12m or 12m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            if verbose==True:
                print('No 12m+7m+tp mom1 found. Using 12m+7m mom1 instead.')
            I_mom1_12m = copy.deepcopy(fits.open(path+filename_12m,mode='update'))
            I_mom1 = I_mom1_12m[0].data
            best_mom1_12m='12m+7m'
        else:
            finalpath = None
            best_mom1_12m = 'None'
        I_mom1_hdul = I_mom1_12m
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
        hdul     = [fits.PrimaryHDU(I_mom1_hybrid,header=hdr),'Dummy list entry, so that I_mom1=hdul[0].data.']
        filename = name+'_co21_'+best_mom1+'_'+masking+'_mom1.fits'
        path = path12m+folder_hybrid
        finalpath = path+filename
        if os.path.isfile(path+filename)==False:
            print(path+filename)
            hdul[0].writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
        I_mom1_hdul = hdul

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
        elif return_mode.lower() in ['hdul','hdulist','hdu_list']:
            return I_mom1_hdul
        elif return_mode.lower() in ['hdu']:
            return I_mom1_hdul[0]
        else:
            print('tools.mom1_get() : Invalid "return_mode"! Must be "data", "path", or "hdu(l)".') 
    
def emom1_get(gal,data_mode='',return_best=False, verbose=True,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             masking='strict',\
             folder_hybrid='jnofech_mom1_hybrid/'):
    '''
    return_best:
        False: returns emom1 (see return_mode for format)
        True:  returns emom1 (see return_mode for format), best_emom1 (str)
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to emom1 file
        'hdul': returns HDU list
                (i.e. fits.open(path))
        'hdu': returns HDU
               (i.e. fits.open(path)[0])
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.emom1_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.emom1_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'
    if masking.lower() in ['broad','strict']:
        path_mask = masking.lower()+'_maps/'
    elif masking.lower() in ['']:
        path_mask = ''
    else:
        raise ValueError('\''+masking+'\' is not a valid \'masking\' setting!')
    
    # Get the emom1 file. In K km/s.
    I_emom1     = None
    I_emom1_7m  = None
    I_emom1_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_name = '7m'
        path = path7m+path_mask
        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_emom1.fits'    # 7m+tp emom1. Ideal.
        filename_7m   = name+'_'+data_mode_name+   '_co21_'+masking+'_emom1.fits'    # 7m emom1. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_emom1_7m = copy.deepcopy(fits.open(path+filename_7mtp,mode='update'))
            I_emom1 = I_emom1_7m[0].data
            best_emom1_7m='7m+tp'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            if verbose==True:
                print('No 7m+tp emom1 found. Using 7m emom1 instead.')
            I_emom1_7m = copy.deepcopy(fits.open(path+filename_7m,mode='update'))
            I_emom1 = I_emom1_7m[0].data
            best_emom1_7m='7m'
        else:
            best_emom1_7m = 'None'
        I_emom1_hdul = I_emom1_7m
        best_emom1 = best_emom1_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m+path_mask
        filename_12mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_emom1.fits'    # 7m+tp emom1. Ideal.
        filename_12m   = name+'_'+data_mode_name+   '_co21_'+masking+'_emom1.fits'    # 7m emom1. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_emom1_12m = copy.deepcopy(fits.open(path+filename_12mtp,mode='update'))
            I_emom1 = I_emom1_12m[0].data
            best_emom1_12m='12m+7m+tp'      # Keeps track of whether 12m or 12m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            if verbose==True:
                print('No 12m+7m+tp emom1 found. Using 12m+7m emom1 instead.')
            I_emom1_12m = copy.deepcopy(fits.open(path+filename_12m,mode='update'))
            I_emom1 = I_emom1_12m[0].data
            best_emom1_12m='12m+7m'
        else:
            finalpath = None
            best_emom1_12m = 'None'
        I_emom1_hdul = I_emom1_12m
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
        hdul     = [fits.PrimaryHDU(I_emom1_hybrid,header=hdr),'Dummy list entry, so that I_emom1=hdul[0].data.']
        filename = name+'_co21_'+best_emom1+'_'+masking+'_emom1.fits'
        path = path12m+folder_hybrid
        finalpath = path+filename
        if os.path.isfile(path+filename)==False:
            print(path+filename)
            hdul[0].writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
        I_emom1_hdul = hdul

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
        elif return_mode.lower() in ['hdul','hdulist','hdu_list']:
            return I_emom1_hdul
        elif return_mode.lower() in ['hdu']:
            return I_emom1_hdul[0]
        else:
            print('tools.emom1_get() : Invalid "return_mode"! Must be "data", "path", or "hdu(l)".') 

def tpeak_get(gal,data_mode='',conbeam=None,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             masking='strict'):
    '''
    Parameters:
    -----------
    gal : Galaxy
    data_mode : str
    conbeam(=None) : Quantity (pc or arcsec)
        Desired resolution of map.
        Will attempt to grab preset resolutions.
        Will convolve from cube+mask if a preset
        is not available (!!!not implemented).
    return_mode(='data'):
        'data': returns data
        'path': returns path to mom0 file
        'hdul': returns HDU list
                (i.e. fits.open(path))
        'hdu': returns HDU
               (i.e. fits.open(path)[0])
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.tpeak_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.tpeak_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'       
    if masking.lower() in ['broad','strict']:
        path_mask = masking.lower()+'_maps/'
    elif masking.lower() in ['']:
        path_mask = ''
    else:
        raise ValueError('\''+masking+'\' is not a valid \'masking\' setting!')
    
    # Conbeam - convert to angle
    conbeam_filename = ''
    if str(conbeam).replace('.0 pc','pc') in ['500pc','750pc','1000pc','1250pc']:
        conbeam_filename = '_'+str(conbeam).replace('.0 pc','pc')
        conbeam = int(str(conbeam).replace('.0 pc','pc').replace('pc',''))*u.pc
    elif type(conbeam) is str:
        raise ValueError('tools.tpeak_get() : \'conbeam\' must be in [\'500pc\',\'750pc\',\'1000pc\',\'1250pc\'] as a string.')
    if conbeam is not None:
        if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
            conbeam_width = conbeam.to(u.pc)                         # Beam width, in pc.
#            conbeam_angle = conbeam_width / gal.distance.to(u.pc) * u.rad  # Beam width, in radians.
#            conbeam_angle = conbeam_angle.to(u.arcsec)               # Beam width, in arcsec.
        elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            conbeam_angle = conbeam.to(u.arcsec)
        else:
            raise ValueError("'conbeam' must have units of pc or arcsec.")
            
    # Get the tpeak file. In K km/s.
    I_tpeak=None
    I_tpeak_hdul=None
    if data_mode=='7m':
        path = path7m+path_mask
        filename_7mtp = name+'_'+data_mode+'+tp_co21_'+masking+'_tpeak'+conbeam_filename+'.fits'    # 7m+tp tpeak. Ideal.
        filename_7m   = name+'_'+data_mode+   '_co21_'+masking+'_tpeak'+conbeam_filename+'.fits'    # 7m tpeak. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_tpeak_hdul = copy.deepcopy(fits.open(path+filename_7mtp))
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            print('No 7m+tp tpeak found. Using 7m tpeak instead.')
            I_tpeak_hdul = copy.deepcopy(fits.open(path+filename_7m))
    elif data_mode=='12m+7m':
        path = path12m+path_mask
        filename_12mtp = name+'_'+data_mode+'+tp_co21_'+masking+'_tpeak'+conbeam_filename+'.fits'    # 12m+tp tpeak. Ideal.
        filename_12m   = name+'_'+data_mode+   '_co21_'+masking+'_tpeak'+conbeam_filename+'.fits'    # 12m tpeak. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_tpeak_hdul = copy.deepcopy(fits.open(path+filename_12mtp))
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            print('No 12m+7m+tp tpeak found. Using 12m+7m tpeak instead.')
            I_tpeak_hdul = copy.deepcopy(fits.open(path+filename_12m))
        else:
            print('tools.tpeak_get(): tpeak maps missing. Calculating tpeak directly from cube.')
            cube = (cube_get(gal,data_mode).unmasked_data[:]).to(u.K).value
            finalpath = None
            I_tpeak = cube.max(axis=0)
            hdr = hdr_get(gal,data_mode,dim=2)
            hdr['BUNIT'] = 'K'
            I_tpeak_hdul = [fits.PrimaryHDU(I_tpeak,header=hdr),'Dummy list entry, so that I_tpeak=I_tpeak_hdul[0].data.']
    else:
        raise ValueError('tools.tpeak_get() : Invalid data_mode-- No tpeak was found!')
    if conbeam is not None and conbeam_filename=='':
        # Convolve cube manually
        raise ValueError('tools.tpeak_get() : Conbeam of '+str(conbeam)+' requested, but convolution not implemented.')
    if I_tpeak_hdul is None:
        print('WARNING: No tpeak was found!')
        return I_tpeak_hdul

    # Clean the header!
    I_tpeak_hdul[0].data    # <-- This is necessary for some reason. The HDUL won't have any data otherwise.
    I_tpeak_hdul[0].header = hdr_clean(I_tpeak_hdul[0].header)
    
    if return_mode=='data':
        return I_tpeak_hdul[0].data
    elif return_mode=='path':
        return finalpath
    elif return_mode.lower() in ['hdul','hdulist','hdu_list']:
        return I_tpeak_hdul
    elif return_mode.lower() in ['hdu']:
        return I_tpeak_hdul[0]
    else:
        print('tools.tpeak_get() : Invalid "return_mode"! Must be "data", "path", or "hdu(l)".')  

def noise_get(gal,data_mode='',cube=None,noisypercent=0.15,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/'):
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
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/'):
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

def hdr_clean(hdr):
    '''
    Cleans a 2D object's header,
    by removing all of its '3D' thingies.
    
    NOTE: If you're cleaning the header of an
        HDUlist and the data gets "wiped out"
        upon cleaning the header, there's a
        workaround.
        Instead of:
        "hdul.header = hdr_clean(hdul.header)",
        consider:
        "hdul.data
         hdul.header = hdr_clean(hdul.header)".
         
         This prevents the data from vanishing upon
         cleaning the header... for some reason.        
    '''
    if hdr['NAXIS']!=2:
        raise ValueError('tools.hdr_clean() : hdr[\'NAXIS\']='+str(hdr['NAXIS'])+', when it should be 2 to be cleanable.')
        
    for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
        if kw in hdr:
            del hdr[kw]
    for i in ['1','2','3']:
        for j in ['1', '2', '3']:
            if ('PC'+i+'_'+j) in hdr:
                del hdr['PC'+i+'_'+j]
    return hdr
            
def cube_get(gal,data_mode,return_best=False,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/'):
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

def mask_get(gal,data_mode,return_boolean=True,\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
             mask_foldername='/'):
    '''
    Parameters:
    gal : Galaxy
    data_mode : str
        '7m' or '12m'.
        Will try to fetch 7m+tp or 12m+tp 
        where available.
    return_boolean(=True) : bool
        Toggles whether to return the mask
        as a BooleanArrayMask (True) or
        as a spectral cube (False).
        
    Note that, if 'mask' is a BooleanArrayMask,
    you can mask cubes easily with
    'cube.with_mask(mask)'.
    '''
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
        raise ValueError('tools.mask_get() : Does not support 12m! Needs PHANGS-ALMA-LP/delivery/cubes support!')
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
        raise ValueError('tools.mask_get() : Does not support 12m! Needs PHANGS-ALMA-LP/delivery/cubes support!')
    # (!!!) END COPY HERE

    # Spectral Cube Mask
    mask=None
    if data_mode=='7m':
        path = path7m+mask_foldername
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
        path = path12m+mask_foldername
        filename = name+'_co21_'+data_mode+'+tp_mask.fits'
        if os.path.isfile(path+filename):
            mask = SpectralCube.read(path+filename)
        else:
            print('WARNING: \''+filename+'\' not found!')
    else:
        print('WARNING: Invalid data_mode-- No mask was found!')
        
    if mask is None:
        print('WARNING: No mask was found!')
        cube = cube_get(gal,data_mode,False,path7m,path12m)
#         from spectral_cube import LazyMask
#         lazymask = LazyMask(True, cube=cube)
        print('tools.mask_get() - Maybe create a BooleanArrayMask here, except everything is True?\nAs opposed to just returning None.')
        return mask
    else:
        if return_boolean==True:
            # Make sure mask and corresponding cube have same WCS.
            # Kind of redundant, but worth doing just in case?
            cube = cube_get(gal,data_mode,False,path7m,path12m)
            mask = BooleanArrayMask(mask=(mask.unmasked_data[:]==True), wcs=cube.wcs)
#             mask = BooleanArrayMask(mask=(mask.unmasked_data[:]==True), wcs=mask.wcs)   # Alternate version, if the cube and mask always have identical WCSs            
    return mask

def sfr_get(gal,hdr=None,conbeam=None,res='7p5',band_uv='nuv',band_ir='w3',autocorrect=False,\
            return_mode='data',\
            verbose=True,\
            path='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    '''
    Recommended: NUV band + WISE Band 3.
    
    The 'conbeam' is the resolution that
        you want the SFR map to be convolved
        to.
    The 'res' can be '7p5' or '15',
        i.e. 7.5" SFR data or 15" data.
        It just selects which resolution
        is read; it may be convolved afterwards.
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
        data if 7.5" is missing (which,
        really, just affects W4 data).
        Recommended to DISABLE if you need
        cubes to be convolved to same res.
        as SFR maps.
    return_mode : str
        'data': returns data (DEFAULT)
        'hdu': returns HDU object for data
               (NOT HDU list)
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
    hdr_copy = copy.deepcopy(hdr)  # Copy of the original header.
        # Certain steps of this code (e.g. getting the HDU from Projection object)
        #   will alter the header, so this is necessary so that the global 'hdr'
        #   doesn't get altered as well.
    map_uv      = band_get(gal,hdr_copy,band_uv,res,sfr_toggle=False)     # Galex UV band.
    map_ir      = band_get(gal,hdr_copy,band_ir,res,sfr_toggle=False)     # WISE IR band.
    
    # Actually generate the SFR map!
    if map_uv is not None and map_ir is not None:
        sfr     = (map_uv*uv_to_sfr + map_ir*ir_to_sfr)   # Sum of SFR contributions, in Msun/yr/kpc**2.
    elif map_uv is None and band_uv==None and map_ir is not None and band_ir!=None:
        # If UV is intentionally missing:
        print('(galaxytools.sfr_get())  WARNING: Only considering IR ('+band_ir+') component.')
        sfr     = (map_ir*ir_to_sfr)             # SFR from just IR contribution.
    elif map_ir is None and band_ir==None and map_uv is not None and band_uv!=None:
        # If IR is intentionally missing:
        print('(galaxytools.sfr_get())  WARNING: Only considering UV ('+band_uv+') component.')
        sfr     = (map_uv*uv_to_sfr)             # SFR from just UV contribution.
    else:
        print('(galaxytools.sfr_get())  WARNING: No '+str(res)+'" '+str(band_uv)\
              +'+'+str(band_ir)+' SFR map was found!')
        sfr = None
    
    # Autocorrect with 15" version?
    if sfr is None and res=='7p5' and autocorrect==True:
        print('(galaxytools.sfr_get())  WARNING: Unable to get 7.5" SFR map! Reverting to 15" instead.')
        sfr = sfr_get(gal,hdr,conbeam,'15',band_uv,band_ir,False,return_mode,path)
        return sfr

    if sfr is not None:
        # sfr is a Projection object right now. Turn it into an HDU, then convolve if necessary.
        sfr_hdu = sfr.hdu
        if conbeam!=None:
            sfr_hdu = convolve_2D(gal,sfr_hdu,conbeam)  # Convolved SFR map.
        if return_mode=='data':
            return sfr_hdu.data
        elif return_mode.lower() in ['hdu']:
            if verbose:
                print('tools.sfr_get() : WARNING: SFR HDU does not have a totally accurate header; BUNIT is missing, etc.')
            if band_uv not in ['nuv']:
                if verbose:
                    print('tools.sfr_get() : WARNING: SFR HDU does NOT have an accurate beam size (BMAJ)!')
            return sfr_hdu
        else:
            raise ValueError('tools.sfr_get() : Invalid "return_mode"! Must be "data" or "hdu".')
    else:
        return sfr


def band_get(gal,hdr=None,band='',res='15',sfr_toggle=False,\
             path='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
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
        map2d = Projection.from_hdu(copy.deepcopy(fits.open(filename)))        # Not necessarily in Msun/yr/kpc**2. Careful!
#         print(filename)
    else:
        print('(galaxytools.band_get()) WARNING: No map was found, for '+(sfr_toggle*'sfr_')+band+'_gauss'+res+'.fits')
        map2d = None
        return map2d
    
    if hdr!=None:
        hdr_copy = copy.deepcopy(hdr)  # Copy of the original header.
            # Certain steps of this code (e.g. getting the HDU from Projection object)
            #   will alter the header, so this is necessary so that the global 'hdr'
            #   doesn't get altered as well.
        map2d_final = map2d.reproject(hdr_copy)
    else:
        map2d_final = map2d
    if map2d_final is not None and ~isinstance(map2d_final,u.Quantity):
        map2d_final = map2d_final*u.K/u.K  # Converts to Quantity to prevent inconsistencies
    return map2d_final
    
def band_check(name,res,band1,band2='None',\
               folder_sfr  ='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    if band1.lower() in ['fuv+w4','nuv+w3'] and band2=='None':
        if band1.lower() in ['fuv+w4']:
            band1 = 'fuv'
            band2 = 'w4'
        else:
            band1 = 'nuv'
            band2 = 'w3'
    elif band1.lower() in ['fuv+w4','nuv+w3'] and band2!='None':
        raise ValueError('check_band() : Either make band1 a combination of bands, or put the second band into band2. Not both!')

    for band in [band1,band2]:
        if band.lower() in ['fuv','nuv','w2','w3','w4']:
            filename = folder_sfr+name.lower()+'_'+band.lower()+'_gauss'+res+'.fits'
            if os.path.isfile(filename):
                sfrmap = copy.deepcopy(fits.open(filename))
            else:
                sfrmap = None
        elif band=='None':
            sfrmap = 9999   # not None
        else:
            raise ValueError('galaxies_gen() : band=[\''+band+'\'] is not a valid band!')
        if band==band1:
            if sfrmap is not None:
                band_failed1 = False
            else:
                band_failed1 = True
        if band==band2:
            if sfrmap is not None:
                band_failed2 = False
            else:
                band_failed2 = True
    return ((not band_failed1) and (not band_failed2))  # i.e. "band_success"

#def peakvels_get(gal,data_mode='',cube=None,mask=None,quadfit=True,write=True,best_cube=None,\
##             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
#             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
#             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/',\
##             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
#             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/eros_masks/',\
#             path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/',\
#             folder_vpeak='jnofech_peakvels/'):
#    '''
#    Returns a 2D map of peak velocities.
#    Can use the quadratic-fit method described
#    in Teague & Foreman-Mackey 2018
#    (https://arxiv.org/abs/1809.10295), 
#    which improves accuracy for cubes of low
#    spectral resolution.
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#    data_mode='12m' or '7m' : str
#        Chooses where to find the output
#        file, based on the selected data.
#        A 'hybrid' mode also exists, which
#        combines 12m and 7m data into a
#        single map.
#    cube(=None) : SpectralCube
#        Spectral cube for the galaxy.
#    mask(=None) : SpectralCube OR Quantity 
#                  OR np.ndarray
#        3D boolean array of cube's resolution,
#        defining where the data is masked.
#    quadfit(=True) : bool
#        Enables the quadratic fit for finding 
#        peak velocity. This means the resulting
#        "peak velocity" will be far more 
#        reliable at low spectral resolutions, 
#        but comes at the cost of slower run 
#        times.
#    write(=True) : bool
#        If no peakvels map is found, this
#        toggles whether to write the output
#        peakvels to a .fits file, in the
#        `path(7/12)m_mask/jnofech_peakvels` 
#        path.
#    best_cube(=None) : str
#        Best image quality-- e.g. '7m', 
#        '7m+12m', etc., for the data cube.
#        Only necessary when write=True.
#        
#    Returns:
#    --------
#    peakvels : np.ndarray
#        2D map of peak velocities, in
#        km/s.
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#    if data_mode == '7m':
#        data_mode = '7m'
#    elif data_mode in ['12m','12m+7m']:
#        data_mode = '12m+7m'  
#    elif data_mode.lower() in ['both','hybrid']:
#        data_mode = 'hybrid'  
#    elif data_mode=='':
#        if verbose==True:
#            print('No data_mode set. Defaulted to 12m+7m.')
#        data_mode = '12m+7m'
#    # Stick with 12m cubes when 'hybrid' is selected!
##     data_mode_peakvels = data_mode
##     if data_mode=='hybrid':
##         data_mode = '12m+7m'
#        

#    # Read from file, if possible!
#    peakvels     = None
#    peakvels_7m  = None
#    peakvels_12m = None
#    if data_mode in ['7m','hybrid']:
#        data_mode_temp = '7m'
#        # Get filename.
#        if best_cube is None:
#            cube_discard, best_cube  = cube_get(gal,data_mode_temp,return_best=True)
#        if (best_cube is None) or (best_cube=='None'):
#            raise ValueError('Best data mode not specified! This is needed to find the peakvels .fits file.')
#        else:
#            filename  = name+'_'+best_cube+'_co21_'+masking+'_peakvels.fits'
#        # Read or write.
#        path=path7m_mask+folder_vpeak
#        if os.path.isfile(path+filename)==True:
#            peakvels_7m = copy.deepcopy(fits.open(path+filename))
#            peakvels    = peakvels_7m[0].data
#        else:
#            print(path+filename)
#            if write==True:
#                print('\''+filename+'\' does not exist. Generating and saving new peakvels map!')
#            else:
#                print('\''+filename+'\' does not exist. Generating new peakvels map!')
#            # Define cube and mask!
#            cube = cube_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
#            mask = mask_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
#            if (mask is not None) and (np.product(mask.shape)!=cube.size):
#                print('WARNING: Mask has different dimensions from cube!')
#                mask = None
#            peakvels = peakvels_gen(gal,data_mode_temp,cube,mask,quadfit,write,best_cube)
#            peakvels_7m = copy.deepcopy(fits.open(path+filename))   # A new file was just created!
#        best_cube_7m = best_cube
#        best_cube    = None                             # Cleanup, for 'hybrid' mode.
#    if data_mode in ['12m','12m+7m','hybrid']:
#        data_mode_temp = '12m+7m'
#        # Get filename.
#        if best_cube is None:
#            cube_discard, best_cube  = cube_get(gal,data_mode_temp,return_best=True)
#        if (best_cube is None) or (best_cube=='None'):
#            raise ValueError('Best data mode not specified! This is needed to find the peakvels .fits file.')
#        else:
#            filename  = name+'_co21_'+best_cube+'_'+masking+'_peakvels.fits'
##             filename  = name+'_'+best_cube+'_co21_peakvels.fits'
#        # Read or write.
#        path=path12m_mask+folder_vpeak
#        if os.path.isfile(path+filename)==True:
#            peakvels_12m = copy.deepcopy(fits.open(path+filename))
#            peakvels     = peakvels_12m[0].data
#        else:
#            if write==True:
#                print('\''+filename+'\' does not exist. Generating and saving new peakvels map!')
#            else:
#                print('\''+filename+'\' does not exist. Generating new peakvels map!')
#            # Define cube and mask!
#            cube = cube_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
#            mask = mask_get(gal,data_mode_temp,path7m=path7m,path12m=path12m)
#            if np.product(mask.shape)!=cube.size:
#                print('WARNING: Mask has different dimensions from cube!')
#                mask = None
#            peakvels = peakvels_gen(gal,data_mode_temp,cube,mask,quadfit,write,best_cube)
#            peakvels_12m = copy.deepcopy(fits.open(path+filename))   # A new file was just created!
#        best_cube_12m = best_cube
#        best_cube     = None                             # Cleanup, for 'hybrid' mode.
#        
#    # Combining the two, if hybrid mode is enabled!
#    if data_mode in ['hybrid']:
#        hdr = peakvels_12m[0].header
#        # Reproject the 7m map to the 12m's dimensions!
#        # Conveniently, the interpolation is also done for us.
#        peakvels_7m_modify = Projection.from_hdu(peakvels_7m)
#        peakvels_7m = peakvels_7m_modify.reproject(peakvels_12m[0].header)
#        # Convert to simple np arrays!
#        peakvels_12m, peakvels_7m = peakvels_12m[0].data, peakvels_7m.value
#        # COMBINE!
#        peakvels_mask = (np.isfinite(peakvels_12m) + np.isfinite(peakvels_7m)).astype('float')
#        peakvels_mask[peakvels_mask == 0.0] = np.nan    # np.nan where _neither_ 12m nor 7m have data.
#        peakvels_hybrid = np.nan_to_num(peakvels_12m) + np.isnan(peakvels_12m)*np.nan_to_num(peakvels_7m) \
#                                                      + peakvels_mask
#        peakvels = peakvels_hybrid
#        # SAVE!
#        best_cube = 'hybrid_'+best_cube_7m+'&'+best_cube_12m
#        hdr['BUNIT'] = 'km  s-1 '  # Write this instead of 'KM/S  '.
#        # Save header and data into a .fits file, if specified!
#        hdu      = fits.PrimaryHDU(peakvels_hybrid,header=hdr)
#        filename = name+'_co21_'+best_cube+'_'+masking+'_peakvels.fits'
#        path = path12m_mask+folder_vpeak
#        if os.path.isfile(path+filename)==False:
#            hdu.writeto(path+filename)
#        else:
#            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
#    if data_mode not in ['7m','12m+7m','hybrid']:
#        print('WARNING: Invalid data_mode-- No mom1 was found!')
#        peakvels = None
#        return peakvels

#    if peakvels is None:
#        if verbose==True:
#            print('WARNING: No peakvels was found!')

#    return peakvels

#def peakvels_gen(gal,data_mode='',cube=None,mask=None,quadfit=True,write=False,best_cube=None,\
##             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
#             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
#             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/',\
##             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
#             path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/eros_masks/',\
#             path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery/cubes/',\
#             folder_vpeak='jnofech_peakvels/'):
#    '''
#    Returns a 2D map of peak velocities.
#    Can use the quadratic-fit method described
#    in Teague & Foreman-Mackey 2018
#    (https://arxiv.org/abs/1809.10295), 
#    which improves accuracy for cubes of low
#    spectral resolution.
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#    data_mode='12m' or '7m' : str
#        Chooses where to save the output
#        file, based on the selected data.
#    cube(=None) : SpectralCube
#        Spectral cube for the galaxy.
#    mask(=None) : SpectralCube OR Quantity 
#                  OR np.ndarray
#        3D boolean array of cube's resolution,
#        defining where the data is masked.
#    quadfit(=True) : bool
#        Enables the quadratic fit for finding 
#        peak velocity. This means the resulting
#        "peak velocity" will be far more 
#        reliable at low spectral resolutions, 
#        but comes at the cost of slower run 
#        times.
#    write(=False) : bool
#        Toggles whether to write the output
#        peakvels to a .fits file, in the
#        `path(7/12)m_mask/jnofech_peakvels` 
#        path.
#    best_cube(=None) : str
#        Best image quality-- e.g. '7m', 
#        '7m+12m', etc., for the data cube.
#        Only necessary when write=True.
#        
#    Returns:
#    --------
#    peakvels : np.ndarray
#        2D map of peak velocities, in
#        km/s.
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#    if data_mode == '7m':
#        data_mode = '7m'
#    elif data_mode in ['12m','12m+7m']:
#        data_mode = '12m+7m'  
#    elif data_mode.lower() in ['both','hybrid']:
#        data_mode = 'hybrid'  
#    elif data_mode=='':
#        print('No data_mode set. Defaulted to 12m+7m.')
#        data_mode = '12m+7m'

#    if data_mode in ['7m', '12m+7m']:
#        # Get the cube, and spectral axis.
#        if cube is None:
#            cube = cube_get(gal,data_mode,path7m,path12m)
#        spec = cube.spectral_axis
#        # Mask the data!
#        if mask is None:
#            print('WARNING: Mask not defined. Using unmasked data; will apply spatial mask later.')
#            mask_provided = False
#            mask = np.ones(cube.size).reshape(cube.shape)
#            mask = BooleanArrayMask(mask=(mask==True), wcs=cube.wcs)
#        elif (isinstance(mask,u.quantity.Quantity) or isinstance(mask,np.ndarray)):
#            mask_provided = True
#            mask = BooleanArrayMask(mask=(mask==True), wcs=cube.wcs)
#        elif isinstance(mask,SpectralCube):
#            mask_provided = True
#            mask = BooleanArrayMask(mask=(mask.unmasked_data[:]==True), wcs=cube.wcs)
#        elif isinstance(mask,BooleanArrayMask):
#            mask_provided = True
#        else:
#            mask_provided = True #(?)
#            print('WARNING: Mask is not a Quantity, SpectralCube, BooleanArrayMask, or array. The code\'s probably about to crash.')    
#        cube_m = cube.with_mask(mask)
#        data = cube_m.filled_data[:].value
#        # Find peak indices from this.
#        data_copy = copy.deepcopy(data)
#        data_copy[np.isnan(data)] = -np.inf  # This prevents 'np.argmax()' from choking on the 'np.nan' values.
#        x0 = np.argmax(data_copy,axis=0)     # Indices of temperature peaks in spectral axis.
#        if quadfit==True:    
#            # Note: This I0 \equiv 'tpeak' will be identical to np.nanmax(data,axis=0).
#            I0 = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Peak intensity, in K.
#            Ip = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Intensity at (x0+1), in K.
#            Im = np.zeros(data[0].size).reshape(data.shape[1],data.shape[2])   # Intensity at (x0-1), in K.
#            # Peak indices, plus or minus 1. These are the two points on a spectrum immediately around the peak.
#            xp = x0+1
#            xm = x0-1
#            # If peak is at either end of the spectrum, then we can't really go higher or lower...
#            # ... but that's almost guaranteed to be noise anyways, so we can just use a bogus value.
#            xp[x0==(data.shape[0]-1)] = data.shape[0]-1
#            xm[x0==0] = 0
#            # Find the intensities, in K, at these indices!
#            for j in range(0,data.shape[1]):
#                for i in range(0,data.shape[2]):   
#                    I0[j,i] = data[x0[j,i],j,i]
#                    Ip[j,i] = data[xp[j,i],j,i]
#                    Im[j,i] = data[xm[j,i],j,i]
#            # Find the quadratic-fitted peak indices!
#            a0 = I0
#            a1 = 0.5*(Ip-Im)
#            a2 = 0.5*(Ip+Im-2*I0)
#            xmax = x0 - (a1/(2.*a2))   # Quad-fitted indices of peak velocity! They're not integers anymore.
#            # Generate 2D map of peak velocity, using these improved peak indices!
#            spec_interp = interpolate.interp1d(np.arange(spec.size),spec, fill_value='extrapolate')  # Units gone.
#            peakvels = spec_interp(xmax)
#        else:
#            # Generate 2D map of peak velocity, using default peak indices!
#            spec_interp = interpolate.interp1d(np.arange(spec.size),spec, fill_value='extrapolate')  # Units gone.
#            peakvels = spec_interp(x0)
#    elif data_mode in ['hybrid','both']:
#        raise ValueError("peakvels_gen() - 'data_mode=hybrid' should appear in peakvels_get(), not here!")
#    else:
#        raise ValueError("peakvels_gen() - invalid \'data_mode\'. Should be '7m', '12m', '12m+7m', or 'hybrid'.")

#    # Adding units back
#    peakvels = peakvels*spec.unit
#    peakvels = peakvels.to(u.km/u.s)
#    
#    # Masking with spatial mask, if no mask was provided
#    if mask_provided==False:
#        print('WARNING: No cube mask provided! Using spatial mask from mom1 map (default file location) instead.')
#        I_mom1 = mom1_get(gal,data_mode,False,False,'data')
##        print(peakvels)
#        print(I_mom1)
#        print(path7m)
#        peakvels[np.isnan(I_mom1)] = np.nan
#    
#    # Give it a header!
#    hdr = cube[0].header      # Take the header of a 2D slice from the cube.
#    hdr['BUNIT'] = 'km  s-1 ' # Write this instead of 'KM/S  '. The 'Projection.from_hdu' will accept these units!
#    
#    # Save header and data into a .fits file, if specified!
#    print('lalalala')
#    print(write)
#    if write==True:
#        hdu      = fits.PrimaryHDU(peakvels.value,header=hdr)
#        if best_cube is None:
#            print('WARNING: Best data mode not specified! Will assume "'+data_mode.lower()+'".')
#            best_cube = data_mode.lower()
#        
#        if data_mode in ['7m']:
#            filename = name+'_'+best_cube+'_co21_'+masking+'_peakvels.fits'
#            path=path7m_mask+folder_vpeak
#        elif data_mode in ['12m+7m','hybrid']:
#            filename = name+'_co21_'+best_cube+'_'+masking+'_peakvels.fits'
#            path=path12m_mask+folder_vpeak
#        
#        print(path+filename)
#        print('WEEEEEEEE')
#        if os.path.isfile(path+filename)==False:
#            hdu.writeto(path+filename)
#            print('aye')
#        else:
#            print('nye')
#            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.')    

#    return peakvels.value

def vpeak_get(gal,data_mode='',return_best=False, verbose=True,\
             return_mode='data',\
             path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/',\
             masking='strict',\
             folder_hybrid='jnofech_vpeak_hybrid/'):
    '''
    return_best:
        False: returns vpeak (see return_mode for format)
        True:  returns vpeak (see return_mode for format), best_evpeak (str)
    return_mode:
        'data': returns data (DEFAULT)
        'path': returns path to vpeak file
        'hdul': returns HDU list
                (i.e. fits.open(path))
        'hdu': returns HDU
               (i.e. fits.open(path)[0])
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if data_mode == '7m+tp':
        print('tools.vpeak_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'
    elif data_mode in ['12m+tp','12m+7m+tp']:
        print('tools.vpeak_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
        data_mode = '12m+7m'
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'
    if data_mode not in ['7m','12m+7m','hybrid']:
        raise ValueError('tools.vpeak_get() : \''+data_mode+'\' is not a valid data_mode!')
    if masking.lower() in ['broad','strict']:
        path_mask = masking.lower()+'_maps/'
    elif masking.lower() in ['']:
        path_mask = ''
    else:
        raise ValueError('\''+masking+'\' is not a valid \'masking\' setting!')
        
    # Get the vpeak file. In K km/s.
    I_vpeak     = None
    I_vpeak_7m  = None
    I_vpeak_12m = None
    if data_mode in ['7m','hybrid']:
        data_mode_name = '7m'
        path = path7m+path_mask
        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_vpeak.fits'    # 7m+tp vpeak. Ideal.
        filename_7m   = name+'_'+data_mode_name+   '_co21_'+masking+'_vpeak.fits'    # 7m vpeak. Less reliable.
#        filename_7mtp = name+'_'+data_mode_name+'+tp_co21_vpeak.fits'    # 7m+tp vpeak. Ideal.
#        filename_7m   = name+'_'+data_mode_name+   '_co21_vpeak.fits'    # 7m vpeak. Less reliable.
        if os.path.isfile(path+filename_7mtp):
            finalpath = path+filename_7mtp
            I_vpeak_7m = copy.deepcopy(fits.open(path+filename_7mtp,mode='update'))
            I_vpeak = I_vpeak_7m[0].data
            best_vpeak_7m='7m+tp'      # Keeps track of whether 7m or 7m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_7m):
            finalpath = path+filename_7m
            if verbose==True:
                print('No 7m+tp vpeak found. Using 7m vpeak instead.')
            I_vpeak_7m = copy.deepcopy(fits.open(path+filename_7m,mode='update'))
            I_vpeak = I_vpeak_7m[0].data
            best_vpeak_7m='7m'
        else:
            best_vpeak_7m = 'None'
        I_vpeak_hdul = I_vpeak_7m
        best_vpeak = best_vpeak_7m
    if data_mode in ['12m+7m','hybrid']:
        data_mode_name = '12m+7m'
        path = path12m+path_mask
        filename_12mtp = name+'_'+data_mode_name+'+tp_co21_'+masking+'_vpeak.fits'    # 7m+tp vpeak. Ideal.
        filename_12m   = name+'_'+data_mode_name+   '_co21_'+masking+'_vpeak.fits'    # 7m vpeak. Less reliable.
        if os.path.isfile(path+filename_12mtp):
            finalpath = path+filename_12mtp
            I_vpeak_12m = copy.deepcopy(fits.open(path+filename_12mtp,mode='update'))
            I_vpeak = I_vpeak_12m[0].data
            best_vpeak_12m='12m+7m+tp'      # Keeps track of whether 12m or 12m+tp is available. Returning this is optional.
        elif os.path.isfile(path+filename_12m):
            finalpath = path+filename_12m
            if verbose==True:
                print('No 12m+7m+tp vpeak found. Using 12m+7m vpeak instead.')
            I_vpeak_12m = copy.deepcopy(fits.open(path+filename_12m,mode='update'))
            I_vpeak = I_vpeak_12m[0].data
            best_vpeak_12m='12m+7m'
        else:
            finalpath = None
            best_vpeak_12m = 'None'
        I_vpeak_hdul = I_vpeak_12m
        best_vpeak = best_vpeak_12m
    if data_mode=='hybrid':
        # Fix both of their headers!
        for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
            del I_vpeak_7m[0].header[kw]
            del I_vpeak_12m[0].header[kw]
        for i in ['1','2','3']:
            for j in ['1', '2', '3']:
                del I_vpeak_7m[0].header['PC'+i+'_'+j]
                del I_vpeak_12m[0].header['PC'+i+'_'+j]
        hdr = I_vpeak_12m[0].header
        # Reproject the 7m map to the 12m's dimensions!
        # Conveniently, the interpolation is also done for us.
        I_vpeak_7m_modify = Projection.from_hdu(I_vpeak_7m)
        I_vpeak_7m = I_vpeak_7m_modify.reproject(I_vpeak_12m[0].header)
        # Convert to simple np arrays!
        I_vpeak_12m, I_vpeak_7m = I_vpeak_12m[0].data, I_vpeak_7m.value
        # COMBINE!
        I_vpeak_mask = (np.isfinite(I_vpeak_12m) + np.isfinite(I_vpeak_7m)).astype('float')
        I_vpeak_mask[I_vpeak_mask == 0.0] = np.nan    # np.nan where _neither_ I_vpeak_12m nor I_vpeak_7m have data.
        I_vpeak_hybrid = np.nan_to_num(I_vpeak_12m) + np.isnan(I_vpeak_12m)*np.nan_to_num(I_vpeak_7m) + I_vpeak_mask
        I_vpeak = I_vpeak_hybrid
        best_vpeak = 'hybrid_'+best_vpeak_7m+'&'+best_vpeak_12m
        
        # SAVE!
        hdr['BUNIT'] = 'km  s-1 '  # Write this instead of 'KM/S  '.
        # Save header and data into a .fits file, if specified!
        hdul     = [fits.PrimaryHDU(I_vpeak_hybrid,header=hdr),'Dummy list entry, so that I_vpeak=hdul[0].data.']
        filename = name+'_co21_'+best_vpeak+'_'+masking+'_vpeak.fits'
        path = path12m+folder_hybrid
        finalpath = path+filename
        if os.path.isfile(path+filename)==False:
            print(path+filename)
            hdul[0].writeto(path+filename)
        else:
            print('WARNING: Did not write to \''+path+filename+'\', as this file already exists.') 
        I_vpeak_hdul = hdul

    if I_vpeak is None:
        if verbose==True:
            print('WARNING: No vpeak was found!')
        return I_vpeak
    if return_best==True:
        return I_vpeak, best_vpeak
    else:
        if return_mode=='data':
            return I_vpeak
        elif return_mode=='path':
            return finalpath
        elif return_mode.lower() in ['hdul','hdulist','hdu_list']:
            return I_vpeak_hdul
        elif return_mode.lower() in ['hdu']:
            return I_vpeak_hdul[0]
        else:
            print('tools.vpeak_get() : Invalid "return_mode"! Must be "data", "path", or "hdu(l)".') 
    
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
        incl = 70.*u.deg
    if gal.name.lower()=='ngc3059':
        print('galaxytools.incl_get(): Using FAKE inclination value, for DiskFit! REMOVE LATER?')
        incl = 15.*u.deg
    return incl

def incl_philipp_get(gal,path='/media/jnofech/BigData/galaxies/drive_tables/',\
                     fname='philipp_Final_orientations_April16.txt'):
    '''
    Gets the inclination for the
    indicated galaxy, using Philipp's
    finalized values from IR imaging
    surveys (S4? Spitzer?)
    
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
    if os.path.isfile(path+fname):
        galaxies, x,x,x,x,x,x              = np.loadtxt(path+fname,dtype=str,unpack=True,skiprows=0)
        x, flags, incls,x,x,RAcens,Deccens = np.genfromtxt(path+fname,unpack=True)
    if name.lower() in galaxies:
        incl = incls[list(galaxies).index(name.lower())]*u.deg
    else:
        print('WARNING: Galaxy does not have custom inclination!')
    return incl
    
def vsys_philipp_get(gal,\
                    path='/media/jnofech/BigData/galaxies/drive_tables/',\
                    fname='philipp_Final_orientations_April16.txt'):
    '''
    Gets the vsys for the
    indicated galaxy.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        
    Returns:
    --------
    vsys : Quantity (float*u.km/u.s)
        Systemic velocity, in km/s.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        print('New Galaxy object created for '+name+'!')
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Get vsys!
    vsys = gal.vsys
    
    # OVERRIDES
    if os.path.isfile(path+fname):
        galaxies, x,x,x,x,x,x              = np.loadtxt(path+fname,dtype=str,unpack=True,skiprows=0)
        x, flags, incls,x,vsyss,RAcens,Deccens = np.genfromtxt(path+fname,unpack=True)
    if name.lower() in galaxies:
        vsys = vsyss[list(galaxies).index(name.lower())]*u.km/u.s
    else:
        print('WARNING: Galaxy does not have custom vsys!')
    return vsys

def coords_philipp_get(gal,\
                      path='/media/jnofech/BigData/galaxies/drive_tables/',\
                      fname='philipp_Final_orientations_April16.txt'):
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
    if os.path.isfile(path+fname):
        galaxies, x,x,x,x,x,x              = np.loadtxt(path+fname,dtype=str,unpack=True,skiprows=0)
        x, flags, incls,x,x,RAcens,Deccens = np.genfromtxt(path+fname,unpack=True)
    if name.lower() in galaxies:
        RA_cen  = RAcens[list(galaxies).index(name.lower())]
        Dec_cen = Deccens[list(galaxies).index(name.lower())]
        # Turn into RA, Dec!
        gal.center_position = SkyCoord(RA_cen,Dec_cen,unit=(u.deg,u.deg), frame='fk5')
    else:
        print('WARNING: Galaxy does not have custom central coords!')

    return gal.center_position

#def coords_get(gal):
#    '''
#    Gets the central RA and Dec
#    for the indicated galaxy, if 
#    the provided ones look a bit off.
#    NOTE: You'll probably need to
#    save this into the headers as well,
#    along with corresponding pixel
#    coordinates, if you're working with
#    a function that grabs central
#    coords in some way!
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#        
#    Returns:
#    --------
#    RA : Quantity (float*u.deg)
#        Right ascension, in degrees.
#    Dec : Quantity (float*u.deg)
#        Declination, in degrees.
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#        print('New Galaxy object created for '+name+'!')
#        gal = Galaxy(name.upper())
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#    
#    raise ValueError('tools.py : coords_get() (NOT the philipp one) should no longer be used.')
#    # Get coords!
#    RA_cen = gal.center_position.ra.value
#    Dec_cen = gal.center_position.dec.value
#    
#    # OVERRIDES
#    # Done by eyeballing mom0 maps, mom1 maps, and optical images!
#    # Custom central coordinates, in case provided values are very incorrect.
#    if gal.name.lower()=='ic5332':
#        # Central coords look more reasonable, but unsure of effect on rotcurve.
#        RA_cen,Dec_cen = 353.61275414100163, Dec_cen
#    if gal.name.lower()=='ngc1385':
#        # Noticeable improvement on RC!
#        RA_cen,Dec_cen = 54.371, -24.502
##     if gal.name.lower()=='ngc1559':
##         # Discard; made RC worse!
##         RA_cen,Dec_cen = 64.39710994945374,-62.78098504774272
#    if gal.name.lower()=='ngc1637':
#        # RC improved!
#        RA_cen,Dec_cen = 70.36711177761812,-2.8589855372620074
#    if gal.name.lower()=='ngc1792':
#        # RC slightly worse than default, but makes more physical sense.
#        RA_cen,Dec_cen = 76.30633574756656,-37.97824645945883
#    if gal.name.lower()=='ngc1809':
#        # mom1-vs-R makes less sense, but RC fit improved?? Unsure where center should be, exactly.
#        RA_cen,Dec_cen = 75.51005604304831,-69.56348536585699
#    if gal.name.lower()=='ngc2090':
#        # Better RC, better mom1 fit!
#        RA_cen,Dec_cen = 86.75809108234091,-34.250416010642695
#    if gal.name.lower()=='ngc2283':
#        # Better RC, better mom1 fit!
#        RA_cen,Dec_cen = 101.469735789141,-18.210409767930265
##     if gal.name.lower()=='ngc2775':
##         # Central coords look more reasonable, and mom1 data is now focused into one "trend"
##         # (with much more scatter) rather than being in two tighter trends. RC looks better.
##         RA_cen,Dec_cen = 137.582, 7.037970066070557
#    if gal.name.lower()=='ngc3511':
#        # Better RC fit, better mom1 fit!
#        RA_cen,Dec_cen = 165.8484352408927,-23.08680662885883
#    if gal.name.lower()=='ngc4207':
#        # Central coords look more reasonable, and mom1 data is focused more into one of
#        # the two "trends". RC looks better, but the error bars are much, much worse.
#        RA_cen,Dec_cen = 183.8765752788386, 9.584930419921875
#    if gal.name.lower()=='ngc4293':
#        # Slightly jaggier RC.
#        RA_cen,Dec_cen = 185.30302908075126,18.3829153738402
#    if gal.name.lower()=='ngc4254':
#        # Central coords look MUCH more reasonable.
#        # Much of the mom1 data is still below zero for some reason. Improved, but nowhere near perfect.
#        RA_cen,Dec_cen = 184.7074087094364, gal.center_position.dec.value
#    if gal.name.lower()=='ngc4424':
#        # RC even more wild, not that it was reliable to begin with.
#        RA_cen,Dec_cen = 186.7959698271931,9.421769823689191
#    if gal.name.lower()=='ngc4457':
#        # Much better RC and mom1 fit!
#        RA_cen,Dec_cen = 187.24653811871107,3.570680697523369
#    if gal.name.lower()=='ngc4569':
#        # No improvement, despite it being a "clean"-looking galaxy. No idea why mom1 fit is still split.
#        RA_cen,Dec_cen = 189.20679998931678,13.163193855398312
##     if gal.name.lower()=='ngc4571':
##         # Better mom1 fit, although DF still hates it for some reason.
##         RA_cen,Dec_cen = 189.2340004103119,14.219281570910422
#    if gal.name.lower()=='ngc4654':
#        # Improved RC and mom1 fit!
#        RA_cen,Dec_cen = 190.98570142483396,13.12708187068814
#    if gal.name.lower()=='ngc4694':
#        # No effect.
#        RA_cen,Dec_cen = 192.06253044717437,10.984342977376253
#    if gal.name.lower()=='ngc4731':
#        # Central coords look somewhat more reasonable. RC is horribly jagged and dips into negatives,
#        # but is still a huge improvement.
#        RA_cen,Dec_cen = 192.750, -6.39
#    if gal.name.lower()=='ngc4781':
#        # Slightly _worse_ RC and mom1 fit, although central coords make more physical sense.
#        RA_cen,Dec_cen = 193.59747846280533,-10.535709989241946
#    if gal.name.lower()=='ngc4826':
#        # No noticeable improvement.
#        RA_cen,Dec_cen = 194.1812342003185,21.68321257606272   # New (should be the same, but makes RC worse?!)
##         RA_cen,Dec_cen = 194.1847900357396,21.68321257606272  # Old
#    if gal.name.lower()=='ngc4951':
#        # Big improvement on RC and mom1 fit!
#        RA_cen,Dec_cen = 196.28196323269697,-6.493484077561956
#    if gal.name.lower()=='ngc5042':
#        # Improved RC and mom1 fit!
#        RA_cen,Dec_cen = 198.8787057293563,-23.98319382130314
#    if gal.name.lower()=='ngc5068':
#        # Central seems better if you compare mom1 image to other NED images, but unsure if the
#        # new coords are more reasonable. Mom1 data went from 2 trends to 1 trend+more scatter.
#        # RC is somewhat jaggier, but still has very small errors.
#        RA_cen,Dec_cen = 199.73321317663428, -21.045
#    if gal.name.lower()=='ngc5530':
#        # Improved RC and mom1 fit!
#        RA_cen,Dec_cen = 214.61009028871453,-43.386535568706506
#    if gal.name.lower()=='ngc5643':
#        # Slightly improved RC and mom1 fit!
#        RA_cen,Dec_cen = 218.169014845381,-44.17440210179905
#    if gal.name.lower()=='ngc6744':
#        # Improved RC and mom1 fit!
#        RA_cen,Dec_cen = 287.4432981513924,-63.857521145661536
#    if gal.name.lower()=='ngc7496':
#        RA_cen,Dec_cen = 347.4464952115924,-43.42790798487375
#    
#    # Turn into RA, Dec!
#    gal.center_position = SkyCoord(RA_cen,Dec_cen,unit=(u.deg,u.deg), frame='fk5')
#    return gal.center_position
    
def logmass_get(gal=None,path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1'):
    '''
    Returns log10(*Stellar* mass / Msun)
    for the specified galaxy, OR
    for every galaxy in 
    galaxies_list if a galaxy is not
    specified.
    
    Parameters:
    -----------
    gal=None : str or Galaxy or list of strings
        Galaxy object/name, or list of galaxy names.
    
    Returns:
    --------
    logmstar : float or array
        All galaxy masses corresponding
        to `galaxy_list`, OR the
        mass of the single galaxy specified.
    '''
    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data

    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    elif isinstance(gal,list):
        galaxies_list = gal
#        print('tools.get_logmstar() : No single galaxy selected! Returning entire array of galaxy masses.')
        logmasses = np.zeros(len(galaxies_list))
        for i in range(0,len(galaxies_list)):
            name = galaxies_list[i]
            if name.upper() in table.field('NAME'):
                logmasses[i] = table.field('LOGMSTAR')[list(table.field('NAME')).index(name.upper())]
            else:
                print('tools.logmass_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
                logmasses[i] = np.nan * u.Mpc
        return logmasses
    else:
        raise ValueError("'gal' must be a str or Galaxy or a list of names!")
    
    if name.upper() in table.field('NAME'):
        logmass = table.field('LOGMSTAR')[list(table.field('NAME')).index(name.upper())]
    else:
        print('tools.logmass_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
        logmass = np.nan * u.Mpc
    return logmass
    
def distance_get(gal=None,path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1'):
    '''
    Returns distance (in Mpc)
    for the specified galaxy, OR
    for every galaxy in 
    galaxies_list if a galaxy is not
    specified.
    
    Parameters:
    -----------
    gal=None : str or Galaxy or list of strings
        Galaxy object/name, or list of galaxy names.
    
    Returns:
    --------
    dist : float or array
        All galaxy distances corresponding
        to `galaxy_list`, OR the
        distance of the single galaxy specified.
    '''
    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data

    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
    elif isinstance(gal,list):
        galaxies_list = gal
#        print('tools.distance_get() : No single galaxy selected! Returning entire array of galaxy distances.')
        distances = np.zeros(len(galaxies_list))*u.Mpc
        for i in range(0,len(galaxies_list)):
            name = galaxies_list[i]
            if name.upper() in table.field('NAME'):
                distances[i] = table.field('DIST')[list(table.field('NAME')).index(name.upper())] * u.Mpc
            else:
                print('tools.distance_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
                distances[i] = np.nan * u.Mpc
        return distances
    else:
        raise ValueError("'gal' must be a str or Galaxy or a list of names!")
    
    if name.upper() in table.field('NAME'):
        dist = table.field('DIST')[list(table.field('NAME')).index(name.upper())] * u.Mpc
    else:
        print('tools.distance_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
        dist = np.nan * u.Mpc
    return dist

#def TFvelocity_get(gal):
#    '''
#    Returns predicted Tully-
#    Fisher velocity (not-projected)
#    for the specified galaxy.
#    
#    Parameters:
#    -----------
#    gal : int or Galaxy
#        Galaxy object.
#    
#    Returns:
#    --------
#    TFv : Quantity
#        Approximate rotational velocity
#        of galaxy.
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")

#    if name.lower()=='ic1954':
#        TFv = 129.3896852*u.km/u.s
#    if name.lower()=='ic5273':
#        TFv = 126.183459*u.km/u.s
#    if name.lower()=='ic5332':
#        TFv = 122.6781319*u.km/u.s
#    if name.lower()=='ngc0628':
#        TFv = 166.1088534*u.km/u.s
#    if name.lower()=='ngc0685':
#        TFv = 131.2281377*u.km/u.s
#    if name.lower()=='ngc1087':
#        TFv = 129.6414964*u.km/u.s
#    if name.lower()=='ngc1097':
#        TFv = 211.1742643*u.km/u.s
#    if name.lower()=='ngc1300':
#        TFv = 211.7199579*u.km/u.s
#    if name.lower()=='ngc1317':
#        TFv = 182.5707277*u.km/u.s
#    if name.lower()=='ngc1365':
#        TFv = 223.2490649*u.km/u.s
#    if name.lower()=='ngc1385':
#        TFv = 157.4249651*u.km/u.s
#    if name.lower()=='ngc1433':
#        TFv = 198.1045651*u.km/u.s
#    if name.lower()=='ngc1511':
#        TFv = 140.4395746*u.km/u.s
#    if name.lower()=='ngc1512':
#        TFv = 187.8490435*u.km/u.s
#    if name.lower()=='ngc1546':
#        TFv = 163.1179861*u.km/u.s
#    if name.lower()=='ngc1559':
#        TFv = 135.9387902*u.km/u.s
#    if name.lower()=='ngc1566':
#        TFv = 205.4559423*u.km/u.s
#    if name.lower()=='ngc1637':
#        TFv = 127.155985*u.km/u.s
#    if name.lower()=='ngc1672':
#        TFv = 168.2787683*u.km/u.s
#    if name.lower()=='ngc1792':
#        TFv = 168.0101957*u.km/u.s
#    if name.lower()=='ngc1809':
#        TFv = 140.5480859*u.km/u.s
#    if name.lower()=='ngc2090':
#        TFv = 144.4697148*u.km/u.s
#    if name.lower()=='ngc2283':
#        TFv = 139.2615752*u.km/u.s
#    if name.lower()=='ngc2566':
#        TFv = 193.6596487*u.km/u.s
#    if name.lower()=='ngc2775':
#        TFv = 206.0931337*u.km/u.s
#    if name.lower()=='ngc2835':
#        TFv = 133.6302972*u.km/u.s
#    if name.lower()=='ngc2903':
#        TFv = 178.6525867*u.km/u.s
#    if name.lower()=='ngc2997':
#        TFv = 185.2073733*u.km/u.s
#    if name.lower()=='ngc3059':
#        TFv = 171.4108286*u.km/u.s
#    if name.lower()=='ngc3137':
#        TFv = 130.6415472*u.km/u.s
#    if name.lower()=='ngc3239':
#        TFv = 131.6274418*u.km/u.s
#    if name.lower()=='ngc3351':
#        TFv = 168.9887324*u.km/u.s
#    if name.lower()=='ngc3507':
#        TFv = 175.4601676*u.km/u.s
#    if name.lower()=='ngc3511':
#        TFv = 124.2545764*u.km/u.s
#    if name.lower()=='ngc3521':
#        TFv = 207.0739332*u.km/u.s
#    if name.lower()=='ngc3596':
#        TFv = 110.4244053*u.km/u.s
#    if name.lower()=='ngc3621':
#        TFv = 182.3324516*u.km/u.s
#    if name.lower()=='ngc3626':
#        TFv = 173.2528181*u.km/u.s
#    if name.lower()=='ngc3627':
#        TFv = 188.9890834*u.km/u.s
#    if name.lower()=='ngc4207':
#        TFv = 120.7242452*u.km/u.s
#    if name.lower()=='ngc4254':
#        TFv = 184.2006293*u.km/u.s
#    if name.lower()=='ngc4293':
#        TFv = 180.4140181*u.km/u.s
#    if name.lower()=='ngc4298':
#        TFv = 144.8225231*u.km/u.s
#    if name.lower()=='ngc4303':
#        TFv = 198.5801286*u.km/u.s
#    if name.lower()=='ngc4321':
#        TFv = 200.0481604*u.km/u.s
#    if name.lower()=='ngc4424':
#        TFv = 92.35864869*u.km/u.s
#    if name.lower()=='ngc4457':
#        TFv = 170.2458161*u.km/u.s
#    if name.lower()=='ngc4535':
#        TFv = 184.7751102*u.km/u.s
#    if name.lower()=='ngc4536':
#        TFv = 165.5247113*u.km/u.s
#    if name.lower()=='ngc4540':
#        TFv = 125.7450402*u.km/u.s
#    if name.lower()=='ngc4548':
#        TFv = 194.1613802*u.km/u.s
#    if name.lower()=='ngc4569':
#        TFv = 214.7993831*u.km/u.s
#    if name.lower()=='ngc4571':
#        TFv = 143.2840114*u.km/u.s
#    if name.lower()=='ngc4579':
#        TFv = 220.2774839*u.km/u.s
#    if name.lower()=='ngc4654':
#        TFv = 163.3954767*u.km/u.s
#    if name.lower()=='ngc4689':
#        TFv = 157.0624416*u.km/u.s
#    if name.lower()=='ngc4694':
#        TFv = 130.9081556*u.km/u.s
#    if name.lower()=='ngc4731':
#        TFv = 113.1246446*u.km/u.s
#    if name.lower()=='ngc4781':
#        TFv = 134.678573*u.km/u.s
#    if name.lower()=='ngc4826':
#        TFv = 154.4822491*u.km/u.s
#    if name.lower()=='ngc4941':
#        TFv = 146.7452837*u.km/u.s
#    if name.lower()=='ngc4951':
#        TFv = 115.4472648*u.km/u.s
#    if name.lower()=='ngc5042':
#        TFv = 115.9642888*u.km/u.s
#    if name.lower()=='ngc5068':
#        TFv = 107.0286549*u.km/u.s
#    if name.lower()=='ngc5128':
#        TFv = 223.8369277*u.km/u.s
#    if name.lower()=='ngc5134':
#        TFv = 164.5652803*u.km/u.s
#    if name.lower()=='ngc5248':
#        TFv = 157.5482845*u.km/u.s
#    if name.lower()=='ngc5530':
#        TFv = 149.1771141*u.km/u.s
#    if name.lower()=='ngc5643':
#        TFv = 170.2398255*u.km/u.s
#    if name.lower()=='ngc6300':
#        TFv = 184.7512441*u.km/u.s
#    if name.lower()=='ngc6744':
#        TFv = 237.2941833*u.km/u.s
#    if name.lower()=='ngc7456':
#        TFv = 90.10775682*u.km/u.s
#    if name.lower()=='ngc7496':
#        TFv = 144.3432589*u.km/u.s
#    
#    return TFv

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
    else:
        raise ValueError("'gal' must be a str or galaxy!")

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
        elif check_has_bar==True:
            return 'Yes'
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

    hdr = hdr_get(gal,data_mode)
    
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
   
   
#def info(gal,conbeam=None,data_mode='',hasmask=False,sfr_autocorrect=False):
#    '''
#    Returns basic info from galaxies.
#    Astropy units are NOT attached to outputs.
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#    conbeam=None : u.quantity.Quantity
#        Width of the beam in pc or ",
#        if you want the output to be
#        convolved.
#    data_mode='12m' or '7m' : str
#        Chooses either 12m data or 7m
#        data.
#    hasmask : bool
#        Determines whether a cube
#        mask is available.
#    sfr_autocorrect=False : bool
#        Attempts to get a 15" SFR
#        map if the 7.5" one fails.
#        
#    Returns:
#    --------
#    hdr : fits.header.Header
#        Header for the galaxy.
#    beam : float
#        Beam width, in deg.
#    I_mom0 : np.ndarray
#        0th moment, in K km/s.
#    I_mom1 : np.ndarray
#        Velocity, in km/s.
#    (if hasmask==True) vpeak : np.ndarray
#        Peak velocity, in km/s.
#    I_tpeak : np.ndarray
#        Peak temperature, in K.
#    cube : SpectralCube
#        Spectral cube for the galaxy.
#    (if hasmask==True) mask : SpectralCube
#        Mask for 'cube'.
#    sfr : np.ndarray
#        2D map of the SFR, in Msun/kpc^2/yr.
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#        
#    if data_mode == '7m+tp':
#        print('tools.cube_get(): WARNING: Changing data_mode from 7m+tp to 7m. Will try to select 7m+tp regardless.')
#        data_mode = '7m'
#    elif data_mode in ['12m','12m+7m']:
#        data_mode = '12m+7m'
#    elif data_mode in ['12m+tp','12m+7m+tp']:
#        print('tools.cube_get(): WARNING: Changing data_mode from '+data_mode+' to 12m. Will try to select 12m+tp regardless.')
#        data_mode = '12m+7m'
#    
#    I_mom0 = mom0_get(gal,data_mode,return_mode='hdu')
#    I_mom1 = mom1_get(gal,data_mode)
#    I_tpeak = tpeak_get(gal,data_mode,return_mode='hdu')
#    hdr = hdr_get(gal,data_mode,dim=2)
#    beam = hdr['BMAJ']                       # In degrees. This is beam size BEFORE convolution,
##                                              #   and we'll need this for the LoS velocity dispersion
##                                              #   (rc.linewidth_iso()).
##     beam_arcsec = (beam*u.deg).to(u.arcsec)  # In arcsec. We want this to be LOWER than the SFR map's 7.5"
##                                              #    beamwidth (i.e. higher resolution), but this often fails
##                                              #    and we need to use 15" SFR maps instead.
#    
##     # Choose appropriate resolution for SFR map, changing 'conbeam' to match it if necessary.
##     res='7p5'
##     if beam_arcsec > 7.5*u.arcsec and conbeam is not None:
##         print('(galaxytools.info())     WARNING: Beam is '+str(beam_arcsec)+', and we want to convolve.')
##         print('                                  This will use a 15" SFR map instead of 7.5"!')
##         res='15'
##         if conbeam==7.5*u.arcsec:
##             print('(galaxytools.info())              We\'ll also use a 15" conbeam.')
##             conbeam = 15.*u.arcsec
#    
#    # Get SFR at this resolution.
#    sfr, sfr_best, conbeam_final = sfr_combine(gal,conbeam,data_mode,return_mode='data')
#    conbeam = conbeam_final
#    
#    # Get cube+mask!
#    cube,bestcube = cube_get(gal,data_mode,return_best=True)
#    if hasmask==True:
#        mask          = mask_get(gal,data_mode)
#        # Get peakvels!
#        peakvels = peakvels_get(gal,data_mode,cube,mask,True,True,bestcube)
#    
#    
#    # CONVOLUTION, if enabled:
#    if conbeam!=None:
#        hdr, cube = cube_convolved(gal,conbeam,data_mode) # CONVOLVED cube, with replacement header.
#        I_mom0  = convolve_2D(gal,I_mom0,conbeam)   # Convolved mom0, from 2D map rather than cube. Maybe unused, as long as the mask is functioning properly.
#        I_tpeak = convolve_2D(gal,I_tpeak,conbeam)  # Same but for tpeak. Maybe unused, as long as the mask is functioning properly.
#    # Ver 1: From convolved cube (proper). REQUIRES MASK
#    if hasmask==True:
##     Convolve the cube, mask it, and then generate mom0/tpeak/etc with it.
##     Replicating the mom0, tpeak in the folder EXACTLY requires some "blurring" of the mask
##     which is kinda difficult to explain, so I'll skip that step.
#        cube2 = cube.with_mask(mask)
#        I_mom0 = cube2.moment0().to(u.K*u.km/u.s).value
#        I_tpeak = cube2.max(axis=0).to(u.K).value
#        return hdr,beam,conbeam,I_mom0,I_mom1,peakvels,I_tpeak,cube,mask,sfr,sfr_best
#    else:
##        raise ValueError('tools.info() - hasmask=False. However, mom0 and tpeak cannot be properly convolved without it. (Is crashing necessary? We did this before.)')
##        I_mom0 = cube.moment0().to(u.K*u.km/u.s).value
##        I_tpeak = cube.max(axis=0).to(u.K).value
#        if conbeam!=None:
#            raise ValueError('tools.info() - convolving data without mask. Not the proper order of operations, so it will be subtly incorrect.')
#        return hdr,beam,conbeam,I_mom0,I_mom1,I_tpeak,cube,sfr,sfr_best
#        
##     # Ver 2: From convolved 2D map
##     # Mask the mom0, tpeak data to match the cube's mask
##     cubeslice = cube[0].value   # 2D slice of cube data. Should theoretically have same mask as all slices
##     I_mom0.data[~np.isfinite(cubeslice)] = np.nan
##     I_tpeak.data[~np.isfinite(cubeslice)] = np.nan
##     if hasmask==True:
##         return hdr,beam,I_mom0.data,I_mom1,peakvels,I_tpeak.data,cube,mask,sfr
##     else:
##         return hdr,beam,I_mom0.data,I_mom1,I_tpeak.data,cube,sfr

def info(gal,data_mode='',conbeam=None,hasmask=False,masking='strict'):
    '''
    Returns basic info from galaxies.
    Astropy units are NOT attached to outputs.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    data_mode='12m' or '7m' : str
        Chooses either 12m data or 7m
        data.
    conbeam=None : u.quantity.Quantity
        Width of the beam in pc or ",
        if you want the output to be
        convolved.
        Can grab preset resolutions of
        500,750,1000,1250pc.
    hasmask : bool
        Determines whether a cube
        mask is available.
        
    Returns:
    --------
    hdr : fits.header.Header
        Unmodified header for the galaxy.
    beam : float
        Beam width, in deg.
    I_mom0 : np.ndarray
        0th moment, in K km/s.
    I_mom1 : np.ndarray
        Velocity, in km/s.
    vpeak : np.ndarray
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
    
    I_mom0 = mom0_get(gal,data_mode,conbeam,return_mode='data',masking=masking)
    I_mom1 = mom1_get(gal,data_mode,masking=masking)
    vpeak = vpeak_get(gal,data_mode,masking=masking)
    I_tpeak = tpeak_get(gal,data_mode,conbeam,return_mode='data',masking=masking)
    hdr = hdr_get(gal,data_mode,dim=2)
    beam = hdr['BMAJ']
    
    # Get SFR at this resolution.
    sfr, sfr_best = sfr_combine(gal,data_mode,conbeam,return_mode='data',return_best=True)
    
    # Get cube+mask!
    cube,bestcube = cube_get(gal,data_mode,return_best=True)
    if hasmask==True:
        raise ValueError('tools.info() : \'hasmask\' set to True. Need to implement 2D map convolution with cube+mask, and vpeak generation with mask!')
        mask          = mask_get(gal,data_mode)
    
    # CONVOLUTION, if enabled:
    # Covered by the XXXX_get() functions themselves.
    
    if hasmask==True:
        return hdr, I_mom0, I_mom1, vpeak, I_tpeak, cube, mask, sfr
    else:
        return hdr, I_mom0, I_mom1, vpeak, I_tpeak, cube, sfr

def sfr_combine(gal,data_mode='',conbeam=None,return_mode='data', return_best=False):
    '''
    Returns "combined" SFR map (e.g. nuv+w3 scaled to
    fuv+w4), convolved to the desired resolution.
    If the desired resolution is too high for the SFR map,
    'None' is returned instead.
    
    Parameters:
    -----------
    gal : str or galaxy
        Galaxy.
    data_mode : str
        Used for grabbing a header, which
        the SFR map will be reprojected to.
    conbeam(=None) : Quantity (pc or arcsec)
        Desired resolution of SFR map.
    return_mode(='data') : str
        Can return as 'data' (np.array)
        or 'hdu' (HDU object).
    return_best(=False) : bool
        Returns the best combination of SFR maps
        as a string afterwards.
        
    Returns:
    --------
    return_best=False:
        sfr
    return_best=True:
        sfr,sfr_best
    '''
    with silence():
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
        hdr = hdr_get(gal,data_mode,2)
        
    # Conbeam - convert to angle
    conbeam_filename = ''
    if conbeam in ['500pc','750pc','1000pc','1250pc']:
        conbeam_filename = '_'+conbeam
        conbeam = int(conbeam.replace('pc',''))*u.pc
    if conbeam is not None:
        if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
            conbeam_width = conbeam.to(u.pc)                         # Beam width, in pc.
            conbeam_angle = conbeam_width / gal.distance.to(u.pc) * u.rad  # Beam width, in radians.
            conbeam_angle = conbeam_angle.to(u.arcsec)               # Beam width, in arcsec.
        elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            conbeam_angle = conbeam.to(u.arcsec)
        else:
            raise ValueError("'conbeam' must have units of pc or arcsec.")
    
    
    # Get useful SFR maps
    # (Ideally, sfr_final = sfr(nuv+w3, 7.5") * (sfr(fuv+w4, 15") / sfr(nuv+w3, 15")) )
    with silence():
        res = '7p5'
        sfr_nuv =     sfr_get(name,hdr,conbeam=None,res=res, band_uv='nuv',band_ir='w3',\
                      autocorrect=False,return_mode='hdu',verbose=False)  # 7.5" ~or 15~"; Best spatial structure
        sfr_scale1 =  sfr_get(name,hdr,conbeam=None,res='15',band_uv='fuv',band_ir='w4',\
                      autocorrect=False,return_mode='hdu',verbose=False)  # 15" fuv+w4, for correcting the scaling
        sfr_scale2 =  sfr_get(name,hdr,conbeam=None,res='15',band_uv='nuv',band_ir='w3',\
                      autocorrect=False,return_mode='hdu',verbose=False)  # 15" nuv+w3, for correcting the scaling
    # COMBINE!
    if sfr_nuv is not None and sfr_scale1 is not None:
        sfr_best = '7.5" nuv+w3, corrected'
        sfr_final = Projection.from_hdu(sfr_nuv) * (sfr_scale1.data)/(sfr_scale2.data)
    elif sfr_nuv is None and sfr_scale1 is None:
        sfr_best = 'missing'
        print('tools.sfr_combine() : fuv+w4 and nuv+w3 both missing.')
        sfr_final = None
    elif sfr_nuv is None:
        sfr_best = '15" fuv'
        sfr_final = Projection.from_hdu(sfr_scale1)
        raise ValueError('tools.sfr_combine() - lolwtf (nuv+w3 missing but fuv+w4 present-- which makes no sense!)')
    elif sfr_scale1 is None:
        sfr_best = res.replace('p','.')+'" nuv+w3'
        sfr_final = Projection.from_hdu(sfr_nuv)
        sfr_final = None  # Actually, don't bother-- better to return a nonsensical SFR map.
        print('tools.sfr_combine() : fuv+w4 missing. We only want galaxies with nuv+w3 AND fuv+w4!')
    else:
        raise ValueError('tools.sfr_combine() - lolwtf')

    # Check that header is behaving correctly
    if sfr_final is not None:
        if sfr_final.hdu.header['BMAJ']!=sfr_nuv.header['BMAJ']:
            raise ValueError('tools.sfr_combine() - Header information was lost while combining maps. Investigate.')
    
    sfr = sfr_final
    
    if sfr is not None:
        # sfr is a Projection object right now. Turn it into an HDU, then convolve if necessary.
        sfr_hdu = sfr.hdu
        if conbeam!=None:
            # CONVOLUTION, if enabled:
            if np.isclose(((sfr_hdu.header['BMAJ']*u.deg).to(u.arcsec)).value, conbeam_angle.value):
                print('tools.sfr_combine() - sfr beam ('+str((sfr_hdu.header['BMAJ']*u.deg).to(u.arcsec))+') is very close to desired conbeam ('+str(conbeam)+'). Will not convolve!')
            else:
                if ((sfr_hdu.header['BMAJ']*u.deg).to(u.arcsec)).value > conbeam.value:
                    sfr_best = 'too low-res ('+str(conbeam_width)+')'
                    sfr = None
                    if return_best==True:
                        return sfr, sfr_best
                    else:
                        return sfr
                else:
                    sfr_hdu = convolve_2D(gal,sfr_hdu,conbeam_angle)  # Convolved sfr_hdu map.
        if return_mode=='data':
            sfr = sfr_hdu.data
        elif return_mode.lower() in ['hdu']:
            sfr = sfr_hdu
        else:
            print('tools.sfr_combine() : Invalid "return_mode"! Must be "data" or "hdu".')
    if return_best==True:
        return sfr, sfr_best
    else:
        return sfr
    
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
    rad : u.Quantity (array-like)
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

def sigmas(gal,hdr=None,I_mom0=None,I_tpeak=None,alpha=6.2,data_mode='',mapmode='mom1',sigmode=''):
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
        Only used for pixel sizes.
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
    x, rad = rc.rotmap(gal,header=hdr,data_mode=data_mode,mapmode=mapmode,rcmode='best')
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
    Sigma = alpha*I_mom0   # Units: Msun pc^-2
    
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

#def cube_convolved(gal,conbeam,data_mode='',\
##                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
#                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
#                  path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
#    '''
#    Extracts the mom0 and tpeak maps from
#        a convolved data cube.
#    If pre-convolved mom0/tpeak/cube data
#        already exists on the PHANGs Drive,
#        then they will be used instead.
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#    conbeam : float
#        Convolution beam width, in pc 
#        OR arcsec. Must specify units!
#    data_mode='12m' or '7m' : str
#        Chooses either 12m data or 7m
#        data.
#        
#    Returns:
#    --------
#    hdrc : fits.header.Header
#        Header for the galaxy's convolved
#        moment maps.
#    I_mom0c : np.ndarray
#        0th moment, in K km/s.
#    I_tpeakc : np.ndarray
#        Peak temperature, in K.
#    cubec : SpectralCube
#        Spectral cube for the galaxy,
#        convolved to the resolution indicated
#        by "conbeam".
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")

#    resolutions = np.array([60,80,100,120,500,750,1000])*u.pc   # Available pre-convolved resolutions,
#                                                                #    in PHANGS-ALMA-v1p0
#    # Units for convolution beamwidth:
#    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
#        if conbeam not in resolutions:
#            conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
#        else:                            # Setting conbeam_filename to use int, for pre-convolved maps
#            conbeam_filename = str(int(conbeam.to(u.pc).value))+'pc'
#    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
#        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
#    else:
#        raise ValueError("'conbeam' must have units of pc or arcsec.")

#    # Read cube
#    if data_mode=='7m':
#        path = path7m
#        filename = path+'cube_convolved/'+name.lower()+'_7m_co21_pbcorr_round_k_'\
#                                                       +conbeam_filename+'.fits'
#        # May be '7m' or '7m+tp', but we'll just name them all as '7m' for simplicity.
#        if os.path.isfile(filename):
#            cubec = SpectralCube.read(filename)
#            cubec.allow_huge_operations=True
#        else:
#            raise ValueError(filename+' does not exist.')
#        I_mom0c = cubec.moment0().to(u.K*u.km/u.s)  # Unused, except for header
##         I_tpeakc = cubec.max(axis=0).to(u.K)
#        hdrc = I_mom0c.header
#    elif data_mode in ['12m','12m+7m']:
#        path = path12m
#        if conbeam not in resolutions:
#            filename = path+'cube_convolved/'+name.lower()+'_co21_12m+7m+tp_pbcorr_round_k_'\
#                                                           +conbeam_filename+'.fits'
#            if os.path.isfile(filename):
#                cubec = SpectralCube.read(filename)
#                cubec.allow_huge_operations=True
#            else:
#                raise ValueError(filename+' does not exist.')
#            I_mom0c = cubec.moment0().to(u.K*u.km/u.s)  # Unused, except for header
##             I_tpeakc = cubec.max(axis=0).to(u.K)
#            hdrc = I_mom0c.header
#        else:    # If pre-convolved 3D data (mom0, tpeak, cube) exist:
#            raise ValueError('tools.cube_convolved() : OUTDATED CODE - Change address from galaxies/phangsdata to proper PHANGS-v1p0 folder!')
#            I_mom0c  = fits.getdata('phangsdata/'+name.lower()+'_co21_12m+7m+tp_mom0_'+conbeam_filename+'.fits')*u.K*u.km/u.s
#            I_tpeakc = fits.getdata('phangsdata/'+name.lower()+'_co21_12m+7m+tp_tpeak_'+conbeam_filename+'.fits')*u.K
#            filename = 'phangsdata/'+name.lower()+'_co21_12m+7m+tp_pbcorr_round_k_'+conbeam_filename+'.fits'
#            if os.path.isfile(filename):
#                cubec = SpectralCube.read(filename)
#                cubec.allow_huge_operations=True
#            else:
#                raise ValueError(filename+' does not exist.')
#            print( "IMPORTANT NOTE: This uses pre-convolved .fits files from Drive.")
#            I_mom0c_DUMMY = cubec.moment0().to(u.K*u.km/u.s)
#            hdrc = I_mom0c_DUMMY.header
#    else:
#        print('ERROR: No data_mode selected in galaxytools.convolve_cube()!')
#        
##     return hdrc,I_mom0c.value, I_tpeakc.value, cubec
#    return hdrc, cubec

def convolve_2D(gal,map2d,conbeam):
    '''
    Returns 2D map (e.g. SFR), convolved 
    to a beam width "conbeam".
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
        Only needed for converting
        "distance" conbeam to "angle"
        conbeam.
    map2d : HDU or HDUlist
        The map (e.g. SFR) that needs to 
        be convolved.
    conbeam : float
        Convolution beam width, in pc 
        OR arcsec. Must have units!

    Returns:
    --------
    map2d_convolved : HDU
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
        conbeam_angle = conbeam_width / gal.distance.to(u.pc) * u.rad  # Beam width, in radians.
        conbeam_angle = conbeam_angle.to(u.arcsec)               # Beam width, in arcsec.
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_angle = conbeam.to(u.arcsec)
    else:
        raise ValueError("'conbeam' must have units of pc or arcsec.")
    
    if isinstance(map2d, HDUList):
        # Not actually necessary; Projection handles HDU or HDUlist just fine either way
        map2dl = map2d
        map2d = map2dl[0]
        print('tools.convolve_2D() - Warning: 2D map for '+gal.name+' was inputted as an HDUlist, but will be returned as an HDU.')
    # Create Projection of 2D map, which can then be convolved
    map2d_proj = Projection.from_hdu(map2d)
    
    # Create a beam object, and then convolve the 2D map to it!
    bm = Beam(major=conbeam_angle,minor=conbeam_angle)    # Actual "beam" object
    
    # Convolve the cube!
    map2d_convolved = map2d_proj.convolve_to(bm)
    
    return map2d_convolved.hdu

#def convolve_cube(gal,cube,conbeam,data_mode='',\
##                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
#                  path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/delivery_v3p3/cubes/',\
#                  path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/'):
#    '''
#    Convolves a cube over a given beam, and
#    then generates and returns the moment
#    maps. The convolved cube is saved as well.
#    
#    Parameters:
#    -----------
#    gal : str OR Galaxy
#        Name of galaxy, OR Galaxy
#        object.
#    cube : SpectralCube
#        Spectral cube for the galaxy.
#    conbeam : float
#        Beam width, in pc OR arcsec.
#        Must specify units!
#    data_mode='12m' or '7m' : str
#        Chooses where to save the output
#        file, based on the selected data.
#            
#    Returns:
#    --------
#    cubec : SpectralCube
#        Spectral cube for the galaxy,
#        convolved to the resolution indicated
#        by "conbeam".
#    '''
#    if isinstance(gal,Galaxy):
#        name = gal.name.lower()
#    elif isinstance(gal,str):
#        name = gal.lower()
#        gal = galaxy(name.upper())
#    else:
#        raise ValueError("'gal' must be a str or galaxy!")
#        
#    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
#        conbeam_width = conbeam.to(u.pc)                     # Beam width in pc.
#        conbeam_angle = conbeam_width / gal.distance.to(u.pc) * u.rad
#        conbeam_angle = conbeam_angle.to(u.arcsec)
#        conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
#    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
#        conbeam_angle = conbeam.to(u.arcsec)                 # Beam width in arcsec.
#        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
#    else:
#        raise ValueError("'beam' must have units of pc or arcsec.")
#    
#    bm = Beam(major=conbeam_angle,minor=conbeam_angle)    # Actual "beam" object, used for convolving cubes
#    print(bm)
#    
#    # Convolve the cube!
#    cube = cube.convolve_to(bm)
#    
#    # Never convolve the cube again!
#    if data_mode=='7m':
#        path = path7m
#        filename = path+'cube_convolved/'+name.lower()+'_7m_co21_pbcorr_round_k_'\
#                                                       +conbeam_filename+'.fits'
#        # May be '7m' or '7m+tp', but we'll just name them all as '7m' for simplicity.
#    elif data_mode in ['12m','12m+7m']:
#        path = path12m
#        filename = path+'cube_convolved/'+name.lower()+'_co21_12m+7m+tp_pbcorr_round_k_'\
#                                                       +conbeam_filename+'.fits'
#    print('Saving convolved cube to... '+filename)
#    if os.path.isfile(filename):
#        os.remove(filename)
#        print(filename+" has been overwritten.")
#    cube.write(filename)
#    
#    return cube

#def gaussian(beam_pixwidth):
##    ____  ____   _____  ____  _      ______ _______ ______ 
##   / __ \|  _ \ / ____|/ __ \| |    |  ____|__   __|  ____|
##  | |  | | |_) | (___ | |  | | |    | |__     | |  | |__   
##  | |  | |  _ < \___ \| |  | | |    |  __|    | |  |  __|  
##  | |__| | |_) |____) | |__| | |____| |____   | |  | |____ 
##   \____/|____/|_____/ \____/|______|______|  |_|  |______|
##
## astropy's Gaussian2DKernel does the same job, except better and with more options.

#    '''
#    Returns a square 2D Gaussian centered on
#    x=y=0, for a galaxy "d" pc away.
#    
#    Parameters:
#    -----------
#    beam : float
#        Desired width of gaussian, in pc.
#    d : float
#        Distance to galaxy, in pc.
#        
#    Returns:
#    --------
#    gauss : np.ndarray
#        2D Gaussian with width "beam".    
#    '''
#    axis = np.linspace(-4*beam_pixwidth,4*beam_pixwidth,int(beam_pixwidth*8))
#    x, y = np.meshgrid(axis,axis)
#    d = np.sqrt(x*x+y*y)
#    
#    sigma, mu = beam_pixwidth, 0.0
#    g = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) / (sigma*np.sqrt(2.*np.pi)))
#    return g

def galaxies_gen(data_mode='7m',data_mode_sfr='either',\
                 exclude_missing_sfr=[''],\
                 exclude_low_incl=False,\
                 exclude_custom=['NGC3239'],\
                 include_sfr=[''],\
                 include_missing_sfr=[''],\
                 include_low_incl=False,\
                 include_custom=[''],\
                 sfr_min_res=999999*u.arcsec,\
                 cube_min_res=999999*u.arcsec,\
                 min_res=None,\
                 folder_cubes='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
                 folder_sfr  ='/media/jnofech/BigData/PHANGS/Archive/galex_and_wise/'):
    '''
    Generates a list of galaxy names based on input parameters.
    Will operate in EXCLUSION or INCLUSION mode, based on which custom
    inputs are changed.
    
    Parameters:
    -----------
        data_mode(='7m') : str
            '7m'  - selects from galaxies with 7m data
            '12m' - selects from galaxies with 12m data
        data_mode_sfr(='either') : str
            '7p5'    - selects from galaxies with 7.5" SFR data
            '15'     - selects from galaxies with 15" SFR data
            'either' - selects from galaxies if criteria are met in 7.5" **OR** 15" SFR data
            'both'   - selects from galaxies if criteria are met in 7.5" **AND** 15" SFR data

        EXCLUSIONS (Excludes certain galaxies from the set specified above):
        -----------
        exclude_low_incl(=False) : bool
            Excludes face-on galaxies.
        exclude_missing_sfr(=['']) : list
            List of SFR map components ('fuv','w4','nuv','w3',
            'fuv+w4','nuv+w3'). If a galaxy DOESN'T have data in ONE of these
            SFR map components in 'data_mode_sfr' resolution, it is EXCLUDED.
        exclude_custom(=['NGC3239']) : list
            Excludes any galaxy in this list. Does NOT trigger exclusion mode.
            
        INCLUSIONS (Includes ONLY certain galaxies, provided they're in the set specified above):
        -----------
        include_low_incl(=False) : bool
            If a galaxy has low incl, it is INCLUDED.
        include_sfr(=[]) : list
            List of SFR map components ('fuv','w4','nuv','w3',
            'fuv+w4','nuv+w3'). If a galaxy DOES have data in ALL of these
            SFR map components in 'data_mode_sfr' resolution, it is INCLUDED.
        include_custom(=[]) : list
            Includes any galaxy in this list.
    '''
    # Resolution restrictions - convert to angle
    if min_res is None:
        if sfr_min_res in ['500pc','750pc','1000pc','1250pc']:
            sfr_min_res = int(sfr_min_res.replace('pc',''))*u.pc
        if sfr_min_res.unit in {u.pc, u.kpc, u.Mpc}:
            sfr_min_res = sfr_min_res.to(u.pc)                         # Maximum beam width, in pc.
        elif sfr_min_res.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            sfr_min_res = sfr_min_res.to(u.arcsec)                     # Maximum beam width, in arcec.
        else:
            raise ValueError("'sfr_min_res' must have units of pc or arcsec.")
        if cube_min_res in ['500pc','750pc','1000pc','1250pc']:
            cube_min_res = int(cube_min_res.replace('pc',''))*u.pc
        if cube_min_res.unit in {u.pc, u.kpc, u.Mpc}:
            cube_min_res = cube_min_res.to(u.pc)                         # Maximum beam width, in pc.
        elif cube_min_res.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            cube_min_res = cube_min_res.to(u.arcsec)                     # Maximum beam width, in arcec.
        else:
            raise ValueError("'cube_min_res' must have units of pc or arcsec.")
    else:
        # Overrides both sfr_min_res and cube_min_res!
        if min_res in ['500pc','750pc','1000pc','1250pc']:
            min_res = int(min_res.replace('pc',''))*u.pc
        if min_res.unit in {u.pc, u.kpc, u.Mpc}:
            sfr_min_res = min_res.to(u.pc)                         # Maximum beam width, in pc.
            cube_min_res = min_res.to(u.pc)                         # Maximum beam width, in pc.
        elif min_res.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
            sfr_min_res = min_res.to(u.arcsec)                     # Maximum beam width, in arcec.
            cube_min_res = min_res.to(u.arcsec)                     # Maximum beam width, in arcec.
        else:
            raise ValueError("'min_res' must have units of pc or arcsec.")
            
    # Select inclusion/exclusion mode:
    mode = ''
    if (include_sfr!=[''] or include_missing_sfr!=[''] or include_low_incl!=False or include_custom!=['']):
        mode='inclusion'
        if include_sfr!=[''] and include_missing_sfr!=['']:
            raise ValueError('fancyplot.galaxies_gen() : Cannot include both SFR and missing-SFR! Pick one, not both!')
    if (exclude_missing_sfr!=[''] or exclude_low_incl!=False):
        if mode=='inclusion':
            raise ValueError('fancyplot.galaxies_gen() : Cannot specify EXCLUSIONS and INCLUSIONS at the same time! Check inputs.')
        mode='exclusion'
    
    # Workaround for if inclusion/exclusion lists are simply strings:
    consideration_list = [exclude_missing_sfr,exclude_custom,include_missing_sfr,include_sfr,include_custom]
    for i in range(0,len(consideration_list)):
        if type(consideration_list[i])==str:
            consideration_list[i] = list([consideration_list[i]])
    exclude_missing_sfr,exclude_custom,include_missing_sfr,include_sfr,include_custom = consideration_list
    
    # Base sets:
    # 7m Dataset
    galaxies_7m    = ['IC1954','IC5273','IC5332','NGC0628','NGC0685',\
                       'NGC1087','NGC1097','NGC1300','NGC1317','NGC1365',\
                       'NGC1385','NGC1433','NGC1511','NGC1512','NGC1546',\
                       'NGC1559','NGC1566','NGC1637','NGC1672','NGC1792',\
                       'NGC1809','NGC2090','NGC2283','NGC2566','NGC2775',\
                       'NGC2835','NGC2903','NGC2997','NGC3059','NGC3137',\
                       'NGC3239','NGC3351','NGC3507','NGC3511','NGC3521',\
                       'NGC3596','NGC3621','NGC3626','NGC3627','NGC4207',\
                       'NGC4254','NGC4293','NGC4298','NGC4303','NGC4321',\
                       'NGC4424','NGC4457','NGC4496A','NGC4535','NGC4536',\
                       'NGC4540','NGC4548','NGC4569','NGC4571','NGC4579',\
                       'NGC4654','NGC4689','NGC4694','NGC4731','NGC4781',\
                       'NGC4826','NGC4941','NGC4951','NGC5042','NGC5068',\
                       'NGC5128','NGC5134','NGC5248','NGC5530','NGC5643',\
                       'NGC6300','NGC6744','NGC7456','NGC7496']

    # 12m Dataset
    # PHANGS-ALMA-v1p0
    # galaxies_list = ['IC5332',  'NGC0628','NGC1672','NGC2835','NGC3351',\
    #                  'NGC3627', 'NGC4254','NGC4303','NGC4321','NGC4535',\
    #                  'NGC5068', 'NGC6744']
    # PHANGS-ALMA-LP/delivery/cubes
    galaxies_12m    = ['IC1954','IC5273','IC5332','NGC0628','NGC0685',\
                        'NGC1317','NGC1365','NGC1511','NGC1672','NGC1809',\
                        'NGC2090','NGC2283','NGC2566','NGC2775','NGC2835',\
                        'NGC3059','NGC3137','NGC3239','NGC3351','NGC3507',\
                        'NGC3511','NGC3596','NGC3626','NGC3627','NGC4207',\
                        'NGC4254','NGC4293','NGC4298','NGC4303','NGC4321',\
                        'NGC4424','NGC4457','NGC4496A','NGC4535','NGC4540','NGC4548',\
                        'NGC4569','NGC4571','NGC4579','NGC4654','NGC4689',\
                        'NGC4694','NGC4826','NGC4941','NGC4951','NGC5042',\
                        'NGC5068','NGC5134','NGC5248','NGC5530','NGC5643',\
                        'NGC6300','NGC6744','NGC7456','NGC7496']
    
    # SFR res settings:
    if data_mode_sfr in ['7p5','7.5']:
        res = ['7p5']
        res_quantity = [7.5]*u.arcsec
    elif data_mode_sfr in ['15']:
        res = ['15']
        res_quantity = [15]*u.arcsec
    elif data_mode_sfr in ['either','both']:
        res = ['7p5','15']
        res_quantity = [7.5,15]*u.arcsec
    else:
        raise ValueError('galaxies_gen() : data_mode_sfr=\''+str(data_mode_sfr)+'\' is invalid')

    # SELECT BASE SET OF GALAXIES.
    if data_mode=='12m':
        galaxies_set = galaxies_12m
    elif data_mode=='7m':
        galaxies_set = galaxies_7m
    else:
        raise ValueError('galaxies_gen() : data_mode=\''+data_mode+'\' not supported! (May include +tp later?)')
        
    # SELECT GALAXIES WITH GOOD/BAD INCLINATIONS.
    galaxies_lowincl = ['IC5332', 'NGC0628', 'NGC1317', 'NGC1365', 'NGC1433', 'NGC1672',
                        'NGC2566', 'NGC3059', 'NGC3507', 'NGC3596', 'NGC3626', 'NGC4303',
                        'NGC4548', 'NGC4731', 'NGC5134', 'NGC5643']
    galaxies_highincl = [e for e in galaxies_set if e not in galaxies_lowincl]     # Galaxies not in low_incl
    
    # SELECT BANDS BASED ON MODE
    bands = 'null'
    if mode=='inclusion':
        bands = list(np.sort(list(set(np.append(include_sfr,include_missing_sfr)))))
        bands = [e for e in bands if e not in ['']]    # Deletes [''] that may show up by default
        bands
    if mode in ['exclusion']:
        bands = exclude_missing_sfr
        bands = [e for e in bands if e not in ['']]    # Deletes [''], but probably redundant
    if mode in ['']:  # DEFAULT; no specific inclusions/exclusions
        bands = ['fuv','nuv','w3','w4']                # Check them all!
#     print('Checking: res='+str(res)+',\n          bands='+str(bands)+',\n          mode='+mode)

    # Check which galaxies are missing SFR maps (initialization)
    galaxies_detectedsfr = np.array([0]*len(galaxies_set)*len(res)).reshape(len(res),len(galaxies_set))
                        # ^  Counts the number of SFR maps detected for each galaxy,
                        #    after checking all checkable bands and resolutions.
                        #        exclusion mode : Galaxy excluded if galaxies_detectedsfr[i]<(nbands*nresolutions)
                        #        inclusion mode : Galaxy included if galaxies_detectedsfr[i]==(nbands*nresolutions)
                        #        inclusion mode, missing : Galaxy included if 
                        #                                  galaxies_detectedsfr[i]<(nbands*nresolutions)
    # Initialize best SFR+CUBE resolutions in each galaxy
    galaxies_set_b = list(np.copy(galaxies_set))               # Backup set of galaxies
    galaxies_bestres = [999999]*len(galaxies_set_b)*u.arcsec   # Best SFR resolution for each galaxy.
    galaxies_bestcuberes = [999999]*len(galaxies_set_b)*u.pc     # Best CUBE resolution for each galaxy
    # Define minimum allowed resolution for each galaxy's SFR+CUBE maps, in " OR pc
    galaxies_sfr_min_res = [sfr_min_res.value]*len(galaxies_set_b)*(sfr_min_res.unit)
    galaxies_cube_min_res = [cube_min_res.value]*len(galaxies_set_b)*(cube_min_res.unit)
    # Read Philipp's galaxy distances
    gal_distances = distance_get(galaxies_set_b)         # Distances of each galaxy, in Mpc
    # Convert SFR resolution restrictions to "
    if galaxies_sfr_min_res.unit==u.pc:
        galaxies_sfr_min_res = (galaxies_sfr_min_res / gal_distances.to(u.pc) * u.rad).to(u.arcsec)  # To arcsec.
    elif galaxies_sfr_min_res.unit==u.arcsec:
        galaxies_sfr_min_res  # Do nothing.
    else:
        raise ValueError('galaxies_sfr_min_res must be in pc or arcsec.') 
    # Convert CUBE resolution restrictions to pc
    if galaxies_cube_min_res.unit==u.pc:
        galaxies_cube_min_res  # Do nothing.
    elif galaxies_cube_min_res.unit==u.arcsec:
        galaxies_cube_min_res = galaxies_cube_min_res.to(u.rad).value * gal_distances.to(u.pc)  # To pc.
    else:
        raise ValueError('galaxies_cube_min_res must be in pc or arcsec.')
    
    for k in range(0,len(res)):
        for j in range(0,len(bands)):
            for i in range(0,len(galaxies_set)):
                name = galaxies_set[i]
                # Define maximum allowed beam width for this galaxy
                # If a useful combination of bands is already specified:
                if bands[j].lower() in ['fuv+w4','nuv+w3']:
                    if bands[j].lower() in ['fuv+w4']:
                        band1 = 'fuv'
                        band2 = 'w4'
                    else:
                        band1 = 'nuv'
                        band2 = 'w3'
                    sfr_hasmap = band_check(name,res[k],band1,band2)
                # If it's an individual band:
                elif bands[j].lower() in ['fuv','nuv','w2','w3','w4']:
                    sfr_hasmap = band_check(name,res[k],bands[j].lower())
                # If no band is specified:
                #   then we only exclude the galaxy if ALL maps are missing.
                elif bands[j].lower() in ['']:
                    raise ValueError('Null (\'\') band detected; this shouldn\'t ever happen!')
                    sfr_hasmap1 = band_check(name,res[k],'fuv'.lower())
                    sfr_hasmap2 = band_check(name,res[k],'nuv'.lower())
                    sfr_hasmap3 = band_check(name,res[k],'w3'.lower())
                    sfr_hasmap4 = band_check(name,res[k],'w4'.lower())
                    if sfr_hasmap1 or sfr_hasmap2 or sfr_hasmap3 or sfr_hasmap4:
                        sfr_hasmap = True  # At least some of the maps are here
                    else:
                        sfr_hasmap = False # All of the maps are missing
#                         print(name+' : '+str(sfr_hasmap))
                # If the band is invalid:
                else:
                    raise ValueError('galaxies_gen() : include/exclude_missing_sfr=[\''+bands[j]+'\'] is not a valid band!')

                if sfr_hasmap==False:
                    galaxies_detectedsfr[k][i]  # Do nothing
                else:
                    galaxies_detectedsfr[k][i] = galaxies_detectedsfr[k][i]+1
                    # Populate galaxies_bestres
                    if galaxies_bestres[i]>res_quantity[k]:     # If detected resolution is better than current best:
                        galaxies_bestres[i] = res_quantity[k]   #   Congrats, we've got a new best resolution.
    if data_mode_sfr in ['either','both']:
        galaxies_nosfr_7p5   = list(np.array(galaxies_set)[galaxies_detectedsfr[0]==0])
        galaxies_somesfr_7p5 = list(np.array(galaxies_set)[(galaxies_detectedsfr[0]>0)&(galaxies_detectedsfr[0]<(len(bands)))])
        galaxies_allsfr_7p5  = list(np.array(galaxies_set)[galaxies_detectedsfr[0]==(len(bands))])
        galaxies_nosfr_15   = list(np.array(galaxies_set)[galaxies_detectedsfr[1]==0])
        galaxies_somesfr_15 = list(np.array(galaxies_set)[(galaxies_detectedsfr[1]>0)&(galaxies_detectedsfr[1]<(len(bands)))])
        galaxies_allsfr_15  = list(np.array(galaxies_set)[galaxies_detectedsfr[1]==(len(bands))])
        galaxies_nosfr = []
        galaxies_somesfr = []
        galaxies_allsfr = []
        if data_mode_sfr in ['either']:
            for i in range(0,len(galaxies_set)):
                if (galaxies_set[i] in galaxies_allsfr_7p5) or (galaxies_set[i] in galaxies_allsfr_15):
                    galaxies_allsfr = np.append(galaxies_allsfr,galaxies_set[i])    # At least one res has all bands
                elif (galaxies_set[i] in galaxies_somesfr_7p5) or (galaxies_set[i] in galaxies_somesfr_15):
                    galaxies_somesfr = np.append(galaxies_somesfr,galaxies_set[i])  # At least 1 res has at least 1 band
                elif (galaxies_set[i] in galaxies_nosfr_7p5) and (galaxies_set[i] in galaxies_nosfr_15):
                    galaxies_nosfr = np.append(galaxies_nosfr,galaxies_set[i])      # Both res's have 0 bands
                else:
                    raise ValueError('gotta be one of them three')
        elif data_mode_sfr in ['both']:
            for i in range(0,len(galaxies_set)):
                if (galaxies_set[i] in galaxies_allsfr_7p5) and (galaxies_set[i] in galaxies_allsfr_15):
                    galaxies_allsfr = np.append(galaxies_allsfr,galaxies_set[i])    # Both res's has all bands
                elif (galaxies_set[i] in galaxies_nosfr_7p5) and (galaxies_set[i] in galaxies_nosfr_15):
                    galaxies_nosfr = np.append(galaxies_nosfr,galaxies_set[i])      # Both res's have 0 bands
                else:
                    galaxies_somesfr = np.append(galaxies_somesfr,galaxies_set[i])  # At least 1 res has at least 1 band
    else:
        galaxies_nosfr  = list(np.array(galaxies_set)[galaxies_detectedsfr[0]==0])
        galaxies_somesfr  = list(np.array(galaxies_set)[(galaxies_detectedsfr[0]>0)&(galaxies_detectedsfr[0]<(len(bands)))])
        galaxies_allsfr = list(np.array(galaxies_set)[galaxies_detectedsfr[0]==(len(bands))])

    # Populate galaxies_bestcuberes
    if cube_min_res!=999999*u.arcsec:
        for i in range(0,len(galaxies_set_b)):  # For the remaining galaxies after the SFR-res purge:
            name = galaxies_set_b[i]
#             with silence():      # Best resolution, from header:
#                 hdr = hdr_get(name,data_mode,dim=3)
#             galaxies_bestcuberes[i] = ((hdr['BMAJ']*u.deg).to(u.rad)).value * gal_distances[i].to(u.pc)
            with silence():      # Best pre-convolved resolution:
                for j in [500*u.pc,750*u.pc,1000*u.pc,1250*u.pc]:
                    mom0 = mom0_get(name,data_mode,conbeam=j)
                    if mom0 is not None:
                        galaxies_bestcuberes[i] = j
                        break
    
    # Excluding galaxies whose CUBE+SFR resolutions are too low!
    galaxies_bestres     = galaxies_bestres.value
    galaxies_bestcuberes = galaxies_bestcuberes.value
    galaxies_sfr_min_res = galaxies_sfr_min_res.value    # Remove units for easy conversions between list<->array
    galaxies_cube_min_res = galaxies_cube_min_res.value  # Remove units for easy conversions between list<->array

    # Selecting final(?) galaxies!
    if mode=='exclusion':
        # Exclude low inclinations?
        if exclude_low_incl==True:
            galaxies_set = [e for e in galaxies_set if e not in galaxies_lowincl]   # Keeps elements NOT in low_incl
        # Exclude galaxies with missing sfr?
        if exclude_missing_sfr!=['']:  # If SFR exclusions are specified:
            galaxies_set = [e for e in galaxies_set if (e not in galaxies_allsfr)]  # Keeps elements NOT in allsfr
        else:
            galaxies_set            # Do nothing
        # Exclude others
        galaxies_set = [e for e in galaxies_set if e not in exclude_custom]
    if mode=='inclusion':
        galaxies_set = []
        # Include low inclinations?
        if include_low_incl==True:
            galaxies_set = np.append(galaxies_set,galaxies_lowincl)
        # Include galaxies with missing sfr?
        if include_missing_sfr!=['']:
            galaxies_set = np.append(galaxies_set,galaxies_nosfr)
        # Include galaxies with specified sfr?
        if include_sfr!=['']:
            galaxies_set = np.append(galaxies_set,galaxies_allsfr)
        # Exclude others
        galaxies_set = [e for e in galaxies_set if e not in exclude_custom]
        # Include others
        if include_custom!=['']:
            galaxies_set = np.append(galaxies_set,include_custom)
    if mode=='':
        # Exclude others
        galaxies_set = [e for e in galaxies_set if e not in exclude_custom]
    galaxies_set = list(np.sort(list(set(galaxies_set))))

    # Exclude galaxies with crappy resolutions
    galaxies_set = [e for e in galaxies_set if (galaxies_bestres[galaxies_set_b.index(e)]<=galaxies_sfr_min_res[galaxies_set_b.index(e)] \
                                              and galaxies_bestcuberes[galaxies_set_b.index(e)]<=galaxies_cube_min_res[galaxies_set_b.index(e)])]

#     return galaxies_nosfr, galaxies_somesfr, galaxies_allsfr
    return galaxies_set

def hasbar_get(galaxies_set,path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1'):
    '''
    Returns an array of len(galaxies_set)
    with each entry being True or False.
    Can also work with a single galaxy.
    
    Parameters:
    -----------
    galaxies_set : str or list
        Galaxy/galaxies.
    data_mode='7m' : str
        '7m' or '12m'.
        (Defunct?)
    '''
    galaxies_set_backup = None
    if isinstance(galaxies_set,str):
        galaxies_set_backup = galaxies_set
        galaxies_set = [galaxies_set]
    galmax = len(galaxies_set)
    hasbar = [False]*galmax
    
#    with silence():
#        for i in range(0,galmax):
#            name = galaxies_set[i]
#            barcheck = bar_info_get(name,data_mode,radii='arcsec',check_has_bar=True,\
#                     folder='/media/jnofech/BigData/galaxies/drive_tables/',fname='TABLE_Environmental_masks - Parameters')
#            if barcheck=='No':
#                hasbar[i] = False
#            elif barcheck=='Yes':
#                hasbar[i] = True
#            elif barcheck=='Uncertain':
#                hasbar[i] = None
#            else:
#                raise ValueError('houston we have a problem')

#        print('\nMISSING BARS:')
#        for i in range(0,len(np.where(np.array(hasbar)==None)[0])):
#            print(galaxies_set[np.where(np.array(hasbar)==None)[0][i]])

#        # Custom 'hasbar' entries!
#        if 'NGC1317' in galaxies_set:
#            hasbar[galaxies_set.index('NGC1317')] = True
#        if 'NGC2090' in galaxies_set:
#            hasbar[galaxies_set.index('NGC2090')] = False
#        if 'NGC2283' in galaxies_set:
#            hasbar[galaxies_set.index('NGC2283')] = True
#        if 'NGC2566' in galaxies_set:
#            hasbar[galaxies_set.index('NGC2566')] = True
#        if 'NGC2835' in galaxies_set:
#            hasbar[galaxies_set.index('NGC2835')] = True
#        if 'NGC2997' in galaxies_set:
#            hasbar[galaxies_set.index('NGC2997')] = False
#        if 'NGC3059' in galaxies_set:
#            hasbar[galaxies_set.index('NGC3059')] = True
#        if 'NGC3137' in galaxies_set:
#            hasbar[galaxies_set.index('NGC3137')] = False
#        if 'NGC3621' in galaxies_set:
#            hasbar[galaxies_set.index('NGC3621')] = False
#        if 'NGC4457' in galaxies_set:
#            hasbar[galaxies_set.index('NGC4457')] = False
#        if 'NGC5128' in galaxies_set:
#            hasbar[galaxies_set.index('NGC5128')] = False
#        if 'NGC5530' in galaxies_set:
#            hasbar[galaxies_set.index('NGC5530')] = False
#        if 'NGC5643' in galaxies_set:
#            hasbar[galaxies_set.index('NGC5643')] = True
#        if 'NGC6300' in galaxies_set:
#            hasbar[galaxies_set.index('NGC6300')] = True
#        if 'NGC6744' in galaxies_set:
#            hasbar[galaxies_set.index('NGC6744')] = True


    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data
    
    with silence():
        for i in range(0,galmax):
            name = galaxies_set[i]
            if name.upper() in table.field('NAME'):
                hasbar[i] = table.field('BAR')[list(table.field('NAME')).index(name.upper())].astype(bool)
            else:
                raise ValueError('tools.hasbar_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
                hasbar[i] = None

        print('\nMISSING BARS:')
        for i in range(0,len(np.where(np.array(hasbar)==None)[0])):
            print(galaxies_set[np.where(np.array(hasbar)==None)[0][i]])

        hasbar = np.array(hasbar)

    if isinstance(galaxies_set_backup,str):
        return hasbar[0]
    else:
        return hasbar
        
def alphaCO21_gen(galaxies_set,rad,alpha_custom='radial',\
                  path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1',
                  path_jiayi='/media/jnofech/BigData/galaxies/drive_tables/',fname_jiayi='catalog.ecsv'):
    '''
    Returns an array of len(galaxies_set)
    with each entry being True or False.
    Can also work with a single galaxy.
    
    Parameters:
    -----------
    galaxies_set : str or list
        Galaxy/galaxies.
    rad : u.Quantity (array-like)
        Radii map(s), corresponding to galaxies_set.
        Will use u.pc if a quantity is not specified.
    alpha_custom='radial' : str
        'radial' : Returns 2D alpha_CO(2-1) map
        'custom' : Returns 2D alpha_CO(2-1) map, except
                   it's just the radial version at radius Re
                   for each galaxy
        'constant' : Returns Milky Way(?) value (6.2)
    '''
    galaxies_set_backup = None
    rad_backup      = None
    rad_set         = copy.deepcopy(rad)
    if isinstance(galaxies_set,str):
        galaxies_set_backup = galaxies_set
        rad_backup      = copy.deepcopy(rad)
        galaxies_set = [galaxies_set]
        rad_set      = [copy.deepcopy(rad)]
    if not hasattr(rad_set[0],'unit'):
        for i in range(0,len(rad_set)):
            rad_set[i] = rad_set[i]*u.pc
    galmax = len(galaxies_set)
    alpha_CO21 = [None]*galmax

    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data
    table_jiayi = Table.read(path_jiayi+fname_jiayi, format='ascii.ecsv')
    
    Re_mode = 'z0mgs'
    # NOTE: Do not confused "Re" (i.e. effective radius; commonly denoted Re) 
    #    with "R_e" (i.e. my own DiskFit errorbars).        
        
    with silence():
        for i in range(0,galmax):
            name = galaxies_set[i]
            
            # Get stellar mass
            if name.upper() in table.field('NAME'):
                logmass = table.field('LOGMSTAR')[list(table.field('NAME')).index(name.upper())]
            else:
                raise ValueError('tools.alphaCO21_gen() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
                logmass = None
                
            # Get Re
            if Re_mode.lower()=='z0mgs':
                # Takes 'R_e' from the z0mgs WISE1 maps. Covers most targets, but preliminary WIP approach.
                Re = table_jiayi.field('REFF_W1')[list(table_jiayi.field('NAME')).index(name.upper())]*u.kpc
            elif Re_mode.lower()=='2mass':
                # Takes 'R_e' recorded from the 2MASS Large Galaxy Atlas catalog (Jarrett+03). Don't cover full PHANGS sample.
                Re = table_jiayi.field('REFF_K')[list(table_jiayi.field('NAME')).index(name.upper())]*u.kpc
            else:
                raise ValueError('Invalid Re_mode.')
            if np.isnan(Re):
                print('tools.alphaCO21_gen() : WARNING: '+name+' does not have "Re" value from '+Re_mode+' sample!')
                
            # Get metallicity array
            if alpha_custom.lower() in ['rad','radial','2d','map']:
                logOH = XCO.predict_metallicity(10**logmass,Rgal=rad_set[i].to(u.kpc),Re=Re,gradient='Sanchez+14')
            elif alpha_custom.lower() in ['custom','re']:
                logOH = rad_set[i].value*0 + XCO.predict_metallicity(10**logmass,Rgal=Re,Re=Re,gradient='Sanchez+14')
            elif alpha_custom.lower() in ['constant']:
                logOH = rad_set[i].value*0 + 8.7*u.K/u.K  # Unitless; the K is arbitrary
            else:
                raise ValueError('tools.alphaCO21_gen() : Invalid alpha_custom.')
            Z = 10.**(logOH - 8.7)
            alpha_CO10    = XCO.predict_alphaCO10(prescription='PHANGS',PHANGS_Zprime=Z)  # CO(1-0)-to-H2
            alpha_CO21[i] = (alpha_CO10/0.7).value   # CO(2-1)intensity -> H2signal conversion factor

        print('\nMISSING ALPHA CO(2-1):')
        for i in range(0,len(np.where(np.array(alpha_CO21)==None)[0])):
            print(galaxies_set[np.where(np.array(alpha_CO21)==None)[0][i]])
        alpha_CO21 = np.array(alpha_CO21)

    if isinstance(galaxies_set_backup,str):
        return alpha_CO21[0]
    else:
        return alpha_CO21
        
def Re_get(galaxies_set,\
           path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1',
           path_jiayi='/media/jnofech/BigData/galaxies/drive_tables/',fname_jiayi='catalog.ecsv'):
    '''
    Returns the characteristic radius
    'Re' of given galaxy/galaxies, in kpc.
    
    Parameters:
    -----------
    galaxies_set : str or list
        Galaxy/galaxies.
    '''
    galaxies_set_backup = None
    if isinstance(galaxies_set,str):
        galaxies_set_backup = galaxies_set
        galaxies_set = [galaxies_set]
    galmax = len(galaxies_set)
    Re = [False]*galmax

    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data
    table_jiayi = Table.read(path_jiayi+fname_jiayi, format='ascii.ecsv')
    
    Re_mode = 'z0mgs'
    # NOTE: Do not confused "Re" (i.e. effective radius; commonly denoted Re) 
    #    with "R_e" (i.e. my own DiskFit errorbars).        
        
    with silence():
        for i in range(0,galmax):
            name = galaxies_set[i]
           
            # Get Re, in kpc
            if Re_mode.lower()=='z0mgs':
                # Takes 'R_e' from the z0mgs WISE1 maps. Covers most targets, but preliminary WIP approach.
                Re[i] = table_jiayi.field('REFF_W1')[list(table_jiayi.field('NAME')).index(name.upper())]
            elif Re_mode.lower()=='2mass':
                # Takes 'R_e' recorded from the 2MASS Large Galaxy Atlas catalog (Jarrett+03). Don't cover full PHANGS sample.
                Re[i] = table_jiayi.field('REFF_K')[list(table_jiayi.field('NAME')).index(name.upper())]
            else:
                raise ValueError('Invalid Re_mode.')
            if np.isnan(Re[i]):
                print ('tools.Re_get() : WARNING: '+name+' does not have "Re" value from '+Re_mode+' sample!')

        Re = np.array(Re)*u.kpc
        
    if isinstance(galaxies_set_backup,str):
        return Re[0]
    else:
        return Re

def hubbleT_get(galaxies_set,path='/media/jnofech/BigData/galaxies/',fname='phangs_sample_table_v1p1'):
    '''
    Returns an array of len(galaxies_set)
    with each entry being the Hubble stage T.
    Higher values indicate later-type galaxies.
    Can also work with a single galaxy.
    
    Parameters:
    -----------
    galaxies_set : str or list
        Galaxy/galaxies.
    '''
    galaxies_set_backup = None
    if isinstance(galaxies_set,str):
        galaxies_set_backup = galaxies_set
        galaxies_set = [galaxies_set]
    galmax = len(galaxies_set)
    hubbleT = [False]*galmax

    table = copy.deepcopy(fits.open(path+fname+'.fits'))[1].data
    
    with silence():
        for i in range(0,galmax):
            name = galaxies_set[i]
            if name.upper() in table.field('NAME'):
                hubbleT[i] = table.field('T')[list(table.field('NAME')).index(name.upper())]
            else:
                raise ValueError('tools.hubbleT_get() : Galaxy \''+name+'\' not in '+path+fname+'.fits!')
                hubbleT[i] = None

        print('\nMISSING hubbleT:')
        for i in range(0,len(np.where(np.array(hubbleT)==None)[0])):
            print(galaxies_set[np.where(np.array(hubbleT)==None)[0][i]])

        hubbleT = np.array(hubbleT)

    if isinstance(galaxies_set_backup,str):
        return hubbleT[0]
    else:
        return hubbleT
