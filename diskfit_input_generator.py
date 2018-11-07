# Important paths!
import sys
sys.path.insert(0, '/media/jnofech/BigData/galaxies/')
sys.path.insert(0, '/media/jnofech/BigData/galaxies/VLA_Lband/')
sys.path.insert(0, '/media/jnofech/BigData/jnofech_codes/')
sys.path.insert(0, '/media/jnofech/BigData/')

import numpy as np

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from radio_beam import Beam
from galaxies.galaxies import Galaxy
from astropy.coordinates import SkyCoord, Angle, FK5

import pandas as pd

import copy
import os.path
import subprocess

# Import my own code
import rotcurve_tools as rc
import galaxytools as tools

def filename_get(name,data_mode='7m',mapmode='mom1',\
                path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
                path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
                path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
#                 path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/for_inspection/',\
                path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
                folder_vpeak='jnofech_peakvels/',\
                folder_hybrid='jnofech_mom1_hybrid/'):
    name = name.lower()
    gal = tools.galaxy(name.upper())
    if name.upper()=='NGC3239':
        raise ValueError('Bad galaxy.')
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m' 
    if mapmode in ['mom1']:
        mapmode='mom1'
    elif mapmode in ['peakvels','vpeak']:
        mapmode='peakvels'
    else:
        print('No mapmode set. Defaulted to "mom1".')
        data_mode = 'mom1' 
        
    # Find filenames for map and error map!
    if data_mode in ['7m','hybrid']:
        if mapmode=='mom1':
            path = path7m
        elif mapmode=='peakvels':
            path = path7m_mask+folder_vpeak
        data_mode_temp = '7m'
        filename_7mtp = name+'_'+data_mode_temp+'+tp_co21_'+mapmode+'.fits'    # 7m+tp mom1. Ideal.
        filename_7m   = name+'_'+data_mode_temp+   '_co21_'+mapmode+'.fits'    # 7m mom1. Less reliable.
#         print(filename_7m)
        if os.path.isfile(path+filename_7mtp):
            filename_map  = filename_7mtp
            filename_emap = name+'_'+data_mode_temp+'+tp_co21_e'+mapmode+'.fits'
            best_map_7m='7m+tp'
        elif os.path.isfile(path+filename_7m):
            filename_map  = filename_7m
            filename_emap = name+'_'+data_mode_temp+   '_co21_e'+mapmode+'.fits'
            best_map_7m='7m'
        else:
            filename_map  = "7m "+mapmode+" MISSING!"
            filename_emap = "7m "+mapmode+" MISSING!"
            raise ValueError('Neither 7m nor 7m+tp '+mapmode+' map found!')
        best_map = best_map_7m
    if data_mode in ['12m+7m','hybrid']:
        if mapmode=='mom1':
            path = path12m
        elif mapmode=='peakvels':
            path = path12m_mask+folder_vpeak
        data_mode_temp = '12m+7m'
        filename_12mtp = name+'_co21_'+data_mode_temp+'+tp_'+mapmode+'.fits'  # (?) Will all new maps have '+tp'?
        best_map_12m = '12m+7m+tp'                                    # (?) ^
        best_map = best_map_12m
        if os.path.isfile(path+filename_12mtp):
            filename_map  = filename_12mtp
            filename_emap = name+'_co21_'+data_mode_temp+'+tp_e'+mapmode+'.fits'
        else:
            filename_map  = "12m "+mapmode+" MISSING!"
            filename_emap = "12m "+mapmode+" MISSING!"
            raise ValueError('No 12m+tp '+mapmode+' found!')
    if data_mode=='hybrid':
        best_map = 'hybrid_'+best_map_7m+'&'+best_map_12m
        filename = name+'_co21_'+best_map+'_'+mapmode+'.fits'
        if mapmode=='mom1':
            path = path12m+folder_hybrid
        elif mapmode=='peakvels':
            path = path12m_mask+folder_vpeak
        if os.path.isfile(path+filename):
            filename_map  = filename
            filename_emap = name+'_co21_'+best_map+'+_e'+mapmode+'.fits'
        else:
            filename_map  = "Hybrid "+mapmode+" MISSING!"
            filename_emap = "Hybrid "+mapmode+" MISSING!"
            raise ValueError('No hybrid '+mapmode+' found!')
        
    return filename_map, filename_emap

def gen_input(name,data_mode='7m',mapmode='mom1',errors=False,errors_exist=False,iteration=1,  \
              xcen  =np.nan,ycen  =np.nan,PA  =np.nan,eps  =np.nan,vsys  =np.nan,\
              xcen_p=np.nan,ycen_p=np.nan,PA_p=np.nan,eps_p=np.nan,vsys_p=np.nan,\
              alteration=[None]*8,\
              toggle_xcen_over=None,toggle_PA_over=None,toggle_eps_over=None,toggle_vsys_over=None,\
              debug=False,\
              path7m ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/',\
              path12m='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
              path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/osu/eros_masks/',\
#              path7m_mask ='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-LP/working_data/for_inspection/',\
              path12m_mask='/media/jnofech/BigData/PHANGS/Archive/PHANGS-ALMA-v1p0/',\
              folder_vpeak='jnofech_peakvels/',\
              folder_hybrid='jnofech_mom1_hybrid/',\
              diskfit_folder='diskfit_procedural/'):
    '''
    Generates a DiskFit .inp file.
    
    
    Parameters:
    -----------
    name : str
        Galaxy's name.
    data_mode : str
        '7m' (default) - uses 7m data.
        '12m'          - 12m data.
        'hybrid'       - combines 7m and 12m.
    mapmode(='mom1') : str
        'mom1' - uses mom1 map of specified
                 data_mode.
        'peakvels' - uses peakvels map of 
                     specified data_mode.
    errors=False : bool
        Calculates error bars on rotcurves.
        Takes a very long time!
    errors_exist=False : bool
        Toggles whether error maps are
        available in the first place.
    iteration=1 : int
        Iteration number.
    <params>=np.nan : float
        Output parameters from the previous run.
    <params>_p=np.nan : float
        Input parameters from the previous run.
        (i.e. Output parameters from the previous
        previous run.)
    alteration=[False]*8 : bool list
        Indicates whether certain parameters
        have been successfully altered before.
        If one has been altered, its toggle is
        Disabled so the value is kept permanently.
        
    Parameters (debug):
    -------------------
    toggle_xxxx_over=None : str ('T'/'F')
        OVERRIDES for the xcen/PA/eps/vsys toggles.
    debug=False : bool
        If enabled, will return certain outputs
        to keep track of what the input parameters are
        set to.
    '''
    name = name.lower()
    gal = tools.galaxy(name.upper())
    if name.upper()=='NGC3239':
        raise ValueError('Bad galaxy.')
    if data_mode == '7m':
        data_mode = '7m'
    elif data_mode in ['12m','12m+7m']:
        data_mode = '12m+7m'  
    elif data_mode.lower() in ['both','hybrid']:
        data_mode = 'hybrid'  
    elif data_mode=='':
        print('No data_mode set. Defaulted to 12m+7m.')
        data_mode = '12m+7m' 
    if mapmode in ['mom1']:
        mapmode='mom1'
    elif mapmode in ['peakvels','vpeak']:
        mapmode='peakvels'
    else:
        print('Invalid mapmode set. Defaulted to "mom1".')
        data_mode = 'mom1' 
        
        
    # Report which iteration we're on!
    print("\nCreating iteration #"+str(iteration)+"'s input:")
    
    
    # Find filenames for map and error map!
    filename_map, filename_emap = filename_get(name,data_mode,mapmode,path7m,path12m,path7m_mask,path12m_mask,\
                                               folder_vpeak,folder_hybrid)
            

    # Get the mom1 map and stuff!
    # (!) Will need to update for peakvels and noise and such!
    if data_mode=='hybrid':
        data_mode_temp = '12m+7m'
    else:
        data_mode_temp = data_mode
    hdr, beam, x, I_mom1, x, x, x, x, x = tools.info(gal,None,data_mode_temp)
    I_mom1 = tools.mom1_get(gal,data_mode)
    
    # Initialize!
    alteration_label = ['PA_altered','eps_altered','coords_altered','vsys_altered','bar_PA_altered',\
                        'r_w_altered','warp_eps_altered','warp_PA_altered']
    
    # Generate xcen, ycen, as numbers.
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))[0]*u.deg # Pixel width, in deg.
    pixsizes_arcsec = pixsizes_deg.to(u.arcsec)                             # Pixel width, in arcsec.
    skycoord = gal.skycoord_grid(header=hdr)
    RA = skycoord.ra     # 2D map of RA.
    Dec = skycoord.dec   # 2D map of Dec.
    RA_cen = gal.center_position.ra
    Dec_cen = gal.center_position.dec
    xcen_1 = I_mom1.shape[1] * (RA.max() - RA_cen).value / (RA.max() - RA.min()).value
    ycen_1 = I_mom1.shape[0] * (Dec_cen - Dec.min()).value / (Dec.max() - Dec.min()).value
    xcen_1 = "{0:4.2f}".format(xcen_1)   # It's a string now.
    ycen_1 = "{0:4.2f}".format(ycen_1)   # It's a string now.
    
    # Generate all other required values!
    width = str(I_mom1.shape[1])
    height = str(I_mom1.shape[0])
    regrad = "{0:4.2f}".format(beam*u.deg/pixsizes_deg*2.5)
    pixscale = "{0:4.2f}".format(pixsizes_arcsec.value)                # Num. of arcsecs per pixel. Unused.
    m = str(2)                                                         # Harmonic order of noncirc flow.
    delta_ISM = str((6.0*u.km/u.s).value)                              # "Doesn't change much to my knowledge."
    
    # Get PA, Eps, Vsys
    PA_temp = gal.position_angle
    PA_1 = "{0:4.2f}".format(PA_temp.value)
    eps_1 = "{0:4.2f}".format(1.0 - np.cos(gal.inclination))
    vsys_1 = "{0:4.2f}".format(gal.vsys.value)
    
    
    # Default toggles!
    if errors==True:
        toggle_emom1 = 'T'
    else:
        toggle_emom1 = 'F'
    toggle_PA   = 'F'
    toggle_eps  = 'F'
    toggle_xcen = 'T'      # Also covers ycen.
    toggle_vsys = 'F'

    # Replace initial parameters with input (NOT output) from the previous run!
    # These inputs were used as actual inputs (i.e. they've already been checked to be
    #    "good"), so don't worry, there are no bogus values here.
    if ~np.isnan(xcen_p*ycen_p):
        xcen_1 = "{0:4.2f}".format(xcen_p)
        ycen_1 = "{0:4.2f}".format(ycen_p)
    if ~np.isnan(PA_p):
        PA_1 = "{0:4.2f}".format(PA_p.value)
    if ~np.isnan(eps_p):
        eps_1 = "{0:4.2f}".format(eps_p)
    if ~np.isnan(vsys_p):
        vsys_1 = "{0:4.2f}".format(vsys_p.value)
    
    # Toggle parameter-fitting!
    # Enable if a good fit has not been found.
    # Disable permanently if a good fit was found.
    #
    # Also: Go F/F/T/F for iteration 1. Go T/T/?/T for the next iteration. 
    #       Let the code handle itself from there.
    
    if ~np.isnan(xcen):
        xcen = "{0:4.2f}".format(xcen)
        ycen = "{0:4.2f}".format(ycen)
        # Check for bogus results.
        if alteration[alteration_label.index('coords_altered')]==False:
            print(name+' : Enabled PA toggle!')
            print(name+' : Enabled Eps toggle!')
            print(name+' : Enabled Vsys toggle!')
            toggle_xcen = 'T'
            toggle_PA   = 'T'
            toggle_eps  = 'T'
            toggle_vsys = 'T'
        else:
            if float(xcen_1)*float(ycen_1) != float(xcen)*float(ycen):
            # Prevent redundant printing if the current and previous values are the same.
                print(name+'\'s xcen and ycen : ('+xcen_1+','+ycen_1+') to ('+xcen+','+ycen+').')
                print(name+' : Disabled central toggle!')
                print(name+' : Enabled PA toggle!')
                print(name+' : Enabled Eps toggle!')
                print(name+' : Enabled Vsys toggle!')
            toggle_xcen = 'F'
            toggle_PA   = 'T'
            toggle_eps  = 'T'
            toggle_vsys = 'T'
            xcen_1 = xcen
            ycen_1 = ycen
    if ~np.isnan(PA):
        PA = "{0:4.2f}".format(PA.value)
        # Check for bogus results.
        if alteration[alteration_label.index('PA_altered')]==False:
            toggle_PA = 'T'
        else:
            if float(PA_1)!=float(PA):
            # If the current and previous values are the same, prevent redundant printing.
                print(name+'\'s PA : '+PA_1+' -> '+PA+'.')
                print(name+' : Disabled PA toggle!')
            toggle_PA = 'F'
            PA_1 = PA
    if ~np.isnan(eps):
        eps = "{0:4.2f}".format(eps)
        if alteration[alteration_label.index('eps_altered')]==False:
            toggle_eps = 'T'
        else:
            if float(eps_1)!=float(eps):
            # If the current and previous values are the same, prevent redundant printing.
                print(name+'\'s eps : '+eps_1+' -> '+eps+'.')
                print(name+' : Disabled eps toggle!')
            toggle_eps = 'F'
            eps_1 = eps
    if ~np.isnan(vsys):
        vsys = "{0:4.2f}".format(vsys.value)
        if alteration[alteration_label.index('vsys_altered')]==False:
            toggle_vsys = 'T'
        else:
            if float(vsys_1)!=float(vsys):
            # If the current and previous values are the same, prevent redundant printing.
                print(name+'\'s Vsys : '+vsys_1+' -> '+vsys+'.')
                print(name+' : Disabled vsys toggle!')
            toggle_vsys = 'F'
            vsys_1 = vsys
    
    # Overrides from input!
    if toggle_xcen_over!=None:
        print('xcen overwritten: '+toggle_xcen+'->'+toggle_xcen_over)
        toggle_xcen = toggle_xcen_over
    if toggle_PA_over!=None:
        print('PA overwritten: '+toggle_PA+'->'+toggle_PA_over)
        toggle_PA = toggle_PA_over
    if toggle_eps_over!=None:
        print('Eps overwritten: '+toggle_eps+'->'+toggle_eps_over)
        toggle_eps = toggle_eps_over
    if toggle_vsys_over!=None:
        print('Vsys overwritten: '+toggle_vsys+'->'+toggle_vsys_over)
        toggle_vsys = toggle_vsys_over
        
    # CALCULATE RADII. (Now that all the params (PA, encl, etc) have been sorted out.)
#     # METHOD 1
#     # Original.
#     rad = gal.radius(skycoord=skycoord).to(u.pc).value
#     radius_max = np.percentile(rad[~np.isnan(I_mom1)],99.0)
#                  # ^ Max radius used, in pc.
#     radius_max_pix = ((radius_max*u.pc / gal.distance.to(u.pc)).value * u.rad.to(u.arcsec)) / pixsizes_arcsec.value
#     radius_max_pix = np.nanmin([radius_max_pix,np.sqrt(float(xcen_1)**2+float(ycen_1)**2)]) # Upper limit.
#     radius_pix = np.linspace(1,radius_max_pix,30)                              # Radius array, in pixels.
#     # METHOD 2
#     # Calculate radius map of galaxy, in galactic plane.
#     Xgal,Ygal = gal.radius(skycoord=skycoord,header=hdr,returnXY=True)   # Maps of x,y in galactic plane, in pc.
#     Xgal = ((Xgal.to(u.pc) / gal.distance.to(u.pc)).value * u.rad.to(u.arcsec)) / pixsizes_arcsec.value # In pixels.
#     Ygal = ((Ygal.to(u.pc) / gal.distance.to(u.pc)).value * u.rad.to(u.arcsec)) / pixsizes_arcsec.value # In pixels.
#     Rgal = np.sqrt(Xgal**2+Ygal**2)
#     radius_max_pix = np.percentile(Rgal[~np.isnan(I_mom1)],99.0)
#     radius_pix = np.linspace(1,radius_max_pix,30)                              # Radius array, in pixels.
    # METHOD 3
    # Getting radius map just from the image alone. Doesn't adapt with fitted params.
    xrange = np.arange(I_mom1.shape[1])-float(xcen_1)
    yrange = np.arange(I_mom1.shape[0])-float(ycen_1)
    X = np.tile(xrange, (I_mom1.shape[0],1))     # Map of X values, in pixels.
    Y = np.tile(yrange, (I_mom1.shape[1],1)).T   # Map of Y values, in pixels.
    R = np.sqrt(X**2+Y**2)                  # Map of radii, in pixels.
    radius_max_pix = np.percentile(R[~np.isnan(I_mom1)],99.5)
    radius_min_pix = np.percentile(R[~np.isnan(I_mom1)],0.40)
    radius_pix = np.linspace(radius_min_pix,radius_max_pix,25)                  # Radius array, in pixels.
#     print(radius_pix)

    # Generate the actual input file ('input_str')!
    text = []
    text.append("PHANGS AUTO-GENERATED ROTATION CURVE")
    text.append("vels                                ")
    text.append("T  F                                ")
    text.append("'"+filename_map+"                  ")
    if errors_exist==True:
        text.append("'"+filename_emap+"                 ")
    else:
        text.append("None                                ")    # Also emom1.
    text.append("1  1  "+width+" "+height+"          ")
    text.append(""+regrad+" "+PA_1+" "+eps_1+"  2 "+pixscale+"  ")
    text.append("'Output/"+name.lower()+"_v"+str(iteration)+".out")
    text.append(""+toggle_PA+" "+toggle_eps+" "+toggle_xcen+"                   #(!!!) FFT, then TTF!")
    text.append(""+PA_1+"  "+eps_1+"            # PA, eps")
    text.append(""+xcen_1+"  "+ycen_1+"            # xcen, ycen             ")
#     text.append("F T "+PA_1+" "+m+"             #(?) Noncirc/non-axisymm/BAR flow toggle. Avoid radial.")
    text.append("F T "+str(45.0)+" "+m+"             #(?) Noncirc/non-axisymm/BAR flow toggle. Avoid radial.")
    text.append("T F                     #(?) Inner interp toggle + radial flow toggle. Avoid bar.")
    text.append(""+toggle_vsys+" "+vsys_1+" "+delta_ISM+" 25.0       # Vsys")
    text.append("F T T T 90 0 0          #(?) Warp? (CANNOT use with Bar or Radial flow fits!)")
    text.append("0.                                         ")
    text.append("-0.01 -0.01                                ")
    text.append(""+toggle_emom1+" -50 5 -1.0                               ")
    text.append("F                       # Verbose          ")
    text.append("3.00 "+"{0:4.2f}".format(radius_pix.max()/4.)+"               #(?) Min/max bar radii. May need improvement")
    for j in range(radius_pix.size):
        text.append("{0:4.2f}".format(radius_pix[j]))
    input_str = ''
    for i in range(0,len(text)):
        input_str = input_str+text[i]+'\n'

    # Print 'input_str'!
#     print(input_str)

    # Save 'input_str'!
    file = open('/media/jnofech/BigData/galaxies/'+diskfit_folder\
                +gal.name.lower()+'_v'+str(iteration)+'.inp','w')
    file.write(input_str)
    file.close()
    
    if debug==True:
        return toggle_xcen, toggle_PA, toggle_eps, toggle_vsys
        
def read_output(name,iteration=1,alteration=[False]*8,verbose=True,\
              custom_input=False,diskfit_folder='diskfit_procedural/'):
    '''
    Reads the output file for 'iteration',
    and returns the output values as astropy
    Quantities.
    
    -----------
    Parameters:
    -----------
    
    name : str
        Galaxy's name.
    iteration=1 : int
        Iteration number.
    alteration=[False]*8 : bool list
        Indicates whether certain parameters
        have been altered before. If one
        hasn't been altered by the end of this
        function, then it is returned as
        np.nan.
    verbose=True : bool
        Prints what's happening as the output
        files are read.
    custom_input=False : bool
        Toggles whether a custom run is
        read instead of an automated run.
    '''
    # Read the DiskFit output files!
    if verbose==True:
        if custom_input==False:
            print("\nReading iteration #"+str(iteration)+"'s output file:")
        else:
            print("\nReading CUSTOM output file, '_c"+str(iteration)+".inp':")
    extension = custom_input*("_c") + (not custom_input)*("_v")
    output_filename = "/media/jnofech/BigData/galaxies/"+diskfit_folder+"Output/"+name.lower()\
                          +extension+str(iteration)+".out"
        
    if os.path.isfile(output_filename):
        file = open(output_filename,'r')
    else:
        print('WARNING: '+name+'\'s output file missing!')
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
               np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
               alteration,np.nan

    # Initializing. Do not touch.
    i=0                         # Line number, for the 'lines' list. Line no. for the text file is i+1.
    lines = []
    is_output = False
    # Save the file to a list of strings, so we can read more easily!
    for line in file:
        lines = np.append(lines,line)

    # Pick out all the input/output parameters!
    # Also, pick out the rotcurves with error bars.
    PA_in,PA_out,eps_in,eps_out,incl_in,incl_out,xcen_in,xcen_out,ycen_in,ycen_out,\
        bar_PA_in,bar_PA_out,vsys_in,vsys_out,delta_ISM_in,r_w_in,r_w_out,\
        warp_eps_in,warp_eps_out,warp_PA_in,warp_PA_out,chi2_out = [np.nan]*22

    alteration_label = ['PA_altered','eps_altered','coords_altered','vsys_altered','bar_PA_altered',\
                        'r_w_altered','warp_eps_altered','warp_PA_altered']
    # ^ Corresponds to 'alteration', which keeps track of which params have been changed 
    #   from their "default" (i.e. from the `galaxies` object, or from online query) values.
    
    for i in range(0,len(lines)):
        if lines[i][0:19]=='Best fitting values':
            is_output = True    # Everything below this line is output.

        if is_output==False:
            if lines[i][0:24]=='disk PA, phi_d^prime (de':
                PA_in = float(lines[i][36:43])*u.deg
            if lines[i][0:9]=='disk eps:':
                eps_in = float(lines[i][36:43])
                incl_in = (np.arccos(1.-eps_in)*u.rad).to(u.deg)
            if lines[i][0:24]=='x,y center (data units):':
                xcen_in = float(lines[i][36:43])
                ycen_in = float(lines[i][44:51])
            if lines[i][0:23]=='Non-axisymm phib (deg):':
                bar_PA_in = float(lines[i][36:43])*u.deg
            if lines[i][0:11]=='Vsys (km/s)':
                vsys_in = float(lines[i][36:43])*u.km/u.s
            if lines[i][0:17]=='Delta_ISM (km/s):':
                delta_ISM_in = float(lines[i][36:43])*u.km/u.s

            if lines[i][0:17]=='r_w (data units):':
                r_w_in = float(lines[i][36:43])
            if lines[i][0:14]=='Warp eps welm:':
                warp_eps_in = float(lines[i][36:43])
            if lines[i][0:14]=='Warp PA wphim:':
                warp_PA_in = float(lines[i][36:43])*u.deg

        elif is_output==True:
            if lines[i][0:24]=='disk PA, phi_d^prime (de':
                PA_out = float(lines[i][36:43])*u.deg
#                 alteration[alteration_label.index('PA_altered')] = True
            if lines[i][0:9]=='disk eps:':
                eps_out = float(lines[i][36:43])
#                 alteration[alteration_label.index('eps_altered')] = True
            if lines[i][0:9]=='disk incl':
                incl_out = float(lines[i][36:43])*u.deg
            if lines[i][0:24]=='x,y center (data units):':
                xcen_out = lines[i][36:43]
                ycen_out = lines[i][55:62]
                if xcen_out!='*******' and ycen_out!='*******':
                    xcen_out = float(xcen_out)
                    ycen_out = float(ycen_out)
                else:
                    if verbose==True:
                        print('Bad coords! xcen_out='+xcen_out+', ycen_out='+ycen_out)
                    xcen_out = 999999
                    ycen_out = 999999
#                 alteration[alteration_label.index('coords_altered')] = True
            if lines[i][0:23]=='Non-axisymm phib (deg):':
                bar_PA_out = float(lines[i][36:43])*u.deg
                alteration[alteration_label.index('bar_PA_altered')] = True
            if lines[i][0:11]=='Vsys (km/s)':
                vsys_out = lines[i][36:43]
                if vsys_out!='*******':
                    vsys_out = float(vsys_out)*u.km/u.s
                else:
                    if verbose==True:
                        print('Bad vsys! vsys_out='+vsys_out)
                    vsys_out=0.0000001*u.km/u.s
#                 alteration[alteration_label.index('vsys_altered')] = True

            if lines[i][0:17]=='r_w (data units):':
                r_w_out = float(lines[i][36:43])
                alteration[alteration_label.index('r_w_altered')] = True
            if lines[i][0:14]=='Warp eps welm:':
                warp_eps_out = float(lines[i][36:43])
                alteration[alteration_label.index('warp_eps_altered')] = True
            if lines[i][0:14]=='Warp PA wphim:':
                warp_PA_out = float(lines[i][36:43])*u.deg
                alteration[alteration_label.index('warp_PA_altered')] = True
                
            if lines[i][0:20]=='Minimum chi^2 found:':
                chi2_out = lines[i][31:43]
                if chi2_out!='************':
                    chi2_out = float(chi2_out)
                else:
                    chi2_out = float(999999999999)
    # Check for bogus output values! Discard them (i.e. replace with good ones) if they're bad.
    #    Only input-ready values should come out of this function!
    # If the output value is NaN, keep the input and explain why the NaN happened.
    
    # PA
    if ~np.isnan(PA_out):
        if verbose==True:
            print(name+'\'s PA : '+"{0:4.2f}".format(PA_in)+' -> '+"{0:4.2f}".format(PA_out)+'.')
        if (180.*u.deg - np.abs(np.abs(PA_in - PA_out) - 180.*u.deg))>=40*u.deg:
            if verbose==True:
                print('    Bad PA off by >40deg! Result discarded.')
            PA_out = PA_in
        else:
            if verbose==True:
                print('    Good PA = '+"{0:4.2f}".format(PA_out)+'.')
            alteration[alteration_label.index('PA_altered')] = True
    else:
        PA_out = PA_in
        if alteration[alteration_label.index('PA_altered')]==True:
            if verbose==True:
                print('PA output was not generated, as a good value has already been found!')
        else:
            if verbose==True:
                print('PA output either failed or was Disabled.')
                
    # eps
    if ~np.isnan(eps_out):
        if verbose==True:
            print(name+'\'s eps : '+"{0:4.2f}".format(eps_in)+' -> '+"{0:4.2f}".format(eps_out)+'.')
        if False:   # No condition as of yet!
            if verbose==True:
                print('    Bad eps off by >99999%! Result discarded.')
            eps_out = eps_in
        else:
            if verbose==True:
                print('    Good eps = '+"{0:4.2f}".format(eps_out)+'.')
            alteration[alteration_label.index('eps_altered')] = True
    else:
        incl_out = incl_in
        eps_out = eps_in
        if alteration[alteration_label.index('eps_altered')]==True:
            if verbose==True:
                print('eps output was not generated, as a good value has already been found!')
        else:
            if verbose==True:
                print('eps output either failed or was Disabled.')
                
    # vsys
    if ~np.isnan(vsys_out):
        if verbose==True:
            print(name+'\'s Vsys : '+"{0:4.2f}".format(vsys_in)+' -> '+"{0:4.2f}".format(vsys_out)+'.')
        if (np.abs(vsys_in - vsys_out)/vsys_in > 0.15):
            if verbose==True:
                print('    Bad vsys off by >15%! Result discarded.')
            vsys_out = vsys_in
        else:
            if verbose==True:
                print('    Good vsys = '+"{0:4.2f}".format(vsys_out)+'.')
            alteration[alteration_label.index('vsys_altered')] = True
    else:
        vsys_out = vsys_in
        if alteration[alteration_label.index('vsys_altered')]==True:
            if verbose==True:
                print('Vsys output was not generated, as a good value has already been found!')
        else:
            if verbose==True:
                print('Vsys output either failed or was Disabled.')
                
    # Coords
    if ~np.isnan(xcen_out*ycen_out):
        if verbose==True:
            print(name+'\'s xcen and ycen : ('+"{0:4.2f}".format(xcen_in)+','+"{0:4.2f}".format(ycen_in)\
                  +') to ('+"{0:4.2f}".format(xcen_out)+','+"{0:4.2f}".format(ycen_out)+').')
        if np.sqrt((np.abs(xcen_out - xcen_in)/xcen_in)**2 + (np.abs(ycen_out - ycen_in)/ycen_in)**2) > 0.07:
            if verbose==True:
                print('    Bad coords off by >7%! Results discarded.')
            xcen_out = xcen_in
            ycen_out = ycen_in
        else:
            if verbose==True:
                print('    Good coords = ('+"{0:4.2f}".format(xcen_out)\
                      +','+"{0:4.2f}".format(ycen_out)+').')
            alteration[alteration_label.index('coords_altered')] = True
    else:
        xcen_out = xcen_in
        ycen_out = ycen_in
        if alteration[alteration_label.index('coords_altered')]==True:
            if verbose==True:
                print('xcen, ycen outputs were not generated, as good values have already been found!')
        else:
            if verbose==True:
                print('xcen, ycen outputs either failed or was Disabled.')
    
    # Bar PA
    if ~np.isnan(bar_PA_out):
        if verbose==True:
            print(name+'\'s BAR PA : '+"{0:4.2f}".format(bar_PA_in)+' -> '+"{0:4.2f}".format(bar_PA_out)+'.')
        if (180.*u.deg - np.abs(np.abs(bar_PA_in - bar_PA_out) - 180.*u.deg))>=40*u.deg:
            if verbose==True:
                print('    Bad Bar PA off by >40deg! Result discarded.')
            bar_PA_out = bar_PA_in
        else:
            if verbose==True:
                print('    Good Bar PA = '+"{0:4.2f}".format(bar_PA_out)+'.')
            alteration[alteration_label.index('bar_PA_altered')] = True
    else:
        bar_PA_out = bar_PA_in
        if alteration[alteration_label.index('bar_PA_altered')]==True:
            if verbose==True:
                print('Bar PA output was not generated, as a good value has already been found!')
#         else:
#             if verbose==True:
#                 print('Bar PA output either failed or was Disabled.')
    
    # Warp Radius (min radius at which warp begins)
    if ~np.isnan(r_w_out):
        if verbose==True:
            print(name+'\'s r_w : '+"{0:4.2f}".format(r_w_in)+' -> '+"{0:4.2f}".format(r_w_out)+'.')
        if (np.abs(r_w_in - r_w_out)/r_w_in > 0.15):
            if verbose==True:
                print('    Bad r_w off by >15%! Result discarded.')
            r_w_out = r_w_in
        else:
            if verbose==True:
                print('    Good r_w = '+"{0:4.2f}".format(r_w_out)+'.')
            alteration[alteration_label.index('r_w_altered')] = True
    else:
        r_w_out = r_w_in
        if alteration[alteration_label.index('r_w_altered')]==True:
            if verbose==True:
                print('r_w output was not generated, as a good value has already been found!')
#         else:
#             if verbose==True:
#                 print('r_w output either failed or was Disabled.')
                
    # Warp Eps
    if ~np.isnan(warp_eps_out):
        if verbose==True:
            print(name+'\'s warp_eps : '+"{0:4.2f}".format(warp_eps_in)\
                  +' -> '+"{0:4.2f}".format(warp_eps_out)+'.')
        if (np.abs(warp_eps_in - warp_eps_out)/warp_eps_in > 0.15):
            if verbose==True:
                print('    Bad warp_eps off by >15%! Result discarded.')
            warp_eps_out = warp_eps_in
        else:
            if verbose==True:
                print('    Good warp_eps = '+"{0:4.2f}".format(warp_eps_out)+'.')
            alteration[alteration_label.index('warp_eps_altered')] = True
    else:
        warp_eps_out = warp_eps_in
        if alteration[alteration_label.index('warp_eps_altered')]==True:
            if verbose==True:
                print('warp_eps output was not generated, as a good value has already been found!')
#         else:
#             if verbose==True:
#                 print('warp_eps output either failed or was Disabled.')
                
    # Warp PA
    if ~np.isnan(warp_PA_out):
        if verbose==True:
            print(name+'\'s warp_PA : '+"{0:4.2f}".format(warp_PA_in)+' -> '+"{0:4.2f}".format(warp_PA_out)+'.')
        if (np.abs(warp_PA_in - warp_PA_out)/warp_PA_in > 0.15):
            if verbose==True:
                print('    Bad warp_PA off by >15%! Result discarded.')
            warp_PA_out = warp_PA_in
        else:
            if verbose==True:
                print('    Good warp_PA = '+"{0:4.2f}".format(warp_PA_out)+'.')
            alteration[alteration_label.index('warp_PA_altered')] = True
    else:
        warp_PA_out = warp_PA_in
        if alteration[alteration_label.index('warp_PA_altered')]==True:
            if verbose==True:
                print('warp_PA output was not generated, as a good value has already been found!')
#         else:
#             if verbose==True:
#                 print('warp_PA output either failed or was Disabled.')
    
    
    if verbose==True:
        print(alteration)
    
    if chi2_out>150:
        if verbose==True:
            print(' (!) WARNING : chi^2 = '+"{0:4.2f}".format(chi2_out)+'.')
    return xcen_out,ycen_out, PA_out, eps_out, incl_out, vsys_out, bar_PA_out, r_w_out, warp_PA_out, warp_eps_out,\
           xcen_in, ycen_in,  PA_in,  eps_in,  incl_in,  vsys_in,  bar_PA_in,  r_w_in,  warp_PA_in,  warp_eps_in,\
            alteration, chi2_out
    # ^ Edit later on if need be.
    
def checkcustom(name,iteration=1,diskfit_folder='diskfit_procedural/'):
    '''
    Checks if a galaxy has a custom input
    file, for a given "iteration".
    (They're not really iterations anymore,
    BUT higher-numbered files will be
    prioritized. e.g. '_c2' will be read
    before '_c1'.)
    
    -----------
    Parameters:
    -----------
    
    name : str
        Galaxy's name.
    iteration=1 : int
        Iteration number.
    diskfit_folder='diskfit_procedural/' : str
        Folder name, for data mode
        and data resolution.
    '''
    # Read the DiskFit output files!
    output_filename = "/media/jnofech/BigData/galaxies/"+diskfit_folder+"Output/"+name.lower()\
                          +"_c"+str(iteration)+".out"
        
    if os.path.isfile(output_filename):
        return True
    else:
        return False
    
def read_rotcurve(name,data_mode='7m',iteration=1,\
              custom_input=False, diskfit_folder='diskfit_procedural/'):
    '''
    Reads the output file for 'iteration',
    and returns the rotation curve.
    
    -----------
    Parameters:
    -----------
    
    name : str
        Galaxy's name.
    data_mode : str
        '7m' (default) - uses 7m data.
        '12m' (unsupported) - 12m data.
    iteration=1 : int
        Iteration number.
    custom_input=False : bool
        Toggles whether a custom run is
        read instead of an automated run.
    '''
    # Read the DiskFit output files!
    extension = custom_input*("_c") + (not custom_input)*("_v")
    output_filename = "/media/jnofech/BigData/galaxies/"+diskfit_folder+"Output/"+name.lower()\
                          +extension+str(iteration)+".out"
    if os.path.isfile(output_filename):
        file = open(output_filename,'r')
    else:
        print('WARNING: '+name+'\'s output file ("'+extension+str(iteration)+'") missing!')
        return [np.nan]*3

    # Initializing. Do not touch.
    i=0                         # Line number, for the 'lines' list. Line no. for the text file is i+1.
    lines = []
    is_output = False
    # Save the file to a list of strings, so we can read more easily!
    for line in file:
        lines = np.append(lines,line)

    # Pick out all the input/output parameters!
    # Also, pick out the rotcurves with error bars.
    Rd, vrotd, vrot_ed = [np.nan]*3

    for i in range(0,len(lines)):
        if lines[i][104:113]=='---------':
            skiprows = i+1
            
    Rd, x, vrotd, vrot_ed,x,x,x,x,x,x = np.loadtxt(output_filename,skiprows=skiprows,unpack=True)
    
    # Check if output file exists, but rotcurve is entirely zeros:
    if np.nansum(vrotd)+np.nansum(vrot_ed)==0.0:
        return [np.nan]*3
    
    return Rd, vrotd, vrot_ed

def read_all_outputs(gal,mode='params',diskfit_folder='diskfit_procedural/',use_custom=True,verbose=False):
    '''
    Reads through all DiskFit outputs in
    specified folder, and returns either
    fitted parameters OR fitted rotcurve.
    
    Parameters:
    -----------
    gal : Galaxy or str
        Galaxy.
    mode='params' : str
        'params'   - returns parameters (xcen,
                     ycen,PA,eps,incl,vsys).
        'rotcurve' - returns rotation curve
                     (R,vrot,vrot_e).

    Returns:
    --------
    See "mode" in Parameters.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = tools.galaxy(name)
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    if mode.lower() in ['params','parameters']:
        mode='params'
    elif mode.lower() in ['rotcurve','rc','rotationcurve','rotation curve','rotation']:
        mode='rotcurve'
    
    PA_orig   = gal.position_angle
    incl_orig = gal.inclination
    vsys_orig = gal.vsys
    if np.isnan(vsys_orig):
        print('dig.read_all_outputs : WARNING - gal.vsys is NaN! Taking mean value of 7m mom1 map.')
        I_mom1 = tools.mom1_get(gal,'7m')
        vsys_orig = np.nanmean(I_mom1)*u.km/u.s

    # Generate Alterations only!
    alteration_label = ['PA_altered','eps_altered','coords_altered','vsys_altered','bar_PA_altered',\
                        'r_w_altered','warp_eps_altered','warp_PA_altered']
    alteration = [False]*len(alteration_label)

    for j in range(1,5):
        # Read previous outputs!
        # If the run fails at an iteration (say, j=3), then the previous (i.e. most recent "good")
        #    `alteration` will be kept.
        x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x\
        = read_output(name,iteration=j,alteration=alteration,verbose=verbose,\
                          diskfit_folder=diskfit_folder)
    # Now, actually read outputs!
    # Check for custom inputs!
    for ver in range(0,5):
        custom_input = checkcustom(name,ver,diskfit_folder)
        if custom_input==True and use_custom==True:
            print('plotting : CUSTOM INPUT DETECTED! Reading custom outputs.')
            break
        elif custom_input==True and use_custom==False:
            print('plotting : CUSTOM INPUT DETECTED... but, IGNORE and read procedural outputs instead.')
            custom_input=False
            break
    
    # Read outputs!
    iteration=4
    while True:
        xcen_out,ycen_out, PA_out, eps_out, incl_out, vsys_out, bar_PA_out, r_w_out, warp_PA_out, warp_eps_out,\
        xcen_in, ycen_in,  PA_in,  eps_in,  incl_in,  vsys_in,  bar_PA_in,  r_w_in,  warp_PA_in,  warp_eps_in,\
        alteration, chi2_out = read_output(name,iteration=iteration,alteration=alteration,\
                                                  custom_input=custom_input,diskfit_folder=diskfit_folder)    
        if [xcen_out,ycen_out, PA_out, eps_out, incl_out, vsys_out, bar_PA_out, r_w_out, warp_PA_out, warp_eps_out,\
            xcen_in, ycen_in,  PA_in,  eps_in,  incl_in,  vsys_in,  bar_PA_in,  r_w_in,  warp_PA_in,  warp_eps_in,\
            alteration, chi2_out] == [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
            np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
            alteration,np.nan] and iteration!=0:
            print('plotting : OUTPUT FILE NOT DETECTED! Reverting to previous output file.')
            iteration = iteration-1
        elif iteration==0:
            print('plotting : NOT A SINGLE OUTPUT FILE DETECTED! Very sad.')
            break
        else:
            break
    if iteration!=0:
        print('Output for iteration='+str(iteration)+' successful!')
    
    if np.isnan(PA_out):
        print('WARNING: PA never changed!')
        PA_out = PA_orig
    if np.isnan(incl_out):
        print('WARNING: incl never changed!')
        incl_out = incl_orig
    if np.isnan(vsys_out):
        print('WARNING: vsys never changed!')
        vsys_out = vsys_orig
    
    if mode=='params':
        return xcen_out,ycen_out, PA_out, eps_out, incl_out, vsys_out
    elif mode=='rotcurve':
        # Read rotcurves!
        Rd, vrotd, vrot_ed = read_rotcurve(name,iteration=iteration,custom_input=custom_input,\
                                           diskfit_folder=diskfit_folder)
        return Rd,vrotd,vrot_ed
