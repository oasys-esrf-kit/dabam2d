#!/usr/bin/python
# coding: utf-8

###################################################################################
# barc4RefractiveOptics
# Authors/Contributors: Rafael Celestre, Manuel Sanchez del Rio
# Rafael.Celestre@esrf.eu
# creation: 24.06.2019
# last update: 21.11.2022
#
# Check documentation:
# Celestre, R. et al. (2020). Recent developments in x-ray lenses modelling with SRW.
# Proc. SPIE 11493, Advances in Computational Methods for X-Ray Optics V, 11493-17.
# arXiv:2007.15461 [physics.optics] - https://arxiv.org/abs/2007.15461
#
# Copyright (c) 2019-2023 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

"""Legacy wavefront fitters from SpeckleMPy."""

import numpy as np

from .polynomial import (
    calc_chebyshev,
    calc_chebyshev_basis,
    calc_legendre,
    calc_legendre_basis,
    calc_zern_circ_basis,
    calc_zernike_circ,
    calc_zernike_rec,
)


def fit_zernike_circ(wavefront, zern_data={}, nmodes=37, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                rec_zern=True, err=None):
    """
    Fit **nmodes** Zernike modes to a **wavefront**. The **wavefront** will be fit to Zernike modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Zernike mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **zern_data** as cache. If this is not given, it will be generated. See calc_zern_circ_basis()
    for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] zern_data Zernike basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Zernike modes to fit
    @param [in] rad Radius of Zernike modes to fit
    @param [in] rec_zern Reconstruct Zernike modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Zernike mode amplitudes,
    the second element is a full 2D Zernike reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_zern_circ_basis() for details on **zern_data** cache
    """

    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5*max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape)-rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode	< 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        rad = -rad * min(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * min(wavefront.shape)

    # Make cropping slices to select only central part of the wavefront
    xslice = slice(int(center[0]-rad), int(center[0]+rad))
    yslice = slice(int(center[1]-rad), int(center[1]+rad))

    # Compute Zernike basis if absent
    # if (not (zern_data.has_key('modes')):
    if (not ( 'modes' in zern_data.keys())):
        tmp_zern = calc_zern_circ_basis(nmodes, rad)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
        zern_data['modes'][0].shape != (2*rad, 2*rad)):
        tmp_zern = calc_zern_circ_basis(nmodes, rad)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
        # wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec] * weight)).ravel()
        # This is 5x faster:
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask

        # Crop out central region of wavefront, then only select the orthogonal part of the Zernike modes (grid_mask)
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
        # wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec])).ravel()
        # This is 5x faster
        # RC190225:
        # FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M
        # and N are the input matrix dimensions. To use the future default and silence this warning we advise to pass
        # `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
        # wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel())[0]
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(),rcond=None)[0]

    wf_zern_vec[:startmode-1] = 0

    # Calculate full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_circ(wf_zern_vec, zern_data=zern_data, rad=min(wavefront.shape)/2)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid**2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean()**2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)



def fit_legendre(wavefront, leg_data={}, nmodes=44, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                    rec_leg=True, err=None):

    """
    Fit **nmodes** Legendre modes to a **wavefront**. The **wavefront** will be fit to Legendre modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Legendre mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **leg_data** as cache. If this is not given, it will be generated.
    See calc_legendre_basis() for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] leg_data Legendre basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Legendre modes to fit
    @param [in] rad Radius of Legendre modes to fit
    @param [in] rec_leg Reconstruct Legendre modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Legendre mode amplitudes,
    the second element is a full 2D Legendre reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_legendre_basis() for details on **leg_data** cache
    """
    zern_data = leg_data
    rec_zern = rec_leg
    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5*max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape)-rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode	< 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        npix = -rad * 2 * np.array(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * wavefront.shape

    # Convert rad and center if coordinates are fractional
    xslice = slice(int(center[1]-npix[1]/2), int(center[1]+npix[1]/2))
    yslice = slice(int(center[0]-npix[0]/2), int(center[0]+npix[0]/2))

    # Compute Zernike basis if absent
    if (not ( 'modes' in zern_data.keys())):
        tmp_zern = calc_legendre_basis(nmodes, npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
        zern_data['modes'][0].shape != (npix[0], npix[1])):
        tmp_zern = calc_legendre_basis(nmodes, npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(),rcond=None)[0]

    wf_zern_vec[:startmode-1] = 0

    # Calculates full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_rec(wf_zern_vec, zern_data=zern_data, npix=wavefront.shape)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid**2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean()**2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)



def fit_chebyshev(wavefront, leg_data={}, nmodes=44, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                 rec_leg=True, err=None):
    """
    Fit **nmodes** Chebyshev modes to a **wavefront**. The **wavefront** will be fit to Chebyshev modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Chebyshev mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **leg_data** as cache. If this is not given, it will be generated.
    See calc_chebyshev_basis() for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] leg_data Chebyshev basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Chebyshev modes to fit
    @param [in] rad Radius of Chebyshev modes to fit
    @param [in] rec_leg Reconstruct Chebyshev modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Chebyshev mode amplitudes,
    the second element is a full 2D Chebyshev reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_chebyshev_basis() for details on **leg_data** cache
    """
    zern_data = leg_data
    rec_zern = rec_leg
    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5 * max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape) - rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode < 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        npix = -rad * 2 * np.array(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * wavefront.shape

    # Convert rad and center if coordinates are fractional
    xslice = slice(int(center[1] - npix[1] / 2), int(center[1] + npix[1] / 2))
    yslice = slice(int(center[0] - npix[0] / 2), int(center[0] + npix[0] / 2))

    # Compute Zernike basis if absent
    if (not ('modes' in zern_data.keys())):
        tmp_zern = calc_chebyshev_basis(nmodes, npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
          zern_data['modes'][0].shape != (npix[0], npix[1])):
        tmp_zern = calc_chebyshev_basis(nmodes, npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1, -1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1, -1) * weight
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1, -1)
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(), rcond=None)[0]

    wf_zern_vec[:startmode - 1] = 0

    # Calculates full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_rec(wf_zern_vec, zern_data=zern_data, npix=wavefront.shape)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid ** 2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean() ** 2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)
