import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from srxraylib.plot.gol import plot
from oasys.util.oasys_util import write_surface_file
from srxraylib.metrology.profiles_simulation import slopes
from convert_from_srw_format import metadata_set_info
    
def convert(file):


    import scipy.ndimage.filters as sc
    import numpy as np
    import matplotlib.pyplot as plt

    # In[79]:

    # file = 'test.txt'
    x, y, z = np.loadtxt(file, delimiter=';', unpack=True)

    # In[80]:

    xvals = np.unique(x)

    # In[81]:

    len(xvals)

    # In[82]:

    yvals = np.unique(y)
    len(yvals)

    # In[83]:

    xvals

    # In[84]:

    yvals

    # In[85]:

    zvals = z.reshape(75, 3178)

    # In[86]:

    zvals

    # In[87]:

    # get_ipython().run_line_magic('matplotlib', 'inline')

    # In[88]:

    imgplot = plt.imshow(zvals)

    # In[89]:

    fig, ax = plt.subplots()
    ax.set_title('height profile')
    ax.set_ylabel('height [m]')
    ax.set_xlabel('position [m]')
    ax.plot(xvals, zvals[37])  # Plot some data on the axes.

    # In[93]:

    # subtract best sphere/cylinder from line profile across cenre of mirror
    quadfit = np.polyfit(xvals, zvals[37], 2)
    print(quadfit)
    print('radius of curvature = %.2f m' % (1 / (2 * quadfit[0])))
    height_err = zvals[37] - np.polyval(quadfit, xvals)
    slope_err = (zvals[37, :-1] - zvals[37, 1:]) / (xvals[1] - xvals[0])
    slope_err_xvals = (xvals[:-1] + xvals[1:]) / 2
    # remove residual curvature
    linfit = np.polyfit(slope_err_xvals, slope_err, 1)
    slope_err = slope_err - np.polyval(linfit, slope_err_xvals)
    print(xvals[1] - xvals[0])
    print('rms height err = %.2e m, pv height error = %.2e m' % (height_err.std(), height_err.max() - height_err.min()))
    print('rms slope err = %.2e rad, pv slope error = %.2e rad' % (slope_err.std(), slope_err.max() - slope_err.min()))
    # filter slope errors to reduce noise. Gaussian sigma 3 pixels equivalent psf of ~ 2mm fwhm for XFEL data.
    filtered_slope_err = sc.gaussian_filter(slope_err, 3)
    print('rms filtered slope err = %.2e rad, pv slope error = %.2e rad' % (
    filtered_slope_err.std(), filtered_slope_err.max() - filtered_slope_err.min()))

    # In[52]:

    fig, ax = plt.subplots()
    ax.set_title('height error profile')
    ax.set_ylabel('height [m]')
    ax.set_xlabel('position [m]')
    ax.plot(xvals, height_err)  # Plot some data on the axes.

    # In[74]:

    fig, ax = plt.subplots()
    ax.set_title('slope error profile')
    ax.set_ylabel('slope error [rad]')
    ax.set_xlabel('position [m]')
    ax.plot(slope_err_xvals, slope_err)  # Plot some data on the axes.

    # In[56]:

    print('rms height err = %.2e m, pv height error = %.2e m' % (height_err.std(), height_err.max() - height_err.min()))

    # pv_height_err=height_err.max()-height_err.min()

    # In[38]:

    fig, ax = plt.subplots()
    ax.set_title('height profile')
    ax.set_ylabel('height [m]')
    ax.set_xlabel('position [m]')
    ax.plot(x, z)  # Plot some data on the axes.

    # In[57]:

    # fit whole surface with best cylinder
    quadfit = np.polyfit(x, z, 2)
    print(quadfit)
    print('radius of curvature = %.2f m' % (1 / (2 * quadfit[0])))
    height_err_surf = z - np.polyval(quadfit, x)

    # In[58]:

    # 2d array of surface height_error after best cylinder subtraction
    height_err_surf = height_err_surf.reshape(75, 3178)

    # In[59]:

    imgplot = plt.imshow(height_err_surf)

    # In[60]:

    height_err_surf.std(), (height_err_surf.max() - height_err_surf.min())

    # In[102]:

    plt.pcolormesh(xvals, yvals, height_err_surf)

    from srxraylib.plot.gol import plot_image
    plot_image(height_err_surf, yvals, xvals, aspect='auto', show=0)

    plot_image(z.reshape(75, 3178), yvals, xvals, aspect='auto', show=0, title="RAW")

    print(height_err_surf.shape, yvals.shape, xvals.shape)
    # In[ ]:

    plt.show()

    return yvals, xvals, z.reshape(75, 3178)

    # In[ ]:

    # In[ ]:


if __name__ == '__main__':
    import json

    file_name = '/users/srio/OASYS1.2/shadow3-scripts/METROLOGY/test.txt'
    x, y, z = convert(file_name)
    print(z.shape, x.shape, y.shape, z.min(), z.max())

    from srxraylib.plot.gol import plot_image
    plot_image(z*1e6, y*1e3, x*1e3, aspect="auto")
    
    # x,z  = detrend_best_circle(x,y,z,fitting_domain_ratio=0.5, plotting=True)
    #
    # print(z.shape)
    # #plot2d(x,y,z)
    #
    # z2 = app_gaussian(z, sigma_0= 6, sigma_1 = 2)
    #
    # z3 =  scale_profile(z2,1)
    #
    # #plot2d(x,y,z)
    # slp = slopes(z, y, x, silent=0, return_only_rms=0)
    #
    # slp_y = np.round(slp[1][1]*1e6, 3)

    ii = 11
    output_filename = "dabam2d-%03d.h5" % ii

    write_surface_file(z.T, y, x, output_filename, overwrite=True)

    print("write_h5_surface: File for OASYS " + output_filename + " written to disk.")

    # print(">>>>>", z.shape, y.shape, x.shape,)

    # dump metadata
    metadata = metadata_set_info(USER_ADDED_BY="barrett+cammarata+reyes_herrera+srio",
                                 USER_REFERENCE="From file: %s" % file_name,
                                 FACILITY="EuXFEL",
                                 FUNCTION="JTEC mirror",
                                 )

    outFile = "dabam2d-%03d.txt" % ii
    with open(outFile, mode='w') as f1:
        json.dump(metadata, f1, indent=2)
        print("File " + outFile + " containing metadata written to disk.")

