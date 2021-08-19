import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from srxraylib.plot.gol import plot
from oasys.util.oasys_util import write_surface_file
from srxraylib.metrology.profiles_simulation import slopes
from convert_from_srw_format import metadata_set_info
    
def get_shadow_h5(file_name):
    """Function to get an h5 file with OASYS structure
    from 2D measurements """
    
    df = pd.read_csv(file_name, sep=';', header=None, comment='#', skiprows=1)
    
    df.columns = ['x(m)', 'y(m)', 'z(m)']

    # this part is to get the ordinates and the number of abscissas for each
    rows_shape = df.pivot_table(columns=['y(m)'], aggfunc='size')

    #print(rows_shape)

    #n_rows = rows_shape.size
    
    #print(n_rows)
    
    x_coors = []
    x_mins = []
    x_maxs = []
    z_heights = []
    
    for i,y in enumerate(rows_shape.index):
        sub_df = df[df['y(m)'] == y]
        x_coors.append(np.array(sub_df['x(m)']))
        x_mins.append(x_coors[i][0])
        x_maxs.append(x_coors[i][-1])
        z_heights.append(np.array(sub_df['z(m)']))

    # checking that all coordinates along the mirror have the same steps #
    if (all(x==x_mins[0] for x in x_mins)) and (all(x==x_maxs[0] for x in x_maxs)):
        print("All elements in x_coors are the same")
        x = x_coors[0]
        y = rows_shape.index
    else:
        #TODO: define coordinates along the mirror and interpolate all#
        #z for all y coord #
        pass 

        
    return np.array(x), np.array(y), np.array(z_heights)

if __name__ == '__main__':
    import json

    file_name = '/users/srio/OASYS1.2/shadow3-scripts/METROLOGY/ring256_TypbeB_F127001_frontside_ontissue_meas2__avg_2D.txt'
    x, y, z = get_shadow_h5(file_name)
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
    slp = slopes(z, y, x, silent=0, return_only_rms=0)
    #
    # slp_y = np.round(slp[1][1]*1e6, 3)

    ii = 12
    output_filename = "dabam2d-%03d.h5" % ii

    write_surface_file(z.T, y, x, output_filename, overwrite=True)

    print("write_h5_surface: File for OASYS " + output_filename + " written to disk.")

    print(">>>>>", z.T.shape, y.shape, x.shape,)

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

