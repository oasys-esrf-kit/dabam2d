from oasys_srw.srwlib import *
from srxraylib.plot.gol import plot_image
import numpy as np
from oasys.util.oasys_util import write_surface_file


# def write_surface_file(zz, xx, yy, file_name, overwrite=True):
#
#     if (os.path.isfile(file_name)) and (overwrite==True): os.remove(file_name)
#
#     if not os.path.isfile(file_name):  # if file doesn't exist, create it.
#         file = h5py.File(file_name, 'w')
#         # points to the default data to be plotted
#         file.attrs['default']          = subgroup_name
#         # give the HDF5 root some more attributes
#         file.attrs['file_name']        = file_name
#         file.attrs['file_time']        = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
#         file.attrs['creator']          = 'write_surface_file'
#         file.attrs['code']             = 'Oasys'
#         file.attrs['HDF5_Version']     = h5py.version.hdf5_version
#         file.attrs['h5py_version']     = h5py.version.version
#         file.close()
#
#     file = h5py.File(file_name, 'a')
#
#     try:
#         f1 = file.create_group(subgroup_name)
#     except:
#         f1 = file[subgroup_name]
#
#     f1z = f1.create_dataset("Z", data=zz)
#     f1x = f1.create_dataset("X", data=xx)
#     f1y = f1.create_dataset("Y", data=yy)
#
#
#     # NEXUS attributes for automatic plot
#     f1.attrs['NX_class'] = 'NXdata'
#     f1.attrs['signal'] = "Z"
#     f1.attrs['axes'] = [b"Y", b"X"]
#
#     f1z.attrs['interpretation'] = 'image'
#     f1x.attrs['long_name'] = "X [m]"
#     f1y.attrs['long_name'] = "Y [m]"
#
#
#     file.close()


def read_srw_intensity_dat(file_name):
    '''Reader for metrology data to numpy array'''

    image, mesh = srwl_uti_read_intens_ascii(file_name)
    image = np.reshape(image, (mesh.ny, mesh.nx))

    x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    return image, x, y, mesh


def write_generic_h5_surface(s, xx, yy, filename='presurface.hdf5',subgroup_name="surface_file"):
    write_surface_file(s.T, xx, yy, filename, overwrite=True)
    print("write_h5_surface: File for OASYS " + filename + " written to disk.")

def metadata_set_info(
                      YEAR_FABRICATION=None,
                      SURFACE_SHAPE=None,
                      FUNCTION=None,
                      LENGTH=None,
                      WIDTH=None,
                      THICK=None,
                      LENGTH_OPTICAL=None,
                      SUBSTRATE=None,
                      COATING=None,
                      FACILITY=None,
                      INSTRUMENT=None,
                      POLISHING=None,
                      ENVIRONMENT=None,
                      SCAN_DATE=None,
                      CALC_HEIGHT_RMS=None,
                      CALC_HEIGHT_RMS_FACTOR=None,
                      CALC_SLOPE_RMS=None,
                      CALC_SLOPE_RMS_FACTOR=None,
                      USER_EXAMPLE=None,
                      USER_REFERENCE=None,
                      USER_ADDED_BY=None,
                      ):

    metadata = {}
    metadata["YEAR_FABRICATION"] = YEAR_FABRICATION
    metadata["SURFACE_SHAPE"] = SURFACE_SHAPE
    metadata["FUNCTION"] = FUNCTION
    metadata["LENGTH"] = LENGTH
    metadata["WIDTH"] = WIDTH
    metadata["THICK"] = THICK
    metadata["LENGTH_OPTICAL"] = LENGTH_OPTICAL
    metadata["SUBSTRATE"] = SUBSTRATE
    metadata["COATING"] = COATING
    metadata["FACILITY"] = FACILITY
    metadata["INSTRUMENT"] = INSTRUMENT
    metadata["POLISHING"] = POLISHING
    metadata["ENVIRONMENT"] = ENVIRONMENT
    metadata["SCAN_DATE"] = SCAN_DATE
    metadata["CALC_HEIGHT_RMS"] = CALC_HEIGHT_RMS
    metadata["CALC_HEIGHT_RMS_FACTOR"] = CALC_HEIGHT_RMS_FACTOR
    metadata["CALC_SLOPE_RMS"] = CALC_SLOPE_RMS
    metadata["CALC_SLOPE_RMS_FACTOR"] = CALC_SLOPE_RMS_FACTOR
    metadata["USER_EXAMPLE"] = USER_EXAMPLE
    metadata["USER_REFERENCE"] = USER_REFERENCE
    metadata["USER_ADDED_BY"] = USER_ADDED_BY

    return metadata

if __name__ == "__main__":

    directory_in = "/users/srio/Oasys/metrology/"

    # data file
    ii = 0
    for i in range(1,6):
        #
        # lens profile
        #
        ii += 1
        root = "Be_R50_lens_0%d" % i
        filein = directory_in + root + ".dat"
        image, x, y, mesh = read_srw_intensity_dat(filein)
        write_generic_h5_surface(image, x, y, filename="../data/dabam2d-%03d.h5" % ii)

        metadata = metadata_set_info(USER_ADDED_BY="celestre+srio",
                                     USER_REFERENCE="From file: %s.dat " % root,
                                     FACILITY="ESRF",
                                     FUNCTION="2D Be lenses (R=50um;A=~400um)",
                                     )
        # dump metadata
        outFile = "../data/dabam2d-%03d.txt" % ii
        with open(outFile, mode='w') as f1:
            json.dump(metadata, f1, indent=2)
            print ("File "+outFile+" containing metadata written to disk.")

        #
        # lens Zernike fit
        #
        ii += 1
        root = "Be_R50_lens_LF_0%d" % i
        filein = directory_in + root + ".dat"
        image, x, y, mesh = read_srw_intensity_dat(filein)
        # plot_image(image.T, x, y, title=filein)
        write_generic_h5_surface(image, x, y, filename="../data/dabam2d-%03d.h5" % ii)

        metadata = metadata_set_info(USER_ADDED_BY="celestre+srio",
                                     USER_REFERENCE="From file: %s.dat " % root,
                                     FACILITY="ESRF",
                                     FUNCTION="Zernike fit of 2D Be lenses (R=50um;A=~400um)",
                                     )
        # dump metadata
        outFile = "../data/dabam2d-%03d.txt" % ii
        with open(outFile, mode='w') as f1:
            json.dump(metadata, f1, indent=2)
            print ("File "+outFile+" containing metadata written to disk.")

