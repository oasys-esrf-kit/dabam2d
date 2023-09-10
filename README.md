# dabam2d - an open-source database of 2D metrology profiles


Repository with files for 2D optical surfaces. They usually contain surface errors or surface shapes. 

Data files are data/dabam2d-xxxx.h5 and data/dabam2d-xxxx.txt with xxxx being their index. They are enumerated as:
- *0*xxx for single x-ray lenses profiles of several materials and fabrication methods;
- *1*xxx for stacked lenses (CRL);
- *2*xxx for mirror profiles;

The *.h5 file contains the surface data in Oasys surface_file format. Read it with: 

```commandline
import h5py
file = h5py.File(file_name, 'r')
xx = file["surface_file" + "/X"][()]
yy = file["surface_file" + "/Y"][()]
zz = file["surface_file" + "/Z"][()]
file.close()
```

usually X is horizontal and Y vertical with respect to the beam. Z the height. Note that Z.shape is (Y.size, X.size).

The *.txt file contains metadata:

```commandline
metadata = {"YEAR_FABRICATION": YEAR_FABRICATION,
            "SURFACE_SHAPE": SURFACE_SHAPE,
            "FUNCTION": FUNCTION,
            "LENS_R_H": LENS_R_H,             # sagittal radius of curvature
            "LENS_R_V": LENS_R_V,             # meridional radius of curvature
            "LENS_FOCUS": LENS_FOCUS,         # 1 for 1D and 2 for 2D
            "WIDTH": WIDTH,                   # horizontal geometric aperture (for lenses)
            "LENGTH": LENGTH,                 # vertical geometric aperture (for lenses)
            "THICK": THICK,
            "LENGTH_OPTICAL": LENGTH_OPTICAL,
            "SUBSTRATE": SUBSTRATE,           # lens material (eg. Si, Be, Al, diamond, glassy-c, SU-8...)
            "COATING": COATING,
            "FACILITY": FACILITY,             # beamline, if data from at-wavelength metrology
            "INSTRUMENT": INSTRUMENT,         # technique, if data from at-wavelength metrology
            "POLISHING": POLISHING,
            "ENVIRONMENT": ENVIRONMENT,
            "SCAN_DATE": SCAN_DATE,
            "CALC_HEIGHT_RMS": CALC_HEIGHT_RMS,
            "CALC_HEIGHT_RMS_FACTOR": CALC_HEIGHT_RMS_FACTOR,
            "CALC_SLOPE_RMS": CALC_SLOPE_RMS,
            "CALC_SLOPE_RMS_FACTOR": CALC_SLOPE_RMS_FACTOR,
            "USER_EXAMPLE": USER_EXAMPLE,
            "USER_REFERENCE": USER_REFERENCE,
            "USER_ADDED_BY": USER_ADDED_BY,
}
```

Data files can be accessed in Oasys (Surface/FEA Reader) using the URL:
```
https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-xxxx.h5
```

Data files can be seen using a h5 viewer:
```
https://h5web.panosc.eu/h5wasm?url=https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-XXXX.h5 
```

### Contribution:

Contact us to make your profiles available online through DABAM2D - the open-source database of 2D metrology profiles.
