# dabam2d - an open-source database of 2D metrology profiles

![alt text](https://github.com/oasys-esrf-kit/dabam2d/blob/main/poster.png)

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
metadata = {
    "YEAR_FABRICATION": YEAR_FABRICATION,
    "SURFACE_SHAPE": SURFACE_SHAPE,
    "FUNCTION": FUNCTION,
    "RS": RS,                         # sagittal radius of curvature (hor. R if lenses)
    "RM": RM,                         # meridional/tangential radius of curvature (ver. R if lenses)
    "FOCUS_DIR": FOCUS_DIR,           # 1 for 1D and 2 for 2D or -1 for nor defined
    "WIDTH": WIDTH,                   # horizontal geom. aperture if lenses
    "LENGTH": LENGTH,                 # vertical geom. aperture if lenses
    "THICK": THICK,
    "SUBSTRATE": SUBSTRATE,
    "COATING": COATING,
    "MATERIAL": MATERIAL,
    "FACILITY": FACILITY,             # beamline, if data from at-wavelength metrology
    "INSTRUMENT": INSTRUMENT,         # technique, if data from at-wavelength metrology
    "POLISHING": POLISHING,
    "ENVIRONMENT": ENVIRONMENT,
    "SCAN_DATE": SCAN_DATE,
    "CALC_HEIGHT_LENGTH_RMS": CALC_HEIGHT_LENGTH_RMS,
    "CALC_SLOPE__LENGTH_RMS": CALC_SLOPE__LENGTH_RMS,
    "CALC_HEIGHT_WIDTH_RMS": CALC_HEIGHT_WIDTH_RMS,
    "CALC_SLOPE_WIDTH_RMS": CALC_SLOPE_WIDTH_RMS,
    "USER_EXAMPLE": USER_EXAMPLE,
    "USER_REFERENCE": USER_REFERENCE,
    "USER_ADDED_BY": USER_ADDED_BY,
}
```

Data files can be accessed in Oasys (Surface/FEA Reader) using the URL:
```
https://github.com/oasys-esrf-kit/dabam2d/raw/main/data/dabam2d-xxxx.h5
```

Data files can be visualized using a h5 viewer:
```
https://h5web.panosc.eu/h5wasm?url=https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-XXXX.h5 
```

### Contribution:

Contact us to make your **metrology** profiles available online through DABAM2D - the open-source database of 2D metrology profiles.
