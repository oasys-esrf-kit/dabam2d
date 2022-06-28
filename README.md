# dabam2d


Repository with files for 2D optical surfaces. They usually contain surface errors or surface shapes. 

Data files are data/dabam2d-xxx.h5 and data/dabam2d-xxx.txt with xxx the index (they are enumerated). 

*.h5 file contains the surface data in Oasys surface_file format. Read it with: 

``` import h5py
    file = h5py.File(file_name, 'r')
    xx = file["surface_file" + "/X"][()]
    yy = file["surface_file" + "/Y"][()]
    zz = file["surface_file" + "/Z"][()]
    file.close()
```

usually X is horizontal and Y vertical with respect to the beam. Z the height. Note that Z.shape is (Y.size, X.size).

*.txt file contains metadata

Data files can be accessed in Oasys (Surface/FEA Reader) using the URL: https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-xxx.h5

Data files can be seen using a h5 viewer, e.g. https://h5web.panosc.eu/h5wasm?url=https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-001.h5 


