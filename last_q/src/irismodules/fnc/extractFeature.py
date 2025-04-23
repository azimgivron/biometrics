##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
from cv2 import imread
from fnc.encode import encode
from fnc.normalize import normalize
from fnc.segment import segment

##-----------------------------------------------------------------------------
##  Parameters for extracting feature
##	(The following parameters are default for CASIA1 dataset)
##-----------------------------------------------------------------------------
# Segmentation parameters
eyelashes_thres = 80

# Normalisation parameters
radial_res = 20
angular_res = 240

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def extractFeature(im, eyelashes_thres=80, use_multiprocess=True):
    """
    Description:
            Extract features from an iris image

    Input:
            im_filename			- The input iris image
            use_multiprocess	- Use multiprocess to run

    Output:
            template			- The extracted template
            mask				- The extracted mask
            im_filename			- The input iris image
    """
    # Perform segmentation
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)

    # Perform normalization
    polar_array, noise_array = normalize(
        imwithnoise,
        ciriris[1],
        ciriris[0],
        ciriris[2],
        cirpupil[1],
        cirpupil[0],
        cirpupil[2],
        radial_res,
        angular_res,
    )

    # Perform feature encoding
    template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

    # Return
    return template, mask
