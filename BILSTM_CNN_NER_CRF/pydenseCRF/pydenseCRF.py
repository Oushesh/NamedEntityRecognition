import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

d = dcrf.DenseCRF2D(640, 480, 5)  # width, height, nlabels
print ('d',d)


U = U.reshape((5,-1)) # Needs to be flat.
d.setUnaryEnergy(U)

