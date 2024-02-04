from .caltech101 import Caltech101Val, Caltech101Test
from .domainbed import VLCS, PACS, OfficeHome, DomainNet, TerraIncognita
from .flowers102 import Flowers102Val, Flowers102Test
from .imagenet import ImageNet, ImageNetK, ImageNetSubsample, ImageNetTrain, ImageNetSubsampleValClasses, \
    ImageFolderWithPaths
from .imagenet_a import ImageNetAValClasses, ImageNetA
from .imagenet_r import ImageNetRValClasses, ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_vid_robust import ImageNetVidRobustValClasses, ImageNetVidRobust

from .objectnet import ObjectNetValClasses, ObjectNet
from .patchcamelyon import PatchCamelyonVal, PatchCamelyonTest
from .sst2 import sst2Val, sst2Test
from .stanfordcars import StanfordCarsVal, StanfordCarsTest

try:
    # requires wilds package
    from .wilds import IWildCamIDVal, IWildCamID, IWildCamOOD, IWildCamOODVal, IWildCamIDNonEmpty, IWildCamOODNonEmpty, IWildCam
    from .wilds import FMOWIDVal, FMOWID, FMOWOOD, FMOWOODVal, FMOW
except ImportError:
    pass

try:
    from .imagenetv2 import ImageNetV2 # requires ImageNetV2 package
except ImportError:
    pass
