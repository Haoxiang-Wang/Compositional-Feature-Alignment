from .openai_imagenet_template import openai_imagenet_template
from .openai_imagenet_template_reduced import openai_imagenet_template_reduced
from .simple_template import simple_template
from .fmow_template import fmow_template
from .iwildcam_template import iwildcam_template
from .caltech101_template import caltech101_template
from .country211_template import country211_template
from .stanfordcars_template import stanfordcars_template
from .flowers102_template import flowers102_template
from .eurosat_template import eurosat_template
from .sun397_template import sun397_template
from .patchcamelyon_template import patchcamelyon_template
from .sst2_template import sst2_template
from .hatefulmemes_template import hatefulmemes_template
from .domainbed_template import domainbed_template
from .domainbed_env_prompts import PACS_prompts, DomainNet_prompts, OfficeHome_prompts
from .wilds_env_prompts import IWildCam_prompts, FMOW_prompts

no_template = [lambda c: str(c)]