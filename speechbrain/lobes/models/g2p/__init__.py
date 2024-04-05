from speechbrain.utils.importutils import lazy_export_all

__getattr__ = lazy_export_all(__file__, __name__, export_subpackages=True)

from .dataio import *  # noqa
