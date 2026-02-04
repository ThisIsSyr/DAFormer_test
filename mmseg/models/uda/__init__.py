# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from .dacs_wavelet import DACSWithWavelet

__all__ = ['DACS', 'DACSWithWavelet']


