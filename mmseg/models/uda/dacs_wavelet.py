# mmseg/models/uda/dacs_wavelet.py

import torch

from .dacs import DACS
from .wtconv import WaveletPreprocess
from .builder import UDA


@UDA.register_module()
class DACSWithWavelet(DACS):
    """
    在 DACS 的基础上，在进入分割网络之前，
    先对 source_img 和 target_img 做一次 WaveletPreprocess。
    """

    def __init__(self,
                 wavelet_cfg=None,
                 apply_to_source=True,
                 apply_to_target=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.apply_to_source = apply_to_source
        self.apply_to_target = apply_to_target

        default_cfg = dict(
            in_channels=3,
            kernel_size=5,
            wt_levels=1,
            wt_type='db1',
        )
        if wavelet_cfg is not None:
            default_cfg.update(wavelet_cfg)

        self.wavelet_pre = WaveletPreprocess(**default_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      **kwargs):

        # 在进入原始 DACS 逻辑之前，对 source/target 做小波变换
        if self.apply_to_source:
            img = self.wavelet_pre(img)
        if self.apply_to_target:
            target_img = self.wavelet_pre(target_img)

        # 调用原始 DACS 的 forward_train
        return super().forward_train(
            img=img,
            img_metas=img_metas,
            gt_semantic_seg=gt_semantic_seg,
            target_img=target_img,
            target_img_metas=target_img_metas,
            **kwargs
        )
