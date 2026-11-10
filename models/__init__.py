# -*- coding: utf-8 -*-
"""EAGG-UNet model package."""

from .eagg import EAGG, UNetWithFeatureSelect

__all__ = ['EAGG', 'UNetWithFeatureSelect', 'build_model']


def build_model(config):
    """Build an :class:`EAGG` model from a config dict.

    Unknown keys are ignored by ``EAGG`` itself, so configs shared with other
    experiments keep working.
    """
    return EAGG(
        num_classes=config.get('num_classes', 20),
        in_channels=config.get('in_channels', 5),
        sigma_spatial=config.get('sigma_spatial', 0.3),
        sigma_feature=config.get('sigma_feature', 0.8),
        use_spatial=config.get('use_spatial', True),
        use_feature=config.get('use_feature', True),
        gcn_spatial_size=config.get('gcn_spatial_size', 32),
        freeze_bn=config.get('freeze_bn', False),
    )
