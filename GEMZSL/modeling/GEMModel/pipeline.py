from .GEMNet import build_GEMNet

_ZSL_META_ARCHITECTURES = {
    "GEMModel": build_GEMNet,
}

def build_zsl_pipeline(cfg):
    meta_arch = _ZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)