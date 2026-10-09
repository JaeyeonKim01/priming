"""Encoder order, reference filenames, and normalization for the ImageNet protocols."""

SPECS = (
    ('FID(ADM)', 'inception', 'inception', None, 'guided_diffusion_stats.npz'),
    ('convnext', 'convnextv2_base.fcmae_ft_in22k_in1k', 'convnext', 224, 'convnext_in256_t224_stats.npz'),
    ('dinov2_cls', 'vit_large_patch14_dinov2.lvd142m', 'dinov2', 256, 'vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz'),
    ('mae_cls', 'vit_large_patch16_224.mae', 'mae', 224, 'vit_large_patch16_224_mae_in256_t224_stats.npz'),
    ('siglip_cls', 'vit_so400m_patch16_siglip_256.v2_webli', 'siglip', 224, 'vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz'),
    ('clip_cls', 'vit_large_patch14_clip_224.openai', 'clip', 256, 'vit_large_patch14_clip_224_openai_in256_t256_stats.npz'),
)
BASELINES = dict(zip((row[0] for row in SPECS), (1.68, 56.87, 14.19, .04, .60, 5.60)))
PAPER_SPECS = tuple(row for row in SPECS if row[0] != 'convnext')
