def get_model(cfg):
    if cfg.STAGE == 'stage1_vocaset':
        from .vae_trainer import VAETrainer as Model
        model = Model(cfg=cfg)
    elif cfg.STAGE == 'stage1_biwi':
        from .vae_trainer import VAETrainer as Model
        model = Model(cfg=cfg)
    elif cfg.STAGE == 'stage2_gpt':
        from .gpt_trainer import GptTrainer as Model
        model = Model(cfg=cfg)
    elif cfg.STAGE == 'stage2_diffar' or 'longseq':
        from .diff_ar import DIFF_AR as Model
        model = Model(cfg=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.STAGE))
    return model