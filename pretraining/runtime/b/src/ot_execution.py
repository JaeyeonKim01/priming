"""Select preserved feature implementations and optional execution settings."""
import torch


def configure_execution(provider, args):
    encoders = provider.encoders
    encoders.microbatch = int(args.ot_encoder_microbatch)
    if encoders.microbatch < 1:
        raise ValueError('encoder microbatch must be positive')
    if bool(getattr(args, 'ot_compile_encoders', False)):
        if not getattr(encoders, '_ot_execution_compiled', False):
            encoders.encode_family = torch.compile(encoders.encode_family, fullgraph=True)
            encoders._ot_execution_compiled = True
    elif getattr(encoders, '_ot_execution_compiled', False):
        raise ValueError('cannot disable compiled execution on an existing provider')


def create_provider(args, device):
    if args.ot_feature_stack == 'convnext_full_mae':
        from .ot_representations_full.provider import RepresentationProvider

        class FullFeaturewiseProvider(RepresentationProvider):
            def running_state_dict(self):
                state = super().running_state_dict()
                return {**state, 'version': 2, 'feature_stack': self.stack,
                        'loss_contract': self.loss_contract}

        provider = FullFeaturewiseProvider(args, device)
        provider.stack = 'convnext_full_mae'
        provider.encoders.family_names = ('mae', 'convnext')
        provider.loss_contract = dict(kind=args.discrepancy_kind,
            coefficient=float(args.lambda_ot), energy_delta=float(args.energy_delta),
            epsilon_mixture=dict(args.ot_epsilon_mixture),
            iterations=int(args.ot_num_iters), tolerance=float(args.ot_tolerance),
            self_weight=1.0)
    else:
        from .ot_representations.provider import RepresentationProvider
        provider = RepresentationProvider(args, device)
    configure_execution(provider, args)
    return provider
