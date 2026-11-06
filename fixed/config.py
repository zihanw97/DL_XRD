"""Reject recipe declarations that this deliberately narrow runner cannot execute."""
import math


EXPECTED_ARCHITECTURE=dict(input_channels=1,cnn_head='flatten',cnn_output_dim=2048,
    prehead_dim=256,rff_dim=64,rff_gamma=1/64,output_classes=3,
    spectral_normalization_layers=['fc1','fc2'],power_iterations=1,
    ridge_penalty=1.,covariance_momentum=None)


def validate_recipe(config):
    if config.get('schema')!='fixed_manuscript_base_training_v1':
        raise ValueError('Unknown fixed-training recipe schema')
    if config.get('architecture')!=EXPECTED_ARCHITECTURE:
        raise ValueError('This runner supports only the original manuscript flatten/SN-dense/RFF64 architecture')
    parameters=config.get('training',{})
    required={'optimizer','learning_rate','learning_rate_decay','weight_decay','max_epochs',
        'batch_size','patience','minimum_epochs','validation_ema_alpha','minimum_improvement',
        'source_classification_weight','target_anchor_classification_weight',
        'adapt_global_mmd_weight','adapt_anchor_prototype_weight','global_mmd_sigma',
        'requested_target_anchor_fraction','pseudo_label_classification','cluster_refinement',
        'backend_continuation','dense_refinement','retries'}
    if set(parameters)!=required:
        raise ValueError('Missing or unknown training parameters: '+str(set(parameters)^required))
    if parameters['optimizer']!='Adam':
        raise ValueError('This fixed runner implements Adam only')
    for flag in ['pseudo_label_classification','cluster_refinement','backend_continuation','dense_refinement']:
        if parameters[flag] is not False:
            raise ValueError(flag+' is excluded from this fixed manuscript recipe')
    if parameters['retries']!=0:
        raise ValueError('No automatic retry strategy is implemented')
    for key in ['max_epochs','batch_size','patience','minimum_epochs']:
        if isinstance(parameters[key],bool) or not isinstance(parameters[key],int) or parameters[key]<1:
            raise ValueError(key+' must be a positive integer')
    if parameters['minimum_epochs']>parameters['max_epochs']:
        raise ValueError('minimum_epochs cannot exceed max_epochs')
    nonnegative=['weight_decay','minimum_improvement','source_classification_weight',
                 'target_anchor_classification_weight','adapt_global_mmd_weight','adapt_anchor_prototype_weight']
    for key in nonnegative+['learning_rate','learning_rate_decay','validation_ema_alpha','global_mmd_sigma','requested_target_anchor_fraction']:
        value=parameters[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(key+' must be finite and numeric')
        if key in nonnegative and value<0:
            raise ValueError(key+' cannot be negative')
    if parameters['learning_rate']<=0 or parameters['global_mmd_sigma']<=0:
        raise ValueError('Learning rate and MMD bandwidth must be positive')
    for key in ['learning_rate_decay','validation_ema_alpha']:
        if not 0<parameters[key]<=1:
            raise ValueError(key+' must be in (0,1]')
    if not 0<=parameters['requested_target_anchor_fraction']<=1:
        raise ValueError('Requested anchor fraction must be in [0,1]')
    selection=config.get('selection',{})
    if (selection.get('criterion')!='EMA of source mean raw-logit validation CE plus target mean raw-logit validation CE'
        or selection.get('uses_training_accuracy') is not False
        or selection.get('uses_test_scores') is not False
        or selection.get('uses_latent_geometry') is not False):
        raise ValueError('Selection declarations must match the implemented validation-only EMA rule')
    return config
