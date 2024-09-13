# deepsqueak.py
# 
# functions formerly in utils.deepsqueak moved to utils.callbacks
# 
# maintained in utils.deepsqueak for compatibility

def call_mat_stim_trial_loader(*args, **kwargs):
    from callbacks import call_mat_stim_trial_loader
    import warnings

    warnings.warn("Called call_mat_stim_trial_loader from utils.deepsqueak. Switch import to utils.callback!")

    return call_mat_stim_trial_loader(*args, **kwargs)


def multi_index_from_dict(*args, **kwargs):
    from callbacks import multi_index_from_dict
    import warnings

    warnings.warn("Called multi_index_from_dict from utils.deepsqueak. Switch import to utils.files!")

    return multi_index_from_dict(*args, **kwargs)