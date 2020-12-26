"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'paths': {
            'train': str,
            'validation': str,
            'logs': str,
            'train_images': str,
            'train_qeustions': str,
            'train_answers': str,
            'val_images': str,
            'val_qeustions': str,
            'val_answers': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'img_encoder_out_classes': int,
        'img_encoder_batchnorm': bool,
        'img_encoder_dropout': float,
        'text_embedding_tokens': int,
        'text_embedding_features': int,
        'text_lstm_features': int,
        'text_dropout': float,
        'attention_mid_features': int,
        'attention_glimpses': int,
        'attention_dropout': float,
        'classifier_dropout': float, 
        'classifier_mid_features': int,
        'classifier_out_classes': int,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
