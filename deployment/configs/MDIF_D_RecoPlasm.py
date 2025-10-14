_base_ = [
    #'mmpretrain/configs/resnet/resnet50_8xb32_in1k.py',
    '__base__/default_runtime.py',
    '__base__/models/resnet50.py',
    '__base__/schedules/imagenet_bs256.py'
    #'mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
]

work_dir='/kaggle/working/experiment_result'
data_root_ours = '/kaggle/input/malaria-parasite/final_malaria_full_class_classification_cropped'
data_root_iml = '/kaggle/input/iml-malaria-classificaiton/IML_Malaria_classification'
data_root_bbbc = '/kaggle/input/bbbc041-classification/BBBC041_classification'
batch_size = 32 
data_root = data_root_ours


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale = 224, crop_ratio_range=(0.6, 1.0)),
    dict(type='RandomFlip', prob= [0.25, 0.25], direction=['horizontal', 'vertical']), 
    #dict(type='RandAugment', policies=[dict(type='Rotate', magnitude_range=(0, 360))]),
    #dict(type='RandomRotate', max_angle = 180),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = 224),
    dict(type='PackInputs'),
]


train_dataloader_our = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_ours, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix= '',
        ann_file = "5 classes - May 2025/train_annotation_5classes.txt",
        with_label=True,
        pipeline=train_pipeline,
       ),
    sampler=dict(type='MalariaWeightedRandomSampler'),
    collate_fn=dict(type='default_collate'),
    drop_last = True,
)
train_dataloader_iml = dict(
    batch_size = batch_size,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_iml,
        data_prefix='',  
        ann_file = "5 classes - May 2025/train_annotation_5classes.txt",
        with_label=True,
        pipeline=train_pipeline,
       ),
    sampler=dict(type='MalariaWeightedRandomSampler'),
    collate_fn=dict(type='default_collate'),
    drop_last = True,
)
train_dataloader_bbbc = dict(
    batch_size = batch_size,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_bbbc,
        data_prefix='',  
        ann_file = "5 classes - May 2025/train_annotation_5classes.txt",
        with_label=True,
        pipeline=train_pipeline,
       ),
    sampler=dict(type='MalariaWeightedRandomSampler'),
    collate_fn=dict(type='default_collate'),
    drop_last = True,
)
train_dataloader = train_dataloader_our
train_cfg = dict(
    type = "MultiDomainTrainLoop", 
    dataloader = train_dataloader,
    dataloaders_multi_domain = [train_dataloader_our, 
                                train_dataloader_bbbc, 
                                train_dataloader_iml], 
    max_epochs = 50,
    val_interval = 5,
    _delete_ = True
)

val_dataloader_our = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_ours, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "5 classes - May 2025/val_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
val_dataloader_iml = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_iml, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "5 classes - May 2025/val_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
val_dataloader_bbbc = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_bbbc, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "5 classes - May 2025/val_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
val_dataloader = val_dataloader_our
val_cfg = dict(#type = "ValLoop",
                type = "MultiDomainValLoop",
               dataloaders_multi_domain= [val_dataloader_our, 
                                          val_dataloader_bbbc,
                                          val_dataloader_iml],
               domain_names = ['OurPlasmodium', 'BBBC041', 'IMLMalaria'])
val_evaluator = [dict(type='Accuracy'), dict(type='ConfusionMatrix'),
                 dict(type='MinorityMetrics'),
                ]

test_dataloader_our = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_ours, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "5 classes - May 2025/test_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
test_dataloader_bbbc = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_bbbc, 
        data_prefix='',  
        ann_file = "5 classes - May 2025/test_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
test_dataloader_iml = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root_iml, 
        data_prefix='',  
        ann_file = "5 classes - May 2025/test_annotation_5classes.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
test_dataloader = test_dataloader_our
test_cfg = dict(type = "MultiDomainTestLoop",
                dataloaders_multi_domain = [test_dataloader_our, 
                                            test_dataloader_bbbc,
                                            test_dataloader_iml],
                domain_names = ['OurPlasmodium', 'BBBC041', 'IMLMalaria'],
                )
test_evaluator = [dict(type='Accuracy'), 
                  dict(type='FalseClassification'),
                  dict(type='MinorityMetrics'),                  
                  dict(type='ConfusionMatrix')]


model = dict(
    #type='CustomClassifier',
    type='MultiDomainClassifier', 
    domain_idx = 0,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        #Config to load pretrained model
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    # neck=dict(type='GlobalAveragePooling'),
    neck=dict(type='MultiDomainInformationFusion',
              input_dim = 2048,
              hidden_dim= 2048,
              output_dim = 2048,
              n_domains = 3),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    #Normalization 
    data_preprocessor = dict(
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    )

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001, _delete_ = True)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer= optimizer,
    clip_grad=None)

param_scheduler = dict(by_epoch=True, gamma=0.1, milestones=[25], type='MultiStepLR')

visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='TensorboardVisBackend')])

# the default value of by_epoch is True
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True,
                                    save_best = ['OurPlasmodium/accuracy/top1',
                                                'OurPlasmodium/recall/parasitized'],
                                    rule = ['greater', 'greater'],
                                    ),
                    )