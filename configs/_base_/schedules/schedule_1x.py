# optimizer
optimizer = dict(type='SGD', lr=0.00003, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnealing',
    min_lr_ratio = 0,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3)
# change the epoch from 12 to 6
total_epochs = 12
