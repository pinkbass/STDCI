checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
)
save_image_config = dict(
    interval=10000,
)
optimizer = dict(type='Adam', lr = 0.0001)

# loss = dict(type='MSELoss')
loss = dict(type='L1Loss' )

runner = dict(max_epochs=200)

Lr = dict(init_lr=0.0001,final_lr=0.000001)
Lr_D = dict(init_lr_d=2e-4,final_lr_d=2e-6)
Lr_G = dict(init_lr_g = 1e-4, final_lr_g = 1e-6)


check = None
resume = None
