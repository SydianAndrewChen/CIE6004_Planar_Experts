import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import copy
from mnh.dataset_replica import ReplicaDataset
from mnh.dataset_TanksAndTemples import TanksAndTemplesDataset
from mnh.model_mnh import *
from mnh.stats import StatsLogger, WandbLogger
from mnh.utils import *
from mnh.utils_model import freeze_model
import teacher_forward
from experts_forward import *

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs/experts')
TEST_CONFIG = 'test'
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints/experts')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


@hydra.main(config_path=CONFIG_DIR, config_name=TEST_CONFIG)
def main(cfg: DictConfig):
    # Set random seed for reproduction
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set device for training
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cfg.cuda))
    else:
        device = torch.device('cpu')

    # set DataLoader objects
    train_path = os.path.join(CURRENT_DIR, cfg.data.path, 'train')
    valid_path = os.path.join(CURRENT_DIR, cfg.data.path, 'valid')
    test_path  = os.path.join(CURRENT_DIR, cfg.data.path, 'test')
    train_dataset, valid_dataset = None, None
    if 'replica' in cfg.data.path:
        train_dataset = ReplicaDataset(folder=train_path, read_points=True, sample_points=cfg.data.batch_points)
        valid_dataset = ReplicaDataset(folder=valid_path)
    elif 'Tanks' in cfg.data.path or 'BlendedMVS' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=valid_path,
        )
    elif 'Synthetic' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=test_path,
        )
    
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=False)
    valid_loader = DataLoader(valid_dataset, collate_fn=lambda x: x, shuffle=False)

    teacher = teacher_forward.get_model_from_config(cfg)
    teacher.to(device)
    model = get_model_from_config(cfg)
    model.to(device)

    # load checkpoints
    run_id = None
    stats_logger = None
    optimizer_state = None   
    start_epoch = 0 

    checkpoint_path = os.path.join(CURRENT_DIR, 'checkpoints/teacher', cfg.checkpoint.teacher)
    pretrained_teacher = os.path.isfile(checkpoint_path)
    if pretrained_teacher: 
        print('Load teacher from checkpoint: {}'.format(checkpoint_path))
        loaded_data = torch.load(checkpoint_path, map_location=device)
        teacher.load_state_dict(loaded_data['model'])
    else:
        print('WARNING: no pretrained weight for teacher network')
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.distill)
    if cfg.train.resume and os.path.isfile(checkpoint_path):
        print('Resume from checkpoint: {}'.format(checkpoint_path))
        loaded_data = torch.load(checkpoint_path, map_location=device)
        run_id = loaded_data['run_id']
        model.load_state_dict(loaded_data['model'])
        stats_logger = pickle.loads(loaded_data['stats'])
        start_epoch = stats_logger.epoch
        optimizer_state = loaded_data['optimizer']
    else:
        if pretrained_teacher:
            print('[Init] Copy plane geometry from teacher ...')
            model.plane_geo = copy.deepcopy(teacher.plane_geo)
        else:
            print('[Init] Initialize plane geometry')
            points = train_dataset.dense_points.to(device)
            model.plane_geo.initialize(
                points, 
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
            )

            # model.plane_geo.random_init(points, wh=cfg.model.init.wh)
            
            # model.plane_geo.initialize_with_box(
            #     points, 
            #     lrf_neighbors=cfg.model.init.lrf_neighbors,
            #     wh=cfg.model.init.wh,
            #     box_factor=cfg.model.init.box_factor, 
            #     random_rate=cfg.model.init.random_rate,
            # )
            del points

    if cfg.train.freeze_geometry:
        print('Freeze plane geometry')
        freeze_model(model.plane_geo)
    
    # set optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr
    )
    if optimizer_state != None:
        optimizer.load_state_dict(optimizer_state)
        optimizer.last_epoch = start_epoch
    
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # set StatsLogger, WandbLogger objects
    if stats_logger == None:
        stats_logger = StatsLogger()

    config = OmegaConf.to_container(cfg)
    config['model_param'] = parameter_number(model)
    wandb_logger = WandbLogger(
        run_name=cfg.name,
        notes=cfg.notes,
        config=config,
        resume_id=run_id,
        project='2021-mnh-distill'
    )
    
    img_folder = os.path.join(CURRENT_DIR, 'output_images/experts', cfg.name, 'output')
    os.makedirs(img_folder, exist_ok=True)
    print('[Traing: Geometry + Transparency + Texture] Start ...')

    for epoch in range(start_epoch, cfg.train.epoch):
        model.train()
        stats_logger.new_epoch()

        for i, data in enumerate(train_loader):
            data = data[0]
            if epoch < cfg.train.epoch_teach:
                train_stats = learn_from_teacher(
                    data, 
                    model, 
                    teacher,
                    device, 
                    cfg, 
                    optimizer
                )
            else:
                train_stats, _ = forward_pass(
                    data, 
                    model,
                    device,
                    cfg, 
                    optimizer,
                    training=True,
                )
            stats_logger.update('train', train_stats)
        
        stats_logger.print_info('train')
        train_info_epoch = stats_logger.get_info('train')
        wandb_logger.upload(
            step=(epoch+1), 
            info=train_info_epoch
        )
        lr_scheduler.step()

        # Checkpoint
        if (epoch+1) % cfg.train.checkpoint_epoch == 0:
            print('store checkpoints ...')
            checkpoint = {
                'run_id': wandb_logger.get_run_id(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stats': pickle.dumps(stats_logger)
            }
            torch.save(checkpoint, checkpoint_path)

        # validation
        if (epoch+1) % cfg.train.validation_epoch == 0:
            model.eval()
            for i, data in enumerate(valid_loader):
                data = data[0]
                valid_stats, valid_images = forward_pass(
                    data, 
                    model,
                    device,
                    cfg,
                    training=False,
                )
                stats_logger.update('valid', valid_stats)

                for key, img in valid_images.items():
                    img = tensor2Image(img)
                    path = os.path.join(img_folder, 'valid-{:0>5}-{}.png'.format(i, key))
                    img.save(path)

            stats_logger.print_info('valid')
            valid_info = stats_logger.get_info('valid')
            wandb_logger.upload(
                step=(epoch+1), 
                info=valid_info
            )

if __name__ == '__main__':
    main()