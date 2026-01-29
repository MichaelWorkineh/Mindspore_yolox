import os
import datetime
import argparse

from mindspore import nn, Model
from mindspore import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from yolox.models.losses import YOLOLossCell
from yolox.models.yolox import YOLOX
from yolox.exp.yolo_base import YOLOXCB
from yolox.exp.build import get_exp
from yolox.config import config
from yolox.logger import get_logger
from datasets.yolox_dataset import create_yolox_dataset


def get_parser():
    parser = argparse.ArgumentParser(description='Yolox train.')
    parser.add_argument('--name', type=str, default="yolox-tiny", help='model name, yolox-s, yolox-m, yolox-l, yolox-x')
    parser.add_argument('--data_dir', type=str, default="/home/psy/workplace/datasets/coco2017",
                        help='Location of data.')
    parser.add_argument('--log_dir', type=str, default='./logs/train', help='Location of logs.')
    parser.add_argument('--save_dir', type=str, default='./save/yolox', help='Location of ckpt.')
    parser.add_argument('--backbone', type=str, default="yolopafpn", help='yolofpn or yolopafpn')
    parser.add_argument('--device_target', type=str, default="GPU", help='Ascend or GPU')
    parser.add_argument('--rank', type=int, default=0, help='logger related, rank id')
    parser.add_argument('--log_interval', type=int, default=30, help='logging related, log_interval')
    parser.add_argument('--per_batch_size', type=int, default=4, help='dataset related, batch_size')
    parser.add_argument('--group_size', type=int, default=1, help='dataset related, device_num')
    parser.add_argument('--data_aug', type=bool, default=True, help='dataset related, mosaic or not')
    parser.add_argument('--resume', type=bool, default=False, help='training related, resume or not')
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/yolox_map47.3.ckpt',
                        help='training related, pretrained model path')
    parser.add_argument('--lr', type=int, default=0.0001, help='hyperparameters, learning_rate')
    parser.add_argument('--keep_ckpt_max', type=int, default=20, help='the max num of saving models')
    parser.add_argument('--max_epoch', type=int, default=10, help='the epoch of training models')
    return parser


def run_train():
    args = get_parser().parse_args()
    exp = get_exp(args.name)

    # coco数据集
    data_root = os.path.join(args.data_dir, 'val2017')
    annFile = os.path.join(args.data_dir, 'annotations/instances_val2017.json')

    # 设置logger
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    logger = get_logger(log_dir, args.rank)
    logger.save_args(config)

    # 静态图 GRAPH_MODE, 动态图 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, save_graphs=False)

    # 定义dataset
    ds = create_yolox_dataset(image_dir=data_root, anno_path=annFile, batch_size=args.per_batch_size,
                              device_num=args.group_size, rank=args.rank, data_aug=args.data_aug)
    steps_per_epoch = ds.get_dataset_size()
    # 定义网络
    base_network = YOLOX(config, backbone=args.backbone, exp=exp)
    network = YOLOLossCell(base_network, config)

    # 加载预训练模型
    if args.resume:
        pre_ckpt = args.pretrain_model
        param_dict = load_checkpoint(pre_ckpt)
        load_param_into_net(network, param_dict)
        logger.info('load pretrained model!')
    else:
        logger.info('training from scratch!')

    # 定义优化器
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=args.lr, momentum=0.9)
    # 定义loss
    # 定义loss
    if args.device_target == "CPU":
        # CPU does not support/need DynamicLossScaleManager with NPU/GPU ops
        train_network = nn.TrainOneStepCell(network, optimizer)
    else:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 22)
        update_cell = loss_scale_manager.get_update_cell()
        # 封装训练网络
        train_network = nn.TrainOneStepWithLossScaleCell(network, optimizer, scale_sense=update_cell)
    model = Model(train_network, amp_level="O0")
    # 设置网络为训练模式
    train_network.set_train(True)
    # 设置保存模型的配置信息
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=args.keep_ckpt_max)
    # 定义callback
    cb = [YOLOXCB(logger, steps_per_epoch, lr=args.lr, per_print_times=args.log_interval),
          ModelCheckpoint(prefix=args.name, directory=args.save_dir, config=config_ck)]
    model.train(args.max_epoch, ds, callbacks=cb, dataset_sink_mode=True, sink_size=-1)


if __name__ == '__main__':
    run_train()
