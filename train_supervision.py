import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch,gc
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.trainloss = config.trainloss
        self.valloss = config.valloss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x,imagename=None):
        # only net is used in the prediction/inference
        seg_pre = self.net(x,imagename)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.trainloss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # supervision stage
        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value,"\n")

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value,"\n")
        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.valloss(prediction, mask)
        return {"loss_val": loss_val}

    def validation_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        f1_per_class = self.metrics_val.F1()
        # eval_value = []
        # for class_name, class_iou, class_f1 in zip(self.config.classes, iou_per_class, f1_per_class):
        #     # print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        #     eval_value.append('F1_{}:{}'.format(class_name,class_f1))
        #     eval_value.append('IOU_{}:{}'.format(class_name,class_iou))
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        # eval_value.append(eval_value1)
        print('val:', eval_value)
        iou_value = {}
        f1_value = {}
        for class_name, iou, f1 in zip(self.config.classes, iou_per_class, f1_per_class):
            iou_value["IOU_" + class_name] = iou
            f1_value["F1_" + class_name] = f1

        print(iou_value)

        self.metrics_val.reset()
        loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        log_dict.update(f1_value)
        log_dict.update(iou_value)
        self.log_dict(log_dict, prog_bar=True)
        gc.collect()
        torch.cuda.empty_cache()
    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    
    trainer = pl.Trainer(devices=config.gpus,max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger,progress_bar_refresh_rate=2)
    trainer.fit(model=model)


if __name__ == "__main__":
   main()
