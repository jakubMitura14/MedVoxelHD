# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import warp as wp
import statistics


wp.init()
devicesWarp = wp.get_devices()
print_config()
import softWarpLoss

data_dir = '/home/data/spleenDat/Task09_Spleen'
root_dir='/home/data/spleenDat/Task09_Spleen'


# def mainPartWarpLoss(index,a,bList,radius):
#     # a=a[1,:,:,:]
#     # b=bList[index][0,:,:,:]
#     argss= mainWarpLoss.prepare_tensors_for_warp_loss(a[1,:,:,:], bList[index][0,:,:,:],radius,devicesWarp[1])
#     mainWarpLoss.getHausdorff.apply(*argss)
#     # print(argss[0])
#     # print(argss[1])
#     return torch.stack([torch.mean(argss[0]),torch.mean(argss[1])])

# def meanWarpLoss(aList, bList):
#     radius = 500.0#TODO increase
#     # summ=torch.zeros(1, requires_grad=True).to('cuda')
#     # lenSum=torch.zeros(1, requires_grad=True).to('cuda')
#     return torch.mean(torch.stack(list(map(
#         lambda tupl: mainPartWarpLoss(tupl[0],tupl[1]
#                                         ,decollate_batch(bList.bool()),radius) 
#                                     ,enumerate(decollate_batch(aList.bool()))))))
  
#     #print(f"waarp loss  {torch.nanmean(catRes)}")

#     #return torch.nanmean(catRes).to('cuda')

# def mainPartWarpLossSingleBatch(a,b):
#     radius=300.0
#     # a=a[1,:,:,:]
#     # b=bList[index][0,:,:,:]
#     argss= mainWarpLoss.prepare_tensors_for_warp_loss(a[0,1,:,:,:].bool(), b[0,0,:,:,:].bool(),radius,devicesWarp[1])
#     mainWarpLoss.getHausdorff.apply(*argss)
#     # print(argss[0])
#     # print(argss[1])
#     return torch.mean(torch.stack([torch.mean(argss[0]),torch.mean(argss[1])]))





def mainPartWarpLossSingleBatch(b,a):
    radius = 500.0#TODO increase
    #b= b.float()
    a=a[0,0,:,:,:].bool()
    b=b[0,1,:,:,:]
    #print(devicesWarp)

    # argss= prepare_tensors_for_warp_loss(a, b,radius,devicesWarp[1])
    # ress=

    points_in_grid, points_labelArr,  y_hat, counts_arr ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false= softWarpLoss.prepare_tensors_for_warp_loss(a, b,radius, wp.get_devices()[1])
    softWarpLoss.getHausdorff_soft.apply(points_in_grid, points_labelArr,  y_hat, counts_arr ,radius,device,dim_x,dim_y,dim_z,num_points_gold, num_points_gold_false)
    res= torch.sub(torch.nanmean(counts_arr)
                ,torch.div(torch.nansum(y_hat[a.bool()])  , (num_points_gold/((dim_x+dim_y+dim_z)/20) ) ))       
    # argss= warpLoss.softWarpLoss.prepare_tensors_for_warp_loss(a, b,radius,devicesWarp[1])
    # warpLoss.softWarpLoss.getHausdorff_soft.apply(*argss)
    print(f"waarp loss  {res}")
    return res


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image', 'label'],
                #                     mode=('bilinear', 'nearest'),
                #                     prob=1.0,
                #                     spatial_size=(96, 96, 96),
                #                     rotate_range=(0, 0, np.pi/15),
                #                     scale_range=(0.1, 0.1, 0.1)),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=1.0, num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files, transform=val_transforms,
            cache_rate=1.0, num_workers=4,
        )
#         self.train_ds = monai.data.Dataset(
#             data=train_files, transform=train_transforms)
#         self.val_ds = monai.data.Dataset(
#             data=val_files, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=1, shuffle=True,
            num_workers=4, collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        #print("in training")
        # print(output.shape)
        # print(labels.shape)
        loss = mainPartWarpLossSingleBatch(output,labels)
        #print(loss.item())
        #loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 1
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward)
        #print("in validation")
        # print(outputs.shape)
        # print(labels.shape)
        loss = mainPartWarpLossSingleBatch(outputs,labels)
        #print(loss)
        # loss = meanWarpLoss(decollate_batch(outputs.bool()),decollate_batch(labels.bool()))

        #print(decollate_batch(outputs)[0][1,:,:,:].shape)

        #loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].nansum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return {"log": tensorboard_logs}

# initialise the LightningModule
net = Net()

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
    save_dir=log_dir
)

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=600,#600
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=16,
)

# train
trainer.fit(net)
print(
    f"train completed, best_metric: {net.best_val_dice:.4f} "
    f"at epoch {net.best_val_epoch}")