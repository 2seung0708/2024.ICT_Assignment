import math
import sys
import time
import torch

from libs.functions import Evaluator
from . import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, obj, logger=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Obj: {} Epoch: [{}]'.format(obj, epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images1, images2, targets1, targets2 in metric_logger.log_every(data_loader, print_freq, header, logger):

        images = list(image.to(device) for image in images1)
        if images2[0] is not None: ########################### temp ################################################
            images.extend(list(image.to(device) for image in images2))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets1]
        if targets2[0] is not None: ########################## temp #################################################
            targets.extend([{k: v.to(device) for k, v in t.items()} for t in targets2])

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def kp_vis(kps, image, targets, dir_p=None):
    from PIL import ImageDraw
    import os
    from torchvision.transforms.functional import to_pil_image
    os.makedirs(f'{dir_p}/kpt_vis1/',exist_ok=True)
    os.makedirs(f'{dir_p}/kpt_vis2/',exist_ok=True)
    os.makedirs(f'{dir_p}/kpt_vis3/',exist_ok=True)
    img = to_pil_image(image[0])
    
    kp1 = kps[0]['keypoints1'][0]
    kp2 = kps[0]['keypoints2'][0]
    kp3 = kps[0]['keypoints3'][0]
    gt_kpt = targets[0]['keypoints'][0]
    image_id = targets[0]['image_id'].item()
    
    draw = ImageDraw.Draw(img)
    for i in range(len(kp1)):
        draw.point((gt_kpt[i][:2]).tolist(), fill=(0, 255, 0))
        draw.point(kp1[i][:2].tolist(), fill=(255, 0, 0))
    img.save('{}/kpt_vis1/kpt_result_{}.png'.format(dir_p,str(image_id).zfill(6)))
    del img
    
    img = to_pil_image(image[0])
    draw = ImageDraw.Draw(img)
    for i in range(len(kp2)):
        draw.point(((gt_kpt[i][:2])).tolist(), fill=(0, 255, 0))
        draw.point(kp2[i][:2].tolist(), fill=(255, 0, 0))
    img.save('{}/kpt_vis2/kpt_result_{}.png'.format(dir_p,str(image_id).zfill(6)))
    del img
    
    img = to_pil_image(image[0])
    draw = ImageDraw.Draw(img)
    for i in range(len(kp3)):
        draw.point(((gt_kpt[i][:2])).tolist(), fill=(0, 255, 0))
        draw.point(kp3[i][:2].tolist(), fill=(255, 0, 0))
    img.save('{}/kpt_vis3/kpt_result_{}.png'.format(dir_p,str(image_id).zfill(6)))

@torch.no_grad()
def evaluate(model, data_loader, device, logger=None,cfg=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    evaluator = Evaluator()
    header = 'Test:'
    
    count_n=0

    for images, targets in metric_logger.log_every(data_loader, 100, header, logger):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        
        # import pdb;pdb.set_trace()
        # kp_vis(outputs,images, targets, dir_p="{}/{}/{}".format(cfg.OUTPUT_DIR,cfg.obj,cfg.log_name))

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        # if count_n >10:
        #     break
        # else:
        #     count_n+=1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluator.gather_all()
    return evaluator

