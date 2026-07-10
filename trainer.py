import argparse
import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 新增：用于计算softmax

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入数据集和工具类
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.dataset_ACDC import ACDCdataset
from utils.utils import powerset, DiceLoss, val_single_volume


# ====================== 新增：仅前景区域正交互斥损失 ======================
class ForegroundOrthogonalLoss(nn.Module):
    def __init__(self):
        super(ForegroundOrthogonalLoss, self).__init__()

    def forward(self, inputs):
        """
        inputs: [B, C, H, W] 网络输出的未经过激活的 Logits
        """
        # 1. 转化为概率图
        probs = torch.softmax(inputs, dim=1)
        B, C, H, W = probs.shape

        if C <= 1:
            return torch.tensor(0.0, device=inputs.device)

        # 2. 仅前景互斥：剔除背景类 (Class 0)，保留所有前景器官
        fg_probs = probs[:, 1:, :, :]  # [B, C-1, H, W]

        # 3. 展平空间维度以计算内积 [B, C-1, N]
        fg_probs = fg_probs.contiguous().view(B, C - 1, -1)

        # 4. 计算前景类之间的内积矩阵 P * P^T
        # 为了防止 H*W (像素总数) 过大导致数值爆炸，除以空间维度大小进行均值化
        inner_product = torch.bmm(fg_probs, fg_probs.transpose(1, 2)) / (H * W)

        # 5. 提取非对角线元素（不同器官类别间的概率重叠/黏连）
        mask = torch.ones(C - 1, C - 1, device=inputs.device) - torch.eye(C - 1, device=inputs.device)
        mask = mask.unsqueeze(0)  # [1, C-1, C-1]

        # 6. 计算非对角线元素的平均值，作为惩罚项
        loss = (inner_product * mask).mean()
        return loss


# ========================================================================


def inference(args, model, best_performance):
    """
    推理函数：同时支持 Synapse 和 ACDC 数据集
    在完整 3D 体积上滑窗推理，计算平均 Dice（已优化）
    """
    if args.dataset == "ACDC":
        db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    else:
        db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol",
                                  list_dir=args.list_dir, nclass=args.num_classes)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    metric_list = []  # ← 修复：使用 list 更安全

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            # val_single_volume 内部自动处理 3D 滑窗和不同数据集的尺寸差异
            metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                         patch_size=[args.img_size, args.img_size],
                                         case=case_name, z_spacing=args.z_spacing)
            metric_list.append(np.array(metric_i))

    # 更稳健的平均计算
    metric_array = np.mean(np.stack(metric_list), axis=0)

    # 原版口径：直接对所有类别求平均（包含背景）
    performance = metric_array.mean()

    logging.info('Testing performance: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance


def trainer_synapse(args, model, snapshot_path):
    """
    主训练函数（最终融合优化版）
    支持 Synapse + ACDC + 原版超强深监督 + Poly LR + 延迟验证
    """
    # ====================== 1. 日志与目录准备 ======================
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 清理旧 handler，避免日志重复
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(snapshot_path, "log.txt")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("==== Training Start ====")
    logging.info(str(args))

    # ====================== 2. 数据集加载 ======================
    if args.dataset == "ACDC":
        from utils.dataset_ACDC import RandomGenerator as ACDC_RG
        db_train = ACDCdataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([ACDC_RG(output_size=[args.img_size, args.img_size])])
        )
    else:
        db_train = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size = args.batch_size * args.n_gpu  # 支持多卡
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # ====================== 3. 模型 & 设备 & 损失 ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    # <--- 初始化前景正交互斥损失
    ortho_loss = ForegroundOrthogonalLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(
        "{} iterations per epoch. {} max iterations".format(len(trainloader), args.max_epochs * len(trainloader)))

    # ====================== 4. 训练主循环 ======================
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # 第一次迭代时确定监督策略子集（原版超强逻辑）
    ss = None
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].squeeze(1).cuda()  # [B,1,H,W] -> [B,H,W]

            # 前向传播（返回多个输出头的 list）
            outputs = model(image_batch)  # EMCADNet 等模型直接调用
            if not isinstance(outputs, list):
                outputs = [outputs]

            # 第一次 batch 时确定监督子集
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(outputs)
                out_idxs = list(range(n_outs))
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs) if x]  # 所有非空子集
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]  # 每个头单独监督
                else:
                    ss = [[-1]]
                print("✅ 当前监督策略子集:", ss)

            # ====================== 计算损失 ======================
            loss = 0.0

            # <--- 新增 w_ortho=0.3，基线 0.3 和 0.7 坚决不动
            w_ce, w_dice, w_ortho = 0.3, 0.7, 0.5

            for s in ss:
                if not s:
                    continue
                # mutation 的灵魂：把当前子集的所有输出头直接相加
                iout = sum(outputs[idx] for idx in s)

                loss_ce = ce_loss(iout, label_batch.long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss_ortho_val = ortho_loss(iout)  # <--- 计算前景互斥损失

                # <--- 线性叠加惩罚项
                loss += w_ce * loss_ce + w_dice * loss_dice + w_ortho * loss_ortho_val

                # ====================== 反向传播 ======================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Poly 学习率衰减
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f'
                             % (iter_num, epoch_num, loss.item(), lr_))

        # ====================== 每个 epoch 结束 ======================
        logging.info('epoch %d : loss : %f, lr: %f' % (epoch_num, loss.item(), lr_))

        # === 只改这里：100轮之后每5轮测试一次（最后一轮必测），其他完全不变 ===
        if epoch_num >= 140:
            logging.info("Starting validation for epoch %d..." % epoch_num)
            performance = inference(args, model, best_performance)

            if performance > best_performance:
                best_performance = performance
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'best.pth'))
                logging.info("✅ epoch {}: saved NEW best.pth (Dice: {:.4f})".format(epoch_num, performance))

            torch.cuda.empty_cache()  # 释放显存

        # 最后保存 final 模型
        if epoch_num == max_epoch - 1:
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'final_model.pth'))
            logging.info("Save final model to final_model.pth")

    writer.close()
    return "Training Finished!"