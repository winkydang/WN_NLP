# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the training details of the emotion classifier.
@All Right Reserve
'''

from optim import Adam, NoamOpt
import torch
import os
import torch.nn as nn
import torch.distributed
# import torch.tensor
from dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ClsTrainer:
    def __init__(self, args, model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None, distributed=False):
        # Initialize attributes
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.rank = torch.distributed.get_rank() if distributed else -1
        # Init the writer for tensorboard  # 初始化训练集和验证集的日志记录器
        # 使用了TensorBoardX库中的SummaryWriter类，该类用于记录训练过程中的各种指标和信息，以便在TensorBoard中进行可视化。
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train_cls'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid_cls'))
        else:
            self.valid_writer = valid_writer
        # Move the model to the right device
        self.model = model.to(device, non_blocking=True)  # 将模型移动到指定的设备上，通常是GPU。non_blocking=True 参数表示如果可能的话，返回控制权给主机，而传输在后台进行。
        # Init criterion 
        self.criterion = nn.CrossEntropyLoss().to(device)

        # Init optimizer. Note that we copied the implementation of Adaom from the original torch lib.
        base_optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        # Set up optimizer for learning rate scheduling with warmup
        if hasattr(self.model, 'config'):
            self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)
        else:
            self.optimizer = NoamOpt(self.model.module.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        # Init the sampler
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        # Init dataloader
        self.train_dataloader = DataLoader(
            train_dataset, sampler=self.train_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.valid_dataloader = DataLoader(
            valid_dataset, sampler=self.valid_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _eval_train(self, epoch):  # 在给定的训练周期（epoch）内评估（通常实际上是执行训练）模型的性能
        self.model.train()  # Set the model to training mode 将模型设置为训练模式，这意味着启动BatchNormalization和dropout，这些在训练时需要，在评估模式时不需要。

        loss, acc, step_count = 0, 0, 0  # 初始化累计的损失、准确率、步数，在训练过程中进行更新
        total = len(self.train_dataloader)  # 计算训练数据加载器（train_dataloader）将迭代的次数，通常这等于训练数据集的总批次数。
        if self.rank in [-1, 0]:    # Set up the progress bar (only for non-distributed training or the master process in distributed training)
            TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                        dynamic_ncols=True, total=total)
        else:
            TQDM = enumerate(self.train_dataloader)

        for i, data in TQDM:
            ################################################
            # 4. TODO complete the code to train the model #
            ################################################
            # pass
            # move the input and target data to the correct device (e.g. GPU)
            inputs, labels = data['utt'].to(self.device), data['label'].to(self.device)
            outputs = self.model(inputs)  # 前向传播，计算模型输出  # return loss, logits
            loss_value = self.criterion(outputs[1], labels)  # 计算损失

            # zero the parameter gradients  清空梯度
            self.optimizer.zero_grad()
            loss_value.backward()  # 反向传播：计算梯度
            self.optimizer.step()  # 更新权重

            # 计算统计信息，如累计损失和准确率
            loss += loss_value.item()
            _, preds = torch.max(outputs[1], 1)  # outputs: tuple(loss, logits)
            acc += torch.sum(preds == labels.data).item()
            step_count += 1

            # 可选：在进度条中更新显示的统计信息
            TQDM.set_postfix(loss=loss / step_count, accuracy=acc / (step_count * inputs.size(0)))

            # # 如果需要，打印统计信息
            # if i % 500 == 0:  # 每500个小批量打印一次
            #     print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, loss / 500))
            #     loss = 0  # 在打印了周期内的平均损失后被重置为0，为下一个打印周期做准备.通常对每个打印周期内的平均损失更感兴趣，而不是自训练开始以来的累计损失.
            #     # 这种重置操作是为了在下一组迭代中能够正确地计算新的平均损失值，而不是让之前的损失值影响新的计算

        # 在 epoch 结束时，计算平均损失和准确率
        avg_loss = loss / total  # total = len(self.train_dataloader)
        avg_acc = acc / (total * self.train_dataloader.batch_size)  # 通过累加预测正确的数量，并在每个epoch结束时计算平均准确率
        print(f'Epoch {epoch} - avg_loss: {avg_loss}, avg_acc: {avg_acc}')

    def _eval_test(self, epoch, step):
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
            ###################################################
            # 5. TODO complete the code to evaluate the model #
            ###################################################
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in self.valid_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)  # 前向传播，计算模型输出

                loss = self.criterion(outputs, labels)  # 计算损失
                total_loss += loss.item()  # 累计损失

                # 在 torch.max 的情况下，它实际上返回两个值：最大值和最大值的索引。第一个值（即最大值本身）在这里不重要，我们只关心每个样本最大值的索引，因为这个索引代表了模型认为最可能的类别。
                _, predicted = torch.max(outputs.data, 1)  # 获得预测结果  # torch.max(…, 1)：这个函数返回给定维度上的最大值。(batch_size, num_classes)。也就是num_classes维度。
                total_correct += (predicted == labels).sum().item()  # 累加正确的数量
                total_samples += labels.size(0)  # 累加样本数量  # babels.size(0) = self.valid_dataloader.batch_size 吗？？

            avg_loss = total_loss / len(self.valid_dataloader)  # 计算平均损失
            accuracy = total_correct / total_samples  # 计算准确率  # total_samples = len(self.valid_dataloader) * self.valid_dataloader.batch_size

        self.model.train()  # 将模型设置为训练模式

        # 打印或记录性能指标
        print(f"Epoch: {epoch}, Step: {step}, Avg loss: {avg_loss}, Accuracy: {accuracy}")
        # 可能还想将这些指标记录到日志文件或tensorboard等

    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        # Train the model
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch'.format(epoch))
            # Reshuffle the train data in each epoch
            if hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            # Call the training function
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                # Call the after epoch function
                func(epoch, self.device)
        # self.model.train()  # 在函数的末尾，将模型的模式重新设置为训练模式。这是为了防止_eval_train函数将模型设置为评估模式，因为在评估模式下某些模型层（如Dropout和BatchNorm）的行为会与训练时不同。
