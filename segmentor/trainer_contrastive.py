from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from lib.utils.tools.average_meter import AverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from lib.vis.embedding_visualizer import EmbeddingVisualizer
from lib.vis.labelmap_visualizer import LabelmapVisualizer

import matplotlib.pyplot as plt
import numpy as np


class Trainer(object):
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)
        if configer.exists('embedding_visualizer'):
            self.embedding_visualizer = EmbeddingVisualizer(configer)
        if configer.exists('labelmap_visualizer'):
            self.labelmap_visualizer = LabelmapVisualizer(configer)

        # Tensorboard:
        self.summary_writer = SummaryWriter(log_dir=configer.get("tensorboard", "log_dir"))

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

        # from here on different from Trainer
        self.with_contrast = True if self.configer.exists("contrast") else False
        if self.configer.exists("contrast", "warmup_iters"):
            self.contrast_warmup_iters = self.configer.get("contrast", "warmup_iters")
        else:
            self.contrast_warmup_iters = 0

        self.with_memory = self.configer.exists('contrast', 'with_memory')
        if self.with_memory:
            self.memory_size = self.configer.get('contrast', 'memory_size')
            self.pixel_update_freq = self.configer.get('contrast', 'pixel_update_freq')

        self.network_stride = self.configer.get('network', 'stride')

        Log.info("with_contrast: {}, warmup_iters: {}, with_memory: {}".format(
            self.with_contrast, self.contrast_warmup_iters, self.with_memory))

        self.last_labelmap_vis_iter = 0
        self.last_embedding_vis_iter = 0

        self.configer.add(('with_embed',), True)
        self.configer.add(('is_eval',), False)

        # self.experiment = keepsake.init(
        #     path='keepsake',
        #     params={"[HP] learning_rate": self.configer.get('lr', 'base_lr'),
        #             "[HP] train_bs": self.configer.get('train', 'batch_size'),
        #             "[NET] loss": self.configer.get('loss', 'loss_type'),
        #             "[NET] backbone": self.configer.get('network', 'backbone'),
        #             "[NET] model_name": self.configer.get('network', 'model_name'),
        #             "[CONTRAST] proj_dim": self.configer.get('contrast', 'proj_dim'),
        #             "[CONTRAST] temperature": self.configer.get('contrast', 'temperature'),
        #             "[CONTRAST] max_samples": self.configer.get('contrast', 'max_samples'),
        #             "[CONTRAST] warmup_iters": self.configer.get('contrast', 'warmup_iters'),
        #             "[CONTRAST] loss_weight": self.configer.get('contrast', 'loss_weight')}
        # )

    def _dequeue_and_enqueue(self, keys, labels,
                             segment_queue, segment_queue_ptr,
                             pixel_queue, pixel_queue_ptr):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = labels[:, ::self.network_stride, ::self.network_stride]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x > 0]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                # segment feature vector is average embedding of all pixels of one class in one image
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)     # region feature vector
                ptr = int(segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % self.memory_size

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            assert torch.is_tensor(data_dict['img'])
            assert torch.is_tensor(data_dict['labelmap'])

            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'),
                    self.scheduler, self.optimizer, backbone_list=[0, ]
                )

            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            assert torch.is_tensor(inputs[0])
            assert torch.is_tensor(targets)

            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()

            # different behaviour from Trainer if with_contrast == True
            with_embed = True if self.configer.get('iters') >= self.contrast_warmup_iters else False

            self.configer.update(('with_embed',), with_embed)
            self.configer.update(('is_eval',), False)

            if self.with_contrast is True:
                if self.with_memory is True:
                    outputs = self.seg_net(*inputs, targets)

                    outputs['pixel_queue'] = self.seg_net.module.pixel_queue
                    outputs['pixel_queue_ptr'] = self.seg_net.module.pixel_queue_ptr
                    outputs['segment_queue'] = self.seg_net.module.segment_queue
                    outputs['segment_queue_ptr'] = self.seg_net.module.segment_queue_ptr
                else:
                    outputs = self.seg_net(*inputs)
            else:
                outputs = self.seg_net(*inputs)

            self.foward_time.update(time.time() - foward_start_time)
            loss_start_time = time.time()
            if is_distributed():
                import torch.distributed as dist
                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that 
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                loss = self.pixel_loss(outputs, targets, with_embed=with_embed)
                backward_loss = loss
                display_loss = reduce_tensor(backward_loss) / get_world_size()
            else:
                backward_loss = display_loss = self.pixel_loss(outputs, targets, with_embed=with_embed)

            # Sanity Check:
            if torch.isnan(display_loss):
                Log.info("Loss is NaN. Aborting training...")
                exit()

            # memory bank update
            if self.with_memory and 'key' in outputs and 'lb_key' in outputs:
                self._dequeue_and_enqueue(outputs['key'], outputs['lb_key'],
                                          segment_queue=self.seg_net.module.segment_queue,
                                          segment_queue_ptr=self.seg_net.module.segment_queue_ptr,
                                          pixel_queue=self.seg_net.module.pixel_queue,
                                          pixel_queue_ptr=self.seg_net.module.pixel_queue_ptr)

            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            # Tensorboard:
            n_iter = self.configer.get('iters')
            self.summary_writer.add_scalar("Loss/Total", display_loss, n_iter)
            self.summary_writer.add_scalar("Loss/Seg", self.pixel_loss.loss, n_iter)
            self.summary_writer.add_scalar("Loss/Contrast", self.pixel_loss.loss_contrast, n_iter)
            self.summary_writer.add_scalar("Learning Rate", self.module_runner.get_lr(self.optimizer)[0], n_iter)

            backward_start_time = time.time()
            self.optimizer.zero_grad()
            backward_loss.backward()

            self.optimizer.step()
            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                    (not is_distributed() or get_rank() == 0):
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),
                    self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                    foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                    data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
                    self.configer.get('iters') > normal_max_iters and \
                    ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or \
                     self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
                self.optimizer.update_swa()

            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

            del display_loss
            del outputs

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()

        data_loader = self.val_loader if data_loader is None else data_loader

        visualize_labels = False
        if self.configer.exists('labelmap_visualizer') and self.configer.get('iters') - \
                self.last_labelmap_vis_iter >= self.configer.get('labelmap_visualizer', 'save_interval'):
            visualize_labels = True
            self.last_labelmap_vis_iter = self.configer.get('iters')

        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                Log.info('{} images processed\n'.format(j))

            if self.configer.get('dataset') == 'lip':
                (inputs, targets, inputs_rev, targets_rev), batch_size = self.data_helper.prepare_data(data_dict,
                                                                                                       want_reverse=True)
            else:
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            assert torch.is_tensor(inputs[0]), f"Inputs: {type(inputs[0])}\n{inputs[0]}"
            assert torch.is_tensor(targets), f"Targets: {type(targets)}\n{targets}"

            with torch.no_grad():
                if self.configer.get('dataset') == 'lip':
                    inputs = torch.cat([inputs[0], inputs_rev[0]], dim=0)

                    outputs = self.seg_net(inputs)

                    outputs_ = self.module_runner.gather(outputs)
                    if isinstance(outputs_, (list, tuple)):
                        outputs_ = outputs_[-1]
                    outputs = outputs_[0:int(outputs_.size(0) / 2), :, :, :].clone()
                    outputs_rev = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), :, :, :].clone()
                    if outputs_rev.shape[1] == 20:
                        outputs_rev[:, 14, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 15, :, :]
                        outputs_rev[:, 15, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 14, :, :]
                        outputs_rev[:, 16, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 17, :, :]
                        outputs_rev[:, 17, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 16, :, :]
                        outputs_rev[:, 18, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 19, :, :]
                        outputs_rev[:, 19, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 18, :, :]
                    outputs_rev = torch.flip(outputs_rev, [3])
                    outputs = (outputs + outputs_rev) / 2.
                    self.evaluator.update_score(outputs, data_dict['meta'])

                elif self.data_helper.conditions.diverse_size:
                    if is_distributed():
                        outputs = [self.seg_net(inputs[i]) for i in range(len(inputs))]
                    else:
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)

                    for i in range(len(outputs)):
                        loss = self.pixel_loss(outputs[i], targets[i].unsqueeze(0))
                        self.val_losses.update(loss.item(), 1)
                        outputs_i = outputs[i]['seg']
                        if isinstance(outputs_i, torch.Tensor):
                            outputs_i = [outputs_i]
                        self.evaluator.update_score(outputs_i, data_dict['meta'][i:i + 1])

                else:
                    self.configer.update(('is_eval',), True)
                    self.configer.update(('with_embed',), False)
                    outputs = self.seg_net(*inputs)

                    # Labelmap Visualizer:
                    if visualize_labels:
                        self.labelmap_visualizer.save_label(data_dict, outputs['seg'])

                    try:
                        loss = self.pixel_loss(outputs, targets)
                    except AssertionError as e:
                        print(len(outputs), len(targets))

                    if not is_distributed():
                        outputs = self.module_runner.gather(outputs)
                    self.val_losses.update(loss.item(), batch_size)

                    if isinstance(outputs, dict):
                        self.evaluator.update_score(outputs['seg'], data_dict['meta'])
                    else:
                        self.evaluator.update_score(outputs, data_dict['meta'])

            if j % 10 == 0 and self.configer.exists("embedding_visualizer", "encodings"):
                self.embedding_visualizer.visualize_embeddings(outputs["encode"], targets, name="encode")

            # Embedding Visualizer:
            if j == 0 and self.configer.exists('embedding_visualizer'):
                if self.configer.get('iters') - self.last_embedding_vis_iter >= \
                        self.configer.get('embedding_visualizer', 'save_interval'):
                    (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
                    self.embedding_visualizer.visualize_embeddings(outputs['embed'], targets)
                    self.last_embedding_vis_iter = self.configer.get('iters')

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()

        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance', experiment=None)
        self.module_runner.save_net(self.seg_net, save_mode='val_loss', experiment=None)
        cudnn.benchmark = True

        # Print the log info & reset the states.
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.evaluator.print_scores()

        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        self.seg_net.train()
        self.pixel_loss.train()

    def save_features(self):
        self.seg_net.eval()
        self.pixel_loss.eval()

        name = self.configer.get('save_features', 'name')
        Log.info(f'Saving features for {name}')

        data_loader = self.val_loader
        if self.configer.exists('save_features', 'set'):
            if self.configer.get('save_features', 'set') == 'train':
                data_loader = self.train_loader

        for j, data_dict in enumerate(data_loader):
            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            if name in data_dict['name']:
                with torch.no_grad():
                    self.configer.update(('is_eval',), True)
                    self.configer.update(('with_embed',), False)

                    outputs = self.seg_net(*inputs)
                    idx = data_dict['name'].index(name)

                    encode = outputs['encode'][idx].cpu().numpy()
                    seg = outputs['seg'][idx].cpu().numpy()
                    embed = outputs['embed'][idx].cpu().numpy()
                    label = targets[idx].cpu().numpy()

                save_path = self.configer.get('save_features', 'path')
                Log.info(f'Saving features at {save_path}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(save_path + '/' + name + '_encode.npy', encode)
                np.save(save_path + '/' + name + '_seg.npy', seg)
                np.save(save_path + '/' + name + '_embed.npy', embed)
                np.save(save_path + '/' + name + '_label.npy', label)

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
                return
            # return

        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            return

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

    def summary(self):
        from lib.utils.tools.summary import get_model_summary
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
