#!/usr/bin/env python3
# coding: utf8

"""
Model of the road segmentatjion neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from comet_ml import Experiment
import os
import sys

import time
import random
from io import StringIO

import numpy as np
import torch
import source.helpers.metricslogging as metricslogging
# import torchmetrics as torchmetrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from source.configuration import Configuration
from source.helpers.img_utils import trans_coords_from_patch_to_org_3d
from source.helpers.metrics import AverageMeter, AverageMeterDict

from source.logcreator.logcreator import Logcreator
from source.lossfunctions.loss_helpers import get_result_func
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory

from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.scheduler.lr_schedulerfactory import LRSchedulerFactory
from source.data.datasetfactory import DataSetFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:

    def __init__(self):
        # fix random seeds
        seed = 49626446
        self.fix_random_seeds(seed)

        self.model = ModelFactory.build().to(DEVICE)
        self.optimizer = OptimizerFactory.build(self.model)
        self.lr_scheduler = LRSchedulerFactory.build(self.optimizer)
        self.loss_function = LossFunctionFactory.build(self.model).to(DEVICE)
        # self.scaler = torch.cuda.amp.GradScaler()  # I assumed we always use gradscaler, thus no factory for this

        self.patch_size = Configuration.get("data_collection.image_size")
        self.result_func = get_result_func()

        # initialize tensorboard logger
        Configuration.tensorboard_folder = os.path.join(Configuration.output_directory, "tensorboard")
        if not os.path.exists(Configuration.tensorboard_folder):
            os.makedirs(Configuration.tensorboard_folder)
        self.writer = SummaryWriter(log_dir=Configuration.tensorboard_folder)

        # Init comet
        self.comet = metricslogging.init_comet()

        # Print model summary
        self.print_modelsummary()

        # Log used device
        Logcreator.info("Following device will be used for training: " + DEVICE,
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    def fix_random_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epoch_nr=0):
        train_dataset = DataSetFactory.load(Configuration.get('data_collection'),
                                            is_train=True)

        valid_dataset = DataSetFactory.load(Configuration.get('data_collection'),
                                            is_train=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Configuration.get("training.general.batch_size"),  # * len(gpus)
            shuffle=Configuration.get("training.general.shuffle_data"),
            num_workers=Configuration.get("training.general.num_workers"),
            pin_memory=True
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=Configuration.get("training.general.batch_size"),  # * len(gpus)
            shuffle=False,
            num_workers=Configuration.get("training.general.num_workers"),
            pin_memory=True
        )

        # Load training parameters from config file
        train_parms = Configuration.get('training.general')

        epoch = 0
        if epoch_nr != 0:  # Check if continued training
            epoch = epoch_nr + 1  # plus one to continue with the next epoch

        best_perf = 0.0
        while epoch < train_parms.num_epochs:
            Logcreator.info(f"Epoch {epoch}, lr: {self.get_lr():.3e}, lr-step: {self.lr_scheduler.last_epoch}")

            train_loss, preds_in_patch_with_score = self.train_step(train_loader, epoch)
            self.evaluate(epoch, preds_in_patch_with_score, train_loader,
                          final_output_path=None,
                          debug=False,
                          writer_dict=False)

            val_loss, preds_in_patch_with_score = self.validate(valid_loader, epoch)
            acc = self.evaluate(epoch, preds_in_patch_with_score, valid_loader,
                                final_output_path=None,
                                debug=False,
                                writer_dict=False)

            # TODO: set dataset in config
            # perf_indicator = 500. - acc if config.DATASET.DATASET == 'h36m' or 'mpii_3dhp' or 'jta' else acc
            perf_indicator = 500. - acc if 'h36m' == 'h36m' or 'mpii_3dhp' or 'jta' else acc

            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True  # TODO save the best model separately (start saving after 30 epochs)
            else:
                best_model = False

            # TODO check if saving works and extend according to original code (save best model, ...)
            # save model
            if (epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1) or (
                    epoch + 1 == train_parms.num_epochs and DEVICE == "cuda"):
                self.save_model(epoch)
                self.save_checkpoint(epoch, train_loss, val_loss)

            epoch += 1

    def train_step(self, data_loader, epoch):
        """
        Train model for 1 epoch.
        """
        print("Train step")
        self.model.train()

        # initialize metrics
        # TODO maybe add metrics on training set

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_metric = AverageMeter()
        loss_metric_dict = AverageMeterDict(dictionary_keys=['heatmap_loss', 'regression_loss'])

        # progressbar
        loop = tqdm(data_loader, file=sys.stdout, colour='green')

        end = time.time()

        h36m_preds_in_patch_with_score = []
        h36m_preds_in_patch_with_score_idx = []

        # for all batches
        for batch_idx, data in enumerate(loop):
            # measure data loading time
            data_time.update(time.time() - end)

            batch_data, batch_label, batch_label_weight, meta = data

            self.optimizer.zero_grad()

            batch_data = batch_data.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            batch_label_weight = batch_label_weight.to(DEVICE)

            batch_size = batch_data.size(0)

            # runs the forward pass with autocasting (improve performance while maintaining accuracy)
            # with torch.cuda.amp.autocast():
            predictions = self.model(batch_data)

            loss_rv = self.loss_function(predictions, batch_label, batch_label_weight)
            del batch_data, batch_label, batch_label_weight

            if isinstance(loss_rv, dict):
                loss = loss_rv['loss']
                loss_metric_dict.update(loss_rv)
            else:
                loss = loss_rv

            # backward according to https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            if False:  # TODO: do we use scaler?
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()  # TODO: They use it also twice in original code why ?
                loss.backward()
                self.optimizer.step()

            # update loss
            # record loss
            loss_metric.update(loss.item(), batch_size)

            # update tqdm progressbar
            if isinstance(loss_rv, dict):
                loop.set_postfix(**{k: v.item() for k, v in loss_rv.items()})
            else:
                loop.set_postfix(loss=loss.item())

            del loss

            # update metrics
            preds_with_score = self.result_func(patch_width=self.patch_size[0],
                                                patch_height=self.patch_size[1],
                                                preds=predictions)
            # filter h36m samples
            h36m_samples_batch_idx = np.asarray(meta["name"]) == "h36m"
            h36m_preds_in_patch_with_score.append(preds_with_score[h36m_samples_batch_idx])

            h36m_preds_in_patch_with_score_idx.append((meta["idx"][h36m_samples_batch_idx]).detach().cpu().numpy())
            del predictions, preds_with_score

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # TODO maybe print to stdout time of one batch
            if False and (batch_idx % 100 == 0 or batch_idx == len(data_loader) - 1):
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch, batch_idx, len(data_loader), batch_time=batch_time,
                    speed=batch_size / batch_time.val,
                    data_time=data_time, loss=loss_metric)
                Logcreator.info(msg)

        loop.close()

        # to array
        h36m_preds_in_patch_with_score = np.vstack(h36m_preds_in_patch_with_score)
        # TODO instead of sorting pass the index to the evaluate function
        sorted_idx = np.hstack(h36m_preds_in_patch_with_score_idx).argsort()
        h36m_preds_in_patch_with_score = h36m_preds_in_patch_with_score[sorted_idx]

        train_loss = loss_metric.avg

        loss_metric_dict.log(self.writer, self.comet, epoch, "train")
        loss_metric.log(self.writer, self.comet, epoch, "train")

        self.lr_scheduler.step()  # decay learning rate over time

        return train_loss, h36m_preds_in_patch_with_score

    def validate(self, data_loader, epoch, only_prediction=False):
        """
        Evaluate model on validation data.
        """
        print("Validation step")
        self.model.eval()

        loss_metric = AverageMeter()
        loss_metric_dict = AverageMeterDict(dictionary_keys=['heatmap_loss', 'regression_loss'])

        # initialize the progressbar
        loop = tqdm(data_loader, file=sys.stdout, colour='green')

        preds_in_patch_with_score = []
        with torch.no_grad():
            for i, data in enumerate(loop):
                batch_data, batch_label, batch_label_weight, meta = data

                batch_data = batch_data.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                batch_label_weight = batch_label_weight.to(DEVICE)

                batch_size = batch_data.size(0)

                # compute output
                predictions = self.model(batch_data)

                if not only_prediction:
                    loss_rv = self.loss_function(predictions, batch_label, batch_label_weight)

                    if isinstance(loss_rv, dict):
                        loss = loss_rv['loss']
                        loss_metric_dict.update(loss_rv)
                    else:
                        loss = loss_rv

                    # update loss
                    loss_metric.update(loss.item(), batch_size)

                    # update tqdm progressbar
                    if isinstance(loss_rv, dict):
                        loop.set_postfix(**{k: v.item() for k, v in loss_rv.items()})
                    else:
                        loop.set_postfix(loss=loss.item())

                    del loss

                del batch_data, batch_label, batch_label_weight

                preds_in_patch_with_score.append(self.result_func(patch_width=self.patch_size[0],
                                                                  patch_height=self.patch_size[1],
                                                                  preds=predictions))
                del predictions

            loop.close()

            # to array
            preds_in_patch_with_score = np.vstack(preds_in_patch_with_score)

            val_loss = loss_metric.avg

            if not only_prediction:
                loss_metric_dict.log(self.writer, self.comet, epoch, "val")
                loss_metric.log(self.writer, self.comet, epoch, "val")

            return val_loss, preds_in_patch_with_score

    def evaluate(self, epoch, preds_in_patch_with_score, val_loader, final_output_path, debug=False, writer_dict=None):
        print("Evaluation step")

        # TODO also evaluate on mpii?
        for image_set in val_loader.dataset.datasets:  # iterate through all datasets
            if image_set.name == "h36m":  # evaluate on h36m
                imdb_list = image_set.db
                imdb = image_set

                # From patch to original image coordinate system
                preds_in_img_with_score = []

                for n_sample in range(len(image_set)):
                    # TODO we not neccessaraly pick the "preds_in_patch_with_score[n_sample]" of the h36m set (could also be the mpii),
                    #  it depends on the order of the paramter "dataset" : ["h36m", "mpii"] in the config file
                    preds_in_img_with_score.append(
                        trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample],
                                                          imdb_list[n_sample]['center_x'],
                                                          imdb_list[n_sample]['center_y'],
                                                          imdb_list[n_sample]['width'],
                                                          imdb_list[n_sample]['height'],
                                                          patch_width=self.patch_size[0],
                                                          patch_height=self.patch_size[1],
                                                          rect_3d_width=2000, rect_3d_height=2000))

                preds_in_img_with_score = np.asarray(preds_in_img_with_score)

                # Evaluate
                if 'joints_3d' in imdb.db[0].keys():
                    name_value, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path,
                                                     debug=debug,
                                                     epoch=epoch,
                                                     writer=self.writer,
                                                     comet=self.comet,
                                                     writer_dict=writer_dict)
                else:
                    Logcreator.info('Test set is used, saving results to %s', final_output_path)
                    _, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path,
                                            debug=debug,
                                            writer_dict=writer_dict)
                    perf = 0.0

                Logcreator.info("Mean per joint position error", perf)

                return perf

        return 0.0

    def save_model(self, epoch_nr):
        """ This function saves entire model incl. modelstructure"""
        Configuration.model_save_folder = os.path.join(Configuration.output_directory, "whole_model_backups")
        if not os.path.exists(Configuration.model_save_folder):
            os.makedirs(Configuration.model_save_folder)
        file_name = str(epoch_nr) + "_whole_model_serialized.pth"
        torch.save(self.model, os.path.join(Configuration.model_save_folder, file_name))

    def save_checkpoint(self, epoch, tl, vl):
        Configuration.weights_save_folder = os.path.join(Configuration.output_directory, "weights_checkpoint")
        if not os.path.exists(Configuration.weights_save_folder):
            os.makedirs(Configuration.weights_save_folder)
        file_name = str(epoch) + "_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_loss': tl,
            'val_loss': vl,
        }, os.path.join(Configuration.weights_save_folder, file_name))

    def load_checkpoints(self, path=None):
        Logcreator.info("Loading checkpoint file:", path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']

        print("Loaded model checkpoint")
        print("Total model parameters: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        return epoch, train_loss, val_loss

    def print_modelsummary(self):
        image_size = Configuration.get("data_collection.image_size")
        input_size = tuple(np.insert(image_size, 0, values=3))

        Logcreator.debug(self.model)

        # redirect stdout to our logger
        sys.stdout = my_stdout = StringIO()
        summary(self.model, input_size=input_size, device=DEVICE)
        # reset stdout to original
        sys.stdout = sys.__stdout__
        Logcreator.info(my_stdout.getvalue())

        Logcreator.debug("Model '%s' initialized with %d parameters." %
                         (Configuration.get('training.model.name'),
                          sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
