import torch

from ConfusionMatrix import ConfusionMatrix
import time
from torch import distributed as dist
import os
from torch.nn import functional


class Trainer(object):

    def __init__(self, model, loss, optimizer,
                 scheduler, train_loader, test_loader,
                 train_sampler, test_sampler, train_config):
        """
        模型训练类，包含了模型训练，模型存储，模型加载以及模型评估
        :param model:
        :param loss:
        :param optimizer:
        :param scheduler:
        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param train_config:
        """

        super(Trainer, self).__init__()
        self.model = model
        self.train_config = train_config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.logs = {'train_loss': [],
                     'train_acc': [],
                     'train_percision': [],
                     'train_recall': [],
                     'train_f1': [],
                     'test_loss': [],
                     'test_acc': [],
                     'test_percision': [],
                     'test_recall': [],
                     'test_f1': [],
                     'epoch': []}

    def train_epoch(self, matrix):
        """
        训练模型
        :return: 当前批次的损失 float
        """

        self.model.train()
        # 当前批次模型总损失值
        total_loss = 0

        iteration_index = 0

        for X, labels in self.train_loader:
            iteration_index += 1
            # 将样本数据加载到gpu上
            X, labels = self._load_gpu(X, labels)

            # 得到模型预测结果[N,C,H,W]
            output = self.model(X)

            # update confusion matrix
            # labes:[B]
            matrix.updateMatrix(pred=torch.argmax(output, dim=1), ground_truth=labels)

            # 计算模型损失
            # labels=functional.one_hot(labels,self.train_config.get('num_class')).float()
            loss = self.loss(output, labels)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        acc, percision, recall, f1 = matrix.get_metrics(avg=True)

        return total_loss / iteration_index, acc.item(), percision.item(), recall.item(), f1.item()

    def train(self):

        # Mean Intersection over Union
        best_mean_accuracy = 0
        matrix = ConfusionMatrix(self.train_config.get("num_class"))

        for epoch in range(1, self.train_config.get('epochs') + 1):
            start_time = time.time()
            self.train_sampler.set_epoch(epoch)
            # 训练模型
            train_epoch_loss, train_epoch_acc, train_epoch_perision, train_epoch_recall, train_epoch_f1 = self.train_epoch(
                matrix)

            end_time = time.time()
            train_time = end_time - start_time

            self.logs['train_loss'].append(train_epoch_loss)
            self.logs['train_acc'].append(train_epoch_acc)
            self.logs['train_percision'].append(train_epoch_perision)
            self.logs['train_recall'].append(train_epoch_recall)
            self.logs['train_f1'].append(train_epoch_f1)

            print(
                "[process %s] [%s]: training_time:%.2f秒\t train_loss:%.3f\t train_accuracy:%.3f\ttrain_percision:%.3f\ttrain_recall:%.3f\ttrain_f1:%.3f\tlr:%.6f" % (
                    dist.get_rank(), epoch, train_time, train_epoch_loss, train_epoch_acc, train_epoch_perision,
                    train_epoch_recall, train_epoch_f1,
                    self.scheduler.get_lr()[0]))

            # 使用验证集，验证模型效果，并保存模型
            avg_loss, avg_accuracy,avg_percision,avg_recall,avg_f1 = self.evaluate()
            self.logs['test_loss'].append(avg_loss)
            self.logs['test_acc'].append(avg_accuracy)
            self.logs['test_percision'].append(avg_percision)
            self.logs['test_recall'].append(avg_recall)
            self.logs['test_f1'].append(avg_f1)

            if best_mean_accuracy < avg_accuracy and dist.get_rank() == 0:
                best_mean_accuracy = avg_accuracy
                self.save_model(epoch)
                print("[Validation]\tavg_loss:%.3f\tavg_f1:%.3f\taccuracy:%.3f\n" % (
                    avg_loss, avg_f1, avg_accuracy))

            self.scheduler.step()
            time.sleep(self.train_config.get('sleep_time'))

    def evaluate(self):
        self.model.eval()
        matrix = ConfusionMatrix(self.train_config.get('num_class'))
        total_loss = torch.zeros(1).cuda()

        step = 0
        with torch.no_grad():
            for X, label in self.test_loader:
                step += 1
                X, labels = self._load_gpu(X, label)
                output = self.model(X)

                matrix.updateMatrix(pred=torch.argmax(output, dim=1), ground_truth=labels)

                # labels=functional.one_hot(labels, self.train_config.get('num_class')).float()
                loss = self.loss(output, labels)
                total_loss += loss

        acc,percision,recall,f1 = matrix.get_metrics(avg=True)
        avg_loss = self.reduce_value(total_loss / step, avg=True)
        avg_percision = self.reduce_value(percision, avg=True)
        avg_recall = self.reduce_value(recall, avg=True)
        avg_f1 = self.reduce_value(f1, avg=True)
        avg_accuracy = self.reduce_value(acc, avg=True)
        return avg_loss.item(), avg_accuracy.item(), avg_percision.item(), avg_recall.item(),avg_f1.item()

    def reduce_value(self, value, avg=True):

        world_size = dist.get_world_size()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if avg:
            return value / world_size
        else:
            return value

    def _load_gpu(self, X, labels):
        """
        加载样本数据到gpu
        :param X:
        :param labels:
        :return:
        """
        return X.float().cuda(), labels.cuda()

    def save_model(self, epoch):
        """
        保存模型
        :param epoch: 训练模型的批次
        :return:
        """


        save_model_path = os.path.join(self.train_config.get('experiment_folder'),
                                       '%s_%s.pkl' % (self.train_config.get('model_name'), epoch))
        torch.save(self.model.state_dict(), save_model_path)
        print("save model:%s" % save_model_path)
