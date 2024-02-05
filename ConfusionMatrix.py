import torch


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = torch.zeros(size=(num_classes, num_classes)).cuda()

    def updateMatrix(self, pred, ground_truth):
        k = (0 <= ground_truth) & (ground_truth < self.num_classes)
        ground_truth_ = ground_truth[k]
        index = self.num_classes * ground_truth_ + pred
        self.matrix += torch.bincount(input=index, minlength=self.num_classes ** 2).reshape(self.num_classes,
                                                                                            self.num_classes)

    def Percision(self,certain_item=0,avg=True):
        epslion=1e-8
        if avg:
            diag_matrix=torch.diag(self.matrix)
            denominator = torch.sum(self.matrix, dim=1)
            re = torch.mean(diag_matrix / denominator)
        else:
            TP=self.matrix[certain_item,certain_item]
            denominator=torch.sum(self.matrix[:,certain_item])+epslion
            re=TP/denominator
        return re

    def Recall(self,certain_item=0,avg=True):
        epslion=1e-8
        if avg:
            diag_matrix=torch.diag(self.matrix)
            denominator=torch.sum(self.matrix,dim=0)
            re=torch.mean(diag_matrix/denominator)
        else:
            TP = self.matrix[certain_item, certain_item]
            denominator = torch.sum(self.matrix[certain_item, :]) + epslion
            re = TP / denominator

        return re

    def F1_score(self,certain_item=0,avg=True):
        epslion = 1e-8
        percision=self.Percision(certain_item,avg=avg)
        recall=self.Recall(certain_item,avg=avg)
        f1=2*percision*recall/(percision+recall)+epslion
        return f1

    def Accuracy(self):
        diag_matrix=torch.diag(self.matrix)
        acc=torch.sum(diag_matrix)/torch.sum(self.matrix)
        return acc

    def get_metrics(self,certain_item=0,avg=True):
        percision=self.Percision(certain_item=certain_item,avg=avg)
        recall=self.Recall(certain_item=certain_item,avg=avg)
        f1=self.F1_score(certain_item=certain_item,avg=avg)
        acc=self.Accuracy()

        return acc,percision,recall,f1


