import torch


class SegMetrics:
    
    def __init__(self, n_class, device):
        self.n_class = n_class
        self.confusion_matrix = torch.zeros((n_class, n_class)).to(device)
        
    def update(self, true, pred):
        for t, p in zip(true, pred):
            self.confusion_matrix += self._fast_hist(t.flatten(), p.flatten())
            
    def _fast_hist(self, true, pred):
        mask = (true >= 0) & (true < self.n_class)
        bin_flat = torch.bincount(self.n_class * true[mask].to(dtype=int) + pred[mask], minlength=self.n_class**2)
        return bin_flat.reshape(self.n_class, self.n_class)
    
    def get_results(self):
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        acc_cls = torch.nanmean(acc_cls)
        iou = torch.diag(hist) / (hist.sum(dim=0) + hist.sum(dim=1) - torch.diag(hist))
        mean_iou = torch.nanmean(iou)
        cls_iou = dict(zip(range(self.n_class), iou))
        
        return {
            "Overall acc": acc,
            "Mean acc": acc_cls,
            "Mean IoU": mean_iou,
            "Class IoU": cls_iou
        }
        
    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_class, self.n_class))