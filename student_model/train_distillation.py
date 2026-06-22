class DistillationLoss(nn.Module):
    def __init__(self, w=10, epsilon=2, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.wing_loss = WingLoss(w=w, epsilon=epsilon)
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha # Balance between GT and Teacher

    def forward(self, student_pred, gt_target, teacher_pred):
        """
        student_pred: (N, 21, 2)
        gt_target:    (N, 21, 2) - Ground Truth labels
        teacher_pred: (N, 21, 2) - Teacher model predictions
        """
        # 1. Loss against hard labels (Ground Truth)
        loss_gt = self.wing_loss(student_pred, gt_target)
        python
        # 2. Loss against soft labels (Teacher's Prediction)
        loss_distill = self.l1_loss(student_pred, teacher_pred)
        
        
        # 3. Combined Loss
        # alpha=0.5 means we trust the Teacher and GT equally
        total_loss = (self.alpha * loss_gt) + ((1 - self.alpha) * loss_distill)
        
        return total_loss, loss_gt, loss_distill