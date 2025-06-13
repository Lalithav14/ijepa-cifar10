import torch
import torch.nn as nn
import torch.nn.functional as F

class IJepaStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.teacher_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.mask_ratio = 0.5

    def mask_image(self, x):
        # Create random binary mask
        B, C, H, W = x.shape
        mask = (torch.rand(B, 1, H, W, device=x.device) > self.mask_ratio).float()
        return x * mask, mask

    def forward(self, x, epoch):
        student_input, mask = self.mask_image(x)

        # Teacher input logic
        if epoch < 2:
            teacher_input = x  # full image
        elif 2 <= epoch < 10:
            masked_teacher, _ = self.mask_image(x)
            noise = torch.randn_like(x) * 0.1
            teacher_input = masked_teacher + noise
        else:
            teacher_input, _ = self.mask_image(x)

        student_feat = self.student_encoder(student_input)
        teacher_feat = self.teacher_encoder(teacher_input)

        # Loss: L2 only on masked patches
        loss = F.mse_loss(student_feat * mask, teacher_feat * mask)
        return loss
