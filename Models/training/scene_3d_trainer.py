import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.augmentations import Augmentations
from Models.model_components.scene_3d.scene_3d_network import Scene3DNetwork
from Models.model_components.scene_3d.scene_seg_network import SceneSegNetwork


class Scene3DTrainer:
    def __init__(
        self,
        checkpoint_path="",
        pretrained_checkpoint_path="",
        is_pretrained=False,
    ):
        # Image and Ground Truth
        self.image = 0
        self.gt = 0

        # Tensors
        self.image_tensor = 0
        self.gt_tensor = 0

        # Model and Prediction
        self.model = 0
        self.prediction = 0

        # Losses
        self.loss = 0
        self.edge_loss = 0
        self.mAE_loss = 0
        self.gram_loss = 0
        self.statistics_loss = 0
        self.edge_scale_factor = 4

        # Checking devices (GPU vs CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for training")

        if is_pretrained:
            # Instantiate Model for validation or resuming training - load full Scene3D weights
            if len(checkpoint_path) > 0:
                scene_seg_network = SceneSegNetwork(pretrained=False)
                self.model = Scene3DNetwork(scene_seg_network)
                self.model.load_state_dict(
                    torch.load(checkpoint_path, weights_only=True, map_location=self.device)
                )
                print(
                    "Loading pre-trained model weights of Scene3D and upstream SceneSeg weights as well"
                )
            else:
                raise ValueError(
                    "Please ensure Scene3D network weights are provided for downstream elements"
                )
        else:
            # Instantiate Model for training - load pre-trained SceneSeg weights only
            if len(pretrained_checkpoint_path) > 0:
                scene_seg_network = SceneSegNetwork(pretrained=False)
                scene_seg_network.load_state_dict(
                    torch.load(
                        pretrained_checkpoint_path,
                        weights_only=True,
                        map_location=self.device,
                    )
                )
                self.model = Scene3DNetwork(scene_seg_network)
                print(
                    "Loading pre-trained model weights of upstream SceneSeg only, Scene3D initialized with random weights"
                )
            else:
                raise ValueError(
                    "Please ensure SceneSeg network weights are provided for upstream elements"
                )

        # Model to device
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Gradient filters
        self.gx_filter = (
            torch.tensor(
                [[0.125, 0, -0.125], [0.25, 0, -0.25], [0.125, 0, -0.125]],
                dtype=torch.float32,
            )
            .view((1, 1, 3, 3))
            .to(self.device)
        )

        self.gy_filter = (
            torch.tensor(
                [[0.125, 0.25, 0.125], [0, 0, 0], [-0.125, -0.25, -0.125]],
                dtype=torch.float32,
            )
            .view((1, 1, 3, 3))
            .to(self.device)
        )

    # Learning Rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Image augmentations
    def apply_augmentations(self, is_train):
        if is_train:
            augTrain = Augmentations(is_train=True, data_type="DEPTH")
            augTrain.setData(self.image, self.gt)
            self.image, self.gt = augTrain.applyTransformDepth(
                image=self.image, ground_truth=self.gt
            )
        else:
            augVal = Augmentations(is_train=False, data_type="DEPTH")
            augVal.setData(self.image, self.gt)
            self.image, self.gt = augVal.applyTransformDepth(
                image=self.image, ground_truth=self.gt
            )

    # Load Data
    def load_data(self):
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

        gt_tensor = torch.from_numpy(self.gt)
        gt_tensor = gt_tensor.permute(2, 0, 1)
        gt_tensor = gt_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.type(torch.FloatTensor)
        self.gt_tensor = gt_tensor.to(self.device)

    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        prediction_ssi = self.get_ssi_nom_tensor(self.prediction)
        gt_ssi = self.get_ssi_nom_tensor(self.gt_tensor)
        self.mAE_loss = self.calc_mAE_ssi_loss_robust(prediction_ssi, gt_ssi)
        self.edge_loss = self.calc_multi_scale_ssi_edge_loss(prediction_ssi, gt_ssi)
        self.loss = self.mAE_loss + self.edge_scale_factor * self.edge_loss

    def get_ssi_nom_tensor(self, tensor):
        denom = (torch.max(tensor) - torch.mean(tensor)).clamp_min(1e-6)
        ssi_tensor = (tensor - torch.min(tensor)) / denom
        return ssi_tensor

    def get_ssi_tensor(self, tensor):
        median_val = torch.median(tensor)
        shifted_tensor = tensor - median_val
        scale_val = torch.mean(torch.abs(shifted_tensor)).clamp_min(1e-6)
        ssi_tensor = shifted_tensor / scale_val
        return ssi_tensor

    def calc_mAE_ssi_loss_robust(self, prediction_ssi, gt_ssi):
        mAE = torch.abs(prediction_ssi - gt_ssi)
        mAE_robust_val = torch.quantile(mAE, 0.9, interpolation="linear")
        mAE_robust = mAE[mAE < mAE_robust_val]
        if mAE_robust.numel() == 0:
            return torch.mean(mAE)
        mAE_loss = torch.mean(mAE_robust)
        return mAE_loss

    def calc_multi_scale_ssi_edge_loss(self, prediction_ssi, gt_ssi):
        downsample = nn.AvgPool2d(2, stride=2)

        prediction_ssi_d1 = downsample(prediction_ssi)
        prediction_ssi_d2 = downsample(prediction_ssi_d1)
        prediction_ssi_d3 = downsample(prediction_ssi_d2)
        gt_ssi_d1 = downsample(gt_ssi)
        gt_ssi_d2 = downsample(gt_ssi_d1)
        gt_ssi_d3 = downsample(gt_ssi_d2)

        edge_loss_d0 = self.calc_edge_ssi_loss(prediction_ssi, gt_ssi)
        edge_loss_d1 = self.calc_edge_ssi_loss(prediction_ssi_d1, gt_ssi_d1)
        edge_loss_d2 = self.calc_edge_ssi_loss(prediction_ssi_d2, gt_ssi_d2)
        edge_loss_d3 = self.calc_edge_ssi_loss(prediction_ssi_d3, gt_ssi_d3)

        multi_scale_edge_loss = (
            edge_loss_d0 + edge_loss_d1 + edge_loss_d2 + edge_loss_d3
        ) / 4

        return multi_scale_edge_loss

    def calc_edge_ssi_loss(self, prediction_ssi, gt_ssi):
        G_x_pred = nn.functional.conv2d(prediction_ssi, self.gx_filter, padding=1)
        G_y_pred = nn.functional.conv2d(prediction_ssi, self.gy_filter, padding=1)

        G_x_gt = nn.functional.conv2d(gt_ssi, self.gx_filter, padding=1)
        G_y_gt = nn.functional.conv2d(gt_ssi, self.gy_filter, padding=1)

        edge_diff_mAE = torch.abs(G_x_pred - G_x_gt) + torch.abs(G_y_pred - G_y_gt)
        edge_loss = torch.mean(edge_diff_mAE)

        return edge_loss

    # Loss Backward Pass
    def loss_backward(self):
        self.loss.backward()

    def get_loss(self):
        return self.loss.item()

    def get_mAE_loss(self):
        return self.mAE_loss.item()

    def get_edge_loss(self):
        return self.edge_loss.item()

    def get_gram_loss(self):
        return self.gram_loss.item()

    def get_statistics_loss(self):
        return self.statistics_loss.item()

    # Logging Loss
    def log_loss(self, log_count):
        self.writer.add_scalars(
            "Train",
            {
                "total_loss": self.get_loss(),
                "edge_loss": self.get_edge_loss() * self.edge_scale_factor,
                "mAE_loss": self.get_mAE_loss(),
            },
            log_count,
        )

    def log_val_loss(self, total_loss, mAE_loss, edge_loss, log_count):
        self.writer.add_scalars(
            "Validation",
            {
                "total_loss": total_loss,
                "mAE_loss": mAE_loss,
                "edge_loss": edge_loss * self.edge_scale_factor,
            },
            log_count,
        )

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_train_mode(self):
        self.model = self.model.train()

    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save predicted visualization
    def save_visualization(self, log_count):
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = prediction_vis.numpy()

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(self.image)
        axs[0].set_title("Image", fontweight="bold")
        axs[1].imshow(self.gt)
        axs[1].set_title("Ground Truth", fontweight="bold")
        axs[2].imshow(prediction_vis)
        axs[2].set_title("Prediction", fontweight="bold")
        self.writer.add_figure(
            "predictions vs. actuals", fig, global_step=log_count
        )
        plt.close(fig)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    # Run Validation and calculate metrics
    def validate(self, image, gt):
        self.set_data(image, gt)
        self.apply_augmentations(is_train=False)
        self.load_data()

        self.prediction = self.model(self.image_tensor)

        gt_ssi = self.get_ssi_nom_tensor(self.gt_tensor)
        prediction_ssi = self.get_ssi_nom_tensor(self.prediction)

        val_mAE_loss = self.calc_mAE_ssi_loss_robust(prediction_ssi, gt_ssi)
        val_mEL_loss = self.calc_multi_scale_ssi_edge_loss(prediction_ssi, gt_ssi)

        val_mAE = val_mAE_loss.detach().cpu().numpy()
        val_mEL = val_mEL_loss.detach().cpu().numpy()

        return val_mAE, val_mEL

    # Run network on test image and visualize result
    def test(self, image_test, save_path):
        frame = cv2.imread(str(image_test), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((640, 320))

        test_image_tensor = self.image_loader(image_pil)
        test_image_tensor = test_image_tensor.unsqueeze(0)
        test_image_tensor = test_image_tensor.to(self.device)
        test_output = self.model(test_image_tensor)

        test_output = test_output.squeeze(0).cpu().detach()
        test_output = test_output.permute(1, 2, 0)
        test_output = test_output.numpy()
        test_output = cv2.resize(test_output, (frame.shape[1], frame.shape[0]))

        plt.imsave(save_path, test_output, cmap="viridis")

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("Finished Training")
