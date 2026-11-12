
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_3d_network import Scene3DNetwork
from data_utils.augmentations import Augmentations

class Scene3DTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if(is_pretrained):

            # Instantiate Model for validation or inference - load both pre-traiend SceneSeg and SuperDepth weights
            if(len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = Scene3DNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                    (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of Scene3D and upstream SceneSeg weights as well')
            else:
                raise ValueError('Please ensure Scene3D network weights are provided for downstream elements')
            
        else:

            # Instantiate Model for training - load pre-traiend SceneSeg weights only
            if(len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load \
                    (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
                    
                # Loading model with pre-trained upstream weights
                self.model = Scene3DNetwork(sceneSegNetwork)
                print('Loading pre-trained model weights of upstream SceneSeg only, Scene3D initialised with random weights')
            else:
                raise ValueError('Please ensure SceneSeg network weights are provided for upstream elements')
        
       
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Gradient filters
        # Gradient - x
        self.gx_filter = torch.Tensor([[0.125, 0, -0.125],
        [0.25, 0, -0.25],
        [0.125, 0, -0.125]])
        self.gx_filter = self.gx_filter.view((1,1,3,3))
        self.gx_filter = self.gx_filter.type(torch.cuda.FloatTensor)
        self.gx_filter.to(self.device)
        
        # Gradient - y
        self.gy_filter = torch.Tensor([[0.125, 0.25, 0.125],
        [0, 0, 0],
        [-0.125, -0.25, -0.125]])
        self.gy_filter = self.gy_filter.view((1,1,3,3))
        self.gy_filter = self.gy_filter.type(torch.cuda.FloatTensor)
        self.gy_filter.to(self.device)

    # Learning Rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    
    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Data for training
            augTrain = Augmentations(is_train=True, data_type='DEPTH')
            augTrain.setData(self.image, self.gt)
            self.image, self.gt = \
                augTrain.applyTransformDepth(image=self.image,ground_truth=self.gt)
            
        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='DEPTH')
            augVal.setData(self.image, self.gt)
            self.image, self.gt = \
                augVal.applyTransformDepth(image=self.image,ground_truth=self.gt)
            

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
        self.loss = self.mAE_loss + self.edge_scale_factor*self.edge_loss

    def get_ssi_nom_tensor(self, tensor):
        ssi_tensor = (tensor - torch.min(tensor)) \
            /(torch.max(tensor) - torch.mean(tensor))
        return ssi_tensor

    def get_ssi_tensor(self, tensor):
        median_val = torch.median(tensor)
        shifted_tensor = tensor - median_val
        scale_val = torch.mean(torch.abs(shifted_tensor))
        ssi_tensor = shifted_tensor/scale_val
        return ssi_tensor

    def calc_mAE_ssi_loss_robust(self, prediction_ssi, gt_ssi):
        mAE = torch.abs(prediction_ssi - gt_ssi)
        mAE_robust_val = torch.quantile(mAE, 0.9, interpolation='linear')
        mAE_robust = mAE[mAE < mAE_robust_val]
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

        multi_scale_edge_loss = \
            (edge_loss_d0 + edge_loss_d1 + edge_loss_d2 + edge_loss_d3)/4

        return multi_scale_edge_loss

    
    def calc_edge_ssi_loss(self, prediction_ssi, gt_ssi):
        G_x_pred = nn.functional.conv2d(prediction_ssi, self.gx_filter, padding=1)
        G_y_pred = nn.functional.conv2d(prediction_ssi, self.gy_filter, padding=1)

        G_x_gt = nn.functional.conv2d(gt_ssi, self.gx_filter, padding=1)
        G_y_gt = nn.functional.conv2d(gt_ssi, self.gy_filter, padding=1)

        edge_diff_mAE = torch.abs(G_x_pred - G_x_gt) + \
                            torch.abs(G_y_pred - G_y_gt)
        edge_loss = torch.mean(edge_diff_mAE)

        return edge_loss
    

    # Loss Backward Pass
    def loss_backward(self): 
        self.loss.backward()

    # Get mAE loss value
    def get_loss(self):
        return self.loss.item()
    
    # Get edge loss
    def get_mAE_loss(self):
        return self.mAE_loss.item()
    
    # Get edge loss
    def get_edge_loss(self):
        return self.edge_loss.item()
    
    # Get gram loss
    def get_gram_loss(self):
        return self.gram_loss.item()
    
    # Get statistics loss
    def get_statistics_loss(self):
        return self.statistics_loss.item()

    # Logging Loss
    def log_loss(self, log_count):
        self.writer.add_scalars("Train",{
            'total_loss': self.get_loss(),
            'edge_loss': self.get_edge_loss()*self.edge_scale_factor,
            'mAE_loss': self.get_mAE_loss()
        }, (log_count))
           
        
    # Logging Loss
    def log_val_loss(self, total_loss, mAE_loss, edge_loss, log_count):
        self.writer.add_scalars("Validation",{
            'total_loss': total_loss,
            'mAE_loss': mAE_loss,
            'edge_loss': edge_loss*self.edge_scale_factor
        }, (log_count))
         
    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()
    
    # Save predicted visualization
    def save_visualization(self, log_count):

        # Converting prediction outputs to visualization
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = prediction_vis.numpy()

        # Displaying on figure
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(self.gt)
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[2].imshow(prediction_vis)
        axs[2].set_title('Prediction',fontweight ="bold")
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))   
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    # Run Validation and calculate metrics
    def validate(self, image, gt):

        # Set Data
        self.set_data(image, gt)

        # Augmenting Image
        self.apply_augmentations(is_train=False)
        
        # Converting to tensor and loading
        self.load_data()

        # Running model
        self.prediction = self.model(self.image_tensor)

        # Calculate loss
        gt_ssi = self.get_ssi_nom_tensor(self.gt_tensor)
        prediction_ssi = self.get_ssi_nom_tensor(self.prediction)
        
        val_mAE_loss = self.calc_mAE_ssi_loss_robust(prediction_ssi, gt_ssi)
        val_mEL_loss = self.calc_multi_scale_ssi_edge_loss(prediction_ssi, gt_ssi)

        val_mAE = val_mAE_loss.detach().cpu().numpy()
        val_mEL = val_mEL_loss.detach().cpu().numpy()

        return val_mAE, val_mEL
    
    # Run network on test image and visualize result
    def test(self, image_test, save_path):

        frame = cv2.imread(image_test, cv2.IMREAD_COLOR)
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

        plt.imsave(save_path, test_output, cmap='viridis')

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')