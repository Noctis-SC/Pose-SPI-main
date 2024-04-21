from data_loader import CocoKeypointsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.model_grayscale_input import KeypointDetection
from torch.utils.tensorboard import SummaryWriter
import torch
from pycocotools.coco import COCO
from utils import evaluate_keypoint_detection, AverageMeter, draw_keypoints
import argparse
import os

"""Setting for having reproducable experiments in torch"""
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Human keypoint detection")
parser.add_argument("--data_path", default='./toydataset/images/train2017', type=str,help="Path to images")
parser.add_argument("--annotation_path", default='./toydataset/annotations/annotations.json', type=str,help="Path to json annotation")

parser.add_argument("--epochs", default=1000, type=int, help="Number of total epochs")
parser.add_argument("--batch", default=4, type=int, help="mini-batch size per GPU.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--output_dir", default="weights", help="Path to save weights ")
parser.add_argument("--num_keypoints", type=int, default=13, help="number of keypoints for the model construction and coco evaluation")


args = parser.parse_args()
print(args)

try:
    os.makedirs(args.output_dir)
except OSError:
    pass

# Initialize COCO dataset object
coco = COCO(args.annotation_path)

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
])

# Initialize COCO-style keypoints dataset
train_dataset = CocoKeypointsDataset(coco, args.data_path, transform)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)


# TODO CHANGE PATHS WHEN VALIDATION SET WILL BE AVAILABLE
val_dataset = CocoKeypointsDataset(coco, args.data_path, transform)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

# Load model
model = KeypointDetection(num_keypoints=args.num_keypoints)

# Set device for training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

criterion = torch.nn.MSELoss()

# Define optimizer and learning rate scheduler, RMSPROP
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# Create tensorboard instance for logging
summary = SummaryWriter()

#TODO scheduler can be added if there will be need in future experiments

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20], gamma=10)

# Training loop
for epoch in range(args.epochs):
    model.train()
    for idx, (images, keypoints, image_id) in enumerate(train_data_loader):

        keypoints = keypoints.to(device)
        images = images.to(device)

        outputs = model(images)

        loss = criterion(outputs, keypoints)
        loss_total = loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if idx % 50 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}")
            summary.add_scalar("loss/train", loss.item(), global_step=epoch * len(train_data_loader) + idx)
    # scheduler.step()

    # Evaluation loop
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for val_idx, (images, keypoints, _) in enumerate(val_data_loader):
            images = images.to(device)
            keypoints = keypoints.to(device)

            outputs = model(images)

            loss = criterion(outputs, keypoints)
            loss_meter.update(loss.item(), images.size(0))

            if val_idx % 20 == 0:
                result_img = draw_keypoints(pred_keypoints=outputs[0], gt_keypoints=keypoints[0], images=images)
                summary.add_image("results", result_img, global_step=epoch * len(val_data_loader) + idx)
        mse_val = loss_meter.avg
        # TODO printing, model saving and coco-style evaluation can be uncommented when real dataset will be used
        # print("epoch{}, mse: {}".format(epoch, mse_val))
        # state = {'model': model.state_dict(), 'metric': mse_val, 'epoch': epoch, "optim": optimizer.state_dict()}
        # torch.save(state, os.path.join(args.output_dir, str(epoch) + 'ckpt.pth'))
        # summary.add_scalar('mse_val', mse_val, epoch)

        # evaluate_keypoint_detection(model=model, device=device, ann_file=args.annotation_path, data_loader=train_data_loader, num_keypoints=args.num_keypoints)

        # Close the SummaryWriter
        summary.close()
