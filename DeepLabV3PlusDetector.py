import os
import utils
import network

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T

from datasets import VOCSegmentation, Cityscapes, cityscapes

class DeepLabV3PlusDetector:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }
        self.valid_dataset = ['voc', 'cityscapes']
        self.checkpoint = None
        self.dataset = None
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0
        # [8, 16]
        self.output_stride = 16
        self.separable_conv = False
        self.device = None
        self.gpu_id = '0'
        self.crop_val = False
        self.crop_size = 513
        self.save_val_results_to = "./result"
        self.val_batch_size = 4
        self.num_classes = None
        self.decode_fn = None
        self.transform = None
        return

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def startTimer(self):
        self.time_start = time.time()

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")

    def getAverageTime(self):
        if self.detected_num == 0:
            return -1

        return 1.0 * self.total_time_sum / self.detected_num

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def setDataset(self, dataset):
        self.dataset = dataset
        if self.dataset == 'voc':
            self.num_classes = 21
            self.decode_fn = VOCSegmentation.decode_target
        elif dataset == 'cityscapes':
            self.num_classes = 19
            self.decode_fn = Cityscapes.decode_target
        return

    def loadModel(self, checkpoint, dataset):
        self.reset()

        self.checkpoint = checkpoint

        self.setDataset(dataset)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

        self.model = self.model_map["deeplabv3plus_resnet50"](num_classes=self.num_classes, output_stride=self.output_stride)

        if self.separable_conv and 'plus' in self.model:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        
        if self.checkpoint is not None and os.path.isfile(checkpoint):
            checkpoint_data = torch.load(self.checkpoint, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint_data["model_state"])
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            print("Resume model from %s" % self.checkpoint)
            del checkpoint_data
        else:
            print("[!] Retrain")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
        if self.crop_val:
            self.transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),])
        else:
            self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])

        self.model = self.model.eval()
        return

    def detect(self, image):
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0) # To tensor of NCHW
            image = image.to(self.device)
            result = self.model(image).max(1)[1].cpu().numpy()[0] # HW
        return result
 
if __name__ == '__main__':
    image_folder_path = "./samples/"
    checkpoint = "./best_deeplabv3plus_resnet50_voc_os16.pth"
    dataset = 'voc'
    save_val_results_to = "./result/"
    if save_val_results_to is not None:
        os.makedirs(save_val_results_to, exist_ok=True)

    deeplabv3plus_detector = DeepLabV3PlusDetector()

    deeplabv3plus_detector.loadModel(checkpoint, dataset)

    image_file_name_list = os.listdir(image_folder_path)

    for image_file_name in image_file_name_list:
        if image_file_name[-4:] not in [".jpg", ".png"]:
            continue
        image_file_path = os.path.join(image_folder_path, image_file_name)
        image = Image.open(image_file_path).convert('RGB')

        result = deeplabv3plus_detector.detect(image)

        colorized_result = deeplabv3plus_detector.decode_fn(result).astype('uint8')
        colorized_result = Image.fromarray(colorized_result)

        if save_val_results_to:
            colorized_result.save(os.path.join(save_val_results_to, image_file_name.split('.')[0] +'.png'))

