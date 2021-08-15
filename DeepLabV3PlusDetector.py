import os
import utils
import network
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms as T

from datasets import VOCSegmentation, Cityscapes, cityscapes

os.makedirs("./result/", exist_ok=True)

class DeepLabV3PlusDetector:
    def __init__(self):
        self.reset()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
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
        self.valid_model_name_list = ['deeplabv3_resnet50',
                                      'deeplabv3plus_resnet50',
                                      'deeplabv3_resnet101',
                                      'deeplabv3plus_resnet101',
                                      'deeplabv3_mobilenet',
                                      'deeplabv3plus_mobilenet']
        self.valid_backend_name_list = ['resnet50', 'resnet101', 'mobilenet']
        self.valid_dataset_name_list = ['voc', 'cityscapes']
        self.model_name = None
        self.checkpoint = None
        self.checkpoint_ready = False
        self.dataset_name = None
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

    def getAverageTimeMS(self):
        if self.detected_num == 0:
            return -1

        return int(1000.0 * self.total_time_sum / self.detected_num)

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def setCheckPoint(self, checkpoint):
        self.checkpoint = checkpoint
        self.checkpoint_ready = False

        checkpoint_file_name = os.path.basename(checkpoint)

        is_plus = False
        backend_index = -1
        dataset_index = -1

        for i in range(len(self.valid_backend_name_list)):
            if self.valid_backend_name_list[i] in checkpoint_file_name:
                backend_index = i
                break

        for i in range(len(self.valid_dataset_name_list)):
            if self.valid_dataset_name_list[i] in checkpoint_file_name:
                dataset_index = i
                break

        if 'deeplabv3' not in checkpoint_file_name:
            print("Cannot load this checkpoint type!")
            return

        if 'plus' in checkpoint_file_name:
            is_plus = True

        if backend_index == -1 or dataset_index == -1:
            print("Cannot load this checkpoint type!")
            return

        self.model_name = 'deeplabv3'
        if is_plus:
            self.model_name += 'plus'
        self.model_name += '_'
        self.model_name += self.valid_backend_name_list[backend_index]
        self.dataset_name = self.valid_dataset_name_list[dataset_index]

        self.checkpoint_ready = True
        return

    def setDataset(self):
        if self.dataset_name == 'voc':
            self.num_classes = 21
            self.decode_fn = VOCSegmentation.decode_target
        elif self.dataset_name == 'cityscapes':
            self.num_classes = 19
            self.decode_fn = Cityscapes.decode_target
        return

    def initEnv(self, checkpoint):
        self.reset()

        self.setCheckPoint(checkpoint)
        if not self.checkpoint_ready:
            return

        self.setDataset()
        return

    def loadModel(self, checkpoint):
        self.initEnv(checkpoint)

        if not self.checkpoint_ready:
            return

        self.model = self.model_map[self.model_name](num_classes=self.num_classes, output_stride=self.output_stride)

        if self.separable_conv and 'plus' in self.model:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        
        if self.checkpoint is not None and os.path.isfile(checkpoint):
            checkpoint_data = torch.load(self.checkpoint, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint_data["model_state"])
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            print("Resume model from %s" % self.checkpoint)
            print("Model loaded : ", self.model_name)
            print("Dataset loaded : ", self.dataset_name)
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
        self.model_ready = True
        return

    def detect(self, image):
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0) # To tensor of NCHW
            image = image.to(self.device)
            result = self.model(image).max(1)[1].cpu().numpy()[0] # HW
        return result

    def test(self, image_folder_path, run_episode=-1, timer_skip_num=5):
        if not self.model_ready:
            print("Model not ready yet, Please loadModel or check your model path first!")
            return

        if run_episode == 0:
            print("No detect run with run_episode=0!")
            return
 
        file_name_list = os.listdir(image_folder_path)
        image_file_name_list = []
        for file_name in file_name_list:
            if file_name[-4:] in [".jpg", ".png"]:
                image_file_name_list.append(file_name)

        if run_episode < 0:
            self.resetTimer()
            timer_skipped_num = 0

            while True:
                for image_file_name in image_file_name_list:
                    image_file_path = os.path.join(image_folder_path, image_file_name)

                    self.startTimer()

                    image = Image.open(image_file_path).convert('RGB')

                    result = self.detect(image)

                    #  colorized_result = deeplabv3plus_detector.decode_fn(result).astype('uint8')
                    #  colorized_result = Image.fromarray(colorized_result)
                    #  colorized_result.save(os.path.join("./result/", image_file_name.split('.')[0] +'.png'))

                    if timer_skipped_num < timer_skip_num:
                        self.endTimer(False)
                        timer_skipped_num += 1
                    else:
                        self.endTimer()

                    print("\rNet: " + self.model_name +
                          "\tDetected: " + str(self.detected_num) +
                          "\tAvgTime: " + str(self.getAverageTimeMS()) + "ms"
                          "\tAvgFPS: " + str(self.getAverageFPS()) +
                          "    ", end="")

            print()

            return

        self.resetTimer()
        total_num = run_episode * len(image_file_name_list)
        timer_skipped_num = 0

        for i in range(run_episode):
            for image_file_name in image_file_name_list:
                image_file_path = os.path.join(image_folder_path, image_file_name)

                self.startTimer()

                image = Image.open(image_file_path).convert('RGB')

                result = self.detect(image)

                #  colorized_result = deeplabv3plus_detector.decode_fn(result).astype('uint8')
                #  colorized_result = Image.fromarray(colorized_result)
                #  colorized_result.save(os.path.join("./result/", image_file_name.split('.')[0] +'.png'))

                if timer_skipped_num < timer_skip_num:
                    self.endTimer(False)
                    timer_skipped_num += 1
                else:
                    self.endTimer()

                print("\rNet: " + self.model_name +
                      "\tDetected: " + str(self.detected_num) + "/" + str(total_num - timer_skip_num) +
                      "\t\tAvgTime: " + str(self.getAverageTimeMS()) + "ms" +
                      "\tAvgFPS: " + str(self.getAverageFPS()) +
                      "    ", end="")

        print()


if __name__ == '__main__':
    checkpoint = "./best_deeplabv3plus_resnet101_voc_os16.pth"
    image_folder_path = "./sample_images/"

    deeplabv3plus_detector = DeepLabV3PlusDetector()

    deeplabv3plus_detector.loadModel(checkpoint)

    deeplabv3plus_detector.test(image_folder_path)

