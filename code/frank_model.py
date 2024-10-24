import torch
from ultralytics import YOLOv10
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List

class Digit:
  def __init__(self, label, conf, coords):
    self.label = label
    self.conf = conf
    self.coords = coords

class Result():
    def __init__(self, result1, result2, number, image):
      self.result1 = result1
      self.result2 = result2
      self.number = number
      self.image = image

class Frank(torch.nn.Module):
    def __init__(self, model1_path=None, model2_path=None):
        super(Frank, self).__init__()
        self.model1 = YOLOv10(model1_path) if model1_path else YOLOv10()
        self.model2 = YOLOv10(model2_path) if model2_path else YOLOv10()

    def forward(self, image_path):
        image = cv2.imread(image_path)
        result1 = self.model1(image)[0]
        cropped_image = self.crop(result1, image)
        turned_over_image = self.turn_over(cropped_image)
        pred1 = self.model2(cropped_image)[0]
        pred2 = self.model2(turned_over_image)[0]
        digits1 = self.get_digits(pred1)
        digits2 = self.get_digits(pred2)

        if digits1 is None or digits2 is None: # если цифры не обнаружены
            result = Result(result1, pred1, None, cropped_image)
            return result

        # first = True - вероятна исходная картинка, False - вероятна перевернутая
        number, first = self.get_number(digits1, digits2)
        if first:
            result = Result(result1, pred1, number, cropped_image)
        else:
            result = Result(result1, pred2, number, turned_over_image)

        return result


    def train(self, config_path1, config_path2, epochs=20, batch=32):
        print(f"Training model 1 with {config_path1}")
        results_model1 = self.model1.train(data=config_path1, epochs=epochs, batch=batch)

        print(f"Training model 2 with {config_path2}")
        results_model2 = self.model2.train(data=config_path2, epochs=epochs, batch=batch)

        return results_model1, results_model2

    def from_pretrained(self, path1=None, path2=None):
        if path1:
            self.model1 = YOLOv10.from_pretrained(path1)
        if path2:
            self.model2 = YOLOv10.from_pretrained(path2)

    def state_dict(self):
        return self.model1.state_dict(), self.model2.state_dict()

    def load_state_dict(self, state_dict):
        self.model1.load_state_dict(state_dict[0])
        self.model2.load_state_dict(state_dict[1])

    def get_bbox_vertices(self, bbox_tensor: torch.Tensor):
        bbox_tensor = bbox_tensor.cpu().detach() if bbox_tensor.is_cuda else bbox_tensor
        x_min, y_min, x_max, y_max = bbox_tensor.tolist()[0]
        top_left = [x_min, y_min]
        top_right = [x_max, y_min]
        bottom_right = [x_max, y_max]
        bottom_left = [x_min, y_max]

        return np.array([top_left, top_right, bottom_right, bottom_left]).reshape((-1, 1, 2)).astype(np.int32)

    def crop(self, result_first, image):
        if len(result_first) == 0:  # Ничего не обнаружено
            print('Область не обнаружена!')
            return image
        location = result_first[0].boxes.xyxy
        location = self.get_bbox_vertices(location)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)

        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        width = x2 - x1
        height = y2 - y1
        new_x1, new_y1 = x1 - width // 2, y1 - height // 1.8 #TODO: Сделать гиперпараметрами
        new_x2, new_y2 = x2 + width // 2, y2 + height // 1.8

        if new_x1 >= 0:
            x1 = int(new_x1)
        if new_y1 >= 0:
            y1 = int(new_y1)
        if new_x2 <= image.shape[0]:
            x2 = int(new_x2)
        if new_y2 <= image.shape[1]:
            y2 = int(new_y2)

        cropped_image = image[x1:x2+1, y1:y2+1]

        return cropped_image

    def turn_over(self, image): # Додумать. Бывает, что у перевернутой выше conf, хотя это неправильно
        # return cv2.rotate(image, cv2.ROTATE_180)
        return image

    def get_digits(self, results):
        labels = results.boxes.cls.cpu().detach()
        confs = results.boxes.conf.cpu().detach()
        coords = results.boxes.xyxy.cpu().detach()
        if len(labels) == 0: # если цифры не обнаружены
            return None

        digits = [Digit(int(label.cpu().detach().item()),
                      conf.cpu().detach().numpy(),
                      coord.cpu().detach().numpy()) for (label, conf, coord) in zip(labels, confs, coords)]

        return digits

    def get_number(self, digits_1: List[Digit], digits_2: List[Digit]) -> str: # результат для изображения и его отображения на 180
        if self.get_conf(digits_1) > self.get_conf(digits_2): # какая из ориентаций более вероятна (0 или 180)
            digits = digits_1
            first = True # выбираем первую картинку
        else:
            digits = digits_2
            first = False # выбираем второую картинку

        result = self.read_number(digits, self.is_vertical(digits))
        return result, first

    def get_conf(self, digits):
        return np.prod([digit.conf for digit in digits])

    def read_number(self, digits: List[Digit], vertical: bool) -> str:
        if not vertical: # обычная горизонтальная ориентация фотографии
            sorted_digits = sorted(digits, key=lambda d: d.coords[0])
        else: # вертикальная ориентация
            sorted_digits = sorted(digits, key=lambda d: d.coords[1])
        return ''.join(map(str, [digit.label for digit in sorted_digits]))

    def is_vertical(self, digits: Digit) -> bool:
        x_mins = [digit.coords[0] for digit in digits]
        y_mins = [digit.coords[1] for digit in digits]

        x_range = max(x_mins) - min(x_mins)
        y_range = max(y_mins) - min(y_mins)

        if y_range > x_range:
            return True
        return False