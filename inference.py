import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import argparse
from train import SymbolClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes_mapping = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'),
                   ord('6'), ord('7'), ord('8'), ord('9'), ord('A'), ord('B'),
                   ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'),
                   ord('J'), ord('K'), ord('L'), ord('M'), ord('N'), ord('P'),
                   ord('R'), ord('S'), ord('T'), ord('U'), ord('V'), ord('W'),
                   ord('X'), ord('Y'), ord('Z')]


def inference_images_from_folder(folder_path):
    model_path = './model.pth'

    model = SymbolClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path).convert('L')

            pixels = np.array(image).T
            pixels = pixels.flatten().tolist()
            pixels = pd.Series(pixels)
            image_tensor = pixels.values.astype(np.float32).reshape(28, 28)
            image_tensor = torch.tensor(image_tensor) / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_class = torch.max(output, 1)

            results.append((predicted_class.item(), file_path))

    for predicted_class, file_path in results:
        print(f"{classes_mapping[predicted_class]}, {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input', type=str, help='Path to folder with images', required=True)
    args = parser.parse_args()
    folder_path = args.input
    inference_images_from_folder(folder_path)
