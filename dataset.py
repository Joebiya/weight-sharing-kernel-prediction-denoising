import numpy as np
import os
from tqdm import tqdm
import pyexr
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def Padding(img, w):
    return np.pad(img, ((w, w), (w, w), (0, 0)))


class DataBase:
    def __init__(self, crop_size=128):
        folder_name = os.path.join("dataset")
        train_set = "/data/hjy/exrset_test/output"
        scene_names = []
        # scene_num = 1024
        scene_num = 9
        for scene in os.listdir(train_set):
            scene_names.append(scene)

        img_num_per_scene = 64
        albedo_file_names = []
        irradiance_file_names = []
        reference_file_names = []
        normal_file_names = []
        depth_file_names = []
        
        for i in tqdm(range(len(scene_names)), desc="Loading scenes"):
            scene_name = scene_names[i]
            scene_path = os.path.join(train_set, scene_name)
            for frame in os.listdir(scene_path):
                frame_path = os.path.join(scene_path, frame)
                albedo_file_names.append(os.path.join(frame_path, "diffuse.exr"))
                irradiance_file_names.append(os.path.join(frame_path, "color.exr"))
                reference_file_names.append(os.path.join(frame_path, "reference.exr"))
                normal_file_names.append(os.path.join(frame_path, "normal.exr"))
                depth_file_names.append(os.path.join(frame_path, "position.exr"))

        self.train_inputs, self.train_targets = [], []
        self.test_inputs, self.test_targets = [], []
        for i in tqdm(range(len(reference_file_names))):
            irradiance_img = pyexr.read(irradiance_file_names[i])[
                :, :, :3
            ]  # ignore alpha channel
            albedo_img = pyexr.read(albedo_file_names[i])
            reference_img = pyexr.read(reference_file_names[i])
            normal_img = pyexr.read(normal_file_names[i])
            normal_img = normal_img * 0.5 + 0.5
            depth_img = pyexr.read(depth_file_names[i])[:, :, 0:1]
            depth_img = (depth_img - np.min(depth_img)) / (
                np.max(depth_img) - np.min(depth_img)
            )

            if i < 60 * (len(scene_names) - 1):
                inputs = np.concatenate(
                    (
                        Padding(irradiance_img, crop_size),
                        Padding(albedo_img, crop_size),
                        Padding(normal_img, crop_size),
                        Padding(depth_img, crop_size),
                    ),
                    axis=2,
                )
                targets = Padding(reference_img, crop_size)

                self.train_inputs.append(inputs)
                self.train_targets.append(targets)
            else:
                inputs = np.concatenate(
                    (irradiance_img, albedo_img, normal_img, depth_img), axis=2
                )
                targets = reference_img

                self.test_inputs.append(inputs)
                self.test_targets.append(targets)

        H, W, _ = self.test_targets[0].shape
        self.img_h, self.img_w = H - crop_size, W - crop_size


class BMFRFullResAlDataset(Dataset):
    def __init__(
        self,
        database,
        use_train=False,
        use_val=False,
        use_test=False,
        train_crops_every_frame=77,
        val_crops_every_frame=20,
        crop_size=128,
    ):  # BMFR
        self.database = database
        self.use_train = use_train
        self.use_val = use_val
        self.use_test = use_test
        self.train_crops_every_frame = train_crops_every_frame
        self.val_crops_every_frame = val_crops_every_frame
        self.crop_size = crop_size

        def rotate90(inputs):
            inputs = torch.rot90(inputs, 1, (1, 2))
            return inputs

        def rotate270(inputs):
            inputs = torch.rot90(inputs, -1, (1, 2))
            return inputs

        self.transforms = [TF.hflip, TF.vflip, rotate90, rotate270]

    def _apply_transform(self, input_img, target_img):
        if self.use_train or self.use_val:
            # Random crop and convert ndarray to tensor
            i, j = np.random.randint(
                self.database.img_h - self.crop_size
            ), np.random.randint(self.database.img_w - self.crop_size)
            input_crop = TF.to_tensor(
                input_img[i : i + self.crop_size, j : j + self.crop_size].astype(
                    np.float32
                )
            )
            target_crop = TF.to_tensor(
                target_img[i : i + self.crop_size, j : j + self.crop_size].astype(
                    np.float32
                )
            )

            if np.random.rand() > 0.5:
                transform = np.random.choice(self.transforms)
                input_crop = transform(input_crop)
                target_crop = transform(target_crop)
        elif self.use_test:
            input_crop = TF.to_tensor(input_img.astype(np.float32))
            target_crop = TF.to_tensor(target_img.astype(np.float32))

        return input_crop, target_crop

    def __getitem__(self, idx):
        if self.use_test:
            frame_idx = idx
            return self._apply_transform(
                self.database.test_inputs[frame_idx],
                self.database.test_targets[frame_idx],
            )
        elif self.use_train:
            frame_idx = idx // self.train_crops_every_frame
        elif self.use_val:
            frame_idx = idx // self.val_crops_every_frame
        return self._apply_transform(
            self.database.train_inputs[frame_idx],
            self.database.train_targets[frame_idx],
        )

    def __len__(self):
        if self.use_train:
            return len(self.database.train_targets) * self.train_crops_every_frame
        elif self.use_val:
            return len(self.database.train_targets) * self.val_crops_every_frame
        elif self.use_test:
            return len(self.database.test_targets)
