"""Upstream-referenced RGB weights with the canonical RGB execution graph.

The DCTNet ResNet constructor has no forward and changes layer2 stride to 1.
Restore the RGB graph only in this experiment; the DCT model stays unchanged.
"""
import io
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset

import backend as B
if B.PROFILE == "efun":
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
    CHECKPOINT = Path(os.environ.get("EFUN_RGB_CHECKPOINT", B.DEFAULT_CHECKPOINT_DIR /
                                    "efficientnet_b0_rgb_official" / WEIGHTS.url.rsplit("/", 1)[-1]))
    INTERPOLATION = T.InterpolationMode.BICUBIC
else:
    from models.imagenet.resnet import ResNet, Bottleneck
    from models.imagenet.mobilenetv2 import MobileNetV2
    CHECKPOINT = B.DEFAULT_CHECKPOINT_DIR / (
        "mobilenetv2_rgb_official/mobilenetv2_1.0-0c6065bc.pth" if B.PROFILE.startswith("mobilenet")
        else "resnet50_rgb_official/resnet50-19c8e357.pth")
    INTERPOLATION = T.InterpolationMode.BILINEAR
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


if B.PROFILE != "efun":
    class RGBResNet(ResNet):
        def __init__(self):
            super().__init__(Bottleneck, [3, 4, 6, 3])
            self.layer2[0].conv2.stride = (2, 2)
            self.layer2[0].downsample[0].stride = (2, 2)
            self.layer2[0].stride = 2

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(torch.flatten(self.avgpool(x), 1))


    class RGBMobileNetV2(MobileNetV2):
        def __init__(self):
            # The upstream default removes a stride for DCT. upscale=True retains
            # the canonical RGB strides; the official base class lacks forward.
            super().__init__(upscale=True)

        def forward(self, x):
            x = self.avgpool(self.conv(self.features(x)))
            return self.classifier(torch.flatten(x, 1))


def model():
    if B.PROFILE == "efun":
        net = efficientnet_b0(weights=None)
    else:
        net = RGBMobileNetV2() if B.PROFILE.startswith("mobilenet") else RGBResNet()
    # Legacy DCTNet RGB downloads use PyTorch tar files, requiring weights_only=False.
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=B.PROFILE == "efun")
    net.load_state_dict(state, strict=True)
    return net.eval().requires_grad_(False)


class Inputs(Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.transform = (WEIGHTS.transforms() if B.PROFILE == "efun" else
                          T.Compose([T.Resize(256, interpolation=INTERPOLATION),
                                     T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        e = self.entries[index]
        t = time.perf_counter()
        blob = Path(e["path"]).read_bytes()
        read = time.perf_counter() - t
        t = time.perf_counter()
        with Image.open(io.BytesIO(blob)) as image:
            image = image.convert("RGB")
        decode = time.perf_counter() - t
        t = time.perf_counter()
        x = self.transform(image)
        return x, e["model_label"], e["ordinal"], read, decode, time.perf_counter()-t, len(blob)


def worker_init(_):
    torch.set_num_threads(1)


def dali_loader(entries, batch_size, workers, prefetch_queue_depth=2):
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    @pipeline_def
    def pipeline():
        encoded, ordinal = fn.readers.file(files=[s["path"] for s in entries],
                                          labels=[s["ordinal"] for s in entries],
                                          random_shuffle=False, pad_last_batch=False, name="Reader")
        images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
        images = fn.resize(images, device="gpu", resize_shorter=256,
                           interp_type=types.INTERP_CUBIC if INTERPOLATION == T.InterpolationMode.BICUBIC
                           else types.INTERP_LINEAR, antialias=True)
        images = fn.crop_mirror_normalize(images, device="gpu", dtype=types.FLOAT,
                                         output_layout="CHW", crop=(224,224),
                                         crop_pos_x=0.499999, crop_pos_y=0.499999,
                                         mean=[255*v for v in MEAN], std=[255*v for v in STD])
        return images, ordinal

    pipe = pipeline(batch_size=batch_size, num_threads=workers, device_id=0,
                    seed=11997733, prefetch_queue_depth=prefetch_queue_depth)
    pipe.build()
    return DALIGenericIterator([pipe], output_map=["image", "ordinal"], reader_name="Reader",
                               auto_reset=False, last_batch_policy=LastBatchPolicy.PARTIAL)
