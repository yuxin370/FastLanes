import unittest
import numpy as np
from PIL import Image
import torch
from galp.benchmarks.system_dct_major.pipeline import _normal_rgb_resize_crop


class NormalPreprocessTest(unittest.TestCase):
    def test_bicubic_checkpoint_preprocessing(self):
        torch.set_num_threads(1)
        pixels=np.random.default_rng(123).integers(0,256,(512,512,3),dtype=np.uint8)
        expected=Image.fromarray(pixels).resize((256,256),Image.Resampling.BICUBIC).crop((16,16,240,240))
        expected=torch.from_numpy(np.array(expected)).permute(2,0,1).float()/127.5-1
        actual=_normal_rgb_resize_crop(torch.from_numpy(pixels).permute(2,0,1)[None],
            {'preprocess':{'rgb':{'resize_shorter':256,'crop_size':[224,224], 'torch_interpolation':'bicubic'}}})[0]
        self.assertLessEqual(float((actual-expected).abs().max()),.25)
        self.assertLess(float((actual-expected).abs().mean()),.008)

    def test_matches_checkpoint_pil_geometry_and_antialiasing(self):
        torch.set_num_threads(1)
        pixels=np.random.default_rng(123).integers(0,256,(512,512,3),dtype=np.uint8)
        expected=Image.fromarray(pixels).resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
        expected=torch.from_numpy(np.array(expected)).permute(2,0,1).float()/127.5-1
        actual=_normal_rgb_resize_crop(torch.from_numpy(pixels).permute(2,0,1)[None],
            {'preprocess':{'rgb':{'resize_shorter':256,'crop_size':[224,224]}}})[0]
        self.assertEqual(tuple(actual.shape),(3,224,224))
        # PIL rounds its intermediate separable pass; tensor interpolation rounds once.
        self.assertLessEqual(float((actual-expected).abs().max()),1/127.5+1e-6)
        self.assertLess(float((actual-expected).abs().mean()),.008)


if __name__=='__main__': unittest.main()
