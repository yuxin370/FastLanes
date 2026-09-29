import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from galp.benchmarks.system_dct_major.pipeline import CoorDLAdapter, _normal_rgb_resize_crop


class NormalPreprocessTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CoorDL preprocessing requires CUDA")
    def test_coordl_jpeg_chroma_edges_match_pil(self):
        try:
            from nvidia.dali import backend
        except ImportError:
            self.skipTest("requires the CoorDL Python environment")

        if "cache_size" not in backend.GetSchema("FileReader").GetArgumentNames():
            self.skipTest("requires the CoorDL Python environment")
        # Saturated chroma edges expose the legacy mixed JPEG decoder's error.
        rng = np.random.default_rng(123)
        pixels = rng.integers(0, 2, (128, 128, 3), dtype=np.uint8) * 255
        pixels = pixels.repeat(4, axis=0).repeat(4, axis=1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chroma.jpg"
            Image.fromarray(pixels).save(path, quality=95, subsampling=2)
            with Image.open(path) as image:
                expected = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                expected = np.array(expected.crop((16, 16, 240, 240)))
            expected = torch.from_numpy(expected).permute(2, 0, 1).float() / 127.5 - 1
            samples = [{"path": str(path), "ordinal": 0, "label": 0}]
            contract = {
                "execution": {"batch_size": 1, "workers": 1, "seed": 17},
                "preprocess": {"rgb": {
                    "resize_shorter": 256, "crop_size": [224, 224], "torch_interpolation": "bicubic",
                }},
                "pipelines": {"coordl": {
                    "file_list": str(Path(temporary) / "files.txt"), "cache_size": 1,
                    "device_id": 0, "prefetch_queue_depth": 2,
                }},
            }
            adapter = CoorDLAdapter(contract, samples, torch.device("cuda"), "coordl")
            try:
                adapter.begin_repeat()
                actual = adapter.load(samples).inputs[0][0].cpu()
                difference = (actual - expected).abs()
                self.assertLessEqual(float(difference.max()), .25)
                self.assertLessEqual(float(difference.mean()), .008)
            finally:
                adapter.close()

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
