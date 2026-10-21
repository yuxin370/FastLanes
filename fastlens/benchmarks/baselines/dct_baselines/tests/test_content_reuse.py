import copy
import os
from unittest.mock import patch

from fastlens.benchmarks.baselines.dct_baselines.store import StoreReader, build_store
from .test_baselines import TemporaryTest, image


class ContentReuseTests(TemporaryTest):
    def test_payload_audit_is_reused_only_for_unchanged_identity_and_file_states(self):
        root=self.root/"store"
        build_store([image(0),image(1)],root,"ic_matched_batched",group_size=2)
        with StoreReader(root) as reader:
            evidence=reader.verify_content()
            self.assertGreater(evidence["bytes"],0)
            with patch.object(reader,"_check_content",side_effect=AssertionError("payload hash repeated")):
                reused=reader.verify_content(evidence)
            self.assertTrue(reused["reused"])
            self.assertEqual(reused["bytes"],0)
            wrong=copy.deepcopy(evidence)
            wrong["store"]["content_fingerprint"]="changed"
            with self.assertRaisesRegex(ValueError,"content or file state changed"):
                reader.verify_content(wrong)
            path=root/next(iter(reader.manifest["files"]))
            state=path.stat()
            payload=bytearray(path.read_bytes());payload[0]^=1
            path.write_bytes(payload)
            os.utime(path,ns=(state.st_atime_ns,state.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,"content or file state changed"):
                reader.verify_content(evidence)
