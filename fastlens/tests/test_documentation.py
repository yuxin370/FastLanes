"""Check public Markdown links, anchors, wording and introductory Python examples."""

import html
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
INLINE_LINK = re.compile(r"\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)


def prose(source):
    return FENCES.sub("", source)


def anchors(source):
    source = prose(source)
    result = set(re.findall(r'<[^>]+\b(?:id|name)=[\"\']([^\"\']+)[\"\']', source))
    for match in re.finditer(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", source, re.MULTILINE):
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
        heading = html.unescape(re.sub(r"<[^>]+>", "", heading))
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        candidate, suffix = slug, 0
        while candidate in result:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        result.add(candidate)
    return result


def links(source):
    source = re.sub(r"`[^`\n]+`", "", prose(source))
    return [m[1].strip("<>") for pattern in (INLINE_LINK, REFERENCE)
            for m in pattern.finditer(source)]


def public_files():
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    ).decode().split("\0")
    return {Path(name) for name in names if name and (ROOT / name).is_file()}


def link_errors(document, source, files):
    errors = []
    for link in links(source):
        target = urlsplit(link)
        if target.scheme or target.netloc:
            continue
        path = unquote(target.path)
        resolved = ((ROOT / path.lstrip("/")) if path.startswith("/") else
                    (ROOT / document.parent / path) if path else ROOT / document).resolve()
        if not resolved.is_relative_to(ROOT):
            errors.append(f"{document}: link leaves repository: {link}")
            continue
        relative = resolved.relative_to(ROOT)
        if relative not in files and not any(relative in file.parents for file in files):
            errors.append(f"{document}: missing published target: {link}")
        elif target.fragment and resolved.suffix.lower() == ".md":
            if unquote(target.fragment) not in anchors(resolved.read_text()):
                errors.append(f"{document}: missing heading anchor: {link}")
    return errors


def introductory_examples():
    for name in ("README.md", "fastlens/ARCHITECTURE.md", "fastlens/docs/DIRECT_DCT.md"):
        source = (ROOT / name).read_text()
        for match in FENCES.finditer(source):
            if match[1] == "python" and re.search(
                r"^from fastlens\.torch import (?:DirectDctReader|DirectDctPipeline)\b",
                match[2], re.MULTILINE,
            ):
                yield f"{name}:{source[:match.start()].count(chr(10)) + 2}", match[2]


def hygiene_errors(source):
    # Match internal narrative, not technical uses of submission, phases or legacy APIs.
    patterns = (
        r"\breviewers? (?:should|guide|entry point|reproduction)\b",
        r"\bexcluded from (?:this|the) submission\b",
        r"\b(?:in this session|the user's H100|the sandbox itself|supplied plan|final fairness revision)\b",
        r"\b(?:after the user|the user confirmed)\b",
        r"\bPhase[- ]\d+\b",
    )
    errors = [m.group() for pattern in patterns for m in re.finditer(pattern, prose(source), re.I)]
    errors += re.findall(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", source, re.I)
    errors += re.findall(r"/(?:home/[^/\s`]+/tmp|tmp|var/tmp)/[^\s`\"']*\d{8}[^\s`\"']*", source)
    return errors


class DocumentationTests(unittest.TestCase):
    def test_public_links_and_anchors(self):
        files = public_files()
        documents = sorted(file for file in files if file.suffix.lower() == ".md"
                           and (len(file.parts) == 1 or file.parts[0] == "fastlens")
                           and "upstream" not in file.parts)
        errors = [error for document in documents
                  for error in link_errors(document, (ROOT / document).read_text(), files)]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_user_guides_avoid_internal_narrative(self):
        files = public_files()
        # Manuscript build notes retain experiment provenance.
        documents = sorted(file for file in files if file.suffix.lower() == ".md"
                           and (len(file.parts) == 1 or file.parts[0] == "fastlens")
                           and "upstream" not in file.parts
                           and not file.is_relative_to("fastlens/docs/paper"))
        errors = [f"{document}: {phrase}" for document in documents
                  for phrase in hygiene_errors((ROOT / document).read_text())]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_introductory_python_syntax(self):
        examples = list(introductory_examples())
        self.assertTrue(examples)
        for location, code in examples:
            with self.subTest(location=location):
                compile(code, location, "exec")

    def test_introductory_examples_exclude_application_fragments(self):
        reader = "from fastlens.torch import DirectDctReader\n"
        pipeline = "from fastlens.torch import DirectDctPipeline\n"
        fragment = ("for batch in batches:  # Your DirectDctReader iterator.\n"
                    "    model(batch.y, batch.cbcr)\n")
        source = "".join(f"```python\n{code}```\n" for code in (reader, pipeline, fragment))
        with patch.object(Path, "read_text", return_value=source):
            examples = list(introductory_examples())
        self.assertEqual({code for _, code in examples}, {reader, pipeline})

    def test_links_ignore_code_and_accept_reference_destinations(self):
        self.assertEqual(links('`[sample](missing.md)`\n```text\n[x](no.md)\n```\n'
                               '[guide](guide.md#install)\n[ref]: <a%20b.md>\n'),
                         ['guide.md#install', 'a%20b.md'])
        errors = link_errors(Path('README.md'), '[x](missing-evidence.json)', {Path('README.md')})
        self.assertIn('missing published target', errors[0])
        errors = link_errors(Path('README.md'), '[x](#1-install-dependencies)', {Path('README.md')})
        self.assertIn('missing heading anchor', errors[0])

    def test_heading_slugs_preserve_duplicate_and_formatted_titles(self):
        self.assertEqual(anchors('# **Install** `API`!\n## Install API\n'
                                 '```text\n# Hidden\n```\n<a id="manual"></a>\n'),
                         {'install-api', 'install-api-1', 'manual'})

    def test_hygiene_distinguishes_technical_terms(self):
        self.assertEqual(hygiene_errors('Asynchronous submission and legacy APIs.\n'
                                        '```bash\n--phase all\n```'), [])
        self.assertTrue(hygiene_errors('In this session we completed Phase-4.'))
        self.assertTrue(hygiene_errors('After the user confirmed the GPU was idle, measurements completed.'))
        self.assertTrue(hygiene_errors('CUDA_VISIBLE_DEVICES=GPU-12345678-1234-1234-1234-123456789abc'))


if __name__ == "__main__":
    unittest.main()
