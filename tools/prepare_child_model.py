"""Build the optional approximate child ensemble from pinned public ONNX files.

Only this export tool needs tokenizers==0.23.3. Inference needs no new packages.
No video, personal data or model requests are sent to a remote inference service.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import onnx
from onnx import helper, inliner, numpy_helper, TensorProto
import onnxruntime as ort

from core.onnx_runtime import cpu_options

CLIP_ROOT = 'https://huggingface.co/onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX/resolve/9463a9c508a344c837ffefe9d724f3827bf2dc79/'
AGE_URL = 'https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/resolve/8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36/mivolo_v2_age.onnx'
ASSETS = [
    ('tinyclip.onnx', CLIP_ROOT+'onnx/model_quantized.onnx', '10921310ddef06557ec1598d1260470a0a4db53f70ffe0deb60b946dcad6d27a'),
    ('tokenizer.json', CLIP_ROOT+'tokenizer.json', '6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35'),
    ('age.onnx', AGE_URL, '2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951'),
    ('LICENSE.child_vision.txt', 'https://raw.githubusercontent.com/microsoft/Cream/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/LICENSE', '4ecaa5a5bc48d91c899216ba3ab8d9fccbb543e106b75f5792e455631d599ae5'),
    ('LICENSE.child_age.txt', 'https://huggingface.co/iitolstykh/mivolo_v2/resolve/57d119f0fe72a6831dbc28c53c7b1b06873a469a/LICENSE', '0d20b855a292bab4204ee5a5c5d34db0634b22b90a291dd0d23be0d476fdc61c'),
]
PROMPTS = [
    'a photo of a baby', 'a photo of a toddler', 'a photo of a small preschool child', 'a photo of a very young child',
    'a photo of an adult man', 'a photo of an adult woman', 'a photo of an elderly person', 'a photo of a teenager',
    'a photo of a school-aged child', 'a photo of a person with their face hidden', 'a close up photo of winter clothing',
]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(cache, output):
    from tokenizers import Tokenizer
    cache, output = Path(cache), Path(output)
    names = ['child_age_v1.onnx', 'child_vision_v1.onnx', 'child_filter_v1.json']
    if any((output/name).exists() for name in names):
        raise FileExistsError('Child model output already exists; use a separate export directory')
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, url, expected in ASSETS:
        path = cache/name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as source, path.open('wb') as target:
                while chunk := source.read(1024*1024):
                    target.write(chunk)
        if digest(path) != expected:
            raise ValueError(f'Source checksum differs: {name}')
    onnx.utils.extract_model(str(cache/'tinyclip.onnx'), str(output/'child_vision_v1.onnx'),
                             ['pixel_values'], ['image_embeds'])
    # Store weights in FP16, then cast constants back to FP32. Computation
    # remains FP32 on CPU; unlike INT8 this avoids large age-estimate drift.
    model = inliner.inline_local_functions(onnx.load(str(cache/'age.onnx')))
    del model.graph.value_info[:]
    casts = []
    for i, tensor in enumerate(list(model.graph.initializer)):
        if tensor.data_type == TensorProto.FLOAT and np.prod(tensor.dims) > 16:
            original = tensor.name
            packed = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), original+'__storage16')
            model.graph.initializer[i].CopyFrom(packed)
            casts.append(helper.make_node('Cast', [packed.name], [original], to=TensorProto.FLOAT,
                                          name=original+'__restore_float32'))
    nodes = [*casts, *model.graph.node]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(output/'child_age_v1.onnx'))
    tokenizer = Tokenizer.from_file(str(cache/'tokenizer.json'))
    tokenizer.enable_padding(pad_id=1, pad_token='<|endoftext|>')
    tokens = tokenizer.encode_batch(PROMPTS)
    session = ort.InferenceSession(str(cache/'tinyclip.onnx'), sess_options=cpu_options(1),
                                   providers=['CPUExecutionProvider'])
    inputs = {'input_ids': np.asarray([t.ids for t in tokens], np.int64),
              'attention_mask': np.asarray([t.attention_mask for t in tokens], np.int64),
              'pixel_values': np.zeros((1, 3, 224, 224), np.float32)}
    embeddings = session.run(['text_embeds'], inputs)[0]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    manifest = {
        'format': 'approx_child_ensemble_v1', 'status': 'experimental',
        'description': 'Approximate preschool-child appearance, not verified chronological age. Unknown passengers count.',
        'vision': {'file': 'child_vision_v1.onnx', 'sha256': digest(output/'child_vision_v1.onnx')},
        'age': {'file': 'child_age_v1.onnx', 'sha256': digest(output/'child_age_v1.onnx')},
        'vision_mean': [.48145466, .4578275, .40821073], 'vision_std': [.26862954, .26130258, .27577711],
        'prompts': PROMPTS, 'embeddings': embeddings.tolist(),
        'minimum_child_margin': .005, 'maximum_child_age': 7.0, 'adult_veto_age': 12.0,
        'source_assets': [{'file': name, 'url': url, 'sha256': sha} for name, url, sha in ASSETS],
        'licenses': {'vision': 'MIT', 'age': 'Apache-2.0'},
        'age_license_revision': 'https://huggingface.co/iitolstykh/mivolo_v2/blob/57d119f0fe72a6831dbc28c53c7b1b06873a469a/LICENSE',
    }
    (output/'child_filter_v1.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode())
    for name in ['LICENSE.child_vision.txt', 'LICENSE.child_age.txt']:
        (output/name).write_bytes((cache/name).read_bytes())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.local/child_model_sources'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.cache, args.output)
    print(json.dumps({k: result[k] for k in ['format', 'status', 'vision', 'age']}, indent=2))
