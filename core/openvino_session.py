"""Optional FP32 person-detector adapter; no OpenVINO import on the default path."""
from types import SimpleNamespace

import numpy as np


class OpenVINOSession:
    """Synchronous adapter preserving the existing ONNX preprocessing and NMS."""

    def __init__(self, model, threads):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError('OpenVINO is optional; use the CPU benchmark image or install openvino==2026.4.1') from exc
        core = ov.Core()
        self.compiled = core.compile_model(str(model), 'CPU', {
            'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': threads,
            'NUM_STREAMS': 1, 'PERFORMANCE_HINT': 'LATENCY',
            # Multiple archive workers must not pin their caller/main thread
            # to the same first core, also slowing the following ReID/vest work.
            'ENABLE_CPU_PINNING': False,
        })
        self.request = self.compiled.create_infer_request()
        self.inputs = [SimpleNamespace(name=p.get_any_name()) for p in self.compiled.inputs]
        self.outputs = [SimpleNamespace(name=p.get_any_name()) for p in self.compiled.outputs]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, inputs):
        if output_names not in (None, [row.name for row in self.outputs]):
            raise ValueError('Unexpected detector output names')
        self.request.infer(inputs)
        # Requests reuse output buffers; callers must own their frame's result.
        return [np.array(self.request.get_output_tensor(i).data, copy=True)
                for i in range(len(self.outputs))]
