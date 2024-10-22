# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/tasks/cc/genai/inference/calculators/llm_gpu_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.tasks.cc.genai.inference.proto import llm_file_metadata_pb2 as mediapipe_dot_tasks_dot_cc_dot_genai_dot_inference_dot_proto_dot_llm__file__metadata__pb2
from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2 as mediapipe_dot_tasks_dot_cc_dot_genai_dot_inference_dot_proto_dot_llm__params__pb2
from mediapipe.tasks.cc.genai.inference.proto import sampler_params_pb2 as mediapipe_dot_tasks_dot_cc_dot_genai_dot_inference_dot_proto_dot_sampler__params__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nGmediapipe/tasks/cc/genai/inference/calculators/llm_gpu_calculator.proto\x12\x10odml.infra.proto\x1a@mediapipe/tasks/cc/genai/inference/proto/llm_file_metadata.proto\x1a\x39mediapipe/tasks/cc/genai/inference/proto/llm_params.proto\x1a=mediapipe/tasks/cc/genai/inference/proto/sampler_params.proto\"\xf8\x04\n\x17LlmGpuCalculatorOptions\x12\x13\n\x0bweight_path\x18\x01 \x01(\t\x12N\n\x0egpu_model_info\x18\n \x01(\x0b\x32\x36.odml.infra.proto.LlmGpuCalculatorOptions.GpuModelInfo\x12\x19\n\x11num_decode_tokens\x18\x0c \x01(\x05\x12\x1b\n\x13sequence_batch_size\x18\x0e \x01(\x05\x12\x11\n\tlora_path\x18\x13 \x01(\t\x12\x37\n\x0ellm_parameters\x18\x14 \x01(\x0b\x32\x1f.odml.infra.proto.LlmParameters\x12\x18\n\x10num_output_heads\x18\x16 \x01(\x05\x12\x12\n\nlora_ranks\x18\x1d \x03(\x05\x12;\n\x0esampler_params\x18\x1f \x01(\x0b\x32#.odml.infra.proto.SamplerParameters\x1a\xc0\x01\n\x0cGpuModelInfo\x12\x1c\n\x14\x61llow_precision_loss\x18\x01 \x01(\x08\x12\x1a\n\x12\x65nable_fast_tuning\x18\x02 \x01(\x08\x12\x1b\n\x13\x65nable_winograd_opt\x18\x03 \x01(\x08\x12\x15\n\ruse_low_power\x18\x04 \x01(\x08\x12\x1e\n\x16prefer_texture_weights\x18\x05 \x01(\x08\x12\"\n\x1a\x65nable_host_mapped_pointer\x18\x06 \x01(\x08J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x07\x10\x08J\x04\x08\x08\x10\tJ\x04\x08\t\x10\nJ\x04\x08\x0b\x10\x0cJ\x04\x08\r\x10\x0eJ\x04\x08\x17\x10\x1cJ\x04\x08\x1e\x10\x1f\x42;\n\x1b\x63om.google.odml.infra.protoB\x1cLlmGpuCalculatorOptionsProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.tasks.cc.genai.inference.calculators.llm_gpu_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\033com.google.odml.infra.protoB\034LlmGpuCalculatorOptionsProto'
  _globals['_LLMGPUCALCULATOROPTIONS']._serialized_start=282
  _globals['_LLMGPUCALCULATOROPTIONS']._serialized_end=914
  _globals['_LLMGPUCALCULATOROPTIONS_GPUMODELINFO']._serialized_start=650
  _globals['_LLMGPUCALCULATOROPTIONS_GPUMODELINFO']._serialized_end=842
# @@protoc_insertion_point(module_scope)
