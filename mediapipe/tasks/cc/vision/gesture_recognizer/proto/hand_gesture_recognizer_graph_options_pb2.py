# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2
from mediapipe.framework import calculator_options_pb2 as mediapipe_dot_framework_dot_calculator__options__pb2
from mediapipe.tasks.cc.core.proto import base_options_pb2 as mediapipe_dot_tasks_dot_cc_dot_core_dot_proto_dot_base__options__pb2
from mediapipe.tasks.cc.vision.gesture_recognizer.proto import gesture_classifier_graph_options_pb2 as mediapipe_dot_tasks_dot_cc_dot_vision_dot_gesture__recognizer_dot_proto_dot_gesture__classifier__graph__options__pb2
from mediapipe.tasks.cc.vision.gesture_recognizer.proto import gesture_embedder_graph_options_pb2 as mediapipe_dot_tasks_dot_cc_dot_vision_dot_gesture__recognizer_dot_proto_dot_gesture__embedder__graph__options__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n^mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.proto\x12/mediapipe.tasks.vision.gesture_recognizer.proto\x1a$mediapipe/framework/calculator.proto\x1a,mediapipe/framework/calculator_options.proto\x1a\x30mediapipe/tasks/cc/core/proto/base_options.proto\x1aYmediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options.proto\x1aWmediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_embedder_graph_options.proto\"\xde\x04\n!HandGestureRecognizerGraphOptions\x12=\n\x0c\x62\x61se_options\x18\x01 \x01(\x0b\x32\'.mediapipe.tasks.core.proto.BaseOptions\x12t\n\x1egesture_embedder_graph_options\x18\x02 \x01(\x0b\x32L.mediapipe.tasks.vision.gesture_recognizer.proto.GestureEmbedderGraphOptions\x12\x7f\n\'canned_gesture_classifier_graph_options\x18\x03 \x01(\x0b\x32N.mediapipe.tasks.vision.gesture_recognizer.proto.GestureClassifierGraphOptions\x12\x7f\n\'custom_gesture_classifier_graph_options\x18\x04 \x01(\x0b\x32N.mediapipe.tasks.vision.gesture_recognizer.proto.GestureClassifierGraphOptions2\x81\x01\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xd4\xf1\xf9\xdc\x01 \x01(\x0b\x32R.mediapipe.tasks.vision.gesture_recognizer.proto.HandGestureRecognizerGraphOptionsBc\n9com.google.mediapipe.tasks.vision.gesturerecognizer.protoB&HandGestureRecognizerGraphOptionsProto')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.tasks.cc.vision.gesture_recognizer.proto.hand_gesture_recognizer_graph_options_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_HANDGESTURERECOGNIZERGRAPHOPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n9com.google.mediapipe.tasks.vision.gesturerecognizer.protoB&HandGestureRecognizerGraphOptionsProto'
  _globals['_HANDGESTURERECOGNIZERGRAPHOPTIONS']._serialized_start=462
  _globals['_HANDGESTURERECOGNIZERGRAPHOPTIONS']._serialized_end=1068
# @@protoc_insertion_point(module_scope)
