import tensorflow as tf
from core.backbone import darknet53
from core.yolov3 import YOLOV3
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


with tf.Graph().as_default() as graph:
    image=tf.placeholder(dtype=tf.float32,shape=[1,544,544,3],name="input")
    out=YOLOV3(image,True,1)
    stats_graph(graph)