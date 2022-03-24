
import tensorflow as tf
if tf.__version__ > '2':
    import tensorflow.compat.v1 as tf  

import numpy as np

PATH_TO_TENSORFLOW_MODEL = 'models/savedmodel.pb'

def model_load(tf_model_path):
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(tf_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                return sess, detection_graph


def tf_inference(sess, detection_graph, img_arr):
  
    image_tensor = detection_graph.get_tensor_by_name('data_1:0')
    detection_bboxes = detection_graph.get_tensor_by_name('loc_branch_concat_1/concat:0')
    detection_scores = detection_graph.get_tensor_by_name('cls_branch_concat_1/concat:0')
    bboxes, scores = sess.run([detection_bboxes, detection_scores],
                            feed_dict={image_tensor: img_arr})

    return bboxes, scores

