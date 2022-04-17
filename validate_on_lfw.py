import os,math,cv2
import numpy as np
import tensorflow
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn import metrics
from six.moves import xrange


if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version: ",tf.__version__)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0 ,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs >0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list
def add_extension(path):
    if os.path.exists(path +'.jpg'):
        return path +'.jpg'
    elif os.path.exists(path +'.png'):
        return path +'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)
def model_restore_from_pb(pb_path,node_dict):
    config = tf.ConfigProto(log_device_placement=True,#Print out whether the process is using the CPU or GPU
                            allow_soft_placement=True,#allow tf to continue with the available device
                            )
    config.gpu_options.allow_growth = True#Automatically adjust usage

    sess = tf.Session(config=config)
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

      
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        tf.import_graph_def(graph_def, name='')  # Import calculation graph

    sess.run(tf.global_variables_initializer())
    for key,value in node_dict.items():
        node = sess.graph.get_tensor_by_name(value)
        node_dict[key] = node
    return sess,node_dict

def get_epoch_data(data_array,start,end,shape):
    

    epoch_data = np.zeros(shape,dtype=float)
    for i in range(start ,end ,1):
        img = cv2.imread(data_array[i])
        if img is not None:
            img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img ,(shape[2] ,shape[1]))
            img = img.astype('float32')
            img = img / 255
        else:
            print("{} read failed".format(data_array[i]))
        epoch_data[i - start] = img

    return epoch_data


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    #evaluation statistics calculation
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds,
                                       distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                      distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance calculation
        diff = np.subtract(embeddings1, embeddings2)
        diff = diff.astype(np.float16)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
def eval_on_lfw(lfw_dir,lfw_pairs_path,pb_path,node_dict):

    batch_size = 12

    #get label and path from lfw dataset folder and pairs.txt

    # Read pairs.txt
    pairs = read_pairs(os.path.expanduser(lfw_pairs_path))
    # Get the paths for images
    paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)

    #restore pb model
    sess, node_dict = model_restore_from_pb(pb_path, node_dict)
    tf_input = node_dict['input']
    tf_embeddings = node_dict['embeddings']


    if "phase_train" in node_dict.keys():
        tf_phase_train = node_dict['phase_train']

    print("embeddings shape = ",tf_embeddings.shape)



    # embedding collecting
    iterations = math.ceil(len(paths)/batch_size)

    for i in range(iterations):
        n_start = i * batch_size
        n_end = n_start + batch_size
        if n_end > len(paths):
            n_end = len(paths)
            n_start = n_end - batch_size

        epoch_data = get_epoch_data(paths, n_start, n_end,
                                         (n_end - n_start, 160, 160, 3))
        if "phase_train" in node_dict.keys():
            feed_dict = {tf_input:epoch_data,tf_phase_train:False}
        else:
            feed_dict = {tf_input: epoch_data}
        sess_out = sess.run(tf_embeddings,feed_dict=feed_dict)

        if i == 0:
            embeddings = sess_out
        else:
            embeddings = np.concatenate((embeddings,sess_out),axis=0)

    print(embeddings.shape)
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    

if __name__ == "__main__":

    node_dict = {'input':'input:0',
                 'phase_train':'phase_train:0',
                 'embeddings':r'embeddings:0',
                 }
    eval_on_lfw(lfw_dir, lfw_pairs_path, pb_path, node_dict)