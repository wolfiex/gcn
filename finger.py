from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

load = 0

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
try:
    flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 45200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 2**4, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-8, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 1000000, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
except:None

# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
########################################
'''['vec_smiles',
 'smiles',
 'finger_mqn',
 'embed_fn ',#space!!
 'finger_maccs',
 'names',
 'vec_spec',
 'fngroups']'''

import load 
adj,finger = load.get_data()



y_val =  np.array(finger['fngroups']) #[[i] for i in finger['fngroups']])

#np.repeat(True,len(y_val))
from scipy import sparse
features = finger['fngroups']

#features = sparse.lil_matrix(finger['fngroups'])


# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
features = sparse.lil_matrix(features)


y_test = y_train = y_val
train_mask=val_mask=test_mask= y_val.sum(axis=1) > -10

#train_mask[:] = True

#val_mask=test_mask=train_mask
# Some preprocessing
features = preprocess_features(features.astype(np.float32))


if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)



# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

cost_val = []
accuracy = []
# Initialize session
sess = tf.Session()
#saver = tf.train.Saver()
# Init variables
sess.run(tf.global_variables_initializer())

#if fload:model.load(sess)



cost_val = []
accuracy = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")
accuracy.append(acc)
model.save(sess)
#save_path = saver.save(sess, "/tmp/model.ckpt")


# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


predict = sess.run(model.predict(),feed_dict = feed_dict)

support = [preprocess_adj(np.zeros(adj.shape))]
feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
predict2 = sess.run(model.predict(),feed_dict = feed_dict)

#out = sess.run(model.outputs,feed_dict = feed_dict)
print('end')
#https://github.com/tkipf/gcn/issues/44