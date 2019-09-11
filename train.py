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
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
except:None

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
########################################
#delfeature = features

import networkx as nx
G = nx.karate_club_graph()
adj = nx.adjacency_matrix(G)
from scipy import sparse
#from infomap import infomap
#from ...graph.infomap import infomap

import sys, os
print ( '/'.join(os.path.dirname(__file__).split('/')[:-2])+ '/graph/infomap'  )
sys.path.append('/'+'/'.join(os.path.dirname(__file__).split('/')[:-2])+ '/graph/infomap')
import infomap

def im(G):
    infomapWrapper = infomap.Infomap("--two-level -z -u -N 5 -y 2")

    for e in G.edges():
    	infomapWrapper.addLink(*e)


    infomapWrapper.run()

    tree = infomapWrapper.tree

    print("Found %d modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))

    print("\n#node module")

    nodedict = {}
    for node in tree.leafIter():
    	print("%d %d" % (node.physIndex, node.moduleIndex()))
        nodedict[node.physIndex] = node.moduleIndex()

    return nodedict


group = im(G)
print (group)

y_val =  np.zeros(shape=(adj.shape[0],adj.shape[0]))
for i in range(adj.shape[0]):
    y_val[i,group[i]] = 1

y_test = y_train = y_val
train_mask=val_mask=test_mask= y_val > -10
#np.repeat(True,len(y_val))

deg = G.degree()
features = np.array([ [deg[i]] for i in range(adj.shape[0])]).astype(float)
features = sparse.lil_matrix(features)

print(features)


# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

#train_mask[:] = True

#val_mask=test_mask=train_mask
# Some preprocessing
features = preprocess_features(features)
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
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
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

if load:
    model.load(sess)



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
