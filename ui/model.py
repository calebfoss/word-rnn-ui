import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import random
import numpy as np

from .beam import BeamSearch


#model that is created during training for word rnn
class Model():
    def __init__(self, args, infer=False):
        #parse arguments passed in
        self.args = args
        if infer:
            args.batch_size = 1 #default batch size 1
            args.seq_length = 1 #default sequence length 1

        if args.model == 'rnn': #rnn model
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru': #gru model
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm': #lstm model
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size) #set cell_fn given argument rnn_size

        #create cell with given number of layers
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        #initialize all tensor flow variables
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        #tensorflow summaries for visualization
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        #tensorflow scope
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"): #tensorflow device
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        #rnn decoder given input
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size]) #save output
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits], #save loss
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length #save cost
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state #save final state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables() #save trainable variables
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr) #save optimizer
        self.train_op = optimizer.apply_gradients(zip(grads, tvars)) #save train optimizer with gradients

    #sample function
    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0):
        state = sess.run(self.cell.zero_state(1, tf.float32)) #run session and save state
        if not len(prime) or prime == " ":
            prime  = random.choice(list(vocab.keys()))  #save prime
        print (prime) #print prime
        # loop through words in prime
        for word in prime.split()[:-1]:
            print (word) #print word
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)
        
        #pick weights
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        #pick beam samples given weights
        def beam_search_pick(weights):
            probs[0] = weights
            samples, scores = BeamSearch(probs).beamsearch(None, vocab.get(prime), None, 2, len(weights), False)
            sampleweights = samples[np.argmax(scores)]
            t = np.cumsum(sampleweights)
            s = np.sum(sampleweights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        word = prime.split()[-1]
        #loop through num
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if pick == 1:
                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)
            elif pick == 2:
                sample = beam_search_pick(p)

            pred = words[sample]
            ret += ' ' + pred
            word = pred
        return ret


