import tensorflow as tf
import numpy as np


class OptionCritic(object):
    def __init__(self, sess, d_in, n_actions, n_options, h1_units, h2_units, lr=0.0002, critic_lr=0.0004,
                 gamma=0.99, ddqn=False, terminal_reg=0, entropy_reg=0, update_target_turns=100, clip_loss=1, baseline=False,
                 batch_size=32):
        self.sess = sess
        self.s_dim = d_in
        self.a_dim = n_actions
        self.h1_size = h1_units
        self.h2_size = h2_units
        self.o_lr = lr
        self.c_lr = critic_lr
        self.n_options = n_options
        self.gamma = gamma
        self.ddqn = ddqn
        self.term_reg = terminal_reg
        self.entropy_reg = entropy_reg
        self.freeze_interval = update_target_turns
        self.batch_size = batch_size

        CLIP_LOSS = clip_loss
        BASELINE = baseline

        # placeholders
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='belief')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], name='belief_')
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.o = tf.placeholder(tf.int32, [None], name='option')
        self.a = tf.placeholder(tf.int32, [None], name='action')
        self.terminal = tf.placeholder(tf.float32, [None], name='is_terminal')
        self.tau = tf.placeholder(tf.float32, [], name='tau') # a scalar

        # initialization
        self.init_w, self.init_b = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.)

        # build nets
        self.feat = self._build_state_net(self.s, "state_net_eval")
        self.feat_ = self._build_state_net(self.s_, "state_net_eval", reuse=True)
        next_feat_prime = self._build_state_net(self.s_, "state_net_target", trainable=False)

        # return Q value over options
        self.Q = self._build_Q_net(self.feat, "Q_net_eval")
        Q_next = self._build_Q_net(self.feat_, "Q_net_eval", reuse=True)
        Q_next_prime = self._build_Q_net(next_feat_prime, "Q_net_target", trainable=False)

        o_idx = tf.stack([tf.range(tf.shape(self.o)[0]), self.o], axis=1)
        self.termination_probs = self._build_terminal_net(tf.stop_gradient(self.feat), 'term_net')
        self.option_term_prob = tf.gather_nd(self.termination_probs, o_idx)
        # don't want beta backprob with respect to next feature value
        next_termination_probs = self._build_terminal_net(tf.stop_gradient(self.feat_), 'term_net', reuse=True)
        next_option_term_prob = tf.gather_nd(next_termination_probs, o_idx)
        disc_option_term_prob = tf.stop_gradient(next_option_term_prob)

        self.action_probs = self._build_options_net(self.feat, 'option_net')
        # TODO: the author use raw state as input for options, should I change this?
        # self.sample_actions = tf.argmax(self.action_probs, axis=1).astype("int32")

        # net params
        self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state_net_eval')
        self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state_net_target')
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_net_eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_net_target')

        self.term_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='term_net')
        self.option_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='option_net')

        # self.o_params = self.se_params + self.option_params # include state weights
        # self.t_params = self.se_params + self.term_params
        self.q_eval_params = self.se_params + self.qe_params
        self.q_target_params = self.st_params + self.qt_params
        actor_params = self.term_params + self.option_params

        with tf.variable_scope('update_q_net'):
            # self.update_target_op = [tf.assign(t, e) for t, e in zip(self.q_target_params, self.q_eval_params)]
            self.update_target_op = [self.q_target_params[i].assign(
                tf.multiply(self.tau, self.q_eval_params[i]) + tf.multiply(1 - self.tau, self.q_target_params[i]))
                for i in range(len(self.q_target_params))]

        # =======================CRITIC=====================================
        max_idx = tf.stack([tf.range(tf.shape(Q_next)[0]), tf.argmax(Q_next, axis=1, output_type=tf.int32)], axis=1)
        if self.ddqn:
            print "training with ddqn algorithm"
            y = tf.squeeze(self.r, axis=1) + (1 - self.terminal) * self.gamma * (
                (1 - disc_option_term_prob) * tf.gather_nd(Q_next_prime, o_idx) +
                disc_option_term_prob * tf.gather_nd(Q_next_prime, max_idx))
        else:
            print "training with dqn"
            y = tf.squeeze(self.r, axis=1) + (1 - self.terminal) * self.gamma * (
                (1 - disc_option_term_prob) * tf.gather_nd(Q_next_prime, o_idx) +
                disc_option_term_prob * tf.reduce_max(Q_next_prime, axis=1)
            )

        y = tf.stop_gradient(y)
        Q_option = tf.gather_nd(self.Q, o_idx)
        td_errors = y - Q_option

        if CLIP_LOSS > 0:
            td_cost = self.huber_loss(td_errors, CLIP_LOSS)
        else:
            td_cost = 0.5 * td_errors ** 2

        with tf.variable_scope('closs'):
            # TODO: whether I should optimize actor loss with critic loss together
            self.closs = np.sum(td_cost)

        # initial an optimizer
        opt_c = tf.train.AdamOptimizer(self.c_lr)
        grad_c = opt_c.compute_gradients(self.closs, self.q_eval_params)
        # choose if clip gradients
        with tf.variable_scope('ctrain'):
            self.ctrain = opt_c.apply_gradients(grad_c)

        # ===============================ACTOR================================
        # actor gradients and updates w.r.t. policy params and terminal params
        opt = tf.train.AdamOptimizer(self.o_lr)
        dis_Q = tf.stop_gradient(Q_option)
        dis_V = tf.stop_gradient(tf.reduce_max(self.Q, axis=1))

        with tf.variable_scope('actor_grad'):
            a_idx = tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=1)
            term_grad = tf.reduce_sum(self.option_term_prob * (dis_Q - dis_V + self.term_reg))
            # entropy = - np.sum(tf.reduce_sum(self.action_probs*tf.log(self.action_probs), axis=1))
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs), axis=[0, 1])

            if not BASELINE:
                policy_grad = -tf.reduce_sum(tf.log(tf.gather_nd(self.action_probs, a_idx)) * y) - self.entropy_reg * entropy
            else:
                policy_grad = -tf.reduce_sum(tf.log(tf.gather_nd(self.action_probs, a_idx)) * (y - dis_Q)) - self.entropy_reg * entropy

            gvs = opt.compute_gradients(term_grad + policy_grad, actor_params)

        if CLIP_LOSS > 0:
            clip = tf.constant(CLIP_LOSS, dtype=tf.float32)
            gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]

        with tf.variable_scope('otrain'):
            self.otrain = opt.apply_gradients(gvs)

    def _build_state_net(self, s, scope_name, trainable=True, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            e1 = tf.layers.dense(s, self.h1_size, tf.nn.relu, kernel_initializer=self.init_w,
                                 bias_initializer=self.init_b, name='share_layer1', trainable=trainable)
            return tf.layers.dense(e1, self.h2_size, tf.nn.relu, kernel_initializer=self.init_w,
                                   bias_initializer=self.init_b, name='share_layer2', trainable=trainable)

    def _build_Q_net(self, feat, scope_name, trainable=True, reuse=None):
        # build Q net over options
        with tf.variable_scope(scope_name, reuse=reuse):
            return tf.layers.dense(feat, self.n_options, None, kernel_initializer=self.init_w,
                                   bias_initializer=self.init_b, name='Q_omega', trainable=trainable)

    def _build_terminal_net(self, feat, scope_name, reuse=None):
        # build terminal function net
        with tf.variable_scope(scope_name, reuse=reuse):
            return tf.layers.dense(feat, self.n_options, tf.nn.sigmoid, kernel_initializer=self.init_w,
                                   bias_initializer=self.init_b, name='terminal')

    def _build_options_net(self, feat, scope_name, reuse=None):
        # build intra policy net for each option
        with tf.variable_scope(scope_name, reuse=reuse):
            self.intro_option_policies = []
            for i in range(self.n_options):
                intro_option = self._build_option_layer(feat, name='option_policy_{}'.format(i))
                self.intro_option_policies.append(intro_option)
        self.intro_option_policies = tf.stack(self.intro_option_policies, axis=1)
        # self.intro_option_policies = tf.tile(self.intro_option_policies, tf.shape(self.o)[0])
        idx = tf.stack([tf.range(tf.shape(self.o)[0]), self.o], axis=1)
        # option_action_probs = []
        # for i in tf.range(tf.shape(self.o)[0]):
        #     current_prob = self.intro_option_policies[self.o[i]]
        #     option_action_probs.append(current_prob)
        return tf.gather_nd(self.intro_option_policies, idx)

    def _build_option_layer(self, feat, name=None):
        layer = None
        layer = tf.layers.dense(feat, self.a_dim, activation=tf.nn.softmax, name=name)
        return layer

    def get_feat_value(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.feat, feed_dict={self.s: s})

    def get_q_value(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.Q, feed_dict={self.s: s})

    def get_terminate_probs(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.termination_probs, feed_dict={self.s: s})

    def get_intrapolicy_probs(self, s, o):
        s = np.array(s)
        o = np.array(o, dtype=np.int32)
        if o.ndim < 1:
            o = np.array([o], dtype=np.int32)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.action_probs, feed_dict={self.s: s, self.o: o})

    def train(self, state, option, action, reward, next_state, terminal):
        print "training ...."
        self.sess.run([self.otrain, self.ctrain],
                      feed_dict={self.s: state, self.o: option, self.a: action, self.r: reward, self.s_: next_state, self.terminal: terminal})

    def update_target(self, tau):
        print "freeze interval:", self.freeze_interval
        print "Updating eval net to target net..."
        self.sess.run(self.update_target_op, feed_dict={self.tau:tau})

    def predict_option(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        Q = self.get_q_value(s)
        return np.argmax(Q, axis=1)

    def terminal_sample(self, s, o):
        # predict terminal and sample the next option
        s = np.array(s)
        o = np.array(o, dtype=np.int32)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        if o.ndim < 1:
            o = np.array([o], dtype=np.int32)
        Q = self.sess.run(self.Q, feed_dict={self.s: s})
        probs = self.sess.run(self.termination_probs, feed_dict={self.s: s})
        option_probs = probs[0, o]
        termination = np.greater(option_probs, np.random.uniform(size=(1, 1)))
        return termination, np.argmax(Q)

    def choose_action(self, input, option, non):
        # decide the current action based on the current max Q
        input = input[np.newaxis, :]
        action_probs = self.get_intrapolicy_probs(input, option)
        # sample action
        action = np.argmax(action_probs, axis=1)
        return action

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, load_filename)
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully loaded:", load_filename
        except BaseException:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving a2c-network...'
        self.saver.save(self.sess, save_filename)

    def huber_loss(self, err, d):
        return tf.where(tf.abs(err) < d, 0.5 * tf.square(err), d * (tf.abs(err) - d / 2))
