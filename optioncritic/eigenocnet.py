import tensorflow as tf
import numpy as np

CLIP_LOSS = 1.0
BASELINE = True # using baseline for option critic


class OptionCritic(object):
    def __init__(self, sess, d_in, n_actions, n_options, h1_units=200, h2_units=100, lr=0.0002, critic_lr=0.0004,
                 gamma=0.99, ddqn=False, terminal_reg=0, entropy_reg=0, update_target_turns=100,
                 batch_size=32):
        self.sess = sess
        self.s_dim = d_in
        self.a_dim = n_actions
        self.h1_size = h1_units
        self.h2_size = h2_units
        self.p_lr = lr        # learning rate for policy gradient
        self.c_lr = critic_lr # learning rate for Q(s,o,a) and Q(s, o)
        self.n_options = n_options
        self.gamma = gamma
        self.ddqn = ddqn
        self.term_reg = terminal_reg
        self.entropy_reg = entropy_reg
        self.freeze_interval = update_target_turns
        self.batch_size = batch_size


        # placeholders
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='S')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], name='S_')
        self.ro = tf.placeholder(tf.float32, [None, 1], name='external_reward')
        self.ra = tf.placeholder(tf.float32, [None, 1], name='combined_reward')
        self.o = tf.placeholder(tf.int32, [None], name='option')
        self.a = tf.placeholder(tf.int32, [None], name='action')
        self.terminal = tf.placeholder(tf.float32, [None], name='is_terminal') # 0 or 1
        self.tau = tf.placeholder(tf.float32, [], name='tau') # a scalar

        # initialization
        self.init_w, self.init_b = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.)

        # build nets
        with tf.variable_scope('state_net'):
            self.feat = self._build_state_net(self.s, "state_net_eval")
            self.feat_ = self._build_state_net(self.s_, "state_net_eval", reuse=True)
            next_feat_prime = self._build_state_net(self.s_, "state_net_target", trainable=False)

        # Q(s,o) option based state option value function
        with tf.variable_scope('Q_s_o'):
            self.Q = self._build_Q_net(self.feat, self.n_options,  "Q_net_eval") # Q(s,o)
            Q_next = self._build_Q_net(self.feat_, self.n_options, "Q_net_eval", reuse=True) # Q(s_, o)
            Q_next_prime = self._build_Q_net(next_feat_prime, self.n_options, "Q_net_target", trainable=False) # Q'(s_, o)

        # Q(s,o,a) state action value function for intra-policym, use "u" indicates the intra-Q
        with tf.variable_scope('Q_s_o_a'):
            self.Qu = self._build_options_Q_net(self.feat, "Qu_net_eval")  # Q(s,o,a)
            # self.Qu_next = self._build_options_Q_net(self.feat_,  "Qu_net_eval", reuse=True)  # Q(s_, o, a)
            # self.Qu_next_prime = self._build_options_Q_net(next_feat_prime,  "Qu_net_target", trainable=False)  # Q'(s_, o, a)

        self.termination_probs = self._build_terminal_net(tf.stop_gradient(self.feat), 'term_net')
        o_idx = tf.stack([tf.range(tf.shape(self.o)[0]), self.o], axis=1)
        a_idx = tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=1)
        self.option_term_prob = tf.gather_nd(self.termination_probs, o_idx)
        next_termination_probs = self._build_terminal_net(self.feat_, 'term_net', reuse=True)
        next_option_term_prob = tf.gather_nd(next_termination_probs, o_idx)
        disc_option_term_prob = tf.stop_gradient(next_option_term_prob)
        # self.termination_sample = tf.greater(next_option_term_prob, tf.random_uniform((self.s.shape[0],1)))

        self.action_probs = self._build_options_net(tf.stop_gradient(self.feat), 'option_net')
        # self.action_probs = self._build_options_net(self.feat, 'option_net')

        # net params
        self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state_net/state_net_eval')
        self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state_net/state_net_target')
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_s_o/Q_net_eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_s_o/Q_net_target')
        self.que_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_s_o_a/Qu_net_eval')
        self.qut_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_s_o_a/Qu_net_target')

        self.term_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='term_net')
        self.option_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='option_net')

        # self.o_params = self.se_params + self.option_params # include state weights
        # self.t_params = self.se_params + self.term_params
        self.q_eval_params = self.se_params + self.qe_params
        self.q_target_params = self.st_params + self.qt_params
        self.qu_eval_params = self.se_params + self.que_params
        self.qu_target_params = self.st_params + self.qut_params
        actor_params = self.term_params + self.option_params

        with tf.variable_scope('update_q_net'):
            # if tau is None:
            #     self.update_target_op = [tf.assign(t, e) for t, e in zip(self.q_target_params, self.q_eval_params)]
            #     self.update_u_target_op = [tf.assign(t, e) for t, e in zip(self.qu_target_params, self.qu_eval_params)]
            # else:
            self.update_target_op = [self.q_target_params[i].assign(
                tf.multiply(self.tau , self.q_eval_params[i]) + tf.multiply(1 - self.tau, self.q_target_params[i])) for i in range(len(self.q_target_params))]
            self.update_u_target_op = [self.qu_target_params[i].assign(
                self.tau * self.qu_eval_params[i] + (1 - self.tau) * self.qu_target_params[i]) for i in  range(len(self.qu_target_params))]

        # =======================define loss=====================================
        max_idx = tf.stack([tf.range(tf.shape(Q_next)[0]), tf.argmax(Q_next, axis=1, output_type=tf.int32)], axis=1)
        if self.ddqn:
            print "training with ddqn algorithm"
            g = (1 - self.terminal) * self.gamma * (
                (1 - disc_option_term_prob) * tf.gather_nd(Q_next_prime, o_idx) +
                disc_option_term_prob * tf.gather_nd(Q_next_prime, max_idx))
            y = tf.squeeze(self.ro, axis=1) + g
            y_u = tf.squeeze(self.ra, axis=1) + g
        else:
            print "training with dqn"
            g = (1 - self.terminal) * self.gamma * (
                (1 - disc_option_term_prob) * tf.gather_nd(Q_next_prime, o_idx) +
                disc_option_term_prob * tf.reduce_max(Q_next_prime, axis=1))
            y = tf.squeeze(self.ro, axis=1) + g
            y_u = tf.squeeze(self.ra, axis=1) + g

        y = tf.stop_gradient(y)
        y_u = tf.stop_gradient(y_u)
        Q_option = tf.gather_nd(self.Q, o_idx)
        Q_u = tf.gather_nd(self.Qu, a_idx)
        td_errors = y - Q_option
        td_action_err = y_u  - Q_u

        if CLIP_LOSS > 0:
            td_cost = self.huber_loss(td_errors, CLIP_LOSS)
            td_cost_u = self.huber_loss(td_action_err, CLIP_LOSS)
        else:
            td_cost = 0.5 * td_errors ** 2
            td_cost_u = 0.5 * td_action_err ** 2


        with tf.variable_scope('critic_loss'):
            self.closs = np.sum(td_cost)
            self.uloss = np.sum(td_cost_u)

        # initial an optimizer
        opt_c = tf.train.AdamOptimizer(self.c_lr)
        # grad_c = opt_c.compute_gradients(self.closs, self.q_eval_params)
        # choose if clip gradients
        with tf.variable_scope('ctrain'):
            # self.ctrain = opt_c.apply_gradients(grad_c)
            self.ctrain = opt_c.minimize(self.closs + self.uloss)

        # actor gradients and updates w.r.t. policy params and terminal params
        opt = tf.train.AdamOptimizer(self.p_lr)
        dis_Q = tf.stop_gradient(Q_option)
        dis_V = tf.stop_gradient(tf.reduce_max(self.Q, axis=1))

        with tf.variable_scope('actor_grad'):
            term_grad = tf.reduce_sum(self.option_term_prob * (dis_Q - dis_V + self.term_reg))
            # entropy = - np.sum(tf.reduce_sum(self.action_probs*tf.log(self.action_probs), axis=1))
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs), axis=[0, 1])

            if not BASELINE:
                policy_grad = -tf.reduce_sum(tf.log(tf.gather_nd(self.action_probs, a_idx)) * y_u) - self.entropy_reg * entropy
            else:
                policy_grad = -tf.reduce_sum(tf.log(tf.gather_nd(self.action_probs, a_idx)) * (y_u - dis_Q)) - self.entropy_reg * entropy

            gvs = opt.compute_gradients(term_grad + policy_grad, actor_params)

        if CLIP_LOSS > 0:
            with tf.variable_scope('clip_grad'):
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

    def _build_Q_net(self, feat, output, scope_name, trainable=True, reuse=None):
        # build Q net over options
        with tf.variable_scope(scope_name, reuse=reuse):
            return tf.layers.dense(feat, output, None, kernel_initializer=self.init_w,
                                   bias_initializer=self.init_b, name='Q_o', trainable=trainable)

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
                intro_option = self._build_option_layer(feat, activate=tf.nn.softmax, name='option_policy_{}'.format(i))
                self.intro_option_policies.append(intro_option)
        self.intro_option_policies = tf.stack(self.intro_option_policies, axis=1)
        # self.intro_option_policies = tf.tile(self.intro_option_policies, tf.shape(self.o)[0])
        idx = tf.stack([tf.range(tf.shape(self.o)[0]), self.o], axis=1)
        # option_action_probs = []
        # for i in tf.range(tf.shape(self.o)[0]):
        #     current_prob = self.intro_option_policies[self.o[i]]
        #     option_action_probs.append(current_prob)
        return tf.gather_nd(self.intro_option_policies, idx)

    def _build_options_Q_net(self, feat, scope_name, reuse=None, trainable=True):
        # build Q  net for each option, return Q(s,o,a)
        with tf.variable_scope(scope_name, reuse=reuse):
            self.intro_option_q = []
            for i in range(self.n_options):
                Qu_o = self._build_option_layer(feat,  name='Q_u_s_{}'.format(i), trainable=trainable)
                self.intro_option_q.append(Qu_o)
        self.q_options = tf.stack(self.intro_option_q, axis=1)
        # self.intro_option_policies = tf.tile(self.intro_option_policies, tf.shape(self.o)[0])
        idx = tf.stack([tf.range(tf.shape(self.o)[0]), self.o], axis=1)
        return tf.gather_nd(self.q_options, idx)

    def _build_option_layer(self, feat, activate=None, name=None, trainable=True):
        layer = None
        layer = tf.layers.dense(feat, self.a_dim, activation=activate, name=name, trainable=trainable)
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

    def train(self, state, option, action, re, ra, next_state, terminal):
        # ra: intrisic policy reward signal = r_e + r_i; re: external reward
        self.sess.run([self.otrain, self.ctrain],
                      feed_dict={self.s: state, self.o: option, self.a: action, self.ro: re,
                                 self.ra: ra, self.s_: next_state, self.terminal: terminal})

    def update_target(self, tau):
        print "freeze interval:", self.freeze_interval
        print "Updating eval net to target net for Q(s,o) and Q(s,o,a) ..."
        self.sess.run([self.update_target_op, self.update_u_target_op], feed_dict={self.tau:tau})

    def predict_option(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        Q = self.get_q_value(s)
        return np.argmax(Q, axis=1)

    def predict_intra_q(self, s, o):
        return self.sess.run(self.Qu, feed_dict={self.s:s, self.o: o})

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

    def choose_action(self, input, option):
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
        return tf.where(tf.abs(err) < d, 0.5 * tf.square(err), d * (tf.abs(err) - d/2))

