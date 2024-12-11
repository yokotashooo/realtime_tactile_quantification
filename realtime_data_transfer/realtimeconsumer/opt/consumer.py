from kafka import KafkaConsumer
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
import numpy as np
from scipy import integrate
import pickle
import statistics
from operator import itemgetter
from numba.experimental import jitclass
from numba import f8, i8
import time
import json
import ast


def kernel(t):
    """ 畳み込みカーネル """
    return np.exp(-t / 20)


@jitclass([('N', i8), ('dt', f8), ('td', f8), ('r', f8[:])])
class SingleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=5e-3):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.r = np.zeros(N)

    def initialize_states(self):
        self.r = np.zeros(self.N)

    def func(self, spike):
        r = self.r*(1-self.dt/self.td) + spike/self.td
        self.r = r
        return r
    #  def __call__(self, spike):
    #    r = self.r*(1-self.dt/self.td) + spike/self.td
    #    self.r = r
    #    return r


@jitclass([('N', i8), ('dt', f8), ('tref', f8), ('tc_m', f8), ('vrest', i8),
          ('vreset', i8), ('vthr', i8), ('vpeak', i8), ('e_exc', i8),
          ('e_inh', i8), ('v', f8[:]), ('v_', f8[:]), ('tlast', f8),
          ('tcount', i8)])
class ConductanceBasedLIF:
    def __init__(self, N, dt=1e-4, tref=5e-3, tc_m=1e-2,
                 vrest=-60, vreset=-60, vthr=-50, vpeak=20,
                 e_exc=0, e_inh=-100):
        """
        Conductance-based Leaky integrate-and-fire model.

        Args:
            N (int)       : Number of neurons.
            dt (float)    : Simulation time step in seconds.
            tc_m (float)  : Membrane time constant in seconds.
            tref (float)  : Refractory time constant in seconds.
            vreset (float): Reset membrane potential (mV).
            vrest (float) : Resting membrane potential (mV).
            vthr (float)  : Threshold membrane potential (mV).
            vpeak (float) : Peak membrane potential (mV).
            e_exc (float) : equilibrium potential of excitatory synapses (mV).
            e_inh (float) : equilibrium potential of inhibitory synapses (mV).
        """
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m
        self.vrest = vrest
        self.vreset = vreset
        self.vthr = vthr
        self.vpeak = vpeak

        self.e_exc = e_exc  # 興奮性シナプスの平衡電位
        self.e_inh = e_inh  # 抑制性シナプスの平衡電位

        self.v = self.vreset*np.ones(N)
        self.v_ = np.zeros(N)  # change
        self.tlast = 0
        self.tcount = 0

    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vthr -
                                                           self.vreset)
        else:
            self.v = self.vreset*np.ones(self.N)
        self.tlast = 0
        self.tcount = 0

    def func(self, g_exc, g_inh):
        I_synExc = g_exc*(self.e_exc - self.v)
        I_synInh = g_inh*(self.e_inh - self.v)
        # Voltage equation with refractory period
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m
        v = self.v + ((self.dt*self.tcount) > (self.tlast +
                                               self.tref))*dv*self.dt

        s = 1*(v >= self.vthr)  # 発火時は1, その他は0の出力
        self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s  # 最後の発火時の更新
        v = v*(1-s) + self.vpeak*s  # 閾値を超えると膜電位をvpeakにする
        self.v_ = v  # 発火時の電位も含めて記録するための変数
        self.v = v*(1-s) + self.vreset*s  # 発火時に膜電位をリセット
        self.tcount += 1

        return s


@jitclass([('N', i8), ('dt', f8), ('tref', f8), ('tc_m', f8), ('vrest', i8),
          ('vreset', i8), ('vrest', i8), ('vreset', i8), ('init_vthr', i8),
          ('vpeak', i8), ('theta_plus', f8), ('theta_max', i8),
          ('tc_theta', f8), ('vthr', i8), ('vpeak', i8), ('e_exc', i8),
          ('e_inh', i8), ('v', f8[:]), ('theta', f8[:]), ('v_', f8[:]),
          ('tlast', f8), ('tcount', i8)])
class DiehlAndCook2015LIF:
    def __init__(self, N, dt=1e-3, tref=5e-3, tc_m=1e-1,
                 vrest=-65, vreset=-65, init_vthr=-52, vpeak=20,
                 theta_plus=0.05, theta_max=35,
                 tc_theta=1e4, e_exc=0, e_inh=-100):
        """
        Leaky integrate-and-fire model of Diehl and Cooks (2015)
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full

        Args:
            N (int)       : Number of neurons.
            dt (float)    : Simulation time step in seconds.
            tc_m (float)  : Membrane time constant in seconds.
            tref (float)  : Refractory time constant in seconds.
            vreset (float): Reset membrane potential (mV).
            vrest (float) : Resting membrane potential (mV).
            vthr (float)  : Threshold membrane potential (mV).
            vpeak (float) : Peak membrane potential (mV).
            e_exc (float) : equilibrium potential of excitatory synapses (mV).
            e_inh (float) : equilibrium potential of inhibitory synapses (mV).
        """
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m
        self.vreset = vreset
        self.vrest = vrest
        self.init_vthr = init_vthr
        self.theta = np.zeros(N)
        self.theta_plus = theta_plus
        self.theta_max = theta_max
        self.tc_theta = tc_theta
        self.vpeak = vpeak

        self.e_exc = e_exc  # 興奮性シナプスの平衡電位
        self.e_inh = e_inh  # 抑制性シナプスの平衡電位

        self.v = self.vreset*np.ones(N)
        self.vthr = self.init_vthr
        self.v_ = np.zeros(N)  # change
        self.tlast = 0
        self.tcount = 0

    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vthr
                                                           - self.vreset)
        else:
            self.v = self.vreset*np.ones(self.N)
        self.vthr = self.init_vthr
        self.theta = np.zeros(self.N)
        self.tlast = 0
        self.tcount = 0

    def func(self, g_exc, g_inh):
        I_synExc = g_exc*(self.e_exc - self.v)
        I_synInh = g_inh*(self.e_inh - self.v)
        # Voltage equation with refractory period
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m
        v = self.v + ((self.dt*self.tcount) > (self.tlast
                                               + self.tref))*dv*self.dt

        s = 1*(v >= self.vthr)  # 発火時は1, その他は0の出力
        theta = (1-self.dt/self.tc_theta)*self.theta + self.theta_plus*s
        self.theta = np.clip(theta, 0, self.theta_max)
        self.vthr = self.theta + self.init_vthr
        self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s  # 最後の発火時の更新
        v = v*(1-s) + self.vpeak*s  # 閾値を超えると膜電位をvpeakにする
        self.v_ = v  # 発火時の電位も含めて記録するための変数
        self.v = v*(1-s) + self.vreset*s  # 発火時に膜電位をリセット
        self.tcount += 1

        return s


# @jitclass([('W',f8[:,:])])
class FullConnection:
    def __init__(self, N_in, N_out, initW=None):
        """
        FullConnection
        """
        if initW is not None:
            self.W = initW
        else:
            self.W = 0.1*np.random.rand(N_out, N_in)

    def backward(self, x):
        return np.dot(self.W.T, x)  # self.W.T @ x

    def func(self, x):
        return np.dot(self.W, x)  # self.W @ x


@jitclass([('N', i8), ('nt_delay', i8), ('state', f8[:, :])])
class DelayConnection:
    def __init__(self, N, delay, dt=1e-4):
        """
        Args:
            delay (float): Delay time
        """
        self.N = N
        self.nt_delay = round(delay/dt)  # 遅延のステップ数
        self.state = np.zeros((N, self.nt_delay))

    def initialize_states(self):
        self.state = np.zeros((self.N, self.nt_delay))

    def func(self, x):
        out = self.state[:, -1]  # 出力

        self.state[:, 1:] = self.state[:, :-1]  # 配列をずらす
        self.state[:, 0] = x  # 入力

        return out


np.random.seed(seed=0)


class DiehlAndCook2015Network:
    # exc_neurons: exc_neurons_type
    # inh_neurons: inh_neurons_type
    # input_synapse:input_synapse_type
    # exc_synapse:exc_synapse_type
    # inh_synapse: inh_synapse_type
    # input_synaptictrace:  input_synaptictrace_type
    # exc_synaptictrace:exc_synaptictrace_type
    # input_conn:input_conn_type
    # # delay_input :delay_input_type
    # # delay_exc2inh:delay_exc2inh_type
    def __init__(self, n_in=4, n_neurons=100, wexc=2.25, winh=0.875,
                 dt=1e-3, wmin=0.0, wmax=5e-2, lr=(1e-2, 1e-4),
                 update_nt=100, norm=0.1):
        """
        Network of Diehl and Cooks (2015)
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full

        Args:
            n_in: Number of input neurons.
            Matches the 1D size of the input data.
            n_neurons: Number of excitatory,
            inhibitory neurons.
            wexc: Strength of synapse weights
            from excitatory to inhibitory layer.
            winh: Strength of synapse weights
            from inhibitory to excitatory layer.
            dt: Simulation time step.
            lr: Single or pair of learning rates
            for pre- and post-synaptic events, respectively.
            wmin: Minimum allowed weight on input to excitatory synapses.
            wmax: Maximum allowed weight on input to excitatory synapses.
            update_nt: Number of time steps of weight updates.
        """

        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax = wmax
        self.wmin = wmin

        # Neurons
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt, tref=5e-3,
                                               tc_m=1e-1,
                                               vrest=-65, vreset=-65,
                                               init_vthr=-52,
                                               vpeak=20, theta_plus=0.05,
                                               theta_max=35,
                                               tc_theta=1e4,
                                               e_exc=0, e_inh=-100)

        self.inh_neurons = ConductanceBasedLIF(n_neurons, dt=dt, tref=2e-3,
                                               tc_m=1e-2,
                                               vrest=-60, vreset=-45,
                                               vthr=-40, vpeak=20,
                                               e_exc=0, e_inh=-85)
        # Synapses
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)

        self.input_synaptictrace = SingleExponentialSynapse(n_in, dt=dt,
                                                            td=2e-2)
        self.exc_synaptictrace = SingleExponentialSynapse(n_neurons, dt=dt,
                                                          td=2e-2)

        # Connections
        initW = 1e-3*np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons,
                                         initW=initW)
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/(n_neurons
                                - 1))*(np.ones((n_neurons, n_neurons))
                                       - np.eye(n_neurons))

        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)

        self.norm = norm
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.update_nt = update_nt
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.s_in_ = np.zeros((self.update_nt, n_in))
        self.s_exc_ = np.zeros((n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, n_in))
        self.x_exc_ = np.zeros((n_neurons, self.update_nt))

    # スパイクトレースのリセット
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.n_in))
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in))
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0

    # 状態の初期化
    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()

    def __call__(self, s_in, stdp=True):
        # 入力層
        c_in = self.input_synapse.func(s_in)
        x_in = self.input_synaptictrace.func(s_in)
        g_in = self.input_conn.func(c_in)

        # 興奮性ニューロン層
        s_exc = self.exc_neurons.func(self.delay_input.func(g_in), self.g_inh)
        c_exc = self.exc_synapse.func(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_synaptictrace.func(s_exc)

        # 抑制性ニューロン層
        s_inh = self.inh_neurons.func(self.delay_exc2inh.func(g_exc), 0)
        c_inh = self.inh_synapse.func(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        if stdp:
            # スパイク列とスパイクトレースを記録
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.x_in_[self.tcount] = x_in
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1

            # Online STDP
            if self.tcount == self.update_nt:
                W = np.copy(self.input_conn.W)

                # postに投射される重みが均一になるようにする
                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum

                # STDP則
                dW = self.lr_p*(self.wmax - W)*np.dot(self.s_exc_, self.x_in_)
                dW -= self.lr_m*W*np.dot(self.x_exc_, self.s_in_)
                clipped_dW = np.clip(dW / self.update_nt, -1e-3, 1e-3)
                self.input_conn.W = np.clip(W + clipped_dW,
                                            self.wmin, self.wmax)
                self.reset_trace()  # スパイク列とスパイクトレースをリセット

        return s_exc


def collect_data(queue_in):
    consumer = KafkaConsumer(
        'experi',  # トピック名
        bootstrap_servers='broker:29092',  # ブートストラップサーバー
        auto_offset_reset='earliest',  # トピックの最初から読み取る
        enable_auto_commit=True,
        group_id='g1',
        value_deserializer=lambda x: x.decode('utf-8')
    )
    print("Starting Consumer...")
    for message in consumer:
        data_str = json.loads(message.value)
        data_list = ast.literal_eval(data_str)
        transformed = list(map(list, zip(*data_list)))
        for elem in transformed:
            queue_in.put(elem)


def count_ones_by_label(labels, predictions):
    count_label_1 = 0
    count_label_2 = 0
    for lbl, pred in zip(labels, predictions):
        if pred == 1:
            if lbl == 1:
                count_label_1 += 1
            elif lbl == 2:
                count_label_2 += 1
    return (count_label_1, count_label_2)


def classification(queue_in, queue_out):
    n_in = 6
    n_neurons = 20
    wexc = 4.544910490280396
    winh = 1.3082028299035284
    lr = (0.0312611326044722, 0.03392142258355323)
    norm = 0.12856140170464153
    dt = 2e-4

    network_test = DiehlAndCook2015Network(n_in=n_in, n_neurons=n_neurons,
                                           dt=dt, wexc=wexc, winh=winh,
                                           lr=lr, norm=norm)
    # 学習済みの重みと閾値の読み込み
    weight_path = '/weight_epoch9.npy'
    theta_path = '/exc_neurons_epoch9.npy'
    assignments_path = '/assignment_epoch9.npy'  # 割り当て情報

    network_test.input_conn.W = np.load(weight_path)
    network_test.exc_neurons.theta = np.load(theta_path)
    network_test.exc_neurons.theta_plus = 0

    assignments = np.load(assignments_path)
    while True:
        item = queue_in.get()
        s_exc_test = network_test(item, stdp=False)
        result = count_ones_by_label(assignments, s_exc_test)
        queue_out.put(result)


def decision_making(queue_out):
    sliding_queue = Queue()
    sum_x, sum_y = 0, 0
    while True:
        x, y = queue_out.get()
        sliding_queue.put((x, y))
        sum_x += x
        sum_y += y
        if sliding_queue.qsize() > 100:
            old_x, old_y = sliding_queue.get()
            sum_x -= old_x
            sum_y -= old_y
        print((sum_x, sum_y))


if __name__ == '__main__':
    n_in = 6
    n_neurons = 20
    wexc = 4.544910490280396
    winh = 1.3082028299035284
    lr = (0.0312611326044722, 0.03392142258355323)
    norm = 0.12856140170464153
    dt = 2e-4

    network_test = DiehlAndCook2015Network(n_in=n_in, n_neurons=n_neurons,
                                           dt=dt, wexc=wexc, winh=winh,
                                           lr=lr, norm=norm)
    # 学習済みの重みと閾値の読み込み
    weight_path = '/weight_epoch9.npy'
    theta_path = '/exc_neurons_epoch9.npy'
    assignments_path = '/assignment_epoch9.npy'  # 割り当て情報

    network_test.input_conn.W = np.load(weight_path)
    network_test.exc_neurons.theta = np.load(theta_path)
    network_test.exc_neurons.theta_plus = 0

    assignments = np.load(assignments_path)

    queue_in = Queue()
    queue_out = Queue()

    # プロセスの作成
    producer = Process(target=collect_data, args=(queue_in,))
    processor = Process(target=classification, args=(queue_in, queue_out))
    consumer = Process(target=decision_making, args=(queue_out,))

    # プロセスの開始
    processor.start()
    consumer.start()
    producer.start()

    # 全プロセスが終了するまで待機
    producer.join()
    processor.join()
    consumer.join()

    print("All processes finished.")
