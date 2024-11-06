from kafka import KafkaConsumer
from concurrent.futures import ProcessPoolExecutor
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


def lif(currents, time: int, dt: float = 1.0, rest=-65,
        th=-52, ref=3, tc_decay=100):
    """ simple LIF neuron """
    time = int(time / dt)

    # initialize
    tlast = 0  # 最後に発火した時刻
    vpeak = 20  # 膜電位のピーク(最大値)
    spikes = np.zeros(time)
    v = rest  # 静止膜電位

    monitor = []  # monitor voltage

    # Core of LIF
    # 微分方程式をコーディングするときは，このように時間分解能dtで離散的に計算することが多い
    for t in range(time):
        dv = ((dt * t) > (tlast + ref)) * (-v + rest +
                                           currents[t]) / tc_decay  # 微小膜電位増加量
        v = v + dt * dv  # 膜電位を計算

        tlast = tlast + (dt * t - tlast) * (v >= th)  # 発火したら発火時刻を記録
        v = v + (vpeak - v) * (v >= th)  # 発火したら膜電位をピークへ

        monitor.append(v)

        spikes[t] = (v >= th) * 1  # スパイクをセット

        v = v + (rest - v) * (v >= th)  # 静止膜電位に戻す

    return spikes, monitor


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


def process_channel(n, data):
    duration = 2000
    convolve_dt = 0.2
    t = np.arange(0, duration, convolve_dt)
    f = open(f"/convolve123456_{n}.txt", "rb")
    k = 9
    GF = 100
    row = pickle.load(f)

    convolve_list = []
    spikes_7 = np.zeros(int(duration / convolve_dt))
    input_data_7 = GF * data[:, n]
    spikes_7, voltage_7 = lif(input_data_7, duration, convolve_dt)
    print(time.time())
    convolve_7 = np.convolve(spikes_7, kernel(t))[:int(duration / convolve_dt)]

    for m in range(600):
        convolve_list.append(integrate.trapz((convolve_7 - row[m]) ** 2, t))

    convolve_list_sort = np.array(convolve_list).argsort()
    knn = (convolve_list_sort + 100) // 100
    mode = statistics.multimode(knn[:k])
    if len(mode) == 2:
        index1 = np.where(knn[:k] == mode[0])
        index2 = np.where(knn[:k] == mode[1])
        sum1 = sum(itemgetter(*index1[0])(sorted(convolve_list)))
        sum2 = sum(itemgetter(*index2[0])(sorted(convolve_list)))
        if sum1 > sum2:
            index = mode[1]
        else:
            index = mode[0]
    else:
        index = mode[0]
    print(index)
    print(time.time())
    return [index, spikes_7]


def parallel_processing(data):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_channel, range(4), [data]*4))
    print(results)
    index_list, spike_list = zip(*results)
    print(index_list)
    spikes_array = np.array(spike_list, dtype=int).T
    print(index_list)
    return index_list, spikes_array


def prediction(spikes, assignments, n_labels):
    # print(n_samples)

    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_labels)).astype(np.float32)

    # spikes:1*60
    # print(assignments)

    for i in range(n_labels):
        n_assigns = np.sum(assignments == i).astype(np.uint8)
        # print(n_assigns)

        if n_assigns > 0:
            indices = np.where(assignments == i)[0]
        # print(indices)

        # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
        # rates[i] = np.sum(spikes[indices], axis=1) / n_assigns
            rates[i] = np.sum(spikes[indices]) / n_assigns
    # print(rates)

    predicted_label = np.argmax(rates).astype(np.uint8)
    # print(f"Type of predicted_label: {type(predicted_label).__name__}")
    # message = str(int(predicted_label)).encode('utf-8')
    # producer.send('topic-01', message)
    # レイヤーの平均発火率が最も高いラベルを出力
    return predicted_label  # (n_samples, )


def processing(data, network_test, assignments):
    # producer = KafkaProducer(bootstrap_servers=['broker:29092'])
    # GF =100
    # duration = 2000
    # convolve_dt = 0.2
    index_list, spikes_array = parallel_processing(data)
    print(time.time())

    print(index_list)

    A = index_list[0]
    B = index_list[1]
    C = index_list[2]
    D = index_list[3]

    e = np.zeros((10000, 28))

    b = spikes_array

    # bの各列を特定の位置に挿入
    e[:, A*4-4] = b[:, 0]
    e[:, B*4-3] = b[:, 1]
    e[:, C*4-2] = b[:, 2]
    e[:, D*4-1] = b[:, 3]

    # リストに行列と追加のデータを追加
    list_test = e

    # print(list_test)

    # para_dic= {'w_exc': 4.544910490280396, 'w_inh': 1.3082028299035284,
    #  'lr1': 0.0312611326044722, 'lr2': 0.03392142258355323
    # , 'Norm': 0.12856140170464153}
    n_labels = 8  # ラベルの数
    dt = 2e-4  # タイムステップ(sec)
    t_inj = 2.0  # 刺激入力時間(sec)
    nt_inj = round(t_inj/dt)

    n_neurons = 60  # 興奮性/抑制性ニューロンの数
    n_labels = 8  # ラベル数

    spikes_val = np.zeros((1, n_neurons)).astype(np.uint8)
    input_spikes_val = list_test
    spike_list = []  # サンプルごとにスパイクを記録するリスト
    # 画像刺激の入力
    print(time.time())
    for t in range(nt_inj):
        s_exc_test = network_test(input_spikes_val[t], stdp=False)
        spike_list.append(s_exc_test)
    print(time.time())

    spikes_val = np.sum(np.array(spike_list), axis=0)
    # print(spikes_val)
    predicted_labels_val = prediction(spikes_val, assignments, n_labels)
    # predicted_labels_val = prediction(spikes_val, assignments
    # , n_labels, producer)
    print('Val prediction:\n', predicted_labels_val)
    print(time.time())
    # producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
    # value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    # データをJSON形式でエンコードして送信
    # producer.send("final-topic", value=predicted_labels_val)
    # producer.flush()


def main():
    n_in = 28
    n_neurons = 60
    wexc = 4.544910490280396
    winh = 1.3082028299035284
    lr = (0.0312611326044722, 0.03392142258355323)
    norm = 0.12856140170464153
    dt = 2e-4

    network_test = DiehlAndCook2015Network(n_in=n_in, n_neurons=n_neurons,
                                           dt=dt, wexc=wexc, winh=winh,
                                           lr=lr, norm=norm)
    # 学習済みの重みと閾値の読み込み
    weight_path = '/weight_epoch3.npy'
    theta_path = '/exc_neurons_epoch3.npy'
    assignments_path = '/assignment_epoch3.npy'  # 割り当て情報

    network_test.input_conn.W = np.load(weight_path)
    network_test.exc_neurons.theta = np.load(theta_path)
    network_test.exc_neurons.theta_plus = 0

    assignments = np.load(assignments_path)

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
        print(time.time())
        data_str = json.loads(message.value)
        data_list = ast.literal_eval(data_str)
        data_array = np.array(data_list)
        processing(data_array, network_test, assignments)


if __name__ == '__main__':
    main()
