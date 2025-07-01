from .Predictor import Predictor
try:
    from thrift.transport import TSocket, TTransport
    from thrift.protocol import TBinaryProtocol
    from thrift.server import TServer
except ImportError:
    raise ImportError("Thrift library is required. Please install it using 'pip install thrift'.")



from multiprocessing import Queue, Process
from threading import Thread, Lock
import queue
from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod
import time
import traceback


HD_T = TypeVar('HD_T',              bound="BaseHandler")

# class ProcessWithQueue(Process, Generic[HD_T]):
#     def __init__(self, handler:"BaseHandler"):
#         super().__init__()
#         self.handler:HD_T = handler

#     def run(self):
#         while True:
#             _input = self.handler.get_input() #从队列中获取
#             _output = self.handler.process_one(_input) #处理
#             self.handler.put_output(_output) #结果放入队列

def encode_image_to_binary(image) -> bytes:
    """
    将图像编码为 PNG 格式的字节流，适合用于 Thrift 的 binary 传输
    :param image: 输入图像（np.ndarray）
    :return: 编码后的字节流（bytes）
    """
    import cv2
    # 压缩等级：cv2.IMWRITE_PNG_COMPRESSION 范围 0-9（0 无压缩）
    success, encoded_image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    return encoded_image.tobytes()

def decode_binary_to_image(binary_data: bytes):
    """
    从二进制字节流解码为 OpenCV 图像
    :param binary_data: 输入字节流（bytes）
    :return: 解码后的图像（np.ndarray）
    """
    import cv2
    import numpy as np
    np_data = np.frombuffer(binary_data, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode image from binary data.")
    return image

def _put(_queue:Queue, item, block = True, timeout:float = None):
    try:
        _queue.put(item, block, timeout=timeout)
        return True
    except queue.Full:
        return False

def _get(_queue:Queue, block = True, timeout:float = None):
    try:
        return _queue.get(block, timeout=timeout)
    except queue.Empty:
        return None

def _clear(_queue:Queue):
    """
    清空队列中的所有元素
    :param _queue: 要清空的队列
    """
    while not _queue.empty():
        _queue.get()

# def _get(_queue:Queue, _gen_default:Callable, block = True):
#     try:
#         return _queue.get(block)
#     except queue.Empty:
#         return _gen_default()

class ThreadWithQueue(Thread, Generic[HD_T]):
    def __init__(self, func:Callable, in_queue:Queue, out_queue:Queue):
        super().__init__()
        self.func      = func
        self.in_queue  = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            _input = _get(self.in_queue, block=True)
            if _input is None:
                time.sleep(0.001)  # 如果队列为空，稍作等待
                continue
            _output = self.func(_input)
            _put(self.out_queue, _output, block=True)

class ProcessWithQueue(Process, Generic[HD_T]):
    def __init__(self, func:Callable, in_queue:Queue, out_queue:Queue):
        super().__init__()
        self.func      = func
        self.in_queue  = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            _input = _get(self.in_queue, block=True)
            _output = self.func(_input)
            _put(self.out_queue, _output, block=True)

DATA_T = TypeVar('DATA_T')
INFO_T = TypeVar('INFO_T')
RECV_T = TypeVar('RECV_T')
INPUT_T = TypeVar('INPUT_T')
OUTPUT_T = TypeVar('OUTPUT_T')
SEND_T = TypeVar('SEND_T')


class ThreadWatcher(Thread):
    def __init__(self, handler:"BaseHandler"):
        super().__init__()
        self.handler = handler

    def run(self):
        while True:
            if self.handler._pause:
                time.sleep(0.1)  # 暂停时稍作等待
                continue
            
            print(f"Receiving queue size: {self.handler._receiving_queue.qsize()}/{self.handler._queue_length}")
            print(f"Input queue size: {self.handler._input_queue.qsize()}/{self.handler._queue_length}")
            print(f"Output queue size: {self.handler._output_queue.qsize()}/{self.handler._queue_length}")
            print(f"Sending queue size: {self.handler._sending_queue.qsize()}/{self.handler._queue_length}")
            print()

            if self.handler._last_time_send != 0.0 and time.time() - self.handler._last_time_send > 5.0:
                print("No data sent in the last 5 seconds, clearing queues...")
                self.handler.clear_queue()
                time.sleep(1)
                self.handler.clear_queue()
                self.handler._last_time_send = 0.0  # 重置最后发送时间

            time.sleep(1)  # 每秒检查一次

class BaseHandler(Generic[RECV_T, INPUT_T, OUTPUT_T, SEND_T, HD_T], ABC):
    TEST = False

    def __init__(self, queue_length = 10) -> None:
        # 创建4个队列，分别用于接受、输入、输出、发送
        # 创建两个子线程，接受->输入，输出->发送
        self._queue_length = queue_length
        self._receiving_queue:Queue[RECV_T]     = Queue(queue_length)
        self._input_queue:Queue[INPUT_T]        = Queue(queue_length)
        self._output_queue:Queue[OUTPUT_T]      = Queue(queue_length)
        self._sending_queue:Queue[SEND_T]       = Queue(queue_length)

        # self.sub_process = ProcessWithQueue(self)
        self.thread_pre_process:ThreadWithQueue[HD_T]  = ThreadWithQueue(self.pre_process, self._receiving_queue, self._input_queue)
        self.thread_process:ThreadWithQueue[HD_T]      = ThreadWithQueue(self.process, self._input_queue, self._output_queue)
        self.thread_post_process:ThreadWithQueue[HD_T] = ThreadWithQueue(self.post_process, self._output_queue, self._sending_queue)
        self.thread_watcher = ThreadWatcher(self)

        self._last_time_send = 0.0

        self._pause = False

    @abstractmethod
    def pre_process(self, _input:RECV_T) -> INPUT_T:
        pass

    @abstractmethod
    def process(self, _input:INPUT_T) -> OUTPUT_T:
        pass

    @abstractmethod
    def post_process(self, _input:OUTPUT_T) -> SEND_T:
        pass

    @staticmethod
    def print_decorator(func:Callable) -> Callable:
        def wrapper(obj:BaseHandler, *args, **kwargs):
            if obj.TEST:
                print(f"--enter {func.__name__}")
            rlt = func(obj, *args, **kwargs)
            if obj.TEST:
                print(f"--exit {func.__name__}")
            return rlt
        return wrapper

    def clear_queue(self):
        while (not self._receiving_queue.empty()):
            self._receiving_queue.get()
        while (not self._input_queue.empty()):
            self._input_queue.get()
        while (not self._output_queue.empty()):
            self._output_queue.get()
        while (not self._sending_queue.empty()):
            self._sending_queue.get()

    def put_receiving(self, data:RECV_T, block = True):
        """
        将接收的数据放入接收队列
        :param data: 接收的数据
        :param block: 是否阻塞等待
        """
        return _put(self._receiving_queue, data, block)
    
    def get_sending(self, block = True) -> SEND_T:
        self._last_time_send = time.time()
        return _get(self._sending_queue, block=block)

    def start(self):
        self.thread_pre_process.start()
        self.thread_process.start()
        self.thread_post_process.start()
        self.thread_watcher.start()


class DataWithInfo(Generic[DATA_T, INFO_T]):
    def __init__(self, data: DATA_T = None, info: INFO_T = None):
        self.data = data
        self.info = info

# class ThreadJoinDataWithInfo(JoinMutilChannels["InteGrasperHandler"]):
#     def __init__(self, handler:"InteGrasperHandler"):
#         super().__init__(handler.receiving_queue, 2, DataWithInfo)

#     def put_data(self, data:DATA_T):
#         self.queue_list[0].put(data, block=True)

#     def put_info(self, info:INFO_T):
#         self.queue_list[1].put(info, block=True)

# class ThreadSplitDataWithInfo(SplitMutliChannels["InteGrasperHandler"]):
#     def __init__(self, handler:"InteGrasperHandler"):
#         super().__init__(handler.sending_queue, ["data", "info"])

#     def get_data(self) -> DATA_T:
#         return self.get(0)

#     def get_info(self) -> INFO_T:
#         return self.get(1)


class InteGrasperHandlerMetaClass(type(BaseHandler)):
    __RECV_STRUCT__:type = None
    __SEND_STRUCT__:type = None

    def __new__(cls, name, bases, dct):
        # 强制要求在类定义时，必须提供一个叫 __RECV_STRUCT__ 的属性
        if '__RECV_STRUCT__' not in dct:
            raise TypeError(f"Class '{name}' must define a '__RECV_STRUCT__' attribute")
        if '__SEND_STRUCT__' not in dct:
            raise TypeError(f"Class '{name}' must define a '__SEND_STRUCT__' attribute")
        
        # 检查 __RECV_STRUCT__ 和 __SEND_STRUCT__ 是否具有data 和 info 属性
        for _type in [dct['__RECV_STRUCT__'], dct['__SEND_STRUCT__']]:
            if _type is None and name == "InteGrasperHandler":
                continue  # 如果是基类，可以不检查
            obj = _type()
            assert hasattr(obj, 'data'), f"{_type} must have a 'data' attribute in class '{name}'"
            assert hasattr(obj, 'info'), f"{_type} must have an 'info' attribute in class '{name}'"
        
        return super().__new__(cls, name, bases, dct)
        
class InteGrasperHandler(BaseHandler[DataWithInfo[RECV_T, INFO_T], 
                                     DataWithInfo[INPUT_T, INFO_T], 
                                     DataWithInfo[OUTPUT_T, INFO_T], 
                                     DataWithInfo[SEND_T, INFO_T], "InteGrasperHandler"], metaclass=InteGrasperHandlerMetaClass):
    __RECV_STRUCT__:type = None
    __SEND_STRUCT__:type = None
    
    def __init__(self, predictor:Predictor, queue_length=10):
        super().__init__(queue_length)
        self.predictor = predictor
        self.inverse_kinematics = Co605inverse_kinematics()
        self.ifsave_recv = False  # 是否保存接收数据到文件
        self.ifsave_input = False  # 是否保存输入数据到文件

    @abstractmethod
    def cvtRecvToInput(self, recv_data:RECV_T) -> INPUT_T:
        pass

    @abstractmethod
    def cvtOutputToSend(self, output_data:OUTPUT_T) -> SEND_T:
        pass

    def pre_process(self, recv:DataWithInfo[RECV_T, INFO_T]) -> DataWithInfo[INPUT_T, INFO_T]:
        if self.ifsave_recv:
            import pickle
            with open(f"recv_data-{time.time()}.pkl", "wb") as f:
                pickle.dump(recv, f)
        _input_data = self.cvtRecvToInput(recv.data)
        _input = DataWithInfo[INPUT_T, INFO_T](_input_data, recv.info)
        return _input

    def process(self, _input:DataWithInfo[INPUT_T, INFO_T]) -> DataWithInfo[OUTPUT_T, INFO_T]:
        if self.ifsave_input:
            import pickle
            with open(f"input_data-{time.time()}.pkl", "wb") as f:
                pickle.dump(_input, f)
        if _input is None:
            return None

        try:
            result = self.predictor.predict_one(_input.data)
        except Exception as e:
            print("Error in process:")
            traceback.print_exc()

            result = None

        return DataWithInfo[OUTPUT_T, INFO_T](result, _input.info)

    def post_process(self, output:DataWithInfo[OUTPUT_T, INFO_T]) -> DataWithInfo[SEND_T, INFO_T]:
        if output is None:
            return None
        send_data = self.cvtOutputToSend(output.data)
        return self.__SEND_STRUCT__(send_data, output.info)
    
    def flow_test_from_input(self, pkl):
        if isinstance(pkl, str):
            import pickle
            with open(pkl, "rb") as f:
                datas = pickle.load(f)
        else:
            datas = pkl
        result = self.process(datas)
        result = self.post_process(result)
        return result

    def flow_test_from_receiving(self, pkl:str):
        if isinstance(pkl, str):
            import pickle
            with open(pkl, "rb") as f:
                datas = pickle.load(f)
        else:
            datas = pkl
        result = self.pre_process(datas)
        result = self.process(result)
        result = self.post_process(result)
        return result
        
import numpy as np
from scipy.spatial.transform import Rotation as R
class Co605inverse_kinematics():
    def __init__(self):
        self.dh_params = np.array(
            [[0.000/180*np.pi,   321.500/1000,     0.000/1000  ,    -90.000/180*np.pi ],
            [-90.000/180*np.pi,   0.000,           400.000/1000,    0.000/180*np.pi  ],
            [180.000/180*np.pi,   0.000,           0.000/1000  ,    90.000/180*np.pi ],
            [0.000/180*np.pi  ,   400.000/1000,    0.000/1000  ,    -90.000/180*np.pi],
            [90.000/180*np.pi ,   0.000,           0.000/1000  ,    90.000/180*np.pi ],
            [0.000/180*np.pi  ,   205.000/1000,    0.000/1000  ,    0.000/180*np.pi  ]])
        
        self.ref_Jpos = np.array([0, -np.pi/2, np.pi, 0.0, np.pi/2, 0.0])

        self.EE_rotation_symmetry:int = 1  # 末端执行器旋转对称角度，默认不对称

        self.Joint_limits = np.array([[ -180.0 / 180.0 * np.pi, -220.0 / 180 * np.pi, -55.0 / 180.0 * np.pi, -180.0 / 180.0 * np.pi, -135.0 / 180.0 * np.pi, -180.0 / 180.0 * np.pi ],
                                      [  180.0 / 180.0 * np.pi,  40.0 / 180 * np.pi,   235.0 / 180.0 * np.pi,  180.0 / 180.0 * np.pi,  135.0 / 180.0 * np.pi,  180.0 / 180.0 * np.pi ]])

    def run(self, xyz_ZYXeuler, ref_Jpos=None, return_diff = False):
        ref_Jpos = ref_Jpos if ref_Jpos is not None else self.ref_Jpos

        _input_matrix = False
        if isinstance(xyz_ZYXeuler, tuple):
            assert len(xyz_ZYXeuler) == 2
            t = np.array(xyz_ZYXeuler[0])
            eular = np.array(xyz_ZYXeuler[1])
        elif isinstance(xyz_ZYXeuler, np.ndarray):
            if xyz_ZYXeuler.shape == (6, ):
                t = xyz_ZYXeuler[:3]
                eular = xyz_ZYXeuler[3:]
            elif xyz_ZYXeuler.shape == (2, 3):
                t = xyz_ZYXeuler[0]
                eular = xyz_ZYXeuler[1]
            elif xyz_ZYXeuler.shape == (4, 4):
                T06 = xyz_ZYXeuler
                _input_matrix = True
            else:
                raise ValueError("Invalid shape for xyz_ZYXeuler")

        # T06 = trvec2tform * eul2tform
        if not _input_matrix:
            T_trans = np.eye(4)
            T_trans[:3, 3] = t
            T_rot = np.eye(4)
            T_rot[:3, :3] = R.from_euler('ZYX', eular, degrees=False).as_matrix()
            T06 = T_trans @ T_rot

        # 提取位姿分量
        nx, ox, ax, px = T06[0, :]
        ny, oy, ay, py = T06[1, :]
        nz, oz, az, pz = T06[2, :]

        # DH 参数
        d1 = self.dh_params[0, 1]
        d4 = self.dh_params[3, 1]
        d6 = self.dh_params[5, 1]
        a2 = self.dh_params[1, 2]

        # t1（只有一个解，后面扩展成 2x2）默认丢弃反方向的解
        t1_tan = (py - ay * d6) / (px - ax * d6)
        t1_1 = np.arctan(t1_tan)
        t1_2 = t1_1 + np.pi if t1_1 < 0 else t1_1 - np.pi
        t1 = np.array([t1_1, t1_2])  # shape: (2,)
        # t1 = np.full((2, 2), t1)

        # t3（两个解）
        t3_sin = ((d1 - pz + az * d6) ** 2 +
                (px * np.cos(t1) - d6 * (ax * np.cos(t1) + ay * np.sin(t1)) +
                py * np.sin(t1)) ** 2 - a2 ** 2 - d4 ** 2) / (2 * a2 * d4)
        t3_1:np.ndarray = np.arcsin(t3_sin)
        t3_2:np.ndarray = t3_1.copy()
        t3_2[t3_2 > 0] = np.pi - t3_2[t3_2 > 0]
        t3_2[t3_2 < 0] = -np.pi - t3_2[t3_2 < 0]
        t3 = np.array([t3_1, t3_2])

        # t2（广播表达式）
        ct1 = np.cos(t1)
        st1 = np.sin(t1)
        ct3 = np.cos(t3)
        st3 = np.sin(t3)

        sin_t2 = (a2*d1 - a2*pz + d1*d4*np.sin(t3) - d4*pz*np.sin(t3) + a2*az*d6 + az*d4*d6*np.sin(t3) +
                  d4*px*np.cos(t1)*np.cos(t3) + d4*py*np.cos(t3)*np.sin(t1) - ax*d4*d6*np.cos(t1)*np.cos(t3) - ay*d4*d6*np.cos(t3)*np.sin(t1)) \
                /(a2**2 + 2*np.sin(t3)*a2*d4 + d4**2)

        cos_t2 = -(d1*d4*np.cos(t3) - d4*pz*np.cos(t3) - a2*py*np.sin(t1) - a2*px*np.cos(t1) - d4*py*np.sin(t1)*np.sin(t3) + 
                   a2*ax*d6*np.cos(t1) + az*d4*d6*np.cos(t3) + a2*ay*d6*np.sin(t1) - d4*px*np.cos(t1)*np.sin(t3) + 
                   ax*d4*d6*np.cos(t1)*np.sin(t3) + ay*d4*d6*np.sin(t1)*np.sin(t3)) \
                    /(a2**2 + 2*np.sin(t3)*a2*d4 + d4**2)

        t2 = np.arctan2(sin_t2, cos_t2)

        # t5（两个解）
        ct2 = np.cos(t2)
        st2 = np.sin(t2)
        cos_t5 = (az * ct2 * ct3 - az * st2 * st3 +
                ax * ct1 * ct2 * st3 + ax * ct1 * ct3 * st2 +
                ay * ct2 * st1 * st3 + ay * ct3 * st1 * st2)
        t5_1 = np.arccos(cos_t5)
        t5_2 = -t5_1
        t5 = np.stack([t5_1, t5_2], axis=0)  # shape: (2,2)

        # t4
        sin_t4 = (ay * ct1 - ax * st1) / np.sin(t5)
        cos_t4 = -(az * ct2 * st3 + az * ct3 * st2 -
                ax * ct1 * ct2 * ct3 - ay * ct2 * ct3 * st1 +
                ax * ct1 * st2 * st3 + ay * st1 * st2 * st3) / np.sin(t5)
        t4 = np.arctan2(sin_t4, cos_t4)

        # t6
        sin_t6 = (oz * ct2 * ct3 - oz * st2 * st3 +
                ox * ct1 * ct2 * st3 + ox * ct1 * ct3 * st2 +
                oy * ct2 * st1 * st3 + oy * ct3 * st1 * st2) / np.sin(t5)

        cos_t6 = -(nz * ct2 * ct3 - nz * st2 * st3 +
                nx * ct1 * ct2 * st3 + nx * ct1 * ct3 * st2 +
                ny * ct2 * st1 * st3 + ny * ct3 * st1 * st2) / np.sin(t5)

        t6 = np.arctan2(sin_t6, cos_t6)

        t1 = np.tile(t1[:,None,None], (1,2,2))
        t2 = np.stack([t2, t2], 0)
        t3 = np.stack([t3, t3], 0)

        result = np.stack([t1, t2, t3, t4, t5, t6], axis=-1)
        result = np.reshape(result, (-1, 6))  # 展平为 (N, 6) 的形状

        assert isinstance(self.EE_rotation_symmetry, int), "EE_rotation_symmetry must be an integer"
        if self.EE_rotation_symmetry > 1:
            for i in range(1, self.EE_rotation_symmetry):
                temp = result.copy()
                temp[:, 5] += i * 2 * np.pi / self.EE_rotation_symmetry
                result = np.concatenate([result, temp], axis=0)

        # 将result约束在 关节角限位内
        # 排除超限位的

        # 将result约束在 [-pi, pi]范围内
        result = result % (2 * np.pi)  # 将角度限制在 [0, 2π]
        result[result > self.Joint_limits[1]] -= 2 * np.pi  # 将角度转换到 [-π, π]
        # result[result < self.Joint_limits[0]] += 2 * np.pi  # 将角度转换到 [-π, π]

        reachable = np.all((result >= self.Joint_limits[0]) & (result <= self.Joint_limits[1]), axis=1) # 检查可达性 [N]
        reachable = reachable & ~np.isnan(result).any(axis=-1)
        result = result[reachable]  # 只保留可达的解

        if reachable.sum() == 0:
            result = np.full((6, ), np.nan)  # 如果没有可达解，返回 NaN 解
            result_diff = np.nan
        else:
            # 选解，取距离参考姿态最近的
            diff = result - ref_Jpos[None, :]
            diff_weight = np.array([1, 1, 1, 0.2, 0.2, 0.1])  # 权重
            diff[diff < -np.pi] += 2 * np.pi
            diff[diff >  np.pi] -= 2 * np.pi
            diff = np.linalg.norm(diff * diff_weight, axis=-1)
            diff[np.isnan(diff)] = np.inf  # 将 NaN 替换为无穷大，避免影响 argmin

            mindiff_arg = np.argmin(diff, axis=0)
            result = result[mindiff_arg, :] # 选择最小差值的解
            result_diff = diff[mindiff_arg]

        if return_diff:
            return result, result_diff
        else:
            return result

    def forward_kinematics(self, q):
        """
        :param q: 关节角数组，shape=(6,)
        :return: T06 齐次变换矩阵，shape=(4,4)
        """
        T = np.eye(4)
        for i in range(6):
            theta, d, a, alpha = self.dh_params[i]
            theta += q[i]

            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            A = np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0,       sa,      ca,     d   ],
                [0,        0,       0,     1   ]
            ])
            T = T @ A
        return T


class Transporter:
    def __init__(self, ip, port, handler:BaseHandler, ProcessorType:type):
        assert isinstance(handler, BaseHandler), "handler must be an instance of BaseHandler"
        self.handler = handler
        self.processor = ProcessorType(handler)
        self.transport = TSocket.TServerSocket(host="127.0.0.1", port=9090)
        self.tfactory = TTransport.TBufferedTransportFactory()
        self.pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        self.server = TServer.TThreadedServer(
            self.processor, self.transport, self.tfactory, self.pfactory)

    def start(self):
        print(f"Starting server on {self.transport.host}:{self.transport.port}")
        self.handler.start()  # 启动处理线程
        self.server.serve()
        print("Server stopped.")