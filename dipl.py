from tensorflow import keras
import tvm
from tvm import relay
import numpy as np
import tvm.testing
import json
import zmq

class LSTMPredictor:

    def compile(self):
        self.model = keras.models.load_model('./lstm2.keras')
        shape_dict = {"lstm_2_input": (1,1,36)}
        mod, params = relay.frontend.from_keras(self.model, shape_dict)
        target = "llvm"
        dev = tvm.device('llvm')
        with tvm.transform.PassContext(opt_level=3):
            self.model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

    def predict(self, move_num):
        if self.sequence_ind>= self.max_seq_len:
            return []

        dtype = "float32"
        x = len(move_num)+1
        if move_num in self.move_map:
            x = self.move_map[move_num]
        else:
            self.move_map[move_num] = x
        self.sequence[self.sequence_ind] = x
        self.sequence_ind +=1
        tvm_out = self.model(tvm.nd.array([[self.sequence]].astype(dtype))).numpy()[0]
        indices = np.argpartition(tvm_out, -3)[-3:]
        rez = []
        for x in indices:
            rez.append((tvm_out[x]*100, self.class_map[str(x)]))

        def sortFunc(e):
            return e[0]
        rez.sort(key=sortFunc,reverse=True)

        return rez

    def __init__(self):
        self.compile()
        self.sequence = [-1.0]*36
        self.sequence_ind=0
        self.max_seq_len = 36
        with open('class_map.json') as json_file:
            self.class_map = json.load(json_file)
        with open('move_map.json') as json_file:
            self.move_map = json.load(json_file)

lstm = LSTMPredictor()
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    bytes_received = socket.recv(3136)
    array_received = np.frombuffer(bytes_received,dtype=np.float32)
    print("RECEIVED: ",array_received)
    pred = lstm.predict(array_received)
    print("PRED: ",pred)
    bytes_to_send = pred.tobytes()
    socket.send(bytes_to_send)
