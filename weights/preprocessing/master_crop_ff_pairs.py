from flask import Flask, request
import os
import json
from threading import Lock
import pickle
from copy import deepcopy

app = Flask(__name__)


def rep(x, codec, fs='c0'):
    y = deepcopy(x)
    y['imgs_path'] = y['imgs_path'].replace(fs, codec)
    y['video_path'] = y['video_path'].replace(fs, codec)
    return y


class job:
    def __init__(self):
        with open('datasets.json') as f:
            data = json.load(f)['ff']
        data_ = []
        for i in data:
            if 'Deepfakes' in i['imgs_path']:
                data_.append(i)
                data_.append(rep(i, 'c23'))
                data_.append(rep(i, 'c40'))
        with open('FF-checked.pkl', 'rb') as f:
            models = pickle.load(f)
        gts = {i['id'][1][:3]: i['gt'] for i in models}
        dd = []
        for (num, i) in enumerate(data_):
            para = {'input_file': i['video_path'], 'aligned_output_dir': i['imgs_path'], 'aligned_image_size': 384,
                    'zoom_in'   : 1, 'gt': gts[i['id'][1][:3]]}
            dd.append([{'para': para, 'num': num}, {'id': '0', 'gt': ''}])
        self.dd = dd
        self.all = len(dd)
        self.sent = 0
        self.recv = 0
        self.mutex = Lock()

    def get(self, num):
        with self.mutex:
            if self.sent == self.all:
                return []
            curr = self.sent
            cnxt = min(curr + num, self.all)
            self.sent = cnxt
        return [self.dd[i][0] for i in range(curr, cnxt)]

    def put(self, x):
        for i in x:
            self.dd[i['num']][1]['gt'] = i['gt']
        with self.mutex:
            self.recv += len(x)
            print('total %s, sent %s, returned %s' % (self.all, self.sent, self.recv))
        if self.recv == self.all:
            return True
        return False

    def dump(self, name):
        with open(name, 'wb') as f:
            pickle.dump([i[1] for i in self.dd], f)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        num = int(request.args.get('num', 100))
        return pickle.dumps(JOB.get(num))
    if request.method == "POST":
        data = request.get_data(cache=False)
        if JOB.put(pickle.loads(data)):
            print('ended')
        #            JOB.dump('result.pkl')
        return 'ok'


if __name__ == '__main__':
    JOB = job()
    app.run(host="0.0.0.0", port='10087')
