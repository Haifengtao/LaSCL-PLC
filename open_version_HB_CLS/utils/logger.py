
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   logger.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/28 13:53   Bot Zhao      1.0         None
"""

# import lib
# import lib
import pandas as pd
import os


class Logger_csv(object):
    def __init__(self, title, out_dir, save_name):
        """
        save the msg during iteration.
        :param title: table's title;
        :param out_dir: output path;
        """
        self.dict = {}
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.save_path = os.path.join(out_dir, save_name)
        for i in title:
            self.dict[i] = []

    def update(self, msg):
        try:
            for i in self.dict.keys():
                self.dict[i].append(msg[i])
        except Exception as err:
            print(err)
        df = pd.DataFrame(self.dict)
        df.to_csv(self.save_path)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    a = ['1','wq']
    logger = Logger_csv(a,'../data2/', 'test.csv')
    x={"1":2121,'wq':32}
    logger.update(x)