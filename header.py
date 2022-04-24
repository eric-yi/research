#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import logging
import __main__ as main
import time
from enum import Enum
import collections
import inspect
from datetime import datetime
import re
from functools import reduce
import uuid
from collections import OrderedDict
try:
    import numpy as np
    import pandas as pd
    import csv
    import json
except Exception as e:
    pass

FORMAT = '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

WORKDIR = os.path.dirname(__file__)


LOGO = f'''                                                             
,------.                                           ,--.      
|  .--. ' ,---.  ,---.  ,---.  ,--,--.,--.--. ,---.|  ,---.  
|  '--'.'| .-. :(  .-' | .-. :' ,-.  ||  .--'| .--'|  .-.  | 
|  |\  \ \   --..-'  `)\   --.\ '-'  ||  |   \ `--.|  | |  | 
`--' '--' `----'`----'  `----' `--`--'`--'    `---'`--' `--' 
                                                              '''


class TermStyle:
    RESET = '\33[0m'
    BOLD = '\33[1m'
    ITALIC = '\33[3m'
    URL = '\33[4m'
    BLINK = '\33[5m'
    BLINK2 = '\33[6m'
    SELECTED = '\33[7m'
    BLACK = '\33[30m'
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    WHITE = '\33[37m'
    BLACKBG = '\33[40m'
    REDBG = '\33[41m'
    GREENBG = '\33[42m'
    YELLOWBG = '\33[43m'
    BLUEBG = '\33[44m'
    VIOLETBG = '\33[45m'
    BEIGEBG = '\33[46m'
    WHITEBG = '\33[47m'
    GREY = '\33[90m'
    RED2 = '\33[91m'
    GREEN2 = '\33[92m'
    YELLOW2 = '\33[93m'
    BLUE2 = '\33[94m'
    VIOLET2 = '\33[95m'
    BEIGE2 = '\33[96m'
    WHITE2 = '\33[97m'
    GREYBG = '\33[100m'
    REDBG2 = '\33[101m'
    GREENBG2 = '\33[102m'
    YELLOWBG2 = '\33[103m'
    BLUEBG2 = '\33[104m'
    VIOLETBG2 = '\33[105m'
    BEIGEBG2 = '\33[106m'
    WHITEBG2 = '\33[107m'


def get_screen_width():
    from screeninfo import get_monitors
    monitors = list(filter(lambda monitor: monitor.is_primary, get_monitors()))
    if len(monitors) > 0:
        monitor = monitors[0]
        return monitor.width
    return -1


def pretty_label(width, color=TermStyle.GREEN2):
    return '{color}{label_line}{reset}'.format(
        color=color, label_line=''.join(('*' for _ in range(width))),
        reset=TermStyle.RESET)


def color_line(line, **kwargs):
    border = kwargs['border'] if 'border' in kwargs else False
    offset = kwargs['offset'] if 'offset' in kwargs else 0
    color = kwargs['color'] if 'color' in kwargs else TermStyle.BLUE2
    label_color = kwargs['label_color'] if 'label_color' in kwargs else TermStyle.GREEN2
    width = len(line) + offset * 2
    color_line = color_string(line, width, offset, color, label_color)
    if border:
        label = pretty_label(width, label_color)
        return f'{label}\n{color_line}\n{label}'
    return color_line


def color_lines(lines, **kwargs):
    label_color = kwargs['label_color'] if 'label_color' in kwargs else TermStyle.GREEN2
    line_color = kwargs['line_color'] if 'line_color' in kwargs else TermStyle.BLUE2
    offset = kwargs['offset'] if 'offset' in kwargs else 0
    width = len(reduce(lambda s1, s2: s1 if len(s1) > len(s2)
                       else s2, lines.splitlines())) + offset * 2
    label = pretty_label(width, label_color)
    colors = [label]
    for line in lines.splitlines():
        colors.append(color_string(
            line, width, offset, line_color, label_color))
    colors.append(label)
    return '\n'.join(colors)


def color_string(s, width, offset, color, label_color):
    prefix = ''
    if offset > 0:
        prefix = f'{label_color}*{TermStyle.RESET}'
        prefix += ' '.join(['' for _ in range(offset)])
    postfix = ''
    if offset > 0:
        postfix = ' '.join(['' for _ in range(width - offset - len(s))])
        postfix += f'{label_color}*{TermStyle.RESET}'
    return f'{prefix}{color}{s}{TermStyle.RESET}{postfix}'


def color_slogan():
    return color_lines(SLOGAN, label_color=TermStyle.BLUE2, line_color=TermStyle.RED2, offset=2)


def shell(statement, **kwargs):
    print(color_line(statement))
    from subprocess import Popen, PIPE, STDOUT
    from threading import Thread
    waiting = kwargs['waiting'] if 'waiting' in kwargs else True
    valued = kwargs['valued'] if 'valued' in kwargs else False
    cwd = kwargs['cwd'] if 'cwd' in kwargs else WORKDIR
    logger.debug(f'run at "{cwd}"')

    class Output():
        pid = None
        stdout = []
        stderr = []

        def __repr__(self):
            return str(self.__dict__)

        def single_value(self):
            return self.stdout[0] if len(self.stdout) > 0 else ''

    def log_subprocess_output(processor, output):
        with processor.stdout:
            for data in iter(processor.stdout.readline, b''):
                line = data.decode('utf-8').rstrip()
                output.stdout.append(line)
                print(line)
        processor.wait()

    try:
        output = Output()
        processor = Popen(statement, cwd=cwd, shell=True,
                          stdout=PIPE, stderr=STDOUT)
        output.pid = processor.pid
        thread = Thread(target=log_subprocess_output,
                        args=(processor, output,))
        thread.setDaemon(not waiting)
        thread.start()
        if waiting:
            thread.join()
        if valued:
            return output.single_value()
        return output
    except KeyboardInterrupt:
        logger.error(f'programming interrupt by ctrl-c, shell:{statement}')
    except Exception as err:
        logger.error(f'execute shell:{statement} exception: {err}')


def now():
    return datetime.now()


def list_files(dir):
    return list(
        map(lambda f: os.path.join(os.path.abspath(dir), f), filter(lambda f: not os.path.isdir(f), os.listdir(dir))))


def filename_of(path):
    return os.path.basename(path).split('.')[0]


def read(filepath):
    with open(filepath, 'r') as fd:
        return fd.read()


def write(filepath, content):
    with open(filepath, 'w') as fd:
        fd.write(content)
        fd.flush()


def url_to_filename(url):
    from urllib.parse import urlparse
    u = urlparse(url)
    site = u.netloc
    path = u.path.replace('/', '_').split('.')[0]
    return f'{site}{path}'


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Matcher:
    @staticmethod
    def ChineseMatcher(s):
        return Matcher._match_first_(s, r'[\u4e00-\u9fa5]+')

    @staticmethod
    def ChineseMatcher(s):
        return Matcher._match_first_(s, r'[\u4e00-\u9fa5]+')

    @staticmethod
    def MobileMatcher(s):
        return Matcher._match_first_(s, r'1[358]\d+')

    @staticmethod
    def TelMatcher(s):
        return Matcher._match_first_(s, r'\(?0\d{2,3}[)-]?\d{7,8}')

    @staticmethod
    def IntegerMatcher(s):
        return Matcher._match_first_(s, r'\d+,?\d+')

    @staticmethod
    def AddressMatcher(s):
        return Matcher._match_first_(s, r'[\u4e00-\u9fa5a-zA-Z0-9]+')

    @staticmethod
    def find(s, pattern):
        group = re.findall(pattern, s)
        if len(group) > 0:
            return group[0]
        return []

    @staticmethod
    def _match_first_(s, pattern):
        matches = re.findall(pattern, s)
        if len(matches) > 0:
            return matches[0]
        return ''


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.now = now()

    def elapse(self):
        class TimeDelta:
            def __init__(self, delta):
                self.delta = delta
                self.s = round(delta.total_seconds(), 3)
                self.ms = self.s * 1000

            def __repr__(self):
                return f'{self.s} seconds'

        delta = now() - self.now
        return TimeDelta(delta)

    @staticmethod
    def code():
        return now().strftime("%Y%d%m%H%M%S")

    @staticmethod
    def timestamp():
        return now().timestamp()

    @staticmethod
    def from_string(s, fmp='%Y-%m-%d %H:%M:%S'):
        return datetime.strptime(s, fmp)

    @staticmethod
    def from_date_string(s):
        return Timer.from_string(s, '%Y-%m-%d')

    @staticmethod
    def to_string(d, fmt):
        return d.strftime(fmt)

    @staticmethod
    def to_date_string(d):
        return Timer.to_string(d, '%Y-%m-%d')


try:
    class ObjectEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "_dict_"):
                return self.default(obj.to_json())
            elif hasattr(obj, "__dict__"):
                d = dict(
                    (key, value)
                    for key, value in inspect.getmembers(obj)
                    if not key.startswith("__")
                    and not inspect.isabstract(value)
                    and not inspect.isbuiltin(value)
                    and not inspect.isfunction(value)
                    and not inspect.isgenerator(value)
                    and not inspect.isgeneratorfunction(value)
                    and not inspect.ismethod(value)
                    and not inspect.ismethoddescriptor(value)
                    and not inspect.isroutine(value)
                )
                return self.default(d)
            return obj
except Exception as e:
    pass


def html_filepath(district=None):
    filename = f'living_shanghai_2022_{Timer.code()}'
    if district:
        filename += f'_{district}'
    return os.path.join(MANUAL_DIR, f'{filename}.html')


class UnitTests:
    def __init__(self, filepath=__file__):
        self.name = self.__class__.__name__
        self.cls = self.__class__.__mro__[0]
        self.filepath = filepath

    @property
    def _skip_(self):
        return self.filepath != main.__file__

    def _setup_(self):
        pass

    def _teardown_(self):
        pass

    def run(self):
        if self._skip_:
            return
        self._setup_()

        fns = [getattr(self.cls, fn)
               for fn in dir(self.cls) if not fn.startswith('__')]
        for fn in fns:
            if inspect.isfunction(fn):
                fun_name = fn.__name__
                annotations = fn.__annotations__
                if 'test' in annotations or fun_name.endswith('_test'):
                    if 'test' in annotations and annotations['test'] == 'skip':
                        print(color_line(
                            f'== {self.name}:{fn.__name__} skipped ==', color=TermStyle.RED2))
                        continue
                    print(color_line(
                        f'== {self.name}:{fn.__name__} run ==', color=TermStyle.GREEN2))
                    timer = Timer()
                    eval(f'self.{fn.__name__}()')
                    print(color_line(f'{self.name}:{fn.__name__} elapse {timer.elapse()}', offset=4, border=True,
                                     color=TermStyle.GREEN2, label_color=TermStyle.BLUE2))

        self._teardown_()

    @staticmethod
    def test(fn):
        fn.__annotations__['test'] = 'normal'
        return fn

    @staticmethod
    def skip(fn):
        fn.__annotations__['test'] = 'skip'
        return fn


class HeaderUnitTests(UnitTests):
    def __init__(self):
        super().__init__(__file__)

    @UnitTests.skip
    def shell_test(self):
        output = shell('ls')
        assert output is not None
        logger.debug(f'$ ls \n {output}')
        assert output.pid > 0
        output = shell('ls', waiting=True)
        logger.debug(f'$ ls \n {output}')
        assert output.pid > 0

    @UnitTests.skip
    def list_files_test(self):
        files = list_files(WORKDIR)
        assert len(files) > 0
        print(files)
        files = list_files(os.path.join('data', 'input', '0000001'))
        assert len(files) > 0
        print(files)

    @UnitTests.skip
    def filename_test(self):
        name = filename_of('data/input/0000001/001.csv')
        print(name)
        assert name == '001'

    @UnitTests.skip
    def read_test(self):
        tpl_body = read(os.path.join(TEMPlATE_DIR, 'living_manual_body.tpl'))
        assert tpl_body is not None
        logger.debug(tpl_body)

    @UnitTests.skip
    def timer_test(self):
        timer = Timer()
        time.sleep(1)
        time_delta = timer.elapse()
        assert time_delta.s >= 1
        assert time_delta.ms >= 1000
        logger.debug(time_delta)
        d = Timer.from_date_string('2022-4-1')
        assert Timer.to_date_string(d) == '2022-04-01'

    @UnitTests.skip
    def color_string_test(self):
        s = color_line('one line string')
        print(s)
        s = color_line('one line string with offset', offset=4)
        print(s)
        s = color_line('one line string with offset and border',
                       offset=4, border=True)
        print(s)
        s = color_lines(
            '\n'.join(['multi line string', 'multi line string', 'multi line string']))
        print(s)
        s = color_lines('\n'.join(
            ['multi line string with offset', 'multi line string with offset', 'multi line string with offset']),
            offset=4)
        print(s)


HeaderUnitTests().run()
