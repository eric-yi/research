#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path

from header import *
import argparse
import signal
import json


class ServiceStatus(Enum):
    IDLE = 'idle'
    RUNNING = 'running'


def color_print(s):
    print(color_line(s, offset=4, border=True))


class Service:
    def __init__(self, name='service', command=None):
        self.name = name
        self.command = command
        self.pid_path = os.path.join(WORKDIR, f'.{name}.pid')
        self.info = None
        self.status = ServiceStatus.IDLE
        self.parameters = {}

    def start(self):
        self.status = ServiceStatus.RUNNING
        self._print_action_('start')
        if os.path.isfile(self.pid_path):
            logger.warning(f'{self.name} is running')
            return
        self.info = shell(self.command, waiting=False)
        if self.info is not None:
            pid = str(self.info.pid)
            logger.info(f'start {self.name} at {pid}')
            with open(self.pid_path, 'w') as pid_file:
                pid_file.write(pid)

    def stop(self):
        self._print_action_('stop')
        if os.path.exists(self.pid_path):
            with open(self.pid_path, 'r') as pid_file:
                pid = int(pid_file.read())
            if pid is not None:
                logger.info(f'stop {self.name} at {pid}')
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.remove(self.pid_path)
                except Exception as e:
                    logger.error(e)
        else:
            logger.warning(f'{self.name} is not running')
        self.status = ServiceStatus.IDLE

    def restart(self):
        self.stop()
        self.start()

    def _print_action_(self, action):
        color_print(f'{action} {self.name}')


class Action(Enum):
    INFO = 'info'
    SETUP = 'setup'
    INIT = 'init'
    START = 'start'
    STOP = 'stop'
    RESTART = 'restart'

    @staticmethod
    def from_args(args):
        if args.action:
            for action in Action:
                if action.value == args.action:
                    return action
            return None
        return None


class ServiceFactory:
    def __init__(self):
        self.services = [
            Service('notebook', 'jupyter lab --ip=0.0.0.0'),
        ]

    @property
    def exist_conda(self):
        conda_home = os.path.join(os.path.expanduser('~'), 'miniconda')
        if os.path.exists(conda_home):
            logger.debug('miniconda exists')
            conda_bin_files = list_files(os.path.join(conda_home, 'bin'))
            for conda_bin_file in conda_bin_files:
                self.__setattr__(os.path.basename(
                    conda_bin_file), conda_bin_file)
            return True
        return False

    def execute(self, action):
        if action == Action.INFO:
            if self.exist_conda:
                shell(f'{self.conda} info --envs', waiting=True, valued=True)
                shell(f'python --version', waiting=True, valued=True)
            return

        if action == Action.SETUP:
            install_conda_scripts = (
                'wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh',
                'bash ~/miniconda.sh -b -p $HOME/miniconda',
                'rm -rf ~/miniconda.sh',
                '~/miniconda/bin/conda init bash',
                '~/miniconda/bin/conda init zsh',
            )
            for install_conda_script in install_conda_scripts:
                shell(install_conda_script, waiting=True, valued=True)

            return

        if action == Action.INIT:
            shell('python3 -m pip install -r requirements.txt',
                  waiting=True, valued=True)
            if self.exist_conda:
                conda_libs = (
                    'jupyterlab',
                    'jupyterlab-drawio',
                    'jupyterlab-spreadsheet',
                    'jupyterlab-variableInspector',
                    'jupyter-matplotlib',
                    'jupyterlab-plotly',

                )
                for conda_lib in conda_libs:
                    try:
                        shell(f'{self.conda} install -c conda-forge {conda_lib}',
                              waiting=True, valued=True)
                    except Exception as e:
                        logger.warn(e)

            return

        for service in self.services:
            self._call_(service, action)

    def _call_(self, service, action):
        call_of_service = getattr(service, action.value)
        call_of_service()


def run(args):
    service = Service()
    action = Action.from_args(args)
    if action:
        ServiceFactory().execute(action)


if __name__ == '__main__':
    print(color_lines(LOGO, offset=4))
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=[
                        'info', 'setup', 'init', 'start', 'stop', 'restart', 'update'], help='action')
    args = parser.parse_args()
    run(args)
