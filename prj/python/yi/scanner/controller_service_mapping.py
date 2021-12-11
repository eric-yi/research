#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pytest
import os
import sys
import re
from types import DynamicClassAttribute
import csv
import logging
logging.basicConfig(level=logging.DEBUG)


class Rule:
    def __init__(self):
        self.java_src_path = 'src/main/java'
        self.package_root_path = 'com/icbc/devops/flow/engine'
        self.controller_path_prefix =  'application/archive/web'
        self.controller_file_pattern = r'\w+Controller.java'
        self.export = True
        self.export_csv = True
        self.export_graph = True

    def __repr__(self):
        return str(self.__dict__)

rule = Rule()


class Node:
    def __init__(self, name, hit = False, parent=None):
        self.name = name
        self.hit = hit
        self.parent = parent
        self.children = []

    def add(self, child):
        if child is not None and child.hit:
            child.parent = self
            self.children.append(child)

    def __repr__(self):
        return str(self.__dict__)

class ControllerNode(Node):
    def __init__(self, name):
        Node.__init__(self, name)
    
    def parse(self, line):
        if re.match(r'^class\b.+', line):
            self.hit = True
            logging.debug(line)
            return False
        return self.hit

    @DynamicClassAttribute
    def functions(self):
        return self.children

class ControllerFunctionNode(Node):
    def __init__(self, name='unknow', controller=None, url='unknow', method='unknow'):
        Node.__init__(self, name, False, controller)
        self.url = url
        self.method = method
        self.is_funciton = False
        function_header = ''
        self.pattern = r'(public\ ){0,}(@\S+\ ){0,}(\S+\ )(\w+)(.*)'
        self.function_header = ''


    def parse(self, line): 
        if not self.hit:
            if not self.is_funciton:
                if re.match(self.pattern, line):
                    self.is_funciton = True
            if self.is_funciton:
                self.function_header += line.strip()
            if '{' in  line:
                logging.debug(self.function_header)
                function_rule = re.match(self.pattern, self.function_header)
                if function_rule:
                    self.name = function_rule[4]
                self.hit = True
                return False
        return self.hit

    
    @DynamicClassAttribute
    def services(self):
        return self.children

class ServiceFunctionNode(Node):
    def __init__(self, name, controller_function=None):
        Node.__init__(self, name, True, controller_function)

class Scanner:
    def __init__(self, path):
        self.path = path
        self.nodes = []
    
    def export_csv(self, csv_path):
        with open(csv_path, mode='w') as csv_fd:
            csv_writer = csv.writer(csv_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Controller', 'API Function', 'API URL', 'API Http Method', 'Service Functions'])
            for node in self.nodes:
                for function in node.functions:
                    csv_writer.writerow([node.name, function.name, function.url, function.method, '\n'.join([service.name for service in function.services])])

    def export_graph(self, graph_dir):
        import matplotlib.pyplot as plt
        import networkx as nx
           
        def draw_controller_service(controller):

            def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
                if not nx.is_tree(G):
                    raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
                if root is None:
                    if isinstance(G, nx.DiGraph):
                        root = next(iter(nx.topological_sort(G)))
                    else:
                        root = random.choice(list(G.nodes))
                def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
                    if pos is None:
                        pos = {root:(xcenter,vert_loc)}
                    else:
                        pos[root] = (xcenter, vert_loc)
                    children = list(G.neighbors(root))
                    if not isinstance(G, nx.DiGraph) and parent is not None:
                        children.remove(parent)  
                    if len(children)!=0:
                        dx = width/len(children) 
                        nextx = xcenter - width/2 - dx/2
                        for child in children:
                            nextx += dx
                            pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                                vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                                pos=pos, parent = root)
                    return pos
                return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

            def format_name(name):
                return ''.join(list(map(lambda x: f'\n{x[1]}' if x[0] % 10 == 0 else x[1], enumerate(name))))

            G = nx.DiGraph()
            G.clear()
            edge_colors = []
            controller_name = format_name(controller.name)
            G.add_node(controller_name)
            color = 'g'
            for function in controller.functions:
                edge_colors.append(color)
                function_name = format_name(function.name)
                G.add_node(function_name)
                G.add_edge(controller_name, function_name)
                for service in function.services:
                    service_name = format_name(service.name)
                    G.add_node(service_name)
                    G.add_edge(function_name, service_name)
            pos = hierarchy_pos(G)
            nx.draw(G, pos=pos, with_labels=False, arrows=True, node_color='orange', edge_color=edge_colors)
            for node, (x, y) in pos.items():
                plt.text(x, y, node, fontsize=6, ha='center', va='center')

        for node in self.nodes:
            draw_controller_service(node)
            plt.title('{node.name} service mapping')
            plt.savefig(os.path.join(graph_dir, f'{node.name}_service_mapping.png'))
            plt.clf()

    def scan(self):
        class ControllerFileScanner:
            def __init__(self, controller_file):
                self.controller_file = controller_file
                self.node = None

            def scan(self):
                logging.info(f'=== start scan {self.controller_file} ===')
                with open(self.controller_file) as controller_fd:
                    code = controller_fd.readlines()
                    code = self._format_(code)
                    self.node = self._parse__(code)
                    

            def _format_(self, code):
                code = [x.strip() for x in code]
                def keep(line):
                    return re.match(r'.+', line) and not re.match(r'(^\/\/).*', line) and not re.match(r'(^import).*', line)
                code = [x for x in code if keep(x)]
                return code

            def _parse__(self, code):
                controller_node = ControllerNode(os.path.splitext(os.path.basename(self.controller_file))[0])
                controller_function_node = None
                for line in code:
                    if not controller_node.parse(line):
                        continue
                    api = re.match(r'^@(get|post|put|delete|)mapping\(\"(.+)\"\)', line.lower())
                    if api:
                        controller_node.add(controller_function_node)
                        controller_function_node = ControllerFunctionNode('unknow', controller_node, api[2], api[1])
                        logging.debug(controller_function_node)
                        continue
                    if not controller_function_node:
                        continue
                    if not controller_function_node.parse(line):
                        continue
                    service_function = re.match(r'(.+)service\.(.+)\((.*)\).*', line.lower())
                    service_function = re.match(r'(.+\(\b|\b)(\w+)service\.(.+)\((.*)\).*', line.lower())
                    service_function = re.match(r'(\S+\ ){0,}(\w+)service\.(\w+)\((.*)\)\S+', line.lower())
                    if service_function:
                        service_name = service_function[2]
                        if ' ' in service_name:
                            names = service_name.split(' ')
                            service_name = names[-1]
                        service_function = f'{service_name}Service.{service_function[3]}({service_function[4]})'
                        service_function_node = ServiceFunctionNode(service_function)
                        controller_function_node.add(service_function_node)
                controller_node.add(controller_function_node)
                return controller_node 

        controller_files = self._all_controller_files_()
        for controller_file in controller_files:
            controller_file_scanner = ControllerFileScanner(controller_file)
            controller_file_scanner.scan()
            self.nodes.append(controller_file_scanner.node)

    def _all_controller_files_(self):
        controller_package = os.path.join(self.path, rule.java_src_path, rule.package_root_path, rule.controller_path_prefix)
        logging.debug(f'controller package: {controller_package}')
        if not os.path.isdir(controller_package):
            logging.error(f'controller package({controller_package}) is not directory')
            raise
        controller_files = []
        for (dirpath, dirnames, filenames) in os.walk(controller_package):
            for filename in filenames:
                if re.match(rule.controller_file_pattern, filename):
                    controller_file = os.path.join(dirpath, filename)
                    logging.debug(f'controller file: {controller_file}')
                    controller_files.append(controller_file)
        return controller_files


    def __repr__(self):
        return str(self.__dict__)
    

def scan(path):
    logging.debug(f'scan path: {path}, rule: {rule}')
    scanner = Scanner(path)
    scanner.scan()
    logging.debug(scanner.nodes)
    export_path = os.path.join(path, 'export')
    if not os.path.exists(export_path):
        os.mkdir(export_path)
    if rule.export:
        if rule.export_csv:
            scanner.export_csv(os.path.join(export_path, 'export.csv'))
        if rule.export_graph:
            scanner.export_graph(export_path)

class Test(object):
    # @pytest.mark.skip
    def test_scan(self):
        logging.debug('=== test scan ===')
        path = 'src/test'
        scan(path)
        assert True

    @pytest.mark.skip
    def test_parse_controller_funcgtion(self):
        code_lines = '''public @sdfdsf JsonResult<String> createFlow(@patha("flowid") Long flowid,
                                                    @pathsdf("userid) String userid,
                                                    @requestBody FlowDto FlowDto) {
        '''
        pattern = r'(public\ ){0,}(@\S+\ ){0,}(\S+\ )(\w+)(.*)'
        is_funciton = False
        function_header = ''
        for code_line in code_lines.splitlines():
            if re.match(pattern, code_line):
                is_funciton = True
            if is_funciton:
                function_header += code_line.strip()
            if '{' in  code_line:
                is_funciton = False
                break
        logging.debug(function_header)
        function_rule = re.match(pattern, function_header)
        if function_rule:
            logging.debug(function_rule[1])
            logging.debug(function_rule[2])
            logging.debug(function_rule[3])
            logging.debug(function_rule[4])
            logging.debug(function_rule[5])



if __name__ == '__main__':
    logging.info('==== Controller and Service Mapping ===')
    sys.argv.append('test')
    if (len(sys.argv) < 2):
        logging.error('err')
        sys.exit(1)
    if sys.argv[1] == 'test':
        sys.exit(pytest.main([__file__]))
    else:
        scan(sys.argv[1])