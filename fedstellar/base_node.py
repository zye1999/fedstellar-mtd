# 
# This file an adaptation and extension of the p2pfl library (https://pypi.org/project/p2pfl/).
# Refer to the LICENSE file for licensing information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import logging
import os
import socket
import sys
from concurrent import futures
from logging import Formatter, FileHandler

import grpc

from fedstellar.messages import NodeMessages
from fedstellar.neighbors import Neighbors
from fedstellar.proto import node_pb2
from fedstellar.proto import node_pb2_grpc


class BaseNode(node_pb2_grpc.NodeServicesServicer):
    """
    This class represents a base node in the network (without **FL**). It is a thread, so it's going to process all messages in a background thread using the CommunicationProtocol.

    Args:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If False, communication will be encrypted.

    Attributes:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If the node is in simulation mode or not. Basically, simulation nodes don't have encryption and metrics aren't sent to network nodes.
        heartbeater (Heartbeater): The heartbeater of the node.
        gossiper (Gossiper): The gossiper of the node.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(self, experiment_name, host="127.0.0.1", port=None, encrypt=False, config=None):
        self.experiment_name = experiment_name
        # Node Attributes
        self.host = socket.gethostbyname(host)
        self.port = port
        self.encrypt = encrypt
        self.simulation = config.participant["scenario_args"]["simulation"]
        self.config = config

        # Message handlers
        self.__msg_callbacks = {}
        self.add_message_handler(NodeMessages.BEAT, self.__heartbeat_callback)

        self.addr = f"{self.host}:{self.port}"

        # Neighbors
        self._neighbors = Neighbors(self.addr, config)

        # Server
        self.__running = False
        opts = [("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_ping_strikes", 0),
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024)]
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=50), options=opts)

        # Logging
        self.log_dir = os.path.join(config.participant['tracking_args']["log_dir"], self.experiment_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = f"{self.log_dir}/participant_{config.participant['device_args']['idx']}"
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        console_handler, file_handler, file_handler_only_debug, exp_errors_file_handler = self.setup_logging(self.log_filename)

        level = logging.DEBUG if config.participant["device_args"]["logging"] else logging.CRITICAL
        logging.basicConfig(level=level,
                            handlers=[
                                console_handler,
                                file_handler,
                                file_handler_only_debug,
                                exp_errors_file_handler
                            ])
        
        # Save the current configuration in config folder
        config_dir = config.participant['tracking_args']["config_dir"]
        with open(f"{config_dir}/participant_{config.participant['device_args']['idx']}.json", 'w') as f:
            f.write(config.to_json())


    def get_addr(self):
        """
        Returns:
            tuple: The address of the node.
        """
        return self.host, self.port

    def get_name(self):
        """
        Returns:
            str: The name of the node.
        """
        return str(self.get_addr()[0]) + ":" + str(self.get_addr()[1])

    def setup_logging(self, log_dir):
        CYAN = "\x1b[0;36m"
        RESET = "\x1b[0m"
        info_file_format = f"%(asctime)s - %(message)s"
        debug_file_format = f"%(asctime)s - %(message)s\n[in %(pathname)s:%(lineno)d]"
        log_console_format = f"{CYAN}[%(levelname)s] - %(asctime)s - {self.get_name()}{RESET}\n%(message)s"

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.config.participant["device_args"]["logging"] else logging.CRITICAL)
        console_handler.setFormatter(Formatter(log_console_format))

        file_handler = FileHandler('{}.log'.format(log_dir), mode='w')
        file_handler.setLevel(logging.INFO if self.config.participant["device_args"]["logging"] else logging.CRITICAL)
        file_handler.setFormatter(Formatter(info_file_format))

        file_handler_only_debug = FileHandler('{}_debug.log'.format(log_dir), mode='w')
        file_handler_only_debug.setLevel(logging.DEBUG if self.config.participant["device_args"]["logging"] else logging.CRITICAL)
        # Add filter to file_handler_only_debug for only add debug messages
        file_handler_only_debug.addFilter(lambda record: record.levelno == logging.DEBUG)
        file_handler_only_debug.setFormatter(Formatter(debug_file_format))

        exp_errors_file_handler = FileHandler('{}_error.log'.format(log_dir), mode='w')
        exp_errors_file_handler.setLevel(logging.WARNING if self.config.participant["device_args"]["logging"] else logging.CRITICAL)
        exp_errors_file_handler.setFormatter(Formatter(debug_file_format))

        return console_handler, file_handler, file_handler_only_debug, exp_errors_file_handler

    #######################
    #   Node Management   #
    #######################

    def assert_running(self, running):
        """
        Asserts that the node is running or not running.

        Args:
            running (bool): True if the node must be running, False otherwise.

        Raises:
            Exception: If the node is not running and running is True, or if the node is running and running is False.
        """
        running_state = self.__running
        if running_state != running:
            raise Exception(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait=False):
        """
        Starts the node: server and neighbors(gossip and heartbeat).

        Args:
            wait (bool): If True, the function will wait until the server is terminated.

        Raises:
            Exception: If the node is already running.
        """
        # Check not running
        self.assert_running(False)
        # Set running
        self.__running = True
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.__server)
        self.__server.add_insecure_port(self.addr)
        logging.info(f"({self.addr}) Starting gRPC thread at {self.addr}...")
        self.__server.start()
        logging.info(f"({self.addr}) gRPC started.")
        # Heartbeat and Gossip
        self._neighbors.start()
        
    def grpc_wait(self):
        logging.info(f"({self.addr}) Waiting for gRPC to terminate...")
        self.__server.wait_for_termination()
        logging.info(f"({self.addr}) gRPC terminated.")

    def stop(self):
        """
        Stops the node: server and neighbors(gossip and heartbeat).

        Raises:
            Exception: If the node is not running.
        """
        logging.info(f"({self.addr}) Stopping node...")
        # Check running
        self.assert_running(True)
        # Stop server
        self.__server.stop(0)
        # Stop neighbors
        self._neighbors.stop()
        # Set not running
        self.__running = False

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr):
        """
        Connects a node to another.

        Args:
            addr (str): The address of the node to connect to.

        Returns:
            bool: True if the node was connected, False otherwise.
        """
        # Check running
        self.assert_running(True)
        # Connect
        logging.info(f"({self.addr}) connecting to {addr}...")
        return self._neighbors.add(addr, handshake_msg=True)

    def get_neighbors(self, only_direct=False, only_undirected=False):
        """
        Returns the neighbors of the node.

        Args:
            only_direct (bool): If True, only the direct neighbors will be returned.

        Returns:
            list: The list of neighbors.
        """
        return self._neighbors.get_all(only_direct=only_direct, only_undirected=only_undirected)

    def disconnect_from(self, addr):
        """
        Disconnects a node from another.

        Args:
            addr (str): The address of the node to disconnect from.
        """
        # Check running
        self.assert_running(True)
        # Disconnect
        logging.info(f"({self.addr}) removing {addr}...")
        self._neighbors.remove(addr, disconnect_msg=True)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def handshake(self, request, _):
        """
        GRPC service. It is called when a node connects to another.
        """
        logging.info(f"({self.addr}) handshake (gRPC) | from {request.addr}")
        if self._neighbors.add(request.addr, handshake_msg=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(
                error="Cannot add the node (duplicated or wrong direction)"
            )

    def disconnect(self, request, _):
        """
        GRPC service. It is called when a node disconnects from another.
        """
        self._neighbors.remove(request.addr, disconnect_msg=False)
        return node_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def send_message(self, request, _):
        """
        GRPC service. It is called when a node sends a message to another.
        More in detail, it is called when a neighbor use your stub to send a message to you.
        Then, you process the message and gossip it to your neighbors.
        """
        # logging.info(f"({self.addr}) received message from {request.source} | {request.cmd} {request.args}")
        # If not processed
        if self._neighbors.add_processed_msg(request.hash):
            # Gossip
            # logging.info(f"({self.addr}) gossiping message from {request.source} | {request.cmd} {request.args}")
            self._neighbors.gossip(request)
            # Process message
            if request.cmd in self.__msg_callbacks.keys():
                try:
                    # logging.info(f"({self.addr}) running callback for {request.cmd} {request.args}")
                    self.__msg_callbacks[request.cmd](request)
                except Exception as e:
                    error_text = f"[{self.addr}] Error while processing command: {request.cmd} {request.args}: {e}"
                    logging.error(error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            else:
                # disconnect node
                logging.error(
                    f"[{self.addr}] Unknown command: {request.cmd} from {request.source}"
                )
                return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")
        return node_pb2.ResponseMessage()

    def add_model(self, request, _):
        raise NotImplementedError

    ####
    # Message Handlers
    ####

    def add_message_handler(self, cmd, callback):
        """
        Adds a function callback to a message.

        Args:
            cmd (str): The command of the message.
            callback (function): The callback function.
        """
        self.__msg_callbacks[cmd] = callback

    def __heartbeat_callback(self, request):
        time = float(request.args[0])
        self._neighbors.heartbeat(request.source, time)
