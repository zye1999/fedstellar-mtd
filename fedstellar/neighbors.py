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
import random
import threading
import time
import sys
from datetime import datetime

import grpc

from fedstellar.messages import NodeMessages
from fedstellar.proto import node_pb2, node_pb2_grpc


class Neighbors:
    """
    Class that manages the neighbors of a node (GRPC connections). It provides the following functionalities:
        - Add neighbors (check duplicates)
        - Remove neighbors
        - Get neighbors
        - Heartbeat: remove neighbors that not send a heartbeat in a period of time
        - Gossip: resend messages to neighbors allowing communication between non-direct connected nodes

    Args:
        self_addr (str): Address of the node itself.
    """

    def __init__(self, self_addr, config):
        self.__self_addr = self_addr
        self.__config = config
        self.__neighbors = {}  # private to avoid concurrency issues
        self.__neighbors_location = {}  # private to avoid concurrency issues
        self.__nei_lock = threading.Lock()

        # Heartbeat
        self.__heartbeat_terminate_flag = threading.Event()

        # Gossip
        self.__pending_msgs = []
        self.__pending_msgs_lock = threading.Lock()
        self.__gossip_terminate_flag = threading.Event()
        self.__processed_messages = []
        self.__processed_messages_lock = threading.Lock()

    def start(self):
        """
        Start the heartbeater and gossiper threads.
        """
        self.__start_heartbeater()
        self.__start_gossiper()

    def stop(self):
        """
        Stop the heartbeater and gossiper threads. Also, close all the connections.
        """
        self._stop_heartbeater()
        self._stop_gossiper()
        self.clear_neis()

    ####
    # Message
    ####

    def build_msg(self, cmd, args=[], round=None):
        """
        Build a message to send to the neighbors.

        Args:
            cmd (string): Command of the message.
            args (list): Arguments of the message.
            round (int): Round of the message.

        Returns:
            node_pb2.Message: Message to send.
        """
        hs = hash(
            str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000))
        )
        # logging.info(f"({self.__self_addr}) Building message with hash {hs}")
        # logging.info(f"({self.__self_addr}) Building message with hash {cmd}: {args} at round {round} ")
        args = [str(a) for a in args]
        return node_pb2.Message(
            source=self.__self_addr,
            ttl=self.__config.participant["TTL"],
            hash=hs,
            cmd=cmd,
            args=args,
            round=round,
        )

    def send_message(self, nei, msg):
        """
        Send a message to a neighbor.

        Args:
            nei (str): Address of the neighbor.
            msg (node_pb2.Message): Message to send.
        """
        try:
            # logging.info(f"({self.__self_addr}) Sending message (gRPC) {msg.cmd} to {self.__neighbors[nei][1]}")
            res = self.__neighbors[nei][1].send_message(
                msg, timeout=self.__config.participant["GRPC_TIMEOUT"]
            )
            if res.error:
                logging.error(
                    f"[{self.__self_addr}] send_message (gRPC) | Error while sending a message: {msg.cmd} {msg.args}: {res.error}"
                )
                self.remove(nei, disconnect_msg=True)
            else:
                pass
                # logging.info(
                #     f"({self.__self_addr}) send_message (gRPC) | Message {msg.cmd} sent to {nei} with hash {msg.hash}"
                # )
        except Exception as e:
            # Remove neighbor
            logging.error(
                f"({self.__self_addr}) send_message (gRPC) | Cannot send message {msg.cmd} to {nei}. Error: {str(e)}"
            )
            self.remove(nei)

    def broadcast_msg(self, msg, node_list=None):
        """
        Broadcast a message to all the neighbors.

        Args:
            msg (node_pb2.Message): Message to send.
            node_list (list): List of neighbors to send the message. If None, send to all the neighbors.
        """
        # Node list
        if node_list is not None:
            node_list = node_list
        else:
            node_list = self.get_all(only_direct=True)
        # Send
        logging.info(f"({self.__self_addr}) Broadcasting\n{msg}--> to {node_list}")
        for n in node_list:
            self.send_message(n, msg)

    def send_model(self, nei, round, serialized_model, contributors=[], weight=1):
        """
        Send a model to a neighbor.

        Args:
            nei (str): Address of the neighbor.
            round (int): Round of the model.
            serialized_model (bytes): Serialized model.
            contributors (list): List of contributors of the model.
            weight (float): Weight of the model.
        """
        try:
            logging.info(
                f"({self.__self_addr}) Sending model to {nei} with round {round}: contributors={contributors}, weight={weight} | size={sys.getsizeof(serialized_model) / (1024 ** 2) if serialized_model is not None else 0} MB"
            )
            stub = self.__neighbors[nei][1]
            # if not connected, create a temporal stub to send the message
            if stub is None:
                channel = grpc.insecure_channel(nei)
                stub = node_pb2_grpc.NodeServicesStub(channel)
            else:
                channel = None
            res = stub.add_model(
                node_pb2.Weights(
                    source=self.__self_addr,
                    round=round,
                    weights=serialized_model,
                    contributors=contributors,
                    weight=weight,
                ),
                timeout=self.__config.participant["GRPC_TIMEOUT"],
            )
            # Handling errors -> however errors in aggregation stops the other nodes and are not raised (decoding/non-matching/unexpected)
            if res.error:
                logging.error(f"[{self.__self_addr}] Error while sending a model: {res.error}")
                self.remove(nei, disconnect_msg=True)
            if not (channel is None):
                channel.close()

        except Exception as e:
            # Remove neighbor
            logging.info(
                f"({self.__self_addr}) Cannot send model to {nei}. Error: {str(e)}"
            )
            self.remove(nei)

    ####
    # Neighbors management
    ####

    def non_direct_add_node(self, addr):
        """
        Add a non-direct connected neighbor.

        Args:
            addr (str): Address of the neighbor.

        Returns:

        """

        logging.info(f"({self.__self_addr}) Adding non direct connected neighbor {addr}")
        self.__nei_lock.acquire()
        self.__neighbors[addr] = [None, None, time.time()]
        self.__nei_lock.release()
        return True

    def direct_add_node(self, handshake_msg, addr):
        """
        Add a direct connected neighbor.

        Args:
            handshake_msg (bool): If True, send a handshake message to the neighbor.
            addr (str): Address of the neighbor.

        Returns:

        """
        logging.info(f"({self.__self_addr}) Adding direct connected neighbor {addr}")
        try:
            # Create channel and stub
            channel = grpc.insecure_channel(addr)
            stub = node_pb2_grpc.NodeServicesStub(channel)

            # Handshake
            if handshake_msg:
                res = stub.handshake(
                    node_pb2.HandShakeRequest(addr=self.__self_addr),
                    timeout=self.__config.participant["GRPC_TIMEOUT"],
                )
                if res.error:
                    logging.info(
                        f"({self.__self_addr}) Cannot add a neighbor: {res.error}"
                    )
                    channel.close()
                    return False

            # Add neighbor
            self.__nei_lock.acquire()
            self.__neighbors[addr] = [channel, stub, time.time()]
            # Update config
            if self.__config.participant["network_args"]["neighbors"] == "":
                self.__config.participant["network_args"]["neighbors"] = addr
            else:
                if addr not in self.__config.participant["network_args"]["neighbors"]:
                    self.__config.participant["network_args"]["neighbors"] += " " + addr
            self.__nei_lock.release()
            return True

        except Exception as e:
            logging.info(f"({self.__self_addr}) Crash while adding a neighbor: {e}")
            # Try to remove neighbor
            try:
                self.remove(addr)
            except:
                pass
            return False

    def add(self, addr, handshake_msg=True, non_direct=False):
        """
        Add a neighbor if it is not itself or already added. It also sends a handshake message to check if the neighbor is available and create a bidirectional connection.

        Args:
            addr (str): Address of the neighbor.
            handshake_msg (bool): If True, send a handshake message to the neighbor.
            non_direct (bool): If True, add a non-direct connected neighbor (without creating a direct GRPC connection).

        Returns:
            bool: True if the neighbor was added, False otherwise.
        """
        # Cannot add itself
        if addr == self.__self_addr:
            logging.info(f"({self.__self_addr}) Cannot add itself")
            return False

        # Cannot add duplicates
        self.__nei_lock.acquire()
        duplicated = addr in self.__neighbors.keys()
        logging.info(f"({self.__self_addr}) Attempting to add DIRECT connection {addr} (duplicated={duplicated})") if not non_direct else logging.info(f"({self.__self_addr}) Attempting to add NON DIRECT connection {addr} (duplicated={duplicated})")
        self.__nei_lock.release()
        # Avoid adding if duplicated and not non_direct neighbor (otherwise, connect creating a channel)
        if duplicated:
            if not non_direct:  # Upcoming direct connection
                # Duplicated but the node wants to add it as a direct connected neighbor
                # Check if it is already connected as a non-direct connected neighbor.
                # If so, upgrade to direct connected neighbor
                if self.__neighbors[addr][1] is None:
                    logging.info(
                        f"({self.__self_addr}) Upgrading non direct connected neighbor {addr}"
                    )
                    return self.direct_add_node(handshake_msg, addr)
                else:  # Upcoming undirected connection
                    logging.info(f"({self.__self_addr}) Already direct connected neighbor {addr}")
                    return False

            elif non_direct:
                # Duplicated but the node wants to add it as a non-direct connected neighbor
                logging.info(f"({self.__self_addr}) Cannot add a duplicate {addr} (undirected connection), already connected")
                return False
        else:
            # Add non-direct connected neighbors
            if non_direct:
                return self.non_direct_add_node(addr)
            else:
                return self.direct_add_node(handshake_msg, addr)

    def remove(self, nei, disconnect_msg=True):
        """
        Remove a neighbor.

        Args:
            nei (str): Address of the neighbor.
            disconnect_msg (bool): If True, send a disconnect message to the neighbor.
        """
        logging.info(f"({self.__self_addr}) Removing {nei}")
        self.__nei_lock.acquire()
        try:
            try:
                # If the other node still connected, disconnect
                if disconnect_msg:
                    self.__neighbors[nei][1].disconnect(
                        node_pb2.HandShakeRequest(addr=self.__self_addr)
                    )
                # Close channel
                self.__neighbors[nei][0].close()
            except:
                pass
            # Remove neighbor
            del self.__neighbors[nei]
            # Remove neighbor from config
            current_neighbors = self.get_all(only_direct=True)
            logging.info(f"({self.__self_addr}) Current neighbors: {current_neighbors}")
            final_neighbors = ""
            if current_neighbors == nei:
                final_neighbors = ""
            else:
                for n in current_neighbors.split(" "):
                    if n != nei:
                        final_neighbors += n + " "
                # Check if there is a space at the end
                if final_neighbors[-1] == " ":
                    final_neighbors = final_neighbors[:-1]
            self.__config.participant["network_args"]["neighbors"] = final_neighbors
            logging.info(f"({self.__self_addr}) Final neighbors: {final_neighbors} (config updated))")
        except:
            pass
        self.__nei_lock.release()

    def get(self, nei):
        """
        Get a neighbor.

        Args:
            nei (str): Address of the neighbor.

        Returns:
            node_pb2_grpc.NodeServicesStub: Stub of the neighbor.
        """
        return self.__neighbors[nei][1]

    def get_all(self, only_direct=False, only_undirected=False):
        """
        Get all the neighbors (names).

        Args:
            only_direct (bool): If True, get only the direct connected neighbors.

        Returns:
            list: List of neighbor addresses.
        """
        neis = self.__neighbors.copy()
        
        if only_direct and only_undirected:
            return list(neis.keys())
        elif only_direct:
            return [k for k, v in neis.items() if v[1] is not None]
        elif only_undirected:
            return [k for k, v in neis.items() if v[1] is None]
        
        return list(neis.keys())

    def clear_neis(self):
        nei_copy = self.__neighbors.copy()
        for nei in nei_copy.keys():
            self.remove(nei)

    ####
    # Heartbeating
    ####

    def heartbeat(self, nei, time):
        """
        Update the time of the last heartbeat of a neighbor. If the neighbor is not added, add it.

        Args:
            nei (str): Address of the neighbor.
            time (float): Time of the heartbeat.
        """
        # Check if it is itself
        if nei == self.__self_addr:
            return
        # Add / update
        self.__nei_lock.acquire()
        if nei not in self.__neighbors.keys():
            self.__nei_lock.release()
            # Add non-direct connected neighbor
            self.add(nei, non_direct=True)

        else:
            # Update time
            if self.__neighbors[nei][2] < time:
                self.__neighbors[nei][2] = time
            self.__nei_lock.release()

    def __start_heartbeater(self):
        logging.info(f"({self.__self_addr}) Starting heartbeater thread")
        threading.Thread(target=self.__heartbeater).start()

    def _stop_heartbeater(self):
        self.__heartbeat_terminate_flag.set()

    def __heartbeater(
            self
    ):
        period = self.__config.participant["HEARTBEAT_PERIOD"]
        timeout = self.__config.participant["HEARTBEAT_TIMEOUT"]
        toggle = False
        
        while not self.__heartbeat_terminate_flag.is_set():
            t = time.time()

            # Check heartbeats (every 2 periods)
            if toggle:
                nei_copy = self.__neighbors.copy()
                for nei in nei_copy.keys():
                    if t - nei_copy[nei][2] > timeout:
                        logging.info(
                            f"({self.__self_addr}) Heartbeat timeout for {nei} ({t - nei_copy[nei][2]}). Removing..."
                        )
                        self.remove(nei)
            else:
                toggle = True

            # Send heartbeat
            nei_copy = self.__neighbors.copy()
            msg = self.build_msg(NodeMessages.BEAT, args=[str(time.time())])
            self.add_processed_msg(msg)
            for nei, (_, stub, _) in nei_copy.items():
                if stub is None:
                    continue
                try:
                    stub.send_message(msg, timeout=self.__config.participant["GRPC_TIMEOUT"])
                except Exception as e:
                    logging.info(
                        f"({self.__self_addr}) Cannot send heartbeat to {nei}. Error: {str(e)}"
                    )
                    self.remove(nei)

            # Sleep to allow the periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)

    ####
    # Gossiping
    ####

    def add_processed_msg(self, msg):
        """
        Add a message to the list of processed messages.

        Args:
            msg (node_pb2.Message): Message to add.

        Returns:
            bool: True if the message was added, False if it was already processed.
        """
        self.__processed_messages_lock.acquire()
        # Check if message was already processed
        if msg in self.__processed_messages:
            self.__processed_messages_lock.release()
            return False
        # If there are more than X messages, remove the oldest one
        if len(self.__processed_messages) > self.__config.participant["AMOUNT_LAST_MESSAGES_SAVED"]:
            self.__processed_messages.pop(0)
        # Add message
        # logging.debug(f"({self.__self_addr}) Adding processed message\n{msg}")
        self.__processed_messages.append(msg)
        self.__processed_messages_lock.release()
        return True

    def gossip(self, msg):
        """
        Add a message to the list of pending messages to gossip.

        Args:
            msg (node_pb2.Message): Message to add.
        """
        # logging.debug(f"({self.__self_addr}) Gossiping\n{msg}")
        # logging.debug(f"({self.__self_addr}) TTL: {msg.ttl}")
        if msg.ttl > 1:
            # Update ttl and broadcast
            msg.ttl -= 1

            # Add to pending messages
            self.__pending_msgs_lock.acquire()
            pending_neis = [n for n in self.__neighbors.keys() if n != msg.source]
            # logging.debug(f"({self.__self_addr}) Adding pending message\n{msg}")
            self.__pending_msgs.append((msg, pending_neis))
            # logging.debug(f"Pending messages to gossip:\n{str(self.__pending_msgs)}")
            self.__pending_msgs_lock.release()

    def __start_gossiper(self):
        logging.info(f"({self.__self_addr}) Starting gossiper thread")
        threading.Thread(target=self.__gossiper).start()

    def _stop_gossiper(self):
        self.__gossip_terminate_flag.set()

    def __gossiper(
            self
    ):
        period = self.__config.participant["GOSSIP_PERIOD"]
        messases_per_period = self.__config.participant["GOSSIP_MESSAGES_PER_PERIOD"]
        while not self.__gossip_terminate_flag.is_set():
            t = time.time()
            messages_to_send = []
            messages_left = messases_per_period

            # Lock
            self.__pending_msgs_lock.acquire()

            # Select the max amount of messages to send
            while messages_left > 0 and len(self.__pending_msgs) > 0:
                head_msg = self.__pending_msgs[0]
                # Select msgs
                if len(head_msg[1]) < messages_left:
                    # Select all
                    messages_to_send.append(head_msg)
                    # Remove from pending
                    self.__pending_msgs.pop(0)
                else:
                    # Select only the first neis
                    messages_to_send.append((head_msg[0], head_msg[1][:messages_left]))
                    # Remove from pending
                    self.__pending_msgs[0][1] = self.__pending_msgs[0][1][
                                                messages_left:
                                                ]

            # Unlock
            self.__pending_msgs_lock.release()
            for msg, neis in messages_to_send:
                for nei in neis:
                    # send only if direct connected (also add a try to deal with disconnections)
                    try:
                        if self.__neighbors[nei][1] is not None:
                            logging.debug(
                                f"({self.__self_addr}) Sending message (gossip)\n{msg}--> to {nei}"
                            )
                            self.send_message(nei, msg)
                    except KeyError:
                        pass
            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)

    def __str__(self):
        return str(self.__neighbors.keys())
    
    def get_neighbors_location(self):
        return self.__neighbors_location
    
    def set_neighbors_location(self, neighbors_location):
        self.__neighbors_location = neighbors_location
