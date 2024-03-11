#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

import json

from fedstellar.utils.topologymanager import TopologyManager


# set_test_settings()

def test_topology():
    topologymanager = TopologyManager(n_nodes=4, b_symmetric=True,
                                      undirected_neighbor_num=3)
    topologymanager.generate_topology()
    topology = topologymanager.get_topology()
    topologymanager.draw_graph()
    print("\n")
    print(topology)


def test_topology_6():
    topologymanager = TopologyManager(n_nodes=6, b_symmetric=True,
                                      undirected_neighbor_num=5)
    topologymanager.generate_topology()
    topology = topologymanager.get_topology()
    topologymanager.draw_graph()
    print("\n")
    print(topology)


def test_ring_topology():
    topologymanager = TopologyManager(scenario_name="example", n_nodes=5, b_symmetric=True)
    topologymanager.generate_ring_topology()
    topology = topologymanager.get_topology()
    topologymanager.draw_graph()
    print("\n")
    print(topology)


def test_ring_topology2():
    # Import configuration file
    with open("/fedstellar/config/topology.json.example") as json_file:
        config = json.load(json_file)
    n_nodes = len(config['nodes'])

    # Create a partially connected network (ring-structured network)
    topologymanager = TopologyManager(n_nodes=n_nodes, b_symmetric=True)
    topologymanager.generate_ring_topology()
    topology = topologymanager.get_topology()
    print(topology)

    nodes_ip_port = []
    for i in config['nodes']:
        nodes_ip_port.append((i['ip'], i['port']))

    topologymanager.add_nodes(nodes_ip_port)
    topologymanager.draw_graph()

def test_topology_centralized():

    # Create a partially connected network (ring-structured network)
    topologymanager = TopologyManager(n_nodes=5, b_symmetric=True, server=True)
    topologymanager.generate_topology()
    topology = topologymanager.get_topology()
    print(topology)
    topologymanager.draw_graph()
