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

from functools import partial
import logging
import threading


class Aggregator:
    """
    Class to manage the aggregation of models. It is a thread so, aggregation will be done in background if all models were added or timeouts have gone.
    Also, it is an observable so, it will notify the node when the aggregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown", config=None):
        self.node_name = node_name
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self.__train_set = []
        self.__waiting_aggregated_model = False
        self.__models = {}
        self.__round = 0

        # Locks
        self.__agg_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()

    def aggregate(self, models):
        """
        Aggregate the models.
        """
        print("Not implemented")

    def set_nodes_to_aggregate(self, l):
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            l: List of nodes to aggregate. Empty for no aggregation.

        Raises:
            Exception: If the aggregation is running.
        """
        if not self.__finish_aggregation_lock.locked():
            logging.info(f"({self.node_name}) set_nodes_to_aggregate | Setting nodes to aggregate: {l}")
            self.__train_set = l
            logging.info(f"({self.node_name}) set_nodes_to_aggregate | Clearing __models.")
            self.__models = {}
            logging.info(
                f"({self.node_name}) set_nodes_to_aggregate | Acquiring __finish_aggregation_lock (timeout={self.config.participant['AGGREGATION_TIMEOUT']})."
            )
            self.__finish_aggregation_lock.acquire(timeout=self.config.participant["AGGREGATION_TIMEOUT"])
        else:
            raise Exception(
                "It is not possible to set nodes to aggregate when the aggregation is running."
            )

    def set_waiting_aggregated_model(self, nodes):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        The model only will receive a model, and then it will be used as an aggregated model.
        """
        self.__waiting_aggregated_model = True

    def clear(self):
        """
        Clear the aggregation (remove train set and release locks).
        """
        logging.info(
            f"({self.node_name}) clear | Acquiring __agg_lock."
        )
        self.__agg_lock.acquire()
        self.__train_set = []
        self.__models = {}
        try:
            logging.info(f"({self.node_name}) clear | Releasing __finish_aggregation_lock.")
            if self.__finish_aggregation_lock.locked():
                self.__finish_aggregation_lock.release()
        except:
            pass
        logging.info(f"({self.node_name}) clear | Releasing __agg_lock.")
        if self.__agg_lock.locked():
            self.__agg_lock.release()

    def get_round(self):
        """
        Get the round of the aggregation.
        """
        return self.__round

    def set_round(self, current_round):
        """
        Set the round of the aggregation.
        """
        self.__round = current_round

    def get_aggregated_models(self):
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that collaborated to get the model.
        """
        # Get a list of nodes added
        models_added = [n.split() for n in list(self.__models.keys())]
        # Flatten list
        models_added = [element for sublist in models_added for element in sublist]
        return models_added

    def get_aggregated_models_weights(self):
        return self.__models

    def add_model(self, model, contributors, weight, source=None, round=None, local=False):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            contributors: Nodes that collaborated to get the model.
            weight: Number of samples used to get the model.
            source: Node that sent the model.
            round: Round of the aggregation.
        """

        nodes = list(contributors)
        logging.info(
            f"({self.node_name}) add_model (aggregator) | source={source} | __models={self.__models.keys()} | contributors={nodes} | train_set={self.__train_set} | get_aggregated_models={self.get_aggregated_models()}")

        # Verify that contributors are not empty
        if contributors == []:
            logging.info(
                f"({self.node_name}) Received a model without a list of contributors."
            )
            logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __agg_lock.")
            if self.__agg_lock.locked():
                self.__agg_lock.release()
            return None

        # Check again if the round is the same as the current one, if not, ignore the model (it is from a previous round)
        if round != self.__round:
            logging.info(
                f"({self.node_name}) add_model (aggregator) | Received a model from a previous round."
            )
            logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __agg_lock.")
            if self.__agg_lock.locked():
                self.__agg_lock.release()
            return None

        # Diffusion / Aggregation
        if self.__waiting_aggregated_model and not local:
            logging.info(
                    f"({self.node_name}) add_model (aggregator) | __waiting_aggregated_model (True)")
            if set(contributors) == set(self.__train_set):
                logging.info(
                    f"({self.node_name}) add_model (aggregator) | __waiting_aggregated_model (True) | Ignoring add_model functionality...")
                logging.info(
                    f"({self.node_name}) add_model (aggregator) | __waiting_aggregated_model (True) | Received an aggregated model because all contributors are in the train set (me too). Overwriting __models with the aggregated model.")
                self.__models = {}
                self.__models = {" ".join(nodes): (model, 1)}
                self.__waiting_aggregated_model = False
                logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __finish_aggregation_lock.")
                if self.__finish_aggregation_lock.locked():
                    self.__finish_aggregation_lock.release()
                return contributors
            else:
                logging.info(
                    f"({self.node_name}) add_model (aggregator) | __waiting_aggregated_model (True) | Ignoring add_model functionality...")

        else:
            logging.info(
                f"({self.node_name}) add_model (aggregator) | Acquiring __agg_lock."
            )
            self.__agg_lock.acquire()

            # Check if aggregation is needed
            if len(self.__train_set) > len(self.get_aggregated_models()):
                # Check if all nodes are in the train_set
                if all([n in self.__train_set for n in nodes]):
                    logging.info(
                        f'({self.node_name}) add_model (aggregator) | All contributors are in the train set. Adding model.')
                    # Check if the model is a full/partial aggregation
                    if len(nodes) == len(self.__train_set):
                        logging.info(
                            f'({self.node_name}) add_model (aggregator) | The number of contributors is equal to the number of nodes in the train set. --> Full aggregation.')
                        self.__models = {" ".join(nodes): (model, weight)}
                        logging.info(
                            f"({self.node_name}) add_model (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )
                        # Finish agg
                        logging.info(
                            f"({self.node_name}) add_model (aggregator) | Releasing __finish_aggregation_lock.")
                        if self.__finish_aggregation_lock.locked():
                            self.__finish_aggregation_lock.release()
                        # Unlock and Return
                        logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __agg_lock.")
                        if self.__agg_lock.locked():
                            self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif all([n not in self.get_aggregated_models() for n in nodes]):
                        logging.info(
                            f'({self.node_name}) add_model (aggregator) | All contributors are not in the aggregated models. --> Partial aggregation.')
                        # Aggregate model
                        self.__models[" ".join(nodes)] = (model, weight)
                        logging.info(
                            f"({self.node_name}) add_model (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )

                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            logging.info(
                                f"({self.node_name}) add_model (aggregator) | All models were added. Finishing aggregation."
                            )
                            logging.info(
                                f"({self.node_name}) add_model (aggregator) | Releasing __finish_aggregation_lock.")
                            if self.__finish_aggregation_lock.locked():
                                self.__finish_aggregation_lock.release()

                        # Unlock and Return
                        logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __agg_lock.")
                        if self.__agg_lock.locked():
                            self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif any([n in self.get_aggregated_models() for n in nodes]):
                        logging.info(
                            f'({self.node_name}) BETA add_model (aggregator) | Some contributors are in the aggregated models.')

                        logging.info(
                            f'({self.node_name}) BETA add_model (aggregator) | __models={self.__models.keys()}')

                        # Obtain the list of nodes that are not in the aggregated models
                        nodes_not_in_aggregated_models = [n for n in nodes if n not in self.get_aggregated_models()]
                        logging.info(
                            f'({self.node_name}) BETA add_model (aggregator) | nodes_not_in_aggregated_models={nodes_not_in_aggregated_models}')

                        # For each node that is not in the aggregated models, aggregate the model with the aggregated model
                        for n in nodes_not_in_aggregated_models:
                            self.__models[n] = (model, weight)

                        logging.info(
                            f'({self.node_name}) BETA add_model (aggregator) | __models={self.__models.keys()}')

                        logging.info(
                            f"({self.node_name}) BETA add_model (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )
                        logging.info(f"({self.node_name}) BETA add_model (aggregator) | self.aggregated_models={self.get_aggregated_models()}")
                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            logging.info(
                                f"({self.node_name}) BETA add_model (aggregator) | All models were added. Finishing aggregation."
                            )
                            logging.info(
                                f"({self.node_name}) BETA add_model (aggregator) | Releasing __finish_aggregation_lock.")
                            if self.__finish_aggregation_lock.locked():
                                self.__finish_aggregation_lock.release()

                        # Unlock and Return
                        logging.info(f"({self.node_name}) BETA add_model (aggregator) | Releasing __agg_lock.")
                        if self.__agg_lock.locked():
                            self.__agg_lock.release()
                        return self.get_aggregated_models()

                    # elif any([n in self.get_aggregated_models() for n in nodes]):
                    #     logging.info(
                    #         f'({self.node_name}) BETA add_model (aggregator) | Some contributors are in the aggregated models. --> Partial aggregation.')
                    #     # Logging __models
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | __models={self.__models.keys()}")
                    #     # Topology of 3 nodes connected in triangle. N1 sends the M1 model to N0, N0 locally adds the M0 and does a partial aggregation between the two (M1+M0). Now N2 sends to N0 a M1+M2 (same procedure as above).
                    #     nodes_all = list(set(nodes) | set(self.get_aggregated_models()))
                    #     # Get overlapping nodes
                    #     overlapping_nodes = list(set(nodes) & set(self.get_aggregated_models()))
                    #     logging.info(f"({self.node_name}) BETA add_model (aggregator) | nodes_all={nodes_all}")
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | overlapping_nodes={overlapping_nodes}")
                    #     # overlapping_models as a dictionary
                    #     overlapping_models = {n: self.__models[n] for n in self.__models.keys() if
                    #                           any([n in overlapping_nodes for n in n.split()])}
                    #     # Remove overlapping models
                    #     for n in overlapping_models.keys():
                    #         del self.__models[n]
                    #
                    #     # Aggregate overlapping models and model (include "received" at the end of the name)
                    #     overlapping_models[" ".join(nodes) + " received"] = (model, weight)
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | overlapping_models={overlapping_models.keys()}")
                    #     aggregated_model = self.aggregate(overlapping_models)
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | aggregated_model={aggregated_model}")
                    #
                    #     # Add aggregated model
                    #     self.__models[" ".join(set(nodes_all))] = (aggregated_model, weight)
                    #
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | __models={self.__models.keys()}")
                    #
                    #     logging.info(
                    #         f"({self.node_name}) BETA add_model (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                    #     )
                    #
                    #     # Check if all models were added
                    #     if len(self.get_aggregated_models()) >= len(self.__train_set):
                    #         logging.info(
                    #             f"({self.node_name}) BETA add_model (aggregator) | All models were added. Finishing aggregation."
                    #         )
                    #         logging.info(
                    #             f"({self.node_name}) BETA add_model (aggregator) | Releasing __finish_aggregation_lock.")
                    #         self.__finish_aggregation_lock.release()
                    #
                    #     # Unlock and Return
                    #     logging.info(f"({self.node_name}) BETA add_model (aggregator) | Releasing __agg_lock.")
                    #     self.__agg_lock.release()
                    #     return self.get_aggregated_models()

                    else:
                        logging.info(
                            f"({self.node_name}) add_model (aggregator) | Can't add a model that has already been added {nodes}"
                        )
                else:
                    logging.info(
                        f"({self.node_name}) add_model (aggregator) | Can't add a model from a node ({nodes}) that is not in the training test."
                    )
            else:
                logging.info(
                    f"({self.node_name}) add_model (aggregator) | Received a model when is not needed."
                )
            logging.info(f"({self.node_name}) add_model (aggregator) | Releasing __agg_lock.")
            if self.__agg_lock.locked():
                self.__agg_lock.release()
            return None

    def wait_and_get_aggregation(self):
        """
        Wait for aggregation to finish.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.
        """
        timeout = self.config.participant["AGGREGATION_TIMEOUT"]
        # Wait for aggregation to finish (then release the lock again)
        logging.info(
            f"({self.node_name}) wait_and_get_aggregation | Acquiring __finish_aggregation_lock (timeout={self.config.participant['AGGREGATION_TIMEOUT']})."
        )
        self.__finish_aggregation_lock.acquire(timeout=timeout)
        try:
            logging.info(f"({self.node_name}) wait_and_get_aggregation | Releasing __finish_aggregation_lock.")
            if self.__finish_aggregation_lock.locked():
                self.__finish_aggregation_lock.release()
        except:
            pass

        # If awaiting an aggregated model, return it
        if self.__waiting_aggregated_model:
            logging.info(
                f"({self.node_name}) wait_and_get_aggregation | __waiting_aggregated_model (True)"
            )
            if len(self.__models) == 1:
                logging.info(
                    f"({self.node_name}) wait_and_get_aggregation | Received an aggregated model. Overwriting my model with the aggregated model."
                )
                return list(self.__models.values())[0][0]
            elif len(self.__models) == 0:
                logging.info(
                    f"({self.node_name}) wait_and_get_aggregation | Timeout reached by waiting for an aggregated model. Continuing with the local model."
                )
            raise Exception(
                f"Waiting for an an aggregated but several models were received: {self.__models.keys()}"
            )
        # Start aggregation
        logging.info(f'({self.node_name}) wait_and_get_aggregation | Starting aggregation.')
        n_model_aggregated = sum(
            [len(nodes.split()) for nodes in list(self.__models.keys())]
        )
        logging.info(
            f"({self.node_name}) wait_and_get_aggregation | n_model_aggregated={n_model_aggregated} | len(self.__train_set)={len(self.__train_set)}"
        )

        # Timeout / All models
        if n_model_aggregated != len(self.__train_set):
            logging.info(
                f"({self.node_name}) wait_and_get_aggregation | Aggregating models, timeout reached. Missing models: {set(self.__train_set) - set(self.__models.keys())}"
            )
        else:
            logging.info(f"({self.node_name}) wait_and_get_aggregation | Aggregating models.")

        # Notify node
        logging.info(f"({self.node_name}) wait_and_get_aggregation | Aggregating models: {self.__models.keys()}")
        return self.aggregate(self.__models)

    def get_partial_aggregation(self, except_nodes):
        """
        Obtain a partial aggregation.

        Args:
            except_nodes (list): List of nodes to exclude from the aggregation.

        Returns:
            Aggregated model, nodes aggregated and aggregation weight.
        """
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n, (m, s) in list(models.items()):
            spplited_nodes = n.split()
            if all([n not in except_nodes for n in spplited_nodes]):
                dict_aux[n] = (m, s)
                nodes_aggregated += spplited_nodes
                aggregation_weight += s

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            return None, None, None

        logging.info(
            f"({self.node_name}) get_partial_aggregation | Aggregating models: dict_aux={dict_aux.keys()}"
        )

        return self.aggregate(dict_aux), nodes_aggregated, aggregation_weight
    
    def get_local_model(self):
        """
        Get my local model in __models.
        """
        if self.node_name in self.__models.keys():
            return self.__models[self.node_name][0], [self.node_name], self.__models[self.node_name][1]
        else:
            return None

    def print_model_size(self, model):
        """
        Get the size of the model.

        Returns:
            Size of the model.
        """
        total_params = 0
        total_memory = 0
        
        for layer, param in model.items():
            num_params = param.numel()
            total_params += num_params
            
            memory_usage = param.element_size() * num_params
            total_memory += memory_usage
        
        total_memory_in_mb = total_memory / (1024 ** 2)
        logging.info(f"({self.node_name}) print_model_size | Model size: {total_memory_in_mb} MB")

def create_malicious_aggregator(aggregator, attack):
    # It creates a partial function aggregate that wraps the aggregate method of the original aggregator. 
    aggregate = partial(aggregator.aggregate)  # None is the self (not used)

    # This function will replace the original aggregate method of the aggregator.
    def malicious_aggregate(self, models):
        # it first calls the original aggregate function with the models argument to get the initial aggregation result.
        accum = aggregate(models)
        logging.info(f"({self.node_name}) malicious_aggregate | original aggregation result={accum}")
        if models is not None:
            accum = attack(accum)
            logging.info(f"({self.node_name}) malicious_aggregate | attack aggregation result={accum}")
        return accum

    # It replaces the aggregate method of the aggregator with the malicious_aggregate function.
    # This is done using the partial function again to bind the aggregator as the self argument of malicious_aggregate.
    aggregator.aggregate = partial(malicious_aggregate, aggregator)
    return aggregator
