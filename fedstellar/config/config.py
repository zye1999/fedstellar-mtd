# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


"""
Module to define constants for the DFL system.
"""
import json
import logging

###################
#  Global Config  #
###################


class Config:
    """
    Class to define global config for the DFL system.
    """
    topology = {}
    participant = {}

    participants = []  # Configuration of each participant (this information is stored only in the controller)
    participants_path = []

    def __init__(self, entity, topology_config_file=None, participant_config_file=None):

        self.entity = entity

        if topology_config_file is not None:
            self.set_topology_config(topology_config_file)

        if participant_config_file is not None:
            self.set_participant_config(participant_config_file)

    def __getstate__(self):
        # Return the attributes of the class that should be serialized
        return {'topology': self.topology, 'participant': self.participant}

    def __setstate__(self, state):
        # Set the attributes of the class from the serialized state
        self.topology = state['topology']
        self.participant = state['participant']

    def get_topology_config(self):
        return json.dumps(self.topology, indent=2)

    def get_participant_config(self):
        return json.dumps(self.participant, indent=2)

    def _set_default_config(self):
        """
        Default values are defined here.
        """
        pass
    
    def to_json(self):
        # Return participant configuration as a json string
        return json.dumps(self.participant, sort_keys=False, indent=2)

    # Read the configuration file scenario_config.json, and return a dictionary with the configuration
    def set_participant_config(self, participant_config):
        with open(participant_config) as json_file:
            self.participant = json.load(json_file)

    def set_topology_config(self, topology_config_file):
        with open(topology_config_file) as json_file:
            self.topology = json.load(json_file)

    def add_participant_config(self, participant_config):
        with open(participant_config) as json_file:
            self.participants.append(json.load(json_file))

    def set_participants_config(self, participants_config):
        self.participants = []
        self.participants_path = participants_config
        for participant in participants_config:
            self.add_participant_config(participant)
    
    def add_participants_config(self, participants_config):
        self.participants_path = participants_config
        for participant in participants_config:
            self.add_participant_config(participant)