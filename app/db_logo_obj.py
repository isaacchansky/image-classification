#! /usr/bin/python

'''
Author: Isaac Chansky

Object definitions for database objects
'''


class db_obj:
    """docstring for db_obj"""
    def __init__(self, logo_name):
        self.logo_name = logo_name

    def get_local_features():
        return 0

    def get_global_features():
        return 0

    def serialize_data():
        return 0


class local_feature:
    """docstring for LocalFeature object"""
    def __init__(self, feature_vector, std_dev, rel_size):
        self.feature_vector = feature_vector
        self.std_dev = std_dev
        self.rel_size = rel_size

    def get_feature_vector():
        return 0

    def get_std_dev():
        return 0

    def get_rel_size():
        return 0


class Global_feature:
    """docstring for global_feature object"""
    def __init__(self, feature_vector, std_dev, rel_size):
        self.feature_vector = feature_vector
        self.std_dev = std_dev
        self.rel_size = rel_size

    def get_feature_vector():
        return 0

    def get_std_dev():
        return 0

    def get_rel_size():
        return 0
