# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.misc
import scipy.io



class MatConvNet:

    def __init__(self, data_path):

        self.data_path = data_path

    def rec_dic(self, dat_type):

        _dic = {}
        if isinstance(dat_type, dict):
            for key, value in dat_type.items():  # iteritems loops through key, value pairs
                _dic.update({key: value})

        return _dic

    def decoding(self, d, indent=0, nkeys=0):

        layers_seq = []
        _layers = {}
        _meta = {}

        """Pretty print nested structures from .mat files
        Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
        """

        # Subset dictionary to limit keys to print.  Only works on first level
        if nkeys > 0:
            d = {k: d[k] for k in d.keys()[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

        if isinstance(d, dict):
            for key, value in d.iteritems():  # iteritems loops through key, value pairs
                # print '\t' * indent + 'Key: ' + str(key)
                if key == 'layers':
                    _num_layers = d[key].shape[0]
                    for i in xrange(0, _num_layers):
                        _dic = {}
                        mat_struct = d[key][i]

                        layer_name = str(mat_struct.name)
                        layers_seq.append(layer_name)

                        # retrieving content for each struct
                        for k, v in mat_struct.__dict__.items():
                            _dic.update({k: v})

                        _layers.update({layer_name: _dic})
                elif key == 'meta':
                    mat_struct = d[key]

                    for k, v in mat_struct.iteritems():  # iteritems loops through key, value pairs
                        k_dic = self.rec_dic(v)
                        _meta.update({k: k_dic})

        #TODO: to remove
        if isinstance(d, np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
            for n in d.dtype.names:  # This means it's a struct, it's bit of a kludge test.
                print '\t' * indent + 'Field: ' + str(n)
                self.decoding(d[n], indent + 1)

        # unsqueeze the weights
        for i, layer_name in enumerate(layers_seq):

            if _layers[layer_name]['weights'].size != 0 and _layers[layer_name]['weights'][0].ndim < 4:
                _layers[layer_name]['weights'][0] = np.expand_dims(_layers[layer_name]['weights'][0], axis=0)
                _layers[layer_name]['weights'][0] = np.expand_dims(_layers[layer_name]['weights'][0], axis=0)

        return layers_seq, _layers, _meta

    def loadmat(self):
        '''
        this function should be called instead of direct scipy.io.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects

        from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
        '''
        data = scipy.io.loadmat(self.data_path, struct_as_record=False, squeeze_me=True)

        return self.check_keys(data)

    def check_keys(self, dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
                dict[key] = self.todict(dict[key])
        return dict

    def todict(self, matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                dict[strg] = self.todict(elem)
            else:
                dict[strg] = elem
        return dict