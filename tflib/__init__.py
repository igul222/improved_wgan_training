import numpy as np
import tensorflow as tf

import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.
    
    Creates and returns theano shared variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    result = _params[name]
    i = 0
    while result in _param_aliases:
        # print 'following alias {}: {} to {}'.format(i, result, _param_aliases[result])
        i += 1
        result = _param_aliases[result]
    return result

def params_with_name(name):
    return [p for n,p in _params.items() if name in n]

def delete_all_params():
    _params.clear()

def alias_params(replace_dict):
    for old,new in replace_dict.items():
        # print "aliasing {} to {}".format(old,new)
        _param_aliases[old] = new

def delete_param_aliases():
    _param_aliases.clear()

# def search(node, critereon):
#     """
#     Traverse the Theano graph starting at `node` and return a list of all nodes
#     which match the `critereon` function. When optimizing a cost function, you 
#     can use this to get a list of all of the trainable params in the graph, like
#     so:

#     `lib.search(cost, lambda x: hasattr(x, "param"))`
#     """

#     def _search(node, critereon, visited):
#         if node in visited:
#             return []
#         visited.add(node)

#         results = []
#         if isinstance(node, T.Apply):
#             for inp in node.inputs:
#                 results += _search(inp, critereon, visited)
#         else: # Variable node
#             if critereon(node):
#                 results.append(node)
#             if node.owner is not None:
#                 results += _search(node.owner, critereon, visited)
#         return results

#     return _search(node, critereon, set())

# def print_params_info(params):
#     """Print information about the parameters in the given param set."""

#     params = sorted(params, key=lambda p: p.name)
#     values = [p.get_value(borrow=True) for p in params]
#     shapes = [p.shape for p in values]
#     print "Params for cost:"
#     for param, value, shape in zip(params, values, shapes):
#         print "\t{0} ({1})".format(
#             param.name,
#             ",".join([str(x) for x in shape])
#         )

#     total_param_count = 0
#     for shape in shapes:
#         param_count = 1
#         for dim in shape:
#             param_count *= dim
#         total_param_count += param_count
#     print "Total parameter count: {0}".format(
#         locale.format("%d", total_param_count, grouping=True)
#     )

def print_model_settings(locals_):
    print "Uppercase local vars:"
    all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T' and k!='SETTINGS' and k!='ALL_SETTINGS')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)


def print_model_settings_dict(settings):
    print "Settings dict:"
    all_vars = [(k,v) for (k,v) in settings.items()]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)