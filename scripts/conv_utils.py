import numpy as np


def get_final_conv_shape(input_shape, conv_number,
                         filter_base_length, filter_length,
                         filter_base_stride, filter_stride,
                         filter_base, filter_factor, padded):
    ''' calculates final convolutional layer output shape '''
    if padded:
        p = 1
    else:
        p = 0
    out_filters = input_shape[2]*filter_base*filter_factor**(conv_number-1)
    out_dim = (np.array(input_shape[:2], dtype=int)-filter_base_length+p)//filter_base_stride+1
    for i in range(1, conv_number):
        out_dim = (out_dim-filter_length+p)//filter_stride+1
    return tuple(out_dim)+(out_filters,)


def get_filter_number(conv_iter, filter_base, filter_factor):
    ''' calculates the filter count for a given convolutional iteration '''
    return filter_base*filter_factor**(conv_iter)


def get_filter_length_stride(conv_iter, filter_base_length, filter_base_stride, filter_length, filter_stride):
    ''' calculates filter length and stride for a given convolutional iteration '''
    if conv_iter == 0:
        return filter_base_length, filter_base_stride
    else:
        return filter_length, filter_stride