def output_dim_conv2d(input_dim, kernel_size, stride, padding):
    """Return the output dimension (width and height) of a 2D convolution layer used in DQN.
    
    Parameters
    --------------------
    input_dim: tuple
        input dimension of a 2D convolution layer
        
    kernel_size: int
        kernel size of 2D convolution layer
        
    stride: int
        stride of a 2D convolution layer
        
    padding: int
        padding of a 2D convolution layer"""
    
    assert input_dim[0] > 0 and input_dim[1] > 0, "Input dimension of a 2D convolution layer must be only positive integers."

    w = input_dim[0]
    h = input_dim[1]
    return (w + 2 * padding - kernel_size) // stride + 1, (h + 2 * padding - kernel_size) // stride + 1,