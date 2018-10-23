#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


void get_max_pool(matrix in, matrix out, int x, int y, layer l, int output, int outw, int outh) {
    float max_val = 0;
    int size = l.size;
    int first = 1;
    for (int c = 0; c < l.channels; c++) {
        int offset = l.width * l.height * c;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int xcoor = x - size / 2 + i; // relative x position
                int ycoor = y - size / 2 + j; // relative y position
                float val = in.data[offset + x * in.cols + y];

                if (first) {
                    max_val = val;
                    first = 0;
                } else {
                    max_val = max_val < val ? val : max_val;
                }
            }
        }

        out.data[outw*outh*c + output] = max_val;
    }
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    int num_conv = 0;
    for (int i = 0; i < in.rows; i += l.stride){
        for (int j = 0; j < in.cols; j+= l.stride) {
            // iterate over every pixel in the image
            get_max_pool(in, out, i, j, l, num_conv, outw, outh);
            // float max = get_max_pool(in, out, num_conv, l.size, i, j);
            num_conv++;
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

