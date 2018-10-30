#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


void get_max_pool(matrix in, matrix out, int x, int y, int c, layer l, int output, int outw, int outh) {
    float max_val = 0;
    int size = l.size;
    int first = 1;
    int offset = l.width * l.height * c;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int xcoor = x - size / 2 + i; // relative x position
            int ycoor = y - size / 2 + j; // relative y position
            float val;
            if (xcoor < 0 || ycoor < 0 || xcoor >= in.rows || ycoor >= in.cols) {
                val = 0;
            } else {
                val = in.data[offset + xcoor * in.cols + ycoor];
            }

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
    for (int batch = 0; batch < in.rows; batch++) {
        matrix batch_image = make_matrix(l.height * l.channels, l.width);
        batch_image.data = in.data + batch * in.cols;
        matrix batch_out = make_matrix(outh*l.channels, outw);
        batch_out.data = out.data + batch * out.cols;
        for (int c = 0; c < l.channels; c++) {
            num_conv = 0;
            for (int i = 0; i < l.height; i += l.stride){
                for (int j = 0; j < l.width; j+= l.stride) {
                    // iterate over every pixel in the image
                    get_max_pool(batch_image, batch_out, i, j, c, l, num_conv, outw, outh);
                    // float max = get_max_pool(in, out, num_conv, l.size, i, j);
                    num_conv++;
                }
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

void set_max_pool(matrix in, matrix prev_delta, matrix delta, int x, int y, int c, layer l, int output, int outw, int outh) {
    // x = l.stride * x;
    // y = l.stride * y;
    // int size = l.size;
    // int first = 1;
    // int offset = l.width * l.height * c;
    // float max_val = in.data[outw*outh*c + output];
    // for (int i = 0; i < size; i++) {
    //     for (int j = 0; j < size; j++) {
    //         int xcoor = x - size / 2 + i; // relative x position
    //         int ycoor = y - size / 2 + j; // relative y position
    //         delta.data[offset + xcoor * delta.cols + ycoor] = delta_val;
    //     }
    // }
    float max_val = 0;
    long max_index = 0;
    int size = l.size;
    int first = 1;
    int offset = l.width * l.height * c;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int xcoor = x - size / 2 + i; // relative x position
            int ycoor = y - size / 2 + j; // relative y position
            float val;
            long index;
            if (xcoor < 0 || ycoor < 0 || xcoor >= in.rows || ycoor >= in.cols) {
                val = 0;
                index = 0;
            } else {
                index = offset + xcoor * in.cols + ycoor;
                val = in.data[index];
            }

            if (first) {
                max_val = val;
                max_index = index;
                first = 0;
            } else {
                int is_max = max_val < val;
                max_val = is_max ? val : max_val;
                max_index = is_max? index : max_index;
            }
        }
    }
    // printf("max_index : %ld max val: %f\n", max_index, max_val);
    // delta.data[outw*outh*c + output] += prev_delta.data[max_index];
    prev_delta.data[max_index] = delta.data[outw*outh*c + output];
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
    // printf("prev_delta (row: %d, cols: %d) vs out (row: %d, cols: %d) vs in(row: %d, cols: %d) vs delta (row: %d, col: %d)\n", prev_delta.rows, prev_delta.cols, out.rows, out.cols, in.rows, in.cols, delta.rows, delta.cols);

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (int batch = 0; batch < in.rows; batch++) {
        matrix batch_image;
        batch_image.rows = l.height * l.channels;
        batch_image.cols = l.width;
        batch_image.data = in.data + batch * in.cols;

        matrix batch_delta;
        batch_delta.rows = outh * l.channels;
        batch_delta.cols = outw;
        batch_delta.data = delta.data + batch * delta.cols;

        matrix batch_prev_delta;
        batch_prev_delta.rows = l.height * l.channels;
        batch_prev_delta.cols = l.width;
        batch_prev_delta.data = prev_delta.data + batch * prev_delta.cols;
        for (int c = 0; c < l.channels; c++) {
            int num_conv = 0;
            for (int i = 0; i < l.height; i += l.stride){
                for (int j = 0; j < l.width; j+= l.stride) {
                    // iterate over every pixel in the image
                    set_max_pool(batch_image, batch_prev_delta, batch_delta, i, j, c, l, num_conv, outw, outh);
                    // float max = get_max_pool(in, out, num_conv, l.size, i, j);
                    num_conv++;
                }
            }
        }
    }

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

