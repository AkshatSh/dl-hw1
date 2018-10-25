#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"
#include "uwnet.h"

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void test_im2col() {
    float input_data[] = {
        1,2,3,
        4,5,6
    };

    // matrix x = make_matrix(2,3);
    image example = float_to_image(input_data, 3, 2, 1);
    matrix x = im2col(example, 3, 2);
    for (int i = 0; i < x.rows; i++) {
        printf("");
        for (int j =0; j < x.cols; j++) {
            printf(", %f", x.data[i * x.cols + j]);
        }
        printf("\n");
    }
}

void run_tests()
{
    test_matrix_speed();
    test_im2col();
    //printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

