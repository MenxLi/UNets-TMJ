/*
 * Copyright (c) 2020 Mengxun Li.
 *
 * This file is part of LabelSys
 * (see https://bitbucket.org/Mons00n/mrilabelsys/).
 */
#include "c_utils.h"
#include <stdlib.h>
#include <math.h>

/*{{{ Encode*/
int intArray2Bool(int * arr, int size, int bit_len, int * dst_arr){
        for(int idx = 0; idx < size; idx++){
                int2Bool(arr[idx], dst_arr, bit_len);
                dst_arr += bit_len;
        }
        return 0;
}


void int2Bool(int num, int * buffer, int buffer_len){
        buffer += (buffer_len-1);
        for(int i = 0; i<buffer_len; i++){
                *buffer = num&1;
                buffer--;
                num >>= 1;
        }
}

int biArray2B64Str(int * bi_arr, char * dst_arr, int bi_size){
        for(int i=0; i< bi_size; i += 6){
                bi6Digit2Char(bi_arr+i, dst_arr + i/6);
        }
        return 0;
}

void bi6Digit2Char(int * bi_arr, char * buffer){
        // convert 6 digits binary code into decimal
        static int operator[6] = {32, 16, 8, 4, 2, 1};
        int decimal = 0;
        for(int i=0; i < 6; i++){
                decimal += bi_arr[i]*operator[i];
        }
        // convert to char
        static char b64_table[64] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/' };
        * buffer = b64_table[decimal];
}
/*}}}*/

/*{{{ Decode*/
void Char2Binary(char * c, int * buffer){
        switch((char)*c){/*{{{*/
                case 'A':
                        int2Bool(0, buffer, 6); break;
                case 'B':
                        int2Bool(1, buffer, 6); break;
                case 'C':
                        int2Bool(2, buffer, 6); break;
                case 'D':
                        int2Bool(3, buffer, 6); break;
                case 'E':
                        int2Bool(4, buffer, 6); break;
                case 'F':
                        int2Bool(5, buffer, 6); break;
                case 'G':
                        int2Bool(6, buffer, 6); break;
                case 'H':
                        int2Bool(7, buffer, 6); break;
                case 'I':
                        int2Bool(8, buffer, 6); break;
                case 'J':
                        int2Bool(9, buffer, 6); break;
                case 'K':
                        int2Bool(10, buffer, 6); break;
                case 'L':
                        int2Bool(11, buffer, 6); break;
                case 'M':
                        int2Bool(12, buffer, 6); break;
                case 'N':
                        int2Bool(13, buffer, 6); break;
                case 'O':
                        int2Bool(14, buffer, 6); break;
                case 'P':
                        int2Bool(15, buffer, 6); break;
                case 'Q':
                        int2Bool(16, buffer, 6); break;
                case 'R':
                        int2Bool(17, buffer, 6); break;
                case 'S':
                        int2Bool(18, buffer, 6); break;
                case 'T':
                        int2Bool(19, buffer, 6); break;
                case 'U':
                        int2Bool(20, buffer, 6); break;
                case 'V':
                        int2Bool(21, buffer, 6); break;
                case 'W':
                        int2Bool(22, buffer, 6); break;
                case 'X':
                        int2Bool(23, buffer, 6); break;
                case 'Y':
                        int2Bool(24, buffer, 6); break;
                case 'Z':
                        int2Bool(25, buffer, 6); break;
                case 'a':
                        int2Bool(26, buffer, 6); break;
                case 'b':
                        int2Bool(27, buffer, 6); break;
                case 'c':
                        int2Bool(28, buffer, 6); break;
                case 'd':
                        int2Bool(29, buffer, 6); break;
                case 'e':
                        int2Bool(30, buffer, 6); break;
                case 'f':
                        int2Bool(31, buffer, 6); break;
                case 'g':
                        int2Bool(32, buffer, 6); break;
                case 'h':
                        int2Bool(33, buffer, 6); break;
                case 'i':
                        int2Bool(34, buffer, 6); break;
                case 'j':
                        int2Bool(35, buffer, 6); break;
                case 'k':
                        int2Bool(36, buffer, 6); break;
                case 'l':
                        int2Bool(37, buffer, 6); break;
                case 'm':
                        int2Bool(38, buffer, 6); break;
                case 'n':
                        int2Bool(39, buffer, 6); break;
                case 'o':
                        int2Bool(40, buffer, 6); break;
                case 'p':
                        int2Bool(41, buffer, 6); break;
                case 'q':
                        int2Bool(42, buffer, 6); break;
                case 'r':
                        int2Bool(43, buffer, 6); break;
                case 's':
                        int2Bool(44, buffer, 6); break;
                case 't':
                        int2Bool(45, buffer, 6); break;
                case 'u':
                        int2Bool(46, buffer, 6); break;
                case 'v':
                        int2Bool(47, buffer, 6); break;
                case 'w':
                        int2Bool(48, buffer, 6); break;
                case 'x':
                        int2Bool(49, buffer, 6); break;
                case 'y':
                        int2Bool(50, buffer, 6); break;
                case 'z':
                        int2Bool(51, buffer, 6); break;
                case '0':
                        int2Bool(52, buffer, 6); break;
                case '1':
                        int2Bool(53, buffer, 6); break;
                case '2':
                        int2Bool(54, buffer, 6); break;
                case '3':
                        int2Bool(55, buffer, 6); break;
                case '4':
                        int2Bool(56, buffer, 6); break;
                case '5':
                        int2Bool(57, buffer, 6); break;
                case '6':
                        int2Bool(58, buffer, 6); break;
                case '7':
                        int2Bool(59, buffer, 6); break;
                case '8':
                        int2Bool(60, buffer, 6); break;
                case '9':
                        int2Bool(61, buffer, 6); break;
                case '+':
                        int2Bool(62, buffer, 6); break;
                case '/':
                        int2Bool(63, buffer, 6); break;
        }/*}}}*/
}

int str2intArray(char * c, int * int_arr, int bit_len, int str_len){
        int i;
        // Construct bit array
        int * bi_arr = malloc(str_len*6*sizeof(int));
        for(i=0; i<str_len; i++){
                Char2Binary(c, bi_arr);
                c += 1;
                bi_arr += 6;
        }
        bi_arr -= 6*str_len;
        
        // Construct operator to convert binary in to int
        int * operator_ = malloc(bit_len * sizeof(int));
        for(i=0; i<bit_len; i++){
                operator_[bit_len-1-i] = pow(2, i);
        }
        // Calculate int array
        int accumulator = 0;
        for(i=0; i<str_len*6; i += bit_len){
                for(int j=0; j<bit_len; j ++){
                        accumulator += operator_[j]*bi_arr[i + j];
                }
                * int_arr = accumulator;
                int_arr ++; 
                accumulator = 0;
        }

        free(bi_arr); free(operator_);
}
/*}}}*/

int main() {
    return 0;
}