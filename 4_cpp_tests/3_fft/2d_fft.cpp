#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;

void f_fft2d(int xsize,int ysize){
    // 2D FFT 
    int idx;
    double *real_arr;
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);
    real_arr= (double*) fftw_malloc(sizeof(double) * xsize*ysize);

    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            in[idx][0]=(double)x*10.0+y;
            in[idx][1]=(double)x*1.0+y*2.0;
            out[idx][0]=0.0;
            out[idx][1]=0.0;
    
            cout<<in[idx][0]<<","<<in[idx][1]<<": :";
        }
        cout<<endl;
     }
    
    // FFT part
    p = fftw_plan_dft_2d(xsize,ysize,in,out, FFTW_FORWARD, FFTW_ESTIMATE);
    //p = fftw_plan_dft_r2c_2d(xsize,ysize,in,out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    cout<<endl;   
    
    fftw_cleanup();
    
    // Print FFT result
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            cout<<out[idx][0]<<","<<out[idx][1]<<'\t';
        }
        cout<<endl;
    }
    // Absolute value
    cout<<endl;
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            real_arr[idx]=sqrt(pow(out[idx][0],2)+pow(out[idx][1],2));
            cout<<real_arr[idx]<<'\t';
        }
        cout<<endl;
    }

fftw_free(real_arr);
}

void f_fft2d_r2c(int xsize, int ysize){
    // 2D FFT 
    int idx;
    double *real_arr, *in;
    fftw_complex *out;
    fftw_plan p;
    
    in= (double*) fftw_malloc(sizeof(double) * xsize*ysize);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);
    real_arr= (double*) fftw_malloc(sizeof(double) * xsize*ysize);


    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            in[idx]=(double)x*10.0+y;
            out[idx][0]=0.0;
            out[idx][1]=0.0;
    
            cout<<in[idx]<<": :";
        }
        cout<<endl;
     }
    
    // FFT part
    p = fftw_plan_dft_r2c_2d(xsize,ysize,in,out,FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    cout<<endl;   
    
    fftw_cleanup();
    
    // Print FFT result
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            cout<<out[idx][0]<<","<<out[idx][1]<<'\t';
        }
        cout<<endl;
    }
    // Absolute value
    cout<<endl;
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            real_arr[idx]=sqrt(pow(out[idx][0],2)+pow(out[idx][1],2));
            cout<<real_arr[idx]<<'\t';
        }
        cout<<endl;
    }

fftw_free(real_arr);
}

int main()
{
    int xsize=5;
    int ysize=5;

//    f_fft2d(xsize,ysize);
    f_fft2d_r2c(xsize,ysize);
}

