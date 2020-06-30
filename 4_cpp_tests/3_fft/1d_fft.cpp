#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;
int xsize=5;
int ysize=5;


void f_fft1d(int N){

    fftw_complex *in, *out;
    fftw_plan p;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int i=0;i<N;i++){
//        in[i][0]=(double)(i+1)/10.0;
        in[i][0]=pow(i,3.2);
        in[i][1]=(i+1.0)*0.0/(-10);
        //in[i][0]=0.0;
        //in[i][1]=0;
        //if (i==0)in[i][0]=1.0;
        out[i][0]=0.0;
        out[i][0]=0.0;
    } 

    for (int i=0;i<N;i++){
        cout<<in[i][0]<<"+ "<<in[i][1]<<"i \t";
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
    
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p); /* repeat as needed */
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    cout<<endl;   

    cout<<"Result"<<endl;
    for (int i=0;i<N;i++){
//        cout<<i<<endl;
    //    cout<<in[i][0]<<"+ "<<in[i][1]<<"i \t";
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
}
void f_fft1d_r2c(int N){

    double *in;
    fftw_complex *out;
    fftw_plan p;
    in= (double*) fftw_malloc(sizeof(double) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int i=0;i<N;i++){
        in[i]=pow(i,3.2);
        out[i][0]=0.0;
        out[i][0]=0.0;
    } 

    for (int i=0;i<N;i++){
        cout<<in[i]<<"\t";
//        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
    
//    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(p); /* repeat as needed */
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    cout<<endl;   
    
    cout<<"Result"<<endl;
    for (int i=0;i<N;i++){
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
}




int main()
{
    int N=5;
    cout<<endl<<"Real to complex"<<endl;
    f_fft1d_r2c(N);
    cout<<endl<<"Complex to complex"<<endl;
    f_fft1d(N);
}

