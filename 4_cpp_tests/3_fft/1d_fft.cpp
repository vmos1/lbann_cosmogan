#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;

void f_fft1d(int N){
    
    double *real_arr;
    fftw_complex *in, *out;
    fftw_plan p;
    int sgn;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    real_arr= (double*) fftw_malloc(sizeof(double) * N);
    
    // Create plan
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    for (int i=0;i<N;i++){
//        in[i][0]=(double)(i+1)/10.0;
        sgn=1.0;
        //if (N%2==0) sgn=pow(-1,i); // One way to implement fftshift
        in[i][0]=(i+1)*sgn*12.0;
        in[i][1]=(i+1.0)*0*(-1);
        out[i][0]=0.0;
        out[i][0]=0.0;
    } 

    for (int i=0;i<N;i++){
        cout<<in[i][0]<<"+ "<<in[i][1]<<"i \t";
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
    
    // Compute fft 
    fftw_execute(p); /* repeat as needed */

    cout<<endl<<"Result"<<endl;
    for (int i=0;i<N;i++){
    //    cout<<in[i][0]<<"+ "<<in[i][1]<<"i \t";
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }

    // Absolute value
    for (int i=0;i<N;i++){
        real_arr[i]=sqrt(pow(out[i][0],2)+pow(out[i][1],2));
        cout<<real_arr[i]<<'\t';
    }
    cout<<endl;

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    fftw_cleanup(); 
}


void f_fft1d_r2c(int N){

    double *in, *real_arr;
    fftw_complex *out;
    fftw_plan p;
    int sgn;

    in= (double*) fftw_malloc(sizeof(double) * N);
    out= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    real_arr= (double*) fftw_malloc(sizeof(double) * N);

    // Create plan
    p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    
    for (int i=0;i<N;i++){

        sgn=1.0;
//        if (N%2==0) sgn=pow(-1,i); // One way to implement fftshift

        in[i]=(i+1)*sgn*12.0;
        out[i][0]=0.0;
        out[i][0]=0.0;
    } 

    for (int i=0;i<N;i++){
        cout<<in[i]<<"\t";
//        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
    
    // Compute fft 
    fftw_execute(p); /* repeat as needed */

    cout<<endl<<"Result"<<endl;
    for (int i=0;i<N;i++){
        cout<<out[i][0]<<"+ "<<out[i][1]<<"i "<<endl;
    }
    
    // Absolute value
    for (int i=0;i<N;i++){
        real_arr[i]=sqrt(pow(out[i][0],2)+pow(out[i][1],2));
        cout<<real_arr[i]<<'\t';
    }
    cout<<endl;

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    fftw_cleanup(); 
}

/*################*/
// Main code 
int main()
{
    int N=6;
    cout<<endl<<"Real to complex"<<endl;
    f_fft1d_r2c(N);
    cout<<endl<<"Complex to complex"<<endl;
    f_fft1d(N);
}

