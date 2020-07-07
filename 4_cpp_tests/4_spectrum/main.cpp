#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;

void f_create_arr(double * img,int xsize, int ysize){
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            img[y*xsize+x]=rand() % 40;
    }} 
}

void f_read_file(string fname, double * img_arr){
    
    string line;
    ifstream f;

    f.open(fname);  
    int i=0;
    cout<<endl<<"File "<<fname<<endl;
    if (f.is_open()){
        while (getline(f,line,',')){
            img_arr[i]=stof(line);
            i++;
        } 
    } 
    else cout<<"File"<<fname<<"not found";        
    f.close();
}

void f_print_arr(double * img, int xsize, int ysize){
    for (int y=0;y<ysize;y++){ 
        for (int x=0;x<xsize;x++){
            cout<<""<<img[y*xsize+x]<<"\t";
        }
        cout<<endl;
    } 
} 

void f_radial_profile(double *img, double *rprof, int xsize, int ysize, int max_r){
    int r_bins [max_r]; 
    double r_arr [max_r]; 
    double r, center_x, center_y;
    int r_int;

    center_x=((xsize-1)-0)/2.0;
    center_y=((ysize-1)-0)/2.0;
     
    //Initialize values to 0
    for(int i=0; i<max_r; i++) {r_bins[i]=0; r_arr[i]=0;}
    
    for(int y=0; y<ysize; y++){
        for(int x=0; x<xsize; x++){
            r=sqrt(pow((x-center_x),2.0)+pow((y-center_y),2.0));
//            cout<<"x,y,r\t"<<x<<y<<r;
            r_int=int(r);
            r_bins[r_int]++;
            r_arr[r_int]+=img[y*xsize+x];
        }}
    
    for(int i=0;i<max_r;i++){
        if (r_bins[i]!=0)  rprof[i]=r_arr[i]/r_bins[i];  
    }
}

void f_fft2d(double *input_arr, double *output_arr, int xsize,int ysize){
    // 2D FFT 
    int idx;
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);

    // Create plan
    p = fftw_plan_dft_2d(xsize,ysize,in,out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            in[idx][0]=input_arr[idx];
            in[idx][1]=0.0;
            out[idx][0]=0.0;
            out[idx][1]=0.0;
    
//            cout<<in[idx][0]<<"+i "<<in[idx][1]<<"\t";
        }
//        cout<<endl;
     }
    
    // Compute FFT
    fftw_execute(p);
    // Print FFT result
 //   cout<<"result"<<endl;
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
//            cout<<out[idx][0]<<"+i "<<out[idx][1]<<'\t';
        }
//        cout<<endl;
    }

    // Absolute value
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            idx=y*xsize+x;
            output_arr[idx]=sqrt(pow(out[idx][0],2)+pow(out[idx][1],2));
        }}

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    fftw_cleanup();
}

void f_write_file(string fname,double *r_prof,int max_r){
    string line;
    ofstream f;
    f.open(fname);  
    for (int i=0; i<max_r; i++){
        f<<r_prof[i]<<",";
        }
    f.close();
 }

/*## Main code## */
int main()
{
    int xsize,ysize;
    string fname; 
    string op_fname;
    int max_r;
    double *img_arr, *output_arr, *r_prof;
    
    xsize=128;
    ysize=128; 
    max_r=(int)sqrt(2*xsize*ysize) ;
    img_arr= (double*) fftw_malloc(sizeof(double) * xsize*ysize);
    output_arr= (double*) fftw_malloc(sizeof(double) * xsize*ysize);
    r_prof= (double*) fftw_malloc(sizeof(double) * max_r);
    
    for (int i=0; i<max_r; i++) r_prof[i]=0;
    
    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";
    f_read_file(fname,img_arr);

    // Create image 
//    f_create_arr(img_arr,xsize,ysize); 
//    f_print_arr(img_arr,xsize,ysize);
    
    // FFT of image
    f_fft2d(img_arr,output_arr,xsize,ysize);
    // Compute radial profile of image 
//    cout<<"Modulus of output array"<<endl;
//    f_print_arr(output_arr,xsize,ysize);

    f_radial_profile(output_arr,r_prof, xsize, ysize, max_r);
    cout<<endl<<"Radial profile"<<endl;
    for(int i=0;i<max_r;i++) cout<<r_prof[i]<<'\t';
    cout<<endl;
    
    op_fname="../data/op.csv";
    f_write_file(op_fname,r_prof,max_r);

    fftw_free(img_arr); fftw_free(output_arr);fftw_free(r_prof);
}

