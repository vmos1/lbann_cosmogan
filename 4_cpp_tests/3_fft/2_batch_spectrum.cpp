/* Code to compute spectrum of a batch of 2D images array with multiple channels
array dimension is (batch_size, num_channels, xsize,ysize)
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;

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

void f_print_arr(double * img, int batch_size, int num_channels, int xsize, int ysize){
    for (int a=0;a<batch_size;a++){ 
        for (int b=0;b<num_channels;b++){ 
            for (int y=0;y<ysize;y++){ 
                for (int x=0;x<xsize;x++){
                    cout<<""<<img[x+y*xsize+(xsize*ysize)*b+(xsize*ysize*num_channels)*a]<<"\t";
                }cout<<"###  ";
            }
            cout<<"channel"<<b<<endl;
            }        
        cout<<"----"<<endl<<"next sample"<<endl;
        }
} 

void f_fft2d(double *input_arr, double *output_arr, int batch_size, int num_channels, int xsize,int ysize){
    // 2D FFT 
    int idx1, idx2;
    fftw_complex *in, *out;
    fftw_plan p;
    
    in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xsize*ysize);

    // Create plan
    p = fftw_plan_dft_2d(xsize,ysize,in,out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int a=0;a<batch_size;a++){ 
        for (int b=0;b<num_channels;b++){ 
            for (int y=0;y<ysize;y++){
                for (int x=0;x<xsize;x++){
                    idx1=x+y*xsize+(xsize*ysize)*b+(xsize*ysize*num_channels)*a; // Index for batch array
                    idx2=x+y*xsize; // Index within each 2D array
                    in[idx2][0]=input_arr[idx1];
                    in[idx2][1]=0.0;
                    out[idx2][0]=0.0;
                    out[idx2][1]=0.0;
                    }}
    
            // Compute FFT
            fftw_execute(p);
            // Absolute value
            for (int y=0;y<ysize;y++){
                for (int x=0;x<xsize;x++){
                    idx1=x+y*xsize+(xsize*ysize)*b+(xsize*ysize*num_channels)*a;
                    idx2=x+y*xsize; 
                    output_arr[idx1]=sqrt(pow(out[idx2][0],2)+pow(out[idx2][1],2));
                }} }}

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    fftw_cleanup();
}

void f_radial_profile(double *img, double *rprof, int batch_size, int num_channels, int xsize, int ysize, int max_r){
    int r_bins [max_r]; 
    double r_arr [max_r]; 
    double r, center_x, center_y;
    int r_int,idx1,idx2;

    center_x=((xsize-1)-0)/2.0;
    center_y=((ysize-1)-0)/2.0;

    for(int a=0;a<batch_size;a++){ 
        for(int b=0;b<num_channels;b++){ 
            //Initialize values to 0
            for(int i=0; i<max_r; i++) {r_bins[i]=0; r_arr[i]=0;}
            
            for(int y=0; y<ysize; y++){
                for(int x=0; x<xsize; x++){
                    r=sqrt(pow((x-center_x),2.0)+pow((y-center_y),2.0));
        //            cout<<"x,y,r\t"<<x<<y<<r;
                    r_int=int(r);
                    r_bins[r_int]++;
                    idx1=x+y*xsize+(xsize*ysize)*b+(xsize*ysize*num_channels)*a;
                    r_arr[r_int]+=img[idx1];
            }}
            for(int i=0;i<max_r;i++){
                idx2=i+max_r*b+(max_r*num_channels)*a;                
                if (r_bins[i]!=0)  rprof[idx2]=r_arr[i]/r_bins[i];  
            }
        }}
}

void f_avg_spec(double *rprof, double *spec_mean, double *spec_sdev, int batch_size, int num_channels, int xsize, int ysize, int max_r){
    
    int idx1,idx2;
    
    for (int b=0;b<num_channels;b++){ 
        for(int i=0; i<max_r; i++){
            idx2=i+max_r*b;                
            for (int a=0;a<batch_size;a++){
                idx1=i+max_r*b+(max_r*num_channels)*a;                
                spec_mean[idx2]+=rprof[idx1]; //mean
                spec_sdev[idx2]+=pow(rprof[idx1],2); //variance
                }
            spec_mean[idx2]=spec_mean[idx2]/(batch_size*1.0);
            spec_sdev[idx2]=sqrt(spec_sdev[idx2]/(batch_size*1.0)-pow(spec_mean[idx2],2));
            
            }}
    }

void f_write_file(string fname,double *r_prof, int num_channels, int max_r){
    string line;
    ofstream f;
    f.open(fname);  
    
    for (int i=0; i<num_channels; i++){
        for (int j=0; j<max_r; j++) f<<r_prof[j+max_r*i]<<",";
        }
    f.close();
 }

/*## Main code## */
int main()
{
    int xsize,ysize,batch_size,num_channels;
    string fname; 
    string op_fname;
    int max_r;
    double *img_arr, *output_arr, *r_prof, *spec_mean, *spec_sdev;
    
    xsize=128;
    ysize=128; 
    num_channels=5;
    batch_size=20;
    
    max_r=(int)sqrt(xsize*ysize) ;
    img_arr    = (double*) fftw_malloc(sizeof(double) * batch_size * num_channels * xsize* ysize);
    output_arr = (double*) fftw_malloc(sizeof(double) * batch_size * num_channels * xsize* ysize);
    r_prof     = (double*) fftw_malloc(sizeof(double) * batch_size * num_channels * max_r);
    spec_mean  = (double*) fftw_malloc(sizeof(double) * num_channels * max_r);
    spec_sdev  = (double*) fftw_malloc(sizeof(double) * num_channels * max_r);

    for (int i=0; i<(batch_size * num_channels * max_r); i++) r_prof[i]=0;
    for (int i=0; i<(num_channels * max_r); i++) {spec_mean[i]=0; spec_sdev[i]=0;}
    
    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";
    f_read_file(fname,img_arr);

    // Create image 
//    f_create_arr(img_arr,xsize,ysize); 
//     f_print_arr(img_arr,batch_size, num_channels,xsize,ysize);
    
    // FFT of image
    f_fft2d(img_arr,output_arr,batch_size, num_channels, xsize,ysize);
    // Compute radial profile of image 
//    cout<<"Modulus of output array"<<endl;
//     f_print_arr(output_arr,batch_size, num_channels, xsize,ysize);
    
    f_radial_profile(output_arr,r_prof, batch_size, num_channels, xsize, ysize, max_r);
    /*
    cout<<endl<<"Radial profile"<<endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j< num_channels; j++){
            for (int k=0; k<max_r; k++)  cout<<r_prof[k+j*(max_r)+i*(num_channels*max_r)]<<'\t';
            cout<<endl;}
        cout<<endl;}
     */
    // Compute average and stdev of spectrum over batches
    f_avg_spec(r_prof,spec_mean,spec_sdev,batch_size,num_channels,xsize,ysize,max_r);
    
    cout<<endl<<"Mean spectrum"<<endl;
    for (int j=0; j< num_channels; j++){
        for (int k=0; k<max_r; k++)  cout<<spec_mean[k+j*(max_r)]<<'\t';
        cout<<"----";
        for (int k=0; k<max_r; k++)  cout<<spec_sdev[k+j*(max_r)]<<'\t';
    cout<<endl<<endl;}
    
    op_fname="../data/op_mean.csv";
    f_write_file(op_fname,spec_mean,num_channels,max_r);
    
    op_fname="../data/op_sdev.csv";
    f_write_file(op_fname,spec_sdev,num_channels,max_r);

    fftw_free(img_arr); fftw_free(output_arr);fftw_free(r_prof); fftw_free(spec_mean); fftw_free(spec_sdev);
}

