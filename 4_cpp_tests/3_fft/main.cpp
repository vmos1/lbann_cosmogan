#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>

using namespace std;

int xsize=5;
int ysize=5;

void f_create_arr(float * img){
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            img[y*xsize+x]=rand() % 40;
        }} 
}

void f_read_image(string fname, float * img){
    
    ifstream f;
    string line;

//    cout<<fname<<endl;
    f.open(fname);  
    int i=0;

    if (f.is_open()){
        while (getline(f,line,',')){
//            cout<<line<<'\t';
            img[i]=stof(line);
            i++;
        } 
    } 
    else cout<<"File"<<fname<<"not found";        
    
    f.close();
} 


void f_print_arr(float * img){
    for (int y=0;y<ysize;y++){ 
        for (int x=0;x<xsize;x++){
            cout<<""<<img[y*xsize+x]<<"\t";
        }
        cout<<endl;
    } 
} 

void f_radial_profile(float *img, float *rprof, int max_r){
    int r_bins [max_r]; 
    float r_arr [max_r]; 
    float r; 
    float center_x, center_y;
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
        cout<<r_arr[i]<<'\t'<<r_bins[i];
    }
}

int main()
{
    string fname; 
    int max_r;  
    
    max_r=(int)sqrt(40) ;
//    cout<<"max r"<<max_r<<endl;
    
    float *img_arr= new float[xsize*ysize];
    float *rprof= new float[max_r];
    for (int i=0; i<max_r; i++) rprof[i]=0;

    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";
    
    // Get image 
//    f_create_arr(img_arr); 
    f_read_image(fname, img_arr);
//    f_print_arr(img_arr);

    // Compute radial profile of image 
//    f_radial_profile(img_arr,rprof, max_r);
//    for(int i=0;i<max_r;i++) cout<<rprof[i]<<' ';
    
    delete [] img_arr;
    delete [] rprof;
}


