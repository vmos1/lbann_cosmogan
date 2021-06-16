/* Code to compute spectrum of a batch of 2D images array with multiple channels
array dimension is (batch_size, num_channels, xsize,ysize)
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

double f_transform(double x){
    // Transformation function
    double ans;
    ans=((2.0 * x )/(x+4 )) -1 ;
    return ans;
}

double f_invtransform(double s){
    // Transformation function
    double ans;
    ans=(4 * (1 + s)/(1-s));
    return ans;
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
    else cout<<"File "<<fname<<" not found";        
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

void f_hist2d(double *input_arr, double *hist_arr, float *bin_edges, int batch_size, int num_channels, int xsize,int ysize, int num_bins){

    int i,j,a,b,idx1,idx2;
    
    for(a=0;a<batch_size;a++){
        for(b=0;b<num_channels;b++){
            for(i=0;i<(xsize*ysize);i++){
                idx1=i+b*(xsize*ysize)+a*(num_channels*xsize*ysize);
                for(j=0;j<num_bins;j++){  // Iterate over bin edges
                    idx2=j+b*(num_bins)+a*(num_channels*num_bins);
                    if(input_arr[idx1]<=bin_edges[j+1]){
                        hist_arr[idx2]++;
//                        printf("%d %d %d %d && ",a,b,i,j);
//                        cout<<input_arr[idx1]<<" "<<bin_edges[j+1]<<" "<<hist_arr[idx2]<<'\t'; 
                        break;
                       }
            }} }} cout<<endl;
}




void f_avg_samples(double *ip_arr, double *arr_mean, double *arr_sdev, int batch_size, int num_channels, int xsize, int ysize, int N){
    
    int idx1,idx2;
    
    for (int b=0;b<num_channels;b++){ 
        for(int i=0; i<N; i++){
            idx2=i+N*b;                
            for (int a=0;a<batch_size;a++){
                idx1=i+N*b+(N*num_channels)*a;                
                arr_mean[idx2]+=ip_arr[idx1]; //mean
                arr_sdev[idx2]+=pow(ip_arr[idx1],2); //variance
                }
            arr_mean[idx2]=arr_mean[idx2]/(batch_size*1.0);
            arr_sdev[idx2]=sqrt(arr_sdev[idx2]/(batch_size*1.0)-pow(arr_mean[idx2],2));
            
            }}
    }

void f_write_file(string fname,double *r_prof, int num_channels, int num_bins){
    string line;
    ofstream f;
    f.open(fname);  
    
    for (int i=0; i<num_channels; i++){
        for (int j=0; j<num_bins; j++) f<<r_prof[j+num_bins*i]<<",";
        }
    f.close();
 }

/*## Main code## */
int main()
{
    int xsize,ysize,batch_size,num_channels;
    int i;
    float f;
    string fname; 
    string op_fname;
    int bin_size,num_bins;
    double *img_arr, *hist_arr, *hist_mean, *hist_sdev;

    num_bins=30;
    float bin_edges[num_bins+1];
    int bin_centers[num_bins];
    xsize=128;
    ysize=128; 
    num_channels=2;
    batch_size=200;
    
    // Defining the bin edges 
    for (i=0,f=-0.5;i<10,f<10.5;i++,f++) bin_edges[i]=f;
    for (i=10,f=10;i<20,f<110;i++,f=f+10) bin_edges[i]=f;
    for (i=20,f=300;i<num_bins+1,f<2400;i++,f=f+200) bin_edges[i]=f;
    // Define bin centers
    for (int i=0;i<num_bins;i++) bin_centers[i]=(bin_edges[i+1]-bin_edges[i])/2.0+bin_edges[i];
    for (int i=0;i<num_bins+1;i++) cout<<bin_edges[i]<<","; cout<<endl;
    // Define bin edges with transformation
    for (int i=0;i<num_bins+1;i++) bin_edges[i]=f_transform(bin_edges[i]);
    // print transformed bin edges
    //for (int i=0;i<num_bins+1;i++) cout<<bin_edges[i]<<"\t";
    //cout<<endl;
    for (int i=0;i<num_bins;i++) {cout<<bin_centers[i]<<"\t";}
     
    img_arr    = (double*) malloc(sizeof(double) * batch_size * num_channels * xsize* ysize);
    hist_arr   = (double*) malloc(sizeof(double) * batch_size * num_channels * num_bins);
    hist_mean  = (double*) malloc(sizeof(double) * num_channels * num_bins);
    hist_sdev  = (double*) malloc(sizeof(double) * num_channels * num_bins);

    for (int i=0; i<(batch_size * num_channels * num_bins); i++) hist_arr[i]=0;
    for (int i=0; i<(num_channels * num_bins-1); i++) {hist_mean[i]=0; hist_sdev[i]=0;}
    
    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";
    f_read_file(fname,img_arr);

    // Create image 
    //f_print_arr(img_arr,batch_size, num_channels,xsize,ysize);
    
    // Histogram of image
    f_hist2d(img_arr, hist_arr, bin_edges, batch_size, num_channels, xsize, ysize, num_bins);
    
//    for (int i=0; i<(batch_size * num_channels * num_bins); i++) cout<<hist_arr[i]<<'\t';
    
    // Compute average and stdev of spectrum over batches
    f_avg_samples(hist_arr,hist_mean,hist_sdev,batch_size,num_channels,xsize,ysize,num_bins);
    
    /*
    cout<<endl<<"Mean Histogram"<<endl;
    for (int j=0; j< num_channels; j++){
        for (int k=0; k<num_bins; k++)  cout<<hist_mean[k+j*(num_bins)]<<'\t';
        cout<<endl<<"----"<<endl;
        for (int k=0; k<num_bins; k++)  cout<<hist_sdev[k+j*(num_bins)]<<'\t';
    cout<<endl<<endl;}
    */     

    op_fname="../data/op_hist_mean.csv";
    f_write_file(op_fname,hist_mean,num_channels,num_bins);
    
    op_fname="../data/op_hist_sdev.csv";
    f_write_file(op_fname,hist_sdev,num_channels,num_bins);

    free(img_arr); free(hist_arr); free(hist_mean); free(hist_sdev);
}

