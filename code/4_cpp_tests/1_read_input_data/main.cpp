#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void f_read_file(string fname, float * img_arr){
    
    string line;
    ifstream f;

    f.open(fname);  
    int i=0;

    if (f.is_open()){
        while (getline(f,line,',')){
//            cout<<line<<'\t';
            img_arr[i]=stof(line);
            i++;
        } 
    } 
    else cout<<"File"<<fname<<"not found";        
    
    f.close();
 

}


int main()
{
    int xsize,ysize;
    string fname;

    xsize=5;
    ysize=5;
    float img_arr [xsize*ysize];
    
    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";

    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            img_arr[y*xsize+x]=0.0;
        }
        } 


    // Read entries from file
    f_read_file(fname,img_arr); 
   
    // print results
    for (int y=0;y<ysize;y++){ 
        for (int x=0;x<xsize;x++){
            cout<<img_arr[y*xsize+x]<<"\t";
        }
        cout<<endl;
    } 

}
