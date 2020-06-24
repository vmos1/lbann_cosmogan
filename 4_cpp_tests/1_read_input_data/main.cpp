#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main()
{
    int xsize,ysize;

    xsize=5;
    ysize=5;
    float img_arr [xsize*ysize];
    
    for (int y=0;y<ysize;y++){
        for (int x=0;x<xsize;x++){
            img_arr[y*xsize+x]=0.0;
//            cout<<img_arr[y*xsize+x]<<"\t";
        }
//        cout<<endl;
        } 


    // Read entries from file
    string fname;
    ifstream f;
    string line;
    float val;

    fname="/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/4_cpp_tests/data/images.csv";
//    cout<<fname;
   
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
    
    // print data
    for (int y=0;y<ysize;y++){ 
        for (int x=0;x<xsize;x++){
            cout<<img_arr[y*xsize+x]<<"\t";
        }
        cout<<endl;
    } 

}
