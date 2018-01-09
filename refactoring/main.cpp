//! @file Simple 2D Heat Diffusion simulator

#include <mpi.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

#include <string.h>
#include <math.h>
using namespace std;

/// @struct Spot
/// @brief The struct with parameters of a spot with given temperature
struct Spot {
    unsigned int x; ///< X-coordination of the spot
    unsigned int y; ///< y-coordination of the spot
    unsigned int temperature; ///< temperature of the spot

    /// operator== Comparison of spots from coordination point of view
    /// @param b - Spot for comparation
    bool operator==(const Spot& b) const
    {
        return (this->x == b.x) && (this->y == b.y);
    }
};

/// readInstance - Method for reading the input instance file
/// @param instanceFileName - File name of the input instance
/// @return Tuple of (Width of the space; Height of the Space; Vector of the Spot)
tuple<unsigned int, unsigned int, vector<Spot>> readInstance(const char *instanceFileName) {
    unsigned int width, height;
    vector<Spot> spots;
    string line;

    ifstream file(instanceFileName);
    if (file.is_open()) {
        int lineId = 0;
        while (std::getline(file, line)) {
            stringstream ss(line);
            if (lineId == 0) {
                ss >> width;
            }
            else if (lineId == 1) {
                ss >> height;
            }
            else {
                unsigned int i, j, temperature;
                ss >> i >> j >> temperature;
                spots.push_back({i, j, temperature});
            }
            lineId++;
        }
        file.close();
    }
    else {
        throw runtime_error("It is not possible to open instance file!\n");
    }
    return make_tuple(width, height, spots);
}

/// writeOutput - Method for creating resulting ppm image
/// @param myRank - Rank of the process
/// @param width - Width of the 2D space (image)
/// @param height - Height of the 2D space (image)
/// @param image - Linearized image
void writeOutput(const int myRank, const int width, const int height, const string instanceFileName, const float *image){//vector<int> image){
  // Draw the output image
  ofstream file(instanceFileName);
  if (file.is_open())
  {
    if (myRank == 0) {
      file << "P2\n" << width << "\n" << height << "\n" << 255 <<  "\n";
      for (unsigned long i = 0; i < width*height; i++) {
        file << static_cast<int> (image[i]) << " ";
      }
    }
  }
  file.close();
}
bool isSpot(int pixelx, int pixely, Spot * spots, int spotSize, int bw, int bh){//counts with global positions
   // pixel -= bw;//remove the upper line

    for (int i = 0; i < spotSize; ++i) {
        if(spots[i].y == pixely && spots[i].x == pixelx){
            return true;
        }
    }
    return false;
}
#define DIFF 0.00001
int filterOnInner(float * block, float * newBlock, int bw, int bh, Spot * spots, int spotSize, int rank){

    int different = 0; //1 if blocks are still not convergent, 0 if we done
    for (int y = 2; y < bh - 2; ++y) {
        for (int x = 0; x < bw; ++x) {
            if(newBlock[y*bw+x]>=0){//if(isSpot(x,(y-1)+(bh-2)*rank,spots,spotSize,bw,bh)) {//(bh-2)
                //newBlock[y*bw+x] = block[y*bw+x];
                continue;
            }
            int divide = 9;
            float sum = 0;
            //do 3 common to all
            sum+=block[y*bw + x];
            sum+=block[(y-1)*bw + x];
            sum+=block[(y+1)*bw + x];
            if (x==0) {//if start of line
                divide = 6;

            }else{
                //do 3 on the left
                sum+=block[y*bw + x-1];
                sum+=block[(y-1)*bw + x-1];
                sum+=block[(y+1)*bw + x-1];
            }
            if (x==bw-1) divide = 6;
            else{
                sum+=block[y*bw + x+1];
                sum+=block[(y-1)*bw + x+1];
                sum+=block[(y+1)*bw + x+1];
            }
            sum/=(float)divide;

            if (sum >255 ) sum = 255;
            newBlock[y*bw+x] = sum;
            if(fabs(sum-block[y*bw + x])>=DIFF) different = 1;//there is still a point to convolute
        }

    }
    return different;
}
int filterOnReceived(float * block, float * newBlock, int bw, int bh, int rank, int worldsize, Spot * spots, int spotSize){
    int  num = 0,  y=1,different = 0;
    float sum=0;
    for (int x = 0; x < bw; ++x) {
        if(newBlock[y*bw+x]>=0){//if(isSpot(x,(y-1)+(bh-2)*rank,spots,spotSize,bw,bh)){
            //newBlock[y*bw+x] = block[y*bw+x];
            continue;
        }
        num=0;
        sum=0;

        for (int y1 = -1; y1 < 2; ++y1) {
            if(rank == 0 && ( y1==-1)) continue;
            for (int x1 = -1; x1 < 2; ++x1) {
                if((x==0&&x1==-1)||(x==bw-1 && x1==1) ) continue;
                num++;
                sum+=block[(y+y1)*bw + x+x1];

            }
        }
        sum/=(float)num;

        if (sum >255 ) sum = 255;
        newBlock[y*bw+x] = sum;
        if(fabs(sum-block[y*bw + x])>=DIFF) different = 1;//there is still a point to convolute
    }

    y=bh-2;
    for (int x = 0; x < bw; ++x) {
        if(newBlock[y*bw+x]>=0){//if(isSpot(x,(y-1)+(bh-2)*rank,spots,spotSize,bw,bh)) {
            //newBlock[y*bw+x] = block[y*bw+x];
            continue;
        }
        num=0;
        sum=0;

        for (int y1 = -1; y1 < 2; ++y1) {
            if(rank == worldsize-1 && ( y1==1)) continue;
            for (int x1 = -1; x1 < 2; ++x1) {
                if((x==0&&x1==-1)||(x==bw-1 && x1==1) ) continue;
                num++;
                sum+=block[(y+y1)*bw + x+x1];

            }
        }
        sum/=(float)num;
        if (sum >255 ) sum = 255;
        newBlock[y*bw+x] = sum;
        if(fabs(sum-block[y*bw + x])>=DIFF) different = 1;//there is still a point to convolute
    }

    return different;

}
void burstSpotsIntoBlock(int rank, float * block, Spot * spots, int spotSize,int bw, int bh){//get there the original block size
    for (int i = 0; i < spotSize; ++i) {
        if  (spots[i].y < bh*rank+bh && spots[i].y >= bh*rank){
            block[((spots[i].y+1-bh*rank)*bw + spots[i].x)] = spots[i].temperature;
            //printf("ADD temp %d on %d,%d to block %d\n", spots[i].temperature,spots[i].x,spots[i].y,rank);
        }
    }
}
float * convoluteImage(int rank, int worldSize, float * block, int bw, int bh, Spot * spots, int spotsSize){
    MPI_Request reqSendUpper, reqRecvUpper,reqSendLower, reqRecvLower;
    int numIter = 0;
    float * newBlock = (float *)malloc(bw*(bh+2)* sizeof(float));

    while(1){
        for (int i = 0; i < bw*(bh+2); ++i) {
            newBlock[i]=-1;
        }
        if (rank!=0){
            MPI_Isend(block+bw, bw,MPI_FLOAT, rank-1,1,MPI_COMM_WORLD, &reqSendUpper);
            MPI_Irecv(block, bw,MPI_FLOAT, rank-1,1,MPI_COMM_WORLD, &reqRecvUpper);

        }
        if (rank < worldSize-1) {
            MPI_Isend(block + bw * bh, bw,MPI_FLOAT, rank+1,1,MPI_COMM_WORLD, &reqSendLower);
            MPI_Irecv(block + bw * bh + bw, bw,MPI_FLOAT, rank+1,1,MPI_COMM_WORLD, &reqRecvLower);
        }
        //printf("After waiting %d \n",rank);
        int endOfConvolution = 0; //0 done, 1 not yet done

        burstSpotsIntoBlock(rank,newBlock,spots,spotsSize,bw,bh);
        endOfConvolution |= filterOnInner(block,newBlock, bw,bh+2, spots,spotsSize, rank);
        //printf("After inner filter and about to wait for new data\n");
        MPI_Status s;
        if (rank < worldSize-1) {
            MPI_Wait(&reqRecvLower, NULL);
        }
        if (rank!=0) {
            MPI_Wait(&reqRecvUpper,NULL);
        }

        endOfConvolution |= filterOnReceived(block,newBlock,bw,bh+2,rank,worldSize, spots, spotsSize);

        int globalEnd = 0;
        MPI_Allreduce(&endOfConvolution, &globalEnd, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        //printf("Sent %d to allreduce, returned %d\n",endOfConvolution,globalEnd);
        if(globalEnd==0) {//if there is no difference, we are done for
            break;
        }
        if(rank < worldSize -1) MPI_Wait(&reqSendLower,NULL);
        if(rank!=0) MPI_Wait(&reqSendUpper,NULL);
        float *switcher = newBlock;
        newBlock=block;
        block=switcher;
        //if (rank==0){
           // printf("%d iteration done and still going on \n",numIter);
            numIter++;
            //if(numIter > 150000) break;
        //}
        //break;
    }
    free(newBlock);//if done, free the new block which is now same as previous
    return block;
}
MPI_Datatype createSpotType(){
    const  int nitems = 3;
    int blocklenghts[3]={1,1,1};
    MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};
    MPI_Datatype mpi_spot;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Spot, x);
    offsets[1] = offsetof(Spot, y);
    offsets[2] = offsetof(Spot, temperature);
    MPI_Type_create_struct(nitems,blocklenghts,offsets,types,&mpi_spot);
    MPI_Type_commit(&mpi_spot);
    return mpi_spot;
}


/// main - Main method
int main(int argc, char **argv) {
    // Initialize MPI
    int worldSize, myRank;
    int initialised;
    MPI_Initialized(&initialised);
    if(!initialised)
        MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Datatype mpi_spot = createSpotType();

    if (argc > 1) {
        // read the input instance
        unsigned int width, height, spotsSize, blockWidth,blockHeight;
        vector<Spot> spots;
        Spot *spotData;
        float * image; // linearized image
        if (myRank == 0) {
            tie(width, height, spots) = readInstance(argv[1]);
            spotsSize = static_cast<unsigned int>(spots.size());

            // round the spacesize of the image to be multiple of the number of used processes
            if (height % worldSize != 0) height = ((height / worldSize) + 1) * worldSize;
            if (width % worldSize != 0) width = ((width/worldSize) + 1) * worldSize;
            blockWidth = width;//we are sending chunks by rows only
            blockHeight = height / worldSize; //a row chunk
            image = new float[ width * height];
            memset(image,'\0',width*height*4);//memset works with bytes
        }

        //-----------------------\\
        // Insert your code here \\
        //        |  |  |        \\
        //        V  V  V        \\

       // printf("Hello from node %d \n", myRank);
        if (myRank==0){

            for (Spot &s : spots) {
                image[s.y*width + s.x] = s.temperature;
            }

            unsigned int *tmpWH = new unsigned int[3];
            tmpWH[0] = blockWidth;
            tmpWH[1] = blockHeight;
            tmpWH[2] = spotsSize;
            int size = (blockHeight+2)*blockWidth;//make some space for rows from other processors
            float *blockBuff = (float *)malloc(size*sizeof(float));//new int [size];
            memset(blockBuff,'\0',size*4);
            MPI_Bcast(tmpWH,3,MPI_INT,0,MPI_COMM_WORLD);
          //  printf("width and height sent \n");
            spotData = spots.data();
            MPI_Bcast(spotData,spotsSize,mpi_spot,0,MPI_COMM_WORLD);
            burstSpotsIntoBlock(myRank,blockBuff,spotData,spotsSize,blockWidth,blockHeight);
            //instead of Scatter, Bcast the Spots
            //MPI_Scatter(image,blockWidth*blockHeight,MPI_INT, blockBuff+blockWidth,blockWidth*blockHeight, MPI_INT, 0, MPI_COMM_WORLD);
            //printf("After Scatter 0\n Received block with %d at 0,0\n",blockBuff[blockWidth]);
            blockBuff = convoluteImage(myRank,worldSize,blockBuff,blockWidth,blockHeight, spotData, spotsSize);
          //  printf("Convolution done %d, lets gather the img\n",myRank);
            MPI_Gather(blockBuff+blockWidth,blockWidth*blockHeight,MPI_FLOAT,image,blockWidth*blockHeight,MPI_FLOAT,0,MPI_COMM_WORLD);
            free(blockBuff);
            delete(tmpWH);
        }else{
            int * tmpWH = new int[3];;

            MPI_Bcast(tmpWH,3,MPI_INT,0,MPI_COMM_WORLD);
            int bw = tmpWH[0], bh = tmpWH[1];
            spotsSize = tmpWH[2];
            int size = bw*(bh+2);
            float *blockBuff = (float *)malloc(size*sizeof(float));//new int [bw*bh];
            //memset(blockBuff,'\0',size*4);
            spotData = (Spot *)malloc(spotsSize* sizeof(Spot));
          //  printf("node %d received width %d and height %d from bcast\n",myRank,bw,bh);
            MPI_Bcast(spotData,spotsSize,mpi_spot,0,MPI_COMM_WORLD);

            burstSpotsIntoBlock(myRank, blockBuff,spotData,spotsSize,bw,bh);
            //MPI_Scatter(NULL,bw*bh,MPI_INT, blockBuff + bw,bw*bh, MPI_INT, 0, MPI_COMM_WORLD);//get the data on second row address
            //instea//instead of Scatter, Bcast the Spots
            //printf("After Scatter 1\n");
            blockBuff = convoluteImage(myRank,worldSize,blockBuff,bw,bh, spotData,spotsSize);
           // printf("Convolution done %d, lets gather the img\n",myRank);
            MPI_Gather(blockBuff+bw,bw*bh,MPI_FLOAT,image,bw*bh,MPI_FLOAT,0,MPI_COMM_WORLD);


            free(blockBuff);
            delete(tmpWH);
        }

        //-----------------------\\

       	if(myRank == 0)
       	{
        	string outputFileName(argv[2]);
        	writeOutput(myRank, width, height, outputFileName, image);
    	}
    }
    else {
        if (myRank == 0)
            cout << "Input instance is missing!!!\n" << endl;
    }
    MPI_Finalize();
    return 0;
}
