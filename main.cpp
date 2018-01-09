//! @file Simple 2D Heat Diffusion simulator

#include <mpi.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

#include <string.h>

#define EPSILON 0.00001

using namespace std;

/// @struct Spot
/// @brief The struct with parameters of a spot with given temperature
struct Spot {
    unsigned int x; ///< X-coordination of the spot
    unsigned int y; ///< y-coordination of the spot
    unsigned int temperature; ///< temperature of the spot

    /// operator== Comparison of spots from coordination point of view
    /// @param b - Spot for comparation
    bool operator==(const Spot &b) const {
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
            } else if (lineId == 1) {
                ss >> height;
            } else {
                unsigned int i, j, temperature;
                ss >> i >> j >> temperature;
                spots.push_back({i, j, temperature});
            }
            lineId++;
        }
        file.close();
    } else {
        throw runtime_error("It is not possible to open instance file!\n");
    }
    return make_tuple(width, height, spots);
}

/// writeOutput - Method for creating resulting ppm image
/// @param myRank - Rank of the process
/// @param width - Width of the 2D space (image)
/// @param height - Height of the 2D space (image)
/// @param image - Linearized image
void writeOutput(const int myRank, const int width, const int height, const string instanceFileName, const double *image) {
    // Draw the output image
    ofstream file(instanceFileName);
    if (file.is_open()) {
        if (myRank == 0) {
            file << "P2\n" << width << "\n" << height << "\n" << 255 << "\n";
            for (unsigned long i = 0; i < width * height; i++) {
                file << static_cast<int> (image[i]) << " ";
            }
        }
    }
    file.close();
}

double ceilDouble(double value) {
    if(value>255) return 255;
    return value;
}

int differsLocal(double *oldBlock, double *newBlock, int blockWidth, int blockHeight) {
    bool differs = false;
    for (int y = 2; y < blockHeight - 2; ++y)
        for (int x = 0; x < blockWidth; ++x) {
            if (newBlock[y * blockWidth + x] >= 0) continue;
            double divide = 9.0f;
            double sum = oldBlock[y * blockWidth + x] + oldBlock[(y - 1) * blockWidth + x] + oldBlock[(y + 1) * blockWidth + x];
            if (x == 0)
                divide = 6.0f;
            else
                sum += oldBlock[y * blockWidth + x - 1] + oldBlock[(y - 1) * blockWidth + x - 1] + oldBlock[(y + 1) * blockWidth + x - 1];

            if (x == blockWidth - 1)
                divide = 6.0f;
            else
                sum += oldBlock[y * blockWidth + x + 1] + oldBlock[(y - 1) * blockWidth + x + 1] + oldBlock[(y + 1) * blockWidth + x + 1];
            sum /= (float) divide;
            newBlock[y * blockWidth + x] = ceilDouble(sum);
            if (fabs(newBlock[y * blockWidth + x] - oldBlock[y * blockWidth + x]) >= EPSILON) differs = true;
        }
    return differs;
}

int differsRemote(double *block, double *newBlock, int bw, int bh, int rank, int worldsize) {
    int num = 0, y = 1, different = 0;
    float sum = 0;
    for (int x = 0; x < bw; ++x) {
        if (newBlock[y * bw + x] >= 0) continue;
        num = 0;
        sum = 0;
        for (int y1 = -1; y1 < 2; ++y1) {
            if (rank == 0 && (y1 == -1)) continue;
            for (int x1 = -1; x1 < 2; ++x1) {
                if ((x == 0 && x1 == -1) || (x == bw - 1 && x1 == 1)) continue;
                num++;
                sum += block[(y + y1) * bw + x + x1];
            }
        }
        newBlock[y * bw + x] = ceilDouble(sum /(float) num);
        if (fabs(newBlock[y * bw + x] - block[y * bw + x]) >= EPSILON) different = 1;//there is still a point to convolute
    }

    y = bh - 2;
    for (int x = 0; x < bw; ++x) {
        if (newBlock[y * bw + x] >= 0)continue;
        num = 0;
        sum = 0;
        for (int y1 = -1; y1 < 2; ++y1) {
            if (rank == worldsize - 1 && (y1 == 1)) continue;
            for (int x1 = -1; x1 < 2; ++x1) {
                if ((x == 0 && x1 == -1) || (x == bw - 1 && x1 == 1)) continue;
                num++;
                sum += block[(y + y1) * bw + x + x1];
            }
        }
        newBlock[y * bw + x] = ceilDouble(sum /(float) num);
        if (fabs(newBlock[y * bw + x] - block[y * bw + x]) >= EPSILON) different = 1;//there is still a point to convolute
    }
    return different;
}

void splitSpots(int rank, Spot *spots, int spotsCount, double *block, int blockWidth, int blockHeight) {
    for (int i = 0; i < spotsCount; ++i)
        if (spots[i].y < blockHeight * rank + blockHeight && spots[i].y >= blockHeight * rank)
            block[((spots[i].y + 1 - blockHeight * rank) * blockWidth + spots[i].x)] = spots[i].temperature;
}

double *convolution(int rank, int worldSize, Spot *spots, int spotsCount, double *block, int blockWidth, int blockHeight) {
    MPI_Request sendRequestUpper, sendRequestLower, receiveRequestUpper, receiveRequestLower;
    double *newBlock = (double *) calloc(blockWidth * (blockHeight + 2), sizeof(double));
    while (true) {
        for (int i = 0; i < blockWidth * (blockHeight + 2); ++i) newBlock[i] = -1;
        if (rank != 0) {
            MPI_Isend(block + blockWidth, blockWidth, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &sendRequestUpper);
            MPI_Irecv(block, blockWidth, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &receiveRequestUpper);
        }
        if (rank < worldSize - 1) {
            MPI_Isend(block + blockWidth * blockHeight, blockWidth, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &sendRequestLower);
            MPI_Irecv(block + blockWidth * blockHeight + blockWidth, blockWidth, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &receiveRequestLower);
        }
        int stopConvolution = 0;
        splitSpots(rank, spots, spotsCount, newBlock, blockWidth, blockHeight);
        stopConvolution |= differsLocal(block, newBlock, blockWidth, blockHeight + 2);
        MPI_Status s;
        if (rank < worldSize - 1) MPI_Wait(&receiveRequestLower, NULL);
        if (rank != 0) MPI_Wait(&receiveRequestUpper, NULL);
        stopConvolution |= differsRemote(block, newBlock, blockWidth, blockHeight + 2, rank, worldSize);
        int shouldStop = 0;
        MPI_Allreduce(&stopConvolution, &shouldStop, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (shouldStop == 0) break;
        if (rank < worldSize - 1) MPI_Wait(&sendRequestLower, NULL);
        if (rank != 0) MPI_Wait(&sendRequestUpper, NULL);
        double *temporaryBlock = newBlock;
        newBlock = block;
        block = temporaryBlock;
    }
    free(newBlock);
    return block;
}

MPI_Datatype createMPIDatatype() {
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_spot;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Spot, x);
    offsets[1] = offsetof(Spot, y);
    offsets[2] = offsetof(Spot, temperature);
    int blocklenghts[3] = {1, 1, 1};
    MPI_Type_create_struct(3, blocklenghts, offsets, types, &mpi_spot);
    MPI_Type_commit(&mpi_spot);
    return mpi_spot;
}

/// main - Main method
int main(int argc, char **argv) {
    // Initialize MPI
    int worldSize, myRank;
    int initialised;
    MPI_Initialized(&initialised);
    if (!initialised) MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Datatype mpi_spot_t = createMPIDatatype();

    if (argc > 1) {
        // read the input instance
        unsigned int width, height, spotsCount, blockWidth, blockHeight;
        vector<Spot> spots;
        Spot *spotData;
        double *image; // linearized image
        if (myRank == 0) {
            tie(width, height, spots) = readInstance(argv[1]);
            spotsCount = static_cast<unsigned int>(spots.size());
            // round the spacesize of the image to be multiple of the number of used processes
            if (height % worldSize != 0) height = ((height / worldSize) + 1) * worldSize;
            if (width % worldSize != 0) width = ((width / worldSize) + 1) * worldSize;
            blockWidth = width;
            blockHeight = height / worldSize;
            image = new double[width * height];
            memset(image, '\0', width * height * 4);
        }

        //-----------------------\\
        // Insert your code here \\
        //        |  |  |        \\
        //        V  V  V        \\

        double *block;
        unsigned int *subproblemSize = new unsigned int[3];
        if (myRank == 0) {
            for (Spot &s : spots) image[s.y * width + s.x] = s.temperature;
            subproblemSize[0] = blockWidth;
            subproblemSize[1] = blockHeight;
            subproblemSize[2] = spotsCount;
            int size = (blockHeight + 2) * blockWidth;
            block = (double *) calloc(size, sizeof(double));
            MPI_Bcast(subproblemSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
            spotData = spots.data();
        } else {
            MPI_Bcast(subproblemSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
            blockWidth = subproblemSize[0];
            blockHeight = subproblemSize[1];
            spotsCount = subproblemSize[2];
            block = (double *) calloc((blockWidth * (blockHeight + 2)),sizeof(double));
            spotData = (Spot *) calloc(spotsCount,sizeof(Spot));
        }
        MPI_Bcast(spotData, spotsCount, mpi_spot_t, 0, MPI_COMM_WORLD);
        splitSpots(myRank, spotData, spotsCount, block, blockWidth, blockHeight);
        block = convolution(myRank, worldSize, spotData, spotsCount, block, blockWidth, blockHeight);
        MPI_Gather(block + blockWidth, blockWidth * blockHeight, MPI_DOUBLE, image, blockWidth * blockHeight, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(block);
        delete (subproblemSize);
        //-----------------------\\

        if (myRank == 0) {
            string outputFileName(argv[2]);
            writeOutput(myRank, width, height, outputFileName, image);
        }
    } else {
        if (myRank == 0)
            cout << "Input instance is missing!!!\n" << endl;
    }
    MPI_Finalize();
    return 0;
}
