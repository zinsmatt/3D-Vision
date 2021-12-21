#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    // read the images and database  
    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab("vocabulary.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // bad results (maybe the vocab file was created with another version ?)
    if (vocab.empty()) {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }
    cout << "reading images... " << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "../data/" + to_string(i + 1) + ".png";
        images.push_back(imread(path));
    }

    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    std::vector<DBoW3::BowVector> bow_vectors;
    for (Mat &image:images) {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        DBoW3::BowVector bv;
        vocab.transform(descriptor, bv);
        bow_vectors.push_back(bv);
    }

    for (int i = 0; i < bow_vectors.size(); ++i)
    {
        auto score = vocab.score(bow_vectors[0], bow_vectors[i]);
        std::cout << "img " << i << " : " << score << "\n";
    }

}