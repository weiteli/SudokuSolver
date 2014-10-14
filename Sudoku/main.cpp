//
//  main.cpp
//  Sudoku
//
//  Created by Wei-Te Li on 14/6/18.
//  Copyright (c) 2014年 wade. All rights reserved.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SudokuSolver.h"

#define FEATUREDIMENSION 64
#define numClass 10
#define AREATHRES 36

using namespace std;
using namespace cv;

string int_to_string(const int& port) {
    stringstream ss;
    ss << port;
    return ss.str();
}

void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255)){
    if(line[1]!=0){
        float m = -1/tan(line[1]);
        float c = line[0]/sin(line[1]);
        
        cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
    }
    else{
        cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb);
    }
}

void getTrain(Mat &TrainingData, Mat &TrainingClass){
    FileStorage fs;
    string trainFile = "/Users/wei-teli/Documents/Developer/Sudoku/Sudoku/DigitData.xml";
    fs.open(trainFile, FileStorage::READ);
    vector<int> vectorClass;
    
    fs["TrainingData"] >> TrainingData;
    fs["classes"] >> vectorClass;
    
    TrainingClass = Mat::zeros(TrainingData.rows, numClass, CV_8UC1);
    
    for (int i=0; i<TrainingClass.rows; i++){
        TrainingClass.at<uchar>(i, vectorClass[i]) = 1;
    }
    
    TrainingClass.convertTo(TrainingClass, CV_32FC1);
    TrainingData.convertTo(TrainingData, CV_32FC1);
}

void findTwoExtremePoints(float rho, float theta, int width, int height, Point &pt1, Point &pt2){
    if (theta > CV_PI*45/180 && theta < CV_PI*135/180){
        pt1.x = 0;
        pt1.y = rho/sin(theta);
        
        pt2.x = width;
        pt2.y = pt2.x/tan(theta)+rho/sin(theta);
    } else{
        pt1.x = rho /cos(theta);
        pt1.y = 0;
        
        pt2.y = height;
        pt2.x = pt2.y/tan(theta)+rho/cos(theta);
    }
}

void mergeRelatedLines(vector<Vec2f> *lines, Mat &img){
    vector<Vec2f>::iterator current;
    //if the line has been fused together -> we will set rho = 0, theta = -100
    for(current=lines->begin();current!=lines->end();current++){
        if((*current)[0]==0 && (*current)[1]==-100) continue;
        
        float p1 = (*current)[0];
        float theta1 = (*current)[1];
        
        //Find two points for the line
        Point pt1current, pt2current;
        findTwoExtremePoints(p1, theta1, img.size().width, img.size().height, pt1current, pt2current);
        
        vector<Vec2f>::iterator pos;
        for(pos=lines->begin();pos!=lines->end();pos++){
            if(*current==*pos) continue;
            if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180){
                float p = (*pos)[0];
                float theta = (*pos)[1];
                Point pt1, pt2;
                findTwoExtremePoints(p, theta, img.size().width, img.size().height, pt1, pt2);
                
                if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) &&
                    ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64))
                {
                    // Merge the two
                    (*current)[0] = ((*current)[0]+(*pos)[0])/2;
                    (*current)[1] = ((*current)[1]+(*pos)[1])/2;
                    
                    (*pos)[0]=0;
                    (*pos)[1]=-100;
                }
            }
        }
    }
    
}


int main(int argc, const char * argv[]){
    bool visualize = true;
    Mat sudoku = imread("/Users/wei-teli/Desktop/pic/cbhsudoku.jpg",0);
    if (visualize){
        imshow("Original", sudoku);
    }
    Mat original = sudoku.clone();
    
    Mat box = Mat(sudoku.size(), CV_8UC1);
    GaussianBlur(sudoku, sudoku, Size(11,11), 0);
    adaptiveThreshold(sudoku, box, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 2);
    
    
    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(box,box, kernel);
    
    //Assumed the biggest thing in the picture to be the puzzle.
    int max=-1;
    Point maxPt;
    for(int y=0;y<box.size().height;y++){
        for(int x=0;x<box.size().width;x++){
            if (box.at<uchar>(y,x) >= 128){
                int area = floodFill(box, Point(x,y), CV_RGB(0,0,64));
                if(area>max){
                    maxPt = Point(x,y);
                    max = area;
                }
            }
        }
    }
    
    floodFill(box, maxPt, CV_RGB(255,255,255));
    if (visualize){
        imshow("Detected Sudoku", box);
    }
    
    for(int y=0;y<box.size().height;y++){
        for(int x=0;x<box.size().width;x++){
            if (box.at<uchar>(y, x)==64 && x!=maxPt.x && y!=maxPt.y){
                floodFill(box, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
    erode(box, box, kernel);

    vector<Vec2f> lines;
    HoughLines(box, lines, 1, CV_PI/180, 200);
    
    if (visualize){
        for (int i=0; i<lines.size(); i++){
            drawLine(lines[i], box);
        }
        imshow("All lines detected", box);
    }
    
    mergeRelatedLines(&lines, sudoku);
    
    //Find lines that forms the quad
    Vec2f topEdge = Vec2f(1000,0);
    Vec2f bottomEdge = Vec2f(-1000,0);
    Vec2f leftEdge = Vec2f(0,1000);
    Vec2f rightEdge = Vec2f(0,-1000);
    
    double leftXIntercept=100000;
    double rightXIntercept=0;
    
    for(int i=0;i<lines.size();i++){
        Vec2f current = lines[i];
        float p = current[0];
        float theta = current[1];
        if(p==0 && theta==-100) continue;
        double xIntercept, yIntercept;
        xIntercept = p/cos(theta);
        yIntercept = p/sin(theta);
        
        //Only consider vertical and horizontal lines
        if(theta>CV_PI*80/180 && theta<CV_PI*100/180){
            if(p<topEdge[0])
                topEdge = current;
            if(p>bottomEdge[0])
                bottomEdge = current;
        } else if(theta<CV_PI*10/180 || theta>CV_PI*170/180){
            if(xIntercept>rightXIntercept){
                rightEdge = current;
                rightXIntercept = xIntercept;
            } else if(xIntercept<=leftXIntercept){
                leftEdge = current;
                leftXIntercept = xIntercept;
            }
        }
    }
    
    if (visualize) {
        drawLine(topEdge, sudoku);
        drawLine(bottomEdge, sudoku);
        drawLine(leftEdge, sudoku);
        drawLine(rightEdge, sudoku);
        imshow("Draw lines", sudoku);
    }
    
    //Calculate the intersection points
    Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
    int height=box.size().height;
    int width=box.size().width;
    
    //Find two points from each line
    if(leftEdge[1]!=0){
        left1.x=0;        left1.y=leftEdge[0]/sin(leftEdge[1]);
        left2.x=width;    left2.y=-left2.x/tan(leftEdge[1]) + left1.y;
    } else{
        left1.x=leftEdge[0]/cos(leftEdge[1]);
        left1.y=0;
        left2.x=left1.x - height*tan(leftEdge[1]);
        left2.y=height;
    }
    
    if(rightEdge[1]!=0){
        right1.x=0;
        right1.y=rightEdge[0]/sin(rightEdge[1]);
        right2.x=width;
        right2.y=-right2.x/tan(rightEdge[1]) + right1.y;
    } else{
        right1.y=0;
        right1.x=rightEdge[0]/cos(rightEdge[1]);
        right2.y=height;
        right2.x=right1.x - height*tan(rightEdge[1]);
    }
    
    bottom1.x=0;
    bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);
    bottom2.x=width;
    bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;
    
    top1.x=0;
    top1.y=topEdge[0]/sin(topEdge[1]);
    top2.x=width;
    top2.y=-top2.x/tan(topEdge[1]) + top1.y;
    
    double leftA = left2.y-left1.y;
    double leftB = left1.x-left2.x;
    double leftC = leftA*left1.x + leftB*left1.y;
    
    double rightA = right2.y-right1.y;
    double rightB = right1.x-right2.x;
    double rightC = rightA*right1.x + rightB*right1.y;
    
    double topA = top2.y-top1.y;
    double topB = top1.x-top2.x;
    double topC = topA*top1.x + topB*top1.y;
    
    double bottomA = bottom2.y-bottom1.y;
    double bottomB = bottom1.x-bottom2.x;
    double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;
    
    // Intersection of left and top
    double detTopLeft = leftA*topB - leftB*topA;
    Point ptTopLeft = Point((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);
    
    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;
    Point ptTopRight = Point((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);
    
    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    Point ptBottomRight = Point((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);
    
    // Intersection of bottom and left
    double detBottomLeft = leftA*bottomB-leftB*bottomA;
    Point ptBottomLeft = Point((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);
    
    int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
    int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);
    if(temp>maxLength) maxLength = temp;
    
    temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);
    if(temp>maxLength) maxLength = temp;
    
    temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);
    if(temp>maxLength) maxLength = temp;
    
    maxLength = sqrt((double)maxLength);
    
    Point2f src[4], dst[4];
    src[0] = ptTopLeft;            dst[0] = Point2f(0,0);
    src[1] = ptTopRight;        dst[1] = Point2f(maxLength-1, 0);
    src[2] = ptBottomRight;        dst[2] = Point2f(maxLength-1, maxLength-1);
    src[3] = ptBottomLeft;        dst[3] = Point2f(0, maxLength-1);
    
    Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    warpPerspective(original, undistorted, cv::getPerspectiveTransform(src, dst), Size(maxLength, maxLength));
    if (visualize){
        imshow("distorted", undistorted);
    }
    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 111, 1);
    
    int maxx=-1;
    Point maxxPt;
    for(int y=0;y<undistortedThreshed.size().height;y++){
        for(int x=0;x<undistortedThreshed.size().width;x++){
            if (undistortedThreshed.at<uchar>(y,x) >= 128){
                int area = floodFill(undistortedThreshed, Point(x,y), CV_RGB(0,0,255));
                if(area>maxx){
                    maxxPt = Point(x,y);
                    maxx = area;
                }
            }
        }
    }
    
    floodFill(undistortedThreshed, maxxPt, CV_RGB(0,0,0));
    if (visualize) {
        imshow("Sudoku Threshold",undistortedThreshed);
    }
    
    Mat inv_undistorted;
    threshold(undistortedThreshed, inv_undistorted, 0, 255, CV_THRESH_BINARY_INV);
    if (visualize){
        imshow("Invert Sudoku", inv_undistorted);
    }
    vector<vector<Point> >digitContours;
    Mat tmp_undistorted = undistortedThreshed.clone();
    findContours(tmp_undistorted, digitContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    vector<vector<Point> >::iterator it = digitContours.begin();
    
    vector<Mat> rectMat;
    vector<Point> pts;
    
    while (it!=digitContours.end()) {
        Rect bb= boundingRect(Mat(*it));
        if (bb.area() > AREATHRES){
            pts.push_back(Point(bb.x, bb.y));
            Mat tmp(inv_undistorted, bb);
            resize(tmp, tmp, Size(8,8));
            tmp = tmp.reshape(1,1);
            rectMat.push_back(tmp);
        }
        it++;
    }
    
    //Artifical Neural Network
    Mat layers(3,1, CV_32S);
    layers.at<int>(0,0) = FEATUREDIMENSION;
    layers.at<int>(1,0) = 16;
    layers.at<int>(2,0) = numClass;
    
    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);
    CvANN_MLP_TrainParams params(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
    
    Mat TrainingData;
    Mat TrainingClass;
    
    getTrain(TrainingData, TrainingClass);
    nnetwork.train(TrainingData, TrainingClass, Mat(), Mat(), params);
    Mat classificationResult(1, numClass, CV_32F);
    vector<int>results;
    for (int i=0; i<rectMat.size(); i++) {
        Mat r = rectMat[i];
        r.convertTo(r, CV_32FC1);
        nnetwork.predict(r, classificationResult);
        Point maxloc;
        double maxVal;
        minMaxLoc(classificationResult, 0, &maxVal, 0, &maxloc);
        int idx = maxloc.x;
        results.push_back(idx);
    }
    
    int block_size = maxLength /9;
    vector<vector<int> >data(9, vector<int>(9,0));
    for(int i=0; i<pts.size(); i++){
        int j = (int)(pts[i].x)/block_size;
        int k = (int)(pts[i].y)/block_size;
        data[k][j] = results[i];
    }
    
    cout << "Problem: " << endl;
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            cout << data[i][j] << "\t";
        }
        cout << endl;
    }
    
    SudokuSolver solver(data);
    solver.init();
    solver.Solve(0, 0);
    solver.print();
    
    while(1){
        int c;
        c = cvWaitKey(10);
        if((char)c==27)
            break;
    }
    return 0;
}
