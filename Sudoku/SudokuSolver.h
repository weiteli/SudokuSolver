//
//  SudokuSolver.h
//  Sudoku
//
//  Created by Wei-Te Li on 14/6/22.
//  Copyright (c) 2014年 wade. All rights reserved.
//

#ifndef __Sudoku__SudokuSolver__
#define __Sudoku__SudokuSolver__

#include <iostream>
#include <vector>
using namespace std;


class SudokuSolver{
private:
    bool mx[9][10] = {false};
    bool my[9][10] = {false};
    bool mg[3][3][10]={false};
    vector<vector<int> > board;
    vector<vector<int> > sol;
    bool flag = false;
public:
    SudokuSolver(vector<vector<int> >data);
    void Solve(int x, int y);
    void init();
    void print();
};
#endif /* defined(__Sudoku__SudokuSolver__) */
