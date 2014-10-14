//
//  SudokuSolver.cpp
//  Sudoku
//
//  Created by Wei-Te Li on 14/6/22.
//  Copyright (c) 2014年 wade. All rights reserved.
//

#include "SudokuSolver.h"

SudokuSolver::SudokuSolver(vector<vector<int> >data){
    board = data;
    sol = data;
}

void SudokuSolver::init(){
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            if (board[i][j]){
                int tmp = board[i][j];
                mx[i][tmp] = my[j][tmp] = mg[i/3][j/3][tmp] = true;
            }
        }
    }
}

void SudokuSolver::Solve(int x, int y){
    if (y==9){
        x++;
        y=0;
    }
    if (x==9){
        flag = true;
        return;
    }
    if (board[x][y]) {
        Solve(x,y+1);
        return;
    }
    
    for (int n=1; n<=9; n++){
        if (!mx[x][n] && !my[y][n] && !mg[x/3][y/3][n]){
            mx[x][n] = true; my[y][n] = true; mg[x/3][y/3][n]=true;
            sol[x][y] = n;
            Solve(x,y+1);
            if (flag) return;
            mx[x][n]=false; my[y][n]=false; mg[x/3][y/3][n]=false;
        }
    }
}

void SudokuSolver::print(){
    cout << "Solved: " << endl;
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            cout << sol[i][j] << "\t";
        }
        cout << endl;
    }
}



