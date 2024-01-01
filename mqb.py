# -*- coding=utf-8 -*-

"""
Skeleton for TP project: Searching Maximum Quasi-Bicliques
Student:

Install Gurobi Solver with free academic licence using your University account at: https://www.gurobi.com/features/academic-named-user-license/

Install Pandas, using PyPI or Conda. A detailed installation tutorial can be find at: https://pandas.pydata.org/docs/getting_started/install.html
"""

from pulp import (
    GUROBI_CMD,
    PULP_CBC_CMD,
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)

import sys
from argparse import ArgumentParser
import pandas as pd

# ============================================================================ #
#                     LP MODEL - MINIMUM WEIGHTED COVER                        #
# ============================================================================ #


def min_model(rows_data, cols_data, edges):
    """
    Implement the LP model for minimum weighted cover.
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * edges: list of tuple of the coordination (row,col) of edges/undesired cell of the matrix.
    """

    # ------------------------------------------------------------------------ #
    # Model with minimization
    # ------------------------------------------------------------------------ #
    model = LpProblem(name='min_weighted_cover', sense=LpMinimize)

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #
    lpRows = {row: (LpVariable(f'row_{row}', cat='Binary',
                    lowBound=0), degree) for row, degree in rows_data}
    lpCols = {col: (LpVariable(f'col_{col}', cat='Binary',
                    lowBound=0), degree) for col, degree in cols_data}
    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #
    model += lpSum([degree*lpvar for lpvar, degree in lpRows.values()] +
                   [degree*lpvar for lpvar, degree in lpCols.values()]), 'min_weighted_cover'

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #
    for row, col in edges:
        model += (lpRows[row][0]+lpCols[col][0] >= 1), f'edge_{row}_{col}'

    # model += (lpSum([lpvar for lpvar, degree in lpRows.values()]) >= 1), f'row_constraint'
    # model += (lpSum([lpvar for lpvar, degree in lpCols.values()]) >= 1), f'col_constraint'

    # model += (lpSum([1-lpvar for lpvar, degree in lpRows.values()]) >= 5), f'row_constraint'
    # model += (lpSum([1-lpvar for lpvar, degree in lpCols.values()]) >= 5), f'col_constraint'

    return model

# ============================================================================ #
#                LP MODEL - PRECISE MINIMUM WEIGHTED COVER                     #
# ============================================================================ #


def min_model_rc(rows_data, cols_data, edges, epsilon=0.3):
    """
    Implement the LP model for minimum weighted cover. 
    In this model, we introduce sigma and epsilon with the goal of alowing error
    in the result. 
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * edges: list of tuple of the coordination (row,col) of edges/undesired cell of the matrix.
    * sigma: percentage of deleted edges over all edges (sensitivity)
    * epsilon: percentage of errors (accepted edges) in the final result submatrix
    """

    # ------------------------------------------------------------------------ #
    # Model with minamization
    # ------------------------------------------------------------------------ #
    model = LpProblem(name='precise_min_weighted_cover', sense=LpMinimize)

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #
    lpRows = {row: (LpVariable(f'row_{row}', cat='Integer',
                    lowBound=0, upBound=1), degree) for row, degree in rows_data}
    lpCols = {col: (LpVariable(f'col_{col}', cat='Integer',
                    lowBound=0, upBound=1), degree) for col, degree in cols_data}
    lpEdges = {(row, col): LpVariable(f'edge_{row}_{col}', cat='Integer',
                                      lowBound=0, upBound=1) for row, col in edges}

    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #
    model += lpSum([degree*lpvar for lpvar, degree in lpRows.values()] +
                   [degree*lpvar for lpvar, degree in lpCols.values()]), 'precise_min_weighted_cover'

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #
    for row, col in edges:
        model += (lpRows[row][0]+lpCols[col][0] >=
                  lpEdges[(row, col)]), f'edge_{row}_{col}'
        
    adjustment_ = 1

    model += (lpSum(lpEdges) >= (1-epsilon*adjustment_) * len(edges)), f'sensitivity'

    return model
# ============================================================================ #
#                     LP MODEL - MAXIMIZE DESIRED CELL                         #
# ============================================================================ #


def max_model(rows_data, cols_data, edges, epsilon=0.3):
    """
    In this model, we try to maximize the selection of desired cell directly
    without using Konig's Theorem.
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * edges: list of tuple of the coordination (row,col) of edges/undesired cell of the matrix.
    * epsilon: percentage of accepted undesired cell over accepted desired cell
    """
    # TODO: Implement the third model

    # ------------------------------------------------------------------------ #
    # Model with maximization
    # ------------------------------------------------------------------------ #
    model = LpProblem(name='maximize_desired_cell', sense=LpMaximize)

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #
    lpRows = {row: (LpVariable(f'row_{row}', cat='Integer',
                    lowBound=0, upBound=1), degree) for row, degree in rows_data}
    lpCols = {col: (LpVariable(f'col_{col}', cat='Integer',
                               lowBound=0, upBound=1), degree) for col, degree in cols_data}
    lpCells = {}
    for row, _ in rows_data:
        for col, _ in cols_data:
            if (row, col) in edges:
                lpCells[(row, col)] = (LpVariable(
                    f'cell_{row}_{col}', cat='Integer', lowBound=0, upBound=1), 0)
            else:
                lpCells[(row, col)] = (LpVariable(  
                    f'cell_{row}_{col}', cat='Integer', lowBound=0, upBound=1), 1)

    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #
    model += lpSum([cellValue*lpvar for lpvar,
                   cellValue in lpCells.values()]), 'maximize_desired_cell'
    #model += lpSum([lpvar for lpvar, _ in lpRows.values()]+[lpvar for lpvar, _ in lpCols.values()]), 'max_vertices'

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

    for row, col in lpCells:
        model += (lpRows[row][0] >= lpCells[(row, col)][0]), f'cell_{row}_{col}_1'
        model += (lpCols[col][0] >= lpCells[(row, col)][0]), f'cell_{row}_{col}_2'
        model += (lpRows[row][0]+lpCols[col][0] -1 <= lpCells[(row, col)][0]), f'cell_{row}_{col}_3'

    model += (lpSum([(1-cellValue)*lpvar for lpvar, cellValue in lpCells.values()]) <= epsilon *
              lpSum([lpvar for lpvar, _ in lpCells.values()])), f'err_rate'

    return model


# ============================================================================ #
#                         LP MODEL - KNAPSACK MODEL                            #
# ============================================================================ #

def knapsack_model(rows_data, cols_data, edges, epsilon=0.3):
    """
    In this model, we use knapsack model
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * epsilon: percentage of accepted undesired cell over accepted desired cell
    """
    # TODO: Implement the model

    # ------------------------------------------------------------------------ #
    # Model with minimization
    # ------------------------------------------------------------------------ #
    model = LpProblem(name='knapsack_problem', sense=LpMinimize)

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #

    lpRows = [(LpVariable(f'row_{row}', cat='Integer',
                    lowBound=0, upBound=1), degree) for row, degree in rows_data]
    lpCols = [(LpVariable(f'col_{col}', cat='Integer',
                    lowBound=0, upBound=1), degree) for col, degree in cols_data]

    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #
    model += lpSum([degree*lpvar for lpvar, degree in lpRows] +
                   [degree*lpvar for lpvar, degree in lpCols]), 'knapsack'

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

    nb_edges = sum([(len(cols_data)-degree) for _, degree in rows_data])

    model += (lpSum([(len(cols_data)-degree)*lpvar for lpvar, degree in lpRows]) + 
              lpSum([(len(rows_data)-degree)*lpvar for lpvar, degree in lpCols])  >= (1-epsilon) * nb_edges), f'sensitivity'

    return model

# ============================================================================ #
#                                    SOLVING                                   #
# ============================================================================ #


def solve(path_to_data, model, epsilon=0.2):
    """
    Function to solve the maximum biclique problem, this function reads the data,
    create a LP model using the data and return a list of rows and a list of 
    columns as a result. 
    ARGUMENTS:
    ----------
    * path_to_data: the path to the csv file.
    * model: the model to be use.
    * epsilon: percentage of errors (accepted edges) in the final result submatrix
    """
    

    rows, cols, edges, row_names, col_names = get_data(path_to_data)
    df = pd.read_csv(path_to_data, header=0 ,index_col=0 )
    model_name = model
    if model == 'min_model':
        model = min_model(rows, cols, edges)
    elif model == 'min_model_rc':
        model = min_model_rc(rows, cols, edges, epsilon)
    elif model == 'max_model':
        model = max_model(rows, cols, edges, epsilon)
    elif model == 'knapsack_model':
        model = knapsack_model(rows, cols, edges, epsilon)
    #create the model using one of the previously implemented models

    #solve the model using GUROBI_CMD. it is possible for the solver to take a long time
    #the time limit is set to 2 hours. The solver will be automatically stop after 2h.
    model.solve(GUROBI_CMD(msg=True, timeLimit= 7200),)
    #model.solve(PULP_CBC_CMD(msg=True, timeLimit= 7200),)

    #read the result from the solver
    rows_res = []
    cols_res = []
    if model_name == 'max_model':
        for var in model.variables():
            if var.varValue == 1:
                if var.name[:3] == "row":
                    rows_res = rows_res + [var.name[4:]]
                elif var.name[:3] == "col":
                    cols_res = cols_res + [var.name[4:]]
    else:
        for var in model.variables():
            if var.varValue == 0:
                if var.name[:3] == "row":
                    rows_res = rows_res + [var.name[4:]]
                elif var.name[:3] == "col":
                    cols_res = cols_res + [var.name[4:]]

    # print_log_output(model)

    print(rows_res,cols_res)
    print("Cardinality: ", len(rows_res), "+", len(cols_res), "=", len(rows_res) + len(cols_res))
    
    nb_0 = df.iloc[[int(r) for r in rows_res],[int(c) for c in cols_res]].size- df.iloc[[int(r) for r in rows_res],[int(c) for c in cols_res]].sum().sum()

    print("epsilon = ",epsilon)
    print("sparsity", nb_0/(len(rows_res) * len(cols_res)))
    print("density", 1-(nb_0/(len(rows_res) * len(cols_res))))

    return rows_res, cols_res

def get_data(path:str):

    rows_data = []
    cols_data = []
    edges = []

    df = pd.read_csv(path, header=0 ,index_col=0 )

    df = (df-1)*(-1)

    row_degrees = df.sum(axis=1)
    row_names = row_degrees.index
    rows_data = list(zip(range(len(row_degrees)), row_degrees))

    col_degrees = df.sum(axis=0)
    col_names = col_degrees.index
    cols_data = list(zip(range(len(col_degrees)), col_degrees))

    df=df.reset_index(drop=True)
    df = df.T.reset_index(drop = True).T
    edges = list(df[df == 0].stack().index)

    return rows_data, cols_data, edges , row_names, col_names

def print_log_output(prob):
    """Print the log output and problem solutions.
    ARGUMENTS:
    ----------
    * prob: an solved LP model (pulp.LpProblem)
    """
    print()
    print('-' * 40)
    print('Stats')
    print('-' * 40)
    print()
    print(f'Number variables: {prob.numVariables()}')
    print(f'Number constraints: {prob.numConstraints()}')
    print()
    print('Time:')
    print(f'- (real) {prob.solutionTime}')
    print(f'- (CPU) {prob.solutionCpuTime}')
    print()

    print(f'Solve status: {LpStatus[prob.status]}')
    print(f'Objective value: {prob.objective.value()}')

    print()
    print('-' * 40)
    print("Variables' values")
    print('-' * 40)
    print()
    # for v in prob.variables():
        # print(v.name, v.varValue)


def parse_arguments():
    """Parse the input arguments and retrieve the choosen resolution method and
    the instance that must be solve."""
    argparser = ArgumentParser()

    argparser.add_argument(
        '--filepath', dest='filepath', required=True, default='',
        help='Select the data',
    )

    argparser.add_argument(
        '--model', dest='model', required=False, default='min_weighted',
        help='Select the model to use',
    )

    argparser.add_argument(
        '--epsilon', dest='epsilon', required=False, default=0.1, type=float,
        help='Select the error rate value',
    )

    arg = argparser.parse_args()

    if arg.model not in ['min_model', 'min_model_rc', 'max_model', 'knapsack_model']:
        argparser.print_help()
        sys.exit(1)

    return (arg.filepath, arg.model, arg.epsilon)


if __name__ == '__main__':

    # Read the arguments
    file_path, selected_model, epsilon = parse_arguments()

    solve(file_path,selected_model,epsilon)



