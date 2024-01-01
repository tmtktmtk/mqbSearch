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
    # Model with minamization
    # ------------------------------------------------------------------------ #
    model = LpProblem()

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

    return model

# ============================================================================ #
#                LP MODEL - PRECISE MINIMUM WEIGHTED COVER                     #
# ============================================================================ #


def min_model_rc(rows_data, cols_data, edges, epsilon=0.3):
    """
    Implement the LP model for minimum weighted cover. 
    In this model, we introduce epsilon with the goal of alowing error
    in the result. 
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * edges: list of tuple of the coordination (row,col) of edges/undesired cell of the matrix.
    * epsilon: percentage of errors (accepted edges) in the final result submatrix
    """

    # ------------------------------------------------------------------------ #
    # Model with minamization
    # ------------------------------------------------------------------------ #
    model = LpProblem()

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #

    # Objective function
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

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
    model = LpProblem()

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

    return model


# ============================================================================ #
#                         LP MODEL - KNAPSACK MODEL                            #
# ============================================================================ #

def knapsack_model(rows_data, cols_data, edges, epsilon=0.3):
    """
    Implement the LP model for minimum weighted cover. 
    In this model, we introduce epsilon with the goal of alowing error
    in the result. 
    ARGUMENTS:
    ----------
    * rows_data: list of the tuples (row, degree) of rows the matrix.
    * cols_data: list of the tuples (col, degree) of columns the matrix.
    * edges: list of tuple of the coordination (row,col) of edges/undesired cell of the matrix.
    * epsilon: percentage of errors (accepted edges) in the final result submatrix
    """

    # ------------------------------------------------------------------------ #
    # Model with minamization
    # ------------------------------------------------------------------------ #
    model = LpProblem()

    # ------------------------------------------------------------------------ #
    # Variables
    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #

    # Objective function
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------------ #

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
    
    data = get_data(path_to_data)
    #create the model using one of the previously implemented models
    model = None

    #solve the model using GUROBI_CMD. it is possible for the solver to take a long time
    #the time limit is set to 2 hours. The solver will be automatically stop after 2h.
    # model.solve(GUROBI_CMD(msg=True, timeLimit= 7200),)
    model.solve(PULP_CBC_CMD(msg=True, timeLimit= 7200),)

    #read the result from the solver
    rows_res, cols_res = [],[]

    return rows_res, cols_res
  
def get_data(path:str):

    rows_data = []
    cols_data = []
    edges = []

    df = pd.read_csv(path, header=0 ,index_col=0 )

    rows = df.sum(axis=1)
    rows_data = list(zip(rows.index, rows))

    cols = df.sum(axis=0)
    cols_data = list(zip(cols.index, cols))

    edges = list(df[df == 0].stack().index)

    return rows_data, cols_data, edges

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
    for v in prob.variables():
        print(v.name, v.varValue)


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



