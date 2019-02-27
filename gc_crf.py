
'''
*
* crf
*
* Purpose: the module finds the minimum energy semi-connected path between 1st and last row
*
* Inputs: observations_matrix
          N = energy "deadzone" for small transitions
          T = threshold to allow large transitions (between obstacles)
          W_trans =
*
* Outputs:
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018 (framework originates from excellent https://crosleythomas.github.io/blog/)
*
'''

# from __future__ import absolute_import

import numpy as np

############################################
###   Viterbi algorithm implementation   ###
############################################

# Creates a rectangular matrix weighting the transition between cell xi to cell xj
# dim = matrix width/height
# N =  "deadzone" for small enough transitions
# T =  threshold to allow transition between obstacles
# Wb = transition cost weight (in relation to Wu - the unary energy)

def create_transition_matrix(dim, N, T, Wb):

    transition_mat = np.zeros((dim ,dim))
    for row in range(dim):
        for column in range (dim):
            transition_mat[column ,row] = max ((abs(column -row)) - N, 0)
            transition_mat[column, row] = min(transition_mat[column ,row], T)
            transition_mat[column, row] = transition_mat[column ,row] * Wb
        #print(transition_mat[row,:])

    #print(transition_mat)
    return transition_mat

# implement the Viterbi dynamic programming algorithm
# Go through the columns and compute the following Viterbi function: Wu*P + Wb*min(max((|y1-y2|-N,0),T)

def viterbi(observations_matrix, N, T, W_trans):

    # Init
    W_unary = 1
    observations_shape = np.shape(observations_matrix)
    stixel_dim = observations_shape[1]
    border_length = observations_shape[0]
    #print('stixel_cells_num = {}'.format(stixel_dim))
    new_energy_vec = np.zeros((stixel_dim,1))
    energy_vec = -np.log(observations_matrix[0 ,:] ) * W_unary
    #print(energy_vec)
    path_matrix = np.zeros((border_length,stixel_dim))
    #print(np.shape(path_matrix))
    transition_matrix = create_transition_matrix(stixel_dim ,N ,T, W_trans)

    # Go through each row
    for to_row in range(1,border_length):
        # find max probability cell in "to_row" ????
        from_row_max_prob = max(observations_matrix[to_row, :])
        for index, i in enumerate(observations_matrix[to_row, :]):
            if i == from_row_max_prob:
                max_cell_index = index
        #print('\nanalyze transitions to row {}, max cell = {}'.format(to_row, max_cell_index))

        # Go through each cell and find the lowest energy path to it
        for to_cell in range(stixel_dim):
            # Init lowest transition energy
            min_energy_cell_num = 0
            min_energy = energy_vec[0] + transition_matrix[0, to_cell]
            #print('{} -> {}: {}'.format(0, to_cell, int(energy_vec[0] + transition_matrix[0, to_cell])))
            # Go through all possible from_cells
            for from_cell in range(1,stixel_dim):
                #print('{} -> {}: {} + {}'.format(from_cell, to_cell, int(energy_vec[from_cell]), transition_matrix[from_cell,to_cell]))
                transition_energy = energy_vec[from_cell] + transition_matrix[from_cell, to_cell]
                if (transition_energy < min_energy):
                    min_energy = transition_energy
                    min_energy_cell_num = from_cell
                elif transition_energy == min_energy:
                    #print('found another trail')
                    if (min_energy_cell_num > 0) and (abs(from_cell - max_cell_index) < abs(min_energy_cell_num - max_cell_index)):
                        min_energy = transition_energy
                        min_energy_cell_num = from_cell
                        #print('* fix min to {} (closer to max prob cell)'.format(min_energy_cell_num))

            # Update new energy vec & path
            #print('min path to {} =  {} (energy {}) \n'.format(to_cell, min_energy_cell_num,int(min_energy)))
            #new_energy_vec[cell_num] = -np.log(observations_matrix[to_row, min_energy_cell_num]) * W_unary + min_energy
            #print('adding {},{} energy {}'.format(to_row,cell_num, -np.log(observations_matrix[to_row, cell_num])))
            new_energy_vec[to_cell] = -np.log(observations_matrix[to_row, to_cell]) * W_unary + min_energy
            path_matrix[to_row,to_cell] = int(min_energy_cell_num)

        # Save new energy vec to energy vec
        for index, val in enumerate(new_energy_vec):
            energy_vec[index] = val
        #print(energy_vec.T)


    # find the best path
    #print(path_matrix)
    #print('Find the best trail from end point to first one (max probability cell {}):'.format(max_cell_index))
    a = min(energy_vec)
    end_of_trail = -1
    for index,i in enumerate(energy_vec):
        if (i == a):
            if (end_of_trail==-1):
                # first min - update end cell
                end_of_trail = index
            elif (abs(index-max_cell_index) < abs(end_of_trail-max_cell_index)):
                # choose the index closest to the max-probability cell
                end_of_trail = index

    # Go through the path_matrix and create the (reverse) optimal path
    best_path_r = []
    best_path_r.append(end_of_trail)
    #print(end_of_trail)
    for row_num in range(border_length-1,0,-1):
        end_of_trail = int(path_matrix[row_num,end_of_trail])
        best_path_r.append(end_of_trail)
        #print(end_of_trail)

    # Reverse the list
    best_path = list(reversed(best_path_r))
    #print(best_path)

    return best_path

#############################################################
###   Visualizing predictions and creating output video   ###
#############################################################

def main(grid, N, T, W_trans):

    # Use CRF to find the best path
    best_path = viterbi(grid.T, N, T, W_trans)
    for index, path in enumerate(best_path):
        print('{}: {}'.format(index,path))


if __name__ == '__main__':

    # define a grid example
    rows = 50
    columns = 50

    grid = np.zeros((rows, columns)) + 1e-9

    '''
    grid[0, 2] = 0.7
    grid[1, 2] = 0.8
    grid[2, 3] = 0.8
    grid[3, 4] = 0.8
    '''

    grid[0,0] = 0.7
    grid[0,1] = 0.3
    grid[1,2] = 0.8
    grid[1,3] = 0.2
    grid[2,2] = 0.2
    grid[2,3] = 0.8
    grid[3,2] = 0.8
    grid[4,3] = 0.9
    grid[5,4] = 0.8
    grid[40,40] = 0.9

    main(grid.T, N=5, T=10, W_trans=5)







