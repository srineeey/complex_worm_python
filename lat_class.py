import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

import os
import glob
import time
import IPython.display


class lattice_sim:
    """
    Lattice simulation class object
    For MCMC simulations of the dualized kl lattice formulation
    the k flux constraint requires a specialized sampling method
    using the worm algorithm
    """

    
    def __init__(self):
        """Lattice variables"""
        
        ##chemical potential mu
        self.mu = 0.90
        

        
        ##size of the lattice
        nt = 10
        nx = 3
        ny = 3
        nz = 4
        #lat_size = [nt, nx, ny, nz]
        self.lat_size = [nx, ny]
        self.dim = len(self.lat_size)


        ##size of one link field on the lattice
        self.link_size = np.concatenate(([len(self.lat_size)], self.lat_size))
        
        ##size of one link field on the lattice
        conf_dof = 2
        self.conf_size = np.concatenate(([conf_dof], [len(self.lat_size)], self.lat_size))
        print(self.conf_size)
        
        ##initialize with zeros
        self.lat_links = np.zeros(shape=self.conf_size, dtype=int)
        self.l_links = np.zeros(shape=self.link_size, dtype=int)
        self.k_links = np.zeros(shape=self.link_size, dtype=int)

        ##an array saving the current status fo the f function
        self.f = np.zeros(shape=self.lat_size, dtype=int)
        
        ##the weights whic hare needed for calculating probabilities
        #W = np.fromfile("weights.dat", sep=",")
        self.W = np.ones(10000)
        
        
        """Worm variables"""
        #self.df = 0.
        #self.df0 = 0.
        
        ##change to be kept track of
        ##changes[0] = df0
        ##changes[1] = df
        ##changes[2] = k_change
        self.changes = np.zeros(3,dtype=int)
        
        self.k_delta = np.random.randint(0,2)*2 - 1
        ##the change of k values which is picked randomly from [-1,1]
        """self.k_delta = np.random.randint(0,1)*2 - 1"""
        self.k_change = 3
        
        """
        lat_size_gpu = cuda.to_device(lat_size)
        dim_gpu = cuda.to_device(dim)
        conf_size_gpu = cuda.to_device(conf_size)
        lat_links_gpu = cuda.to_device(lat_links)
        f_gpu = cuda.to_device(f)
        w_gpu = cuda.to_device(w)
        """
        
        self.head = np.zeros(self.dim, dtype = int)
        """Does python create a reference or copy?"""
        """Update each time the head is modified!!!"""
        self.tail = self.head.copy()
        
        ##the link at which the worm is pointing atm
        self.worm_link_coord = np.zeros(self.dim+1, dtype = int)
        self.worm_link_coord[1:] = self.head.copy()
        
        ##the proposed head position after the move would be executed
        self.prop_head = self.head.copy()
        
        
        ##the viable moves the worm can make
        self.move_set = np.array([
                    [1,0],
                    [0,1],
                    [-1,0],
                    [0,-1],
        ])
        ##number of moves to pick from
        self.n_moves = len(self.move_set)
        
        ##variables describing the move
        ##dir_i = dimension index
        ##sign = positive (=0) or negative direction (=1)
        ##move_i = move index from moveset
        self.sign = 0
        self.dir_i = 0
        self.dir_i_sign = np.zeros(2,dtype=int)
        self.move_i = 0
        


        #move variables to GPU
        """
        head_gpu = cuda.to_device(head)
        tail_gpu = cuda.to_device(tail)
        move_set_gpu = cuda.to_device(move_set)
        n_moves_gpu = cuda.to_device(n_moves)
        """
        ##keep track fo the worms evolution
        trajectory = []
         
        print(f"Setting worm to tail: {self.tail}, head: {self.head}\n")
        print(f"with k_delta: {self.k_delta}\n")

    """Lattice functions"""
    #@jit(nopython=True)
    def read_lat(self, l_links, k_links, read_path):
        """
        read the lattice conf from a file
        """
        lat_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)
        l_links = lat_links[0]
        k_links = lat_links[1]
        #l_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)

    #@jit(nopython=True)
    def save_lat(self, links, save_path):
        """
        save the lattice conf to a file
        """
        links.flatten().tofile(save_path, sep=" ")

    #@jit(nopython=True)
    """DEPRECATED???"""
    """change to lattice index tuple (more practical!)"""
    def get_link_index_worm(self, lat_coord, dir_i, sign, move_set, lat_size, dim):
    #def get_link_index_worm(lat_coord, dir_i, sign):    
        #links are save unambigiously in an array
        #but in reality one link belongs to two different points and all 2*d directions
        #get the exact index in order to access a specific link value 

        move_i = dir_i + sign*dim

        print(f"possibly invalid link index: {dir_i}, {lat_coord}")

        new_lat_coord = lat_coord.copy()
        if sign == 1:
            print(f"negative direction, sign =  {sign}")
            new_lat_coord += move_set[move_i]
        self.per_lat_coord(new_lat_coord, lat_size, dim)

        link_index = np.concatenate(([dir_i], new_lat_coord))
        #link_index = [dir_i] + new_lat_coord
        print(f"valid link index: {link_index}")
        return link_index

    #@jit(nopython=True)
    def transform_link_index_worm(self, link_coord, sign, move_set, lat_size, dim):
        """
        transforms a lattice index to a valid value.
        Needed for links in the negative direction sign=1
        because of the special way configurations are saved
        not all links from a particular site can be accessed by index [lattice_coord, link_index]
        """

        move_i = link_coord[0] + sign*dim

        print(f"possibly invalid link index: {link_coord}")

        """what is faster?"""
        if sign == 1:
            link_coord[1:] += move_set[move_i]
        #link_coord[1:] += sign * move_set[move_i]
        #link_coord[1:] = link_coord[1:] + (sign * move_set[move_i])

        ##impose periodic boundary conditions
        self.per_lat_coord(link_coord[1:], lat_size, dim)

        #link_index = [dir_i] + new_lat_coord
        print(f"valid link index: {link_coord}")
        #return link_index

    #@jit(nopython=True)
    def per_lat_coord(self, lat_coord, lat_size, dim):
        """
        impose periodic boundary conditions on the lattice
        lat_coord that lie outside of lat size
        have to be glued to the other lattice end
        """
        
        #for d in np.arange(start=0, stop=dim, step=1, dtype=int):
        for d in range(0, int(dim), 1):
        #for d in range(0, len(lat_size), 1):
            #if lat_coord is outside of the possible index domain

            if lat_coord[d] >= lat_size[d]:
                lat_coord[d] = (lat_coord[d] - lat_size[d])
            elif lat_coord[d] < 0:
                lat_coord[d] = (lat_coord[d] + lat_size[d])

        #return lat_coord

    """
    #@jit(nopython=True)
    #def update_link_worm(links, lat_size, dim, lat_coord, dir_i, sign, move_set, value):
    def update_link_worm(links, lat_coord, dir_i, sign, value):

        link_index = get_link_index_worm(lat_coord, dir_i, sign, move_set, lat_size, dim)

        #print(f"Updating link degree of freedom {link_dof} at position {tuple(link_coord)} and direction {move_i}")

        #link_coord = [move_i] + link_coord
        #print(link_coord)

        #sign = 1 positive direction
        #links[tuple(link_index)] = (2*sign -1) * value

        #sign = 0 positive direction
        links[tuple(link_index)] += (1-2*sign) * value
        print(f"updating links[{link_index}] += {(1-2*sign) * value}")
    """
    #@jit(nopython=True)
    #def update_link_worm(links, lat_size, dim, lat_coord, dir_i, sign, move_set, value):
    def update_link(self, links, link_coord, value):
        """
        update links[links_coord] by addig value 
        """

        links[tuple(link_coord)] += value
        print(f"updating links[{link_coord}] += {value}")

    #@jit(nopython=True)
    #def get_prob_k(self, l_links, k_links, mu, head, prop_head, tail, worm_link_coord, k_change, W, f, df):
    def get_prob_k(self, l_links, k_links, mu, head, prop_head, tail, worm_link_coord, changes, W, f):
        """
        Calculate the aceptance probability for a worm move
        lat_size is the size of the lattice as an array
        dim is the number of dimensions
        mu is the chemical potential
        head is the coordinate vector of the worms head
        tail is the coordinate vectore of the worms tail
        dir_i is the dimension of the proposed move
        sign is the orientation of that move: 0=positive, 1=negative
        move_set is an array which has saved translation vectors, moving is then simply a_trans += move_set[move_i]
        value is the random delta
        W the physical weights as one long vector
        f as an array of lattice size
        """
        
        p = 1.0
        ##the proposed changes of that particular move
        ##needed for calculating the acceptance probability
        k_change = changes[2]
        df = changes[1]
        #print(f"dir_i {dir_i}, sign {sign}")

        #link_index = get_link_index_worm(head, dir_i, sign, move_set, lat_size, dim)
        #print(f"link_index {link_index}")

        ##just some values needed for acceptance probability
        k_old_link = k_links[tuple(worm_link_coord)]
        k_proposed_link = k_links[tuple(worm_link_coord)] + k_change
        #k_proposed_link = k_links[tuple(worm_link_coord)] + changes[2]
        l_old_link = l_links[tuple(worm_link_coord)]

        print(f"k_old {k_old_link}")
        print(f"k_change {k_change}")
        #print(f"df {df}")
        print(f"k_prop {k_proposed_link}")
        #print(f"l_old {l_old_link}")

        f_old = f[tuple(head)]
        f_prop_head = f[tuple(prop_head)]

        #print(f"f_old {f_old}")
        #print(f"f_prop_head {f_prop_head}")

        #different factor in acceptance probability for changing modulus of k link
        #accounts for factorial coefficient in formula
        if abs(k_proposed_link) > abs(k_old_link):
            p = p/(abs(k_proposed_link) + l_old_link)
        else:
            p = p*(abs(k_old_link) + l_old_link)

        ##calculate proposed change of f[head] : df
        ##df = int( 2*(abs(k_proposed_link) - abs(k_old_link)) )
                
        #df = -2*k_change
        #df = 2*k_change
        #df = 2*abs(k_change)
        
        
        """referenced before assignment"""
        print(f"prob df {df}")
        """Find a more efficient way"""
        if not self.check_head_tail(head, tail):
            #worm has already started

            #multiply p with W[f_prop/2]
            """divide by 2"""
            p *= W[int((f_old + df)/2)]/W[int(f_prop_head/2)]

            ##dont update f to keep it even
            ##but saved df (outside of function in changes[1]!)
        else:
            ##worm has not yet started
            ##multiply p with W[f[head + move]]
            p *= 1./W[int(f_prop_head/2)]

        ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
        if worm_link_coord[0] == 0:
            #p *= np.exp((1. - 2.*sign)*mu*value)
            p *= np.exp(float(k_change)*mu)

        return p

    #@jit(nopython=True)
    def update_f_df(self, f, lat_coord, df):
        """
        update f function array with df value
        """
        print(f"updating f[{lat_coord}] += {int(df)}")
        f[tuple(lat_coord)] += f[tuple(lat_coord)] + int(df)

    #@jit(nopython=True)
    def print_env_worm(self, lat_size, head, tail):
        """
        simple function for plotting worms head and tail
        for visualization
        """
        
        """
        l_links_host = l_links.copy_to_host()
        k_links_host = k_links.copy_to_host()
        head_host = head.copy_to_host()
        tail_host = tail.copy_to_host()
        """
        image = np.zeros(lat_size)
        #image = np.zeros(k_links_host[0].shape)
        head_val = 10
        tail_val = -10
        #print(f"printing tail: {tail}")
        #print(f"printing head: {head}")
        image[tuple(head)] = head_val
        image[tuple(tail)] = tail_val

        #print(image)
        plt.imshow(image)
        plt.show()
        
    
    
    
    """Monte Carlo functions"""

    
    def sweep(self, lat_size, dim, links, sites):
        """
        goal: do a parallel sweep through all given lattice sites (sites)
        """
        
        n_sites = int(np.prod(lat_size))
        

        #save the proposed mcmc modifications in an array
        #rand_mods = np.zeros(conf_size)

        """
        implement checkerboard
        black or white sweep?
        """
        for site in sites:
            """call parallel function mc_site"""


    def draw_site(self, mod_dim, mod_value, dim) :
        """
        choose random modification at random link dimension
        """
        """find better method"""
        mod_dim = np.random.randint(dim)
        mod_value = int(2*np.random.randint(0,1) - 1)

    def mc_site(self, lat_coord, W, f):
        
        """
        MCMC for one single lattice site
        """

        mod_dim = 0
        mod_value = 0
        df = 0
        df_target = 0
        
        ##draw link and link_modification at random
        draw_site(mod_dim, mod_value, dim) 
        
        target_lat_coord = lat_coord + move_set[mod_dim]
        
        """find better method"""
        link_coord = np.concatenate([mod_dim], lat_coord)
        
        ##get acceptance probability
        p = get_prob_l(l_links, k_links, lat_coord, link_coord, target_lat_coord, mod_value, W, f)

        if np.random.uniform < p:
            #accept
            update_link(l_links, link_coord, mod_value)


    #modify to get probability for l change
   #def get_prob_k(l_links, k_links, mu, head, prop_head, tail, sign, move_set, worm_link_coord, k_change, lat_size, dim, W, f):
    def get_prob_l(self, l_links, k_links, mu, link_coord, target_lat_coord, mod_value, W, f):
        p = 1.0

        #change in f function at lat_coord
        """save this change globally?"""
        df = 2 * mod_value

        #change in f function at link_target_coord
        """save this change globally?"""
        df_target = -df

        f_old = f[tuple(link_coord[1:])]
        f_target = f[tuple(target_lat_coord)]

        l_old_link = l_links[tuple(link_coord[1:])]
        l_proposed_link = l_links[tuple(link_coord[1:])] + mod_value
        k_old_link = k_links[tuple(link_coord[1:])]
        print(f"k_old {l_old_link}")
        print(f"k_prop {l_proposed_link}")


        #different factor in acceptance probability for changing modulus of l link
        #accounts for factorial coefficient
        if abs(l_proposed_link) > abs(l_old_link):
            p = p/((abs(k_old_link) + l_proposed_link) * l_proposed_link)
        else:
            p = p*(abs(k_old_link) + l_old_link) * l_old_link

        p *= W[int((f_old + df)/2)]/W[int(f_prop_head/2)]

        return p


    """ SEE ABOVE DEFINITION        
    #@jit(nopython=True)
    def update_link(links, link_coord, sign, value):

        links[tuple(link_coord)] += (1-2*sign) * value
        print(f"updating links[{link_coord}] += {(1-2*sign) * value}")

    """

    """NOT NEEDED
    #function that draws according to the Metropolis algorithm given two probabilities
    def metropolis_p(p_x_new, p_x_old):

        if p_x_new > p_x_old:
            return True
        else:
            if np.random.randint() < (p_x_new/p_x_old):
                return True
            else:
                return False"""
    
    
    


    """Worm functions"""
    

    #@jit(nopython=True)
    #def propose_move(self, dir_i, sign, move_i, move_set, head, prop_head, worm_link_coord, k_delta, k_change, lat_size, dim):
    def propose_move(self, dir_i, sign, move_i, move_set, head, prop_head, worm_link_coord, k_delta, changes, lat_size, dim):
        """
        Draws a random move from the moveset
        as well as a random value for changing the link value
        and prepares some values needed lateron for the probabalistic sampling step
        """
        
        ##draw a ranom move dimension (dir_i) and orientation (sign)
        dir_i = np.random.randint(low=0,high=dim)
        print(f"proposing dir_i {dir_i}")
        sign = np.random.randint(low=0,high=2)
        print(f"proposing sign {sign}")
        
        ##calculate the corresponding move_index for the move_set
        move_i = dir_i + sign*dim
        print(f"proposing move_i {move_i}")
        
        ##the link value which is to be modified if the move is accepted
        worm_link_coord[0] = dir_i
        worm_link_coord[1:] = head.copy()
        ##check whether the link index has to be changed in order to be accessed
        self.transform_link_index_worm(worm_link_coord, sign, move_set, lat_size, dim)
        
        ##the new proposed head if the move is accepted
        #prop_head = head + move_set[move_i]
        prop_head += move_set[move_i]
        ##check whether the new head has to be changed for periodic bc
        self.per_lat_coord(prop_head, lat_size, dim)
        
        
        ##change to be kept track of
        ##changes[0] = df0
        ##changes[1] = df
        ##changes[2] = k_change
        changes[2] = (1-2*sign) * k_delta
        self.k_change = changes[2]
        changes[1] = 2*changes[2]
        print(f"proposing move {move_i}")
        print(f"proposing move {move_set[move_i]}")
        print(f"prop_head {prop_head}")
        #print(f"k change {k_change}")
        print(f"df {changes[1]}")
        print(f"k change {changes[2]}")
        #print(f"move_i {rand_i}")
        #return dir_i

    #@jit(nopython=True)
    #def extend(head, dir_i, sign, move_set, lat_size, dim):
    def extend(self, head, worm_link_coord, move_i, move_set, lat_size, dim):
        """
        extend the worms head via a move
        """

        print(f"move_set {move_set}")
        print(f"extend old head {head}")
        head += move_set[move_i]
        print(f"extend move {move_set[move_i]}")
        print(f"extend new head {head}")

        self.per_lat_coord(head, lat_size, dim)
        #prop_head = head.copy()
        #print(f"per head {head}")

    def set_head(self, head, prop_head):
        """
        set the worms head directly
        """
        print(f"head {head}")
        print(f"prop_head {prop_head}")
        head[:] = prop_head[:]

    #@jit(nopython=True)             
    def check_head_tail(self, check_head, check_tail):
        """
        check whether head and tail coincide
        """
        return np.all(check_head == check_tail)

    #@jit(nopython=True)
    def reset_worm(self, tail, head, worm_link_coord, k_delta, lat_size, dim):
        """
        reset the worm randomly
        choose the value of k_delta (fixed!)
        """
        for d in range(dim):
            tail[d] = np.random.randint(0,lat_size[d])
        head[:] = tail[:]
        worm_link_coord[1:] = head[:]
        worm_link_coord[0] = 0
        """k_delta = np.random.randint(0,1)*2 - 1"""
        self.k_delta = np.random.randint(0,1)*2 - 1
        print(f"Resetting worm to tail: {tail}, head: {head}\n")
        print(f"with k_delta: {self.k_delta}\n")