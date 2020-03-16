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

    
    def __init__(self):
        """Lattice variables"""
        self.mu = 0.90
        nt = 10
        nx = 3
        ny = 3
        nz = 4
        #lat_size = [nt, nx, ny, nz]
        self.lat_size = [nx, ny]
        self.dim = len(self.lat_size)
        conf_dof = 2
        self.link_size = np.concatenate(([len(self.lat_size)], self.lat_size))
        self.conf_size = np.concatenate(([conf_dof], [len(self.lat_size)], self.lat_size))
        print(self.conf_size)
        self.lat_links = np.zeros(shape=self.conf_size, dtype=int)
        self.l_links = np.zeros(shape=self.link_size, dtype=int)
        self.k_links = np.zeros(shape=self.link_size, dtype=int)

        self.f = np.zeros(shape=self.lat_size, dtype=int)
        #W = np.fromfile("weights.dat", sep=",")
        self.W = np.ones(10000)
        
        #self.df = 0.
        #self.df0 = 0.
        
        #change in f to be kept track of
        #changes[0] = df0
        #changes[1] = df
        #changes[2] = k_change
        self.changes = np.zeros(3,dtype=int)
        
        """
        lat_size_gpu = cuda.to_device(lat_size)
        dim_gpu = cuda.to_device(dim)
        conf_size_gpu = cuda.to_device(conf_size)
        lat_links_gpu = cuda.to_device(lat_links)
        f_gpu = cuda.to_device(f)
        w_gpu = cuda.to_device(w)
        """
        

        """Worm variables"""
        self.head = np.zeros(self.dim, dtype = int)
        """Does python create a reference or copy?"""
        """Update each time the head is modified!!!"""
        self.tail = self.head.copy()
        self.worm_link_coord = np.zeros(self.dim+1, dtype = int)
        self.worm_link_coord[1:] = self.head.copy()
        self.prop_head = self.head.copy()
        
        self.k_delta = np.random.randint(0,1)*2 - 1
        #self.k_change = 3
        
        self.move_set = np.array([
                    [1,0],
                    [0,1],
                    [-1,0],
                    [0,-1],
        ])
        self.n_moves = len(self.move_set)

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
        trajectory = []


    """Lattice functions"""
    #@jit(nopython=True)
    def read_lat(self, l_links, k_links, read_path):
        lat_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)
        l_links = lat_links[0]
        k_links = lat_links[1]
        #l_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)

    #@jit(nopython=True)
    def save_lat(self, links, save_path):
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
    """transforms a lattice index to a valid value. Needed for links in the negative direction sign=1"""
    def transform_link_index_worm(self, link_coord, sign, move_set, lat_size, dim):
    #def get_link_index_worm(lat_coord, dir_i, sign):    
        #links are save unambigiously in an array
        #but in reality one link belongs to two different points and all 2*d directions
        #get the exact index in order to access a specific link value 

        move_i = link_coord[0] + sign*dim

        print(f"possibly invalid link index: {link_coord}")

        """what is faster?"""
        if sign == 1:
            link_coord[1:] += move_set[move_i]
        #link_coord[1:] += sign * move_set[move_i]
        #link_coord[1:] = link_coord[1:] + (sign * move_set[move_i])

        #impose periodic boundary conditions
        self.per_lat_coord(link_coord[1:], lat_size, dim)

        #link_index = [dir_i] + new_lat_coord
        print(f"valid link index: {link_coord}")
        #return link_index

    #@jit(nopython=True)
    def per_lat_coord(self, lat_coord, lat_size, dim):
        #impose periodic boundary conditions

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

        links[tuple(link_coord)] += value
        print(f"updating links[{link_coord}] += {value}")

    #@jit(nopython=True)
    """
    Calculate the aceptance probability for a worm move
    lat_size is the size of the lattice as an array
    dim is the number of dimensions
    mu is the chemical potential
    head is the coordinate vector of the worms head
    tail is the coordinate vectore of the worms tail
    dir_i is the dimension of the proposed move
    sign is the orientation of htat move: 0=positive, 1=negative
    move_set is a array which has saved translation vectors, moving is then simply a_trans += move_set[move_i]
    value is the random delta
    W the physical weights as one long vector
    f as an array of lattice size
    """
    
    #def get_prob_k(self, l_links, k_links, mu, head, prop_head, tail, worm_link_coord, k_change, W, f, df):
    def get_prob_k(self, l_links, k_links, mu, head, prop_head, tail, worm_link_coord, changes, W, f):
        p = 1.0
        k_change = changes[2]
        df = changes[1]
        #print(f"dir_i {dir_i}, sign {sign}")

        #link_index = get_link_index_worm(head, dir_i, sign, move_set, lat_size, dim)

        
        #print(f"link_index {link_index}")

        #just some values needed for acceptance p
        k_old_link = k_links[tuple(worm_link_coord)]
        k_proposed_link = k_links[tuple(worm_link_coord)] + k_change
        #k_proposed_link = k_links[tuple(worm_link_coord)] + changes[2]
        l_old_link = l_links[tuple(worm_link_coord)]

        print(f"k_old {k_old_link}")
        print(f"k_change {k_change}")
        #print(f"k_change {changes[2]}")
        print(f"k_prop {k_proposed_link}")
        #print(f"l_old {l_old_link}")

        f_old = f[tuple(head)]
        f_prop_head = f[tuple(prop_head)]

        #print(f"f_old {f_old}")
        #print(f"f_prop_head {f_prop_head}")

        #different factor in acceptance probability for changing modulus of k link
        #accounts for factorial coefficient
        if abs(k_proposed_link) > abs(k_old_link):
            p = p/(abs(k_proposed_link) + l_old_link)
        else:
            p = p*(abs(k_old_link) + l_old_link)

        """???"""
        #calculate proposed change of f[head] : df
        #df = int( 2*(abs(k_proposed_link) - abs(k_old_link)) )
        
        if k_change == -1:
            print(f"k change {-1}")
            df = 2
        elif k_change == 1:
            print(f"k change {1}")
            df = -2
        

        """referenced before assignment"""
        #print(f"prob df {df}")
        if not self.check_head_tail(head, tail):
            #worm has already started

            #multiply p with W[f_prop/2]
            """divide by 2"""
            p *= W[int((f_old + df)/2)]/W[int(f_prop_head/2)]

            #dont updafte f to keep it even
            #but save df (outside of function in main part!)
        else:
            #worm has not yet started
            #multiply p with W[f[head + move]]
            p *= 1./W[int(f_prop_head/2)]

        # if direction is timelike (0) multiply by exp(-change*mu)
        if worm_link_coord[0] == 0:
            #p *= np.exp((1. - 2.*sign)*mu*value)
            p *= np.exp(float(k_change)*mu)

        return p

    #@jit(nopython=True)
    def update_f_df(self, f, lat_coord, df):
        print(f"updating f[{lat_coord}] += {int(df)}")
        f[tuple(lat_coord)] += f[tuple(lat_coord)] + int(df)

    #@jit(nopython=True)
    def print_env_worm(self, lat_size, head, tail):
        """
        l_links_host = l_links.copy_to_host()
        k_links_host = k_links.copy_to_host()
        head_host = head.copy_to_host()
        tail_host = tail.copy_to_host()

        image = np.zeros(l_links_host[0].shape)
        #image = np.zeros(k_links_host[0].shape)
        head_val = 10
        tail_val = -10
        #print(f"printing tail: {tail}")
        #print(f"printing head: {head}")
        image[tuple(head_host)] = head_val
        image[tuple(tail_host)] = tail_val
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

    #goal: do a parallel sweep through all given lattice sites
    def sweep(self, lat_size, dim, links, sites):
        n_sites = int(np.prod(lat_size))

        #save the proposed mcmc modifications in an array
        #rand_mods = np.zeros(conf_size)

        """
        implement checkerboard
        black or white sweep?
        """
        for site in sites:
            """#call parallel function mc_site"""


    def draw_site(self, dim):
        #choose random modification at random link dimension

        mod_dim = np.random.randint(dim)
        mod_value = int(2*np.random.randint(0,1) - 1)

    def mc_site(self, lat_coord, W, f):

        mod_dim = 0
        mod_value = 0
        df = 0
        df_target = 0
        draw_site(dim) 
        target_lat_coord = lat_coord + move_set[mod_dim]
        """find better method"""
        link_coord = np.concatenate([mod_dim], lat_coord)
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
        dir_i = np.random.randint(low=0,high=dim)
        print(f"proposing dir_i {dir_i}")
        sign = np.random.randint(low=0,high=2)
        print(f"proposing sign {sign}")
        move_i = dir_i + sign*dim
        
        worm_link_coord[0] = dir_i
        worm_link_coord[1:] = head.copy()
        #check whether the link index has to be changed in order to be accessed
        self.transform_link_index_worm(worm_link_coord, sign, move_set, lat_size, dim)
        
        #prop_head = head + move_set[move_i]
        prop_head += move_set[move_i]
        self.per_lat_coord(prop_head, lat_size, dim)
        
        #k_change = int((1-2*sign) * k_delta)
        #changes[2] = int((1-2*sign) * k_delta)
        changes[2] = (1-2*sign) * k_delta
        changes[1] = 2*changes[2]
        print(f"proposing move {move_i}")
        print(f"proposing move {move_set[move_i]}")
        print(f"prop_head {prop_head}")
        #print(f"k change {k_change}")
        print(f"k change {changes[2]}")
        #print(f"move_i {rand_i}")
        #return dir_i

    #@jit(nopython=True)
    #def extend(head, dir_i, sign, move_set, lat_size, dim):
    def extend(self, head, worm_link_coord, move_i, move_set, lat_size, dim):

        print(f"move_set {move_set}")
        print(f"extend old head {head}")
        head += move_set[move_i]
        print(f"extend move {move_set[move_i]}")
        print(f"extend new head {head}")

        self.per_lat_coord(head, lat_size, dim)
        #prop_head = head.copy()
        #print(f"per head {head}")

    def set_head(self, head, prop_head):
        print(f"head {head}")
        print(f"prop_head {prop_head}")
        head[:] = prop_head[:]

    #@jit(nopython=True)             
    def check_head_tail(self, check_head, check_tail):
        return np.all(check_head == check_tail)

    #@jit(nopython=True)
    def reset_worm(self, tail, head, worm_link_coord, lat_size, dim):
        for d in range(dim):
            tail[d] = np.random.randint(0,lat_size[d])
        head[:] = tail[:]
        worm_link_coord[1:] = head[:]
        worm_link_coord[0] = 0
        k_delta = np.random.randint(0,1)*2 - 1
        #print(f"Resetting worm to tail: {tail}, head: {head}\n")