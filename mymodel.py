#####################################################################
# analytical geothermal doublet analyses TU/e
#####################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

open("parameters.txt", "r")
open("well.txt", "r")
# from sfepy.discrete.fem import Mesh, FEDomain, Field


## Define the model parameters
class Aquifer:

    def __init__(self):
        # aquifer characteristics
        with open('parameters.txt') as json_file:
            data = json.load(json_file)
            for aq in data['aquifer']:
                self.d_top = aq['d_top']        # depth top aquifer at production well
                self.labda = aq['labda']        # geothermal gradient
                self.H     = aq['H']        # thickness aquifer
                self.T_surface = aq['T_surface']
                self.porosity = aq['porosity']
                self.rho_f = aq['rho_f']
                self.mhu =  aq['viscosity']
                self.K =  aq['K']

class Well:

    def __init__(self, t0, tEnd):
        # well properties
        with open('parameters.txt') as json_file:
            data = json.load(json_file)
            for well in data['well']:
                self.r = well['r']  # well radius; assume radial distance for monitoring drawdown
                self.Q = well['Q']  # pumping rate from well (negative value = extraction)
                self.L = well['L']  # distance between injection well and production well
                self.Ti_inj = well['Ti_inj'] # initial temperature of injection well (reinjection temperature)
                self.epsilon = well['epsilon']
                self.D_in = 2*self.r

## Assemble doublet system
class DoubletGenerator:
    """Generates all properties for a doublet

    Args:

    """
    def __init__(self, aquifer, well):

        self.aquifer = aquifer
        self.well = well

        self.cp = 4183 # water heat capacity
        self.rhos = 2711 #density limestone
        self.labdas = 1.9 # thermal conductivity solid [W/mK]
        self.cps = 910 #heat capacity limestone [J/kg K]
        self.g = 9.81 # gravity constant
        self.time = 365*24*60*60 #1 year [s]
        self.mdot = self.well.Q * self.aquifer.rho_f
        self.lpipe = self.aquifer.d_top + 0.5 * self.aquifer.H
        self.rhotest = self.rho(20,1e5/1e6)
        # print(self.rhotest)

        self.Dx = self.well.L * 3  # domain of x
        self.Dy = - (2 * self.aquifer.d_top + self.aquifer.H)  # domain of y
        self.Nx = 24  # number of nodes by x
        self.Ny = 10  # number of nodes by y
        self.nNodes = self.Nx * self.Ny  # total number of nodes
        self.ne = (self.Nx - 1) * (self.Ny - 1)
        self.dx = self.Dx / self.Nx  # segment length of x
        self.dy = self.Dy / self.Ny  # segment length of y
        self.domain = np.array([self.dx, self.dy])
        self.x_grid, self.y_grid = self._make_grid()
        self.x_well, self.y_well = self._construct_well()
        self.nodes_grid = self._make_nodes_grid()
        self.coordinate_grid = self._make_coordinates_grid()

        self.P_pump = self._get_P_pump()
        self.T_aquifer = self._get_T(self.lpipe)
        self.P_aquifer = self._get_P(self.lpipe)
        self.P_wellbore = self._get_P_wb()
        self.T_wellbore = self.T_aquifer

        self.lpipe_divide = np.linspace(self.lpipe, 0, 200)
        self.q_heatloss_pipe = self._get_T_heatloss_pipe(self.well.D_in, self.lpipe_divide)
        self.P_HE = self._get_P_HE(self.well.D_in)
        self.T_HE = self._get_T_HE(self.well.D_in, self.lpipe_divide)
        self.Power_HE = self.mdot * self.cp * (self.T_HE - self.well.Ti_inj)

        self.P_grid = self._compute_P_grid()
        self.T_grid = self._compute_T_grid()

        print(self._get_P(900)/1e5)
        print(self._get_P(1100)/1e5)

    # def _get_gaussian_points
    def _compute_T_grid(self):
        T_grid = self._get_T(-self.y_grid)
        # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
        # P_grid[5][16] = self.P_wellbore
        # P_grid[4][16] = self.P_wellbore
        T_grid[5][8] = self.well.Ti_inj
        T_grid[4][8] = self.well.Ti_inj

        return T_grid

    def _compute_P_grid(self):
        P_grid = self._get_P(-self.y_grid)
        # P_grid[self.Ny/2][self.Nx/3] = self.P_wellbore
        P_grid[5][16] = self.P_wellbore
        P_grid[4][16] = self.P_wellbore
        P_grid[5][8] = self.P_wellbore
        P_grid[4][8] = self.P_wellbore

        return P_grid

    def _get_P_pump(self):
        P_pump = 0

        return P_pump

    def _get_P_HE(self, D_in):
        P_HE = self.P_wellbore - self._get_P(self.aquifer.d_top + 0.5 * self.aquifer.H) -\
        ( self._get_f( D_in) * self.aquifer.rho_f * self.get_v_avg( D_in ) * (self.aquifer.d_top + 0.5 * self.aquifer.H) ) / 2 * D_in\
               + self.P_pump

        return P_HE

    def _get_T_heatloss_pipe(self, D_in, length_pipe):
        alpha = self.labdas / ( self.rhos * self.cps) #thermal diffusion of rock
        gamma = 0.577216 #euler constant

        q_heatloss_pipe = 4 * math.pi * self.labdas * ( self.T_wellbore - self._get_T(length_pipe) ) / math.log( ( 4 * alpha * self.time ) / (math.exp(gamma) * (D_in/2)**2 ) )

        return q_heatloss_pipe

    def _get_T_HE(self, D_in, length_pipe):
        T_HE = self.T_wellbore

        for i in range(len(length_pipe)-1):
            T_HE -= length_pipe[-2] * self.q_heatloss_pipe[i] / ( self.mdot * self.cp )

        return T_HE

    def _get_f(self, D_in):
        f = ( 1.14 - 2 * math.log10( self.well.epsilon / D_in + 21.25 / ( self.get_Re( D_in )**0.9 ) ) )**-2
        return f

    def get_v_avg(self, D_in):
        v_avg = 4 * self.well.Q / ( math.pi * ( D_in ** 2 ) )
        return v_avg

    def get_Re(self, D_in):
        Re = ( self.aquifer.rho_f * self.get_v_avg( D_in ) ) / self.aquifer.mhu
        return Re

    def _get_P_wb(self):
        """ Computes pressure at wellbore

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        P_wb (float): value of pressure at well bore
        """
        P_wb = self.P_aquifer + ( ( self.well.Q * self.aquifer.mhu ) / ( 2 * math.pi * self.aquifer.K * self.aquifer.H ) ) * np.log ( self.well.L / self.well.r)
        return P_wb

    def _get_T(self, d):
        """ Computes temperature of the aquifer as a function of the depth

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        T (float): value of temperature
        """
        T = self.aquifer.T_surface + d * self.aquifer.labda
        return T

    def _get_P(self, d):
        """ Computes pressure of the aquifer as a function of the depth

        Arguments:
        d (float): depth (downwards from groundlevel is positive)
        Returns:
        T (float): value of temperature
        """
        P_atm = 1e05
        g = 9.81
        P = P_atm + g * self.aquifer.rho_f * d

        return P

    def rho(self, T, p):
        rho = (1 + 10e-6 * (-80 * T - 3.3 * T**2 + 0.00175 * T**3 + 489 * p - 2 * T * p + 0.016 * T**2 * p - 1.3e-5 * T**3\
                           * p - 0.333 * p**2 - 0.002 * T * p**2) )

        return rho

    def _make_nodes_grid(self):
        """ Compute a nodes grid for the doublet

        Returns:
        x_grid_nodes, y_grid_nodes (np.array): arrays of the domain in x and y direction
        """
        i = np.arange(0, self.Nx+1, 1)
        j = np.arange(0, -self.Ny-1, -1)

        i_coords, j_coords = np.meshgrid(i, j)

        nodes_grid = np.array([i_coords, j_coords])

        return nodes_grid

    def _make_coordinates_grid(self):
        coordinates_grid = self.nodes_grid

        coordinates_grid[0,:,:] = self.nodes_grid[0,:,:] * self.domain[0]
        coordinates_grid[1,:,:] = self.nodes_grid[1,:,:] * -self.domain[1]

        return coordinates_grid

    def _make_grid(self):
        """ Compute a cartesian grid for the doublet

        Returns:
        domain (np.array): array of the domain in x and y direction
        """
        x = np.linspace(0, self.well.L * 3, self.Nx)
        y = np.linspace(0,- (2 * self.aquifer.d_top + self.aquifer.H) , self.Ny)
        x_grid, y_grid = np.meshgrid(x, y)

        return x_grid, y_grid

    def _construct_well(self):
        """ Compute two wells for the doublet

        Returns:
        x_well, y_well (np.array): array of the x and y of the well
        """
        # x = np.array([[self.well.L * 5 - self.well.L * 0.5], [self.well.L * 5 + self.well.L * 0.5]])
        # y = np.linspace(0,- (self.aquifer.d_top + self.aquifer.H) , (20 * self.Ny) - 1)
        x_well = np.array([[self.x_grid[0][math.floor(self.Nx/3)]], [self.x_grid[0][2*math.floor(self.Nx/3)]]])
        y_well = self.y_grid[math.floor(self.Ny/2)][0] * np.ones(2)
        # print(self.y_grid)
        # print(y_well)

        # x_well, y_well = np.meshgrid(x, y)
        # print(x_well)
        # print(y_well)

        return x_well, y_well

class Node:
    """Represent node.

    Args:
        ID_x float: ID of x position of the node.
        ID_y float: ID of y position of the node.

    """
    def __init__(self, ID_x, ID_y, domain):

        self.ID_x = ID_x
        self.ID_y = ID_y
        self.pos = [self._get_x_coordinate(self.ID_x, domain), self._get_y_coordinate(self.ID_y, domain)]

    def _get_x_coordinate(self, ID_x, domain):
        """ Calculates x coordinate of node.

        Arguments:
            ID_x (int): x index of node
        Returns:
            x (float): Scalar of x coordinate of node center
        """
        x = domain[0] * ID_x
        return x

    def _get_y_coordinate(self, ID_y, domain):
        """ Calculates y coordinate of node.

        Arguments:
            ID_y (int): y index of node
        Returns:
            y (float): Scalar of x coordinate of node center
        """
        y = domain[1] * ID_y
        return y

## Solve mass balance
class SolveMassBalance:

    def __init__(self, aquifer, well):
        self.aquifer = aquifer
        self.well = well

    # def Integral(self, y):
    #     # integral term for the well
    #     return x



### main script ###
def PumpTest():
    t0 = 0.01
    tEnd = 10
    well = Well(t0, tEnd)
    aquifer = Aquifer()
    Doublet = DoubletGenerator(aquifer, well)
    print("\r\n############## Analytical values model ##############\n"
          "P_aq,i/P_aq,p:   ", round(Doublet.P_aquifer/1e5,2), "Bar\n"
          "P_bh,i/P_bh,p:   ", round(Doublet.P_wellbore/1e5,2), "Bar\n"
          "T_bh,p:          ", Doublet.T_wellbore, "Celcius\n"
          "P_out,p/P_in,HE: ", round(Doublet.P_HE/ 1e5,2), "Bar\n"
          "P_pump,p:        ", Doublet.P_pump/ 1e5, "Bar\n"
          "T_out,p/T_in,HE: ", round(Doublet.T_HE,2), "Celcius\n"
          "P_in,i/P_out,HE: ", round(Doublet.P_HE/ 1e5,2), "Bar\n"
          "T_in,i/T_out,HE: ", Doublet.well.Ti_inj, "Celcius\n"
          "Power,HE:        ", round(Doublet.Power_HE/1e6,2), "MW")

    # print("Node (3,2) info is ", Node(3, 2, Doublet.domain).pos)
    # fig, ax = plt.subplots()
    #
    # fig, axs = plt.subplots(2, sharex=True, sharey=True)
    # fig.suptitle('2D cross section of the doublet')
    #
    # plot0 = axs[0].contourf(Doublet.x_grid, Doublet.y_grid, Doublet.P_grid/1e5, cmap='RdGy')
    # cb0 = fig.colorbar(plot0, ax=axs[0], label="Pressure [Bar]")
    # cb0.ax.invert_yaxis()
    # plt.xlabel('x [m]')
    # plt.ylabel('depth [m]')
    # axs[0].scatter(Doublet.x_grid, Doublet.y_grid, 0.4, color="green")
    #
    # axs[0].arrow(Doublet.x_well[1], Doublet.y_well[1], 0, 0.9*abs(Doublet.y_well[1]), length_includes_head=True,
    #       head_width=100, head_length=10, overhang=10, color='r')
    # axs[0].arrow(Doublet.x_well[0], 0, 0, -0.9 * abs(Doublet.y_well[0]), length_includes_head=True,
    #           head_width=100, head_length=10, overhang=10, color='b')
    # axs[0].add_patch(Rectangle((Doublet.x_well[0], -(Aquifer().d_top + Aquifer().H)), Doublet.x_well[0], Aquifer().H, linewidth=1, edgecolor='k', facecolor='none'))
    #
    #
    # plot1 = axs[1].contourf(Doublet.x_grid, Doublet.y_grid, Doublet.T_grid, cmap='RdGy')
    # cb1 = fig.colorbar(plot1, ax=axs[1], label="Temperature [C]")
    # cb1.ax.invert_yaxis()
    # axs[1].scatter(Doublet.x_grid, Doublet.y_grid, 0.4, color="green")
    #
    # axs[1].arrow(Doublet.x_well[1], Doublet.y_well[1], 0, 0.9 * abs(Doublet.y_well[1]), length_includes_head=True,
    #              head_width=100, head_length=10, overhang=10, color='r')
    # axs[1].arrow(Doublet.x_well[0], 0, 0, -0.9 * abs(Doublet.y_well[0]), length_includes_head=True,
    #              head_width=100, head_length=10, overhang=10, color='b')
    # axs[1].add_patch(
    #     Rectangle((Doublet.x_well[0], -(Aquifer().d_top + Aquifer().H)), Doublet.x_well[0], Aquifer().H, linewidth=1,
    #               edgecolor='k', facecolor='none'))
    # plt.show()

# run script
PumpTest()