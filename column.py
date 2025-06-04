import nest
import json
import numpy as np
from default_params.ctx_params import ctx_params

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

NOTICE = YELLOW
FCOLOR = {
    "E": "\033[34m",
    "I": "\033[31m"
    }
BCOLOR = {
    "E": "\033[44m",
    "I": "\033[41m"
    }

class Column:
    def __init__(self):
        pass

    def __init__(self, col_label, col_params):
        """
        Initialise a column.

        Args:
            col_label (str): Label (or name) of column
            col_params (dict): Column params
            col_conn (str): File name of internal column connections
        
        Returns:
            None
        """
        self.A_scaling = 0.1
        self.col_label = col_label
        self.col_params = col_params.copy()
        self.size = col_params["structure_info"]["region_size"]
        self.connections = col_params["connection_info"]["internal"]
        self.psg = col_params["connection_info"]["poisson"]
        # self.create_column(self.col_params["structure_info"]["region_name"])

    def create_neuron(self, pop_name, pop_params):
        """
        Create neuron model.
        ---

        Args:
            pop_name (str): Population name
            pop_params (dict): Population params
        
        Returns:
            Tuple(float, str):
                - float: Neuron density of population
                - str: Class of population
        """
        params = pop_params.copy()
        neuron_model = params.pop("model")
        pop_class = params.pop("EI")
        pop_density = params.pop("Cellcount_mm2")
        if "cond" in neuron_model:
            pop_params["g_L"] = 250. / params.pop("tau_m")
        if not pop_name in nest.node_models:
            nest.CopyModel(neuron_model, pop_name, params)
            # print(f"({self.col_label}){pop_name}")
        return pop_density, pop_class

    def estimate_population(self, density, h_layer):
        """
        Generate neuron grid and position.

        Args:
            density (float): Neuron density
            h_layer (float): Thickness of layer
        
        Returns:
            position (nest.): NEST grid
        """
        fineness = (density / h_layer) ** (1/3)
        shape = np.concatenate([np.array(self.size[:2]), [h_layer]])
        grid = np.ceil(shape * self.A_scaling * fineness).astype(int)
        position = nest.spatial.grid(shape=grid, center=[0., 0., 0.], extent=shape, edge_wrap=True)
        return position

    def create_column(self, area_name):
        """
        Create a column.

        Args:
            area_name (str): Name of area that this column belongs to
        
        Returns:
            pops (dict): All populations in the column
        """
        # self.pops = {"E": {}, "I": {}}
        self.pops = {}
        self.pop_flags = {}
        print(f"\033[32mCreating Column {self.col_label}\033[0m")
        layerthickness = self.col_params["structure_info"]["layer_thickness"]
        layers = self.col_params["structure_info"]["Layer_Name"]
        for layer_name, layer_params in self.col_params["neuro_info"].items():
        
            # print("_".join([area_name, layer_name])) # Create Populations in Layer
            # print(f"layer_thickness [{layer_name}] = {layerthickness[layers.index(layer_name)]}")
            h_layer = layerthickness[layers.index(layer_name)]
            for pop, pop_params in layer_params.items(): 
                pop_name = "_".join([area_name, layer_name, pop]) # Create a Single Population
                density, flag = self.create_neuron(pop_name, pop_params)
                pos = self.estimate_population(density, h_layer)
                population = nest.Create(pop_name, positions=pos)
                # self.pops[flag][pop_name] = population
                self.pops[pop_name] = population
                self.pop_flags[pop_name] = flag
        return self.pops

    def create_connections(self, col_conn=None, verbose=False):
        """
        Create internal connections in a column.

        Args:
            col_conn (str): Path to connection file
        
        Returns:
            None
        """

        if col_conn == None:
            col_conn = self.connections
        with open(col_conn, 'r') as f:
            connections = json.load(f)
        conn = connections.copy()
        for pre_pop in conn.keys():
            for post_pop, conn_params in conn[pre_pop].items():
                self.connect_layers_ctx(pre_pop, post_pop, conn_params)
                if verbose:
                    print(f"(Col {self.col_label}) {BCOLOR[self.pop_flags[pre_pop]]}{pre_pop:^13}{RESET} ---< {BCOLOR[self.pop_flags[post_pop]]}{post_pop:^13}{RESET}")
                    # SN = len(nest.GetConnections(self.pops[pre_pop], self.pops[post_pop]))
                    # print(f"{COLOR[self.pop_flags[pre_pop]]}{pre_pop:^13}{RESET} --{SN:-^7}-< {COLOR[self.pop_flags[post_pop]]}{post_pop:^13}{RESET}")
                    # del SN
        pass

    def connect_layers_ctx(self, pre, post, conn):
        """
        Connect two layers in the column.

        Args:
            pre (str): Name of presynaptic population
            post (str): Name of postsynaptic population
            conn (dict): Params of connections between specified populations.
        
        Returns:
            None
        """
        sigma_x = conn['sigma']/1000.
        sigma_y = conn['sigma']/1000.
        if conn['p_center'] != 0.0 and sigma_x != 0.0 and conn['weight'] != 0.0:
            weight_distribution=conn['weight_distribution']
            if weight_distribution == 'lognormal':
                weight = nest.random.lognormal(mean=conn['weight'], std=1.0)
            else:
                weight = conn['weight']
            conn_dict = {'rule': 'pairwise_bernoulli',
                         'p': conn['p_center']*nest.spatial_distributions.gaussian2D(
                             nest.spatial.distance.x, nest.spatial.distance.y, std_x=sigma_x, std_y=sigma_y),
                        #  'mask': {'box':
                        #               {'lower_left': [-1., -1., -2.],
                        #                'upper_right': [1., 1., 2.]}},
                         'allow_autapses': False,
                         'allow_multapses': False,
                        #  'allow_oversiSzed_mask': True
                         }
            syn_spec = {
                'weight': weight,
                'delay': conn['delay']
            }
            pre = self.pops[pre]
            post = self.pops[post]
            nest.Connect(pre, post, conn_dict, syn_spec)

    def add_detectors(self, verbose=False):
        """
        Add and connect detectors for each layers.

        Args:
            verbose (bool): To execute this in verbose mode or not
        
        Returns:
            dict: All detectors in the column
        """
        self.detectors = {}
        for pop_name, pop in self.pops.items():
            detector = nest.Create("spike_recorder", params={"record_to": "ascii"})
            nest.Connect(pop, detector)
            self.detectors[pop_name] = detector
            if verbose:
                print(f"{NOTICE}Add detecor for {FCOLOR[self.pop_flags[pop_name]]}{pop_name}{RESET}")
        print(f"{NOTICE}Add detectors for column {self.col_label}{RESET}")
        return self.detectors
    
    def add_posissons(self, verbose=False):
        """
        Add and connect poisson noise generators for a column.

        Args:
            verbose (bool): To execute this in verbose mode or not

        Returns:
            None
        """
        psgs = self.col_params["connection_info"]["poisson"]
        for pop_name, pop in self.pops.items():
            psg_dict = {"rate": psgs[pop_name]["PSG"]["rate"], "start": 500. + psgs[pop_name]["PSG"]["offset"]}
               
            psg = nest.Create("poisson_generator", params=psg_dict)
            nest.Connect(psg, pop, syn_spec=psgs[pop_name]["SYNP"])
            if verbose:
                print(f"Add Poisson's noise for {FCOLOR[self.pop_flags[pop_name]]}{pop_name}{RESET}")
        print(f"{NOTICE}Add Poisson's noise for column {self.col_label}{RESET}")
        pass

    def update_weight(self, pre, post, new_conn):
        """
        Update synapse connection weight.

        Args:
            pre (pop):
            post (pop):
            new_conn (dict):

        Returns:
            None
        """
        pass
