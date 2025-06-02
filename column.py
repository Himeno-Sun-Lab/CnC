import nest
import json
import numpy as np
from default_params.ctx_params import ctx_params

class Column:
    def __init__(self):
        pass

    def __init__(self, col_label, col_params, col_conn):
        """
        Initialise a column.

        Args:
            col_label (str): Label (or name) of column
            col_params (dict): Column params
            col_conn (str): File name of internal column connections
        Returns:
            None
        """
        self.col_label = col_label
        self.col_params = col_params.copy()
        self.size = col_params["structure_info"]["region_size"]
        self.connections = col_conn
        # self.create_column(self.col_params["structure_info"]["region_name"])

    def create_neuron(self, pop_name, pop_params):
        """
        Create neuron model.

        Args:
            pop_name (str): Population name
            pop_params (dict): Population params
        
        Returns:
            pop_density (float): Neuron density of population
            pop_class (str): Class of population
        """
        params = pop_params.copy()
        neuron_model = params.pop("model")
        pop_class = params.pop("EI")
        pop_density = params.pop("Cellcount_mm2")
        if "cond" in neuron_model:
            pop_params["g_L"] = 250. / params.pop("tau_m")
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
        print(f"Creating Column {self.col_label}")
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
    
    def create_connections(self):
        """
        Create connections in a column.

        Args:
            conn_dict (dict): Connections read from connection file
        
        Returns:
            None
        """
        with open(self.connections, 'r') as f:
            connections = json.load(f)
        conn = connections.copy()
        for pre_pop in conn.keys():
            for post_pop, conn_params in conn[pre_pop].items():
                print(f"{pre_pop} --< {post_pop}")
                self.connect_layers_ctx(pre_pop, post_pop, conn_params)
                # Nsyn = self.connect_layers_ctx(pre_pop, post_pop, conn_params)
                # print(f"{Nsyn} synapses")
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
            # SN (int): Synapse number between specified populations.
        """
        sigma_x = conn['sigma']/1000.
        sigma_y = conn['sigma']/1000.
        #print (conn_params)
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
            # SN = len(nest.GetConnections(pre, post))
            # return SN