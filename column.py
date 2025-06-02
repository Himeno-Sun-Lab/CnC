import nest
import json
import numpy as np
from default_params.ctx_params import ctx_params

class Column:
    def __init__(self):
        pass

    def __init__(self, col_label, col_params):
        self.col_label = col_label
        self.col_params = col_params.copy()
        self.size = col_params["size"]
    
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
        print(pop_name)
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
        shape = np.concatenate([np.array(self.size), [h_layer]])
        grid = np.round(shape * self.A_scaling * fineness).astype(int)
        position = nest.spatial.grid(shape=grid, center=[0., 0., 0.], extent=shape, edge_wrap=True)
        return position

    def create_column(self, area_name):
        """
        Create a column.

        Args:
            area_name (str): Name of area that this column belongs to
        
        Returns:
            pops (dict): All populations in all layers
        """
        pops = {"E": {}, "I": {}}
        print(f"Creating Column {self.col_label}")
        layerthickness = self.col_params["structure_info"]["layer_thickness"]
        layers = self.col_params["structure_info"]["Layer_Name"]
        for layer_name, layer_params in self.col_params["neuro_info"].items():
        
            print("_".join([area_name, layer_name])) # Create Populations in Layer
            print(f"layer_thickness [{layer_name}] = {layerthickness[layers.index(layer_name)]}")
            h_layer = layerthickness[layers.index(layer_name)]
            for pop, pop_params in layer_params.items(): 
                pop_name = "_".join([area_name, layer_name, pop]) # Create a Single Population
                density, flag = self.create_neuron(pop_name, pop_params)
                pos = self.estimate_population(density, h_layer)
                population = nest.Create(pop_name, positions=pos)
                pops[flag][pop_name] = population

        return pops
    def read_connection(self):
        with open(self.connections, 'r') as f:
            conn = json.load(f)
        return conn
    
    def create_connection(self):
        pass

    if __name__ == "__main__":
        nest.ResetKernel()
        kk = create_column("S1", ctx_params["S1"])

        # backend.save_params(kk, "Col_I")

        with open("data/S1.json", "r") as f:
            conn = json.load(f)
            # print(conn)