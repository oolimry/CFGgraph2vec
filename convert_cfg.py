from pycfg.pycfg import PyCFG, CFGNode, slurp 
import argparse 
import json
import os.path
import os
import pandas as pd
import glob

exam = "final" #final or midterm
if __name__ == '__main__':
    
    if True: 
        doesnt_exist = []
        couldnt_cfg = []
        score_of_dashes = []
        score_of_zero = []
        works = []
        
        for name in os.scandir("raw_codes"):
            input_file = f"raw_codes/{name.name}" #can change
	     	
            CFGNode.cache = {}
            CFGNode.registry = 0
            cfg = PyCFG()
	        
            if not os.path.isfile(input_file):
                doesnt_exist += [input_file]
                continue
            try:
                cfg.gen_cfg(slurp(input_file).strip())
            except:
                couldnt_cfg += [input_file]
                continue
            g = CFGNode.to_graph()
                
            edges = g.edges()
            edges_new = []
            for count in range(len(edges)):
                edges_new += [[int(edges[count][0]), int(edges[count][1])]]
		    
            nodes = g.nodes()
            features_dictionary = {}
            for count in range(len(nodes)):
                label = g.get_node(count).attr['label'] 
                features_dictionary[nodes[count]] = label.split(': ')[-1] 
            graph2vec_input = {"edges": edges_new, "features": features_dictionary}
		    
            #print(graph2vec_input)
		    
            output_folder = "dataset/"
            output_filename = (input_file.split("_")[3]).replace("py", "json")
            print(output_filename)
            with open(output_folder + output_filename, 'w') as outfile:
                json.dump(graph2vec_input, outfile)
            works += [output_filename]
	
        print("Converted everything to graph2vec input")
        print("=========================================================================")
        print("\nDoesn't exist\n")
        print(doesnt_exist)
        print("\nNo cfg\n")
        print(couldnt_cfg)
        print("\nScore is a dash\n")
        print(score_of_dashes)
        print("\nScore is zero\n")
        print(score_of_zero)
        print("\nWorks\n")
        print(works)

