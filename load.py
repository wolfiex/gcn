def get_data():

    import re
    import numpy as np


    fullstr =  open('/work/home/dp626/DSMACC-testing/mechanisms/full_mcm_2019.kpp','r').read()
    minfull = re.sub(r' |\n|\t|\s|\r','', fullstr).upper()


    eqn = [i.split(':') for i in re.findall(r'[^/]{1,2}\s*\{[\.\W\s\d]*?\}([^;]+)' ,'   '+minfull,re.S|re.M)]

    nocoeff = re.compile(r'\d*\.*\d*([\W\d\w]+)')

    l = []
    for e in eqn:
        e = e[0].split('=')
        for r in e[0].split('+'):
            r=nocoeff.findall(r)[0]
            for p in e[1].split('+'):
                if p!= '':
                    l.append((r,nocoeff.findall(p)[0]
    ))
                



    import networkx as nx

    G = nx.DiGraph()
    G.add_edges_from( l )



    #############
    import pickle
    with open('/work/home/dp626/DSMACC-testing/dsmacc/examples/fingerprintgen/all_fingerprints.pkl', 'rb') as f:
        finger = pickle.load(f)
        

    print set(G.nodes) ^ set(finger['names'])

    ## remove nonexists (inorganics mainly)
    for n in set(G.nodes) ^ set(finger['names']):
        G.remove_node(n)
        

    a = nx.adjacency_matrix(G, nodelist = finger['names'])
    
    return a, finger




print 'fi'